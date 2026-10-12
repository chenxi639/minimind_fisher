"""Cross-process CPU synthetic resume/evaluation tests; never research training."""
import argparse,json,math,subprocess,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind
from model.model_moment_controls import MomentControlMiniMind
from model.model_peft25 import install
from portable.validate_training_paths import cpu_tree,exact
CONFIG={'hidden_size':32,'num_hidden_layers':2,'vocab_size':64,'num_attention_heads':4,'num_key_value_heads':2,'max_position_embeddings':64,'flash_attn':False,'fisher_index':0,'fisher_dim':4,'fisher_k':1,'moment_mode':'current','dropout':.1}
RUNTIME={'device':'cpu','precision':'fp32'}
def setup():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
def build(arm):
    if arm not in ['k1','current','lora']:raise ValueError('unsupported arm')
    torch.manual_seed(21)
    m=(FisherMiniMind if arm=='k1' else MomentControlMiniMind if arm=='current' else MiniMindForCausalLM)(MiniMindConfig(**CONFIG))
    if arm=='lora':install(m,'lora')
    else:m.requires_grad_(False);m.model.layers[0].fisher.requires_grad_(True)
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    return m,opt
def snapshot(m,opt,arm,step,data,losses):
    return {'schema_version':1,'mode':'synthetic_resume_only','arm':arm,'config':CONFIG,'model':cpu_tree(m.state_dict()),'optimizer':cpu_tree(opt.state_dict()),'rng':torch.get_rng_state(),'step':step,'data':data,'losses':losses,'parameter_names':[n for n,p in m.named_parameters() if p.requires_grad]}
def restore(ck):
    if ck['schema_version']!=1 or ck['mode']!='synthetic_resume_only' or ck['config']!=CONFIG or ck['step'] not in [1,2]:raise ValueError('incompatible checkpoint')
    m,opt=build(ck['arm']);m.load_state_dict(ck['model'],strict=True)
    if ck['parameter_names']!=[n for n,p in m.named_parameters() if p.requires_grad]:raise ValueError('parameter ordering mismatch')
    opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);return m,opt
def advance(m,opt,data,step):
    m.train();opt.zero_grad(set_to_none=True);values=[]
    for micro in range(2):
        x=data[(step*2+micro)*2:(step*2+micro+1)*2]
        opt.param_groups[0]['lr']=.0005*(.1+.45*(1+math.cos(math.pi*(step*2+micro+1)/4)))
        loss=m(x,labels=x).loss/2;assert torch.isfinite(loss);loss.backward();values.append(float(loss.detach()))
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step();return values
def evaluate(m,data):
    m.eval();x=data[:4];y=x.clone();y[0,-3:]=-100;y[2,-1:]=-100
    with torch.no_grad():
        losses=torch.nn.functional.cross_entropy(m(x).logits[:,:-1].transpose(1,2),y[:,1:],ignore_index=-100,reduction='none')
    counts=(y[:,1:]!=-100).sum(1);sums=losses.double().sum(1);n=int(counts.sum());total=float(sums.sum())
    return {'tokens':n,'nll':total,'ce':total/n,'ppl':math.exp(total/n),'document_tokens':counts.tolist(),'document_nll':sums.tolist()}
def worker(folder,mode):
    for arm in ['k1','current','lora']:
        ck=torch.load(folder/(arm+('_first.pt' if mode=='resume' else '_resumed.pt')),weights_only=True,map_location='cpu');m,opt=restore(ck)
        if mode=='resume':
            losses=ck['losses']+[advance(m,opt,ck['data'],ck['step'])]
            torch.save(snapshot(m,opt,arm,2,ck['data'],losses),folder/(arm+'_resumed.pt'))
        else:(folder/(arm+'_eval.json')).write_text(json.dumps(evaluate(m,ck['data'])),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--worker',choices=['resume','evaluate']);a=ap.parse_args();setup()
    if a.worker:worker(a.output,a.worker);return
    a.output.mkdir(parents=True,exist_ok=False);result={'status':'running','research_evidence':False,**RUNTIME,'arms':{}}
    def save():(a.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save()
    try:
        data=torch.randint(1,64,(8,12),generator=torch.Generator().manual_seed(77))
        for arm in ['k1','current','lora']:
            m,opt=build(arm);frozen={n:p.clone() for n,p in m.named_parameters() if not p.requires_grad};torch.manual_seed(99);losses=[]
            for step in range(2):
                losses.append(advance(m,opt,data,step))
                if step==0:torch.save(snapshot(m,opt,arm,1,data,losses.copy()),a.output/(arm+'_first.pt'))
            assert all(torch.equal(p,frozen[n]) for n,p in m.named_parameters() if not p.requires_grad)
            torch.save(snapshot(m,opt,arm,2,data,losses),a.output/(arm+'_continuous.pt'))
        for mode in ['resume','evaluate']:
            subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--output',str(a.output.resolve()),'--worker',mode],check=True,cwd=str(a.output.resolve()))
        for arm in ['k1','current','lora']:
            full=torch.load(a.output/(arm+'_continuous.pt'),weights_only=True);resumed=torch.load(a.output/(arm+'_resumed.pt'),weights_only=True);exact(full,resumed,arm)
            m,opt=restore(full);expected=evaluate(m,data);actual=json.loads((a.output/(arm+'_eval.json')).read_text());exact(expected,actual,arm+'.evaluation')
            assert expected['tokens']==40 and abs(expected['ce']-sum(expected['document_nll'])/sum(expected['document_tokens']))<1e-15
            result['arms'][arm]={'status':'passed','cross_process_model_optimizer_rng_losses_exact':True,'evaluation_exact':True,'tokens':expected['tokens']}
        # Restoration must reject incompatible schema and parameter ordering.
        for key,value in [('schema_version',2),('parameter_names',[])]:
            bad=dict(full);bad[key]=value
            try:restore(bad)
            except ValueError:pass
            else:raise AssertionError('invalid resume accepted: '+key)
        result.update(status='passed',invalid_checkpoint_rejected=True);save();print(json.dumps(result))
    except Exception:result.update(status='failed',error=traceback.format_exc());save();raise
if __name__=='__main__':main()
