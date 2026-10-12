"""Nine paired short-run equivalence checks on math SDPA, not historical re-training."""
import argparse,copy,json,math,sys,tempfile,shutil,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind
from model.model_moment_controls import MomentControlMiniMind
from model.model_peft25 import install
from portable.check_assets import check,digest
from portable.train_regression import micro_lr
def cpu_tree(v):
    if torch.is_tensor(v):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu_tree(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu_tree(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu_tree(x) for x in v)
    return copy.deepcopy(v)
def exact(a,b,label='root'):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a,b):raise AssertionError(label)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(label+'.keys')
        for k in a:exact(a[k],b[k],label+'.'+str(k))
    elif isinstance(a,(list,tuple)):
        if len(a)!=len(b):raise AssertionError(label+'.length')
        for i,(x,y) in enumerate(zip(a,b)):exact(x,y,label+'.'+str(i))
    elif a!=b:raise AssertionError(label)
def main():
    ap=argparse.ArgumentParser()
    for n in ['config','manifest','mapping','output']:ap.add_argument('--'+n,type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.config.read_text());audit=check(a.manifest,a.mapping);mp=json.loads(a.mapping.read_text())
    def p(n):
        q=Path(mp[n]);return q if q.is_absolute() else a.mapping.resolve().parent/q
    expected={'schema_version':1,'protocol_id':'stage26-frozen-v1','seeds':[1729,2718,31415],'arms':['k1','current','lora'],'horizon_steps':1024,'batch_size':2,'accumulation_steps':8,'max_length':340,'learning_rate':.0005,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01,'clip_norm':1.,'precision':'bf16','tf32':False}
    assert cfg==expected
    for i in range(8192):assert micro_lr(cfg,i//8,i%8)==.0005*(.1+.45*(1+math.cos(math.pi*(i+1)/8192)))
    for key,rel in [('code_model','model/model_minimind.py'),('code_fisher','model/model_fisher.py'),('code_current','model/model_moment_controls.py'),('code_moment','model/model_moment.py'),('code_peft','model/model_peft25.py'),('code_dataset','dataset/lm_dataset.py')]:assert digest(ROOT/rel)==audit['assets'][key]['sha256']
    a.output.mkdir(parents=True,exist_ok=False);r={'status':'running','research_evidence':False,'backend':'math_sdpa_only','optimizer_updates':0,'schedule_points_exact':8192,'config_sha256':digest(a.config),'script_sha256':digest(Path(__file__)),'asset_audit':audit,'runs':{},'torch_version':str(torch.__version__)}
    def save():(a.output/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    save()
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        with tempfile.TemporaryDirectory() as td:
            for n in ['tokenizer','tokenizer_config']:shutil.copyfile(p(n),Path(td)/(n+'.json'))
            tok=AutoTokenizer.from_pretrained(td,local_files_only=True)
        ds=PretrainDataset(str(p('train')),tok,max_length=340);assert len(ds)==16384
        state=torch.load(p('backbone'),weights_only=True,map_location='cpu')
        for seed in cfg['seeds']:
            order=json.loads(p(f'{seed}_order').read_text());assert order==torch.randperm(16384,generator=torch.Generator().manual_seed(seed)).tolist()
            rows=[ds[i] for i in order[:32]]
            batches=[tuple(torch.stack([rows[j][k] for j in [i,i+1]]).cuda() for k in [0,1]) for i in range(0,32,2)]
            for kind in cfg['arms']:
                key=f'{seed}/{kind}';folder=a.output/str(seed)/kind;folder.mkdir(parents=True);paired={}
                initial=torch.load(p(f'{seed}_{kind}_init'),weights_only=True,map_location='cpu')
                for variant in ['historical','configured']:
                    mc=MiniMindConfig(fisher_k=1,fisher_dim=64,fisher_index=3,moment_mode='current')
                    if kind=='lora':
                        m=MiniMindForCausalLM(mc);m.load_state_dict(state,strict=True);torch.manual_seed(seed+1);install(m,'lora')
                    else:
                        m=(FisherMiniMind if kind=='k1' else MomentControlMiniMind)(mc);m.requires_grad_(False);m.model.layers[3].fisher.requires_grad_(True)
                    m.load_state_dict({**state,**initial},strict=True);m.cuda().eval()
                    if variant=='historical':
                        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):m(batches[0][0])
                    torch.manual_seed(seed);params=[v for v in m.parameters() if v.requires_grad]
                    assert sum(v.numel() for v in params)=={'k1':148352,'current':148224,'lora':150528}[kind]
                    opt=torch.optim.AdamW(params,lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=.01) if variant=='historical' else torch.optim.AdamW(params,lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay'])
                    history=[]
                    for step in range(2):
                        m.train();opt.zero_grad(set_to_none=True);losses=[];tokens=0
                        for micro in range(8):
                            x,y=batches[step*8+micro]
                            lr=.0005*(.1+.45*(1+math.cos(math.pi*(step*8+micro+1)/8192))) if variant=='historical' else micro_lr(cfg,step,micro)
                            opt.param_groups[0]['lr']=lr
                            with torch.autocast('cuda',dtype=torch.bfloat16):
                                loss=m(x,labels=y).loss
                                if variant=='configured':loss=loss/cfg['accumulation_steps']
                            if variant=='historical':loss=loss/8
                            assert torch.isfinite(loss);loss.backward();losses.append(float(loss.detach()));tokens+=int((y[:,1:]!=-100).sum())
                        gradients={n:cpu_tree(v.grad) for n,v in m.named_parameters() if v.requires_grad};assert all(torch.isfinite(v).all() for v in gradients.values())
                        assert all(v.grad is None for v in m.parameters() if not v.requires_grad)
                        norm=torch.nn.utils.clip_grad_norm_(params,1. if variant=='historical' else cfg['clip_norm'],error_if_nonfinite=True);opt.step();r['optimizer_updates']+=1
                        snapshot={'gradients':gradients,'optimizer':cpu_tree(opt.state_dict()),'adaptation':{n:cpu_tree(m.state_dict()[n]) for n in initial},'losses':losses,'tokens':tokens,'lr':lr,'grad_norm':float(norm)}
                        torch.save(snapshot,folder/f'{variant}_{step+1}.pt')
                        if variant=='historical':paired[step]=snapshot
                        else:exact(snapshot,paired[step],f'{key}/step{step+1}')
                        history.append({'step':step+1,'tokens':tokens,'lr':lr,'grad_norm':float(norm)})
                    assert all(torch.equal(m.state_dict()[n].cpu(),v) for n,v in state.items())
                    if kind=='current':assert all(torch.equal(m.state_dict()[n].cpu(),initial[n]) for n in ['model.layers.3.fisher.w','model.layers.3.fisher.b'])
                    del m,opt,params;torch.cuda.empty_cache()
                r['runs'][key]={'status':'passed','both_steps_exact':True,'backbone_unchanged':True,'history':history,'evidence_hashes':{f.name:digest(f) for f in folder.glob('*.pt')}};save();print(key,'PASS',flush=True)
        check(a.manifest,a.mapping);r['status']='passed';save()
    except Exception:r.update(status='failed',error=traceback.format_exc());save();raise
if __name__=='__main__':main()
