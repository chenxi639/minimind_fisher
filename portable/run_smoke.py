"""Asset-free plumbing test, explicitly NOT a reproduction of research metrics."""
import argparse,hashlib,json,math,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind,FisherResidual
from model.model_moment_controls import MomentControlMiniMind
from model.model_peft25 import install
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8'));assert cfg['mode']=='synthetic_smoke_only'
    assert cfg['steps']>=2 and cfg['steps']*cfg['batch_size']<=cfg['train_records']
    assert cfg['eval_records']%cfg['batch_size']==0
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True)
    result={'status':'running','mode':cfg['mode'],'research_evidence':False,'device':'cpu','precision':'fp32','config_sha256':sha(args.config),'arms':{}}
    def save():(out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save();(out/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    try:
        mc=cfg['model'];base=MiniMindForCausalLM(MiniMindConfig(**mc)).eval();state={n:v.clone() for n,v in base.state_dict().items()}
        generator=torch.Generator().manual_seed(cfg['seed']+10)
        train=torch.randint(1,mc['vocab_size'],(cfg['train_records'],cfg['sequence_length']),generator=generator)
        ev=torch.randint(1,mc['vocab_size'],(cfg['eval_records'],cfg['sequence_length']),generator=generator)
        labels=ev.clone();labels[0,-3:]=-100
        def evaluate(m):
            m.eval();nll=0.;count=0
            with torch.no_grad():
                for i in range(0,len(ev),cfg['batch_size']):
                    x=ev[i:i+cfg['batch_size']];y=labels[i:i+cfg['batch_size']]
                    loss=torch.nn.functional.cross_entropy(m(x).logits[:,:-1].transpose(1,2),y[:,1:],ignore_index=-100,reduction='sum')
                    assert torch.isfinite(loss);nll+=float(loss);count+=int((y[:,1:]!=-100).sum())
            return {'nll':nll,'tokens':count,'ce':nll/count,'ppl':math.exp(nll/count)}
        reference=evaluate(base)
        with torch.no_grad():probe=base(ev).logits.clone()
        def build(kind):
            torch.manual_seed(cfg['seed']+1)
            if kind in ['pure','lora']:
                m=MiniMindForCausalLM(MiniMindConfig(**mc));m.load_state_dict(state,strict=True);m.requires_grad_(False)
                if kind=='lora':
                    torch.manual_seed(cfg['seed']+1);install(m,'lora')
            else:
                params={**mc,'fisher_k':8 if kind=='k8' else 1,'moment_mode':'current'}
                m=(MomentControlMiniMind if kind=='current' else FisherMiniMind)(MiniMindConfig(**params))
                torch.manual_seed(cfg['seed']+1);f=FisherResidual(mc['hidden_size'],params['fisher_k'],mc['fisher_dim'])
                # Synthetic smoke uses zero reference offsets, NOT historical fitted centers.
                with torch.no_grad():f.b.zero_()
                initial={f"model.layers.{mc['fisher_index']}.fisher."+n:v for n,v in f.state_dict().items()}
                m.load_state_dict({**state,**initial},strict=True);m.requires_grad_(False);m.model.layers[mc['fisher_index']].fisher.requires_grad_(True)
            return m
        for kind in ['pure','k8','k1','current','lora']:
            m=build(kind);m.eval();initial={n:p.detach().clone() for n,p in m.named_parameters() if p.requires_grad}
            with torch.no_grad():assert torch.equal(m(ev).logits,probe)
            assert evaluate(m)==reference
            params=[p for p in m.parameters() if p.requires_grad];history=[]
            if params:
                opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=.01);torch.manual_seed(cfg['seed'])
                for step in range(cfg['steps']):
                    m.train();opt.zero_grad(set_to_none=True);x=train[step*cfg['batch_size']:(step+1)*cfg['batch_size']]
                    loss=m(x,labels=x).loss;assert torch.isfinite(loss);loss.backward()
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
                    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
                    history.append({'step':step+1,'loss':float(loss.detach()),'gradient_norm':float(norm)})
                assert any(not torch.equal(p,initial[n]) for n,p in m.named_parameters() if p.requires_grad)
            assert all(torch.equal(m.state_dict()[n],v) for n,v in state.items())
            if kind=='current':
                branch=m.model.layers[mc['fisher_index']].fisher;assert torch.equal(branch.w,torch.ones_like(branch.w)) and not branch.b.any()
            m.eval();final=evaluate(m)
            with torch.no_grad():before=m(ev).logits.clone()
            file=out/(kind+'.pt');torch.save({'state_dict':m.state_dict(),'mode':cfg['mode'],'config':cfg,'arm':kind},file)
            restored=build(kind);restored.load_state_dict(torch.load(file,weights_only=True,map_location='cpu')['state_dict'],strict=True);restored.eval()
            with torch.no_grad():assert torch.equal(before,restored(ev).logits)
            assert evaluate(restored)==final
            result['arms'][kind]={'status':'passed','steps':len(history),'trainable_parameters':sum(p.numel() for p in params),'zero_output_identity':True,'backbone_unchanged':True,'reload_exact':True,'eval':final,'history':history,'checkpoint_sha256':sha(file)};save()
        result.update(status='passed',baseline=reference);save();print(json.dumps({'status':'passed','arms':list(result['arms']),'research_evidence':False}))
    except Exception:result.update(status='failed',error=traceback.format_exc());save();raise
if __name__=='__main__':main()
