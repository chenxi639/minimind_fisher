"""Stage23B independent K1/Moment screening; no external or sealed test evaluation."""
import sys,json,hashlib,math,time,argparse
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind
from model.model_moment import MomentMiniMind

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    parser=argparse.ArgumentParser();parser.add_argument('--prepared',type=Path,required=True);args=parser.parse_args()
    prepared=args.prepared.resolve()
    cfg=json.loads((prepared/'config.json').read_text());prep=json.loads((prepared/'preparation.json').read_text())
    assert prep['status']=='passed'
    source=Path(cfg['source_preparation']);weight=Path(cfg['backbone_path'])
    files=dict(cfg['file_hashes'])
    cfg['order_path']=str(source/'order.json')
    files.update({str(source/n):d for n,d in cfg['hashes'].items()})
    files[str(prepared/'common_init.pt')]=prep['common_init_sha256']
    names=['scripts/train_mechanism23.py','report/23B_mechanism_protocol.md']
    files.update({str(ROOT/n):sha(ROOT/n) for n in names})
    def verify():
        for n,d in files.items():assert sha(Path(n))==d,n
    verify()
    run=ROOT/'runs'/('mechanism_pair_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir()
    def save(p,v):p.write_text(json.dumps(v,indent=2),encoding='utf-8')
    cfg.update(formal_training=True,optimizer={'name':'AdamW','betas':[.9,.999],'eps':1e-8,'weight_decay':.01,'amsgrad':False,'foreach':None,'fused':None},file_hashes=files)
    save(run/'manifest.json',cfg);print(run,flush=True)
    state=torch.load(weight,map_location='cpu',weights_only=True)
    tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    train=PretrainDataset(str(source/'train.jsonl'),tok,max_length=340);dev=PretrainDataset(str(source/'dev.jsonl'),tok,max_length=340)
    order=json.loads(Path(cfg['order_path']).read_text());assert sorted(order)==list(range(len(train)))
    assert len(train)==16384 and len(dev)==1024
    def batch(ds,ids):
        rows=[ds[j] for j in ids];return tuple(torch.stack([r[k] for r in rows]).cuda() for k in (0,1))
    def forward(m,x,y):
        with torch.autocast('cuda',dtype=torch.bfloat16):return m(x,labels=y)
    active=None
    def evaluate(m,observer=None):
        nonlocal active
        m.eval();total=0.;count=0
        if observer:observer.start()
        with torch.no_grad():
            for i in range(0,len(dev),2):
                x,y=batch(dev,[i,i+1]);active=torch.zeros_like(y,dtype=torch.bool);active[:,:-1]=y[:,1:]!=-100
                if observer:observer.mask=active
                out=forward(m,x,y);assert torch.isfinite(out.loss)
                n=int(active.sum());total+=float(out.loss)*n;count+=n
        if observer:observer.finish()
        return {'ce':total/count,'ppl':math.exp(total/count),'tokens':count}
    torch.manual_seed(cfg['seed']);pure=MiniMindForCausalLM(MiniMindConfig());pure.load_state_dict(state,strict=True);pure.cuda()
    baseline=evaluate(pure);save(run/'baseline.json',baseline);print('BASELINE',baseline,flush=True);del pure;torch.cuda.empty_cache()
    assert baseline==json.loads((ROOT/'runs/frozen_pair_20261002_145450/baseline.json').read_text())
    exposure=None;summary={'baseline':baseline}
    for arm,cls,name in [('k1',FisherMiniMind,'fisher'),('moment',MomentMiniMind,'fisher')]:
        verify();torch.manual_seed(cfg['seed']);folder=run/arm;folder.mkdir();start=time.perf_counter()
        result={'status':'running','arm':arm,'steps':0};history=[];save(folder/'result.json',result)
        try:
            model_cfg=MiniMindConfig();model_cfg.fisher_k=1;model_cfg.fisher_dim=64;model_cfg.fisher_index=3
            model=cls(model_cfg);initial=torch.load(prepared/'common_init.pt',weights_only=True,map_location='cpu')
            combined={**state,**{f'model.layers.3.{name}.'+k:v for k,v in initial.items()}}
            model.load_state_dict(combined,strict=True);del combined;model.cuda()
            branch=getattr(model.model.layers[3],name)
            for p in model.parameters():p.requires_grad_(False)
            for p in branch.parameters():p.requires_grad_(True)
            params=list(branch.parameters());assert sum(p.numel() for p in params)==prep[arm]['trainable_parameters']
            optimizer=torch.optim.AdamW(params,lr=cfg['lr'],betas=(.9,.999),eps=1e-8,weight_decay=.01)
            observer=None
            initial_dev=evaluate(model,observer);assert initial_dev==baseline
            history.append({'step':0,'dev':initial_dev});save(folder/'history.json',history)
            torch.cuda.reset_peak_memory_stats();seen=0
            for step in range(1024):
                model.train();optimizer.zero_grad(set_to_none=True);nll=0.;count=0
                for micro in range(8):
                    off=(step*8+micro)*2;x,y=batch(train,order[off:off+2])
                    lr=cfg['lr']*(.1+.45*(1+math.cos(math.pi*(step*8+micro+1)/(1024*8))))
                    optimizer.param_groups[0]['lr']=lr
                    out=forward(model,x,y);loss=out.loss/8;assert torch.isfinite(loss);loss.backward()
                    n=int((y[:,1:]!=-100).sum());count+=n;nll+=float(out.loss.detach())*n
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                if step==1:assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in params)
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step();seen+=count
                row={'step':step+1,'tokens':seen,'train_ce':nll/count,'grad_norm':float(norm),'lr':lr}
                if step+1 in [256,512,1024]:row['dev']=evaluate(model,observer)
                if arm=='moment':assert all(torch.equal(branch.state_dict()[k].cpu(),initial[k]) for k in ('w','b'))
                if step+1 in [256,512]:
                    torch.save({'branch':branch.state_dict(),'optimizer':optimizer.state_dict(),'steps':step+1,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'manifest_sha256':sha(run/'manifest.json')},folder/f'checkpoint_{step+1}.pt')
                history.append(row)
                if (step+1)%64==0:save(folder/'history.json',history);print(arm,row,flush=True)
            assert seen==4176264
            assert all(torch.equal(v.cpu(),state[k].to(v.dtype)) for k,v in model.state_dict().items() if k in state)
            this_exposure=[r['tokens'] for r in history[1:]]
            if exposure is None:exposure=this_exposure
            else:assert exposure==this_exposure
            torch.save({'branch':branch.state_dict(),'optimizer':optimizer.state_dict(),'steps':1024,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'manifest_sha256':sha(run/'manifest.json'),'backbone_sha256':sha(weight)},folder/'checkpoint.pt')
            checkpoint_sha=sha(folder/'checkpoint.pt');ck=torch.load(folder/'checkpoint.pt',weights_only=True,map_location='cpu')
            assert all(torch.equal(v.cpu(),ck['branch'][k]) for k,v in branch.state_dict().items())
            branch.load_state_dict(ck['branch'],strict=True)
            if observer:observer.handle.remove()
            full={'tokens':0,'ratio':0.}
            def monitor(layer,args,output):
                delta=output[0]
                h=args[0].float();n=int(active.sum());full['tokens']+=n
                full['ratio']+=float((delta.float()[active].norm(dim=-1)/h[active].norm(dim=-1).clamp_min(1e-12)).sum())
            handle=branch.register_forward_hook(monitor);normal=evaluate(model);handle.remove()
            assert normal==history[-1]['dev']
            def disable(layer,args,output):return (torch.zeros_like(output[0]),output[1])
            handle=branch.register_forward_hook(disable);disabled=evaluate(model);handle.remove();assert disabled==baseline
            assert all(torch.equal(v.cpu(),ck['branch'][k]) for k,v in branch.state_dict().items())
            assert sha(folder/'checkpoint.pt')==checkpoint_sha;verify()
            health={'normal':normal,'disabled':disabled,'disabled_minus_normal_ce':disabled['ce']-normal['ce'],'residual_hidden_ratio':full['ratio']/full['tokens'],'checkpoint_unchanged':True}
            health['state_change_l2']={k:float((v.cpu()-initial[k]).float().norm()) for k,v in branch.state_dict().items()}
            health['assignment']='K=1, q=1 by definition; not evidence of component diversity'
            save(folder/'health.json',health)
            result.update(status='passed',steps=1024,tokens=seen,backbone_exact_unchanged=True,checkpoint_exact=True,checkpoint_sha256=checkpoint_sha,seconds=time.perf_counter()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20,final_dev=normal)
            summary[arm]=result;save(run/'comparison.json',summary)
            del model,branch,optimizer,params,ck;torch.cuda.empty_cache()
        except Exception as exc:result.update(status='failed',error=repr(exc));raise
        finally:save(folder/'history.json',history);save(folder/'result.json',result);print(arm,result,flush=True)
    summary.update(status='passed',per_step_token_exposure_equal=True,initial_dev_exact=True,test_evaluated=False)
    save(run/'comparison.json',summary)
if __name__=='__main__':main()
