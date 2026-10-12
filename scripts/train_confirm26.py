"""Stage26C fixed runs. No external evaluation or checkpoint selection."""
import sys,json,math,time,hashlib,traceback
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind
from model.model_moment_controls import MomentControlMiniMind
from model.model_peft25 import install,enabled
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    prep=ROOT/'runs/confirm26_ready_20261004_194128';ready=json.loads((prep/'preparation.json').read_text());assert ready['status']=='passed'
    files=dict(ready['file_hashes'])
    for p in [prep/'preparation.json',Path(__file__),ROOT/'report/26C_protocol.md']:files[str(p)]=sha(p)
    def verify():
        for p,h in files.items():assert sha(p)==h,p
    verify();run=ROOT/'runs'/('confirm26_train_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir();print(run,flush=True)
    manifest={'file_hashes':files,'prepared_path':str(prep),'seeds':[1729,2718,31415],'arms':['k1','current','lora'],'steps':1024,'formal_training':True}
    save(run/'manifest.json',manifest);summary={'status':'running','external_evaluated':False,'test_evaluated':False,'runs':{}};save(run/'comparison.json',summary)
    try:
        state=torch.load(ROOT/'checkpoints/official_edba70e/download.pth',weights_only=True,map_location='cpu')
        tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True);source=ROOT/'runs/pair_ready_20260929_181458'
        train=PretrainDataset(str(source/'train.jsonl'),tok,max_length=340);dev=PretrainDataset(str(source/'dev.jsonl'),tok,max_length=340)
        def batch(ds,ids):
            rows=[ds[i] for i in ids];return [torch.stack([r[k] for r in rows]).cuda() for k in [0,1]]
        def fw(m,x,y=None,p='bf16'):
            with (torch.autocast('cuda',dtype=torch.bfloat16) if p=='bf16' else nullcontext()):return m(x,labels=y)
        def evaluate(m,p):
            m.eval();total=0.;tokens=0
            with torch.no_grad():
                for i in range(0,len(dev),2):
                    x,y=batch(dev,[i,i+1]);loss=fw(m,x,y,p).loss;assert torch.isfinite(loss);n=int((y[:,1:]!=-100).sum());total+=float(loss)*n;tokens+=n
            assert tokens==265051
            return {'ce':total/tokens,'ppl':math.exp(total/tokens),'tokens':tokens}
        pure=MiniMindForCausalLM(MiniMindConfig());pure.load_state_dict(state,strict=True);pure.cuda().eval()
        baseline={p:evaluate(pure,p) for p in ['fp32','bf16']}
        old=json.loads((ROOT/'runs/peft25_20261004_112438/comparison.json').read_text());assert baseline==old['baseline']
        px,py=batch(dev,[0,64])
        with torch.no_grad():reference=fw(pure,px).logits.cpu()
        del pure;torch.cuda.empty_cache();summary['baseline']=baseline;save(run/'comparison.json',summary);print('Pure consistency passed',flush=True)
        for seed in manifest['seeds']:
            order=json.loads((prep/str(seed)/'order.json').read_text());assert sorted(order)==list(range(16384));exposure=None
            for kind,expected in [('k1',148352),('current',148224),('lora',150528)]:
                verify();folder=run/str(seed)/kind;folder.mkdir(parents=True);history=[];key=f'{seed}/{kind}';result={'status':'running'};summary['runs'][key]=result;save(run/'comparison.json',summary);start=time.perf_counter()
                cfg=MiniMindConfig(fisher_k=1,fisher_dim=64,fisher_index=3,moment_mode='current')
                if kind=='lora':
                    m=MiniMindForCausalLM(cfg);m.load_state_dict(state,strict=True);torch.manual_seed(seed+1);install(m,'lora')
                else:
                    m=(FisherMiniMind if kind=='k1' else MomentControlMiniMind)(cfg);m.requires_grad_(False);m.model.layers[3].fisher.requires_grad_(True)
                initial=torch.load(prep/str(seed)/(kind+'_init.pt'),weights_only=True,map_location='cpu');m.load_state_dict({**state,**initial},strict=True);m.cuda().eval()
                with torch.no_grad():assert torch.equal(reference,fw(m,px).logits.cpu())
                torch.manual_seed(seed);params=[p for p in m.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==expected
                opt=torch.optim.AdamW(params,lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=.01);seen=0
                def adaptation():return {n:m.state_dict()[n].detach().cpu().clone() for n in initial}
                for step in range(1024):
                    m.train();opt.zero_grad(set_to_none=True);count=0;total=0.
                    for micro in range(8):
                        off=(step*8+micro)*2;x,y=batch(train,order[off:off+2]);lr=.0005*(.1+.45*(1+math.cos(math.pi*(step*8+micro+1)/8192)));opt.param_groups[0]['lr']=lr
                        loss=fw(m,x,y).loss;assert torch.isfinite(loss);(loss/8).backward();n=int((y[:,1:]!=-100).sum());count+=n;total+=float(loss.detach())*n
                    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
                    if step==1:assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in params)
                    norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();seen+=count
                    if exposure is not None:assert seen==exposure[step]
                    row={'step':step+1,'tokens':seen,'train_ce':total/count,'grad_norm':float(norm),'lr':lr};history.append(row)
                    if step+1 in [256,512,1024]:
                        torch.save({'adaptation':adaptation(),'optimizer':opt.state_dict(),'steps':step+1,'tokens':seen,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'manifest_sha256':sha(run/'manifest.json')},folder/('checkpoint.pt' if step==1023 else f'checkpoint_{step+1}.pt'))
                    if (step+1)%64==0:save(folder/'history.json',history);print(key,row,flush=True)
                if exposure is None:exposure=[r['tokens'] for r in history]
                assert seen==4176264 and all(torch.equal(m.state_dict()[n].cpu(),v) for n,v in state.items())
                if kind=='current':assert all(torch.equal(m.state_dict()[n].cpu(),initial[n]) for n in ['model.layers.3.fisher.w','model.layers.3.fisher.b'])
                m.eval()
                with torch.no_grad():before=fw(m,px).logits.cpu()
                path=folder/'checkpoint.pt';ck=torch.load(path,weights_only=True,map_location='cpu');m.load_state_dict({**state,**ck['adaptation']},strict=True)
                with torch.no_grad():assert torch.equal(before,fw(m,px).logits.cpu())
                final={p:evaluate(m,p) for p in ['fp32','bf16']}
                if kind=='lora':enabled(m,False)
                else:hook=m.model.layers[3].fisher.register_forward_hook(lambda mod,a,o:(torch.zeros_like(o[0]),o[1]))
                with torch.no_grad():assert torch.equal(reference,fw(m,px).logits.cpu())
                if kind!='lora':hook.remove()
                verify();result.update(status='passed',steps=1024,tokens=seen,trainable_parameters=expected,final_dev=final,backbone_unchanged=True,reload_exact=True,disabled_probe_exact=True,checkpoint_sha256=sha(path),seconds=time.perf_counter()-start)
                save(folder/'result.json',result);save(run/'comparison.json',summary);print(key,result,flush=True);del m,opt,params,ck;torch.cuda.empty_cache()
        summary.update(status='passed',per_seed_exposure_matched=True);save(run/'comparison.json',summary)
    except Exception:
        summary.update(status='failed',error=traceback.format_exc());save(run/'comparison.json',summary);raise
if __name__=='__main__':main()
