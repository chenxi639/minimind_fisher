"""Locked Stage25B training; only full development evaluation."""
import sys,json,math,hashlib,time
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_peft25 import install,enabled,adaptation_state
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    prepared=ROOT/'runs/peft25_ready_20261003_180008';cfg=json.loads((prepared/'locked_config.json').read_text());files=dict(cfg['file_hashes'])
    for p in [Path(__file__),ROOT/'report/25B_protocol.md',prepared/'locked_config.json']:files[str(p)]=sha(p)
    def verify():
        for p,d in files.items():assert sha(p)==d,p
    verify();run=ROOT/'runs'/('peft25_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir();print(run,flush=True)
    cfg.update(file_hashes=files,formal_training=True);save(run/'manifest.json',cfg)
    state=torch.load(cfg['backbone_path'],weights_only=True,map_location='cpu');tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    source=Path(cfg['source_preparation']);train=PretrainDataset(str(source/'train.jsonl'),tok,max_length=340);dev=PretrainDataset(str(source/'dev.jsonl'),tok,max_length=340)
    order=json.loads((source/'order.json').read_text());assert len(train)==16384 and len(dev)==1024 and sorted(order)==list(range(16384))
    def batch(ds,ids):
        rows=[ds[i] for i in ids];return [torch.stack([r[k] for r in rows]).cuda() for k in [0,1]]
    def forward(m,x,y,p='bf16'):
        with (torch.autocast('cuda',dtype=torch.bfloat16) if p=='bf16' else nullcontext()):return m(x,labels=y)
    def evaluate(m,p):
        m.eval();total=0.;tokens=0
        with torch.no_grad():
            for i in range(0,1024,2):
                x,y=batch(dev,[i,i+1]);loss=forward(m,x,y,p).loss;assert torch.isfinite(loss)
                n=int((y[:,1:]!=-100).sum());tokens+=n;total+=float(loss)*n
        assert tokens==265051
        return dict(ce=total/tokens,ppl=math.exp(total/tokens),tokens=tokens)
    m=MiniMindForCausalLM(MiniMindConfig());m.load_state_dict(state,strict=True);m.cuda()
    baseline={p:evaluate(m,p) for p in ['fp32','bf16']};del m;torch.cuda.empty_cache()
    old=json.loads((ROOT/'runs/controls24_20261003_152008/comparison.json').read_text());assert baseline==old['baseline']
    exposure=[r['tokens'] for r in json.loads((ROOT/'runs/controls24_20261003_152008/mean/history.json').read_text())[1:]]
    summary={'status':'running','baseline':baseline};save(run/'comparison.json',summary);print('BASELINE',baseline,flush=True)
    for kind,expected in [('houlsby',159840),('lora',150528)]:
        verify();folder=run/kind;folder.mkdir();history=[];r={'status':'running'};start=time.perf_counter()
        try:
            m=MiniMindForCausalLM(MiniMindConfig());m.load_state_dict(state,strict=True);torch.manual_seed(43);install(m,kind)
            initial=torch.load(prepared/(kind+'_init.pt'),weights_only=True,map_location='cpu');m.load_state_dict({**state,**initial},strict=True);m.cuda();torch.manual_seed(42)
            params=[p for p in m.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==expected
            opt=torch.optim.AdamW(params,lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=.01)
            first=evaluate(m,'bf16');assert first==baseline['bf16'];history.append({'step':0,'dev':first})
            torch.cuda.reset_peak_memory_stats();seen=0
            def checkpoint(step):
                path=folder/('checkpoint.pt' if step==1024 else f'checkpoint_{step}.pt')
                torch.save({'adaptation':adaptation_state(m),'optimizer':opt.state_dict(),'steps':step,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),'manifest_sha256':sha(run/'manifest.json')},path)
            for step in range(1024):
                m.train();opt.zero_grad(set_to_none=True);total=0.;count=0
                for micro in range(8):
                    off=(step*8+micro)*2;x,y=batch(train,order[off:off+2]);lr=.0005*(.1+.45*(1+math.cos(math.pi*(step*8+micro+1)/8192)));opt.param_groups[0]['lr']=lr
                    loss=forward(m,x,y).loss;assert torch.isfinite(loss);(loss/8).backward();n=int((y[:,1:]!=-100).sum());count+=n;total+=float(loss.detach())*n
                assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
                if step==1:assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in params)
                norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();seen+=count;assert seen==exposure[step]
                row={'step':step+1,'tokens':seen,'train_ce':total/count,'grad_norm':float(norm),'lr':lr}
                if step+1 in [256,512,1024]:row['dev']=evaluate(m,'bf16');checkpoint(step+1)
                history.append(row)
                if (step+1)%64==0:save(folder/'history.json',history);print(kind,row,flush=True)
            assert seen==4176264 and all(torch.equal(m.state_dict()[n].cpu(),v) for n,v in state.items())
            path=folder/'checkpoint.pt';digest=sha(path);ck=torch.load(path,weights_only=True,map_location='cpu')
            assert all(torch.equal(v,ck['adaptation'][n]) for n,v in adaptation_state(m).items());m.load_state_dict({**state,**ck['adaptation']},strict=True)
            normal={p:evaluate(m,p) for p in ['fp32','bf16']};assert normal['bf16']==history[-1]['dev']
            enabled(m,False);disabled={p:evaluate(m,p) for p in ['fp32','bf16']};assert disabled==baseline;enabled(m,True)
            assert all(torch.equal(v,ck['adaptation'][n]) for n,v in adaptation_state(m).items());assert sha(path)==digest;verify()
            r.update(status='passed',steps=1024,tokens=seen,trainable_parameters=expected,total_parameters=sum(p.numel() for p in m.parameters()),final_dev=normal,disabled=disabled,backbone_unchanged=True,reload_exact=True,checkpoint_sha256=digest,seconds=time.perf_counter()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20)
            summary[kind]=r;save(run/'comparison.json',summary);del m,opt,params,ck;torch.cuda.empty_cache()
        except Exception as exc:r.update(status='failed',error=repr(exc));raise
        finally:save(folder/'result.json',r);save(folder/'history.json',history);print(kind,r,flush=True)
    summary.update(status='passed',per_step_exposure_matches_stage24=True,test_evaluated=False,external_evaluated=False);save(run/'comparison.json',summary)
if __name__=='__main__':main()
