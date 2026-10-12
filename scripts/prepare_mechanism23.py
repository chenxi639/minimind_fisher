"""Common K1/Moment initial states and real frozen-backbone smoke; no formal training."""
import sys,json,hashlib,math
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind,FisherResidual
from model.model_moment import MomentMiniMind
from dataset.lm_dataset import PretrainDataset
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    old=ROOT/'runs/frozen_ready_20261002_124018';source=ROOT/'runs/pair_ready_20260929_181458'
    cfg=json.loads((old/'config.json').read_text());previous=json.loads((old/'preparation.json').read_text())
    files={str(Path(cfg['backbone_path'])):cfg['backbone_sha256'],str(old/'kmeans_fit.pt'):previous['kmeans']['sha256'],str(old/'fisher_init.pt'):previous['fisher']['init_sha256']}
    files.update({str(source/n):d for n,d in cfg['hashes'].items()})
    for p,d in files.items():assert sha(Path(p))==d,p
    run=ROOT/'runs'/('mechanism_ready_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir();print(run,flush=True)
    r={'status':'running','formal_training':False}
    def save():(run/'preparation.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    try:
        descriptors=torch.load(old/'kmeans_fit.pt',weights_only=True)['descriptors'];assert descriptors.shape==(30000,64)
        mu=descriptors.mean(0,keepdim=True);oldinit=torch.load(old/'fisher_init.pt',weights_only=True)
        torch.manual_seed(43);f=FisherResidual(768,1,64);initial=f.state_dict()
        initial['norm.weight']=oldinit['norm.weight'].clone();initial['input.weight']=oldinit['input.weight'].clone()
        initial['w']=torch.ones(1,64);initial['b']=-mu;initial['output.weight']=torch.zeros(768,128)
        torch.save(initial,run/'common_init.pt');torch.save(mu,run/'training_mean.pt')
        state=torch.load(cfg['backbone_path'],weights_only=True,map_location='cpu')
        tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
        train=PretrainDataset(str(source/'train.jsonl'),tok,max_length=340);dev=PretrainDataset(str(source/'dev.jsonl'),tok,max_length=340)
        order=json.loads((source/'order.json').read_text())
        def batch(ds,ids):
            rows=[ds[i] for i in ids];return [torch.stack([row[k] for row in rows]).cuda() for k in [0,1]]
        def forward(m,x,y=None):
            with torch.autocast('cuda',dtype=torch.bfloat16):return m(x,labels=y)
        pure=MiniMindForCausalLM(MiniMindConfig());pure.load_state_dict(state,strict=True);pure.cuda().eval()
        x,y=batch(dev,[0,64])
        with torch.no_grad():ref=forward(pure,x).logits.cpu()
        del pure
        for arm,cls in [('k1',FisherMiniMind),('moment',MomentMiniMind)]:
            torch.manual_seed(42);m=cls(MiniMindConfig(fisher_k=1,fisher_dim=64,fisher_index=3))
            m.load_state_dict({**state,**{'model.layers.3.fisher.'+k:v for k,v in initial.items()}},strict=True);m.cuda().eval();branch=m.model.layers[3].fisher
            with torch.no_grad():assert torch.equal(ref,forward(m,x).logits.cpu())
            for p in m.parameters():p.requires_grad_(False)
            for p in branch.parameters():p.requires_grad_(True)
            params=list(branch.parameters());count=sum(p.numel() for p in params);assert count== (148352 if arm=='k1' else 148224)
            opt=torch.optim.AdamW(params,lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=.01);rows=[]
            m.train()
            for step in range(2):
                opt.zero_grad(set_to_none=True);tokens=0
                for micro in range(8):
                    off=(step*8+micro)*2;tx,ty=batch(train,order[off:off+2]);lr=.0005*(.1+.45*(1+math.cos(math.pi*(step*8+micro+1)/8192)))
                    opt.param_groups[0]['lr']=lr;loss=forward(m,tx,ty).loss/8;assert torch.isfinite(loss);loss.backward();tokens+=int((ty[:,1:]!=-100).sum())
                grads={n:float(p.grad.norm()) for n,p in branch.named_parameters()}
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                assert grads['output.weight']>0
                if step==1:assert all(v>0 for v in grads.values())
                assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();rows.append({'step':step+1,'tokens':tokens,'grad_norms':grads})
            assert all(torch.equal(v.cpu(),state[k].to(v.dtype)) for k,v in m.state_dict().items() if k in state)
            if arm=='moment':assert torch.equal(branch.b.cpu(),initial['b']) and torch.equal(branch.w.cpu(),initial['w'])
            torch.save(branch.state_dict(),run/(arm+'_smoke.pt'));loaded=torch.load(run/(arm+'_smoke.pt'),weights_only=True)
            m.eval()
            with torch.no_grad():before=forward(m,x).logits.cpu()
            branch.load_state_dict(loaded,strict=True)
            with torch.no_grad():assert torch.equal(before,forward(m,x).logits.cpu())
            def disable(layer,args,output):return torch.zeros_like(output[0]),output[1]
            handle=branch.register_forward_hook(disable)
            with torch.no_grad():assert torch.equal(ref,forward(m,x).logits.cpu())
            handle.remove()
            r[arm]={'status':'passed','trainable_parameters':count,'initial_logits_exact':True,'backbone_unchanged':True,'reload_exact':True,'disabled_returns_pure':True,'steps':rows}
            del m,branch,opt,params;torch.cuda.empty_cache();save();print(arm,r[arm],flush=True)
        for p,d in files.items():assert sha(Path(p))==d
        files[str(run/'common_init.pt')]=sha(run/'common_init.pt')
        for n in ['scripts/prepare_mechanism23.py','model/model_minimind.py','model/model_fisher.py','model/model_moment.py','model/tokenizer.json','model/tokenizer_config.json','dataset/lm_dataset.py']:files[str(ROOT/n)]=sha(ROOT/n)
        cfg.update(k=1,seed=42,initialization='common train-descriptor mean; learned w/b for K1, fixed buffers for Moment',file_hashes=files,formal_training=False,prepared_path=str(run))
        (run/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        r.update(status='passed',common_init_sha256=sha(run/'common_init.pt'),source_files_unchanged=True)
    except Exception as exc:r.update(status='failed',error=repr(exc));raise
    finally:save()
if __name__=='__main__':main()
