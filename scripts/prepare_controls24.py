"""Stage24 controls initial states and two-step frozen-backbone smoke only."""
import sys,json,hashlib,math
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_moment_controls import MomentControlMiniMind,MomentControl
from model.model_matched_mlp import MatchedMiniMind,MatchedResidual
from dataset.lm_dataset import PretrainDataset
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    old=ROOT/'runs/mechanism_ready_20261003_112436';source=ROOT/'runs/pair_ready_20260929_181458'
    cfg=json.loads((old/'config.json').read_text());previous=json.loads((old/'preparation.json').read_text())
    assert previous['status']=='passed'
    files=dict(cfg['file_hashes'])
    files[str(old/'common_init.pt')]=previous['common_init_sha256']
    files.update({str(source/n):d for n,d in cfg['hashes'].items()})
    for p,d in files.items():assert sha(Path(p))==d,p
    run=ROOT/'runs'/('controls24_ready_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir();print(run,flush=True)
    r={'status':'running','formal_training':False}
    def save():(run/'preparation.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    try:
        common=torch.load(old/'common_init.pt',weights_only=True,map_location='cpu')
        initials={}
        for arm in ['mean','second','current','adapter']:
            torch.manual_seed(43)
            branch=MatchedResidual(768,96) if arm=='adapter' else MomentControl(768,64,arm)
            initial=branch.state_dict();initial['norm.weight']=common['norm.weight'].clone()
            if arm!='adapter':
                for key in ['input.weight','w','b']:initial[key]=common[key].clone()
            assert torch.count_nonzero(initial['output.weight'])==0
            initials[arm]=initial;torch.save(initial,run/(arm+'_init.pt'))
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
        for arm in ['mean','second','current','adapter']:
            torch.manual_seed(42);cls=MatchedMiniMind if arm=='adapter' else MomentControlMiniMind
            m=cls(MiniMindConfig(fisher_k=1,fisher_dim=64,fisher_index=3,moment_mode=arm,adapter_width=96))
            initial=initials[arm];name='adapter' if arm=='adapter' else 'fisher'
            m.load_state_dict({**state,**{f'model.layers.3.{name}.'+k:v for k,v in initial.items()}},strict=True);m.cuda().eval();branch=getattr(m.model.layers[3],name)
            with torch.no_grad():assert torch.equal(ref,forward(m,x).logits.cpu())
            for p in m.parameters():p.requires_grad_(False)
            for p in branch.parameters():p.requires_grad_(True)
            params=list(branch.parameters());count=sum(p.numel() for p in params);assert count== (99072 if arm in ['mean','second'] else 148224)
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
            if arm!='adapter':assert torch.equal(branch.b.cpu(),initial['b']) and torch.equal(branch.w.cpu(),initial['w'])
            torch.save(branch.state_dict(),run/(arm+'_smoke.pt'));loaded=torch.load(run/(arm+'_smoke.pt'),weights_only=True)
            m.eval()
            with torch.no_grad():before=forward(m,x).logits.cpu()
            branch.load_state_dict(loaded,strict=True)
            with torch.no_grad():assert torch.equal(before,forward(m,x).logits.cpu())
            def disable(layer,args,output):return torch.zeros_like(output) if arm=='adapter' else (torch.zeros_like(output[0]),output[1])
            handle=branch.register_forward_hook(disable)
            with torch.no_grad():assert torch.equal(ref,forward(m,x).logits.cpu())
            handle.remove()
            r[arm]={'status':'passed','trainable_parameters':count,'initial_logits_exact':True,'backbone_unchanged':True,'fixed_buffers_unchanged':arm!='adapter','reload_exact':True,'disabled_returns_pure':True,'steps':rows,'init_sha256':sha(run/(arm+'_init.pt'))}
            del m,branch,opt,params;torch.cuda.empty_cache();save();print(arm,r[arm],flush=True)
        for p,d in files.items():assert sha(Path(p))==d
        for arm in initials:files[str(run/(arm+'_init.pt'))]=sha(run/(arm+'_init.pt'))
        for n in ['scripts/prepare_controls24.py','tests/test_moment_controls.py','report/24A_preparation_protocol.md','model/model_moment_controls.py','model/model_matched_mlp.py']:files[str(ROOT/n)]=sha(ROOT/n)
        cfg.update(k=1,seed=42,branch_seed=43,arms=['mean','second','current','adapter'],adapter_width=96,initialization='common Stage23A untrained projection and train mean for statistics; seed43 width96 adapter; all zero outputs',file_hashes=files,formal_training=False,prepared_path=str(run),primary_evaluation='fp32_tf32_off',auxiliary_evaluation='bf16')
        (run/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        r.update(status='passed',source_common_init_sha256=sha(old/'common_init.pt'),source_files_unchanged=True,per_arm_smoke_updates=2,test_evaluated=False,external_evaluated=False)
    except Exception as exc:r.update(status='failed',error=repr(exc));raise
    finally:save()
if __name__=='__main__':main()
