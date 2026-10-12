"""Two-step compatibility checks, never formal training or test evaluation."""
import sys,json,math,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_peft25 import install,enabled,is_adaptation,adaptation_state,merged_lora
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=ROOT/'runs/controls24_ready_20261003_142354';cfg=json.loads((old/'config.json').read_text());files=dict(cfg['file_hashes'])
    for n in ['model/model_peft25.py','scripts/prepare_peft25.py','tests/test_peft25.py','report/25A_protocol.md']:files[str(ROOT/n)]=sha(ROOT/n)
    def verify():
        for p,d in files.items():assert sha(p)==d,p
    verify();run=ROOT/'runs'/('peft25_ready_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir();print(run,flush=True)
    result={'status':'running','formal_training':False,'arms':{}}
    def save():(run/'preparation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save();state=torch.load(cfg['backbone_path'],map_location='cpu',weights_only=True)
    tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True);source=Path(cfg['source_preparation'])
    train=PretrainDataset(str(source/'train.jsonl'),tok,max_length=340);dev=PretrainDataset(str(source/'dev.jsonl'),tok,max_length=340)
    order=json.loads((source/'order.json').read_text())
    def batch(ds,ids):
        rows=[ds[i] for i in ids];return [torch.stack([r[k] for r in rows]).cuda() for k in [0,1]]
    def forward(m,x,y=None):
        with torch.autocast('cuda',dtype=torch.bfloat16):return m(x,labels=y)
    pure=MiniMindForCausalLM(MiniMindConfig());pure.load_state_dict(state,strict=True);pure.cuda().eval();x,y=batch(dev,[0,64])
    with torch.no_grad():ref=forward(pure,x).logits.cpu()
    del pure;torch.cuda.empty_cache()
    for kind,expected in [('houlsby',159840),('lora',150528)]:
        m=MiniMindForCausalLM(MiniMindConfig());m.load_state_dict(state,strict=True);torch.manual_seed(43);install(m,kind)
        initial=adaptation_state(m);torch.save(initial,run/(kind+'_init.pt'));files[str(run/(kind+'_init.pt'))]=sha(run/(kind+'_init.pt'))
        params=[p for p in m.parameters() if p.requires_grad];assert sum(p.numel() for p in params)==expected
        m.cuda().eval()
        # Obtain parameters after device move, which is the supported optimizer workflow.
        params=[p for p in m.parameters() if p.requires_grad]
        with torch.no_grad():assert torch.equal(ref,forward(m,x).logits.cpu())
        torch.manual_seed(42);opt=torch.optim.AdamW(params,lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=.01);rows=[]
        m.train()
        for step in range(2):
            opt.zero_grad(set_to_none=True);tokens=0
            for micro in range(8):
                off=(step*8+micro)*2;tx,ty=batch(train,order[off:off+2]);opt.param_groups[0]['lr']=.0005*(.1+.45*(1+math.cos(math.pi*(step*8+micro+1)/8192)))
                loss=forward(m,tx,ty).loss/8;assert torch.isfinite(loss);loss.backward();tokens+=int((ty[:,1:]!=-100).sum())
            gradients={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.requires_grad}
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            if step==1:assert all(v>0 for v in gradients.values())
            assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();rows.append({'step':step+1,'tokens':tokens,'gradient_norms':gradients})
        assert all(torch.equal(m.state_dict()[n].cpu(),v) for n,v in state.items())
        m.eval();torch.save(adaptation_state(m),run/(kind+'_smoke.pt'));loaded=torch.load(run/(kind+'_smoke.pt'),weights_only=True)
        with torch.no_grad():before=forward(m,x).logits.cpu()
        m.load_state_dict({**state,**loaded},strict=True)
        with torch.no_grad():assert torch.equal(before,forward(m,x).logits.cpu())
        enabled(m,False)
        with torch.no_grad():assert torch.equal(ref,forward(m,x).logits.cpu())
        enabled(m,True);merge_error=None
        if kind=='lora':
            merged=merged_lora(m).cuda()
            with torch.no_grad():
                a=m(x).logits;b=merged(x).logits;merge_error=float((a-b).abs().max());torch.testing.assert_close(a,b,atol=2e-4,rtol=1e-4)
            del merged
            assert all(torch.equal(m.state_dict()[n].cpu(),v) for n,v in state.items())
        result['arms'][kind]={'status':'passed','trainable_parameters':expected,'total_parameters':sum(p.numel() for p in m.parameters()),'initial_logits_exact':True,'backbone_unchanged':True,'reload_exact':True,'disabled_returns_pure':True,'merge_fp32_max_abs_error':merge_error,'steps':rows,'init_sha256':files[str(run/(kind+'_init.pt'))]}
        print(kind,{k:v for k,v in result['arms'][kind].items() if k!='steps'},flush=True);save()
        del m,opt,params;torch.cuda.empty_cache()
    verify();cfg.update(arms=['houlsby','lora'],adapter_width=6,lora_rank=7,lora_alpha=7,seed=42,branch_seed=43,file_hashes=files,prepared_path=str(run),formal_training=False,initialization='seed43 install after official load; zero up/B; no center fitting')
    (run/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    result.update(status='passed',source_files_unchanged=True,external_evaluated=False,test_evaluated=False);save()
if __name__=='__main__':main()
