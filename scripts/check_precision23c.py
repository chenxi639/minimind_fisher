"""Fixed Stage23B weights, full development set, zero training."""
import sys,json,math,hashlib
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind
from model.model_moment import MomentMiniMind
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.set_float32_matmul_precision('highest')
    old=ROOT/'runs/mechanism_pair_20261003_113236'
    manifest=json.loads((old/'manifest.json').read_text());comparison=json.loads((old/'comparison.json').read_text())
    files=dict(manifest['file_hashes'])
    for arm in ['k1','moment']:files[str(old/arm/'checkpoint.pt')]=comparison[arm]['checkpoint_sha256']
    for p in [Path(__file__),ROOT/'report/23C_protocol.md',old/'manifest.json',old/'comparison.json']:files[str(p)]=sha(p)
    def verify():
        for p,d in files.items():assert sha(p)==d,p
    verify();run=ROOT/'runs'/('precision23c_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir();print(run,flush=True)
    result={'status':'running','optimizer_updates':0,'file_hashes':files,'tf32':False,'models':{}}
    def save():(run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save();tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    ds=PretrainDataset(str(Path(manifest['source_preparation'])/'dev.jsonl'),tok,max_length=340)
    assert len(ds)==1024
    state=torch.load(manifest['backbone_path'],map_location='cpu',weights_only=True)
    for arm,cls in [('pure',MiniMindForCausalLM),('k1',FisherMiniMind),('moment',MomentMiniMind)]:
        cfg=MiniMindConfig();cfg.fisher_k=1;cfg.fisher_dim=64;cfg.fisher_index=3
        m=cls(cfg);weights=dict(state)
        if arm!='pure':
            ck=torch.load(old/arm/'checkpoint.pt',map_location='cpu',weights_only=True)
            weights.update({'model.layers.3.fisher.'+k:v for k,v in ck['branch'].items()})
        m.load_state_dict(weights,strict=True);m.cuda().float().eval().requires_grad_(False)
        result['models'][arm]={}
        for precision in ['fp32','bf16']:
            repeats=[]
            for repeat in range(2):
                losses=[];counts=[]
                with torch.no_grad():
                    for i in range(0,len(ds),2):
                        rows=[ds[i],ds[i+1]];x,y=[torch.stack([r[k] for r in rows]).cuda() for k in (0,1)]
                        with (torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()):out=m(x,labels=y)
                        assert torch.isfinite(out.loss);losses.append(float(out.loss));counts.append(int((y[:,1:]!=-100).sum()))
                assert sum(counts)==265051
                ce=sum(l*n for l,n in zip(losses,counts))/sum(counts)
                repeats.append({'ce':ce,'ppl':math.exp(ce),'tokens':sum(counts),'batch_ce':losses,'batch_tokens':counts})
                print(arm,precision,repeat,ce,flush=True)
            result['models'][arm][precision]={'repeats':repeats,'max_repeat_batch_difference':max(abs(a-b) for a,b in zip(repeats[0]['batch_ce'],repeats[1]['batch_ce']))}
            if precision=='bf16':assert repeats[0]['ce']==(comparison['baseline'] if arm=='pure' else comparison[arm]['final_dev'])['ce']
            save()
        assert all(torch.equal(v.cpu(),weights[k].to(v.dtype)) for k,v in m.state_dict().items())
        del m,weights;torch.cuda.empty_cache()
    result['differences']={}
    for p in ['fp32','bf16']:
        ce={a:result['models'][a][p]['repeats'][0]['ce'] for a in result['models']}
        result['differences'][p]={'k1_minus_pure':ce['k1']-ce['pure'],'moment_minus_pure':ce['moment']-ce['pure'],'k1_minus_moment':ce['k1']-ce['moment']}
    verify();result.update(status='passed',weights_unchanged=True,external_evaluated=False,test_evaluated=False);save()
    print(json.dumps(result['differences']),flush=True)
if __name__=='__main__':main()
