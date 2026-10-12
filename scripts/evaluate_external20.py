import sys,json,hashlib,math,time,statistics
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind
from model.model_matched_mlp import MatchedMiniMind
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    data=ROOT/'dataset/external_wikipedia_20a';manifest=json.loads((data/'manifest.json').read_text())
    assert sha(data/'eval.jsonl')==manifest['data_sha256']
    assert sha(ROOT/'report/20A_external_protocol.md')==manifest['protocol_sha256']
    for n,d in manifest['code_hashes'].items():assert sha(ROOT/n)==d
    for n,d in manifest['tokenizer_hashes'].items():assert sha(ROOT/'model'/n)==d
    for name,v in manifest['models'].items():assert sha(Path(v.get('path',v.get('backbone'))))==v['sha256']
    run=ROOT/'runs'/('external20_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir();print(run,flush=True)
    r={'status':'running','manifest_sha256':sha(data/'manifest.json'),'models':{}};start=time.perf_counter()
    def save():(run/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    docs=[json.loads(line) for line in (data/'eval.jsonl').read_text(encoding='utf-8').splitlines()]
    xs=[];ys=[]
    for doc in docs:
        ids=[tok.bos_token_id]+doc['content_ids']+[tok.eos_token_id];ids+= [tok.pad_token_id]*(340-len(ids))
        x=torch.tensor(ids);y=x.clone();y[x==tok.pad_token_id]=-100;xs.append(x);ys.append(y)
    xall=torch.stack(xs);yall=torch.stack(ys);state=torch.load(manifest['models']['pure']['backbone'],map_location='cpu',weights_only=True)
    try:
        for name,v in manifest['models'].items():
            cls=MiniMindForCausalLM if name=='pure' else FisherMiniMind if name.startswith('fisher') else MatchedMiniMind
            m=cls(MiniMindConfig());weights=dict(state)
            if name!='pure':
                ck=torch.load(v['path'],map_location='cpu',weights_only=True);branch='fisher' if name.startswith('fisher') else 'adapter'
                weights.update({f'model.layers.3.{branch}.'+k:p for k,p in ck['branch'].items()})
            m.load_state_dict(weights,strict=True);m.cuda().eval();m.requires_grad_(False);r['models'][name]={}
            for precision in ['fp32','bf16']:
                sums=[];counts=[]
                with torch.no_grad():
                    for i in range(0,len(docs),2):
                        x,y=xall[i:i+2].cuda(),yall[i:i+2].cuda()
                        with (torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()):out=m(x)
                        losses=torch.nn.functional.cross_entropy(out.logits[:,:-1].float().transpose(1,2),y[:,1:],reduction='none',ignore_index=-100)
                        sums.extend(losses.double().sum(1).cpu().tolist());counts.extend((y[:,1:]!=-100).sum(1).cpu().tolist())
                ce=sum(sums)/sum(counts);assert math.isfinite(ce) and sum(counts)==manifest['prediction_tokens']
                r['models'][name][precision]={'ce':ce,'ppl':math.exp(ce),'document_nll':sums,'document_tokens':counts}
                save();print(name,precision,ce,math.exp(ce),flush=True)
            assert all(torch.equal(p.cpu(),weights[k].to(p.dtype)) for k,p in m.state_dict().items())
            del m,weights;torch.cuda.empty_cache()
        r['statistics']={};rng=np.random.default_rng(20261002);indices=rng.integers(0,len(docs),size=(2000,len(docs)))
        for precision in ['fp32','bf16']:
            n=np.array(r['models']['pure'][precision]['document_tokens']);den=n[indices].sum(1);baseline=r['models']['pure'][precision]
            differences={};perseed=[]
            for seed in [42,2026,3407]:
                f=r['models'][f'fisher_{seed}'][precision];m=r['models'][f'mlp_{seed}'][precision]
                perseed.append({'seed':seed,'fisher_minus_pure':f['ce']-baseline['ce'],'mlp_minus_pure':m['ce']-baseline['ce'],'fisher_minus_mlp':f['ce']-m['ce']})
            for comparison in ['fisher_minus_pure','fisher_minus_mlp']:
                arr=np.mean([np.array(r['models'][f'fisher_{s}'][precision]['document_nll'])-np.array(r['models']['pure' if comparison=='fisher_minus_pure' else f'mlp_{s}'][precision]['document_nll']) for s in [42,2026,3407]],axis=0)
                dist=arr[indices].sum(1)/den;values=[p[comparison] for p in perseed]
                differences[comparison]={'mean_delta_ce':float(arr.sum()/n.sum()),'document_bootstrap_ci95':np.quantile(dist,[.025,.975]).tolist(),'seed_sample_sd':statistics.stdev(values),'negative_seeds':sum(v<0 for v in values)}
            r['statistics'][precision]={'per_seed':perseed,**differences}
        main=r['statistics']['fp32']['fisher_minus_pure'];aux=r['statistics']['bf16']['fisher_minus_pure']
        r['external_improvement_supported']=main['negative_seeds']==3 and main['document_bootstrap_ci95'][1]<0 and aux['mean_delta_ce']<0
        for name,v in manifest['models'].items():assert sha(Path(v.get('path',v.get('backbone'))))==v['sha256']
        r.update(status='passed',seconds=time.perf_counter()-start,weights_unchanged=True,optimizer_updates=0)
    except Exception as exc:r.update(status='failed',error=repr(exc));raise
    finally:save();print(run,r['status'],flush=True)
if __name__=='__main__':main()
