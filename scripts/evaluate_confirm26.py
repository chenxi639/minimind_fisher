"""Locked Stage26D: final models, no training and no selection."""
import sys,json,hashlib,math,statistics,traceback
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind
from model.model_moment_controls import MomentControlMiniMind
from model.model_peft25 import install
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'runs/confirm26_train_20261004_230533';summary=json.loads((source/'comparison.json').read_text());manifest=json.loads((source/'manifest.json').read_text())
    seeds=[1729,2718,31415];kinds=['k1','current','lora'];assert summary['status']=='passed' and len(summary['runs'])==9
    files=dict(manifest['file_hashes'])
    for p in [source/'comparison.json',source/'manifest.json',Path(__file__),ROOT/'report/26D_protocol.md']:files[str(p)]=sha(p)
    for seed in seeds:
        exposure=[]
        for kind in kinds:
            key=f'{seed}/{kind}';r=summary['runs'][key];assert r['status']=='passed' and r['steps']==1024 and r['tokens']==4176264
            assert all(r[k] for k in ['backbone_unchanged','reload_exact','disabled_probe_exact'])
            files[str(source/key/'checkpoint.pt')]=r['checkpoint_sha256']
            history=json.loads((source/key/'history.json').read_text());assert [x['step'] for x in history]==list(range(1,1025));exposure.append([x['tokens'] for x in history])
        assert exposure[0]==exposure[1]==exposure[2]
    def verify():
        for p,h in files.items():assert sha(p)==h,p
    verify();run=ROOT/'runs'/('confirm26_eval_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir();print(run,flush=True)
    r={'status':'running','training_audit_passed':True,'file_hashes':files,'models':{},'optimizer_updates':0}
    def save():(run/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    save()
    try:
        tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
        docs=[json.loads(l) for l in (ROOT/'dataset/external_csl_26a/eval.jsonl').read_text(encoding='utf-8').splitlines()];assert len(docs)==1024
        rows=[]
        for doc in docs:
            ids=[tok.bos_token_id]+doc['content_ids']+[tok.eos_token_id];assert len(ids)<=340;rows.append(ids+[tok.pad_token_id]*(340-len(ids)))
        xall=torch.tensor(rows);yall=xall.clone();yall[xall==tok.pad_token_id]=-100;assert int((yall[:,1:]!=-100).sum())==219146
        state=torch.load(ROOT/'checkpoints/official_edba70e/download.pth',weights_only=True,map_location='cpu')
        for key in ['pure']+[f'{s}/{k}' for s in seeds for k in kinds]:
            kind=key.split('/')[-1];cfg=MiniMindConfig(fisher_k=1,fisher_dim=64,fisher_index=3,moment_mode='current')
            cls=FisherMiniMind if kind=='k1' else MomentControlMiniMind if kind=='current' else MiniMindForCausalLM
            m=cls(cfg);weights=dict(state)
            if kind=='lora':install(m,'lora')
            if key!='pure':
                ck=torch.load(source/key/'checkpoint.pt',weights_only=True,map_location='cpu');assert ck['steps']==1024 and ck['tokens']==4176264;weights.update(ck['adaptation']);del ck
            m.load_state_dict(weights,strict=True);m.cuda().eval().requires_grad_(False);r['models'][key]={}
            for precision in ['fp32','bf16']:
                sums=[];counts=[]
                with torch.no_grad():
                    for off in range(0,1024,2):
                        x,y=xall[off:off+2].cuda(),yall[off:off+2].cuda()
                        with (torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()):out=m(x)
                        loss=torch.nn.functional.cross_entropy(out.logits[:,:-1].float().transpose(1,2),y[:,1:],reduction='none',ignore_index=-100);assert torch.isfinite(loss).all()
                        sums.extend(loss.double().sum(1).cpu().tolist());counts.extend((y[:,1:]!=-100).sum(1).cpu().tolist())
                assert sum(counts)==219146;ce=sum(sums)/sum(counts)
                r['models'][key][precision]={'ce':ce,'ppl':math.exp(ce),'document_nll':sums,'document_tokens':counts};save();print(key,precision,ce,flush=True)
            assert all(torch.equal(v.cpu(),weights[n]) for n,v in m.state_dict().items());del m,weights;torch.cuda.empty_cache()
        rng=np.random.default_rng(20261004);ix=rng.integers(0,1024,size=(10000,1024));r['statistics']={}
        for precision in ['fp32','bf16']:
            models={k:v[precision] for k,v in r['models'].items()};n=np.array(models['pure']['document_tokens']);den=n[ix].sum(1);stats={}
            for left,right in [('k1','pure'),('current','pure'),('lora','pure'),('k1','lora'),('current','lora')]:
                values=[];arrays=[]
                for s in seeds:
                    a=models[f'{s}/{left}'];b=models['pure' if right=='pure' else f'{s}/{right}'];values.append(a['ce']-b['ce']);arrays.append(np.array(a['document_nll'])-np.array(b['document_nll']))
                avg=np.mean(arrays,axis=0);dist=avg[ix].sum(1)/den;delta=float(avg.sum()/n.sum());ci=np.quantile(dist,[.005,.995]).tolist()
                stats[left+'_minus_'+right]={'per_seed_delta_ce':dict(zip(map(str,seeds),values)),'mean_delta_ce':delta,'seed_sample_sd':statistics.stdev(values),'negative_seeds':sum(v<0 for v in values),'document_bootstrap_ci99':ci,'relative_ppl_change':math.expm1(delta),'criterion_met':bool(all(v<0 for v in values) and ci[1]<0 and (right!='pure' or delta<=math.log(.999)))}
            r['statistics'][precision]=stats
        r['precision_sign_reversal']={k:bool(v['mean_delta_ce']*r['statistics']['bf16'][k]['mean_delta_ce']<0) for k,v in r['statistics']['fp32'].items()}
        verify();r.update(status='passed',weights_unchanged=True,sealed_test_evaluated=False);save();print('PASSED',r['statistics']['fp32'],flush=True)
    except Exception:r.update(status='failed',error=traceback.format_exc());save();raise
if __name__=='__main__':main()
