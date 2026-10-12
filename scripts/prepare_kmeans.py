"""Train-only descriptor fitting; only Fisher b differs from the shared initialization."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json,hashlib,shutil,time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig
from model.model_fisher import FisherMiniMind
from dataset.lm_dataset import PretrainDataset

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.perf_counter();torch.manual_seed(42)
    source=ROOT/'runs/pair_ready_20260929_181458'
    target=ROOT/'runs'/('kmeans_ready_'+datetime.now().strftime('%Y%m%d_%H%M%S'));target.mkdir()
    cfg=json.loads((source/'config.json').read_text())
    for n,d in cfg['hashes'].items():
        assert sha(source/n)==d,n
        shutil.copy2(source/n,target/n)
    shutil.copy2(source/'manifest.json',target/'manifest.json')
    weights=torch.load(source/'backbone_init.pt',weights_only=True,map_location='cpu')
    init=torch.load(source/'fisher_init.pt',weights_only=True,map_location='cpu');weights.update(init)
    model=FisherMiniMind(MiniMindConfig()).cuda().eval();model.load_state_dict(weights,strict=True)
    branch=model.model.layers[3].fisher
    tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    ds=PretrainDataset(str(source/'train.jsonl'),tok,max_length=340)
    indices=np.random.default_rng(42).choice(len(ds),512,replace=False).tolist()
    chunks=[];locations=[];active=None
    def hook(layer,args,output):
        with torch.autocast('cuda',enabled=False):
            x=F.linear(layer.norm(args[0].float()),layer.input.weight.float())
        chunks.append(x[active].cpu())
    handle=branch.register_forward_hook(hook)
    with torch.no_grad():
        for offset in range(0,len(indices),2):
            ids=indices[offset:offset+2];rows=[ds[j] for j in ids]
            x=torch.stack([r[0] for r in rows]).cuda();y=torch.stack([r[1] for r in rows]).cuda()
            active=torch.zeros_like(y,dtype=torch.bool);active[:,:-1]=y[:,1:]!=-100
            for j,t in active.nonzero().cpu().tolist():locations.append([ids[j],t])
            with torch.autocast('cuda',dtype=torch.bfloat16):model(x)
    handle.remove()
    pool=torch.cat(chunks).numpy();chosen=np.random.default_rng(43).choice(len(pool),30000,replace=False)
    descriptors=pool[chosen];assert np.isfinite(descriptors).all()
    with threadpool_limits(limits=4):
        km=KMeans(n_clusters=8,init='k-means++',n_init=10,max_iter=300,random_state=42,algorithm='lloyd').fit(descriptors)
    counts=np.bincount(km.labels_,minlength=8);assert (counts>0).all()
    center=torch.from_numpy(km.cluster_centers_);assert torch.isfinite(center).all()
    modified={k:v.clone() for k,v in init.items()};modified['model.layers.3.fisher.b']=-center
    assert all(torch.equal(v,modified[k]) for k,v in init.items() if not k.endswith('.b'))
    # Zero output projection keeps original logits exactly unchanged.
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):before=model(x).logits
    model.load_state_dict({**weights,**modified},strict=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):after=model(x).logits
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    torch.save(modified,target/'fisher_init.pt')
    torch.save({'descriptors':torch.from_numpy(descriptors),'centers':center,
                'locations':torch.tensor([locations[i] for i in chosen])},target/'kmeans_fit.pt')
    distances=torch.cdist(torch.from_numpy(descriptors),center).square()
    q=(-.5*distances).softmax(-1)
    result={'status':'passed','record_indices':indices,'pool_tokens':len(pool),'fit_tokens':30000,
            'cluster_counts':counts.tolist(),'center_pair_distances':torch.cdist(center,center).tolist(),
            'fit_assignment_entropy':float(-(q*q.clamp_min(1e-30).log()).sum(-1).mean()),
            'fit_soft_occupancy':q.mean(0).tolist(),'fit_mean_max_assignment':float(q.max(-1).values.mean()),
            'inertia':float(km.inertia_),'iterations':int(km.n_iter_),'seconds':time.perf_counter()-start,
            'only_b_changed':True,'initial_logits_exact':True,'fit_file_sha256':sha(target/'kmeans_fit.pt')}
    cfg['hashes']['fisher_init.pt']=sha(target/'fisher_init.pt')
    cfg['initialization']='kmeans_centers_only';cfg['source_preparation']=str(source)
    cfg['code_hashes']={n:sha(ROOT/n) for n in ['model/model_minimind.py','model/model_fisher.py',
        'scripts/train_pair_configured.py','scripts/train_kmeans_monitored.py','scripts/assignment_monitor.py','scripts/prepare_kmeans.py']}
    (target/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (target/'preparation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(target,flush=True);print(json.dumps({k:v for k,v in result.items() if k not in ['record_indices','center_pair_distances']},indent=2))

if __name__=='__main__':main()
