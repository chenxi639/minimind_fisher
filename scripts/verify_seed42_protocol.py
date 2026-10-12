import sys,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from model.model_fisher import FisherResidual
from model.model_matched_mlp import MatchedResidual

def main():
    torch.set_num_threads(4)
    prep=ROOT/'runs/frozen_ready_20261002_124018';pair=ROOT/'runs/frozen_pair_20261002_145450'
    manifest=json.loads((pair/'manifest.json').read_text())
    for p,d in manifest['file_hashes'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==d,p
    source=Path(manifest['source_preparation'])
    assert torch.randperm(16384,generator=torch.Generator().manual_seed(42)).tolist()==json.loads((source/'order.json').read_text())
    for arm,cls in [('fisher',FisherResidual),('mlp',MatchedResidual)]:
        torch.manual_seed(43);branch=cls(768);old=torch.load(prep/(arm+'_init.pt'),weights_only=True)
        for k,v in branch.state_dict().items():
            if arm=='fisher' and k=='b':continue
            assert torch.equal(v,old[k]),(arm,k)
    fit=torch.load(prep/'kmeans_fit.pt',weights_only=True)
    with threadpool_limits(limits=4):km=KMeans(n_clusters=8,n_init=10,max_iter=300,random_state=42,algorithm='lloyd').fit(fit['descriptors'].numpy())
    centers=torch.from_numpy(km.cluster_centers_)
    error=float((centers-fit['centers']).abs().max());relative=float((centers-fit['centers']).norm()/fit['centers'].norm())
    assert error<=1e-6 and relative<=1e-6
    assert torch.equal(-fit['centers'],torch.load(prep/'fisher_init.pt',weights_only=True)['b'])
    result={'status':'passed_with_numerical_caveat','order42_exact':True,'branch43_exact_except_fitted_b':True,'kmeans_refit_saved_descriptors_exact':torch.equal(centers,fit['centers']),'kmeans_max_abs':error,'kmeans_relative_l2':relative,'original_centers_and_b_exact':True,'all_source_manifest_hashes_valid':True,'seed42_eligible':True,'initial_strict_check':'failed; separately diagnosed max_abs 5.96e-8; tolerance amendment registered before new training','limitation':'reuses saved descriptors, not independent re-extraction; multithread reduction can differ below 1e-6; original weights retained'}
    out=ROOT/'runs/seed42_protocol_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
