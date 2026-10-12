"""Report every locked seed; never select a best checkpoint or seed."""
import argparse,json,statistics,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--seed2026',type=Path,required=True);p.add_argument('--seed3407',type=Path,required=True);a=p.parse_args()
    sources={42:ROOT/'runs/frozen_pair_20261002_145450',2026:a.seed2026.resolve(),3407:a.seed3407.resolve()}
    rows=[];hashes={};baseline=None
    for seed,folder in sources.items():
        comparison=json.loads((folder/'comparison.json').read_text());assert comparison['status']=='passed'
        assert comparison['per_step_token_exposure_equal'] and comparison['initial_dev_exact']
        manifest=json.loads((folder/'manifest.json').read_text());assert manifest['seed']==seed
        if baseline is None:baseline=comparison['baseline']
        else:assert baseline==comparison['baseline']
        row={'seed':seed,'role':'retrospective_exploratory_anchor' if seed==42 else 'prospective_replication'}
        for arm in ['fisher','mlp']:
            result=comparison[arm];assert result['status']=='passed' and result['steps']==1024 and result['tokens']==4176264
            assert result['backbone_exact_unchanged'] and result['checkpoint_exact']
            digest=hashlib.sha256((folder/arm/'checkpoint.pt').read_bytes()).hexdigest();assert digest==result['checkpoint_sha256']
            hashes[str(folder/arm/'checkpoint.pt')]=digest
            row[arm]=result['final_dev'];row['delta_'+arm]=result['final_dev']['ce']-baseline['ce']
        row['delta_fisher_minus_mlp']=row['fisher']['ce']-row['mlp']['ce']
        health=json.loads((folder/'fisher/health.json').read_text())
        row['fisher_occupancy']=health['full_dev_assignments'];rows.append(row)
    stats={}
    for key in ['delta_fisher','delta_mlp','delta_fisher_minus_mlp']:
        values=[r[key] for r in rows];stats[key]={'mean':statistics.mean(values),'sample_sd':statistics.stdev(values),'min':min(values),'max':max(values),'negative_count':sum(v<0 for v in values),'n':3,'all_negative':all(v<0 for v in values)}
    result={'status':'passed','baseline':baseline,'rows':rows,'statistics':stats,'checkpoint_hashes':hashes,'primary_endpoint_step':1024,'test_evaluated':False,'claims':'within potentially seen corpus; N=3 descriptive, seed42 known before preregistration'}
    out=ROOT/'runs/frozen_multiseed_summary.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
