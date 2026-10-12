"""Summarize completed paired runs without opening the held-out test split."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('mlp',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    runs=[]
    for folder in [a.baseline,a.mlp]:
        result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
        history=json.loads((folder/'history.json').read_text(encoding='utf-8'))
        assert result['status']=='passed' and not result['smoke'] and result['steps']==1024
        runs.append((result,{r['step']:r for r in history}))
    b,bh=runs[0];f,fh=runs[1]
    assert b['arm']=='baseline' and f['arm']=='mlp'
    assert b['tokens']==f['tokens']
    assert bh[0]['dev']==fh[0]['dev']
    assert all(bh[i]['tokens']==fh[i]['tokens'] for i in range(1,1025))
    rows=[]
    for step in [0,256,512,1024]:
        x,y=bh[step]['dev'],fh[step]['dev']
        assert x['tokens']==y['tokens']
        rows.append(dict(step=step,baseline_ce=x['ce'],mlp_ce=y['ce'],delta_ce=y['ce']-x['ce'],baseline_ppl=x['ppl'],mlp_ppl=y['ppl']))
    summary=dict(baseline=str(a.baseline.resolve()),mlp=str(a.mlp.resolve()),same_initial_dev=True,
                 equal_token_exposure=True,training_tokens=b['tokens'],curves=rows,
                 seconds={'baseline':b['seconds'],'mlp':f['seconds'],'ratio':f['seconds']/b['seconds']},
                 peak_mib={'baseline':b['peak_mib'],'mlp':f['peak_mib']},
                 final_ppl_relative_change=rows[-1]['mlp_ppl']/rows[-1]['baseline_ppl']-1,
                 limitation='One seed, development-set screening; total time includes evaluation and saving, not pure training throughput; test not evaluated.')
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
