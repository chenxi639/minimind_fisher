"""Recompute the preregistered comparisons from text-free exported scores."""
import json,math
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
scores=json.loads((root/'evidence/26D_document_scores.json').read_text())
summary=json.loads((root/'evidence/26D_summary.json').read_text())
indices=np.random.default_rng(20261004).integers(0,1024,size=(10000,1024))
for precision in ['fp32','bf16']:
    models={k:v[precision] for k,v in scores.items()};counts=np.array(models['pure']['document_tokens'])
    assert len(counts)==1024 and counts.sum()==219146
    for name,values in models.items():
        assert values['document_tokens']==counts.tolist()
        ce=sum(values['document_nll'])/sum(values['document_tokens'])
        assert abs(ce-summary['models'][name][precision]['ce'])<1e-12
    for left,right in [('k1','pure'),('current','pure'),('lora','pure'),('k1','lora'),('current','lora')]:
        diffs=[np.array(models[f'{s}/{left}']['document_nll'])-np.array(models['pure' if right=='pure' else f'{s}/{right}']['document_nll']) for s in [1729,2718,31415]]
        avg=np.mean(diffs,axis=0);delta=float(avg.sum()/counts.sum());ci=np.quantile(avg[indices].sum(1)/counts[indices].sum(1),[.005,.995])
        ref=summary['statistics'][precision][left+'_minus_'+right]
        assert abs(delta-ref['mean_delta_ce'])<1e-12
        np.testing.assert_allclose(ci,ref['document_bootstrap_ci99'],rtol=0,atol=1e-12)
        passed=bool(all(d.sum()<0 for d in diffs) and ci[1]<0 and (right!='pure' or delta<=math.log(.999)))
        assert passed==ref['criterion_met']
print('Verified: all model CE, 10 comparison intervals and preregistered decisions; no training.')
