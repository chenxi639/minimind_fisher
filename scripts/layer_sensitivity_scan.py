"""Observed-label tokenwise empirical Fisher traces and hidden gradients; no updates."""
import sys,json,hashlib,time,argparse,math
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind
from model.model_matched_mlp import MatchedMiniMind
from dataset.lm_dataset import PretrainDataset

def sha(path):
    d=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):d.update(b)
    return d.hexdigest()

def parameter_trace(params):
    total=None
    for param in params:
        assert param.grad is not None and torch.isfinite(param.grad).all()
        term=param.grad.float().square().sum()
        total=term if total is None else total+term
    return float(total) if total is not None else 0.

def main():
    p=argparse.ArgumentParser();p.add_argument('--mlp-run',type=Path,required=True);a=p.parse_args()
    run=ROOT/'runs'/('layer_scan_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir()
    source=ROOT/'runs/pair_ready_20260929_181458';cfg=json.loads((source/'config.json').read_text())
    assert sha(source/'dev.jsonl')==cfg['hashes']['dev.jsonl']
    tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True);ds=PretrainDataset(str(source/'dev.jsonl'),tok,max_length=340)
    rng=np.random.default_rng(42);selected=[]
    for i in range(0,1024,64):
        x,y=ds[i];valid=(y[1:]!=-100).nonzero().flatten().numpy()
        selected.append({'record':i,'positions':sorted(rng.choice(valid,4,replace=False).tolist())})
    specs=[('baseline',source/'baseline_train_20260929_191604/checkpoint.pt',MiniMindForCausalLM),
           ('fisher_mid',ROOT/'runs/kmeans_ready_20260930_224711/fisher_train_20260930_224923/checkpoint.pt',FisherMiniMind),
           ('matched_mlp',a.mlp_run/'checkpoint.pt',MatchedMiniMind)]
    result={'status':'running','optimizer_steps':0,'sampling':selected,'models':{},'definition':'mean of individual observed-token NLL squared gradients, not squared batch gradient; traces only, not full FIM'}
    start=time.perf_counter()
    for name,path,cls in specs:
        t0=time.perf_counter();before=sha(path);ck=torch.load(path,map_location='cpu',weights_only=True)
        model=cls(MiniMindConfig()).cuda().eval();model.load_state_dict(ck['model'],strict=True)
        hidden={};handles=[]
        def make_hook(index):
            def hook(layer,args,out):
                hidden[index]=out[0];out[0].retain_grad()
            return hook
        for i,layer in enumerate(model.model.layers):handles.append(layer.register_forward_hook(make_hook(i)))
        groups={str(i):[] for i in range(8)};groups['branch']=[]
        for n,param in model.named_parameters():
            if n.startswith('model.layers.'):
                group='branch' if '.fisher.' in n or '.adapter.' in n else n.split('.')[2]
                groups[group].append(param)
        counts={g:sum(p.numel() for p in ps) for g,ps in groups.items()};rows=[]
        for sample in selected:
            x,y=ds[sample['record']];x=x[None].cuda();y=y[None].cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x)
            for j,pos in enumerate(sample['positions']):
                model.zero_grad(set_to_none=True)
                for h in hidden.values():h.grad=None
                loss=F.cross_entropy(out.logits[:,pos].float(),y[:,pos+1])
                loss.backward(retain_graph=j<3)
                traces={}
                for group,ps in groups.items():
                    traces[group]=parameter_trace(ps)
                hs=[]
                for i in range(8):
                    h=hidden[i].detach().float();g=hidden[i].grad.float()
                    assert torch.isfinite(g).all()
                    # Gradients at future positions must vanish for this target.
                    assert float(g[:,pos+1:].abs().max())==0 if pos+1<g.shape[1] else True
                    gh=g[0,:pos+1];hh=h[0,:pos+1];mass=gh.square().sum(-1);tot=mass.sum()
                    grad_l2=tot.sqrt();rms=hh.square().mean().sqrt()
                    hs.append({'layer':i,'grad_l2':float(grad_l2),'hidden_rms':float(rms),
                        'relative_isotropic_sensitivity':float(grad_l2*rms),
                        'current_position_mass_fraction':float(mass[-1]/tot.clamp_min(1e-30)),
                        'top10pct_positions_mass_fraction':float(mass.topk(max(1,math.ceil(len(mass)*.1))).values.sum()/tot.clamp_min(1e-30))})
                rows.append({'record':sample['record'],'position':pos,'nll':float(loss.detach()),'parameter_trace':traces,'hidden':hs})
            del out
        for handle in handles:handle.remove()
        assert all(torch.equal(v.cpu(),ck['model'][k]) for k,v in model.state_dict().items()) and sha(path)==before
        summary=[]
        for i in range(8):
            traces=np.array([r['parameter_trace'][str(i)] for r in rows]);sens=np.array([r['hidden'][i]['relative_isotropic_sensitivity'] for r in rows])
            summary.append({'layer':i,'parameters':counts[str(i)],'empirical_fisher_trace':float(traces.mean()),
                'empirical_fisher_mean_diagonal':float(traces.mean()/counts[str(i)]),
                'trace_first8_records':float(traces[:32].mean()),'trace_last8_records':float(traces[32:].mean()),
                'hidden_grad_l2_mean':float(np.mean([r['hidden'][i]['grad_l2'] for r in rows])),
                'hidden_relative_sensitivity_mean':float(sens.mean()),
                'hidden_relative_first8':float(sens[:32].mean()),'hidden_relative_last8':float(sens[32:].mean()),
                'current_position_mass_mean':float(np.mean([r['hidden'][i]['current_position_mass_fraction'] for r in rows])),
                'top10pct_mass_mean':float(np.mean([r['hidden'][i]['top10pct_positions_mass_fraction'] for r in rows]))})
        result['models'][name]={'checkpoint':str(path.resolve()),'source_sha256':before,'source_and_memory_unchanged':True,
            'seconds':time.perf_counter()-t0,'targets':len(rows),'layers':summary,'branch_parameters':counts['branch'],
            'branch_trace_mean':float(np.mean([r['parameter_trace']['branch'] for r in rows])),'rows':rows}
        (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(name,'seconds',result['models'][name]['seconds'],'layers',[(r['layer'],r['empirical_fisher_trace'],r['hidden_relative_sensitivity_mean']) for r in summary],flush=True)
        hidden.clear();del model,ck,groups;torch.cuda.empty_cache()
    result.update(status='passed',seconds=time.perf_counter()-start,script_sha256=sha(Path(__file__)))
    (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print('SCAN',run,flush=True)
if __name__=='__main__':main()
