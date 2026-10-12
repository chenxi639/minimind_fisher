"""Frozen development-set ablation; never trains or accesses test records."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import datasets
import json, math, hashlib
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig
from model.model_fisher import FisherMiniMind
from dataset.lm_dataset import PretrainDataset

def main():
    folder=ROOT/'runs/pair_ready_20260929_181458'
    run=folder/('diagnostic_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir()
    def save(n,v):(run/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    cfg=json.loads((folder/'config.json').read_text())
    assert sha(folder/'dev.jsonl')==cfg['hashes']['dev.jsonl']
    for name,digest in cfg['code_hashes'].items():assert sha(ROOT/name)==digest
    path=folder/'fisher_train_20260929_193424/checkpoint.pt'
    checkpoint_hash=sha(path)
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=FisherMiniMind(MiniMindConfig()).cuda().eval();model.load_state_dict(checkpoint['model'],strict=True)
    del checkpoint
    model.requires_grad_(False)
    module=model.model.layers[3].fisher
    initial=torch.load(folder/'fisher_init.pt',map_location='cpu',weights_only=True)
    changes={name:{'norm':float(v.float().norm()),'change_norm':float((v.cpu()-initial['model.layers.3.fisher.'+name]).float().norm())} for name,v in module.state_dict().items()}
    stats={'n':0,'ratio_sum':0.,'delta_sq':0.,'hidden_sq':0.,'entropy_sum':0.,'max_sum':0.,'usage':torch.zeros(8,device='cuda')}
    active_mask=None;disabled=False
    def hook(layer,args,output):
        h=args[0];delta,state=output
        if disabled:return torch.zeros_like(delta),state
        valid=active_mask
        with torch.autocast('cuda',enabled=False):
            x=F.linear(layer.norm(h.float()),layer.input.weight.float())
            z=layer.w[None,None]*(x[:,:,None]+layer.b[None,None])
            q=(-.5*z.square().sum(-1)).softmax(-1)[valid]
            dh=delta.float()[valid];hh=h.float()[valid]
            stats['n']+=len(q)
            stats['ratio_sum']+=float((dh.norm(dim=-1)/hh.norm(dim=-1).clamp_min(1e-12)).sum())
            stats['delta_sq']+=float(dh.square().sum());stats['hidden_sq']+=float(hh.square().sum())
            stats['entropy_sum']+=float(-(q*q.clamp_min(1e-30).log()).sum())
            stats['max_sum']+=float(q.max(-1).values.sum());stats['usage']+=q.sum(0)
    handle=module.register_forward_hook(hook)
    tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    ds=PretrainDataset(str(folder/'dev.jsonl'),tok,max_length=340)
    result={'status':'running','trained':False,'optimizer_steps':0,'checkpoint_sha256':checkpoint_hash}
    save('result.json',result)
    try:
        for mode in ['normal','disabled']:
            disabled=mode=='disabled';rows=[];original_nll=0.
            with torch.inference_mode():
                for i in range(0,len(ds),2):
                    batch=[ds[j] for j in range(i,i+2)]
                    x=torch.stack([r[0] for r in batch]).cuda();y=torch.stack([r[1] for r in batch]).cuda()
                    active_mask=torch.zeros_like(y,dtype=torch.bool);active_mask[:,:-1]=y[:,1:]!=-100
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        out=model(x,labels=y);logits=out.logits
                    original_nll+=out.loss.item()*int((y[:,1:]!=-100).sum())
                    losses=F.cross_entropy(logits[:,:-1].float().reshape(-1,6400),y[:,1:].reshape(-1),ignore_index=-100,reduction='none').reshape(2,-1)
                    assert torch.isfinite(losses).all()
                    for j in range(2):rows.append({'index':i+j,'nll':float(losses[j].sum()),'tokens':int((y[j,1:]!=-100).sum())})
            nll=sum(r['nll'] for r in rows);count=sum(r['tokens'] for r in rows)
            result[mode]={'ce':original_nll/count,'ppl':math.exp(original_nll/count),'tokens':count,'per_example_sum_ce':nll/count}
            save(mode+'_per_example.json',rows);print(mode,result[mode],flush=True)
        usage=(stats.pop('usage')/stats['n']).cpu().tolist()
        result['branch']={'valid_prediction_positions':stats['n'],'mean_token_residual_norm_ratio':stats['ratio_sum']/stats['n'],
            'global_residual_norm_ratio':math.sqrt(stats['delta_sq']/stats['hidden_sq']),
            'mean_assignment_entropy':stats['entropy_sum']/stats['n'],'max_entropy':math.log(8),
            'mean_max_assignment':stats['max_sum']/stats['n'],'mean_usage':usage,
            'effective_components_from_mean_usage':math.exp(-sum(p*math.log(max(p,1e-30)) for p in usage))}
        previous=json.loads((folder/'fisher_train_20260929_193424/history.json').read_text())[-1]['dev']['ce']
        assert abs(result['normal']['ce']-previous)<1e-5
        assert sha(path)==checkpoint_hash
        result.update(status='passed',normal_reproduced=True,checkpoint_unchanged=True,parameter_changes=changes,
                      disabled_minus_normal_ce=result['disabled']['ce']-result['normal']['ce'])
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        handle.remove();save('result.json',result);print('RUN',run);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
