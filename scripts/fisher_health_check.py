"""Read-only parameter health probe: backward is allowed, optimizer updates are not."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import json, math, hashlib
from datetime import datetime
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig
from model.model_fisher import FisherMiniMind, FisherResidual
from model.model_fisher_window import WindowFisherMiniMind

def sha(path):
    d=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):d.update(block)
    return d.hexdigest()

def pair_metrics(v):
    v=v.detach().float();idx=torch.triu_indices(len(v),len(v),1,device=v.device)
    cos=F.normalize(v,dim=-1)@F.normalize(v,dim=-1).T
    dist=torch.cdist(v,v)/math.sqrt(v.shape[1])
    return {'mean_pair_cosine':float(cos[idx[0],idx[1]].mean()),
            'mean_pair_rms_distance':float(dist[idx[0],idx[1]].mean())}

def assignments(layer,h):
    x=F.linear(layer.norm(h.float()),layer.input.weight.float())
    z=layer.w.float()[None,None]*(x[:,:,None]+layer.b.float()[None,None])
    return (-.5*z.square().sum(-1)).softmax(-1),z

def q_summary(q):
    usage=q.mean(0);center=q-usage
    cov=center.T@center/max(1,len(q)-1)
    std=cov.diag().sqrt();corr=cov/(std[:,None]*std[None,:]).clamp_min(1e-20)
    return {'tokens':len(q),'entropy':float(-(q*q.clamp_min(1e-30).log()).sum(-1).mean()),
            'max_entropy':math.log(8),'mean_max_q':float(q.max(-1).values.mean()),
            'mean_usage':usage.tolist(),'token_std_per_component':q.std(0).tolist(),
            'mean_token_l1_from_uniform':float((q-1/8).abs().sum(-1).mean()),
            'hard_winner_counts':torch.bincount(q.argmax(-1),minlength=8).tolist(),
            'centered_response_correlation':corr.tolist()}

def main():
    torch.manual_seed(42)
    folder=ROOT/'runs/pair_ready_20260929_181458'
    run=ROOT/'runs'/('fisher_health_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir()
    cfg=json.loads((folder/'config.json').read_text())
    assert sha(folder/'dev.jsonl')==cfg['hashes']['dev.jsonl']
    init=torch.load(folder/'fisher_init.pt',weights_only=True,map_location='cpu')
    init_local={k.split('fisher.',1)[1]:v for k,v in init.items()}
    initial_branch=FisherResidual(768,8,64).cuda().eval();initial_branch.load_state_dict(init_local)
    tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    ds=PretrainDataset(str(folder/'dev.jsonl'),tok,max_length=340)
    indices=list(range(0,1024,64))
    batches=[]
    for i in range(0,len(indices),2):
        rows=[ds[j] for j in indices[i:i+2]]
        batches.append((torch.stack([r[0] for r in rows]).cuda(),torch.stack([r[1] for r in rows]).cuda()))
    specs=[('initial',None,FisherMiniMind),
           ('prefix_k8',folder/'fisher_train_20260929_193424/checkpoint.pt',FisherMiniMind),
           ('window32',ROOT/'runs/w32_ready_20260930_214307/fisher_train_20260930_214715/checkpoint.pt',WindowFisherMiniMind)]
    result={'optimizer_steps':0,'indices':indices,'split':'dev','batch_size':2,'models':{},'status':'running'}
    for name,path,cls in specs:
        print('CHECK',name,flush=True)
        source_paths=[path] if path else [folder/'backbone_init.pt',folder/'fisher_init.pt']
        before_hash={str(p):sha(p) for p in source_paths}
        if path:
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            weights=checkpoint['model'];saved_optimizer=checkpoint['optimizer'];del checkpoint
        else:
            weights=torch.load(folder/'backbone_init.pt',map_location='cpu',weights_only=True);weights.update(init)
            saved_optimizer=None
        model=cls(MiniMindConfig(fisher_window=32)).cuda().eval();model.load_state_dict(weights,strict=True)
        branch=model.model.layers[3].fisher
        historical={}
        if saved_optimizer is not None:
            ids=[i for group in saved_optimizer['param_groups'] for i in group['params']]
            assert len(ids)==len(list(model.parameters()))
            for (pname,p),pid in zip(model.named_parameters(),ids):
                if '.fisher.' in pname:
                    state=saved_optimizer['state'][pid]
                    assert state['exp_avg'].shape==p.shape
                    historical[pname.split('.fisher.')[1]]={'step':float(state['step']),
                        'exp_avg_norm':float(state['exp_avg'].norm()),'exp_avg_sq_norm':float(state['exp_avg_sq'].norm())}
            del saved_optimizer
        changes={n:{'norm':float(p.detach().float().norm()),'delta_from_init_norm':float((p.detach().cpu()-init_local[n]).norm())}
                 for n,p in branch.named_parameters()}
        params={'w':pair_metrics(branch.w),'b':pair_metrics(branch.b),
                'output_component_blocks':pair_metrics(branch.output.weight.detach().reshape(768,2,8,64).permute(2,0,1,3).reshape(8,-1))}
        grad_rows=[];qs=[];qi=[];qfixed=[];feature_cos=[];ratios=[];scale=1.;collect=True;mask=None;captured=None
        def hook(layer,args,output):
            nonlocal captured
            h=args[0];delta,state=output
            if collect:
                with torch.no_grad(),torch.autocast('cuda',enabled=False):
                    q,z=assignments(layer,h);q0,_=assignments(initial_branch,h)
                    qs.append(q[mask].cpu());qi.append(q0[mask].cpu())
                    x=F.linear(layer.norm(h.float()),layer.input.weight.float())
                    z0=initial_branch.w[None,None]*(x[:,:,None]+initial_branch.b[None,None])
                    qfixed.append((-.5*z0.square().sum(-1)).softmax(-1)[mask].cpu())
                    a=q[...,None]*z;b=q[...,None]*(z.square()-1)/math.sqrt(2)
                    def aggregate(v):
                        c=v.cumsum(1)
                        if hasattr(layer,'window'):
                            w=layer.window;c=c-torch.cat([torch.zeros_like(c[:,:w]),c[:,:-w]],1)
                        return c
                    count=torch.arange(1,h.shape[1]+1,device=h.device).float()
                    if hasattr(layer,'window'):count=count.clamp_max(layer.window)
                    # Inputs are right-padded; valid causal prediction positions see no pads.
                    v=(torch.cat([aggregate(a),aggregate(b)],-1)/count[None,:,None,None])[mask]
                    v=v/torch.sqrt(v.abs()+1e-6)
                    vn=F.normalize(v,dim=-1)
                    sims=vn@vn.transpose(-1,-2);ix=torch.triu_indices(8,8,1,device=h.device)
                    feature_cos.append(sims[:,ix[0],ix[1]].cpu().flatten())
                    ratios.append((delta.float()[mask].norm(dim=-1)/h.float()[mask].norm(dim=-1).clamp_min(1e-12)).cpu())
                    if captured is None:captured=h.detach()
            return delta*scale,state
        handle=branch.register_forward_hook(hook)
        total_nll=0.;nt=0
        for x,y in batches:
            mask=torch.zeros_like(y,dtype=torch.bool);mask[:,:-1]=y[:,1:]!=-100
            model.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,labels=y)
            out.loss.backward()
            n=int(mask.sum());total_nll+=float(out.loss.detach())*n;nt+=n
            row={}
            for pname,p in branch.named_parameters():
                g=p.grad
                assert g is not None and torch.isfinite(g).all(),pname
                if name!='initial':assert g.norm()>0,pname
                row[pname]={'present':g is not None,'finite':bool(torch.isfinite(g).all()) if g is not None else False,
                            'l2':float(g.norm()) if g is not None else None,
                            'rms':float(g.square().mean().sqrt()) if g is not None else None,
                            'nonzero_fraction':float((g!=0).float().mean()) if g is not None else 0}
                if pname in ['w','b'] and g is not None:
                    row[pname]['per_component_l2']=g.norm(dim=-1).tolist()
                    row[pname]['component_gradient_pairs']=pair_metrics(g)
            row['global_unclipped_l2']=math.sqrt(sum(float(p.grad.float().square().sum()) for p in model.parameters() if p.grad is not None))
            grad_rows.append(row)
        collect=False
        ces={'1':total_nll/nt}
        with torch.no_grad():
            for scale in [0.,2.]:
                nll=0.
                for x,y in batches:
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,labels=y)
                    nll+=float(out.loss)*int((y[:,1:]!=-100).sum())
                ces[str(int(scale))]=nll/nt
        handle.remove()
        unchanged=all(torch.equal(v.detach().cpu(),weights[k]) for k,v in model.state_dict().items())
        assert unchanged and all(sha(p)==before_hash[str(p)] for p in source_paths)
        entry={'ce_by_residual_scale':ces,'valid_tokens':nt,'gradients_unclipped_per_batch':grad_rows,
               'saved_adam_state':historical,
               'parameter_changes':changes,'parameter_component_pairs':params,
               'assignments':q_summary(torch.cat(qs)),
               'initial_branch_on_same_hidden':q_summary(torch.cat(qi)),
               'initial_w_b_on_same_current_descriptors':q_summary(torch.cat(qfixed)),
               'mean_component_statistic_cosine':float(torch.cat(feature_cos).mean()),
               'mean_residual_hidden_norm_ratio':float(torch.cat(ratios).mean()),
               'source_sha256':before_hash,'source_and_memory_weights_unchanged':unchanged}
        result['models'][name]=entry
        (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(name,ces,'entropy',entry['assignments']['entropy'],'feature cosine',entry['mean_component_statistic_cosine'],flush=True)
        del model,weights,branch,out;torch.cuda.empty_cache()
    result['status']='passed'
    result['script_sha256']=sha(Path(__file__))
    (run/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('RESULT',run,flush=True)

if __name__=='__main__':main()
