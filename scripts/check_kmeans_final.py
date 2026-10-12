"""Frozen final checkpoint: full-dev residual ablation and sample component health."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import argparse,json,hashlib,math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig
from model.model_fisher import FisherMiniMind
from scripts.assignment_monitor import AssignmentMonitor

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    folder=a.run.parent;path=a.run/'checkpoint.pt';out=a.run/'final_health';out.mkdir(exist_ok=False)
    before=sha(path);ck=torch.load(path,map_location='cpu',weights_only=True)
    model=FisherMiniMind(MiniMindConfig()).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    cfg=json.loads((folder/'config.json').read_text());assert sha(folder/'dev.jsonl')==cfg['hashes']['dev.jsonl']
    initial=torch.load(folder/'fisher_init.pt',map_location='cpu',weights_only=True)
    branch=model.model.layers[3].fisher
    changes={n:float((v.detach().cpu()-initial['model.layers.3.fisher.'+n]).norm()) for n,v in branch.named_parameters()}
    tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    ds=PretrainDataset(str(folder/'dev.jsonl'),tok,max_length=340)
    observer=AssignmentMonitor(model,out/'assignment.json');result={'optimizer_steps':0,'source_sha256':before,'parameter_change_norms':changes}
    full={'tokens':0,'q_sum':torch.zeros(8,device='cuda'),'hard':torch.zeros(8,device='cuda',dtype=torch.long),'entropy_sum':0.,'max_sum':0.}
    def occupancy(layer,args,output):
        if observer.scale!=1:return
        with torch.autocast('cuda',enabled=False):
            x=F.linear(layer.norm(args[0].float()),layer.input.weight.float())
            z=layer.w[None,None]*(x[:,:,None]+layer.b[None,None])
            q=(-.5*z.square().sum(-1)).softmax(-1)[observer.mask]
            full['tokens']+=len(q);full['q_sum']+=q.sum(0);full['hard']+=torch.bincount(q.argmax(-1),minlength=8)
            full['entropy_sum']+=float(-(q*q.clamp_min(1e-30).log()).sum())
            full['max_sum']+=float(q.max(-1).values.sum())
    full_handle=branch.register_forward_hook(occupancy)
    with torch.no_grad():
        for scale in [1.,0.]:
            observer.scale=scale
            if scale==1:observer.start()
            nll=0.;count=0
            for i in range(0,len(ds),2):
                rows=[ds[i],ds[i+1]];x=torch.stack([r[0] for r in rows]).cuda();y=torch.stack([r[1] for r in rows]).cuda()
                observer.mask=torch.zeros_like(y,dtype=torch.bool);observer.mask[:,:-1]=y[:,1:]!=-100
                with torch.autocast('cuda',dtype=torch.bfloat16):prediction=model(x,labels=y)
                n=int((y[:,1:]!=-100).sum());nll+=float(prediction.loss)*n;count+=n
            if scale==1:observer.finish()
            result[str(int(scale))]={'ce':nll/count,'ppl':math.exp(nll/count),'tokens':count}
    observer.handle.remove()
    full_handle.remove()
    result['full_dev_assignments']={'tokens':full['tokens'],'soft_occupancy':(full['q_sum']/full['tokens']).tolist(),
        'hard_counts':full['hard'].tolist(),'entropy':full['entropy_sum']/full['tokens'],'mean_max_q':full['max_sum']/full['tokens']}
    history=json.loads((a.run/'history.json').read_text())
    assert abs(result['1']['ce']-history[-1]['dev']['ce'])<1e-6
    assert all(torch.equal(v.cpu(),ck['model'][k]) for k,v in model.state_dict().items())
    assert sha(path)==before
    result.update(status='passed',checkpoint_unchanged=True,normal_ce_reproduced=True,
                  disabled_minus_normal_ce=result['0']['ce']-result['1']['ce'])
    (out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
