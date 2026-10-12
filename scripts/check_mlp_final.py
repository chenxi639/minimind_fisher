import sys,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig
from model.model_matched_mlp import MatchedMiniMind
from dataset.lm_dataset import PretrainDataset
from scripts.layer_sensitivity_scan import sha

def main():
    run=Path(sys.argv[1]);folder=run.parent;target=run/'final_health';target.mkdir(exist_ok=False)
    path=run/'checkpoint.pt';before=sha(path);ck=torch.load(path,map_location='cpu',weights_only=True)
    cfg=json.loads((folder/'config.json').read_text());assert sha(folder/'dev.jsonl')==cfg['hashes']['dev.jsonl']
    model=MatchedMiniMind(MiniMindConfig()).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    ds=PretrainDataset(str(folder/'dev.jsonl'),AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True),max_length=340)
    scale=1.;mask=None;ratios=[]
    def hook(layer,args,delta):
        if scale==1:ratios.append((delta.float()[mask].norm(dim=-1)/args[0].float()[mask].norm(dim=-1).clamp_min(1e-12)).cpu())
        return delta*scale
    handle=model.model.layers[3].adapter.register_forward_hook(hook);result={}
    with torch.no_grad():
        for scale in [1.,0.]:
            nll=0.;count=0
            for i in range(0,len(ds),2):
                rows=[ds[i],ds[i+1]];x=torch.stack([r[0] for r in rows]).cuda();y=torch.stack([r[1] for r in rows]).cuda()
                mask=torch.zeros_like(y,dtype=torch.bool);mask[:,:-1]=y[:,1:]!=-100
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,labels=y)
                n=int(mask.sum());nll+=float(out.loss)*n;count+=n
            result[str(int(scale))]={'ce':nll/count,'ppl':math.exp(nll/count),'tokens':count}
    handle.remove();assert sha(path)==before
    assert all(torch.equal(v.cpu(),ck['model'][k]) for k,v in model.state_dict().items())
    history=json.loads((run/'history.json').read_text());assert abs(history[-1]['dev']['ce']-result['1']['ce'])<1e-6
    result.update(status='passed',optimizer_steps=0,source_sha256=before,source_and_memory_unchanged=True,
        mean_residual_hidden_ratio=float(torch.cat(ratios).mean()),disabled_minus_normal_ce=result['0']['ce']-result['1']['ce'])
    (target/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
