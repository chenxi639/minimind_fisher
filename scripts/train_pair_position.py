"""One common runner for baseline/Fisher. Test split is never evaluated here."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import datasets
import argparse,json,hashlib,math,time
from datetime import datetime
import torch
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind
from dataset.lm_dataset import PretrainDataset

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',type=Path,required=True);p.add_argument('--arm',choices=['baseline','fisher'],required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    folder=a.prepared.resolve();cfg=json.loads((folder/'config.json').read_text())
    for name,digest in cfg['hashes'].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest,name
    for name,digest in cfg['code_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    run=folder/(a.arm+('_smoke_' if a.smoke else '_train_')+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir(exist_ok=False)
    def save(n,v):(run/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    torch.manual_seed(cfg['seed']);torch.backends.cudnn.benchmark=False
    model=(MiniMindForCausalLM if a.arm=='baseline' else FisherMiniMind)(MiniMindConfig(fisher_index=cfg['index'],fisher_k=cfg['k'],fisher_dim=cfg['dim']))
    weights=torch.load(folder/'backbone_init.pt',map_location='cpu',weights_only=True)
    if a.arm=='fisher':weights.update(torch.load(folder/'fisher_init.pt',map_location='cpu',weights_only=True))
    model.load_state_dict(weights,strict=True);del weights
    model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
    train=PretrainDataset(str(folder/'train.jsonl'),tok,max_length=cfg['length'])
    dev=PretrainDataset(str(folder/'dev.jsonl'),tok,max_length=cfg['length'])
    order=json.loads((folder/'order.json').read_text());assert sorted(order)==list(range(len(train)))
    steps=2 if a.smoke else cfg['steps'];history=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def evaluate():
        model.eval();total=0.;count=0
        with torch.no_grad():
            for i in range(0,4 if a.smoke else len(dev),2):
                batch=[dev[j] for j in range(i,i+2)];x=torch.stack([v[0] for v in batch]).cuda();y=torch.stack([v[1] for v in batch]).cuda()
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,labels=y)
                n=int((y[:,1:]!=-100).sum());total+=out.loss.item()*n;count+=n
        model.train();return {'ce':total/count,'ppl':math.exp(total/count),'tokens':count}
    result={'status':'running','arm':a.arm,'smoke':a.smoke,'steps':0};save('result.json',result)
    try:
        history.append({'step':0,'dev':evaluate()});save('history.json',history)
        seen_tokens=0
        for step in range(steps):
            optimizer.zero_grad(set_to_none=True);weighted=0.;count=0
            for micro in range(cfg['accumulation']):
                offset=(step*cfg['accumulation']+micro)*cfg['batch']
                batch=[train[j] for j in order[offset:offset+cfg['batch']]]
                x=torch.stack([v[0] for v in batch]).cuda();y=torch.stack([v[1] for v in batch]).cuda()
                # Same per-microbatch cosine schedule and mean-loss accumulation as upstream.
                t=step*cfg['accumulation']+micro+1
                lr=cfg['lr']*(.1+.45*(1+math.cos(math.pi*t/(cfg['steps']*cfg['accumulation']))))
                for group in optimizer.param_groups:group['lr']=lr
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,labels=y);loss=out.loss/cfg['accumulation']
                assert torch.isfinite(loss);loss.backward()
                n=int((y[:,1:]!=-100).sum());weighted+=out.loss.item()*n;count+=n
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if a.arm=='fisher' and step==1:
                f=model.model.layers[cfg['index']].fisher
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in f.parameters())
                assert f.w.grad.abs().sum()>0 and f.input.weight.grad.abs().sum()>0
            optimizer.step();seen_tokens+=count
            row={'step':step+1,'train_ce':weighted/count,'tokens':seen_tokens,'grad_norm':float(norm),'lr':lr}
            if step+1 in cfg['eval_steps'] or step+1==steps:row['dev']=evaluate()
            history.append(row)
            if (step+1)%32==0 or step+1==steps:print(a.arm,row,flush=True);save('history.json',history)
        torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict(),'steps':steps,'config':cfg},run/'checkpoint.pt')
        reloaded=torch.load(run/'checkpoint.pt',map_location='cpu',weights_only=True)
        assert all(torch.equal(v.cpu(),reloaded['model'][k]) for k,v in model.state_dict().items())
        result.update(status='passed',steps=steps,tokens=seen_tokens,seconds=time.perf_counter()-start,peak_mib=torch.cuda.max_memory_allocated()/2**20,checkpoint_exact=True)
    except Exception as exc:result.update(status='failed',error=repr(exc));raise
    finally:save('history.json',history);save('result.json',result);print(run,result,flush=True)
if __name__=='__main__':main()
