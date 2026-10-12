"""Stage16: verified external checkpoint, train-only centers, two-update smoke only."""
import sys, json, hashlib, math, time, argparse
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
import numpy as np
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_fisher import FisherMiniMind, FisherResidual
from model.model_matched_mlp import MatchedMiniMind, MatchedResidual

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);args=parser.parse_args();seed=args.seed
    assert seed in [2026,3407]
    torch.set_num_threads(4);torch.manual_seed(seed)
    source=ROOT/'runs/pair_ready_20260929_181458';official=ROOT/'checkpoints/official_edba70e'
    run=ROOT/'runs'/('frozen_ready_seed'+str(seed)+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir()
    result={'status':'running','formal_training':False};start=time.perf_counter()
    def save(): (run/'preparation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(run,flush=True)
    try:
        weight=official/'download.pth'
        assert weight.stat().st_size==137684407
        assert sha(weight)=='0191958e7a96ef7cd1e443d379dc2ebea90cfea72cc4450fbcb4a7b87ff945f6'
        assert json.loads((ROOT/'model/tokenizer.json').read_text(encoding='utf-8'))==json.loads((official/'tokenizer.json').read_text(encoding='utf-8'))
        localcfg=json.loads((ROOT/'model/tokenizer_config.json').read_text(encoding='utf-8'))
        remotecfg=json.loads((official/'tokenizer_config.json').read_text(encoding='utf-8'))
        differences=[k for k in localcfg.keys()|remotecfg.keys() if localcfg.get(k)!=remotecfg.get(k)]
        assert set(differences)<= {'chat_template','extra_special_tokens'}
        published=json.loads((official/'config.json').read_text(encoding='utf-8'));config=MiniMindConfig()
        fields=['hidden_size','num_hidden_layers','vocab_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','rope_theta','rms_norm_eps','tie_word_embeddings','max_position_embeddings']
        assert all(getattr(config,k)==published[k] for k in fields)
        cfg=json.loads((source/'config.json').read_text());cfg['hashes']={n:cfg['hashes'][n] for n in ['train.jsonl','dev.jsonl','order.json']}
        for n,d in cfg['hashes'].items():assert sha(source/n)==d
        state=torch.load(weight,map_location='cpu',weights_only=True)
        assert torch.equal(state['lm_head.weight'],state['model.embed_tokens.weight'])
        pure=MiniMindForCausalLM(config);pure.load_state_dict(state,strict=True);pure.cuda().eval()
        tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
        train=PretrainDataset(str(source/'train.jsonl'),tok,max_length=340)
        dev=PretrainDataset(str(source/'dev.jsonl'),tok,max_length=340)
        order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed)).tolist();assert sorted(order)==list(range(len(train)))
        (run/'order.json').write_text(json.dumps(order),encoding='utf-8')
        cfg['hashes'].pop('order.json');cfg['order_path']=str(run/'order.json');cfg['order_sha256']=sha(run/'order.json')
        def batch(ds,ids):
            rows=[ds[j] for j in ids];return tuple(torch.stack([r[i] for r in rows]).cuda() for i in (0,1))
        x,y=batch(dev,[0,64])
        def forward(m,x,y=None):
            with torch.autocast('cuda',dtype=torch.bfloat16):return m(x,labels=y)
        with torch.no_grad():
            ref=forward(pure,x,y);reference=ref.logits.cpu();ce=float(ref.loss);assert math.isfinite(ce)
            pure.load_state_dict(state,strict=True)
            assert torch.equal(reference,forward(pure,x).logits.cpu())
        del pure,ref
        result.update(weight_sha256=sha(weight),strict_load=True,tied_weights_equal=True,tokenizer_semantic_equal=True,tokenizer_config_differences=differences,config_fields_checked=fields,small_dev_ce=ce,small_dev_ppl=math.exp(ce),small_dev_indices=[0,64],reload_exact=True)
        save()
        for arm,cls,branchcls,name in [('fisher',FisherMiniMind,FisherResidual,'fisher'),('mlp',MatchedMiniMind,MatchedResidual,'adapter')]:
            torch.manual_seed(seed+1);branch=branchcls(768)
            init={f'model.layers.3.{name}.'+k:v for k,v in branch.state_dict().items()}
            model=cls(config);model.load_state_dict({**state,**init},strict=True);model.cuda().eval()
            branch=getattr(model.model.layers[3],name)
            if arm=='fisher':
                chunks=[];locations=[];active=None
                def hook(layer,args,output):
                    with torch.autocast('cuda',enabled=False):desc=layer.input(layer.norm(args[0].float()))
                    chunks.append(desc[active].cpu())
                handle=branch.register_forward_hook(hook)
                indices=np.random.default_rng(42).choice(len(train),512,replace=False).tolist()
                with torch.no_grad():
                    for i in range(0,512,2):
                        ids=indices[i:i+2];tx,ty=batch(train,ids)
                        active=torch.zeros_like(ty,dtype=torch.bool);active[:,:-1]=ty[:,1:]!=-100
                        locations.extend([[ids[j],t] for j,t in active.nonzero().cpu().tolist()])
                        forward(model,tx)
                handle.remove();pool=torch.cat(chunks).numpy()
                chosen=np.random.default_rng(43).choice(len(pool),30000,replace=False);desc=pool[chosen]
                assert np.isfinite(desc).all()
                with threadpool_limits(limits=4):km=KMeans(n_clusters=8,n_init=10,max_iter=300,random_state=42,algorithm='lloyd').fit(desc)
                centers=torch.from_numpy(km.cluster_centers_);counts=np.bincount(km.labels_,minlength=8);assert (counts>0).all()
                with torch.no_grad():branch.b.copy_(-centers.to('cuda'))
                q=(-.5*torch.cdist(torch.from_numpy(desc),centers).square()).softmax(-1)
                torch.save({'descriptors':torch.from_numpy(desc),'centers':centers,'locations':torch.tensor([locations[j] for j in chosen])},run/'kmeans_fit.pt')
                result['kmeans']={'record_indices':indices,'pool_tokens':len(pool),'fit_tokens':30000,'counts':counts.tolist(),'soft_occupancy':q.mean(0).tolist(),'entropy':float(-(q*q.clamp_min(1e-30).log()).sum(-1).mean()),'sha256':sha(run/'kmeans_fit.pt')}
                print('new pretrained K-means completed',flush=True)
            with torch.no_grad():assert torch.equal(reference,forward(model,x).logits.cpu())
            for p in model.parameters():p.requires_grad_(False)
            for p in branch.parameters():p.requires_grad_(True)
            params=list(branch.parameters());assert all(p.requires_grad for p in params)
            initial={k:v.detach().cpu().clone() for k,v in branch.state_dict().items()}
            torch.save(initial,run/(arm+'_init.pt'))
            optimizer=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
            history=[];model.train()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True);tokens=0
                for micro in range(8):
                    off=(step*8+micro)*2;tx,ty=batch(train,order[off:off+2])
                    lr=cfg['lr']*(.1+.45*(1+math.cos(math.pi*(step*8+micro+1)/(1024*8))))
                    optimizer.param_groups[0]['lr']=lr
                    loss=forward(model,tx,ty).loss/8;assert torch.isfinite(loss);loss.backward();tokens+=int((ty[:,1:]!=-100).sum())
                grads={k:float(p.grad.norm()) for k,p in branch.named_parameters()}
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                assert grads['output.weight']>0
                if step==1:assert all(v>0 for v in grads.values())
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                history.append({'step':step+1,'tokens':tokens,'grad_norms_before_clip':grads})
            assert all(torch.equal(v.cpu(),state[k].to(v.dtype)) for k,v in model.state_dict().items() if k in state)
            model.eval();saved={k:v.cpu().clone() for k,v in branch.state_dict().items()};torch.save(saved,run/(arm+'_smoke.pt'))
            with torch.no_grad():before=forward(model,x).logits.cpu()
            branch.load_state_dict(torch.load(run/(arm+'_smoke.pt'),weights_only=True),strict=True)
            with torch.no_grad():
                assert torch.equal(before,forward(model,x).logits.cpu())
                branch.output.weight.zero_();assert torch.equal(reference,forward(model,x).logits.cpu())
            branch.load_state_dict(saved,strict=True)
            result[arm]={'initial_logits_exact':True,'trainable_parameters':sum(p.numel() for p in params),'history':history,'backbone_exact_unchanged':True,'reload_exact':True,'disabled_returns_pure_exact':True,'init_sha256':sha(run/(arm+'_init.pt'))}
            del model,branch,optimizer,params;torch.cuda.empty_cache();save();print(arm+' smoke passed',flush=True)
        cfg.update(seed=seed,source_preparation=str(source),backbone_path=str(weight),backbone_sha256=sha(weight),frozen_backbone=True,branch_seed=seed+1,scope='potentially_seen_corpus_mechanism_screen',formal_training=False)
        cfg['code_hashes']={n:sha(ROOT/n) for n in ['scripts/prepare_frozen_multiseed.py','scripts/train_frozen_multiseed.py','model/model_minimind.py','model/model_fisher.py','model/model_matched_mlp.py']}
        (run/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        result.update(status='passed',seconds=time.perf_counter()-start)
    except Exception as exc:result.update(status='failed',error=repr(exc));raise
    finally:save();print(json.dumps({k:v for k,v in result.items() if k!='kmeans'},indent=2),flush=True)
if __name__=='__main__':main()
