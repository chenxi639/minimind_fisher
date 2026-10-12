"""Stage26B preparation only; smoke checkpoints must not seed formal training."""
import sys,json,math,hashlib,traceback
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherResidual,FisherMiniMind
from model.model_moment_controls import MomentControlMiniMind
from model.model_peft25 import install,enabled,adaptation_state
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    registry=ROOT/'runs/confirm26_registry.json';reg=json.loads(registry.read_text());files=dict(reg['file_hashes'])
    prep=ROOT/'runs/frozen_ready_20261002_124018/preparation.json'
    for p in [registry,prep,Path(__file__),ROOT/'report/26B_protocol.md']:files[str(p)]=sha(p)
    def verify():
        for p,h in files.items():assert sha(p)==h,p
    verify();run=ROOT/'runs'/('confirm26_ready_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir()
    result={'status':'running','formal_training':False,'external_evaluated':False,'test_evaluated':False,'seeds':{},'file_hashes':files}
    def save():(run/'preparation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save();print(run,flush=True)
    try:
        state=torch.load(ROOT/'checkpoints/official_edba70e/download.pth',map_location='cpu',weights_only=True)
        tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True);source=ROOT/'runs/pair_ready_20260929_181458'
        train=PretrainDataset(str(source/'train.jsonl'),tok,max_length=340);dev=PretrainDataset(str(source/'dev.jsonl'),tok,max_length=340)
        def batch(ds,ids):
            rows=[ds[int(i)] for i in ids];return [torch.stack([r[k] for r in rows]).cuda() for k in [0,1]]
        def fw(m,x,y=None):
            with torch.autocast('cuda',dtype=torch.bfloat16):return m(x,labels=y)
        fit=torch.load(ROOT/'runs/frozen_ready_20261002_124018/kmeans_fit.pt',map_location='cpu',weights_only=True)
        loc=fit['locations'];assert loc.shape==(30000,2)
        indices=json.loads(prep.read_text())['kmeans']['record_indices'];assert len(indices)==512
        hidden=torch.empty(30000,768);seen=torch.zeros(30000,dtype=torch.bool)
        pure=MiniMindForCausalLM(MiniMindConfig());pure.load_state_dict(state,strict=True);pure.cuda().eval()
        capture=[];hook=pure.model.layers[3].register_forward_hook(lambda m,a,o:capture.append(o[0].detach().float().cpu()))
        with torch.no_grad():
            for off in range(0,512,2):
                ids=indices[off:off+2];tx,ty=batch(train,ids);capture.clear();fw(pure,tx)
                for j,idx in enumerate(ids):
                    rows=(loc[:,0]==idx).nonzero().flatten();pos=loc[rows,1].long()
                    assert ((pos>=0)&(pos<339)).all() and (ty[j,pos.cuda()+1]!=-100).all()
                    hidden[rows]=capture[0][j,pos];seen[rows]=True
        hook.remove();assert seen.all() and torch.isfinite(hidden).all()
        x,y=batch(dev,[0,64])
        with torch.no_grad():ref=fw(pure,x).logits.cpu()
        del pure,capture;torch.cuda.empty_cache();print('Fixed training locations extracted',flush=True)
        orders=[];projections=[];centers=[]
        for seed in reg['seeds']:
            folder=run/str(seed);folder.mkdir();order=torch.randperm(len(train),generator=torch.Generator().manual_seed(seed)).tolist()
            assert sorted(order)==list(range(16384));orders.append(order)
            (folder/'order.json').write_text(json.dumps(order),encoding='utf-8')
            torch.manual_seed(seed+1);branch=FisherResidual(768,1,64)
            with torch.no_grad():descriptors=branch.input(branch.norm(hidden));mu=descriptors.mean(0,keepdim=True);branch.b.copy_(-mu)
            assert torch.isfinite(mu).all();common={n:v.clone() for n,v in branch.state_dict().items()}
            projections.append(common['input.weight']);centers.append(mu.clone())
            torch.save({'locations':loc,'mu':mu,'descriptors':descriptors},folder/'center_fit.pt');torch.save(common,folder/'common_init.pt')
            entry={'seed':seed,'branch_seed':seed+1,'arms':{},'center_norm':float(mu.norm()),'center_method':'mean of fixed training positions; not K-means'};result['seeds'][str(seed)]=entry
            token_ref=None
            for kind,expected in [('k1',148352),('current',148224),('lora',150528)]:
                cfg=MiniMindConfig(fisher_k=1,fisher_dim=64,fisher_index=3,moment_mode='current')
                if kind=='lora':
                    m=MiniMindForCausalLM(cfg);m.load_state_dict(state,strict=True);torch.manual_seed(seed+1);install(m,'lora');initial=adaptation_state(m)
                else:
                    m=(FisherMiniMind if kind=='k1' else MomentControlMiniMind)(cfg)
                    initial={'model.layers.3.fisher.'+n:v.clone() for n,v in common.items()}
                    m.load_state_dict({**state,**initial},strict=True)
                    m.requires_grad_(False);m.model.layers[3].fisher.requires_grad_(True)
                torch.save(initial,folder/(kind+'_init.pt'));m.cuda().eval();params=[p for p in m.parameters() if p.requires_grad]
                assert sum(p.numel() for p in params)==expected
                with torch.no_grad():assert torch.equal(ref,fw(m,x).logits.cpu())
                torch.manual_seed(seed);opt=torch.optim.AdamW(params,lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=.01);steps=[];m.train()
                for step in range(2):
                    opt.zero_grad(set_to_none=True);tokens=0
                    for micro in range(8):
                        off=(step*8+micro)*2;tx,ty=batch(train,order[off:off+2]);opt.param_groups[0]['lr']=.0005*(.1+.45*(1+math.cos(math.pi*(step*8+micro+1)/8192)))
                        loss=fw(m,tx,ty).loss/8;assert torch.isfinite(loss);loss.backward();tokens+=int((ty[:,1:]!=-100).sum())
                    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
                    gradients={n:float(p.grad.norm()) for n,p in m.named_parameters() if p.requires_grad}
                    if step==1:assert all(v>0 for v in gradients.values())
                    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
                    torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();steps.append({'step':step+1,'tokens':tokens,'gradients':gradients})
                counts=[r['tokens'] for r in steps]
                if token_ref is None:token_ref=counts
                assert counts==token_ref
                assert all(torch.equal(m.state_dict()[n].cpu(),v) for n,v in state.items())
                if kind=='current':
                    assert all(torch.equal(m.model.layers[3].fisher.state_dict()[n].cpu(),common[n]) for n in ['w','b'])
                smoke={n:m.state_dict()[n].detach().cpu().clone() for n in initial};torch.save(smoke,folder/(kind+'_smoke.pt'));m.eval()
                with torch.no_grad():before=fw(m,x).logits.cpu()
                m.load_state_dict({**state,**torch.load(folder/(kind+'_smoke.pt'),weights_only=True)},strict=True)
                with torch.no_grad():assert torch.equal(before,fw(m,x).logits.cpu())
                if kind=='lora':enabled(m,False)
                else:hook=m.model.layers[3].fisher.register_forward_hook(lambda mod,a,o:(torch.zeros_like(o[0]),o[1]))
                with torch.no_grad():assert torch.equal(ref,fw(m,x).logits.cpu())
                if kind!='lora':hook.remove()
                entry['arms'][kind]={'status':'passed','trainable_parameters':expected,'initial_logits_exact':True,'backbone_unchanged':True,'reload_exact':True,'disabled_returns_pure':True,'steps':steps}
                print(seed,kind,'passed',counts,flush=True);save();del m,opt,params;torch.cuda.empty_cache()
            for p in folder.iterdir():files[str(p)]=sha(p)
            save()
        assert all(orders[i]!=orders[j] and not torch.equal(projections[i],projections[j]) and not torch.equal(centers[i],centers[j]) for i in range(3) for j in range(i))
        verify();result.update(status='passed',source_files_unchanged=True,distinct_seeds_verified=True,state_format='full adaptation state keys; common_init branch-only; formal training uses *_init.pt, never *_smoke.pt');save()
    except Exception:
        result.update(status='failed',error=traceback.format_exc());save();raise
if __name__=='__main__':main()
