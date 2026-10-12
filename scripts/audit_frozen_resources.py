"""Read-only Stage18 audit: no optimizer, no backward, no test evaluation."""
import sys,json,hashlib,math,time,unicodedata,re
from pathlib import Path
from datetime import datetime
from contextlib import nullcontext
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind
from model.model_matched_mlp import MatchedMiniMind

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pair=ROOT/'runs/frozen_pair_20261002_145450';cfg=json.loads((pair/'manifest.json').read_text())
    prep=ROOT/'runs/frozen_ready_20261002_124018';source=Path(cfg['source_preparation'])
    run=ROOT/'runs'/('resource_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S'));run.mkdir()
    r={'status':'running','optimizer_updates':0,'test_evaluated':False};start=time.perf_counter()
    def save(): (run/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    print(run,flush=True)
    protected=dict(cfg['file_hashes'])
    for p in [pair/'manifest.json',pair/'fisher/checkpoint.pt',pair/'mlp/checkpoint.pt',pair/'fisher/assignment.json',prep/'fisher_init.pt',ROOT/'scripts/audit_frozen_resources.py']:
        protected[str(p)]=sha(p)
    def verify():
        for p,d in protected.items():assert sha(Path(p))==d,p
    try:
        verify();r['protected_hashes']=protected
        official=ROOT/'checkpoints/official_edba70e';published=json.loads((official/'config.json').read_text())
        mc=MiniMindConfig();keys=['hidden_size','num_hidden_layers','vocab_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','rope_theta','rms_norm_eps','tie_word_embeddings','max_position_embeddings']
        assert all(getattr(mc,k)==published[k] for k in keys)
        tokenizer_equal=json.loads((ROOT/'model/tokenizer.json').read_text(encoding='utf-8'))==json.loads((official/'tokenizer.json').read_text(encoding='utf-8'))
        assert tokenizer_equal
        r['provenance']={'model_repo':'jingyaogong/minimind-3-pytorch','model_revision':'edba70ec15e06bc4280fbb96ac3383d73a7eab91','weight_sha256':sha(Path(cfg['backbone_path'])),'auxiliary_config_repo':'jingyaogong/minimind-3','auxiliary_revision':'f92512d4cd6142fa9acc0d6022375049a8974bf6','checked_config':{k:getattr(mc,k) for k in keys},'tokenizer_json_semantic_equal':tokenizer_equal,'auxiliary_hashes':{n:sha(official/n) for n in ['config.json','tokenizer.json','tokenizer_config.json']}}
        tok=AutoTokenizer.from_pretrained(str(ROOT/'model'),local_files_only=True)
        sets={};counts={}
        for split in ['train','dev']:
            rows=[json.loads(s)['text'] for s in (source/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
            enc=tok(rows,add_special_tokens=False,truncation=True,max_length=338).input_ids
            modes={'stripped':[t.strip() for t in rows],'normalized':[re.sub(r'\s+','',unicodedata.normalize('NFKC',t)).casefold() for t in rows],'truncated_tokens':[tuple(ids) for ids in enc]}
            sets[split]={k:set(v) for k,v in modes.items()};counts[split]={'records':len(rows),'within_duplicate_excess':{k:len(v)-len(set(v)) for k,v in modes.items()}}
        manifest=json.loads((source/'manifest.json').read_text())
        r['data']={'splits':counts,'train_dev_intersections':{k:len(sets['train'][k]&sets['dev'][k]) for k in sets['train']},'source_url':manifest['url'],'source_membership':'Both splits sampled from official pretrain_t2t_mini; actual official checkpoint training exposure unknown','near_semantic_duplicates':'not exhaustively audited','sealed_test':'not read or evaluated'}
        trace=json.loads((pair/'fisher/assignment.json').read_text());assert len(trace)==4
        r['assignment_trajectory']=[{'step':step,**row} for step,row in zip([0,256,512,1024],trace)]
        assert all(row['sample_indices']==list(range(0,1024,64)) for row in trace)
        r['checkpoint_inventory']=[str(p.relative_to(pair)) for p in pair.rglob('*.pt')]
        r['missing_evidence']='No 256/512 checkpoint; logs prove concentration by step256, exact onset and intermediate parameter states unavailable'
        save();print('provenance, data, trajectory audit saved',flush=True)
        state=torch.load(cfg['backbone_path'],weights_only=True,map_location='cpu')
        ds=PretrainDataset(str(source/'dev.jsonl'),tok,max_length=340)
        batches=[]
        for i in range(0,len(ds),2):
            rows=[ds[i],ds[i+1]];batches.append(tuple(torch.stack([v[k] for v in rows]) for k in [0,1]))
        def evaluate(m,precision):
            total=0.;count=0;batch_losses=[]
            with torch.no_grad():
                for cx,cy in batches:
                    x,y=cx.cuda(),cy.cuda()
                    with (torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()):out=m(x,labels=y)
                    assert torch.isfinite(out.loss);n=int((y[:,1:]!=-100).sum());v=float(out.loss)
                    total+=v*n;count+=n;batch_losses.append(v)
            return {'ce':total/count,'ppl':math.exp(total/count),'tokens':count,'batch_mean_losses':batch_losses}
        r['precision']={};r['ablations']={}
        for arm,cls,name in [('pure',MiniMindForCausalLM,None),('fisher',FisherMiniMind,'fisher'),('mlp',MatchedMiniMind,'adapter')]:
            m=cls(mc);weights=dict(state)
            if name:
                ck=torch.load(pair/arm/'checkpoint.pt',map_location='cpu',weights_only=True)
                weights.update({f'model.layers.3.{name}.'+k:v for k,v in ck['branch'].items()})
            m.load_state_dict(weights,strict=True);m.cuda().eval();m.requires_grad_(False)
            r['precision'][arm]={}
            for precision in ['bf16','fp32']:
                repeats=[]
                for repeat in range(2):
                    row=evaluate(m,precision);repeats.append(row);print(arm,precision,repeat,row['ce'],flush=True)
                r['precision'][arm][precision]={'repeats':repeats,'repeat_ce_spread':abs(repeats[0]['ce']-repeats[1]['ce']),'batch_losses_exact_repeat':repeats[0]['batch_mean_losses']==repeats[1]['batch_mean_losses']};save()
            if arm=='fisher':
                projection=m.model.layers[3].fisher.output
                # Actual implementation uses F.linear(weight), so hook the branch and recompute
                # only output projection from the exact feature formula captured via input h.
                branch=m.model.layers[3].fisher
                for label,keep in [(f'drop_{k}',[j for j in range(8) if j!=k]) for k in range(8)]+[('keep_1',[1]),('all',list(range(8))),('none',[])]:
                    mask=torch.zeros(2,8,64,device='cuda');mask[:,keep,:]=1;mask=mask.flatten()
                    def hook(layer,args,output):
                        h=args[0].float();d=layer.input(layer.norm(h));z=layer.w[None,None]*(d[:,:,None]+layer.b[None,None]);q=(-.5*z.square().sum(-1)).softmax(-1)[...,None]
                        a=(q*z).cumsum(1);b=(q*(z.square()-1)/math.sqrt(2)).cumsum(1)
                        raw=torch.cat((a.flatten(2),b.flatten(2)),-1)/torch.arange(1,h.shape[1]+1,device=h.device)[None,:,None]
                        encoded=torch.nn.functional.normalize(raw/torch.sqrt(raw.abs()+1e-6),dim=-1,eps=1e-6)
                        delta=torch.nn.functional.linear(encoded*mask,layer.output.weight)
                        return delta.to(args[0].dtype),output[1]
                    handle=branch.register_forward_hook(hook);row=evaluate(m,'fp32');handle.remove()
                    r['ablations'][label]=row;save();print(label,row['ce'],flush=True)
                assert r['ablations']['all']['ce']==r['precision']['fisher']['fp32']['repeats'][0]['ce']
                assert r['ablations']['none']['ce']==r['precision']['pure']['fp32']['repeats'][0]['ce']
            assert all(torch.equal(v.cpu(),weights[k].to(v.dtype)) for k,v in m.state_dict().items())
            del m,weights;torch.cuda.empty_cache()
        verify();r.update(status='passed',source_files_unchanged=True,seconds=time.perf_counter()-start)
    except Exception as exc:r.update(status='failed',error=repr(exc));raise
    finally:save();print(run,r['status'],flush=True)
if __name__=='__main__':main()
