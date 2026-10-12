"""Rebuild fixed Stage26 initial states from mapped assets, never train."""
import argparse,json,sys,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer,PreTrainedTokenizerFast
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherResidual
from model.model_peft25 import install,adaptation_state
from portable.check_assets import check,digest
def main():
    ap=argparse.ArgumentParser()
    for name in ['manifest','mapping','output']:ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();audit=check(args.manifest,args.mapping)
    mapping=json.loads(args.mapping.read_text());base=args.mapping.resolve().parent
    def path(key):
        p=Path(mapping[key]);return p if p.is_absolute() else base/p
    # Code identities are verified against the files actually imported, not arbitrary mapped copies.
    for name,rel in [('code_model','model/model_minimind.py'),('code_fisher','model/model_fisher.py'),('code_peft','model/model_peft25.py'),('code_dataset','dataset/lm_dataset.py')]:
        assert digest(ROOT/rel)==audit['assets'][name]['sha256'],rel
    args.output.mkdir(parents=True,exist_ok=False);r={'status':'running','optimizer_updates':0,'external_evaluated':False,'asset_audit':audit,'seeds':{}}
    def save():(args.output/'result.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    save()
    try:
        assert torch.cuda.is_available(),'CUDA required for exact historical extraction'
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        state=torch.load(path('backbone'),map_location='cpu',weights_only=True)
        # Isolate the two verified tokenizer files: no unregistered sibling config is loaded.
        import tempfile,shutil
        with tempfile.TemporaryDirectory() as td:
            shutil.copyfile(path('tokenizer'),Path(td)/'tokenizer.json');shutil.copyfile(path('tokenizer_config'),Path(td)/'tokenizer_config.json')
            tok=AutoTokenizer.from_pretrained(td,local_files_only=True)
        train=PretrainDataset(str(path('train')),tok,max_length=340);assert len(train)==16384
        locations=torch.load(path('fixed_locations_container'),map_location='cpu',weights_only=True)['locations'];assert locations.shape==(30000,2)
        indices=json.loads(path('record_order').read_text())['kmeans']['record_indices'];assert len(indices)==len(set(indices))==512
        hidden=torch.empty(30000,768);seen=torch.zeros(30000,dtype=torch.bool)
        model=MiniMindForCausalLM(MiniMindConfig());model.load_state_dict(state,strict=True);model.cuda().eval()
        captured=[];hook=model.model.layers[3].register_forward_hook(lambda m,a,o:captured.append(o[0].detach().float().cpu()))
        with torch.no_grad():
            for offset in range(0,512,2):
                ids=indices[offset:offset+2];rows=[train[i] for i in ids];x,y=[torch.stack([row[k] for row in rows]).cuda() for k in [0,1]];captured.clear()
                with torch.autocast('cuda',dtype=torch.bfloat16):model(x)
                for j,idx in enumerate(ids):
                    rows=(locations[:,0]==idx).nonzero().flatten();pos=locations[rows,1].long()
                    assert ((pos>=0)&(pos<339)).all() and (y[j,pos.cuda()+1]!=-100).all()
                    hidden[rows]=captured[0][j,pos];seen[rows]=True
        hook.remove();assert seen.all() and torch.isfinite(hidden).all();del model,captured;torch.cuda.empty_cache()
        def compare(actual,expected):
            assert actual.keys()==expected.keys()
            for n,v in actual.items():assert torch.equal(v,expected[n]),n
        for seed in [1729,2718,31415]:
            folder=args.output/str(seed);folder.mkdir()
            order=torch.randperm(16384,generator=torch.Generator().manual_seed(seed)).tolist()
            assert order==json.loads(path(f'{seed}_order').read_text())
            (folder/'order.json').write_text(json.dumps(order),encoding='utf-8')
            torch.manual_seed(seed+1);branch=FisherResidual(768,1,64)
            with torch.no_grad():descriptors=branch.input(branch.norm(hidden));mu=descriptors.mean(0,keepdim=True);branch.b.copy_(-mu)
            common={n:v.clone() for n,v in branch.state_dict().items()};center={'locations':locations,'mu':mu,'descriptors':descriptors}
            products={'common_init':common,'center_fit':center}
            products['k1_init']={'model.layers.3.fisher.'+n:v.clone() for n,v in common.items()};products['current_init']={n:v.clone() for n,v in products['k1_init'].items()}
            m=MiniMindForCausalLM(MiniMindConfig());m.load_state_dict(state,strict=True);torch.manual_seed(seed+1);install(m,'lora');products['lora_init']=adaptation_state(m);del m
            evidence={}
            for name,product in products.items():
                expected=torch.load(path(f'{seed}_{name}'),map_location='cpu',weights_only=True);compare(product,expected)
                dest=folder/(name+'.pt');torch.save(product,dest)
                evidence[name]={'tensor_exact':True,'sha256':digest(dest),'reference_sha256':digest(path(f'{seed}_{name}')),'container_bytes_equal':digest(dest)==digest(path(f'{seed}_{name}'))}
            r['seeds'][str(seed)]={'order_exact':True,'products':evidence};save();print(seed,'all initial tensors exact',flush=True)
        check(args.manifest,args.mapping);r['status']='passed';save()
    except Exception:r.update(status='failed',error=traceback.format_exc());save();raise
if __name__=='__main__':main()
