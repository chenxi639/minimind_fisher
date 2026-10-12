"""Two-step regression of the frozen training kernel, not new research training."""
import argparse,json,math,sys,tempfile,shutil,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind
from model.model_moment_controls import MomentControlMiniMind
from model.model_peft25 import install
from portable.check_assets import check,digest
def micro_lr(cfg,step,micro):
    return cfg['learning_rate']*(.1+.45*(1+math.cos(math.pi*(step*cfg['accumulation_steps']+micro+1)/(cfg['horizon_steps']*cfg['accumulation_steps']))))
def main():
    ap=argparse.ArgumentParser()
    for name in ['config','manifest','mapping','output']:ap.add_argument('--'+name,type=Path,required=True)
    a=ap.parse_args();cfg=json.loads(a.config.read_text());audit=check(a.manifest,a.mapping)
    # Only this validated protocol is enabled; new experiments require a new protocol ID.
    expected=json.loads((ROOT/'portable/train_protocol.json').read_text());assert cfg==expected and cfg['protocol_id']=='stage26-frozen-v1'
    mp=json.loads(a.mapping.read_text())
    def path(n):
        p=Path(mp[n]);return p if p.is_absolute() else a.mapping.resolve().parent/p
    for n,rel in [('code_model','model/model_minimind.py'),('code_fisher','model/model_fisher.py'),('code_peft','model/model_peft25.py'),('code_dataset','dataset/lm_dataset.py'),('code_current','model/model_moment_controls.py'),('code_moment','model/model_moment.py')]:assert digest(ROOT/rel)==audit['assets'][n]['sha256']
    a.output.mkdir(parents=True,exist_ok=False);result={'status':'running','formal_training':False,'external_evaluated':False,'config':cfg,'asset_audit':audit,'runs':{}}
    def save():(a.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    save()
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        with tempfile.TemporaryDirectory() as td:
            for n in ['tokenizer','tokenizer_config']:shutil.copyfile(path(n),Path(td)/(n+'.json'))
            tok=AutoTokenizer.from_pretrained(td,local_files_only=True)
        train=PretrainDataset(str(path('train')),tok,max_length=cfg['max_length']);state=torch.load(path('backbone'),weights_only=True,map_location='cpu')
        for seed in cfg['seeds']:
            order=json.loads(path(f'{seed}_order').read_text());assert sorted(order)==list(range(16384));assert order==torch.randperm(16384,generator=torch.Generator().manual_seed(seed)).tolist()
            for kind in cfg['arms']:
                mc=MiniMindConfig(fisher_k=1,fisher_dim=64,fisher_index=3,moment_mode='current')
                if kind=='lora':
                    m=MiniMindForCausalLM(mc);m.load_state_dict(state,strict=True);torch.manual_seed(seed+1);install(m,'lora')
                else:
                    m=(FisherMiniMind if kind=='k1' else MomentControlMiniMind)(mc);m.requires_grad_(False);m.model.layers[3].fisher.requires_grad_(True)
                initial=torch.load(path(f'{seed}_{kind}_init'),weights_only=True,map_location='cpu');m.load_state_dict({**state,**initial},strict=True);m.cuda();torch.manual_seed(seed)
                params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=cfg['learning_rate'],betas=tuple(cfg['betas']),eps=cfg['eps'],weight_decay=cfg['weight_decay']);history=[];seen=0
                original=json.loads(path(f'{seed}_{kind}_history').read_text())
                for step in range(2):
                    m.train();opt.zero_grad(set_to_none=True);tokens=0;total=0.
                    for micro in range(cfg['accumulation_steps']):
                        off=(step*cfg['accumulation_steps']+micro)*cfg['batch_size'];rows=[train[i] for i in order[off:off+cfg['batch_size']]];x,y=[torch.stack([row[k] for row in rows]).cuda() for k in [0,1]]
                        lr=micro_lr(cfg,step,micro);opt.param_groups[0]['lr']=lr
                        with torch.autocast('cuda',dtype=torch.bfloat16):loss=m(x,labels=y).loss/cfg['accumulation_steps']
                        assert torch.isfinite(loss);loss.backward();n=int((y[:,1:]!=-100).sum());tokens+=n;total+=float(loss.detach()*cfg['accumulation_steps'])*n
                    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
                    norm=torch.nn.utils.clip_grad_norm_(params,cfg['clip_norm'],error_if_nonfinite=True);opt.step();seen+=tokens
                    assert seen==original[step]['tokens'] and lr==original[step]['lr'];history.append({'step':step+1,'tokens':seen,'lr':lr,'train_ce':total/tokens,'original_train_ce':original[step]['train_ce'],'grad_norm':float(norm),'original_grad_norm':original[step]['grad_norm']})
                assert all(torch.equal(m.state_dict()[n].cpu(),v) for n,v in state.items())
                actual={n:m.state_dict()[n].detach().cpu().clone() for n in initial};ref=torch.load(path(f'{seed}_{kind}_smoke'),weights_only=True,map_location='cpu')
                assert actual.keys()==ref.keys()
                out=a.output/f'{seed}_{kind}.pt';torch.save(actual,out)
                differences={n:{'max_abs':float((v-ref[n]).abs().max()),'unequal':int((v!=ref[n]).sum())} for n,v in actual.items() if not torch.equal(v,ref[n])}
                if differences:
                    result['runs'][f'{seed}/{kind}']={'status':'failed','history':history,'differences':differences,'diagnostic_checkpoint_sha256':digest(out)};save()
                    raise AssertionError(f'{seed}/{kind}: non-exact smoke tensors; see differences')
                result['runs'][f'{seed}/{kind}']={'status':'passed','smoke_tensors_exact':True,'backbone_unchanged':True,'history':history,'sha256':digest(out)};save();print(seed,kind,'exact',flush=True);del m,opt,params;torch.cuda.empty_cache()
        check(a.manifest,a.mapping);result['status']='passed';save()
    except Exception:result.update(status='failed',error=traceback.format_exc());save();raise
if __name__=='__main__':main()
