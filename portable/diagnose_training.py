"""Bounded controlled diagnostic; does not change either historical implementation."""
import argparse,json,sys,tempfile,shutil,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from transformers import AutoTokenizer
from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig
from model.model_fisher import FisherMiniMind
from portable.check_assets import check
def main():
    ap=argparse.ArgumentParser()
    for n in ['manifest','mapping','output']:ap.add_argument('--'+n,type=Path,required=True)
    ap.add_argument('--repeat-backends',action='store_true')
    ap.add_argument('--math-paths',action='store_true')
    a=ap.parse_args();audit=check(a.manifest,a.mapping);mp=json.loads(a.mapping.read_text())
    def p(n):
        q=Path(mp[n]);return q if q.is_absolute() else a.mapping.resolve().parent/q
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    original_tok=AutoTokenizer.from_pretrained(str(p('tokenizer').parent),local_files_only=True)
    with tempfile.TemporaryDirectory() as td:
        for n in ['tokenizer','tokenizer_config']:shutil.copyfile(p(n),Path(td)/(n+'.json'))
        isolated_tok=AutoTokenizer.from_pretrained(td,local_files_only=True)
    order=json.loads(p('1729_order').read_text());data={}
    for name,tok in [('original',original_tok),('isolated',isolated_tok)]:
        ds=PretrainDataset(str(p('train')),tok,max_length=340);data[name]=[tuple(v.cuda() for v in ds[i]) for i in order[:32]]
    assert all(torch.equal(a,b) for old,new in zip(data['original'],data['isolated']) for a,b in zip(old,new))
    state=torch.load(p('backbone'),map_location='cpu',weights_only=True);init=torch.load(p('1729_k1_init'),map_location='cpu',weights_only=True);ref=torch.load(p('1729_k1_smoke'),map_location='cpu',weights_only=True)
    result={'status':'running','token_batches_exact':True,'cases':{},'optimizer_updates':0}
    def save():(a.output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    saved={}
    cases=[('new',False,False,'isolated'),('outside_div',True,False,'isolated'),('probe_only',False,True,'isolated'),('original_tokenizer',False,False,'original'),('historical_combined',True,True,'original')]
    if a.repeat_backends:cases=[('new',False,False,'isolated'),('repeat',False,False,'isolated'),('math1',False,False,'isolated'),('math2',False,False,'isolated')]
    if a.math_paths:cases=[('new',False,False,'isolated'),('historical_combined',True,True,'original')]
    for name,divide_out,probe,tokenizer in cases:
        if name.startswith('math') or a.math_paths:
            torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
        m=FisherMiniMind(MiniMindConfig(fisher_k=1,fisher_dim=64,fisher_index=3,moment_mode='current'));m.requires_grad_(False);m.model.layers[3].fisher.requires_grad_(True);m.load_state_dict({**state,**init},strict=True);m.cuda().eval()
        if probe:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):m(torch.stack([data[tokenizer][0][0],data[tokenizer][1][0]]))
        torch.manual_seed(1729);params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=.01);history=[]
        for step in range(2):
            m.train();opt.zero_grad(set_to_none=True);losses=[]
            for micro in range(8):
                off=(step*8+micro)*2;x,y=[torch.stack([data[tokenizer][j][k] for j in [off,off+1]]) for k in [0,1]]
                opt.param_groups[0]['lr']=.0005*(.1+.45*(1+math.cos(math.pi*(step*8+micro+1)/8192)))
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss=m(x,labels=y).loss
                    if not divide_out:loss=loss/8
                if divide_out:loss=loss/8
                losses.append(float(loss.detach()));loss.backward()
            gradients={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.requires_grad}
            norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step();result['optimizer_updates']+=1
            history.append({'step':step+1,'scaled_micro_losses':losses,'grad_norm':float(norm)})
            if step==0:torch.save({'gradients':gradients,'optimizer':opt.state_dict()},a.output/(name+'_first_step.pt'))
        actual={n:m.state_dict()[n].detach().cpu().clone() for n in init};saved[name]=actual
        result['cases'][name]={'history':history,'historical_smoke_exact':all(torch.equal(v,ref[n]) for n,v in actual.items()),'max_abs_vs_smoke':{n:float((v-ref[n]).abs().max()) for n,v in actual.items()},'exact_vs_new':all(torch.equal(v,saved['new'][n]) for n,v in actual.items())}
        torch.save(actual,a.output/(name+'_final.pt'));save();print(name,result['cases'][name]['historical_smoke_exact'],history[0]['grad_norm'],flush=True)
        del m,opt,params;torch.cuda.empty_cache()
    if a.repeat_backends:
        result['repeat_exact']={label:all(torch.equal(saved[left][n],saved[right][n]) for n in init) for label,left,right in [('default','new','repeat'),('math','math1','math2')]}
    result['status']='completed';save()
if __name__=='__main__':main()
