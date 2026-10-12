"""CUDA specialization of the synthetic cross-process resume harness."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from portable import validate_resume as base
base.CONFIG['flash_attn']=True
base.RUNTIME.update(device='cuda:0',precision='bf16',attention_backend='math_only',cublas_workspace=os.environ['CUBLAS_WORKSPACE_CONFIG'])
original_build=base.build
def setup():
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_cudnn_sdp(False);torch.backends.cuda.enable_math_sdp(True)
def build(arm):
    m,_=original_build(arm);m.cuda()
    return m,torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0005,betas=(.9,.999),eps=1e-8,weight_decay=.01)
def snapshot(m,opt,arm,step,data,losses):
    return {'schema_version':1,'mode':'synthetic_cuda_resume_only','arm':arm,'config':base.CONFIG,'runtime':base.RUNTIME,'model':base.cpu_tree(m.state_dict()),'optimizer':base.cpu_tree(opt.state_dict()),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),'step':step,'data':data,'losses':losses,'parameter_names':[n for n,p in m.named_parameters() if p.requires_grad]}
def restore(ck):
    if ck['schema_version']!=1 or ck['mode']!='synthetic_cuda_resume_only' or ck['config']!=base.CONFIG or ck['runtime']!=base.RUNTIME or ck['step'] not in [1,2]:raise ValueError('incompatible CUDA checkpoint')
    if len(ck['cuda_rng'])!=torch.cuda.device_count():raise ValueError('CUDA device count changed')
    m,opt=build(ck['arm']);m.load_state_dict(ck['model'],strict=True)
    if ck['parameter_names']!=[n for n,p in m.named_parameters() if p.requires_grad]:raise ValueError('parameter ordering mismatch')
    opt.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);torch.cuda.set_rng_state_all(ck['cuda_rng']);return m,opt
def advance(m,opt,data,step):
    m.train();opt.zero_grad(set_to_none=True);values=[]
    for micro in range(2):
        x=data[(step*2+micro)*2:(step*2+micro+1)*2].cuda();opt.param_groups[0]['lr']=.0005*(.1+.45*(1+math.cos(math.pi*(step*2+micro+1)/4)))
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=m(x,labels=x).loss/2
        assert torch.isfinite(loss);loss.backward();values.append(float(loss.detach()))
    assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
    torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step();return values
def evaluate(m,data):
    m.eval();x=data[:4].cuda();y=x.clone();y[0,-3:]=-100;y[2,-1:]=-100
    with torch.no_grad():
        with torch.autocast('cuda',dtype=torch.bfloat16):logits=m(x).logits
        loss=torch.nn.functional.cross_entropy(logits[:,:-1].float().transpose(1,2),y[:,1:],ignore_index=-100,reduction='none')
    counts=(y[:,1:]!=-100).sum(1);sums=loss.double().sum(1);n=int(counts.sum());total=float(sums.sum())
    return {'tokens':n,'nll':total,'ce':total/n,'ppl':math.exp(total/n),'document_tokens':counts.tolist(),'document_nll':sums.tolist()}
base.setup=setup;base.build=build;base.snapshot=snapshot;base.restore=restore;base.advance=advance;base.evaluate=evaluate
# Child workers must run this CUDA specialization, not the CPU entry point.
base.__file__=__file__
if __name__=='__main__':base.main()
