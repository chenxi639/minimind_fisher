"""Windowed Fisher statistics; backbone attention remains global and causal."""
import math
import torch
from torch.nn import functional as F
from .model_fisher import FisherResidual, FisherMiniMind

class WindowFisherResidual(FisherResidual):
    def forward(self,h,mask=None,state=None):
        valid=torch.ones(h.shape[:2],dtype=torch.bool,device=h.device) if mask is None else mask.bool()
        if valid.shape!=h.shape[:2]:raise ValueError('mask shape mismatch')
        clean=torch.where(valid[...,None],h,torch.zeros_like(h))
        with torch.autocast(h.device.type,enabled=False):
            x=F.linear(self.norm(clean.float()),self.input.weight.float())
            z=self.w.float()[None,None]*(x[:,:,None]+self.b.float()[None,None])
            q=(-.5*z.square().sum(-1)).softmax(-1)[...,None]*valid[:,:,None,None]
            a=q*z;b=q*(z.square()-1)/math.sqrt(2);c=valid.float()[...,None]
            if state is not None:
                if len(state)!=3 or state[0].shape[0]!=h.shape[0] or state[0].shape[2:]!=a.shape[2:]:
                    raise ValueError('invalid window cache')
                a=torch.cat([state[0],a],1);b=torch.cat([state[1],b],1);c=torch.cat([state[2],c],1)
            def rolling(v):
                prefix=torch.cat([torch.zeros_like(v[:,:1]),v.cumsum(1)],1)
                end=torch.arange(1,v.shape[1]+1,device=v.device)
                return (prefix[:,end]-prefix[:,(end-self.window).clamp_min(0)])[:,-h.shape[1]:]
            raw=torch.cat([rolling(a).flatten(2),rolling(b).flatten(2)],-1)/rolling(c).clamp_min(1)
            encoded=F.normalize(raw/torch.sqrt(raw.abs()+1e-6),dim=-1,eps=1e-6)
            delta=F.linear(encoded,self.output.weight.float())*valid[...,None]
            keep=self.window-1
            next_state=tuple(v[:,-keep:] if keep else v[:,:0] for v in (a,b,c))
        return delta.to(h.dtype),next_state

class WindowFisherMiniMind(FisherMiniMind):
    def __init__(self,config):
        super().__init__(config)
        index=getattr(config,'fisher_index',3)
        old=self.model.layers[index].fisher
        new=WindowFisherResidual(config.hidden_size,old.k,old.dim)
        new.window=getattr(config,'fisher_window',32)
        if new.window<1:raise ValueError('window must be positive')
        new.load_state_dict(old.state_dict(),strict=True)
        self.model.layers[index].fisher=new
