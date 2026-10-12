"""Stage24 fixed-reference moment controls; historical models remain unchanged."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model_moment import MomentResidual
from .model_fisher import FisherMiniMind

class MomentControl(MomentResidual):
    def __init__(self, hidden, dim=64, mode='mean'):
        if mode not in ('mean','second','current'):raise ValueError('unknown moment control')
        super().__init__(hidden,dim);self.mode=mode
        self.output=nn.Linear(dim if mode!='current' else 2*dim,hidden,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self,h,mask=None,state=None):
        valid=torch.ones(h.shape[:2],device=h.device,dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape!=h.shape[:2]:raise ValueError('mask shape mismatch')
        if h.shape[1]==0:raise ValueError('empty sequence')
        clean=torch.where(valid[...,None],h,torch.zeros_like(h))
        with torch.autocast(h.device.type,enabled=False):
            x=F.linear(self.norm(clean.float()),self.input.weight.float())
            z=self.w[0].float()*(x+self.b[0].float())
            a=z*valid[...,None];b=(z.square()-1)/math.sqrt(2)*valid[...,None]
            count=valid.float()[...,None]
            if state is not None:
                if len(state)!=3 or state[0].shape!=a[:,-1].shape or state[1].shape!=b[:,-1].shape or state[2].shape!=count[:,-1].shape:
                    raise ValueError('invalid control cache')
            if self.mode!='current':
                a=a.cumsum(1);b=b.cumsum(1);count=count.cumsum(1)
                if state is not None:
                    a=a+state[0][:,None];b=b+state[1][:,None];count=count+state[2][:,None]
                raw=(a if self.mode=='mean' else b)/count.clamp_min(1)
            else:
                raw=torch.cat((a,b),-1)
            encoded=F.normalize(raw/torch.sqrt(raw.abs()+1e-6),dim=-1,eps=1e-6)
            delta=F.linear(encoded,self.output.weight.float())*valid[...,None]
        # Current-only accepts the block's cache contract but never reads history.
        return delta.to(h.dtype),(a[:,-1],b[:,-1],count[:,-1])

class MomentControlMiniMind(FisherMiniMind):
    def __init__(self,config):
        config.fisher_k=1;super().__init__(config)
        self.model.layers[getattr(config,'fisher_index',3)].fisher=MomentControl(
            config.hidden_size,getattr(config,'fisher_dim',64),getattr(config,'moment_mode','mean'))
