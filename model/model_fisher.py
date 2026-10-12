"""Experimental causal Fisher branch; upstream MiniMind stays untouched."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model_minimind import MiniMindForCausalLM, MiniMindBlock, RMSNorm


class FisherResidual(nn.Module):
    def __init__(self, hidden, k=8, dim=64):
        super().__init__()
        self.k, self.dim = k, dim
        self.norm = RMSNorm(hidden)
        self.input = nn.Linear(hidden, dim, bias=False)
        self.w = nn.Parameter(torch.ones(k, dim))
        self.b = nn.Parameter(torch.randn(k, dim) * .01)
        self.output = nn.Linear(2*k*dim, hidden, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, h, mask=None, state=None):
        valid = torch.ones(h.shape[:2], device=h.device, dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape != h.shape[:2]:
            raise ValueError('mask shape mismatch')
        clean = torch.where(valid[..., None], h, torch.zeros_like(h))
        with torch.autocast(h.device.type, enabled=False):
            x = F.linear(self.norm(clean.float()), self.input.weight.float())
            z = self.w.float()[None,None] * (x[:,:,None] + self.b.float()[None,None])
            q = (-.5*z.square().sum(-1)).softmax(-1)[...,None] * valid[:,:,None,None]
            a = (q*z).cumsum(1)
            b = (q*(z.square()-1)/math.sqrt(2)).cumsum(1)
            count = valid.float().cumsum(1)[...,None]
            if state is not None:
                if len(state)!=3 or state[0].shape != a[:,-1].shape or state[2].shape != count[:,-1].shape:
                    raise ValueError('invalid Fisher cache')
                a=a+state[0][:,None]; b=b+state[1][:,None]; count=count+state[2][:,None]
            raw = torch.cat((a.flatten(2), b.flatten(2)), -1)/count.clamp_min(1)
            power = raw/torch.sqrt(raw.abs()+1e-6)
            encoded = F.normalize(power,dim=-1,eps=1e-6)
            delta=F.linear(encoded,self.output.weight.float())*valid[...,None]
        return delta.to(h.dtype), (a[:,-1],b[:,-1],count[:,-1])


class FisherBlock(MiniMindBlock):
    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        if past_key_value is not None and len(past_key_value)!=5:
            raise ValueError('Fisher block requires KV plus Fisher cache')
        kv = past_key_value[:2] if past_key_value is not None else None
        state = past_key_value[2:] if past_key_value is not None else None
        h,present=super().forward(hidden_states,position_embeddings,kv,use_cache,attention_mask)
        mask=attention_mask[:,-h.shape[1]:] if attention_mask is not None else None
        delta,state=self.fisher(h,mask,state)
        return h+delta, (*present,*state) if use_cache else None


class FisherMiniMind(MiniMindForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        index=getattr(config,'fisher_index',3)
        if not 0 <= index < config.num_hidden_layers:
            raise ValueError('invalid insertion index')
        # Attach after upstream post_init, preserving shared names and zero projection.
        old=self.model.layers[index]
        block=FisherBlock(index,config)
        block.load_state_dict(old.state_dict(),strict=True)
        block.fisher=FisherResidual(config.hidden_size,getattr(config,'fisher_k',8),getattr(config,'fisher_dim',64))
        self.model.layers[index]=block
