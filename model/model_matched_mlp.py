"""Stage14 preregistered near-parameter-matched tokenwise residual MLP."""
import torch
from torch import nn
from torch.nn import functional as F
from .model_minimind import MiniMindForCausalLM,MiniMindBlock,RMSNorm

class MatchedResidual(nn.Module):
    def __init__(self,hidden,width=544):
        super().__init__();self.norm=RMSNorm(hidden)
        self.input=nn.Linear(hidden,width,bias=False);self.output=nn.Linear(width,hidden,bias=False)
        nn.init.zeros_(self.output.weight)
    def forward(self,h,mask=None):
        valid=torch.ones(h.shape[:2],device=h.device,dtype=torch.bool) if mask is None else mask.bool()
        if valid.shape!=h.shape[:2]:raise ValueError('mask shape mismatch')
        clean=torch.where(valid[...,None],h,torch.zeros_like(h))
        with torch.autocast(h.device.type,enabled=False):
            x=F.silu(F.linear(self.norm(clean.float()),self.input.weight.float()))
            delta=F.linear(x,self.output.weight.float())*valid[...,None]
        return delta.to(h.dtype)

class MatchedBlock(MiniMindBlock):
    def forward(self,hidden_states,position_embeddings,past_key_value=None,use_cache=False,attention_mask=None):
        h,present=super().forward(hidden_states,position_embeddings,past_key_value,use_cache,attention_mask)
        mask=attention_mask[:,-h.shape[1]:] if attention_mask is not None else None
        return h+self.adapter(h,mask),present

class MatchedMiniMind(MiniMindForCausalLM):
    def __init__(self,config):
        super().__init__(config);index=getattr(config,'fisher_index',3)
        if not 0<=index<config.num_hidden_layers:raise ValueError('invalid index')
        old=self.model.layers[index];block=MatchedBlock(index,config);block.load_state_dict(old.state_dict(),strict=True)
        block.adapter=MatchedResidual(config.hidden_size,getattr(config,'adapter_width',544))
        self.model.layers[index]=block
