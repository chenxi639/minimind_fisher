"""Fixed-reference prefix moments; no learnable component center or scale."""
import torch
from .model_fisher import FisherResidual, FisherMiniMind

class MomentResidual(FisherResidual):
    def __init__(self, hidden, dim=64):
        super().__init__(hidden,k=1,dim=dim)
        del self.w
        del self.b
        self.register_buffer('w',torch.ones(1,dim))
        self.register_buffer('b',torch.zeros(1,dim))

class MomentMiniMind(FisherMiniMind):
    def __init__(self,config):
        if getattr(config,'fisher_k',1)!=1:
            raise ValueError('Moment requires fisher_k=1')
        config.fisher_k=1
        super().__init__(config)
        self.model.layers[getattr(config,'fisher_index',3)].fisher=MomentResidual(
            config.hidden_size,getattr(config,'fisher_dim',64))
