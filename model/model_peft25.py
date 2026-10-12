"""Explicit MiniMind ports: two bottleneck adapters/layer or Q/V LoRA."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model_minimind import MiniMindForCausalLM

class Bottleneck(nn.Module):
    def __init__(self,hidden,width):
        super().__init__();self.down=nn.Linear(hidden,width);self.up=nn.Linear(width,hidden);self.enabled=True
        nn.init.zeros_(self.down.bias);nn.init.zeros_(self.up.weight);nn.init.zeros_(self.up.bias)
    def forward(self,x):
        if not self.enabled:return x
        with torch.autocast(x.device.type,enabled=False):delta=self.up(F.relu(self.down(x.float())))
        return x+delta.to(x.dtype)

class LoRALinear(nn.Module):
    def __init__(self,base,rank=7,alpha=7):
        super().__init__();self.weight=base.weight;self.bias=base.bias;self.scale=alpha/rank;self.enabled=True
        self.lora_A=nn.Parameter(torch.empty(rank,base.in_features));self.lora_B=nn.Parameter(torch.zeros(base.out_features,rank))
        nn.init.kaiming_uniform_(self.lora_A,a=math.sqrt(5))
    def forward(self,x):
        y=F.linear(x,self.weight,self.bias)
        if self.enabled:
            with torch.autocast(x.device.type,enabled=False):delta=F.linear(F.linear(x.float(),self.lora_A),self.lora_B)*self.scale
            y=y+delta.to(y.dtype)
        return y

def is_adaptation(name):return name.startswith('peft_adapters.') or '.lora_' in name

def install(model,kind):
    """Install after loading official weights; no reinitialization of backbone."""
    if kind not in ('houlsby','lora'):raise ValueError(kind)
    model.requires_grad_(False)
    if kind=='houlsby':
        model.peft_adapters=nn.ModuleDict()
        for i,block in enumerate(model.model.layers):
            a=Bottleneck(model.config.hidden_size,6);b=Bottleneck(model.config.hidden_size,6)
            model.peft_adapters[f'attn_{i}']=a;model.peft_adapters[f'ffn_{i}']=b
            block.self_attn.register_forward_hook(lambda layer,args,out,adapter=a:(adapter(out[0]),out[1]))
            block.mlp.register_forward_hook(lambda layer,args,out,adapter=b:adapter(out))
    else:
        for block in model.model.layers:
            for name in ('q_proj','v_proj'):setattr(block.self_attn,name,LoRALinear(getattr(block.self_attn,name)))
    return model

def enabled(model,value):
    for layer in model.modules():
        if isinstance(layer,(Bottleneck,LoRALinear)):layer.enabled=value

def adaptation_state(model):return {n:v.detach().cpu().clone() for n,v in model.state_dict().items() if is_adaptation(n)}

def merged_lora(model):
    weights={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if not is_adaptation(n)}
    for n,layer in model.named_modules():
        if isinstance(layer,LoRALinear):weights[n+'.weight']+=layer.lora_B.detach().cpu()@layer.lora_A.detach().cpu()*layer.scale
    merged=MiniMindForCausalLM(model.config);merged.load_state_dict(weights,strict=True)
    return merged.eval()
