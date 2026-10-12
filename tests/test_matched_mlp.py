import unittest,io
import torch
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_matched_mlp import MatchedMiniMind,MatchedResidual

class MatchedTests(unittest.TestCase):
    def test_identity_causality_cache_reload_gradients(self):
        torch.manual_seed(42)
        cfg=MiniMindConfig(hidden_size=32,num_hidden_layers=2,vocab_size=64,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,flash_attn=False,fisher_index=0,adapter_width=24)
        base=MiniMindForCausalLM(cfg).eval();m=MatchedMiniMind(cfg).eval();m.load_state_dict(base.state_dict(),strict=False)
        x=torch.randint(1,64,(2,9));torch.testing.assert_close(base(x).logits,m(x).logits,rtol=0,atol=0)
        f=m.model.layers[0].adapter
        m(x,labels=x).loss.backward();self.assertEqual(float(f.input.weight.grad.norm()),0.)
        self.assertGreater(float(f.output.weight.grad.norm()),0.)
        m.zero_grad();torch.nn.init.normal_(f.output.weight,std=.05)
        full=m(x).logits;changed=x.clone();changed[:,5:]=1
        torch.testing.assert_close(full[:,:5],m(changed).logits[:,:5])
        first=m(x[:,:4],use_cache=True);last=m(x[:,4:],past_key_values=first.past_key_values,use_cache=True)
        torch.testing.assert_close(full,torch.cat([first.logits,last.logits],1),rtol=1e-4,atol=1e-6)
        m(x,labels=x).loss.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in f.parameters()))
        buffer=io.BytesIO();torch.save(m.state_dict(),buffer);buffer.seek(0);other=MatchedMiniMind(cfg).eval();other.load_state_dict(torch.load(buffer,weights_only=True))
        torch.testing.assert_close(full,other(x).logits,rtol=0,atol=0)
    def test_mask_and_parameter_budget(self):
        f=MatchedResidual(768,544);self.assertEqual(sum(p.numel() for p in f.parameters()),836352)
        torch.nn.init.normal_(f.output.weight,std=.01)
        x=torch.randn(2,7,768);mask=torch.rand(2,7)>.3;x[~mask]=float('nan')
        out=f(x,mask);self.assertTrue(torch.isfinite(out).all());self.assertEqual(float(out[~mask].abs().sum()),0.)
