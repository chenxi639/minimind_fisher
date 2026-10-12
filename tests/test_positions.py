import unittest
import torch
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_fisher import FisherMiniMind

class PositionTests(unittest.TestCase):
    def test_locations_identity_causality_cache_gradients(self):
        for index in [1,3,6]:
            with self.subTest(index=index):
                torch.manual_seed(42)
                cfg=MiniMindConfig(hidden_size=32,num_hidden_layers=8,vocab_size=64,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,flash_attn=False,fisher_index=index,fisher_k=8,fisher_dim=4)
                base=MiniMindForCausalLM(cfg).eval();model=FisherMiniMind(cfg).eval()
                model.load_state_dict(base.state_dict(),strict=False)
                x=torch.randint(1,64,(2,9))
                torch.testing.assert_close(base(x).logits,model(x).logits,rtol=0,atol=0)
                self.assertEqual([i for i,b in enumerate(model.model.layers) if hasattr(b,'fisher')],[index])
                f=model.model.layers[index].fisher
                torch.nn.init.normal_(f.output.weight,std=.05)
                full=model(x).logits;y=x.clone();y[:,5:]=1
                torch.testing.assert_close(full[:,:5],model(y).logits[:,:5])
                a=model(x[:,:4],use_cache=True);b=model(x[:,4:],past_key_values=a.past_key_values,use_cache=True)
                torch.testing.assert_close(full,torch.cat([a.logits,b.logits],1),rtol=1e-4,atol=1e-6)
                model(x,labels=x).loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in f.parameters()))
