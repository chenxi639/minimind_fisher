import unittest
import torch
from model.model_minimind import MiniMindConfig,MiniMindForCausalLM
from model.model_peft25 import install,enabled,is_adaptation,merged_lora

class PeftTests(unittest.TestCase):
    def test_zero_gradient_cache_merge(self):
        torch.set_num_threads(4)
        cfg=MiniMindConfig(hidden_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,vocab_size=128,flash_attn=False)
        for kind in ['houlsby','lora']:
            torch.manual_seed(42);m=MiniMindForCausalLM(cfg).eval();ids=torch.randint(0,128,(2,9));ref=m(ids).logits.detach()
            old={k:v.clone() for k,v in m.state_dict().items()};torch.manual_seed(43);install(m,kind)
            torch.testing.assert_close(m(ids).logits,ref,rtol=0,atol=0)
            params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.001)
            for step in range(2):
                opt.zero_grad();m(ids,labels=ids).loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in params))
                if step:self.assertTrue(all(p.grad.norm()>0 for p in params))
                opt.step()
            self.assertTrue(all(torch.equal(m.state_dict()[k],v) for k,v in old.items()))
            full=m(ids).logits;left=m(ids[:,:4],use_cache=True);right=m(ids[:,4:],past_key_values=left.past_key_values,use_cache=True)
            torch.testing.assert_close(full[:,4:],right.logits,atol=2e-5,rtol=1e-4)
            changed=ids.clone();changed[:,5:]=1;torch.testing.assert_close(full[:,:5],m(changed).logits[:,:5],atol=0,rtol=0)
            if kind=='lora':torch.testing.assert_close(full,merged_lora(m)(ids).logits,atol=2e-5,rtol=1e-4)
            enabled(m,False);torch.testing.assert_close(m(ids).logits,ref,atol=0,rtol=0)
    def test_counts(self):
        for kind,count in [('houlsby',159840),('lora',150528)]:
            m=install(MiniMindForCausalLM(MiniMindConfig()),kind)
            self.assertEqual(sum(p.numel() for p in m.parameters() if p.requires_grad),count)
            self.assertTrue(all(p.requires_grad==is_adaptation(n) for n,p in m.named_parameters()))

if __name__=='__main__':unittest.main()
