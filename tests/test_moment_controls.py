import unittest,math,io
import torch
from torch.nn import functional as F
from model.model_moment_controls import MomentControl,MomentControlMiniMind
from model.model_matched_mlp import MatchedResidual,MatchedMiniMind
from model.model_minimind import MiniMindConfig

class ControlTests(unittest.TestCase):
    def test_counts(self):
        for mode,n in [('mean',99072),('second',99072),('current',148224)]:
            m=MomentControl(768,64,mode);self.assertEqual(sum(p.numel() for p in m.parameters()),n)
            self.assertNotIn('w',dict(m.named_parameters()));self.assertNotIn('b',dict(m.named_parameters()))
        self.assertEqual(sum(p.numel() for p in MatchedResidual(768,96).parameters()),148224)

    def test_formula_causality_padding_cache_reload(self):
        for mode in ('mean','second','current','adapter'):
            with self.subTest(mode=mode):
                torch.manual_seed(43);m=MatchedResidual(12,6) if mode=='adapter' else MomentControl(12,4,mode)
                with torch.no_grad():
                    m.output.weight.normal_(0,.03)
                    if mode!='adapter':m.b.normal_(0,.1)
                h=torch.randn(3,7,12);mask=torch.tensor([[1,1,0,1,1,0,0],[1]*7,[0]*7],dtype=torch.bool)
                h[~mask]=float('nan')
                def run(x,ms):return m(x,ms) if mode=='adapter' else m(x,ms)[0]
                out=run(h,mask);self.assertTrue(torch.isfinite(out).all());self.assertEqual(float(out[~mask].detach().abs().sum()),0)
                x=m.input(m.norm(torch.where(mask[...,None],h,torch.zeros_like(h))))
                if mode=='adapter':expected=m.output(F.silu(x))*mask[...,None]
                else:
                    z=x+m.b[0];a=z*mask[...,None];b=(z.square()-1)/math.sqrt(2)*mask[...,None]
                    raw=torch.cat((a,b),-1) if mode=='current' else (a if mode=='mean' else b).cumsum(1)/mask.float().cumsum(1)[...,None].clamp_min(1)
                    expected=m.output(F.normalize(raw/torch.sqrt(raw.abs()+1e-6),dim=-1,eps=1e-6))*mask[...,None]
                torch.testing.assert_close(out,expected)
                altered=h.clone();altered[:,4:]+=10
                torch.testing.assert_close(out[:,:4],run(altered,mask)[:,:4],rtol=0,atol=0)
                if mode=='adapter':joined=torch.cat((m(h[:,:3],mask[:,:3]),m(h[:,3:],mask[:,3:])),1)
                else:
                    left,state=m(h[:,:3],mask[:,:3]);right,_=m(h[:,3:],mask[:,3:],state);joined=torch.cat((left,right),1)
                torch.testing.assert_close(out,joined,atol=1e-6,rtol=1e-5)
                if mode=='current':
                    changed=h.clone();changed[:,:3]=10
                    torch.testing.assert_close(out[:,3:],run(changed,mask)[:,3:],rtol=0,atol=0)
                stream=io.BytesIO();torch.save(m.state_dict(),stream);stream.seek(0);m.load_state_dict(torch.load(stream,weights_only=True))
                torch.testing.assert_close(out,run(h,mask),rtol=0,atol=0)

    def test_gradient(self):
        for mode in ('mean','second','current','adapter'):
            torch.manual_seed(43);m=MatchedResidual(12,6) if mode=='adapter' else MomentControl(12,4,mode)
            h=torch.randn(2,7,12);target=torch.randn_like(h)
            for step in range(2):
                m.zero_grad();out=m(h) if mode=='adapter' else m(h)[0]
                if step==0:self.assertEqual(float(out.detach().abs().sum()),0)
                (out*target).sum().backward()
                self.assertGreater(float(m.output.weight.grad.norm()),0)
                if step==1:self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in m.parameters()))
                with torch.no_grad():m.output.weight.add_(-.01*m.output.weight.grad)

    def test_model_cache(self):
        for mode in ('mean','second','current','adapter'):
            torch.manual_seed(42)
            cfg=MiniMindConfig(hidden_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=16,vocab_size=128,intermediate_size=128,fisher_index=0,fisher_dim=4,moment_mode=mode,adapter_width=6,flash_attn=False)
            m=(MatchedMiniMind(cfg) if mode=='adapter' else MomentControlMiniMind(cfg)).eval()
            branch=getattr(m.model.layers[0],'adapter' if mode=='adapter' else 'fisher')
            with torch.no_grad():
                branch.output.weight.normal_(0,.02);ids=torch.randint(0,128,(2,7))
                full=m(ids).logits;left=m(ids[:,:3],use_cache=True);right=m(ids[:,3:],past_key_values=left.past_key_values,use_cache=True)
                torch.testing.assert_close(full[:,3:],right.logits,atol=2e-5,rtol=1e-4)

if __name__=='__main__':unittest.main()
