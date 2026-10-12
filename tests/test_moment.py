import unittest,math,io
import torch
from model.model_fisher import FisherResidual
from model.model_moment import MomentResidual

class MomentTests(unittest.TestCase):
    def test_counts(self):
        a=FisherResidual(768,1,64);b=MomentResidual(768,64)
        self.assertEqual(sum(p.numel() for p in a.parameters()),148352)
        self.assertEqual(sum(p.numel() for p in b.parameters()),148224)
        self.assertNotIn('w',dict(b.named_parameters()));self.assertNotIn('b',dict(b.named_parameters()))

    def test_formula_cache_causal_reload(self):
        torch.manual_seed(42);a=FisherResidual(12,1,4);b=MomentResidual(12,4)
        with torch.no_grad():a.output.weight.normal_(0,.02)
        b.load_state_dict(a.state_dict(),strict=True)
        h=torch.randn(2,7,12);mask=torch.tensor([[1,1,1,1,1,0,0],[1,1,1,1,1,1,1]],dtype=torch.bool)
        out,cache=b(h,mask);torch.testing.assert_close(out,a(h,mask)[0],rtol=0,atol=0)
        x=b.input(b.norm(torch.where(mask[...,None],h,torch.zeros_like(h))))
        z=x+b.b[0];zs=z*mask[...,None];z2=(z.square()-1)/math.sqrt(2)*mask[...,None]
        raw=torch.cat((zs.cumsum(1),z2.cumsum(1)),-1)/mask.float().cumsum(1)[...,None].clamp_min(1)
        expected=b.output(torch.nn.functional.normalize(raw/torch.sqrt(raw.abs()+1e-6),dim=-1,eps=1e-6))*mask[...,None]
        torch.testing.assert_close(out,expected)
        left,state=b(h[:,:3],mask[:,:3]);right,_=b(h[:,3:],mask[:,3:],state)
        torch.testing.assert_close(out,torch.cat((left,right),1),atol=1e-6,rtol=1e-5)
        altered=h.clone();altered[:,4:]+=10
        torch.testing.assert_close(out[:,:4],b(altered,mask)[0][:,:4],rtol=0,atol=0)
        stream=io.BytesIO();torch.save(b.state_dict(),stream);stream.seek(0)
        c=MomentResidual(12,4);c.load_state_dict(torch.load(stream,weights_only=True),strict=True)
        torch.testing.assert_close(out,c(h,mask)[0],rtol=0,atol=0)

    def test_zero_and_gradient(self):
        torch.manual_seed(43);m=MomentResidual(12,4);x=torch.randn(2,5,12)
        y=m(x)[0];self.assertEqual(float(y.detach().abs().sum()),0)
        target=torch.randn_like(y);(y*target).sum().backward()
        self.assertGreater(float(m.output.weight.grad.norm()),0)
        self.assertEqual(float(m.input.weight.grad.norm()),0)
        with torch.no_grad():m.output.weight.add_(-.01*m.output.weight.grad)
        m.zero_grad();(m(x)[0]*target).sum().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in m.parameters()))

if __name__=='__main__':unittest.main()
