import io
import unittest
import torch
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_fisher import FisherMiniMind, FisherResidual

class PairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg=MiniMindConfig(hidden_size=32,num_hidden_layers=2,vocab_size=64,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=64,flash_attn=False,fisher_index=0,fisher_k=2,fisher_dim=4)
        self.base=MiniMindForCausalLM(self.cfg).eval()
        self.model=FisherMiniMind(self.cfg).eval()
        missing=self.model.load_state_dict(self.base.state_dict(),strict=False)
        self.assertFalse(missing.unexpected_keys)
        self.assertTrue(all('.fisher.' in k for k in missing.missing_keys))
        self.x=torch.randint(1,64,(2,9))

    def test_identity_and_reload(self):
        torch.testing.assert_close(self.base(self.x).logits,self.model(self.x).logits,rtol=0,atol=0)
        f=io.BytesIO();torch.save(self.model.state_dict(),f);f.seek(0)
        other=FisherMiniMind(self.cfg).eval();other.load_state_dict(torch.load(f,weights_only=True))
        torch.testing.assert_close(other(self.x).logits,self.model(self.x).logits,rtol=0,atol=0)

    def test_active_causality_and_chunk_cache(self):
        torch.nn.init.normal_(self.model.model.layers[0].fisher.output.weight,std=.1)
        full=self.model(self.x).logits
        y=self.x.clone();y[:,5:]=1
        torch.testing.assert_close(full[:,:5],self.model(y).logits[:,:5])
        for sizes in [[1]*9,[3,2,4]]:
            cache=None; pieces=[];start=0
            for size in sizes:
                out=self.model(self.x[:,start:start+size],past_key_values=cache,use_cache=True)
                pieces.append(out.logits);cache=out.past_key_values;start+=size
            torch.testing.assert_close(full,torch.cat(pieces,1),rtol=1e-4,atol=1e-6)

    def test_mask_and_formula(self):
        f=FisherResidual(4,2,3)
        torch.nn.init.normal_(f.output.weight,std=.1)
        x=torch.randn(2,5,4);mask=torch.tensor([[0,1,1,0,1],[0,0,0,0,0]],dtype=torch.bool)
        poisoned=x.clone();poisoned[~mask]=float('nan')
        y,state=f(poisoned,mask)
        self.assertTrue(torch.isfinite(y).all())
        torch.testing.assert_close(y,f(x,mask)[0])
        torch.testing.assert_close(state[2],torch.tensor([[3.],[0.]]))
        # Independent prefix reference, using explicit sums instead of cumsum.
        for t in range(5):
            selected=x[0,:t+1][mask[0,:t+1]]
            if not len(selected):continue
            z=f.w[None]*(f.input(f.norm(selected))[:,None]+f.b[None])
            q=(-.5*z.square().sum(-1)).softmax(-1)[...,None]
            torch.testing.assert_close(f(x[:1,:t+1],mask[:1,:t+1])[1][0][0],(q*z).sum(0))
            torch.testing.assert_close(f(x[:1,:t+1],mask[:1,:t+1])[1][1][0],(q*(z.square()-1)/2**.5).sum(0))

    def test_two_step_gradient_flow(self):
        optimizer=torch.optim.AdamW(self.model.parameters(),lr=.001)
        for step in range(2):
            optimizer.zero_grad();self.model(self.x,labels=self.x).loss.backward()
            f=self.model.model.layers[0].fisher
            for p in f.parameters():self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(f.output.weight.grad.abs().sum().item(),0)
            if step:self.assertGreater(f.w.grad.abs().sum().item(),0)
            optimizer.step()

    def test_masked_chunk_state_and_reset(self):
        f=FisherResidual(4,2,3)
        torch.nn.init.normal_(f.output.weight,std=.1)
        x=torch.randn(2,7,4);mask=torch.tensor([[0,0,1,1,0,1,1],[1,1,1,0,0,0,0]],dtype=torch.bool)
        full,_=f(x,mask)
        first,state=f(x[:,:3],mask[:,:3])
        second,state=f(x[:,3:],mask[:,3:],state)
        torch.testing.assert_close(full,torch.cat([first,second],1))
        torch.testing.assert_close(state[2],torch.tensor([[4.],[3.]]))
        torch.testing.assert_close(f(x,mask)[0],full)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
    def test_cuda_bf16_cache_and_gradients(self):
        model=self.model.cuda()
        torch.nn.init.normal_(model.model.layers[0].fisher.output.weight,std=.02)
        x=self.x.cuda()
        with torch.autocast('cuda',dtype=torch.bfloat16):
            full=model(x,labels=x,use_cache=True)
            self.assertEqual(full.past_key_values[0][2].dtype,torch.float32)
            cache=None;pieces=[]
            for i in range(x.shape[1]):
                out=model(x[:,i:i+1],past_key_values=cache,use_cache=True)
                cache=out.past_key_values;pieces.append(out.logits)
        torch.testing.assert_close(full.logits,torch.cat(pieces,1),rtol=.03,atol=.005)
        full.loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

if __name__=='__main__':unittest.main(verbosity=2)
