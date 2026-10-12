import torch
import io
from tests.test_pair import PairTests
from model.model_fisher import FisherResidual
from model.model_fisher_window import WindowFisherMiniMind,WindowFisherResidual

class WindowTests(PairTests):
    def setUp(self):
        super().setUp();self.cfg.fisher_window=4
        self.model=WindowFisherMiniMind(self.cfg).eval()
        missing=self.model.load_state_dict(self.base.state_dict(),strict=False)
        assert not missing.unexpected_keys and all('.fisher.' in k for k in missing.missing_keys)

    def test_active_window_reload(self):
        torch.nn.init.normal_(self.model.model.layers[0].fisher.output.weight,std=.1)
        file=io.BytesIO();torch.save(self.model.state_dict(),file);file.seek(0)
        other=WindowFisherMiniMind(self.cfg).eval()
        other.load_state_dict(torch.load(file,weights_only=True),strict=True)
        torch.testing.assert_close(self.model(self.x).logits,other(self.x).logits,rtol=0,atol=0)

    def test_window_reference_and_padding(self):
        f=WindowFisherResidual(4,2,3);f.window=4
        torch.nn.init.normal_(f.output.weight,std=.1)
        ref=FisherResidual(4,2,3);ref.load_state_dict(f.state_dict())
        x=torch.randn(2,11,4);mask=torch.rand(2,11)>.3
        poisoned=x.clone();poisoned[~mask]=float('nan')
        actual,state=f(poisoned,mask)
        expected=torch.stack([ref(poisoned[:,max(0,t-3):t+1],mask[:,max(0,t-3):t+1])[0][:,-1] for t in range(11)],1)
        torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-6)
        first,cache=f(poisoned[:,:6],mask[:,:6]);last,cache=f(poisoned[:,6:],mask[:,6:],cache)
        torch.testing.assert_close(actual,torch.cat([first,last],1),rtol=1e-4,atol=1e-6)
        self.assertEqual(cache[0].shape[1],3)
        changed=x.clone();changed[:,:7]+=10
        torch.testing.assert_close(f(x,mask)[0][:,-1],f(changed,mask)[0][:,-1],rtol=1e-4,atol=1e-6)
