from tests.test_pair import PairTests
from model.model_fisher import FisherMiniMind
import torch
from model.model_fisher import FisherResidual

class K1Tests(PairTests):
    def setUp(self):
        super().setUp()
        self.cfg.fisher_k=1
        self.model=FisherMiniMind(self.cfg).eval()
        incompatible=self.model.load_state_dict(self.base.state_dict(),strict=False)
        assert not incompatible.unexpected_keys
        assert all('.fisher.' in k for k in incompatible.missing_keys)

    def test_k1_masked_moment_formula(self):
        f=FisherResidual(4,1,3)
        x=torch.randn(2,5,4);mask=torch.tensor([[0,1,1,0,1],[1,0,1,1,0]],dtype=torch.bool)
        poisoned=x.clone();poisoned[~mask]=float('nan')
        delta,state=f(poisoned,mask)
        self.assertTrue(torch.isfinite(delta).all())
        for b in range(2):
            selected=x[b][mask[b]]
            z=f.w*(f.input(f.norm(selected))+f.b)
            torch.testing.assert_close(state[0][b,0],z.sum(0))
            torch.testing.assert_close(state[1][b,0],((z.square()-1)/2**.5).sum(0))
