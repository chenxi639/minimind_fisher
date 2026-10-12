import torch
from tests.test_window import WindowTests
from model.model_fisher_window import WindowFisherMiniMind,WindowFisherResidual
from model.model_fisher import FisherResidual

class W1Tests(WindowTests):
    def setUp(self):
        super().setUp();self.cfg.fisher_window=1
        self.model=WindowFisherMiniMind(self.cfg).eval()
        self.model.load_state_dict(self.base.state_dict(),strict=False)
    def test_w1_actual_centers_formula_and_empty_cache(self):
        initial=torch.load('runs/kmeans_ready_20260930_224711/fisher_init.pt',weights_only=True)
        state={k.split('.fisher.')[1]:v for k,v in initial.items()}
        f=WindowFisherResidual(768,8,64);f.window=1;f.load_state_dict(state)
        torch.nn.init.normal_(f.output.weight,std=.01)
        ref=FisherResidual(768,8,64);ref.load_state_dict(f.state_dict())
        x=torch.randn(2,340,768);mask=torch.rand(2,340)>.15
        x[~mask]=float('nan')
        actual,cache=f(x,mask)
        self.assertTrue(all(v.shape[1]==0 for v in cache))
        expected=ref(x.reshape(-1,1,768),mask.reshape(-1,1))[0].reshape_as(actual)
        torch.testing.assert_close(actual,expected,rtol=3e-4,atol=3e-5)
        first,c=f(x[:,:137],mask[:,:137]);last,c=f(x[:,137:],mask[:,137:],c)
        torch.testing.assert_close(actual,torch.cat([first,last],1),rtol=3e-4,atol=3e-5)
