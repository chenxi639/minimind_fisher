import unittest
import torch
from scripts.layer_sensitivity_scan import parameter_trace

class SensitivityTests(unittest.TestCase):
    def test_square_before_average_and_no_updates(self):
        p=torch.nn.Parameter(torch.tensor([1.,2.]));before=p.detach().clone();scores=[]
        # Opposing per-example gradients cancel in batch mean but not empirical Fisher.
        for sign in [1.,-1.]:
            p.grad=None;(sign*p.sum()).backward();scores.append(parameter_trace([p]))
        self.assertEqual(sum(scores)/2,2.)
        p.grad=None;((p.sum()-p.sum())/2).backward();self.assertEqual(parameter_trace([p]),0.)
        torch.testing.assert_close(p,before,rtol=0,atol=0)
