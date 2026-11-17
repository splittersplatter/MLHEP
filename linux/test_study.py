import unittest
import numpy as np
import torch
from study import disco_squared, inject, partitions

class ScientificChecks(unittest.TestCase):
    def test_distance_correlation(self):
        x = torch.linspace(-1, 1, 100, dtype=torch.float64, requires_grad=True)
        self.assertAlmostEqual(disco_squared(x,x).item(), 1., places=10)
        self.assertEqual(disco_squared(x*0,x).item(),0.)
        loss=disco_squared(x,x.square())
        self.assertGreater(loss.item(),.1)
        loss.backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertAlmostEqual(disco_squared(x.detach(),x.square().detach()).item(),
                               disco_squared(x.detach()*5+3,x.square().detach()*2).item(),places=8)
    def test_independence_and_constant_gradient(self):
        # Cartesian product has exactly independent empirical marginals.
        grid=torch.linspace(-1,1,10,dtype=torch.float64)
        x=grid.repeat_interleave(10)
        y=grid.repeat(10)
        self.assertAlmostEqual(disco_squared(x,y).item(),0.,places=12)
        constant=torch.ones(20,dtype=torch.float64,requires_grad=True)
        disco_squared(constant,torch.arange(20,dtype=torch.float64)).backward()
        self.assertTrue(torch.isfinite(constant.grad).all())

    def test_injection_and_splits(self):
        truth=np.tile([0,0,0,1],1000)
        mass=np.tile([3200,3400,3800,3500],1000)
        data=dict(truth=truth,mass=mass,row=np.arange(60000,64000))
        tr,va,te=partitions(data,11)
        self.assertFalse(set(tr)&set(va) or set(tr)&set(te) or set(va)&set(te))
        self.assertEqual(set(te),set(partitions(data,22)[2]))
        sr=((mass>=3300)&(mass<3700)).astype(int)
        zero=inject(tr,truth,sr,0,1)
        one=inject(tr,truth,sr,.01,1)
        two=inject(tr,truth,sr,.02,1)
        self.assertFalse(truth[zero].any())
        self.assertTrue(set(one)<=set(two))
        self.assertEqual(int(truth[one].sum()),round(.01/.99*sr[zero].sum()))

if __name__=='__main__': unittest.main()
