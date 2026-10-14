import unittest
import numpy as np
import pandas as pd

from multilingual_megatron.experiments.scaling.stage1_fit import atlas_full_model as m
from multilingual_megatron.experiments.scaling.stage1_fit.atlas_full_pilot import packet


class AtlasFullTests(unittest.TestCase):
    def fixture(self):
        D = np.array([[.5, .1, .2, .1, .1], [1., .3, .4, .2, .2], [4., 2., 3., 1., .5]])*1e9
        return dict(D=D, U=np.ones((3, 5))*1e9, N=[1e8, 2e8, 4e8],
                    physical_floor=-5., training_minimum=-2.)

    def test_null_and_pre_epoch(self):
        D=np.array([0., .5, 1., 2., 10.])
        s,g=m.saturation(D, np.ones(5), 0.)
        np.testing.assert_array_equal(s, D)
        np.testing.assert_array_equal(g[:3], 0.)
        np.testing.assert_allclose(g[3:], [-.5, -40.5])

    def test_invalid_inventory(self):
        with self.assertRaises(ValueError): m.saturation([1.], [0.], .1)
        with self.assertRaises(ValueError): m.saturation([1.], [1.], -.1)
        np.testing.assert_equal(m.saturation([0.], [0.], 1.)[0], [0.])

    def test_saturation_monotonic_and_bounded(self):
        D=np.linspace(0,100,1001)
        s,g=m.saturation(D, np.ones_like(D), .2)
        self.assertTrue(np.all(np.diff(s)>=0))
        self.assertTrue(np.all(s<=D+1e-12))
        self.assertTrue(np.all(s<=6))
        self.assertTrue(np.all(g<=0))

    def test_full_derivatives(self):
        p=self.fixture();x=np.array([.5,.3,.8,.4,.3,.2,.1,.2,.3,.4])
        _,jac=m.predict(x,p)
        for j in range(10):
            h=np.eye(10)[j]*1e-6
            numeric=(m.predict(x+h,p)[0]-m.predict(x-h,p)[0])/2e-6
            np.testing.assert_allclose(jac[:,j],numeric,atol=1e-8,rtol=1e-6)

    def test_chinchilla_limit(self):
        p=self.fixture();x=np.array([.5,.3,.8,.4,.3,0.,0.,0.,0.,0.])
        expected=-3.5+.3*(np.array(p['N'])/1e8)**(-.4)+.8*(p['D'][:,0]/1e9)**(-.3)
        np.testing.assert_allclose(m.predict(x,p)[0],expected)

    def test_small_rate_derivative(self):
        D=np.array([.5,1.,1.01,2.,20.]);U=np.ones(5)
        for rate in [1e-8,1e-5,.1,3.]:
            h=min(1e-6,rate/2)
            numeric=(m.saturation(D,U,rate+h)[0]-m.saturation(D,U,rate-h)[0])/(2*h)
            np.testing.assert_allclose(m.saturation(D,U,rate)[1],numeric,atol=1e-6,rtol=1e-5)

    def test_synthetic_fit(self):
        rng=np.random.default_rng(20260906);n=80
        p=dict(D=np.exp(rng.uniform(np.log(1e7),np.log(1e10),(n,5))),
               U=np.full((n,5),1e9), N=np.exp(rng.uniform(np.log(1e8),np.log(1e9),n)),
               physical_floor=-5.,training_minimum=-2.,weights=np.ones(n),delta=.1)
        x=np.array([.5,.3,.8,.4,.3,.2,.1,.2,.3,.4]);p['y']=m.predict(x,p)[0]
        result=m.fit_start(p,0)
        self.assertTrue(result['success'], result['message'])
        self.assertLess(result['objective'],1e-12)

    def test_prediction_rejects_collapsed_donors(self):
        p=self.fixture();p['D']=p['D'][:,:3];p['U']=p['U'][:,:3]
        with self.assertRaises(ValueError):m.predict(np.ones(10),p)

    def test_no_heldout_influence(self):
        rows=[]
        for t in ['train','test']:
            for k in [1,4]:
                for l in list('abcd')[:k]:
                    rows.append(dict(trajectory_id=t,readout_id=f'{t}{k}',language=l,K=k,
                        D_total=400.,D_t=400./k,N_act=1e8,adjusted_loss=-2.,unigram_nll=5.))
        frame=pd.DataFrame(rows);membership=dict(train_trajectory_ids=['train'],test_trajectory_ids=['test'],train_readout_ids=[])
        before=packet(frame,membership,'a',dict.fromkeys('abcd',200),.1)
        frame.loc[frame.trajectory_id=='test',['adjusted_loss','D_t']]=1e20
        self.assertEqual(before,packet(frame,membership,'a',dict.fromkeys('abcd',200),.1))
        self.assertEqual(before['U'][0][-1],0.)


if __name__ == '__main__': unittest.main()
