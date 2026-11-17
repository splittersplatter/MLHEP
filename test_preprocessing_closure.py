"""Explicit synthetic tests for transform protocols and uncertainty propagation."""
import copy
import unittest
import numpy as np
from preprocessing import VARIANTS,fit,apply
from study import synthetic_data
from local_closure import calibrated_closure,background_controls
from tune_disco import select_strength


class PreprocessingClosureChecks(unittest.TestCase):
    def test_transform_fit_is_train_only_and_incompatible_rejected(self):
        data=synthetic_data()
        training=np.arange(600)
        for transform in VARIANTS:
            x,p,diagnostic=fit(data,training,True,transform)
            other={k:v.copy() for k,v in data.items()}
            other['x'][600:]*=1e7
            other['jet_mass'][600:]*=100
            y,q,_=fit(other,training,True,transform)
            for key in ('mean','scale','active'):
                self.assertTrue(np.array_equal(p[key],q[key]))
            self.assertTrue(np.array_equal(x,apply(data,p,True,transform)))
            self.assertEqual(diagnostic['transform']['name'],transform)
            self.assertTrue(all(r['row'] < 600 for r in diagnostic['floor_events']))
            self.assertTrue(np.isfinite(x).all())
            other_transform=next(t for t in VARIANTS if t != transform)
            with self.assertRaises(ValueError):
                apply(data,p,True,other_transform)

    def test_calibration_uncertainty_and_background_controls(self):
        report=background_controls(replicates=99)
        controls={r['case']:r for r in report['cases']}
        for name in ('flat','smooth_slope'):
            self.assertEqual(controls[name]['status'],'defined')
            self.assertTrue(controls[name]['diagnostic_interval_contains_zero'])
        step=controls['localized_sr_step']
        self.assertFalse(step['diagnostic_interval_contains_zero'])
        self.assertGreater(step['residual_interval_95'][0],0)
        self.assertGreater(step['threshold_interval_95'][1],step['threshold_interval_95'][0])
        self.assertEqual(controls['smooth_slope']['point']['coefficients'],step['point']['coefficients'])
        self.assertIn('SYNTHETIC',report['analysis'])

    def test_insufficient_support_never_gets_interval(self):
        mass=np.array([3200.,3400.,3800.])
        truth=np.zeros(3,dtype=int)
        score=np.array([.1,.2,.3])
        calibration=dict(mass=mass,truth=truth,score=score)
        result=calibrated_closure(calibration,mass,truth,score,.05,9)
        self.assertEqual(result['status'],'unavailable')
        self.assertEqual(result['replicates_attempted'],0)
        self.assertIsNone(result['residual_interval_95'])
        with self.assertRaises(ValueError):
            calibrated_closure(calibration,mass,truth,score+1,.05,9,same_sample=True)

    def test_local_step_is_visible_despite_improved_global_js(self):
        from local_closure import closure
        from scipy.spatial.distance import jensenshannon
        centers=np.arange(3125,3900,50)
        n=20000
        mass=np.repeat(centers,n)
        baseline=.05+(centers-3500)*.00012
        stepped=.05+.02*((centers >= 3300)&(centers < 3700))
        def selected(rates):
            return np.concatenate([np.arange(n) < round(rate*n) for rate in rates])
        first,second=selected(baseline),selected(stepped)
        edges=np.arange(3100,3901,50)
        reference=np.histogram(mass,edges)[0]
        js_first=jensenshannon(reference,np.histogram(mass[first],edges)[0],base=2)**2
        js_second=jensenshannon(reference,np.histogram(mass[second],edges)[0],base=2)**2
        self.assertLess(js_second,js_first)
        self.assertFalse(closure(mass,second)['diagnostic_within_3se'])

    def test_zero_baseline_js_never_qualifies(self):
        rows=[dict(seed=s,fraction=f,**{'lambda':lam},sic=1.,js_divergence=0.)
              for s in [11,22,33] for f in [.005,.01] for lam in [0.,.003]]
        result=select_strength(rows)
        self.assertFalse(result['targets_met'])
        self.assertIsNone(result['candidates'][0]['regimes'][0]['js_ratio'])

    def test_empty_qualification_grid_rejected(self):
        with self.assertRaises(ValueError):
            select_strength([],seeds=[],fractions=[],strengths=[])


if __name__ == '__main__':
    unittest.main()
