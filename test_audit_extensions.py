"""Audit regressions: declared floors, input preservation and independent arithmetic."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import h5py
import numpy as np
from provenance import file_hash
from feature_tail_audit import audit
from inventory_history import inventory
from metrics import metrics_for
from audit_predictions import audit_local


class AuditExtensionChecks(unittest.TestCase):
    def test_raw_audit_uses_declared_floor_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            run=root/'run'
            fit=run/'fits/model'
            fit.mkdir(parents=True)
            np.savez(fit/'predictions.npz',train_rows=np.array([0]))
            (run/'protocol.json').write_text('{}')
            dataset=root/'raw.h5'
            raw=np.zeros((1,2101));raw[0,0]=100;raw[0,3]=100
            with h5py.File(dataset,'w') as handle:
                handle.create_dataset('df/block0_values',data=raw)
            cache=root/'cache.npz'
            np.savez(cache,x=np.array([1.]))
            data=dict(row=np.array([0]),x=np.array([[5e-11,1.]]),truth=np.array([0]),mass=np.array([3500.]),
                constituents=np.array([[1,1]]),jet_pt=np.array([[100.,100.]]),jet_eta=np.zeros((1,2)),jet_rapidity=np.zeros((1,2)))
            protocol=dict(run_id='fixture',cache=dict(path=str(cache)),identity=dict(config=dict(smoke=False,transform='log_floor_1e-10'),
                dataset_sha256=file_hash(dataset),expected_fits=['model']))
            with patch('feature_tail_audit.verify_run',return_value=protocol),patch('feature_tail_audit.verified_data',return_value=data):
                result=audit(run,dataset,root/'high.json')
                self.assertEqual(result['training_floor_event_count'],1)
                self.assertEqual(result['records'][0]['jets'][0]['efp_at_floor'],1)
                self.assertEqual(result['transform']['floor'],1e-10)
                protocol['identity']['config']['transform']='log_floor_1e-12'
                self.assertEqual(audit(run,dataset,root/'low.json')['training_floor_event_count'],0)
                before={p:file_hash(p) for p in [dataset,cache,root/'high.json']}
                for path in before:
                    with self.assertRaises(ValueError):
                        audit(run,dataset,path)
                    self.assertEqual(file_hash(path),before[path])
                with self.assertRaises(ValueError):
                    inventory(dataset,dataset,roots=[run])
                with self.assertRaises(ValueError):
                    inventory(dataset,run/'new.json',roots=[run])

    def test_independent_local_uncertainty_rejects_corruption(self):
        rng=np.random.default_rng(947)
        mass=rng.uniform(3100,3900,6000)
        truth=np.zeros(len(mass),dtype=int)
        score=rng.random(len(mass))+.02*(mass-3500)/800
        calibration=dict(mass=rng.uniform(3100,3900,6000),truth=truth.copy(),score=rng.random(6000))
        sb=(calibration['mass'] < 3300)|(calibration['mass'] >= 3700)
        threshold=float(np.quantile(calibration['score'][sb],.95))
        config=dict(permutations=19,closure_replicates=19)
        row=dict(seed=11,cut=.05,threshold=threshold,**metrics_for(dict(mass=mass,truth=truth),np.arange(len(mass)),score,
            threshold,311,19,calibration=calibration,cut=.05,closure_replicates=19))
        audit_local(row,mass,truth,score,calibration,config)
        for path in [('local','regions','sr','conditional_wilson_95'),('local','js','null_95_interval'),
                     ('closure','coefficients'),('closure_uncertainty','threshold_interval_95')]:
            changed=copy.deepcopy(row)
            target=changed
            for key in path[:-1]:
                target=target[key]
            target[path[-1]][0]+=.001
            with self.assertRaises(ValueError):
                audit_local(changed,mass,truth,score,calibration,config)


if __name__ == '__main__':
    unittest.main()
