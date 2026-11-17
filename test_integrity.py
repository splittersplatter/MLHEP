"""Protocol-v2 regression fixtures. All generated events are explicitly synthetic."""
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from artifacts import writer_lock, complete, verify
from provenance import atomic_json, read_json, file_hash, identity
from splits import allocate, indices, validate_manifest
from metrics import metrics_for, calibrate, disco_squared
from study import synthetic_data, prepare, train_mlp, score_model
from tune_disco import select_strength
from exposure import load_ledger, eligible, record_access
from local_closure import closure
from bootstrap import paired


class IntegrityChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.data = synthetic_data()
        self.manifest = allocate(self.data, [11, 22, 33], 'synthetic', smoke=True)

    def test_five_roles_fixed_across_seeds_and_explicit_smoke(self):
        for seed in [11, 22, 33]:
            roles = indices(self.data, self.manifest, seed)
            union = set()
            for values in roles.values():
                self.assertFalse(union & set(values))
                union.update(values)
            self.assertEqual(len(union), len(self.data['row']))
        for seed,roles in self.manifest['seeds'].items():
            for role, members in roles.items():
                self.assertEqual(sum(self.manifest['role_stratum_counts'][seed][role].values()),len(members))
        for name in ['calibration', 'tuning', 'final']:
            self.assertEqual(self.manifest['seeds']['11'][name], self.manifest['seeds']['22'][name])
        tiny = {k: v[:100] for k, v in self.data.items()}
        with self.assertRaises(ValueError):
            allocate(tiny, [11], 'synthetic')
        bad = copy.deepcopy(self.manifest)
        bad['seeds']['11']['training'].append(bad['seeds']['11']['tuning'][0])
        with self.assertRaises(ValueError):
            validate_manifest(bad)

    def test_undefined_metrics_strict_json_and_independent_valid_arithmetic(self):
        ids = np.arange(len(self.data['mass']))
        scores = np.linspace(0, 1, len(ids))
        none = metrics_for(self.data, ids, scores, 1., 4, 9)
        self.assertIsNone(none['js_divergence'])
        self.assertIsNone(none['js_permutation_p'])
        self.assertIsNone(none['sic'])
        json.dumps(none, allow_nan=False)
        all_selected = metrics_for(self.data, ids, scores, -1., 4, 9)
        self.assertEqual(all_selected['js_divergence'], 0.)
        self.assertEqual(all_selected['js_permutation_p'], 1.)
        tied = metrics_for(self.data, ids, scores*0, 0., 4, 9)
        self.assertEqual(tied['correlation_status'], 'constant_variable')
        self.assertEqual(tied['score_mass_dcor_squared'], 0.)
        result = metrics_for(self.data, ids, scores, .5, 4, 9)
        bg = self.data['truth'] == 0
        sr = (self.data['mass'] >= 3300) & (self.data['mass'] < 3700)
        selected = scores > .5
        es = selected[(~bg) & sr].mean()
        eb = selected[bg & sr].mean()
        # Independent implementation: never delegates to the production helper.
        edges = np.arange(3100, 3901, 50)
        js = jensenshannon(np.histogram(self.data['mass'][bg], edges)[0],
                          np.histogram(self.data['mass'][bg & selected], edges)[0], base=2)**2
        self.assertAlmostEqual(result['sic'], es/np.sqrt(eb), places=12)
        self.assertAlmostEqual(result['js_divergence'], js, places=12)
        missing = dict(mass=self.data['mass'][bg], truth=self.data['truth'][bg])
        empty_class = metrics_for(missing, np.arange(bg.sum()), scores[bg], .5, 4, 9)
        self.assertIsNone(empty_class['signal_auc_sr'])
        self.assertIsNone(empty_class['signal_efficiency'])
        sparse = metrics_for(dict(mass=np.array([3200.]), truth=np.array([0])), [0], np.array([.5]), .1, 4, 9)
        self.assertEqual(sparse['correlation_status'], 'insufficient_samples')
        self.assertIsNone(sparse['local']['regions']['sr']['efficiency'])
        json.dumps(sparse, allow_nan=False)
        with self.assertRaises(ValueError):
            metrics_for(self.data, ids, scores[:-1], .5, 1, 9)

    def test_only_calibration_affects_cuts(self):
        roles = indices(self.data, self.manifest, 11)
        mass = self.data['mass'][roles['calibration']]
        scores = np.linspace(.1, .9, len(mass))
        threshold = calibrate(mass, scores, .05)
        unrelated = np.ones(len(roles['tuning']))*999
        self.assertEqual(threshold, calibrate(mass, scores, .05))
        self.assertNotEqual(threshold, calibrate(mass, scores*.5, .05))
        tied = calibrate(mass, scores*0, .05)
        self.assertEqual(np.count_nonzero(scores*0 > tied), 0)

    def test_artifact_completion_lock_and_content_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'weights').write_bytes(b'weights')
            with self.assertRaises(FileNotFoundError):
                verify(root, 'run')
            complete(root, 'run', {})
            verify(root, 'run')
            with self.assertRaises(ValueError):
                verify(root, 'different_run')
            (root/'weights').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify(root, 'run')
        with tempfile.TemporaryDirectory() as directory:
            with writer_lock(directory):
                with self.assertRaises(ValueError):
                    with writer_lock(directory):
                        pass
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'strict.json'
            with self.assertRaises(ValueError):
                atomic_json(path, {'bad': float('nan')})
            path.write_text('{"x":NaN}')
            with self.assertRaises(ValueError):
                read_json(path)
            path.write_text('{"x":1,"x":2}')
            with self.assertRaises(ValueError):
                read_json(path)

    def test_exact_qualification_grid_and_zero_denominators(self):
        rows = [dict(seed=s, fraction=f, **{'lambda':lam}, sic=1., js_divergence=.01 if lam else .1)
                for s in [11,22,33] for f in [.005,.01] for lam in [0.,.003]]
        self.assertTrue(select_strength(rows)['targets_met'])
        self.assertFalse(select_strength(rows[:-1])['targets_met'])
        with self.assertRaises(ValueError):
            select_strength(rows + [rows[0]])
        rows[0]['sic'] = 0.
        rows[2]['sic'] = 0.
        rows[4]['sic'] = 0.
        rows[6]['sic'] = 0.
        rows[8]['sic'] = 0.
        rows[10]['sic'] = 0.
        self.assertFalse(select_strength(rows)['targets_met'])
        rows[0]['sic'] = None
        json.dumps(select_strength(rows), allow_nan=False)

    def test_telemetry_does_not_change_model_or_rng(self):
        roles = indices(self.data, self.manifest, 11)
        train = roles['training'][self.data['truth'][roles['training']] == 0]
        valid = roles['checkpoint'][self.data['truth'][roles['checkpoint']] == 0]
        x, preprocessing, tails = prepare(self.data, train, True)
        args = SimpleNamespace(epochs=2, telemetry=False, device='cpu')
        a, history_a, _ = train_mlp(x, self.data, train, valid, 11, 'sideband', .003, args)
        state_a = torch.get_rng_state()
        args.telemetry = True
        b, history_b, diagnostics = train_mlp(x, self.data, train, valid, 11, 'sideband', .003, args)
        self.assertTrue(torch.equal(state_a, torch.get_rng_state()))
        self.assertEqual(history_a, history_b)
        self.assertTrue(np.array_equal(score_model(a, x), score_model(b, x)))
        self.assertGreater(diagnostics[0]['bce_gradient_norm'], 0)
        self.assertGreater(diagnostics[0]['weighted_penalty_gradient_norm'], 0)
        self.assertIn(7, tails['constant_columns'])
        json.dumps(diagnostics, allow_nan=False)
        # Tiny score spread exercises the denominator floor with finite gradients.
        score = torch.arange(20, dtype=torch.float64)*1e-14
        score.requires_grad_()
        disco_squared(score, torch.arange(20, dtype=torch.float64)).backward()
        self.assertTrue(torch.isfinite(score.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable in this execution environment')
    def test_cuda_training_cpu_reload_and_telemetry(self):
        roles = indices(self.data, self.manifest, 11)
        train = roles['training'][self.data['truth'][roles['training']] == 0]
        valid = roles['checkpoint'][self.data['truth'][roles['checkpoint']] == 0]
        x, _, _ = prepare(self.data, train, True)
        torch.use_deterministic_algorithms(True)
        args = SimpleNamespace(epochs=2, telemetry=False, device='cuda')
        a, first_history, _ = train_mlp(x, self.data, train, valid, 11, 'sideband', .003, args)
        args.telemetry = True
        b, second_history, telemetry = train_mlp(x, self.data, train, valid, 11, 'sideband', .003, args)
        self.assertEqual(first_history, second_history)
        self.assertTrue(np.array_equal(score_model(a, x), score_model(b, x)))
        expected = score_model(b, x)
        b.cpu()
        self.assertTrue(np.allclose(score_model(b, x), expected, atol=2e-6, rtol=0))
        self.assertTrue(telemetry)

    def test_unknown_and_reused_exposure_never_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'ledger.json'
            ledger = load_ledger(path, 'dataset')
            with self.assertRaises(ValueError):
                eligible(ledger, [200000, 300000])
            ledger.update(history='source_audited', unused_evidence=dict(reference='SYNTHETIC source audit', rows=[1,2]))
            atomic_json(path, ledger)
            eligible(load_ledger(path, 'dataset'), [1,2])
            record_access(path, 'dataset', [1,2], 'attempt_interrupted', 'run', require_unused=True)
            with self.assertRaises(ValueError):
                record_access(path, 'dataset', [1,2], 'simultaneous_second_attempt', 'other_run', require_unused=True)
            with self.assertRaises(ValueError):
                eligible(load_ledger(path, 'dataset'), [1])
            with self.assertRaises(ValueError):
                load_ledger(path, 'changed_dataset')

    def test_concurrent_unused_reservation_has_only_one_winner(self):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'dataset.json'
            ledger = load_ledger(path, 'dataset')
            ledger.update(history='source_audited',unused_evidence=dict(reference='SYNTHETIC audit',rows=[1,2]))
            atomic_json(path,ledger)
            barrier = Barrier(2)
            def reserve(name):
                barrier.wait()
                try:
                    record_access(path,'dataset',[1,2],'independent_evaluation_attempt',name,require_unused=True)
                    return True
                except ValueError:
                    return False
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(reserve,['different_output_one','different_output_two']))
            self.assertEqual(sum(results),1)
            self.assertEqual(len(load_ledger(path,'dataset')['records']),1)

    def test_closure_fits_only_sidebands_and_exposes_step(self):
        centers = np.arange(3125, 3900, 50)
        n = 20000
        mass = np.repeat(centers, n)
        slope = .05 + (centers-3500)*.00003
        def decisions(rates):
            return np.concatenate([np.arange(n) < round(rate*n) for rate in rates])
        smooth = closure(mass, decisions(slope))
        stepped = closure(mass, decisions(slope + ((centers >= 3300)&(centers < 3700))*.05))
        self.assertTrue(smooth['diagnostic_within_3se'])
        self.assertFalse(stepped['diagnostic_within_3se'])
        self.assertEqual(smooth['coefficients'], stepped['coefficients'])
        self.assertGreater(stepped['residual'], .049)

    def test_bootstrap_does_not_drop_invalid_replicates(self):
        mass = np.array([3200., 3400., 3500.])
        truth = np.array([0,0,1])
        decisions = {(11, model): np.array([False,False,True]) for model in ['mass_mlp','sideband_0.003']}
        result = paired(mass, truth, decisions, [11], replicates=10)
        self.assertEqual(result['invalid_sic_replicates'], 10)
        self.assertIsNone(result['sic_interval'])
        self.assertFalse(result['point_targets_met'])
        json.dumps(result, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
