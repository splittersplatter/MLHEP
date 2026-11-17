"""End-to-end SYNTHETIC protocol, audit, freeze, reload, resume and corruption tests."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from study import parser, run_study, synthetic_data, fit_name
from provenance import read_json, atomic_json
from artifacts import verify, complete
from report_study import audit_run, render
from confirm_disco import freeze, evaluate, load_model


class PipelineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='mlhep_SYNTHETIC_tests_')
        cls.root = Path(cls.temporary.name)
        cls.run_dir = cls.root/'run'
        cls.args = parser().parse_args(['--smoke', '--output', str(cls.run_dir), '--epochs', '1', '--seeds', '11', '22',
                                     '--fractions', '.005', '--permutations', '9', '--closure-replicates', '9', '--device', 'cpu', '--telemetry'])
        with contextlib.redirect_stdout(io.StringIO()):
            run_study(cls.args)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def copy_run(self, name):
        import shutil
        target = self.root/name
        shutil.copytree(self.run_dir, target)
        return target

    def test_end_to_end_resume_and_reload(self):
        audit = audit_run(self.run_dir)
        self.assertEqual(audit['fits'], 10)
        before = (self.run_dir/'fits/seed11_f0.005_mass_mlp/model.pt').read_bytes()
        with contextlib.redirect_stdout(io.StringIO()):
            run_study(self.args)
        self.assertEqual(before, (self.run_dir/'fits/seed11_f0.005_mass_mlp/model.pt').read_bytes())
        data = synthetic_data()
        for name in read_json(self.run_dir/'protocol.json')['identity']['expected_fits']:
            directory = self.run_dir/'fits'/name
            marker = verify(directory)
            with np.load(directory/'predictions.npz') as p:
                ids = p['tuning_row']
                subset = {k: v[ids] for k, v in data.items()}
                actual = load_model(directory, marker['metadata'], subset)
                self.assertTrue(np.allclose(actual, p['tuning_score'], atol=2e-6, rtol=0))

    def test_freeze_replay_unknown_independence_refusal(self):
        frozen = self.root/'freeze.json'
        payload = freeze(self.run_dir, frozen)
        self.assertEqual(payload, freeze(self.run_dir, frozen))
        with self.assertRaises(ValueError):
            evaluate(self.run_dir, frozen, self.root/'independent', independent=True)
        with self.assertRaises(ValueError):
            evaluate(self.run_dir, frozen, self.root/'alternate_ledger_replay', ledger_path=self.root/'alternate.json')
        result = evaluate(self.run_dir, frozen, self.root/'replay')
        verify(result)
        from audit_predictions import audit_replay
        self.assertEqual(audit_replay(self.run_dir, result)['status'], 'passed')
        metrics = read_json(result/'metrics.json')
        self.assertEqual(len(metrics), 12)
        self.assertEqual(read_json(result/'interpretation.json')['mode'], 'synthetic_smoke')
        from report_study import render_evaluation
        report = render_evaluation(self.run_dir, result, self.root/'replay_report', diagnostic=True)
        self.assertIn('SYNTHETIC SOFTWARE CHECK', report.read_text())
        changed = read_json(frozen)
        changed['protocol']['prescribed_lambda'] = .01
        atomic_json(self.root/'bad-freeze.json', changed)
        with self.assertRaises(ValueError):
            evaluate(self.run_dir, self.root/'bad-freeze.json', self.root/'bad-replay')

    def test_report_and_no_stale_or_scientific_synthetic_output(self):
        with self.assertRaises(ValueError):
            render(self.run_dir, self.root/'scientific')
        self.assertFalse((self.root/'scientific').exists())
        report = render(self.run_dir, self.root/'report', diagnostic=True)
        content = report.read_text()
        self.assertIn('SYNTHETIC SOFTWARE CHECK', content)
        self.assertIn(read_json(self.run_dir/'protocol.json')['run_id'], content)
        self.assertIn('Every seed', content)
        corrupted = self.copy_run('corrupt_report')
        path = corrupted/'fits/seed11_f0.005_mass_mlp/metrics.json'
        rows = read_json(path)
        rows[0]['sic'] = 9999
        atomic_json(path, rows)
        with self.assertRaises(ValueError):
            render(corrupted, self.root/'corrupt_report_output', diagnostic=True)
        self.assertFalse((self.root/'corrupt_report_output').exists())

    def test_changed_source_split_model_and_config_rejected(self):
        args = copy.copy(self.args)
        args.epochs = 2
        with self.assertRaises(ValueError):
            run_study(args)
        with patch('study.source_identity', return_value={'tampered.py': 'no'}):
            with self.assertRaises(ValueError):
                run_study(self.args)
        for what in ['source', 'split', 'model']:
            run = self.copy_run('tampered_'+what)
            target = {'source': run/'source/study.py', 'split': run/'splits.json',
                      'model': run/'fits/seed11_f0.005_mass_mlp/model.pt'}[what]

            if what == 'split':
                payload = read_json(target)
                payload['seeds']['11']['training'][0] = -1
                atomic_json(target, payload)
            else:
                target.write_bytes(target.read_bytes()+b' ')
            with self.assertRaises(ValueError):
                audit_run(run)

    def test_incomplete_fit_retrained_and_simultaneous_writer_refused(self):
        run = self.copy_run('incomplete')
        (run/'fits/seed11_f0.005_mass_mlp/complete.json').unlink()
        args = copy.copy(self.args)
        args.output = run
        with contextlib.redirect_stdout(io.StringIO()):
            run_study(args)
        audit_run(run)
        (run/'.writer.lock').write_text('SYNTHETIC-other-writer')
        with self.assertRaises(ValueError):
            run_study(args)


if __name__ == '__main__':
    unittest.main()
