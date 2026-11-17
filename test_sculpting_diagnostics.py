"""Synthetic software checks only; these fixtures are not physics results."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
from sculpting_diagnostics import diagnose, js_reference, wilson, checkpoint_diagnostic


class SculptingDiagnosticsTests(unittest.TestCase):
    def test_measured_sidebands_and_strict_cut(self):
        r = diagnose(np.array([3200., 3200., 3400., 3400., 3800., 3800.]),
                     np.zeros(6), np.array([1., 0., 1., 1., 1., 0.]), .5, 19)
        self.assertEqual(r['sr_to_measured_sb_ratio'], 2.)
        self.assertEqual(r['regions']['sb']['selected'], 2)
        tied = diagnose(np.array([3200., 3400., 3800.]), np.zeros(3), np.ones(3), 1., 19)
        self.assertIsNone(tied['js']['p_upper'])
        self.assertIsNone(tied['sr_to_measured_sb_ratio'])

    def test_null_all_selected_and_empty(self):
        counts = np.array([10, 20, 30])
        r = js_reference(counts, counts, 39, 1)
        self.assertEqual(r['observed'], 0.)
        self.assertEqual(r['null_mean'], 0.)
        self.assertEqual(r['p_upper'], 1.)
        self.assertIsNone(js_reference(counts, counts*0, 39, 1)['p_upper'])

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            js_reference(np.array([2, 3]), np.array([3, 2]), 19, 1)
        with self.assertRaises(ValueError):
            diagnose(np.array([3200.]), np.array([0]), np.array([np.nan]), .5)
        with self.assertRaises(ValueError):
            diagnose(np.array([3900.]), np.array([0]), np.array([.5]), .5)

    def test_wilson_boundaries(self):
        self.assertIsNone(wilson(0, 0))
        self.assertAlmostEqual(wilson(0, 10)[0], 0.)
        self.assertAlmostEqual(wilson(10, 10)[1], 1.)

    def test_penalty_can_change_checkpoint(self):
        # Synthetic log: BCE prefers epoch 2; objective prefers epoch 1.
        history = np.array([[1, .7, 0, .69, 0, .69],
                            [2, .7, 0, .689, 1, .692]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'training.csv'
            np.savetxt(path, history, delimiter=',')
            result = checkpoint_diagnostic(path, .003)
            self.assertEqual(result['recorded_objective_epoch'], 1)
            self.assertEqual(result['bce_only_epoch_on_observed_trajectory'], 2)
            with self.assertRaises(ValueError):
                checkpoint_diagnostic(path, .01)


if __name__ == '__main__':
    unittest.main()
