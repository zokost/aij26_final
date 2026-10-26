"""Frozen threshold guards, safe artifact loading and feature parity checks."""
import io
import os
import pickle
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from nc17_calibration import BASE, FrozenUnpickler, choose_thresholds
from nc17_features import handcrafted, map_features, selected_handcrafted


class FixedGains:
    def __init__(self, values):
        self.values = np.asarray(values)

    def predict(self, features):
        return self.values


class CalibrationTest(unittest.TestCase):
    def test_exact_one_percent_is_eligible_and_smaller_mask_is_frozen(self):
        result = choose_thresholds(FixedGains([[.2, .1, 0, 0]] * 3),
                                   np.zeros((3, 276)), np.array([9, 10, 11]), np.array([1000] * 3))
        np.testing.assert_array_equal(result, [.675, .5, .5])

    def test_nonpositive_gain_keeps_baseline(self):
        result = choose_thresholds(FixedGains([[-.1, -.2, -.3, 0], [0, 0, 0, 0], [1e-9, 0, 0, 0]]),
                                   np.zeros((3, 276)), np.array([100] * 3), np.array([1000] * 3))
        np.testing.assert_array_equal(result, [BASE] * 3)

    def test_probability_features_are_finite_and_normalized(self):
        for value in (0., .75, 1.):
            features = map_features(np.full((7, 11), value, np.float32))
            self.assertEqual(features.shape, (22,))
            self.assertTrue(np.isfinite(features).all())
            self.assertAlmostEqual(features[:8].sum(), 1.)

    def test_selected_columns_match_full_frozen_feature_arithmetic(self):
        image = np.random.default_rng(42).integers(0, 256, (47, 67, 3), dtype=np.uint8)
        probability = np.linspace(0, 1, 47 * 67, dtype=np.float32).reshape(47, 67)
        used = {22, 28, 70, 71, 76, 80, 111, 125, 175, 200, 233, 242, 243}
        full = np.concatenate(handcrafted(image, probability))
        selected = np.concatenate(selected_handcrafted(image, probability, used))
        indices = np.array(sorted(i - 22 for i in used))
        np.testing.assert_array_equal(selected[indices], full[indices])

    def test_unknown_pickle_global_is_rejected_without_invocation(self):
        raw = pickle.dumps(os.system)
        with self.assertRaises(pickle.UnpicklingError):
            FrozenUnpickler(io.BytesIO(raw)).load()


if __name__ == '__main__':
    unittest.main()
