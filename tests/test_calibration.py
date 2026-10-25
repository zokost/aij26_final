"""Check the official 1% boundary and lower-only calibration edge cases."""
import unittest
import numpy as np
from stage2.calibration import choose_thresholds, map_features


class FixedGains:
    def __init__(self, values):
        self.values = np.asarray(values)

    def predict(self, features):
        return self.values


class CalibrationChecks(unittest.TestCase):
    def test_guard_includes_exact_one_percent_as_alarm(self):
        model = FixedGains([[.2,.1,0,0]]*3)
        thresholds = choose_thresholds(model, np.zeros((3,276)), np.array([9,10,11]), np.array([1000]*3))
        np.testing.assert_array_equal(thresholds, [.75,.5,.5])

    def test_no_positive_gain_preserves_baseline(self):
        model = FixedGains([[-.1,-.2,-.3,0], [0,0,0,0], [1e-9,0,0,0]])
        thresholds = choose_thresholds(model, np.zeros((3,276)), np.array([100]*3), np.array([1000]*3))
        np.testing.assert_array_equal(thresholds, [.75]*3)

    def test_native_probability_features_handle_constant_maps(self):
        for value in [0.,.75,1.]:
            features = map_features(np.full((7,11), value, np.float32))
            self.assertEqual(features.shape, (22,))
            self.assertTrue(np.isfinite(features).all())
            self.assertAlmostEqual(features[:8].sum(), 1.)


if __name__ == '__main__':
    unittest.main()
