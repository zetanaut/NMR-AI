"""Scientific and evaluation invariants for the independent tutorial labs."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_predictions import summarize
from nmr_lab import group_split, make_features, sample_configurations, simulate


class LabInvariants(unittest.TestCase):
    def test_groups_never_cross_partitions(self):
        groups = np.repeat(np.arange(30), np.arange(1, 31))
        partitions = group_split(groups, 42)
        sets = [set(groups[indices]) for indices in partitions]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        np.testing.assert_array_equal(np.sort(np.concatenate(partitions)), np.arange(len(groups)))

    def test_wing_subtraction_removes_cubic_and_uses_known_cc(self):
        axis = np.linspace(-1, 1, 512)
        baseline = 0.2 - 0.1 * axis + 0.03 * axis**2 + 0.04 * axis**3
        signal = np.where(np.abs(axis) < 0.5, 0.01 * (1 - 4 * axis**2), 0)
        features = make_features(np.array([baseline + signal]), np.array([-1.39]))
        np.testing.assert_allclose(features[0, 1], signal * -1.39, atol=1e-8)
        with self.assertRaises(ValueError):
            make_features(np.array([baseline]), np.array([0]))

    def test_zero_p_has_no_clean_signal(self):
        event = simulate(0, sample_configurations(1)[0], np.random.default_rng(42), noise_level=0)
        np.testing.assert_array_equal(event["lineshape"], np.zeros(512))
        np.testing.assert_array_equal(event["signal"], event["baseline"])

    def test_bias_width_and_relative_units(self):
        metrics = summarize(np.zeros(3), np.array([0.01, 0.02, 0.03]), 0.05)
        self.assertAlmostEqual(metrics["rmse"]**2, metrics["bias"]**2 + metrics["width"]**2)
        self.assertAlmostEqual(metrics["relative_bias_percent_at_p0"], 40)
        self.assertIsNone(summarize([], [], 0.05)["rmse"])


if __name__ == "__main__":
    unittest.main()
