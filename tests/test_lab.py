"""Scientific and evaluation invariants for the independent tutorial labs."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from analyze_predictions import summarize
from lineshape import SIMULATOR, branch, pake_doublet, powder_branch
from nmr_lab import group_split, make_features, sample_configurations, simulate, validate_configurations


class LabInvariants(unittest.TestCase):
    def test_published_spectrum_is_generated_by_current_physics(self):
        source = Path(__file__).resolve().parents[1] / "docs/assets/results.js"
        snapshot = json.loads(source.read_text().split("window.TUTORIAL_RESULTS = ", 1)[1].strip().removesuffix(";"))
        self.assertEqual(snapshot["simulator"], SIMULATOR)
        self.assertEqual(snapshot["validation_status"], "unvalidated-polynomial-baseline-prototype")
        spectrum = snapshot["spectrum"]
        self.assertEqual(spectrum["simulator"], SIMULATOR)
        event = simulate(spectrum["p"], spectrum["configuration"], np.random.default_rng(spectrum["seed"]),
                         noise_level=spectrum["noise_level"])
        for name in ("signal", "baseline", "lineshape", "noise"):
            np.testing.assert_allclose(spectrum[name], event[name], rtol=5e-9, atol=1e-13)
        for run in snapshot["runs"]:
            self.assertEqual(run["config"]["simulator"], SIMULATOR)
            self.assertEqual(run["dataset_settings"]["simulator"], SIMULATOR)

    def test_dulya_branch_matches_independent_convolution_quadrature(self):
        # Integrate the powder-orientation Lorentzian directly, independently of
        # the analytic arctan/log implementation. Include horns, shoulders, wings.
        nodes, weights = np.polynomial.legendre.leggauss(512)
        x = np.array([-4, -2, -1, 0, 0.95, 1, 1.05, 2, 4])
        for eta in (0, 0.03, 0.3):
            for eps in (-1, 1):
                phi, g = 0.4, 0.08
                ymax = np.sqrt(3 - eta * np.cos(2 * phi))
                y = (nodes + 1) * ymax / 2
                b = 1 - eps * x - eta * np.cos(2 * phi)
                integral = ((2*g/np.pi) / ((y[None, :]**2 - b[:, None])**2 + g*g)) @ weights
                expected = integral * ymax / 20  # quadrature interval and 1/10 convention
                np.testing.assert_allclose(branch(x, eps, eta, phi, g), expected, rtol=1e-10, atol=1e-13)

    def test_powder_average_and_axial_limit(self):
        x = np.linspace(-4, 4, 101)
        np.testing.assert_allclose(powder_branch(x, 1, eta=0), branch(x, 1, 0, 0, 0.08), rtol=1e-14)
        phi = np.linspace(0, np.pi/2, 33)
        expected = np.mean([branch(x, 1, 0.3, f, 0.08) * np.sqrt(3/(3-0.3*np.cos(2*f)))
                            for f in phi], axis=0)
        np.testing.assert_allclose(powder_branch(x, 1, eta=0.3), expected, rtol=1e-14)

    def test_pake_horn_shoulder_and_center_are_not_isolated_gaussian_peaks(self):
        x = np.linspace(-3, 3, 6001)
        plus = powder_branch(x, 1, eta=0, g=0.002)
        self.assertLess(abs(x[np.argmax(plus)] - 1), 0.01)
        value = lambda at: float(plus[np.argmin(abs(x-at))])
        self.assertGreater(value(-1.8), 20*value(-2.2))  # shoulder near -2
        self.assertGreater(value(0), 20*value(1.2))  # extended powder support
        np.testing.assert_allclose(powder_branch(x, -1, eta=0, g=0.002), plus[::-1], rtol=1e-11)

    def test_signed_area_components_and_spin_temperature_weights(self):
        x = np.linspace(-5, 5, 512)
        for p in (-1, -0.5, -1e-8, 0, 1e-8, 0.05, 0.5, 1):
            total, plus, minus = pake_doublet(x, p, -1.39)
            self.assertTrue(np.isfinite(total).all())
            np.testing.assert_allclose(total, plus + minus, rtol=1e-14)
            self.assertAlmostEqual(total.sum(), p / -1.39, places=14)
            reflected = pake_doublet(-x, -p, -1.39)[0]
            np.testing.assert_allclose(total, -reflected, rtol=1e-13, atol=1e-18)
            if 0 < abs(p) < 1:
                r = (np.sqrt(4 - 3*p*p) + p) / (2 - 2*p)
                self.assertAlmostEqual(plus.sum() / minus.sum(), r, places=12)
        tiny = pake_doublet(x, 1e-10, -1.39)[0]
        self.assertLess(np.max(np.abs(tiny)), 1e-11)  # no artificial low-P floor

    def test_pake_configuration_validation(self):
        configs = sample_configurations(3)
        validate_configurations(configs)
        configs[0]["eta"] = -0.01
        with self.assertRaises(ValueError):
            validate_configurations(configs)
        for p, cc, eta, g in ((1.1, 1, 0.03, 0.08), (0.05, 0, 0.03, 0.08),
                              (0.05, 1, 1.1, 0.08), (0.05, 1, 0.03, 0)):
            with self.assertRaises(ValueError):
                pake_doublet(np.linspace(-5, 5, 512), p, cc, eta, g)

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
