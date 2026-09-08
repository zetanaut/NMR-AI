"""Independent quadrature and physical limits of the actual production model."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from circuit import (Circuit, cable_parameters, detector_voltage, line_input, node_voltage,
                     nominal_circuit, resonator_impedance, shunted_coil)
from learning_data import SIMULATOR
from lineshape import complex_branch, pake_susceptibility, powder_complex, spin_weights
from nmr_lab import (FREQUENCY_HZ, clean_response, group_split, integration_weights, make_features,
                     noise_factor, sample_configurations, simulate, te_calibration, thermal_polarization)
from analyze_predictions import summarize


class Physics(unittest.TestCase):
    def test_published_snapshot_uses_current_generator(self):
        text = (Path(__file__).resolve().parents[1]/"docs/assets/results.js").read_text()
        snapshot = json.loads(text.split("window.TUTORIAL_RESULTS = ", 1)[1].strip().removesuffix(";"))
        self.assertEqual(snapshot["simulator"], SIMULATOR)
        self.assertEqual(snapshot["residual_definition"], "P_true - P_pred")
        s = snapshot["spectrum"]
        event = simulate(s["p"], s["configuration"], np.random.default_rng(s["seed"]))
        for key in ("signal", "baseline", "lineshape", "noise"):
            np.testing.assert_allclose(s[key], event[key], rtol=5e-9, atol=1e-16)
        for run in snapshot["runs"]:
            self.assertEqual(run["config"]["simulator"], SIMULATOR)
            self.assertEqual(run["dataset_settings"]["simulator"], SIMULATOR)

    def test_complex_kernel_against_independent_orientation_quadrature(self):
        nodes, weights = np.polynomial.legendre.leggauss(512)
        x = np.array([-4, -2, -1, 0, .95, 1, 1.05, 2, 4])
        for eta in (0, .03, .3):
            for eps in (-1, 1):
                phi, g = .4, .08
                y = np.sqrt(3-eta*np.cos(2*phi))
                q = (nodes+1)*y/2
                resonance = eps*(1-eta*np.cos(2*phi)-q*q)
                expected = (1/(resonance[None,:]-x[:,None]+1j*g)) @ weights/(2*np.pi)
                np.testing.assert_allclose(complex_branch(x, eps, eta, phi, g), expected, rtol=1e-10)

    def test_physical_horns_reflection_and_passive_absorption(self):
        x = np.linspace(-3, 3, 6001)
        plus = powder_complex(x, 1, eta=0, g=.002)
        self.assertLess(abs(x[np.argmax(-plus.imag)]-1), .01)
        self.assertTrue(np.all(plus.imag < 0))
        np.testing.assert_allclose(powder_complex(-x, -1, eta=0, g=.002), -plus.conjugate(), rtol=1e-10)
        self.assertGreater(-plus.imag[np.argmin(abs(x))], 20*(-plus.imag[np.argmin(abs(x-1.2))]))

    def test_continuous_normalization_and_no_window_renormalization(self):
        from scipy.integrate import quad
        for eps in (-1, 1):
            area, _ = quad(lambda x: -complex_branch(x, eps, .03, .4, .08).imag,
                           -np.inf, np.inf, epsabs=1e-9)
            self.assertAlmostEqual(area, 1, places=8)
        full = np.linspace(-5, 5, 1001)
        total = pake_susceptibility(full, .05)[0]
        np.testing.assert_allclose(total[200:800], pake_susceptibility(full[200:800], .05)[0], rtol=1e-14)
        for p in (-1, -1e-10, 0, .05, 1):
            a, b = spin_weights(p)
            self.assertAlmostEqual(a+b, p)
            self.assertTrue(np.isfinite(pake_susceptibility(full, p)[0]).all())

    def test_cable_limits_and_passivity(self):
        c = replace(Circuit(), cable_resistance_ohm_per_m=0, cable_conductance_s_per_m=0)
        f, load = 32.7e6, 12+9j
        z0, gamma = cable_parameters(f, c)
        for angle, expected in ((0, load), (np.pi, load), (2*np.pi, load), (np.pi/2, z0*z0/load)):
            actual = line_input(f, replace(c, cable_length_m=angle/gamma.imag), load)
            self.assertLess(abs(actual-expected), 1e-10)
        c = nominal_circuit()
        self.assertAlmostEqual(resonator_impedance(c.reference_hz, c).imag, 0, places=12)
        self.assertTrue(np.all(resonator_impedance(FREQUENCY_HZ, c).real > 0))
        self.assertTrue(np.all(cable_parameters(FREQUENCY_HZ, c)[1].real >= 0))

    def test_coil_stray_zero_loss_and_fill_identifiability(self):
        c, f = Circuit(), 32.7e6
        omega = 2*np.pi*f
        z = c.coil_resistance_ohm+1j*omega*c.coil_inductance_h
        self.assertLess(abs(shunted_coil(f, c)-z), 1e-14)
        changed = shunted_coil(f, c, -.001j)
        self.assertGreater(changed.real, z.real)
        np.testing.assert_array_equal(detector_voltage(FREQUENCY_HZ, c),
                                      detector_voltage(FREQUENCY_HZ, replace(c, filling_factor=.8)))
        with self.assertRaises(ValueError):
            replace(c, stray_capacitance_f=-1)

    def test_exact_loaded_voltage_and_detector_phase(self):
        z = np.array([12+9j, 50-30j, 1+100j, 20+0j])
        u, r0, ri = .5, 500, 50
        y = 1/r0+1/ri
        eq2 = u/r0*(z.real+y*abs(z)**2)/((1+y*z.real)**2+(y*z.imag)**2)
        np.testing.assert_allclose(eq2, node_voltage(z, u, r0, ri).real, rtol=1e-14)
        c = nominal_circuit()
        voltage = node_voltage(resonator_impedance(FREQUENCY_HZ, c), c.drive_v,
                               c.drive_resistance_ohm, c.input_resistance_ohm)
        np.testing.assert_allclose(detector_voltage(FREQUENCY_HZ, replace(c, detector_phase_rad=np.pi/2)),
                                   -c.detector_gain*voltage.imag, atol=1e-17)

    def test_zero_p_and_te_area_calibration(self):
        config = sample_configurations(1)[0]
        e = simulate(0, config, np.random.default_rng(42))
        np.testing.assert_array_equal(e["lineshape"], np.zeros(500))
        np.testing.assert_allclose(e["signal"], e["baseline"]+e["noise"], atol=1e-17)
        p = thermal_polarization(config["center_mhz"]*1e6)
        signal, baseline = clean_response(p, config)
        features = make_features((signal+baseline)[None], np.array([te_calibration(config)]), baseline[None])
        self.assertAlmostEqual(float(features[0, 1].sum()), p, places=9)
        self.assertAlmostEqual(100*thermal_polarization(32.7e6), .069748986608294, places=11)

    def test_covariance_and_group_partition(self):
        covariance = np.diag(np.linspace(1, 2, 500))*1e-18
        factor = noise_factor(covariance)
        np.testing.assert_allclose(factor@factor.T, covariance, atol=1e-30)
        covariance[0,0] = -1e-18
        with self.assertRaises(ValueError):
            noise_factor(covariance)
        groups = np.repeat(np.arange(30), 10)
        parts = group_split(groups, 42)
        sets = [set(groups[p]) for p in parts]
        self.assertFalse(sets[0]&sets[1] or sets[1]&sets[2] or sets[0]&sets[2])

    def test_error_sign_and_units(self):
        m = summarize(np.zeros(3), np.array([.01, .02, .03]), .05)
        self.assertAlmostEqual(m["relative_bias_percent_at_p0"], -40)
        self.assertAlmostEqual(m["rmse"]**2, m["bias"]**2+m["width"]**2)

    def test_baseline_fit_recovers_detector_trace(self):
        from dataclasses import asdict
        from fit_tuned_baseline import CIRCUIT_NAMES, READOUT_NAMES, fit_with_setup, reconstruct_fit
        c = replace(Circuit(), reference_hz=32.68e6, tune_capacitance_f=100e-12,
                    cable_length_m=4, stray_capacitance_f=50e-12,
                    detector_gain=30, detector_phase_rad=.7, dc_offset_v=-.02)
        f = np.linspace(32.28, 33.08, 500)*1e6  # synthetic, not acquisition metadata
        truth = detector_voltage(f, c)
        parameters = {name: {"value": asdict(c)[name], "source": "Synthetic truth"} for name in CIRCUIT_NAMES}
        parameters.update({name: {"profile": True, "source": "Synthetic unknown readout"} for name in READOUT_NAMES})
        prediction, report = fit_with_setup(f, truth, {"parameters": parameters}, starts=1)
        np.testing.assert_allclose(prediction, truth, atol=1e-11)
        self.assertLess(report["rmse_over_trace_range"], 1e-9)
        np.testing.assert_allclose(reconstruct_fit(f, report), truth, atol=1e-11)


if __name__ == "__main__":
    unittest.main()
