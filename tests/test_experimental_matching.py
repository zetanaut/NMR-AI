"""Experimental file preservation, complex readout, fits, and generation."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from baseline_data import acquisition_grid, load_acquisition
from experimental_data import load_signal_csv
from circuit import detector_voltage, cable_parameters
from lineshape import spin_weights, pake_susceptibility
from match_experimental_signals import (conditional_circuit, node_basis, match_scan, reconstruct,
                                         residual_diagnostics, NAMES, DEFAULT_MATCHING)
from generate_matched_data import (generate, covariance_factor, seed_configuration, clean_spectrum,
                                   DEFAULT_COVERAGE, GENERATOR)


class ExperimentalMatching(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(DEFAULT_MATCHING.read_text())
        self.f = acquisition_grid(load_acquisition())/1e6

    def test_public_raw_file_preserved_and_audited(self):
        path = ROOT/"examples/Sample_RawSignal.csv"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "cdbb7e3afa6531694a4b97848d295bbb5c7c03ef62d796b053e3b4e16fbaea5a")
        data, parsing = load_signal_csv(path)
        self.assertEqual(data.shape, (5, 501))
        self.assertEqual(len(parsing["unicode_line_separators"]), 5)
        self.assertEqual(len(parsing["blank_logical_lines_1based"]), 18)
        self.assertEqual(parsing["record_logical_lines_1based"], [1, 6, 11, 16, 22])
        self.assertEqual(parsing["decimal_whitespace_repairs"], [])
        self.assertEqual(parsing["duplicate_of_record_1based"], [None]*5)
        np.testing.assert_array_equal(data[:, 0], [1432662203, 1432682378, 1432683377, 1432679586, 1432681988])
        np.testing.assert_array_equal(data[:, 1], [-.1070683, -.06295195, -.186963, -.1225745, -.07522745])
        self.assertEqual(self.config["grid_status"], "confirmed")

    def test_reader_never_discards_duplicates_or_bad_tokens(self):
        row = ','.join(['1']+['0.001']*500)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"sample.csv"
            path.write_text(row+'\u2028\n\n'+row+'\n')
            data, parsing = load_signal_csv(path)
            self.assertEqual(len(data), 2)
            self.assertEqual(parsing["duplicate_of_record_1based"], [None, 1])
            for token in ('nan', 'inf', '1 2', '1\u20282', '1\u00a02'):
                path.write_text(','.join(['1', token]+['0']*499))
                with self.assertRaises(ValueError):
                    load_signal_csv(path)

    def test_complex_signal_mapping_and_capacitance_are_used(self):
        c = conditional_circuit(self.config)
        parameters = {n: self.config["fit_parameters"][n]["initial"] for n in NAMES}
        parameters.update(phase_slope_rad_per_mhz=.2, phase_curvature_rad_per_mhz2=-.1)
        coefficients = np.array([1500., -80., 4.])
        basis = node_basis(self.f, c, parameters)
        gain = np.hypot(*coefficients[:2])
        actual = replace(c, tune_capacitance_f=10**parameters["log10_tune_capacitance_pf"]*1e-12,
                         detector_gain=gain, detector_phase_rad=np.arctan2(-coefficients[1], coefficients[0]),
                         detector_phase_slope_rad_per_hz=.2e-6,
                         detector_phase_curvature_rad_per_hz2=-.1e-12, dc_offset_v=4.)
        chi = 10**parameters["log10_signal_scale_cgs"]*pake_susceptibility(
            (self.f-parameters["center_mhz"])/parameters["split_mhz"],
            parameters["P_model"], parameters["eta"], parameters["g"])[0]
        np.testing.assert_allclose(basis@coefficients, detector_voltage(self.f*1e6, actual, chi), atol=1e-11)
        parameters["P_model"] = 0.
        np.testing.assert_array_equal(node_basis(self.f, c, parameters), node_basis(self.f, c, parameters, False))
        old = node_basis(self.f, c, parameters, False)
        parameters["log10_tune_capacitance_pf"] += .1
        self.assertGreater(np.max(np.abs(node_basis(self.f, c, parameters, False)-old)), 1e-6)
        self.assertAlmostEqual(cable_parameters(32.7e6, c)[1].imag*c.cable_length_m, np.pi)

    def test_polarization_is_in_branch_ratio_without_te_calibration(self):
        for p in (-.8, -.3, -.05, .05, .3, .8):
            plus, minus = spin_weights(p)
            ratio = plus/minus
            recovered = (ratio**2-1)/(ratio**2+ratio+1)
            self.assertAlmostEqual(recovered, p, places=13)

    def test_noise_proxy_on_known_iid_gaussian_noise(self):
        f = np.arange(20000, dtype=float)
        y = np.random.default_rng(1).normal(0, .002, len(f))
        report = residual_diagnostics(y, f, np.ones(len(f), dtype=bool))
        self.assertAlmostEqual(report["wing_second_difference_sigma_proxy"]/.002, 1, delta=.03)

    def test_synthetic_bounded_signal_recovery_and_reconstruction(self):
        config = deepcopy(self.config)
        true = {n: config["fit_parameters"][n]["initial"] for n in NAMES}
        true.update(P_model=.4, log10_signal_scale_cgs=-3, phase_slope_rad_per_mhz=-.12,
                    phase_curvature_rad_per_mhz2=-.15)
        for n in NAMES:
            # These are synthetic known-neighborhood constraints, not experimental priors.
            width = {"center_mhz": .0001, "split_mhz": .0001, "g": .002, "eta": .002,
                     "P_model": .03, "log10_signal_scale_cgs": .03,
                     "log10_tune_capacitance_pf": .02,
                     "phase_slope_rad_per_mhz": .005, "phase_curvature_rad_per_mhz2": .005}[n]
            config["fit_parameters"][n].update(initial=true[n], bounds=[true[n]-width, true[n]+width])
        c = conditional_circuit(config)
        y = node_basis(self.f, c, true)@np.array([1500., -80., 4.])
        trace, fit = match_scan(self.f, y, config, starts=1)
        np.testing.assert_allclose(trace[:, 2], y, atol=2e-7)
        np.testing.assert_array_equal(reconstruct(self.f, fit), trace[:, 2])
        np.testing.assert_array_equal(reconstruct(self.f, fit, False), trace[:, 3])
        self.assertAlmostEqual(fit["parameters"]["P_model"], .4, delta=.003)
        self.assertEqual(fit["n_free"], 12)
        self.assertTrue(fit["all_500_bins_fitted"])

    def test_published_experimental_fits_reconstruct_every_sample(self):
        metadata = json.loads((ROOT/"docs/assets/experimental-matching.json").read_text())
        data, _ = load_signal_csv(ROOT/"examples/Sample_RawSignal.csv")
        self.assertEqual(len(metadata["fits"]), 5)
        for i, fit in enumerate(metadata["fits"]):
            prediction = reconstruct(self.f, fit)
            residual = data[i, 1:]-prediction
            np.testing.assert_allclose(np.sqrt(np.mean(residual**2)), fit["diagnostics"]["rms_recorded_units"], rtol=1e-7)
            self.assertTrue(fit["all_500_bins_fitted"])
            self.assertTrue(fit["success"])
            self.assertEqual(fit["near_bounds"], {})
            config = seed_configuration(fit)
            raw, base, signal = clean_spectrum(self.f, fit["parameters"]["P_model"], config)
            np.testing.assert_allclose(raw, prediction, atol=1e-11)
            np.testing.assert_allclose(base, reconstruct(self.f, fit, False), atol=1e-11)
            np.testing.assert_allclose(raw-base, signal, atol=0)
        for name, expected in metadata["code_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT/"tools"/name).read_bytes()).hexdigest(), expected)
        publication = metadata["publication"]
        for name, expected in publication["figure_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT/"docs/assets"/name).read_bytes()).hexdigest(), expected)
        self.assertEqual(hashlib.sha256((ROOT/"tools/export_signal_matching.py").read_bytes()).hexdigest(), publication["exporter_sha256"])
        self.assertEqual(hashlib.sha256((ROOT/"tools/generate_matched_data.py").read_bytes()).hexdigest(), publication["generation"]["generator_sha256"])
        import xml.etree.ElementTree as ET
        tree = ET.fromstring((ROOT/"docs/assets/experimental-signal-matches.svg").read_text())
        ns = {"s": "http://www.w3.org/2000/svg"}
        for i in range(1, 6):
            dots = tree.find(f".//s:g[@id='measured-scan-{i}']", ns)
            self.assertIsNotNone(dots)
            self.assertEqual(len(dots.findall('.//s:use', ns)), 500)

    def test_generator_labels_units_reproducibility_and_reference_sharing(self):
        report = json.loads((ROOT/"docs/assets/experimental-matching.json").read_text())
        coverage = json.loads(DEFAULT_COVERAGE.read_text())
        first, configs = generate(report, 20, 10, 91, -.5, .5, coverage)
        second, same_configs = generate(report, 20, 10, 91, -.5, .5, coverage)
        self.assertEqual(configs, same_configs)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])
        self.assertEqual(first["signals"].shape, (20, 500))
        self.assertEqual(str(first["simulator"]), GENERATOR)
        self.assertEqual(str(first["voltage_unit"]), "recorded units")
        self.assertFalse('calibration' in first)
        self.assertTrue(np.all((-0.5 <= first["P"]) & (first["P"] <= .5)))
        self.assertFalse(np.isin(first["P"], [f["parameters"]["P_model"] for f in report["fits"]]).any())
        np.testing.assert_allclose(first["signals"], first["clean_raw"]+first["noise"], atol=0)
        np.testing.assert_allclose(first["clean_raw"]-first["clean_baseline"], first["clean_signal"], atol=0)
        for group in np.unique(first["configuration_id"]):
            idx = np.flatnonzero(first["configuration_id"] == group)
            np.testing.assert_array_equal(first["baselines"][idx[0]], first["baselines"][idx[1]])
            self.assertFalse(np.array_equal(first["noise"][idx[0]], first["noise"][idx[1]]))
            c = __import__('circuit').Circuit(**configs[group]["circuit"])
            self.assertAlmostEqual(cable_parameters(32.7e6, c)[1].imag*3.58, np.pi)
            self.assertLessEqual(abs(c.cable_length_m-3.58), .005+1e-12)
        zero, baseline, signal = clean_spectrum(self.f, 0., configs[0])
        np.testing.assert_array_equal(zero, baseline)
        np.testing.assert_array_equal(signal, np.zeros(500))

    def test_covariance_rejects_wrong_grid_nonfinite_asymmetric_and_negative(self):
        for bad in (np.eye(512), np.eye(500)*-1, np.eye(500)*np.nan):
            with self.assertRaises(ValueError):
                covariance_factor(bad)
        bad = np.eye(500); bad[0, 1] = .1
        with self.assertRaises(ValueError):
            covariance_factor(bad)
        cov = np.eye(500)*1e-8
        cov[:2, :2] = 1e-8  # singular PSD block is valid
        factor = covariance_factor(cov)
        np.testing.assert_allclose(factor@factor.T, cov, atol=1e-20)


if __name__ == "__main__":
    unittest.main()
