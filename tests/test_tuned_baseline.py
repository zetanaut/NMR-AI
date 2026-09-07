"""Synthetic recovery and enforcement of independently supplied tuning records."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
from circuit import Circuit, cable_parameters
from fit_tuned_baseline import (CIRCUIT_NAMES, validate_setup, resolved_circuit, setup_voltage,
                                fit_with_setup, reconstruct_fit)


def known_setup():
    c = asdict(Circuit(reference_hz=32.68e6, tune_capacitance_f=270e-12,
                       stray_capacitance_f=15e-12, cable_length_m=3.68))
    parameters = {k: {"value": c[k], "source": "Synthetic test truth, not an experimental measurement"} for k in CIRCUIT_NAMES}
    parameters.update({k: {"value": v, "source": "Synthetic test truth"} for k, v in
                       (("readout_gain", 20), ("readout_phase_rad", .7), ("readout_offset", -.01))})
    return {"parameters": parameters}


class TunedBaseline(unittest.TestCase):
    def setUp(self):
        # Explicit synthetic grid, not the supplied CSV's acquisition mapping.
        self.f = (32.28 + np.arange(500)*(.8/499))*1e6
        self.setup = known_setup()

    def test_fixed_means_no_optimization_or_hidden_unknowns(self):
        truth = validate_setup(self.setup)
        data = setup_voltage(self.f, truth, self.setup)
        prediction, report = fit_with_setup(self.f, data, self.setup, starts=1)
        self.assertEqual(report["n_free"], 0)
        self.assertEqual(report["resolved_parameters"], truth)
        np.testing.assert_array_equal(prediction, data)
        np.testing.assert_array_equal(reconstruct_fit(self.f, report), data)

    def test_fit_only_unknown_capacitance(self):
        true = validate_setup(self.setup)
        data = setup_voltage(self.f, true, self.setup)
        self.setup["parameters"]["tune_capacitance_f"] = {
            "initial": 250e-12, "bounds": [230e-12, 310e-12], "source": "Synthetic bounded unknown"}
        predicted, report = fit_with_setup(self.f, data, self.setup, starts=1)
        self.assertEqual(report["free_parameters"], ["tune_capacitance_f"])
        self.assertAlmostEqual(report["resolved_parameters"]["tune_capacitance_f"]/270e-12, 1, places=5)
        for key in true.keys()-{"tune_capacitance_f"}:
            self.assertEqual(report["resolved_parameters"][key], true[key])
        np.testing.assert_allclose(predicted, data, atol=1e-10)

    def test_known_half_wave_branch_and_small_unknown_trim(self):
        self.setup["parameters"].pop("cable_length_m")
        self.setup["cable_tuning"] = {"half_wave_multiple": 1, "reference_hz": 32.68e6, "source": "Synthetic tuning record"}
        self.setup["parameters"]["cable_delta_length_m"] = {"value": .002, "source": "Synthetic trim truth"}
        truth = validate_setup(self.setup)
        c = resolved_circuit(truth, self.setup)
        beta = cable_parameters(32.68e6, c)[1].imag
        self.assertAlmostEqual(c.cable_length_m, np.pi/beta+.002)
        data = setup_voltage(self.f, truth, self.setup)
        self.setup["parameters"]["cable_delta_length_m"] = {
            "initial": 0, "bounds": [-.01, .01], "source": "Synthetic connector/trim tolerance"}
        predicted, report = fit_with_setup(self.f, data, self.setup, starts=1)
        self.assertEqual(report["free_parameters"], ["cable_delta_length_m"])
        self.assertAlmostEqual(report["resolved_parameters"]["cable_delta_length_m"], .002, places=7)
        np.testing.assert_allclose(predicted, data, atol=1e-10)

    def test_readout_unknowns_and_nonzero_frequency_dependent_phase(self):
        self.setup["parameters"]["detector_phase_slope_rad_per_hz"]["value"] = 1e-7
        truth = validate_setup(self.setup)
        data = setup_voltage(self.f, truth, self.setup)
        for name, initial, bounds in (("readout_gain", 19, [15, 25]),
                                      ("readout_phase_rad", .68, [.5, .9]),
                                      ("readout_offset", -.009, [-.02, 0])):
            self.setup["parameters"][name] = {"initial": initial, "bounds": bounds, "source": "Synthetic unknown"}
        prediction, report = fit_with_setup(self.f, data, self.setup, starts=1)
        self.assertEqual(report["n_free"], 3)
        np.testing.assert_allclose(prediction, data, atol=1e-9)
        np.testing.assert_array_equal(reconstruct_fit(self.f, report), prediction)

    def test_cli_report_reconstruction_and_export_contract(self):
        import json
        import subprocess
        import tempfile
        from export_baseline_example import load_example, example_html
        from baseline_parameters import parameter_table_html
        root = Path(__file__).resolve().parents[1]
        truth = validate_setup(self.setup)
        data = setup_voltage(self.f, truth, self.setup) + np.random.default_rng(7).normal(0, 1e-6, 500)
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder/"setup.json").write_text(json.dumps(self.setup))
            np.savetxt(folder/"scan.csv", np.concatenate([[1], data])[None], delimiter=",")
            subprocess.run([sys.executable, str(root/"tools/fit_tuned_baseline.py"), str(folder/"scan.csv"),
                            "--setup", str(folder/"setup.json"), "--output-dir", str(folder/"fit"),
                            "--start-mhz", "32.28", "--step-mhz", str(.8/499), "--starts", "1",
                            "--frequency-source", "Explicit synthetic test grid, not experimental acquisition metadata"],
                           check=True, capture_output=True, text=True)
            _, metadata = load_example(folder/"fit")
            html = parameter_table_html(metadata)
            self.assertEqual(metadata["fit_mode"], "tuning-informed")
            self.assertIn("only 0 declared unknowns", html)
            self.assertNotIn("six quantities were fitted", html)
            example = example_html(metadata)
            self.assertIn("no optimizer is run", example)
            self.assertIn("0 exact duplicate records omitted", example)
            self.assertNotIn("five distinct", example)

    def test_priors_need_noise_scale_and_enter_separate_objective(self):
        data = setup_voltage(self.f, validate_setup(self.setup), self.setup)
        self.setup["parameters"]["tune_capacitance_f"] = {
            "initial": 270e-12, "bounds": [230e-12, 310e-12], "source": "Synthetic separate component measurement",
            "prior": {"mean": 280e-12, "sigma": 1e-12}}
        with self.assertRaisesRegex(ValueError, "noise_sigma"):
            validate_setup(self.setup)
        self.setup["noise_sigma_recorded"] = 1.0
        _, report = fit_with_setup(self.f, data, self.setup, starts=1)
        self.assertLess(abs(report["resolved_parameters"]["tune_capacitance_f"]-280e-12), 1e-13)
        self.assertGreater(report["data_objective"], 0)
        self.assertGreaterEqual(report["constraint_objective"], 0)

    def test_reject_missing_contradictory_or_nonidentifiable_inputs(self):
        invalid = deepcopy(self.setup)
        invalid["parameters"].pop("coil_inductance_h")
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_setup(invalid)
        invalid = deepcopy(self.setup)
        for key in ("drive_v", "readout_gain"):
            invalid["parameters"][key] = {"initial": 1, "bounds": [.1, 5], "source": "Unknown"}
        with self.assertRaisesRegex(ValueError, "product is identifiable"):
            validate_setup(invalid)
        invalid = deepcopy(self.setup)
        invalid["parameters"]["filling_factor"] = {"value": .1, "source": "Inactive"}
        with self.assertRaisesRegex(ValueError, "unknown"):
            validate_setup(invalid)
        invalid = deepcopy(self.setup)
        invalid["cable_tuning"] = {"half_wave_multiple": 1, "reference_hz": 32.68e6, "source": "Test"}
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_setup(invalid)  # cannot supply both physical length and derived length

    def test_reject_unfilled_template(self):
        import json
        template = Path(__file__).resolve().parents[1]/"configs/baseline-setup.template.json"
        with self.assertRaises(ValueError):
            validate_setup(json.loads(template.read_text()))


if __name__ == "__main__":
    unittest.main()
