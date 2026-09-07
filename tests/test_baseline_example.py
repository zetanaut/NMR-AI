"""Deuteron publication contract, stale-grid rejection, and all-bin rendering."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from baseline_parameters import CATALOG
from baseline_data import load_acquisition, acquisition_grid, load_baseline_csv
from fit_tuned_baseline import reconstruct_fit, resolved_circuit, validate_setup, DEFAULT_SETUP
from export_baseline_example import draw_example, load_example


class BaselineExample(unittest.TestCase):
    def test_page_keeps_parameter_meanings_without_invalid_fit_claims(self):
        page = (ROOT/"docs/baseline.html").read_text()
        self.assertIn('id="example"', page)
        self.assertIn('id="known-tuning"', page)
        self.assertIn("32.7 MHz", page)
        metadata_path = ROOT/"docs/assets/baseline-example.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["nucleus"], "deuteron")
            self.assertTrue(metadata["frequency_source"])
            self.assertEqual(metadata["start_mhz"], 32.3)
            self.assertEqual(metadata["step_mhz"], .0015287)
            self.assertEqual(metadata["reference_hz"], 32.7e6)
            self.assertEqual(metadata["acquisition"], load_acquisition())
            self.assertEqual(metadata["unique_rows"], 1)
            self.assertEqual(metadata["input_parsing"]["decimal_whitespace_repair_count"], 35)
            selected = sorted(metadata["summaries"], key=lambda s: s["rms_recorded_units"])[len(metadata["summaries"])//2]
            self.assertEqual(metadata["selected_trace_number"], selected["trace_number"])
            for name in CATALOG.keys()-{"cable_delta_length_m"}:
                self.assertIn(f"<code>{name}</code>", page)
            for name, digest in metadata["fit_code_sha256"].items():
                self.assertEqual(hashlib.sha256((ROOT/"tools"/name).read_bytes()).hexdigest(), digest)
            self.assertEqual(hashlib.sha256((ROOT/"docs/assets/baseline-example.svg").read_bytes()).hexdigest(),
                             metadata["svg_sha256"])
            self.assertEqual(hashlib.sha256((ROOT/"tools/export_baseline_example.py").read_bytes()).hexdigest(), metadata["exporter_sha256"])
            self.assertEqual(hashlib.sha256((ROOT/"tools/baseline_parameters.py").read_bytes()).hexdigest(), metadata["parameter_catalog_sha256"])
        else:
            self.fail("The confirmed public acquisition needs its measured-fit example")

    def test_published_fit_reconstructs_against_the_public_csv(self):
        metadata = json.loads((ROOT/"docs/assets/baseline-example.json").read_text())
        records, parsing = load_baseline_csv(ROOT/"examples/deuteron-baseline.csv")
        prediction = reconstruct_fit(acquisition_grid(load_acquisition()), metadata["selected_fit"])
        residual = records[0, 1:]-prediction
        self.assertEqual(metadata["source_sha256"], parsing["source_sha256"])
        summary = metadata["summaries"][0]
        np.testing.assert_allclose(np.sqrt(np.mean(residual**2)), summary["rms_recorded_units"], rtol=1e-8)
        np.testing.assert_allclose(np.max(np.abs(residual)), summary["max_absolute_residual_recorded_units"], rtol=1e-8)
        self.assertLess(summary["rms_percent_of_peak_to_peak"], 1.0)
        fit = metadata["selected_fit"]
        setup = json.loads(DEFAULT_SETUP.read_text())
        self.assertEqual(fit["setup"], setup)
        self.assertEqual(setup["cable_tuning"]["half_wave_multiple"], 1)
        self.assertEqual(setup["cable_tuning"]["half_wave_length_m"], 3.58)
        self.assertEqual(fit["n_free"], 6)
        self.assertEqual(len(fit["nonlinear_parameters"]), 3)
        self.assertEqual(len(fit["profiled_parameters"]), 3)
        self.assertEqual(fit["active_bound_parameters"], fit["nonlinear_parameters"])
        self.assertEqual(fit["near_bounds"], {"stray_capacitance_f": "upper", "cable_delta_length_m": "upper"})
        self.assertIn("not an accepted hardware calibration", (ROOT/"docs/baseline.html").read_text())
        for candidate in fit["multistart_candidates"]:
            c = resolved_circuit(candidate["parameters"], setup)
            self.assertGreaterEqual(c.cable_length_m, 3.58*.97-1e-12)
            self.assertLessEqual(c.cable_length_m, 3.58*1.03+1e-12)
        self.assertNotIn("1.3544", (ROOT/"docs/baseline.html").read_text())

    def test_reject_missing_frequency_provenance_and_proton_grid(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            report = folder/"fit_report.json"
            report.write_text("{}")
            with self.assertRaisesRegex(ValueError, "fresh deuteron fit"):
                load_example(folder)
            report.write_text(json.dumps({"nucleus": "deuteron", "frequency_source": "Wrong grid regression test",
                                          "start_mhz": 212.6, "step_mhz": .0015287}))
            with self.assertRaisesRegex(ValueError, "does not bracket"):
                load_example(folder)
            report.write_text(json.dumps({"nucleus": "deuteron", "frequency_source": "Regression test",
                                          "start_mhz": 32.3, "step_mhz": .0015287}))
            with self.assertRaisesRegex(ValueError, "tuning-informed"):
                load_example(folder)

    def test_vector_figure_retains_all_points_on_synthetic_deuteron_grid(self):
        from circuit import nominal_circuit, detector_voltage
        f = np.linspace(32.28, 33.08, 500)
        prediction = detector_voltage(f*1e6, nominal_circuit())
        residual = np.random.default_rng(10).normal(0, 1e-8, 500)
        data = np.column_stack([f, prediction+residual, prediction, residual])
        with tempfile.TemporaryDirectory() as folder:
            asset = Path(folder)/"synthetic.svg"
            draw_example(data, asset)
            tree = ET.fromstring(asset.read_text())
            ns = {"s": "http://www.w3.org/2000/svg"}
            dots = tree.find(".//s:g[@id='PathCollection_1']", ns)
            self.assertIsNotNone(dots)
            self.assertEqual(len(dots.findall(".//s:use", ns)), 500)
            self.assertIn("recorded minus fitted", asset.read_text())

    def test_fit_cli_requires_complete_alternative_grid_overrides(self):
        import subprocess
        for script in ("fit_baseline.py", "fit_tuned_baseline.py"):
            args = [sys.executable, str(ROOT/"tools"/script), "unused.csv",
                    "--output-dir", "unused-output", "--start-mhz", "32.3"]
            if script == "fit_tuned_baseline.py":
                args.extend(["--setup", "unused.json"])
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            for option in ("--start-mhz", "--step-mhz", "--frequency-source"):
                self.assertIn(option, result.stderr)
        template = json.loads((ROOT/"configs/baseline-setup.template.json").read_text())
        self.assertEqual(template["parameters"]["reference_hz"]["value"], 32.7e6)
        self.assertEqual(template["cable_tuning"]["reference_hz"], 32.7e6)


if __name__ == "__main__":
    unittest.main()
