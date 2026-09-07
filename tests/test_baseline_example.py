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
from baseline_parameters import CATALOG, parameter_reference_html
from export_baseline_example import draw_example, load_example


class BaselineExample(unittest.TestCase):
    def test_page_keeps_parameter_meanings_without_invalid_fit_claims(self):
        page = (ROOT/"docs/baseline.html").read_text()
        self.assertIn('id="example"', page)
        self.assertIn('id="known-tuning"', page)
        self.assertIn("32.68 MHz", page)
        metadata_path = ROOT/"docs/assets/baseline-example.json"
        if not metadata_path.exists():
            for name in CATALOG:
                self.assertIn(f"<code>{name}</code>", page)
            self.assertIn("awaiting the actual scan grid", page)
            self.assertNotIn('src="assets/baseline-example.svg"', page)
            self.assertFalse((ROOT/"docs/assets/baseline-example.svg").exists())
            self.assertIn(parameter_reference_html(), page)
        else:
            metadata = json.loads(metadata_path.read_text())
            self.assertEqual(metadata["nucleus"], "deuteron")
            self.assertTrue(metadata["frequency_source"])
            self.assertLessEqual(metadata["start_mhz"], 32.68)
            self.assertGreaterEqual(metadata["start_mhz"]+499*metadata["step_mhz"], 32.68)
            selected = sorted(metadata["summaries"], key=lambda s: s["rms_recorded_units"])[len(metadata["summaries"])//2]
            self.assertEqual(metadata["selected_trace_number"], selected["trace_number"])
            for name in CATALOG.keys()-{"cable_delta_length_m"}:
                self.assertIn(f"<code>{name}</code>", page)
            for name, digest in metadata["fit_code_sha256"].items():
                self.assertEqual(hashlib.sha256((ROOT/"tools"/name).read_bytes()).hexdigest(), digest)
            self.assertEqual(hashlib.sha256((ROOT/"docs/assets/baseline-example.svg").read_bytes()).hexdigest(),
                             metadata["svg_sha256"])

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

    def test_fit_cli_requires_actual_grid_and_its_source(self):
        import subprocess
        for script in ("fit_baseline.py", "fit_tuned_baseline.py"):
            result = subprocess.run([sys.executable, str(ROOT/"tools"/script), "unused.csv",
                                     "--output-dir", "unused-output"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            for option in ("--start-mhz", "--step-mhz", "--frequency-source"):
                self.assertIn(option, result.stderr)
        template = json.loads((ROOT/"configs/baseline-setup.template.json").read_text())
        self.assertEqual(template["parameters"]["reference_hz"]["value"], 32.68e6)
        self.assertEqual(template["cable_tuning"]["reference_hz"], 32.68e6)


if __name__ == "__main__":
    unittest.main()
