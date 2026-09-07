"""Published measured-fit selection, provenance, and full-bin rendering."""
import hashlib
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


class BaselineExample(unittest.TestCase):
    def test_median_selection_statistics_and_source_provenance(self):
        metadata = json.loads((ROOT/"docs/assets/baseline-example.json").read_text())
        ordered = sorted(metadata["summaries"], key=lambda s: s["rms_recorded_units"])
        selected = ordered[len(ordered)//2]
        self.assertEqual(metadata["selected_trace_number"], selected["trace_number"])
        self.assertEqual(metadata["source_rows"], 6)
        self.assertEqual(metadata["unique_rows"], 5)
        self.assertEqual(metadata["residual_sign"], "recorded minus fitted")
        self.assertNotIn("timestamp", metadata["selected_fit"])
        for row in ordered:
            self.assertEqual(row["bins"], 500)
            self.assertAlmostEqual(row["rms_percent_of_peak_to_peak"],
                                   100*row["rms_recorded_units"]/row["peak_to_peak_recorded_units"])
        for name, digest in metadata["fit_code_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT/"tools"/name).read_bytes()).hexdigest(), digest)
        self.assertEqual(hashlib.sha256((ROOT/"tools/export_baseline_example.py").read_bytes()).hexdigest(),
                         metadata["exporter_sha256"])

    def test_vector_figure_retains_all_measured_points(self):
        metadata = json.loads((ROOT/"docs/assets/baseline-example.json").read_text())
        asset = ROOT/"docs/assets/baseline-example.svg"
        self.assertEqual(hashlib.sha256(asset.read_bytes()).hexdigest(), metadata["svg_sha256"])
        tree = ET.fromstring(asset.read_text())
        ns = {"s": "http://www.w3.org/2000/svg"}
        dots = tree.find(".//s:g[@id='PathCollection_1']", ns)
        self.assertIsNotNone(dots)
        self.assertEqual(len(dots.findall(".//s:use", ns)), 500)
        self.assertIn("recorded minus fitted", asset.read_text())
        page = (ROOT/"docs/baseline.html").read_text()
        self.assertIn('id="example"', page)
        self.assertIn("0.0275%", page)
        self.assertIn('src="assets/baseline-example.svg"', page)


if __name__ == "__main__":
    unittest.main()
