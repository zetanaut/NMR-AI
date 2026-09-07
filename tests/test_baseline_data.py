"""Public CSV integrity, explicit formatting repair, and strict record parsing."""
import csv
from decimal import Decimal
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
from baseline_data import load_baseline_csv

CSV = ROOT/"examples/deuteron-baseline.csv"
SHA256 = "dac7c4598c6ec32250ab763ab1bf99e2a7aa7346de4e452ce00e80f0e2b1eb28"


class BaselineData(unittest.TestCase):
    def test_public_example_preserves_all_samples_and_reports_repairs(self):
        original = CSV.read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(), SHA256)
        records, parsing = load_baseline_csv(CSV)
        self.assertEqual(records.shape, (1, 501))
        self.assertEqual(parsing["decimal_whitespace_repair_count"], 35)
        self.assertEqual(parsing["decimal_whitespace_repairs"][0]["field_0based"], 14)
        fields = next(csv.reader(original.decode("ascii").splitlines()))
        independent = [float(Decimal("".join(field.split()))) for field in fields]
        np.testing.assert_array_equal(records[0], independent)
        self.assertEqual(records[0, 1], -.1217583)
        self.assertEqual(records[0, -1], -.1602448)
        self.assertEqual(CSV.read_bytes(), original)

    def test_normal_rows_and_duplicates_are_not_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"scan.csv"
            row = ",".join(["100"]+["1.5"]*500)
            path.write_text(row+"\n"+row+"\n")
            records, parsing = load_baseline_csv(path)
            self.assertEqual(records.shape, (2, 501))
            self.assertEqual(parsing["decimal_whitespace_repair_count"], 0)

    def test_ambiguous_corrupt_or_nonfinite_values_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"bad.csv"
            for token in ("1 2", "1e -3", "NaN", "inf", "1e999", "", "bad"):
                with self.subTest(token=token):
                    path.write_text(",".join(["100", token]+["0"]*499))
                    with self.assertRaises(ValueError):
                        load_baseline_csv(path)
            for contents in ("", "100,1,2\n", "timestamp,value\n"):
                path.write_text(contents)
                with self.assertRaises(ValueError):
                    load_baseline_csv(path)

    def test_preview_is_all_measured_points_without_an_assumed_frequency_axis(self):
        path = ROOT/"docs/assets/deuteron-baseline-preview.svg"
        metadata = json.loads(path.with_suffix(".json").read_text())
        self.assertEqual(metadata["source_sha256"], SHA256)
        self.assertEqual(metadata["plotted_samples"], 500)
        self.assertEqual(metadata["decimal_whitespace_repair_count"], 35)
        self.assertEqual(metadata["svg_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        for name, digest in metadata["code_sha256"].items():
            self.assertEqual(digest, hashlib.sha256((ROOT/"tools"/name).read_bytes()).hexdigest())
        tree = ET.fromstring(path.read_text())
        ns = {"s": "http://www.w3.org/2000/svg"}
        dots = tree.find(".//s:g[@id='PathCollection_1']", ns)
        self.assertEqual(len(dots.findall(".//s:use", ns)), 500)
        self.assertIn("frequency grid not supplied", path.read_text())


if __name__ == "__main__":
    unittest.main()
