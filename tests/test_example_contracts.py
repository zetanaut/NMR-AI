"""Guard the public working examples against mixed frequency-grid contracts."""
import json
from pathlib import Path
import re
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from baseline_data import acquisition_grid, load_acquisition, load_baseline_csv
from experimental_data import load_signal_csv
from uva_nd3_data import load_nd3


class WorkingExampleContracts(unittest.TestCase):
    def test_every_working_data_file_has_500_bins(self):
        examples = ROOT / "examples"
        # New datasets must be added to this audit before becoming examples.
        self.assertEqual({p.name for p in examples.iterdir() if p.is_file()},
                         {"README.md", "deuteron-baseline.csv", "Sample_RawSignal.csv", "uva-nd3.json"})
        baseline, audit = load_baseline_csv(examples / "deuteron-baseline.csv")
        self.assertEqual(baseline.shape, (1, 501))  # timestamp + 500 amplitudes
        self.assertEqual(audit["decimal_whitespace_repair_count"], 35)
        butanol, _ = load_signal_csv(examples / "Sample_RawSignal.csv")
        self.assertEqual(butanol.shape, (5, 501))
        nd3, audit = load_nd3(examples / "uva-nd3.json")
        self.assertEqual((audit["rows"], audit["bins"]), (5, 500))
        frequency = acquisition_grid(load_acquisition()) / 1e6
        for row in nd3["records"]:
            np.testing.assert_array_equal(row["frequency_mhz"], frequency)
            for channel in ("phase", "baseline", "basesub"):
                self.assertEqual(len(row[channel]), 500)

    def test_all_published_frequency_arrays_and_bin_contracts_match(self):
        frequency = acquisition_grid(load_acquisition()) / 1e6
        found = []

        def inspect(value, path):
            if isinstance(value, dict):
                for key, item in value.items():
                    location = f"{path}.{key}"
                    if key in ("frequency", "frequency_mhz") and isinstance(item, list):
                        np.testing.assert_allclose(item, frequency, atol=1e-13, rtol=0, err_msg=location)
                        found.append(location)
                    if key in ("bins", "samples_per_row", "plotted_samples"):
                        self.assertEqual(item, 500, location)
                    self.assertNotEqual(key, "all_512_bins_fitted", location)
                    inspect(item, location)
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    if isinstance(item, (dict, list)):
                        inspect(item, f"{path}[{index}]")

        for asset in (ROOT / "docs/assets").glob("*.json"):
            inspect(json.loads(asset.read_text()), asset.name)
        script = (ROOT / "docs/assets/results.js").read_text()
        results = json.loads(script.split("=", 1)[1].strip().removesuffix(";"))
        inspect(results, "results.js")
        self.assertGreaterEqual(len(found), 3)

    def test_active_tutorial_text_has_no_old_bin_contract(self):
        paths = [ROOT / "README.md", ROOT / "examples/README.md"]
        paths += list((ROOT / "docs").glob("*.html"))
        paths += list((ROOT / "docs/reference").glob("*.md"))
        for path in paths:
            self.assertIsNone(re.search(r"\b512\b", path.read_text()), str(path))


if __name__ == "__main__":
    unittest.main()
