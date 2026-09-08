"""Audit published errors independently from every retained synthetic prediction."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from export_model_comparison import END, START, render_block
from learning_data import FREQUENCY, PREPROCESSING_VERSION
from nmr_lab import build_model


class ModelComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = json.loads((ROOT/"docs/assets/model-comparison.json").read_text())

    def test_metrics_and_common_test_events_match_every_published_prediction(self):
        snapshot = self.snapshot
        self.assertEqual(snapshot["bins"], 500)
        self.assertEqual(snapshot["input_contract"]["version"], PREPROCESSING_VERSION)
        np.testing.assert_array_equal(snapshot["input_contract"]["frequency_mhz"], FREQUENCY)
        groups = snapshot["group_values"]
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            self.assertFalse(set(groups[a]) & set(groups[b]))
        self.assertEqual(sum(snapshot["rows"].values()), snapshot["events"])
        self.assertEqual(len(set(snapshot["test_row_indices"])), snapshot["rows"]["test"])
        reference = None
        for run in snapshot["runs"]:
            values = np.asarray(run["test_predictions"])
            if reference is None:
                reference = values[:, [0, 2]]
            np.testing.assert_array_equal(reference, values[:, [0, 2]])
            self.assertEqual(set(values[:, 2]), set(groups["test"]))
            error = values[:, 0]-values[:, 1]
            metric = run["metrics"]
            self.assertEqual(len(error), metric["n"])
            self.assertAlmostEqual(metric["bias"], error.mean(), places=15)
            self.assertAlmostEqual(metric["width"], error.std(ddof=0), places=15)
            self.assertAlmostEqual(metric["rmse"]**2, np.mean(error**2), places=15)
            self.assertAlmostEqual(metric["rmse"]**2, metric["bias"]**2+metric["width"]**2, places=15)
            self.assertAlmostEqual(metric["p95_absolute_error"], np.quantile(abs(error), .95), places=15)
            self.assertAlmostEqual(metric["rmse_percentage_points"], 100*metric["rmse"], places=15)
            model = build_model(run["architecture"])
            self.assertEqual(sum(p.numel() for p in model.parameters()), run["parameters"])
            self.assertEqual(run["best_epoch"], min(run["history"], key=lambda row: row["val_loss"])["epoch"])

    def test_publication_table_figure_and_protocol_are_synchronized(self):
        snapshot = self.snapshot
        self.assertEqual(snapshot["protocol"], json.loads((ROOT/"configs/model-comparison.json").read_text()))
        figure_hash = hashlib.sha256((ROOT/"docs/assets/model-comparison.svg").read_bytes()).hexdigest()
        self.assertEqual(figure_hash, snapshot["figure_sha256"])
        page = (ROOT/"docs/index.html").read_text()
        block = START+page.split(START, 1)[1].split(END, 1)[0]+END
        self.assertEqual(block, render_block(snapshot))
        script = (ROOT/"docs/assets/model-comparison.js").read_text()
        embedded = json.loads(script.split("window.MODEL_COMPARISON = ", 1)[1].strip().removesuffix(";"))
        self.assertEqual(embedded, snapshot)

    def test_lineshape_publication_matches_source_holdout_and_saved_predictions(self):
        from export_lineshape_training import START, END, render_block
        snapshot = json.loads((ROOT/"docs/assets/lineshape-training.json").read_text())
        self.assertEqual(snapshot["protocol"], json.loads((ROOT/"configs/lineshape-training.json").read_text()))
        self.assertEqual(snapshot["bins"], 500)
        self.assertEqual(snapshot["input_contract"]["feature_mode"], "lineshape")
        self.assertEqual(snapshot["input_contract"]["voltage_unit"], "recorded units")
        np.testing.assert_array_equal(snapshot["input_contract"]["frequency_mhz"], FREQUENCY)
        groups = snapshot["group_values"]
        self.assertEqual([len(groups[name]) for name in ("train", "validation", "test")], [3, 1, 1])
        self.assertEqual(len(set(sum(groups.values(), []))), 5)
        run = snapshot["runs"][0]
        self.assertEqual(run["prediction_columns"], ["P_true", "P_pred", "source_scan_1based"])
        values = np.asarray(run["test_predictions"])
        self.assertEqual(set(values[:, 2]), set(groups["test"]))
        error = values[:, 0]-values[:, 1]
        self.assertEqual(len(error), snapshot["rows"]["test"])
        for key, expected in (("bias", error.mean()), ("width", error.std()),
                              ("rmse", np.sqrt(np.mean(error**2))),
                              ("p95_absolute_error", np.quantile(abs(error), .95))):
            self.assertAlmostEqual(run["metrics"][key], expected, places=15)
        page = (ROOT/"docs/matching.html").read_text()
        block = START+page.split(START, 1)[1].split(END, 1)[0]+END
        self.assertEqual(block, render_block(snapshot))


if __name__ == "__main__":
    unittest.main()
