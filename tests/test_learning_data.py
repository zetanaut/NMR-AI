"""Protect the acquisition grid, preprocessing units and measured-seed holdouts."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from learning_data import (FREQUENCY, MATCHED_SIMULATOR, SIMULATOR,
                           input_contract, load_dataset, make_features)


class LearningData(unittest.TestCase):
    def arrays(self):
        rng = np.random.default_rng(19)
        groups = np.repeat(np.arange(15), 4)
        return dict(signals=rng.normal(0, .01, (60, 500)), baselines=np.ones((60, 500))*.1,
                    P=rng.uniform(-.6, .6, 60), configuration_id=groups,
                    source_scan_1based=groups % 5 + 1, frequency_mhz=FREQUENCY,
                    simulator=MATCHED_SIMULATOR, voltage_unit="recorded units")

    def test_exact_acquisition_and_subtraction_before_float32(self):
        np.testing.assert_array_equal(FREQUENCY, 32.3 + .0015287*np.arange(500))
        self.assertAlmostEqual(FREQUENCY[-1], 33.0628213, places=12)
        baseline = np.ones((1, 500))
        raw = baseline + 1e-10
        features = make_features(raw, None, baseline, mode="lineshape")
        np.testing.assert_array_equal(features[:, 1], (raw-baseline).astype(np.float32))
        self.assertTrue(np.all(features[:, 1] > 0))
        with self.assertRaisesRegex(ValueError, "without TE"):
            make_features(raw, np.ones(1), baseline, mode="lineshape")
        with self.assertRaisesRegex(ValueError, "require.*calibration"):
            make_features(raw, None, baseline)

    def test_wrong_grid_mode_and_source_membership_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"data.npz"
            original = self.arrays()
            for frequency in (np.linspace(32.3, 33.1, 500), np.linspace(32.3, 33.1, 512), FREQUENCY[::-1]):
                np.savez(path, **{**original, "frequency_mhz": frequency})
                with self.assertRaisesRegex(ValueError, "frequency grid"):
                    load_dataset(path, training=True)
            np.savez(path, **{**original, "voltage_unit": "V"})
            with self.assertRaisesRegex(ValueError, "voltage_unit"):
                load_dataset(path, training=True)
            missing = {k: v for k, v in original.items() if k != "source_scan_1based"}
            np.savez(path, **missing)
            with self.assertRaisesRegex(ValueError, "source_scan_1based"):
                load_dataset(path, training=True)
            sources = original["source_scan_1based"].copy(); sources[0] = 5
            np.savez(path, **{**original, "source_scan_1based": sources})
            with self.assertRaisesRegex(ValueError, "one valid source"):
                load_dataset(path, training=True)

    def test_clean_simulator_arrays_are_never_model_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"data.npz"
            arrays = self.arrays()
            np.savez(path, **arrays)
            expected = load_dataset(path, training=True)["features"]
            np.savez(path, **arrays, clean_signal=np.full((60, 500), np.nan), clean_baseline=np.full((60, 500), 1e20))
            np.testing.assert_array_equal(load_dataset(path, training=True)["features"], expected)
            with self.assertRaisesRegex(ValueError, "do not match"):
                load_dataset(path, expected_contract=input_contract(SIMULATOR))

    def test_lineshape_training_and_label_free_measured_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            path, model = folder/"data.npz", folder/"model"
            arrays = self.arrays(); np.savez(path, **arrays)
            result = subprocess.run([sys.executable, str(ROOT/"tools/train_model.py"), "--data", str(path),
                                     "--output-dir", str(model), "--epochs", "2", "--patience", "2",
                                     "--architecture", "mlp", "--device", "cpu", "--threads", "1"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            split = json.loads((model/"split.json").read_text())
            self.assertEqual(split["group_column"], "source_scan_1based")
            sets = [set(split["group_values"][name]) for name in ("train", "validation", "test")]
            self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            with np.load(model/"scaler.npz") as scaler, np.load(model/"partition.npz") as partition:
                features = make_features(arrays["signals"], None, arrays["baselines"], mode="lineshape")
                np.testing.assert_array_equal(scaler["feature_mean"], features[partition["train"]].mean(axis=(0, 2), keepdims=True))
                np.testing.assert_array_equal(scaler["frequency_mhz"], FREQUENCY)
            measured = folder/"measured.npz"
            np.savez(measured, signals=arrays["signals"][:2], baselines=arrays["baselines"][:2],
                     frequency_mhz=FREQUENCY, feature_mode="lineshape", voltage_unit="recorded units")
            command = [sys.executable, str(ROOT/"tools/predict.py"), "--model-dir", str(model),
                       "--data", str(measured), "--output", str(folder/"predictions.csv")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = np.atleast_1d(np.genfromtxt(folder/"predictions.csv", delimiter=",", names=True))
            self.assertEqual(rows.dtype.names, ("P_pred",))
            self.assertEqual(len(rows), 2)
            self.assertTrue(np.isfinite(rows["P_pred"]).all())
