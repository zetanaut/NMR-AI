"""Check saved model/preprocessing/partition contracts and deferred evaluation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from lineshape import SIMULATOR
from nmr_lab import ARCHITECTURES, FREQUENCY, build_model, make_features


class ModelTraining(unittest.TestCase):
    def test_comparison_runner_saves_shared_partitions_and_protects_results(self):
        rng = np.random.default_rng(17)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data, protocol_path = folder/"data.npz", folder/"protocol.json"
            np.savez(data, signals=rng.normal(size=(48, 512)), baselines=np.zeros((48, 512)),
                     calibration=np.ones(48), P=rng.uniform(-.2, .2, 48),
                     configuration_id=np.repeat(np.arange(12), 4), frequency_mhz=FREQUENCY, simulator=SIMULATOR)
            protocol = {"dataset": str(data), "dataset_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                        "run_directory": str(folder/"runs"), "architectures": ["mlp", "dnn"],
                        "training": {"epochs": 1, "patience": 1, "threads": 1, "device": "cpu",
                                     "seed": 42, "batch_order_seed": 42, "defer_test": True}}
            protocol_path.write_text(json.dumps(protocol))
            command = [sys.executable, str(ROOT/"tools/run_model_comparison.py"), "--protocol", str(protocol_path)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            with np.load(folder/"runs/mlp/partition.npz") as a, np.load(folder/"runs/dnn/partition.npz") as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
            for name in ("mlp", "dnn"):
                self.assertTrue((folder/f"runs/{name}/validation_predictions.csv").exists())
                self.assertTrue((folder/f"runs/{name}/test_predictions.csv").exists())
            record = json.loads((folder/"runs/evaluation.json").read_text())
            self.assertEqual(set(record["checkpoint_sha256"]), {"mlp", "dnn"})
            repeated = subprocess.run(command+["--stage", "evaluate"], capture_output=True, text=True)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("already has a test-evaluation record", repeated.stderr)

    def test_architectures_learn_and_reload_the_same_input_contract(self):
        torch.set_num_threads(1)
        x = torch.randn(4, 2, 512)
        for name in ARCHITECTURES:
            with self.subTest(architecture=name):
                model = build_model(name)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                loss = (model(x)-torch.ones(4, 1)).square().mean()
                loss.backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))
                optimizer.step()
                model.eval()
                restored = build_model(name).eval()
                restored.load_state_dict(model.state_dict())
                with torch.inference_mode():
                    expected, actual = model(x), restored(x)
                self.assertEqual(actual.shape, (4, 1))
                torch.testing.assert_close(actual, expected)

    def test_deferred_training_and_partition_prediction_roundtrip(self):
        rng = np.random.default_rng(4)
        groups = np.repeat(np.arange(12), 4)
        signals = rng.normal(0, .01, (len(groups), 512))
        baselines = rng.normal(0, .001, signals.shape)
        labels = rng.uniform(-.25, .25, len(groups))
        calibration = np.ones(len(groups))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data_path, model_dir = folder/"data.npz", folder/"model"
            np.savez(data_path, signals=signals, baselines=baselines, calibration=calibration,
                     P=labels, configuration_id=groups, frequency_mhz=FREQUENCY, simulator=SIMULATOR)
            result = subprocess.run([
                sys.executable, str(ROOT/"tools/train_model.py"), "--data", str(data_path),
                "--output-dir", str(model_dir), "--architecture", "dnn", "--epochs", "2",
                "--patience", "2", "--device", "cpu", "--threads", "1", "--batch-order-seed", "42",
                "--defer-test"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((model_dir/"test_predictions.csv").exists())
            self.assertFalse((model_dir/"metrics.json").exists())
            self.assertTrue((model_dir/"validation_predictions.csv").exists())
            provenance = json.loads((model_dir/"provenance.json").read_text())
            self.assertFalse(provenance["test_evaluated"])
            self.assertEqual(provenance["dataset_sha256"], hashlib.sha256(data_path.read_bytes()).hexdigest())
            with np.load(model_dir/"partition.npz") as partition:
                indices = {name: partition[name] for name in partition.files}
            self.assertEqual(sorted(np.concatenate(list(indices.values())).tolist()), list(range(len(groups))))
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
                self.assertEqual(len(np.intersect1d(groups[indices[a]], groups[indices[b]])), 0)
            features = make_features(signals, calibration, baselines)
            with np.load(model_dir/"scaler.npz") as scaler:
                np.testing.assert_allclose(scaler["feature_mean"], features[indices["train"]].mean(axis=(0, 2), keepdims=True))
                x = ((features[indices["test"]]-scaler["feature_mean"])/scaler["feature_std"]).astype(np.float32)
                target_scale = float(scaler["target_scale"][0])
            output = folder/"predictions.csv"
            command = [sys.executable, str(ROOT/"tools/predict.py"), "--model-dir", str(model_dir),
                       "--data", str(data_path), "--partition", "test", "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = np.genfromtxt(output, delimiter=",", names=True)
            np.testing.assert_array_equal(rows["configuration_id"], groups[indices["test"]])
            np.testing.assert_array_equal(rows["P_true"], labels[indices["test"]])
            model = build_model("dnn").eval()
            model.load_state_dict(torch.load(model_dir/"model.pt", weights_only=True)["model_state"])
            with torch.inference_mode():
                expected = model(torch.from_numpy(x)).numpy()[:, 0]*target_scale
            np.testing.assert_allclose(rows["P_pred"], expected, rtol=1e-5, atol=1e-8)
            np.testing.assert_allclose(rows["P_residual"], rows["P_true"]-rows["P_pred"])
            # Even a valid NPZ with identical arrays but different bytes must not reuse saved row identities.
            changed_path = folder/"other.npz"
            with np.load(data_path) as original:
                np.savez(changed_path, **dict(original), extra=np.array(1))
            command[command.index("--data")+1] = str(changed_path)
            command[command.index("--output")+1] = str(folder/"wrong.csv")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("original hashed dataset", result.stderr)


if __name__ == "__main__":
    unittest.main()
