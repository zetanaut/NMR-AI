#!/usr/bin/env python3
"""Run the declared MLP/DNN/CNN protocol with shared data and deferred test evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/model-comparison.json")
    parser.add_argument("--runs-dir", type=Path, help="Fresh output directory; defaults to the protocol's directory")
    parser.add_argument("--stage", choices=("train", "evaluate", "all"), default="all")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    data = ROOT/protocol["dataset"]
    if digest(data) != protocol["dataset_sha256"]:
        parser.error("Dataset hash differs from the declared protocol")
    folder = (args.runs_dir or ROOT/protocol["run_directory"]).resolve()
    if args.stage in ("train", "all"):
        if folder.exists() and any(folder.iterdir()):
            parser.error("Use a fresh --runs-dir for a new comparison")
        folder.mkdir(parents=True, exist_ok=True)
        (folder/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
        for architecture in protocol["architectures"]:
            command = [sys.executable, str(ROOT/"tools/train_model.py"), "--data", str(data),
                       "--output-dir", str(folder/architecture), "--architecture", architecture]
            for key, value in protocol["training"].items():
                command.append("--"+key.replace("_", "-"))
                if value is not True:
                    command.append(str(value))
            print(f"Training {architecture}; progress: {folder/architecture}-training.log", flush=True)
            with (folder/f"{architecture}-training.log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=True)
            metrics = json.loads((folder/architecture/"validation_metrics.json").read_text())
            print(f"{architecture}: validation RMSE = {100*metrics['rmse']:.6g} pp", flush=True)
    if args.stage in ("evaluate", "all"):
        if (folder/"evaluation.json").exists():
            parser.error("This comparison already has a test-evaluation record; outputs are preserved")
        if json.loads((folder/"protocol.json").read_text()) != protocol:
            parser.error("Saved protocol differs from the requested comparison")
        reference_partition = None
        for architecture in protocol["architectures"]:
            run = folder/architecture
            config = json.loads((run/"config.json").read_text())
            if config["dataset_sha256"] != protocol["dataset_sha256"] or any(
                    config[key] != value for key, value in protocol["training"].items()):
                parser.error("Comparison runs must use the declared data and training settings")
            with np.load(run/"partition.npz", allow_pickle=False) as saved:
                partition = {name: saved[name] for name in saved.files}
            if reference_partition is None:
                reference_partition = partition
            else:
                for name in reference_partition:
                    np.testing.assert_array_equal(partition[name], reference_partition[name])
            provenance = json.loads((run/"provenance.json").read_text())
            for name, expected in provenance["code_sha256"].items():
                if digest(ROOT/"tools"/name) != expected:
                    parser.error(f"Training source changed since fitting: {name}")
        record = {
            "decision": "All declared architectures retained after validation; no architecture or setting changes. Evaluate each best-validation checkpoint once on the same saved test rows.",
            "protocol_sha256": digest(args.protocol),
            "checkpoint_sha256": {arch: digest(folder/arch/"model.pt") for arch in protocol["architectures"]},
            "prediction_device": "cpu", "prediction_threads": 2, "prediction_batch_size": 128,
            "predictor_sha256": digest(ROOT/"tools/predict.py"),
        }
        (folder/"evaluation.json").write_text(json.dumps(record, indent=2)+"\n")
        for architecture in protocol["architectures"]:
            subprocess.run([sys.executable, str(ROOT/"tools/predict.py"), "--model-dir", str(folder/architecture),
                            "--data", str(data), "--partition", "test",
                            "--output", str(folder/architecture/"test_predictions.csv")], cwd=ROOT, check=True)
        print("Saved held-out predictions. Run tools/export_model_comparison.py to verify and publish.")


if __name__ == "__main__":
    main()
