#!/usr/bin/env python3
"""Train a tutorial network with configuration-grouped validation and saved preprocessing."""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from analyze_predictions import summarize
from lineshape import SIMULATOR
from nmr_lab import ARCHITECTURES, FREQUENCY, build_model, group_split, make_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("local-results/model"))
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="physics_multiscale")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-order-seed", type=int,
                        help="Independent batch shuffle seed for comparisons across architectures")
    parser.add_argument("--defer-test", action="store_true",
                        help="Save validation results only; evaluate test once the comparison is fixed")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1:
        parser.error("Epochs, patience, batch size, and threads must be positive")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output directory is not empty; choose a new run directory")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    with np.load(args.data, allow_pickle=False) as data:
        if "simulator" not in data or str(data["simulator"].item()) != SIMULATOR:
            parser.error("Simulator version mismatch: regenerate the dataset")
        if "frequency_mhz" not in data or not np.array_equal(data["frequency_mhz"], FREQUENCY):
            parser.error("Dataset frequency grid does not match the model contract")
        signals, cc, labels, groups = data["signals"], data["calibration"], data["P"][:, None], data["configuration_id"]
        baselines = data["baselines"]
    if signals.shape != (len(labels), 512) or cc.shape != (len(labels),) or groups.shape != (len(labels),):
        parser.error("Dataset must have aligned signals, references, calibration, labels and groups")
    if not all(np.isfinite(array).all() for array in [signals, baselines, cc, labels, groups]) or np.any(cc == 0):
        parser.error("Data must be finite, with nonzero known calibration")
    train, validation, test = group_split(groups, args.seed)
    features = make_features(signals, cc, baselines)
    mean = features[train].mean(axis=(0, 2), keepdims=True)
    std = features[train].std(axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1
    x = ((features - mean) / std).astype(np.float32)
    target_scale = max(float(labels[train].std()), 1e-6)
    model = build_model(args.architecture)
    if hasattr(model, "calibrate"):
        model.calibrate(x[train], labels[train] / target_scale)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    batch_generator = (None if args.batch_order_seed is None
                       else torch.Generator().manual_seed(args.batch_order_seed))
    loaders = [DataLoader(TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy((labels[idx] / target_scale).astype(np.float32))),
                          batch_size=args.batch_size, shuffle=(i == 0),
                          generator=batch_generator if i == 0 else None)
               for i, idx in enumerate([train, validation, test])]

    def epoch(loader, training):
        model.train(training)
        total = 0
        with torch.set_grad_enabled(training):
            for inputs, truth in loader:
                loss = torch.nn.functional.mse_loss(model(inputs.to(device)), truth.to(device))
                if training:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                total += float(loss.detach()) * len(inputs)
        return total / len(loader.dataset)

    history, best_loss, best_state, stale = [], float("inf"), None, 0
    started = time.perf_counter()
    for number in range(1, args.epochs + 1):
        training_loss, validation_loss = epoch(loaders[0], True), epoch(loaders[1], False)
        if not np.isfinite([training_loss, validation_loss]).all():
            raise RuntimeError("Non-finite loss; inspect data and learning rate")
        history.append({"epoch": number, "train_loss": training_loss, "val_loss": validation_loss,
                        "lr": optimizer.param_groups[0]["lr"]})
        scheduler.step(validation_loss)
        print(f"epoch={number:03d} train={training_loss:.6g} validation={validation_loss:.6g}", flush=True)
        if validation_loss < best_loss:
            best_loss, stale = validation_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    training_seconds = time.perf_counter()-started
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": args.architecture, "model_state": best_state, "simulator": SIMULATOR},
               args.output_dir / "model.pt")
    np.savez(args.output_dir / "scaler.npz", feature_mean=mean, feature_std=std, target_scale=np.array([target_scale]))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["simulator"] = SIMULATOR
    config["dataset_sha256"] = digest(args.data)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    split = {"strategy": "grouped", "group_column": "configuration_id", "rows": {}, "groups": {}, "group_values": {}}
    for name, idx in zip(["train", "validation", "test"], [train, validation, test]):
        split["rows"][name] = len(idx)
        split["groups"][name] = len(np.unique(groups[idx]))
        split["group_values"][name] = np.unique(groups[idx]).tolist()
    (args.output_dir / "split.json").write_text(json.dumps(split, indent=2))
    np.savez(args.output_dir / "partition.npz", train=train, validation=validation, test=test)
    for name, idx, loader in (("validation", validation, loaders[1]), ("test", test, loaders[2])):
        if name == "test" and args.defer_test:
            continue
        with torch.inference_mode():
            prediction = np.concatenate([model(inputs.to(device)).cpu().numpy() * target_scale
                                         for inputs, _ in loader])[:, 0]
        truth = labels[idx, 0]
        metric_file = "metrics.json" if name == "test" else "validation_metrics.json"
        (args.output_dir / metric_file).write_text(json.dumps(summarize(truth, prediction, 0.05), indent=2))
        np.savetxt(args.output_dir / f"{name}_predictions.csv",
                   np.column_stack([truth, prediction, truth-prediction, groups[idx]]), delimiter=",",
                   header="P_true,P_pred,P_residual,configuration_id", comments="")
    with (args.output_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    provenance = {
        "dataset_sha256": config["dataset_sha256"], "partition_sha256": digest(args.output_dir/"partition.npz"),
        "code_sha256": {name: digest(Path(__file__).parent/name) for name in
                        ("nmr_lab.py", "train_model.py", "lineshape.py", "circuit.py", "analyze_predictions.py")},
        "python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
        "device": device, "device_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.processor(),
        "cuda_build": torch.version.cuda, "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "best_epoch": min(history, key=lambda row: row["val_loss"])["epoch"],
        "training_seconds": training_seconds,
        "timing_scope": "Training loop, including validation and best-state copies; excludes data loading and feature/ridge fitting",
        "test_evaluated": not args.defer_test,
    }
    (args.output_dir/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    print(f"Saved best validation checkpoint to {args.output_dir}; device={device}; test_evaluated={not args.defer_test}")


if __name__ == "__main__":
    main()
