#!/usr/bin/env python3
"""Train a tutorial CNN with configuration-grouped validation and saved preprocessing."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from analyze_predictions import summarize
from nmr_lab import build_model, group_split, make_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("local-results/model"))
    parser.add_argument("--architecture", choices=["compact", "physics_multiscale"], default="physics_multiscale")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
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
        signals, cc, labels, groups = data["signals"], data["cc"], data["P"][:, None], data["configuration_id"]
    if signals.shape != (len(labels), 512) or cc.shape != (len(labels),) or groups.shape != (len(labels),):
        parser.error("Dataset must have 512 signal bins and aligned label, cc, and group arrays")
    if not all(np.isfinite(array).all() for array in [signals, cc, labels, groups]) or np.any(cc == 0):
        parser.error("Data must be finite, with nonzero known calibration")
    train, validation, test = group_split(groups, args.seed)
    features = make_features(signals, cc)
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
    loaders = [DataLoader(TensorDataset(torch.from_numpy(x[idx]), torch.from_numpy(labels[idx] / target_scale)),
                          batch_size=args.batch_size, shuffle=(i == 0)) for i, idx in enumerate([train, validation, test])]

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
    with torch.inference_mode():
        prediction = np.concatenate([model(inputs.to(device)).cpu().numpy() * target_scale for inputs, _ in loaders[2]])[:, 0]
    truth = labels[test, 0]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture": args.architecture, "model_state": best_state}, args.output_dir / "model.pt")
    np.savez(args.output_dir / "scaler.npz", feature_mean=mean, feature_std=std, target_scale=np.array([target_scale]))
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2))
    split = {"strategy": "grouped", "group_column": "configuration_id", "rows": {}, "groups": {}}
    for name, idx in zip(["train", "validation", "test"], [train, validation, test]):
        split["rows"][name] = len(idx)
        split["groups"][name] = len(np.unique(groups[idx]))
    (args.output_dir / "split.json").write_text(json.dumps(split, indent=2))
    (args.output_dir / "metrics.json").write_text(json.dumps(summarize(truth, prediction, 0.05), indent=2))
    with (args.output_dir / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    np.savetxt(args.output_dir / "test_predictions.csv", np.column_stack([truth, prediction, prediction - truth, groups[test]]),
               delimiter=",", header="P_true,P_pred,P_residual,configuration_id", comments="")
    print(f"Saved best validation checkpoint and held-out metrics to {args.output_dir}; device={device}")


if __name__ == "__main__":
    main()
