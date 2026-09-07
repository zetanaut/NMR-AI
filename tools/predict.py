#!/usr/bin/env python3
"""Apply a saved tutorial model to another tutorial-format NPZ spectrum file."""

import argparse
from pathlib import Path

import numpy as np
import torch
from nmr_lab import build_model, make_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("local-results/predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1 or args.output.exists():
        parser.error("Use a positive batch size and a new output path")
    torch.set_num_threads(2)
    checkpoint = torch.load(args.model_dir / "model.pt", map_location="cpu", weights_only=True)
    model = build_model(checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with np.load(args.data, allow_pickle=False) as data:
        features = make_features(data["signals"], data["cc"])
        truth = data["P"] if "P" in data.files else None
    with np.load(args.model_dir / "scaler.npz", allow_pickle=False) as scaler:
        x = ((features - scaler["feature_mean"]) / scaler["feature_std"]).astype(np.float32)
        target_scale = float(scaler["target_scale"][0])
    if not len(x) or not np.isfinite(x).all():
        parser.error("Input spectra must be nonempty and finite")
    with torch.inference_mode():
        prediction = np.concatenate([model(torch.from_numpy(x[i:i + args.batch_size])).numpy()
                                     for i in range(0, len(x), args.batch_size)])[:, 0] * target_scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = prediction if truth is None else np.column_stack([truth, prediction])
    np.savetxt(args.output, columns, delimiter=",", comments="", header="P_pred" if truth is None else "P_true,P_pred")
    print(f"Saved {len(prediction)} predictions to {args.output}")


if __name__ == "__main__":
    main()
