#!/usr/bin/env python3
"""Apply a saved tutorial model to another tutorial-format NPZ spectrum file."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from learning_data import FREQUENCY, PREPROCESSING_VERSION, load_dataset, validate_contract
from nmr_lab import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("local-results/predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--partition", choices=["validation", "test"],
                        help="Evaluate only saved partition rows, verifying the original dataset hash")
    args = parser.parse_args()
    if args.batch_size < 1 or args.output.exists():
        parser.error("Use a positive batch size and a new output path")
    torch.set_num_threads(2)
    checkpoint = torch.load(args.model_dir / "model.pt", map_location="cpu", weights_only=True)
    try:
        contract = checkpoint.get("input_contract")
        validate_contract(contract)
        data = load_dataset(args.data, expected_contract=contract, partition=bool(args.partition))
    except ValueError as error:
        parser.error(str(error))
    model = build_model(checkpoint["architecture"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    features, truth = data["features"], data["P"]
    group_column = contract["group_column"]
    if args.partition:
        config = json.loads((args.model_dir/"config.json").read_text())
        if hashlib.sha256(args.data.read_bytes()).hexdigest() != config.get("dataset_sha256"):
            parser.error("Saved-partition evaluation requires the original hashed dataset")
        with np.load(args.model_dir/"partition.npz", allow_pickle=False) as partition:
            rows = partition[args.partition]
        features, truth = features[rows], truth[rows]
    with np.load(args.model_dir / "scaler.npz", allow_pickle=False) as scaler:
        if not np.array_equal(scaler["frequency_mhz"], FREQUENCY) or str(scaler["preprocessing_version"].item()) != PREPROCESSING_VERSION:
            parser.error("Saved scaler frequency/preprocessing version mismatch")
        if any(str(scaler[key].item()) != contract[key] for key in ("feature_mode", "voltage_unit")):
            parser.error("Saved scaler feature mode or units mismatch")
        x = ((features - scaler["feature_mean"]) / scaler["feature_std"]).astype(np.float32)
        target_scale = float(scaler["target_scale"][0])
    if not len(x) or not np.isfinite(x).all():
        parser.error("Input spectra must be nonempty and finite")
    with torch.inference_mode():
        prediction = np.concatenate([model(torch.from_numpy(x[i:i + args.batch_size])).numpy()
                                     for i in range(0, len(x), args.batch_size)])[:, 0] * target_scale
    args.output.parent.mkdir(parents=True, exist_ok=True)
    columns = prediction if truth is None else np.column_stack([truth, prediction])
    header = "P_pred" if truth is None else "P_true,P_pred"
    if args.partition:
        columns = np.column_stack([truth, prediction, truth-prediction, data["identifiers"]["configuration_id"][rows]])
        header = "P_true,P_pred,P_residual,configuration_id"
        if group_column != "configuration_id":
            columns = np.column_stack([columns, data["identifiers"][group_column][rows]])
            header += ","+group_column
    np.savetxt(args.output, columns, delimiter=",", comments="", header=header)
    print(f"Saved {len(prediction)} predictions to {args.output}")


if __name__ == "__main__":
    main()
