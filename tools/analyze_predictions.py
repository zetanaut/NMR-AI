#!/usr/bin/env python3
"""NMR-AI lab: bias, residual width, and relative error from held-out P predictions."""

import argparse
import json

import numpy as np


def summarize(truth, prediction, p0):
    """Use population SD so empirical RMSE² = bias² + width² exactly."""
    error = np.asarray(prediction, dtype=float) - np.asarray(truth, dtype=float)
    if error.size == 0:
        return {"n": 0, "bias": None, "width": None, "rmse": None}
    bias = float(error.mean())
    width = float(error.std(ddof=0))
    rmse = float(np.sqrt(np.mean(error**2)))
    return {
        "n": int(error.size),
        "bias": bias,
        "width": width,
        "rmse": rmse,
        "mae": float(np.abs(error).mean()),
        "p95_absolute_error": float(np.quantile(np.abs(error), 0.95)),
        "bias_percentage_points": 100 * bias,
        "width_percentage_points": 100 * width,
        "rmse_percentage_points": 100 * rmse,
        "relative_bias_percent_at_p0": 100 * bias / abs(p0),
        "relative_width_percent_at_p0": 100 * width / abs(p0),
        "relative_rmse_percent_at_p0": 100 * rmse / abs(p0),
    }


def load_predictions(path):
    rows = np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))
    if not {"P_true", "P_pred"}.issubset(rows.dtype.names or ()):
        raise ValueError("CSV must contain P_true and P_pred")
    truth, prediction = rows["P_true"], rows["P_pred"]
    if not len(truth) or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Predictions must be nonempty and finite")
    return truth, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", help="Trainer's test_predictions.csv")
    parser.add_argument("--p0", type=float, default=0.05, help="Signed reference P as a fraction")
    parser.add_argument("--half-width", type=float, default=0.005, help="Signed P selection half-width")
    args = parser.parse_args()
    if not np.isfinite(args.p0) or args.p0 == 0 or abs(args.p0) > 1:
        parser.error("p0 must be finite, nonzero, and within [-1, 1]")
    if not np.isfinite(args.half_width) or args.half_width <= 0:
        parser.error("half-width must be finite and positive")
    truth, prediction = load_predictions(args.predictions)
    low, high = args.p0 - args.half_width, args.p0 + args.half_width
    selected = (truth >= low) & (truth < high)
    report = {
        "residual_definition": "P_pred - P_true; all absolute metrics use fractional P",
        "reference_p0": args.p0,
        "pooled": summarize(truth, prediction, args.p0),
        "signed_p_band": {"low_inclusive": low, "high_exclusive": high,
                          **summarize(truth[selected], prediction[selected], args.p0)},
        "interpretation": "Pooled relative values are conversions at P0, not conditional performance. "
                          "The signed band estimates local performance; inspect n and uncertainty. "
                          "Width is residual SD, not uncertainty on the mean or a per-event interval.",
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
