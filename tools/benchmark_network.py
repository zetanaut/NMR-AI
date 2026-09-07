#!/usr/bin/env python3
"""Measure architecture forward-pass cost, without loading research checkpoints."""

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.benchmark import Timer

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=Path(__file__).resolve().parents[2] / "gen-NMR")
    parser.add_argument("--architecture", choices=["physics_multiscale", "legacy"], default="physics_multiscale")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not (args.project_dir / "ml" / "train_rgc_joint.py").is_file():
        parser.error("Set --project-dir to your gen-NMR checkout")
    sys.path.insert(0, str(args.project_dir.resolve()))
    from ml.train_rgc_joint import JointRGCNet, LegacyJointRGCNet
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    cls = JointRGCNet if args.architecture == "physics_multiscale" else LegacyJointRGCNet
    model = cls(n_outputs=1).eval().to(args.device)
    inputs = torch.randn(args.batch_size, 2, 512, device=args.device)
    # Timer warms up and synchronizes accelerator work. Model/data stay on device.
    with torch.inference_mode():
        measurement = Timer("model(inputs)", globals={"model": model, "inputs": inputs},
                            num_threads=args.threads).blocked_autorange(min_run_time=1.0)
    print(json.dumps({
        "architecture": args.architecture,
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu",
        "threads": args.threads,
        "batch_size": args.batch_size,
        "median_batch_ms": 1000 * measurement.median,
        "amortized_ms_per_spectrum": 1000 * measurement.median / args.batch_size,
        "batch_time_iqr_ms": 1000 * measurement.iqr,
        "scope": "Random initialized architecture, forward pass only. Excludes preprocessing, "
                 "transfers, file I/O, and startup. Batch amortization is not single-event latency.",
    }, indent=2))


if __name__ == "__main__":
    main()
