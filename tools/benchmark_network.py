#!/usr/bin/env python3
"""Measure forward-pass cost of the standalone tutorial architectures."""

import argparse
import json

import torch
from torch.utils.benchmark import Timer
from nmr_lab import ARCHITECTURES, build_model

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=ARCHITECTURES, default="physics_multiscale")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch-size and threads must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    model = build_model(args.architecture).eval().to(args.device)
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
