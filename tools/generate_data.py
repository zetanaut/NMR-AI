#!/usr/bin/env python3
"""Generate an unvalidated Pake/polynomial prototype, NOT a physical Q-meter dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
from lineshape import SIMULATOR
from nmr_lab import FREQUENCY, sample_configurations, simulate, validate_configurations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local-results/smoke.npz"))
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--num-configurations", type=int, default=100)
    parser.add_argument("--configurations", type=Path, help="Optional JSON configuration list; see nmr_lab.py schema")
    parser.add_argument("--coverage", choices=["narrow", "broad"], default="narrow")
    parser.add_argument("--p-min", type=float, default=-0.25)
    parser.add_argument("--p-max", type=float, default=0.25)
    parser.add_argument("--cc", type=float, default=-1.39)
    parser.add_argument("--noise-level", type=float, default=2.7e-5)
    parser.add_argument("--noise-correlation", type=float, default=0)
    parser.add_argument("--center-jitter", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configuration-seed", type=int, default=17)
    args = parser.parse_args()
    if args.num_samples < 3 or args.num_configurations < 3:
        parser.error("Use at least three samples and configurations")
    if not -1 <= args.p_min < args.p_max <= 1:
        parser.error("Require -1 <= p-min < p-max <= 1")
    if not np.isfinite([args.noise_level, args.noise_correlation, args.center_jitter, args.cc]).all():
        parser.error("Noise, jitter, and calibration values must be finite")
    if args.noise_level < 0 or not -1 < args.noise_correlation < 1 or args.center_jitter < 0 or args.cc == 0:
        parser.error("Require noise/jitter >= 0, |correlation| < 1, and nonzero cc")
    if args.output.suffix != ".npz":
        parser.error("Output must end in .npz")
    if args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Output or its metadata already exists; choose a new output path")
    configs = json.loads(args.configurations.read_text()) if args.configurations else sample_configurations(
        args.num_configurations, args.configuration_seed, args.coverage, args.cc)
    validate_configurations(configs)
    rng = np.random.default_rng(args.seed)
    labels = rng.uniform(args.p_min, args.p_max, args.num_samples)
    groups = rng.integers(0, len(configs), args.num_samples)
    signals = np.empty((args.num_samples, 512), dtype=np.float32)
    cc = np.array([configs[g]["cc"] for g in groups], dtype=np.float32)
    for i, (p, group) in enumerate(zip(labels, groups)):
        signals[i] = simulate(p, configs[group], rng, args.noise_level, args.noise_correlation, args.center_jitter)["signal"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, signals=signals, P=labels.astype(np.float32), cc=cc,
                        configuration_id=groups, frequency_mhz=FREQUENCY, simulator=SIMULATOR)
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata["simulator"] = SIMULATOR
    args.output.with_suffix(".json").write_text(json.dumps({"settings": metadata, "configurations": configs}, indent=2))
    print(f"Saved {args.num_samples} spectra from {len(np.unique(groups))} sampled configurations to {args.output}")


if __name__ == "__main__":
    main()
