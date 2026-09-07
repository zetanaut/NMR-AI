#!/usr/bin/env python3
"""Generate complex-susceptibility Q-meter sweeps and independent references."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from lineshape import SIMULATOR
from nmr_lab import (FREQUENCY, sample_configurations, simulate, validate_configurations,
                     clean_response, te_calibration, noise_factor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("local-results/smoke.npz"))
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--num-configurations", type=int, default=100)
    parser.add_argument("--configurations", type=Path, help="JSON list of physical configurations")
    parser.add_argument("--noise-covariance", type=Path, help="Optional development-data 512x512 NPY covariance in V²")
    parser.add_argument("--coverage", choices=["narrow", "broad"], default="narrow")
    parser.add_argument("--p-min", type=float, default=-0.25)
    parser.add_argument("--p-max", type=float, default=0.25)
    parser.add_argument("--center-jitter", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--configuration-seed", type=int, default=17)
    args = parser.parse_args()
    if args.num_samples < 3 or args.num_configurations < 3 or not -1 <= args.p_min < args.p_max <= 1:
        parser.error("Need >=3 events/configurations and -1 <= p-min < p-max <= 1")
    if not np.isfinite(args.center_jitter) or args.center_jitter < 0:
        parser.error("Center jitter must be nonnegative and finite")
    if args.output.suffix != ".npz" or args.output.exists() or args.output.with_suffix(".json").exists():
        parser.error("Choose a new .npz output path")
    configs = json.loads(args.configurations.read_text()) if args.configurations else sample_configurations(
        args.num_configurations, args.configuration_seed, args.coverage)
    validate_configurations(configs)
    factor = noise_factor(np.load(args.noise_covariance, allow_pickle=False)) if args.noise_covariance else None
    rng = np.random.default_rng(args.seed)
    labels = rng.uniform(args.p_min, args.p_max, args.num_samples)
    groups = rng.integers(0, len(configs), args.num_samples)
    # Reference noise is shared within a configuration, never re-drawn per event.
    references, calibrations = [], []
    for config in configs:
        _, baseline = clean_response(0, config)
        noise = (rng.normal(size=512)*config["noise_rms_v"] if factor is None else factor @ rng.normal(size=512))
        references.append(baseline + noise/np.sqrt(config["reference_averages"]))
        calibrations.append(te_calibration(config))
    signals = np.empty((args.num_samples, 512), dtype=np.float64)
    for i, (p, group) in enumerate(zip(labels, groups)):
        signals[i] = simulate(p, configs[group], rng, args.center_jitter, factor)["signal"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, signals=signals, baselines=np.asarray(references)[groups],
                        calibration=np.asarray(calibrations)[groups], P=labels.astype(np.float32),
                        configuration_id=groups, frequency_mhz=FREQUENCY, simulator=SIMULATOR)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    settings.update(simulator=SIMULATOR, voltage_unit="V", calibration="P_TE / trapezoidal integral(S_TE/f df)",
                    reference_model="Independent noise averaged 16 times by default; shared by configuration",
                    distribution_provenance="Controlled physical-parameter sensitivity study, not a fitted experimental population",
                    dataset_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    if args.noise_covariance:
        settings["noise_covariance_sha256"] = hashlib.sha256(args.noise_covariance.read_bytes()).hexdigest()
    args.output.with_suffix(".json").write_text(json.dumps({"settings": settings, "configurations": configs}, indent=2))
    print(f"Saved {len(labels)} spectra from {len(np.unique(groups))} configurations to {args.output}")


if __name__ == "__main__":
    main()
