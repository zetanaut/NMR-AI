#!/usr/bin/env python3
"""Generate 500-bin training examples anchored to fitted experimental lineshapes.

Sample NEW simulator-known polarizations. Experimental P estimates are never
copied into training labels. Parameter excursions are controlled sensitivity
ranges, not an empirical population inferred from five scans.
"""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from circuit import Circuit, cable_parameters, detector_voltage
from fit_tuned_baseline import calibrated_cable_lc
from lineshape import pake_susceptibility
from match_experimental_signals import VERSION, ROOT, digest


GENERATOR = "experiment-anchored-qmeter-pake-500-v1"
DEFAULT_COVERAGE = ROOT/"configs/matched-generator-coverage.json"


def validated_report(path):
    report = json.loads(Path(path).read_text())
    if report.get("version") != VERSION or not report.get("fits"):
        raise ValueError("Require a complete current experimental matching report")
    if not report["readiness"]["signal_grid_confirmed"]:
        raise ValueError("Confirm the acquisition grid before generating frequency-labeled examples")
    for name, expected in report["code_sha256"].items():
        if digest(ROOT/"tools"/name) != expected:
            raise ValueError(f"Stale matching code: {name}; regenerate the fit report")
    if any(not fit["success"] or fit["near_bounds"] for fit in report["fits"]):
        raise ValueError("Resolve nonconverged or bound-limited selected fits before using them as generator seeds")
    return report


def covariance_factor(covariance):
    cov = np.asarray(covariance, dtype=float)
    if cov.shape != (500, 500) or not np.isfinite(cov).all():
        raise ValueError("Supply a finite 500x500 covariance in recorded-units squared")
    scale = max(float(np.max(np.abs(cov))), 1e-30)
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=scale*1e-12):
        raise ValueError("Covariance must be symmetric")
    values, vectors = np.linalg.eigh(cov)
    if values.min() < -scale*1e-10:
        raise ValueError("Covariance must be positive semidefinite")
    return vectors*np.sqrt(np.maximum(values, 0))


def seed_configuration(fit):
    c = Circuit(**fit["fixed_circuit"])
    p = fit["parameters"]
    a, b, d = fit["quadrature_coefficients"]
    c = replace(c, tune_capacitance_f=10**p["log10_tune_capacitance_pf"]*1e-12,
                susceptibility_scale_cgs=10**p["log10_signal_scale_cgs"],
                detector_gain=float(np.hypot(a, b)), detector_phase_rad=float(np.arctan2(-b, a)),
                dc_offset_v=float(d), detector_phase_slope_rad_per_hz=p["phase_slope_rad_per_mhz"]*1e-6,
                detector_phase_curvature_rad_per_hz2=p["phase_curvature_rad_per_mhz2"]*1e-12)
    return {"source_scan_1based": fit["record_1based"], "circuit": asdict(c),
            **{key: p[key] for key in ("center_mhz", "split_mhz", "g", "eta")},
            "noise_sigma_recorded": fit["diagnostics"]["wing_second_difference_sigma_proxy"]}


def validate_coverage(coverage):
    if not coverage.get("source") or coverage["reference_averages"] < 1 or not isinstance(coverage["reference_averages"], int):
        raise ValueError("Record coverage source and a positive integer reference-averaging count")
    allowed = {"drive_v", "drive_resistance_ohm", "input_resistance_ohm", "damping_resistance_ohm",
               "coil_resistance_ohm", "coil_inductance_h", "tune_capacitance_f", "susceptibility_scale_cgs",
               "detector_gain", "cable_resistance_ohm_per_m"}
    fractional = coverage["circuit_fractional_half_ranges"]
    if set(fractional)-allowed or any(not np.isfinite(v) or not 0 <= v < 1 for v in fractional.values()):
        raise ValueError("Unsupported circuit coverage or invalid fractional half-range")
    positive = [coverage[k] for k in ("center_half_range_mhz", "split_fractional_half_range", "g_fractional_half_range",
                                     "eta_half_range", "phase_half_range_rad", "phase_slope_half_range_rad_per_mhz",
                                     "phase_curvature_half_range_rad_per_mhz2", "offset_half_range_recorded", "cable_trim_half_range_m")]
    if not np.isfinite(positive).all() or min(positive) < 0:
        raise ValueError("Coverage half-ranges must be finite and nonnegative")
    if max(coverage["split_fractional_half_range"], coverage["g_fractional_half_range"]) >= 1:
        raise ValueError("Splitting and width must remain positive")
    if coverage["cable_trim_half_range_m"] > .03*3.58:
        raise ValueError("Do not cross the established +/-3% cable trim interval")


def vary_configuration(seed, coverage, rng):
    validate_coverage(coverage)
    result = deepcopy(seed)
    c = result["circuit"]
    for name, width in coverage["circuit_fractional_half_ranges"].items():
        c[name] *= 1+rng.uniform(-width, width)
    c["cable_length_m"] = 3.58+rng.uniform(-coverage["cable_trim_half_range_m"], coverage["cable_trim_half_range_m"])
    # Changing conductor loss requires recomputing L/C together to preserve the
    # confirmed propagation scale. Do not draw independent L/C/VF parameters.
    inductance, capacitance = calibrated_cable_lc(c["reference_hz"], 3.58, resistance_ohm_per_m=c["cable_resistance_ohm_per_m"])
    c.update(cable_inductance_h_per_m=inductance, cable_capacitance_f_per_m=capacitance)
    for name, scale, width_name in (
            ("detector_phase_rad", 1., "phase_half_range_rad"),
            ("detector_phase_slope_rad_per_hz", 1e-6, "phase_slope_half_range_rad_per_mhz"),
            ("detector_phase_curvature_rad_per_hz2", 1e-12, "phase_curvature_half_range_rad_per_mhz2"),
            ("dc_offset_v", 1., "offset_half_range_recorded")):
        c[name] += scale*rng.uniform(-coverage[width_name], coverage[width_name])
    result["center_mhz"] += rng.uniform(-coverage["center_half_range_mhz"], coverage["center_half_range_mhz"])
    for name, key in (("split_mhz", "split_fractional_half_range"), ("g", "g_fractional_half_range")):
        result[name] *= 1+rng.uniform(-coverage[key], coverage[key])
    result["eta"] = float(rng.uniform(max(0., result["eta"]-coverage["eta_half_range"]),
                                    min(1., result["eta"]+coverage["eta_half_range"])))
    Circuit(**c)
    return result


def clean_spectrum(frequency_mhz, polarization, config):
    c = Circuit(**config["circuit"])
    f = np.asarray(frequency_mhz)
    chi = c.susceptibility_scale_cgs*pake_susceptibility((f-config["center_mhz"])/config["split_mhz"],
                                                     polarization, config["eta"], config["g"])[0]
    baseline = detector_voltage(f*1e6, c)
    raw = detector_voltage(f*1e6, c, chi)
    return raw, baseline, raw-baseline


def generate(report, count, configurations, seed, p_min, p_max, coverage, noise_covariance=None):
    if count < configurations or configurations < len(report["fits"]) or not -1 <= p_min < p_max <= 1:
        raise ValueError("Need samples >= configurations >= fitted scans, and -1 <= P_min < P_max <= 1")
    validate_coverage(coverage)
    rng = np.random.default_rng(seed)
    fit_seeds = [seed_configuration(fit) for fit in report["fits"]]
    configs = [vary_configuration(fit_seeds[i % len(fit_seeds)], coverage, rng) for i in range(configurations)]
    acquisition = report["acquisition"]
    f = acquisition["start_mhz"]+np.arange(500)*acquisition["step_mhz"]
    factor = covariance_factor(noise_covariance) if noise_covariance is not None else None
    groups = np.arange(count) % configurations
    rng.shuffle(groups)
    labels = rng.uniform(p_min, p_max, count)
    references = []
    for config in configs:
        _, base, _ = clean_spectrum(f, 0, config)
        noise = (factor@rng.normal(size=500) if factor is not None else rng.normal(0, config["noise_sigma_recorded"], 500))
        references.append(base+noise/np.sqrt(coverage["reference_averages"]))
    clean, baselines, signal, noise = [np.empty((count, 500), dtype=np.float64) for _ in range(4)]
    for i, (p, group) in enumerate(zip(labels, groups)):
        config = configs[group]
        clean[i], baselines[i], signal[i] = clean_spectrum(f, p, config)
        noise[i] = factor@rng.normal(size=500) if factor is not None else rng.normal(0, config["noise_sigma_recorded"], 500)
    data = {"signals": clean+noise, "clean_raw": clean, "clean_baseline": baselines, "clean_signal": signal,
            "noise": noise, "baselines": np.asarray(references)[groups], "P": labels,
            "configuration_id": groups, "source_scan_1based": np.asarray([configs[g]["source_scan_1based"] for g in groups]),
            "frequency_mhz": f, "simulator": GENERATOR, "label_source": "newly sampled simulator truth, not fitted experimental P",
            "voltage_unit": "recorded units", "polarization_method": "spin-1 lineshape; no TE normalization"}
    return data, configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--num-configurations", type=int, default=100)
    parser.add_argument("--p-min", type=float, default=-.6)
    parser.add_argument("--p-max", type=float, default=.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-covariance", type=Path, help="Optional measured 500x500 covariance in recorded-units squared")
    args = parser.parse_args()
    if args.output.suffix != '.npz' or args.output.exists() or args.output.with_suffix('.json').exists():
        parser.error("Choose a new .npz output path")
    report = validated_report(args.matching_report)
    coverage = json.loads(args.coverage.read_text())
    cov = np.load(args.noise_covariance, allow_pickle=False) if args.noise_covariance else None
    data, configs = generate(report, args.num_samples, args.num_configurations, args.seed, args.p_min, args.p_max, coverage, cov)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    metadata = {"version": GENERATOR, "matching_report_sha256": digest(args.matching_report),
                "source_csv_sha256": report["signal_parsing"]["source_sha256"], "dataset_sha256": digest(args.output),
                "generator_sha256": digest(__file__), "coverage_sha256": digest(args.coverage), "coverage": coverage,
                "seed": args.seed, "num_samples": args.num_samples, "configurations": configs,
                "P_interval": [args.p_min, args.p_max], "acquisition": report["acquisition"],
                "label_source": "New uniform draws in fractional P; fitted scan polarizations are NOT ground-truth training labels",
                "noise_model": "Supplied correlated Gaussian covariance" if cov is not None else "White Gaussian reference with per-seed high-frequency sigma proxy",
                "noise_covariance_sha256": digest(args.noise_covariance) if cov is not None else None,
                "units": "Recorded units, NOT volts; no TE calibration or area normalization",
                "limitations": ["Five development scans do not define a measured population distribution",
                                "White-noise reference does not reproduce all structured fit residuals or endpoint artifacts",
                                "All descendants of a source scan must stay together in experimental holdout splits",
                                "Lineshape preprocessing uses raw/reference recorded units without the optional TE-area benchmark calibration",
                                "Simulation test accuracy is not a measured experimental polarization uncertainty"]}
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    print(f"Saved {args.num_samples} new simulated sweeps, 500 bins, with simulator-known P and independent baseline-reference noise")


if __name__ == "__main__":
    main()
