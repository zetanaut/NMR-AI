#!/usr/bin/env python3
"""Fit recorded Q-curves using the same passive circuit as the generator.

Reads the supplied headerless timestamp + 500-bin CSV without discarding row 0.
Amplitude, phase and DC offset are solved as linear quadratures; three physical
shape parameters are fitted with bounded multi-start least squares. No filling
factor is fitted to a chi=0 trace. Raw measurements are never copied to docs/.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from circuit import Circuit, node_voltage, resonator_impedance


def fit_trace(frequency_hz, values, starts=24, seed=42):
    frequency_hz, values = np.asarray(frequency_hz, dtype=float), np.asarray(values, dtype=float)
    if (frequency_hz.ndim != 1 or values.shape != frequency_hz.shape or len(values) < 7
            or not np.isfinite(frequency_hz).all() or not np.isfinite(values).all()
            or np.any(frequency_hz <= 0) or np.any(np.diff(frequency_hz) <= 0) or starts < 1):
        raise ValueError("Need aligned finite traces on an increasing positive grid and positive starts")
    base = Circuit(reference_hz=32.68e6)
    if not frequency_hz[0] <= base.reference_hz <= frequency_hz[-1]:
        raise ValueError("This deuteron practical requires a sweep around 32.68 MHz; check acquisition metadata")
    scale = max(float(np.ptp(values)), 1e-12)
    target = (values-values.mean())/scale

    def evaluate(parameters):
        capacitance_pf, length_m, stray_pf = parameters
        c = replace(base, tune_capacitance_f=capacitance_pf*1e-12,
                    cable_length_m=length_m, stray_capacitance_f=stray_pf*1e-12)
        z = resonator_impedance(frequency_hz, c)
        voltage = node_voltage(z, c.drive_v, c.drive_resistance_ohm, c.input_resistance_ohm)
        design = np.column_stack([voltage.real, voltage.imag, np.ones(len(values))])
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
        return design@coefficients, coefficients, c

    rng = np.random.default_rng(seed)
    # Search physical F/m coordinates, not the old script's knob/trim variables.
    seeds = [[100.0, 4.0, 50.0]]
    seeds += [[rng.uniform(10, 590), rng.uniform(3.0, 5.0), rng.uniform(0, 400)] for _ in range(starts-1)]
    best, candidates = None, []
    for initial in seeds:
        fit = least_squares(lambda p: evaluate(p)[0]-target, initial,
                            bounds=([0.2, 3.0, 0], [600, 5.0, 400]),
                            x_scale=[100, 1, 100], max_nfev=600,
                            ftol=1e-11, xtol=1e-11, gtol=1e-11)
        candidates.append({"parameters_pf_m_pf": fit.x.tolist(), "normalized_sse": float(np.sum(fit.fun**2)),
                           "success": bool(fit.success), "evaluations": int(fit.nfev)})
        if best is None or np.sum(fit.fun**2) < np.sum(best.fun**2):
            best = fit
    fitted, coefficient, c = evaluate(best.x)
    prediction = fitted*scale + values.mean()
    coefficient *= scale
    coefficient[2] += values.mean()
    residual = values-prediction
    singular = np.linalg.svd(best.jac*np.array([100, 1, 100]), compute_uv=False)
    return prediction, {
        "circuit": asdict(c), "shape_fit_success": bool(best.success),
        "recorded_units_per_node_volt": float(np.hypot(*coefficient[:2])),
        "detector_phase_rad": float(np.arctan2(-coefficient[1], coefficient[0])),
        "dc_offset_recorded_units": float(coefficient[2]),
        "quadrature_coefficients": coefficient.tolist(),
        "rmse_recorded_units": float(np.sqrt(np.mean(residual**2))),
        "rmse_over_trace_range": float(np.sqrt(np.mean(residual**2))/scale),
        "residual_lag1": float(np.corrcoef(residual[:-1], residual[1:])[0, 1]),
        "scaled_shape_jacobian_condition": float(singular[0]/max(singular[-1], 1e-30)),
        "multistart_candidates": candidates,
        "active_bounds": best.active_mask.tolist(),
        "note": "A narrow sweep does not uniquely identify individual components. Amplitude is an effective "
                "DAQ scale, not measured RF drive or susceptibility calibration. Residuals include model mismatch.",
    }


def fitted_curve(frequency_hz, report):
    """Reconstruct a saved fit in recorded units, without rerunning optimization."""
    c = Circuit(**report["circuit"])
    z = resonator_impedance(frequency_hz, c)
    u = node_voltage(z, c.drive_v, c.drive_resistance_ohm, c.input_resistance_ohm)
    a, b, offset = report["quadrature_coefficients"]
    return a*u.real+b*u.imag+offset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-mhz", type=float, required=True, help="Actual acquisition start, not inferred from the center")
    parser.add_argument("--step-mhz", type=float, required=True, help="Actual acquisition spacing between bins")
    parser.add_argument("--frequency-source", required=True, help="Acquisition record establishing the supplied frequency grid")
    parser.add_argument("--starts", type=int, default=24)
    parser.add_argument("--volts-per-unit", type=float, help="Confirmed DAQ conversion; omit if unknown")
    args = parser.parse_args()
    if not args.frequency_source.strip():
        parser.error("Record the source of the actual deuteron frequency grid")
    if (args.starts < 1 or not np.isfinite([args.start_mhz, args.step_mhz]).all()
            or args.start_mhz <= 0 or args.step_mhz <= 0
            or (args.volts_per_unit is not None and (not np.isfinite(args.volts_per_unit) or args.volts_per_unit <= 0))):
        parser.error("Require positive starts, frequency step, and any supplied unit conversion")
    if args.output_dir.exists():
        parser.error("Choose a new output directory")
    records = np.loadtxt(args.data, delimiter=",", ndmin=2)
    if records.shape[1] != 501 or not np.isfinite(records).all():
        parser.error("Expected finite rows: timestamp followed by 500 values, no header")
    frequency = (args.start_mhz + np.arange(500)*args.step_mhz)*1e6
    if not frequency[0] <= 32.68e6 <= frequency[-1]:
        parser.error("The acquisition grid must bracket the approximate 32.68 MHz deuteron reference")
    _, first = np.unique(records, axis=0, return_index=True)
    unique = records[np.sort(first)]
    args.output_dir.mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    reports = []
    for i, row in enumerate(unique):
        fitted, report = fit_trace(frequency, row[1:], args.starts)
        report["timestamp"] = float(row[0])
        if args.volts_per_unit is not None:
            c = report["circuit"].copy()
            c.update(detector_gain=report["recorded_units_per_node_volt"]*args.volts_per_unit,
                     detector_phase_rad=report["detector_phase_rad"],
                     dc_offset_v=report["dc_offset_recorded_units"]*args.volts_per_unit)
            report["circuit_in_confirmed_volts"] = c
        reports.append(report)
        np.savetxt(args.output_dir/f"trace_{i}.csv", np.column_stack([frequency/1e6, row[1:], fitted, row[1:]-fitted]),
                   delimiter=",", header="frequency_mhz,recorded,fit,residual", comments="")
        axes[0].plot(frequency/1e6, row[1:], lw=1, label=f"Trace {i+1}")
        axes[0].plot(frequency/1e6, fitted, "k--", lw=0.8)
        axes[1].plot(frequency/1e6, row[1:]-fitted, lw=0.8)
        print(f"trace {i+1}: normalized RMSE={report['rmse_over_trace_range']:.6g}, "
              f"residual lag1={report['residual_lag1']:.3f}", flush=True)
    axes[0].set_ylabel("Recorded units (conversion separate)")
    axes[1].set_ylabel("Recorded minus fitted")
    axes[1].set_xlabel("Frequency (MHz)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(args.output_dir/"baseline_fits.png", dpi=160)
    import scipy
    output = {"source_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("fit_baseline.py", "circuit.py")},
              "paper": "https://arxiv.org/abs/2603.10146v5", "starts": args.starts, "seed": 42,
              "numpy_version": np.__version__, "scipy_version": scipy.__version__,
              "loss": "Unweighted least squares; no verified measurement-error covariance supplied",
              "fitted_bin_mask": "All 500 bins; no exclusions", "residual_sign": "recorded minus fitted",
              "bounds_pf_m_pf": [[0.2, 3.0, 0], [600, 5.0, 400]],
              "rows": len(records), "unique_rows": len(unique), "volts_per_recorded_unit": args.volts_per_unit,
              "nucleus": "deuteron", "reference_hz": 32.68e6,
              "frequency_mapping": "start_mhz + bin*step_mhz; both supplied explicitly",
              "frequency_source": args.frequency_source,
              "start_mhz": args.start_mhz, "step_mhz": args.step_mhz, "fits": reports}
    (args.output_dir/"fit_report.json").write_text(json.dumps(output, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
