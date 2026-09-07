#!/usr/bin/env python3
"""Conditional experimental matching with the complex spin-1 Q-meter model.

Polarization is inferred from the spin-1 lineshape; TE calibration is not needed.
No polynomial voltage background or two-Gaussian substitute is used.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from baseline_data import load_acquisition, acquisition_grid, load_baseline_csv
from experimental_data import load_signal_csv
from circuit import Circuit, resonator_impedance, node_voltage
from fit_tuned_baseline import validate_setup, resolved_circuit
from lineshape import pake_susceptibility

ROOT = Path(__file__).resolve().parents[1]
VERSION = "conditional-experimental-qmeter-pake-500-v1"
DEFAULT_MATCHING = ROOT/"configs/experimental-matching.json"
NAMES = ("center_mhz", "split_mhz", "g", "eta", "P_model", "log10_signal_scale_cgs",
         "log10_tune_capacitance_pf", "phase_slope_rad_per_mhz", "phase_curvature_rad_per_mhz2")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conditional_circuit(config):
    """Apply known cable setup and explicit nominal component assumptions."""
    setup = json.loads((ROOT/config["component_setup"]).read_text())
    parameters = validate_setup(setup)
    parameters["cable_delta_length_m"] = 0.
    parameters["stray_capacitance_f"] = 0.
    c = resolved_circuit(parameters, setup)
    c = replace(c, tune_capacitance_f=10**config["fit_parameters"]["log10_tune_capacitance_pf"]["initial"]*1e-12,
                detector_gain=1., detector_phase_rad=0., dc_offset_v=0., susceptibility_scale_cgs=1.)
    return c


def validate_matching(config):
    if set(config["fit_parameters"]) != set(NAMES):
        raise ValueError("Declare every conditional matching parameter exactly once")
    for name in NAMES:
        spec = config["fit_parameters"][name]
        low, high = spec["bounds"]
        if not spec.get("source") or not np.isfinite([low, high, spec["initial"]]).all() or not low < high:
            raise ValueError(f"{name}: require source and finite increasing bounds")
        if not low <= spec["initial"] <= high:
            raise ValueError(f"{name}: initial must lie inside bounds")
    p = config["fit_parameters"]
    if p["P_model"]["bounds"][0] < -1 or p["P_model"]["bounds"][1] > 1:
        raise ValueError("Polarization model bounds must stay inside [-1,1]")
    if p["eta"]["bounds"][0] < 0 or p["eta"]["bounds"][1] > 1:
        raise ValueError("EFG asymmetry bounds must stay inside [0,1]")
    if min(p[n]["bounds"][0] for n in ("center_mhz", "split_mhz", "g")) <= 0:
        raise ValueError("Positive center, splitting and broadening required")
    if config["grid_status"] not in ("provisional", "confirmed"):
        raise ValueError("Record whether the signal acquisition mapping is confirmed")
    return conditional_circuit(config)


def node_basis(frequency_mhz, c, parameters, signal_on=True):
    f = np.asarray(frequency_mhz, dtype=float)
    c = replace(c, tune_capacitance_f=10**parameters["log10_tune_capacitance_pf"]*1e-12)
    chi = 0.
    if signal_on:
        x = (f-parameters["center_mhz"])/parameters["split_mhz"]
        chi = 10**parameters["log10_signal_scale_cgs"] * pake_susceptibility(
            x, parameters["P_model"], parameters["eta"], parameters["g"], nphi=32)[0]
    u = node_voltage(resonator_impedance(f*1e6, c, chi), c.drive_v,
                     c.drive_resistance_ohm, c.input_resistance_ohm)
    df = f-c.reference_hz/1e6
    phase = parameters["phase_slope_rad_per_mhz"]*df + parameters["phase_curvature_rad_per_mhz2"]*df**2
    u = u*np.exp(1j*phase)
    return np.column_stack([u.real, u.imag, np.ones_like(f)])


def project_readout(basis, observed, mask):
    coefficients, _, rank, singular = np.linalg.lstsq(basis[mask], observed[mask], rcond=None)
    if rank != 3:
        raise ValueError("Rank-deficient readout; independent gain/phase information is needed")
    return coefficients


def residual_diagnostics(residual, frequency_mhz, wing_mask):
    r = np.asarray(residual)
    # Only consecutive triplets within the specified wings enter this proxy.
    triplet = wing_mask[:-2] & wing_mask[1:-1] & wing_mask[2:]
    second = np.diff(r, n=2)[triplet]
    sigma = float(np.median(np.abs(second-np.median(second)))/(.6744897501960817*np.sqrt(6)))
    return {
        "rms_recorded_units": float(np.sqrt(np.mean(r*r))),
        "max_absolute_recorded_units": float(np.max(np.abs(r))),
        "lag1": float(np.corrcoef(r[:-1], r[1:])[0, 1]) if r.std() > 0 else None,
        "wing_second_difference_sigma_proxy": sigma,
        "sigma_proxy_triplets": int(triplet.sum()),
        "sigma_proxy_assumption": "MAD(second differences)/(0.67449 sqrt(6)); iid Gaussian noise and locally smooth mean. Not a validated noise covariance.",
    }


def match_scan(frequency_mhz, observed, config, starts=6, seed=42, baseline_only=False):
    c = validate_matching(config)
    f, y = np.asarray(frequency_mhz), np.asarray(observed)
    if f.shape != (500,) or y.shape != (500,) or not np.isfinite(y).all() or not np.isfinite(f).all() or np.any(np.diff(f) <= 0):
        raise ValueError("Require 500 finite values on an increasing frequency grid")
    if starts < 1:
        raise ValueError("Need positive starts")
    specs = config["fit_parameters"]
    low = np.array([specs[n]["bounds"][0] for n in NAMES])
    span = np.array([specs[n]["bounds"][1]-specs[n]["bounds"][0] for n in NAMES])
    initial = np.array([specs[n]["initial"] for n in NAMES])
    decode = lambda q: dict(zip(NAMES, low+span*q))
    wing = (f < config["initial_signal_window_mhz"][0]) | (f > config["initial_signal_window_mhz"][1])
    if wing.sum() < 20:
        raise ValueError("Need at least 20 initialization wing bins")
    scale = max(float(np.ptp(y)), 1e-12)

    def solve(q, signal_on, mask):
        parameters = decode(q)
        basis = node_basis(f, c, parameters, signal_on)
        coefficients = project_readout(basis, y, mask)
        return basis@coefficients, coefficients

    # Wing-only baseline is an initialization, never the final signal mask.
    q0 = (initial-low)/span
    init_mask = np.ones(500, dtype=bool) if baseline_only else wing
    def phase_residual(q):
        full = q0.copy()
        full[-3:] = q
        return (solve(full, False, init_mask)[0]-y)[init_mask]/scale
    phase = least_squares(phase_residual, q0[-3:], bounds=(0., 1.), jac="3-point", diff_step=1e-4,
                          max_nfev=500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
    q0[-3:] = phase.x
    mask = np.ones(500, dtype=bool)
    candidates = []
    if baseline_only:
        best_q, success, condition = q0, bool(phase.success), float(np.linalg.cond(phase.jac))
        optimized_names = list(NAMES[-3:])
        active = phase.active_mask.tolist()
    else:
        def objective(q):
            return (solve(q, True, mask)[0]-y)/scale
        rng = np.random.default_rng(seed)
        guesses = [q0.copy()]
        for i in range(starts-1):
            physical = initial.copy()
            physical[0] += rng.uniform(-.002, .002)
            physical[1] *= rng.uniform(.95, 1.05)
            physical[2] *= rng.uniform(.6, 1.5)
            physical[4] = (.3 if i % 2 else -.3)
            physical[5] += rng.uniform(-.5, .5)
            q = np.clip((physical-low)/span, 1e-6, 1-1e-6)
            q[-3:] = q0[-3:]
            guesses.append(q)
        best = None
        for q in guesses:
            result = least_squares(objective, q, bounds=(0., 1.), jac="3-point", diff_step=1e-4,
                                   max_nfev=700, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            sse = float(result.fun@result.fun)
            candidates.append({"parameters": decode(result.x), "range_normalized_sse": sse,
                               "quadrature_coefficients": solve(result.x, True, mask)[1].tolist(),
                               "success": bool(result.success), "evaluations": int(result.nfev)})
            if best is None or sse < float(best.fun@best.fun):
                best = result
        best_q, success = best.x, bool(best.success)
        condition = float(np.linalg.cond(best.jac))
        optimized_names, active = list(NAMES), best.active_mask.tolist()
    parameters = decode(best_q)
    if baseline_only:
        parameters["P_model"] = 0.
    prediction, coefficients = solve(best_q, not baseline_only, mask)
    baseline = node_basis(f, c, parameters, False)@coefficients
    residual = y-prediction
    matching_candidates = [] if baseline_only else [item for item in candidates
        if item["success"] and item["range_normalized_sse"] <= min(k["range_normalized_sse"] for k in candidates)*1.01]
    near = {n: "lower" if best_q[i] <= 1e-4 else "upper" for i, n in enumerate(NAMES)
            if n in optimized_names and min(best_q[i], 1-best_q[i]) <= 1e-4}
    report = {
        "model_version": VERSION, "fit_kind": "baseline-only" if baseline_only else "conditional spin-temperature full-circuit fit",
        "fixed_circuit": asdict(c), "parameters": parameters, "quadrature_coefficients": coefficients.tolist(),
        "resolved_tune_capacitance_pf": float(10**parameters["log10_tune_capacitance_pf"]),
        "nonlinear_parameters": optimized_names, "profiled_readout_parameters": ["a", "b", "d"],
        "n_free": len(optimized_names)+3, "all_500_bins_fitted": True,
        "initialization_wing_mask": wing.tolist(), "initialization_only_mask": not baseline_only,
        "diagnostics": residual_diagnostics(residual, f, wing), "success": success,
        "active_bounds": dict(zip(optimized_names, active)), "near_bounds": near,
        "scaled_profiled_jacobian_condition": condition if np.isfinite(condition) else None,
        "near_best_P_range": ([float(min(k["parameters"]["P_model"] for k in matching_candidates)),
                               float(max(k["parameters"]["P_model"] for k in matching_candidates))]
                              if matching_candidates else None),
        "near_best_rule": "Successful starts within 1% of minimum SSE; optimizer agreement, not a statistical confidence interval",
        "candidates": candidates, "P_status": "not applicable" if baseline_only else "Spin-1 lineshape estimate; no TE calibration required. A fitted estimate is not an independent accuracy benchmark.",
    }
    return np.column_stack([f, y, prediction, baseline, prediction-baseline, residual]), report


def reconstruct(frequency_mhz, fit, signal_on=True):
    c = Circuit(**fit["fixed_circuit"])
    return node_basis(frequency_mhz, c, fit["parameters"], signal_on)@np.asarray(fit["quadrature_coefficients"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=ROOT/"examples/Sample_RawSignal.csv")
    parser.add_argument("--baseline", type=Path, default=ROOT/"examples/deuteron-baseline.csv")
    parser.add_argument("--config", type=Path, default=DEFAULT_MATCHING)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a fresh output directory")
    config = json.loads(args.config.read_text())
    validate_matching(config)
    acquisition = load_acquisition(ROOT/config["acquisition"])
    f = acquisition_grid(acquisition)/1e6
    signals, parsing = load_signal_csv(args.signals)
    base, base_parsing = load_baseline_csv(args.baseline)
    if len(base) != 1:
        parser.error("This practical expects exactly one baseline reference")
    args.output_dir.mkdir(parents=True)
    baseline_data, baseline_fit = match_scan(f, base[0, 1:], config, baseline_only=True)
    np.savetxt(args.output_dir/"baseline.csv", baseline_data, delimiter=",",
               header="frequency_mhz,recorded,fitted,baseline,signal,residual", comments="")
    fits = []
    for index, row in enumerate(signals):
        data, fit = match_scan(f, row[1:], config, args.starts, args.seed+index)
        fit["record_1based"] = index+1
        fits.append(fit)
        (args.output_dir/f"scan_{index+1}_fit.json").write_text(json.dumps(fit, indent=2, allow_nan=False)+"\n")
        np.savetxt(args.output_dir/f"scan_{index+1}.csv", data, delimiter=",",
                   header="frequency_mhz,recorded,fitted,baseline,signal,residual", comments="")
        print(f"scan {index+1}: P_model={fit['parameters']['P_model']:.5g}, RMS={fit['diagnostics']['rms_recorded_units']:.6g}, near bounds={fit['near_bounds']}", flush=True)
    import scipy
    report = {"version": VERSION, "matching_config": config, "acquisition": acquisition,
              "signal_parsing": parsing, "baseline_parsing": base_parsing, "fits": fits, "baseline_fit": baseline_fit,
              "config_sha256": digest(args.config), "component_setup_sha256": digest(ROOT/config["component_setup"]),
              "code_sha256": {name: digest(ROOT/"tools"/name) for name in
                              ("match_experimental_signals.py", "experimental_data.py", "circuit.py", "lineshape.py", "baseline_data.py", "fit_tuned_baseline.py")},
              "seed": args.seed, "starts_per_scan": args.starts, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
              "unit": "recorded units; no DAQ voltage conversion supplied", "residual_sign": "recorded minus fitted",
              "readiness": {"status": "model-development example", "polarization_method": "spin-1 lineshape branch ratio; TE not required",
                            "measured_noise_covariance": False, "held_out_experimental_validation": False,
                            "component_and_detector_calibration": False, "signal_grid_confirmed": config["grid_status"] == "confirmed"}}
    (args.output_dir/"matching_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
