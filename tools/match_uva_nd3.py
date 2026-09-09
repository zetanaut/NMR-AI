#!/usr/bin/env python3
"""Jointly fit raw UVA-ND3 phase sweeps with an ND3 spin-1/full-circuit model.

The raw input retains its baseline. The same signal-on circuit provides the
complete prediction; chi=0 reconstructs its baseline after fitting. Stored
reference subtraction, DAQ polarization and TE calibration are not fit inputs.
Fixed circuit constants are declared tutorial assumptions, not ND3 measurements.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from circuit import Circuit, node_voltage, resonator_impedance
from lineshape import pake_susceptibility
from match_experimental_signals import digest, project_readout, residual_diagnostics
from uva_nd3_data import (confirmed_frequency, linear_resample, load_nd3,
                          sampling_contract, sampling_frequency)

ROOT = Path(__file__).resolve().parents[1]
VERSION = "uva-nd3-raw-qmeter-pake-500-v3"
NAMES = ("center_mhz", "split_mhz", "g", "eta", "P_model", "log10_signal_scale_cgs",
         "log10_tune_capacitance_pf", "phase_slope_rad_per_mhz", "phase_curvature_rad_per_mhz2")


def validate_config(config):
    if config.get("input_channel") != "phase":
        raise ValueError("The ND3 full-circuit example requires raw phase input")
    if set(config["fit_parameters"]) != set(NAMES):
        raise ValueError("Declare every ND3 lineshape and electronics coordinate")
    for name in NAMES:
        spec = config["fit_parameters"][name]
        lo, hi = spec["bounds"]
        if (not spec.get("source") or not np.isfinite([lo, hi, spec["initial"]]).all()
                or not lo < hi or not lo <= spec["initial"] <= hi):
            raise ValueError(f"{name}: require sourced finite bounds and an initial value inside them")
    specs = config["fit_parameters"]
    if specs["P_model"]["bounds"][0] < -1 or specs["P_model"]["bounds"][1] > 1:
        raise ValueError("Polarization bounds must remain in [-1,1]")
    if specs["eta"]["bounds"][0] < 0 or specs["eta"]["bounds"][1] > 1:
        raise ValueError("EFG asymmetry bounds must remain in [0,1]")
    if min(specs[n]["bounds"][0] for n in ("center_mhz", "split_mhz", "g")) <= 0:
        raise ValueError("Require positive frequency, splitting and broadening")
    nominal = config["nominal_circuit"]
    if not nominal.get("source") or nominal.get("status") != "tutorial assumptions; not measured ND3 hardware":
        raise ValueError("Identify nominal ND3 circuit assumptions explicitly")
    return Circuit(**nominal["values"])


def susceptibility(f, parameters, nphi=32):
    """Single-site ND3 spin-temperature response, cgs; vanishes at P=0."""
    x = (np.asarray(f)-parameters["center_mhz"])/parameters["split_mhz"]
    return 10**parameters["log10_signal_scale_cgs"] * pake_susceptibility(
        x, parameters["P_model"], parameters["eta"], parameters["g"], nphi=nphi)[0]


def node_basis(f, c, parameters, signal_on=True, nphi=32, source_frequency=None):
    """Full signal-on circuit or its chi=0 baseline, then the sampling operator."""
    if source_frequency is not None:
        return linear_resample(source_frequency, node_basis(source_frequency, c, parameters, signal_on, nphi), f)
    f = np.asarray(f, dtype=float)
    c = replace(c, tune_capacitance_f=10**parameters["log10_tune_capacitance_pf"]*1e-12)
    chi = susceptibility(f, parameters, nphi) if signal_on else 0.
    u = node_voltage(resonator_impedance(f*1e6, c, chi), c.drive_v,
                     c.drive_resistance_ohm, c.input_resistance_ohm)
    df = f-c.reference_hz/1e6
    phase = parameters["phase_slope_rad_per_mhz"]*df + parameters["phase_curvature_rad_per_mhz2"]*df**2
    u = u*np.exp(1j*phase)
    return np.column_stack([u.real, u.imag, np.ones_like(f)])


def reconstruct(f, fit, background_only=False, nphi=32):
    if not np.array_equal(f, confirmed_frequency()):
        raise ValueError("Require the exact confirmed 500-bin frequency grid")
    source = sampling_frequency(fit["sampling"]) if fit.get("sampling") else None
    design = node_basis(f, Circuit(**fit["fixed_circuit"]), fit["parameters"],
                        not background_only, nphi, source)
    return design@np.asarray(fit["quadrature_coefficients"])


def match_scan(f, y, config, starts=6, seed=42, source_record_1based=None):
    c = validate_config(config)
    f, y = np.asarray(f), np.asarray(y)
    if not np.array_equal(f, confirmed_frequency()) or y.shape != f.shape or not np.isfinite(y).all():
        raise ValueError("Require all 500 finite raw samples on the exact confirmed frequency grid")
    if starts < 1:
        raise ValueError("Need at least one start")
    sampling = sampling_contract(source_record_1based) if source_record_1based is not None else None
    source = sampling_frequency(sampling) if sampling else None
    specs = config["fit_parameters"]
    low = np.array([specs[n]["bounds"][0] for n in NAMES])
    span = np.array([specs[n]["bounds"][1]-specs[n]["bounds"][0] for n in NAMES])
    initial = np.array([specs[n]["initial"] for n in NAMES])
    decode = lambda q: dict(zip(NAMES, low+span*q))
    scale = max(float(np.ptp(y)), 1e-12)
    lo, hi = config["initial_signal_window_mhz"]
    wing = (f < lo) | (f > hi)
    if wing.sum() < 20:
        raise ValueError("Need at least 20 initialization wing bins")
    all_bins = np.ones(len(f), dtype=bool)

    def solve(q, signal_on=True, mask=all_bins):
        design = node_basis(f, c, decode(q), signal_on, source_frequency=source)
        coefficients = project_readout(design, y, mask)
        return design@coefficients, coefficients

    q0 = (initial-low)/span
    def initialize_electronics(q):
        full = q0.copy()
        full[-3:] = q
        return (solve(full, False, wing)[0]-y)[wing]/scale
    init = least_squares(initialize_electronics, q0[-3:], bounds=(0., 1.), jac="3-point", diff_step=1e-4,
                         max_nfev=config["max_nfev"], ftol=1e-11, xtol=1e-11, gtol=1e-11)
    q0[-3:] = init.x

    def objective(q):
        return (solve(q)[0]-y)/scale

    rng = np.random.default_rng(seed)
    guesses = [q0.copy()]
    for i in range(starts-1):
        guess = initial.copy()
        guess[0] += rng.uniform(-.003, .003)
        guess[1] *= rng.uniform(.94, 1.06)
        guess[2] *= rng.uniform(.5, 1.8)
        guess[3] = rng.uniform(.01, .15)
        guess[4] = .35 if i%2 == 0 else -.35
        guess[5] += rng.uniform(-.5, .5)
        q = np.clip((guess-low)/span, 1e-7, 1-1e-7)
        q[-3:] = q0[-3:]
        guesses.append(q)
    candidates, best = [], None
    for q in guesses:
        result = least_squares(objective, q, bounds=(0., 1.), jac="3-point", diff_step=1e-4,
                               max_nfev=config["max_nfev"], ftol=1e-11, xtol=1e-11, gtol=1e-11)
        sse = float(result.fun@result.fun)
        candidates.append({"parameters": decode(result.x), "range_normalized_sse": sse,
                           "quadrature_coefficients": solve(result.x)[1].tolist(),
                           "success": bool(result.success), "evaluations": int(result.nfev)})
        if best is None or sse < float(best.fun@best.fun):
            best = result
    p = decode(best.x)
    prediction, coefficients = solve(best.x)
    background = node_basis(f, c, p, False, source_frequency=source)@coefficients
    condition = float(np.linalg.cond(best.jac))
    diagnostics = residual_diagnostics(y-prediction, f, wing)
    if sampling:
        for key in ("wing_second_difference_sigma_proxy", "sigma_proxy_triplets", "sigma_proxy_assumption"):
            diagnostics.pop(key)
        diagnostics["noise_status"] = "Linear resampling correlates neighboring errors; no iid-noise sigma proxy or measured covariance is inferred from this raw-fit residual"
    fit = {"model_version": VERSION, "fit_kind": "joint raw ND3 spin-temperature full-circuit fit",
           "input_channel": "phase", "stored_reference_used_in_fit": False, "added_baseline": False,
           "parameters": p, "fixed_circuit": asdict(c), "quadrature_coefficients": coefficients.tolist(),
           "profiled_readout_parameters": ["a", "b", "d"], "n_free": 12,
           "resolved_tune_capacitance_pf": float(10**p["log10_tune_capacitance_pf"]),
           "nonlinear_parameters": list(NAMES), "all_500_bins_fitted": True, "sampling": sampling,
           "initialization_wing_mask": wing.tolist(), "initialization_only_mask": True,
           "initialization_success": bool(init.success), "success": bool(best.success), "candidates": candidates,
           "near_bounds": {n: "lower" if q <= 1e-4 else "upper" for n, q in zip(NAMES, best.x) if min(q, 1-q) <= 1e-4},
           "scaled_profiled_jacobian_condition": condition if np.isfinite(condition) else None,
           "diagnostics": diagnostics,
           "P_status": "Conditional ND3 spin-temperature estimate from raw phase; no TE calibration or independently established P truth",
           "hardware_status": config["nominal_circuit"]["status"],
           "baseline_definition": "Same fitted circuit/readout with chi=0; nuclear signal is full fit minus this baseline"}
    return np.column_stack([f, y, prediction, background, prediction-background, y-prediction]), fit


def fit_record(record, config, starts=6, seed=42):
    """Select raw phase explicitly; reference/basesub are provenance only."""
    identifier = record["source_record_1based"]
    trace, fit = match_scan(record["frequency_mhz"], record["phase"], config, starts, seed+identifier,
                            source_record_1based=identifier)
    fit.update(source_record_1based=identifier, start_time=record["start_time"], sweeps=record["sweeps"])
    return trace, fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"examples/uva-nd3.json")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/uva-nd3-matching.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a fresh output directory")
    data, audit = load_nd3(args.data)
    config = json.loads(args.config.read_text())
    validate_config(config)
    args.output_dir.mkdir(parents=True)
    fits = []
    for record in data["records"]:
        trace, fit = fit_record(record, config, args.starts, args.seed)
        i = record["source_record_1based"]
        fits.append(fit)
        np.savetxt(args.output_dir/f"record_{i}.csv", trace, delimiter=",", comments="",
                   header="frequency_mhz,raw_phase,fitted,baseline,signal,residual")
        print(f"UVA-ND3 record {i}: P={fit['parameters']['P_model']:.6f}, RMS={fit['diagnostics']['rms_recorded_units']:.6g}, bounds={fit['near_bounds']}", flush=True)
    report = {"version": VERSION, "dataset": "UVA-ND3 data", "material": "ND3", "audit": audit, "config": config,
              "config_sha256": digest(args.config), "fits": fits, "starts_per_scan": args.starts, "seed": args.seed,
              "input_channel": "phase", "stored_reference_used_in_fit": False, "added_baseline": False,
              "unit": "recorded units", "residual_sign": "resampled raw phase minus full resampled circuit fit",
              "code_sha256": {n: digest(ROOT/"tools"/n) for n in ("match_uva_nd3.py", "uva_nd3_data.py", "prepare_uva_nd3.py", "baseline_data.py", "circuit.py", "lineshape.py", "match_experimental_signals.py")}}
    (args.output_dir/"uva_nd3_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
