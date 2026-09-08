#!/usr/bin/env python3
"""Compare a two-site butanol susceptibility with all five saved single-site fits."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from baseline_data import acquisition_grid, load_acquisition
from circuit import Circuit, node_voltage, resonator_impedance
from experimental_data import load_signal_csv
from material_lineshapes import butanol_components
from match_experimental_signals import NAMES as SINGLE_NAMES, digest, project_readout, residual_diagnostics

ROOT = Path(__file__).resolve().parents[1]
VERSION = "butanol-two-site-full-circuit-v1"
NAMES = SINGLE_NAMES + ("od_split_ratio", "eta_od", "od_fraction")


def susceptibility(f, p, nphi=128):
    # Keep the C-D amplitude convention of the existing single-site model.
    # Multiplying the density by split_cd gives integrated scale = scale_cgs*split_cd.
    split = p["split_mhz"]
    density, cd, od = butanol_components(
        f, p["P_model"], p["center_mhz"], split, split*p["od_split_ratio"],
        split*p["g"], p["eta"], p["eta_od"], p["od_fraction"], nphi)
    amplitude = 10**p["log10_signal_scale_cgs"]*split
    return amplitude*density, amplitude*cd, amplitude*od


def node_basis(f, circuit, p, signal_on=True, nphi=128):
    f = np.asarray(f, dtype=float)
    c = replace(circuit, tune_capacitance_f=10**p["log10_tune_capacitance_pf"]*1e-12)
    chi = susceptibility(f, p, nphi)[0] if signal_on else 0.
    u = node_voltage(resonator_impedance(f*1e6, c, chi), c.drive_v,
                     c.drive_resistance_ohm, c.input_resistance_ohm)
    df = f-c.reference_hz/1e6
    u *= np.exp(1j*(p["phase_slope_rad_per_mhz"]*df + p["phase_curvature_rad_per_mhz2"]*df**2))
    return np.column_stack([u.real, u.imag, np.ones_like(f)])


def reconstruct(f, fit, signal_on=True, nphi=128):
    return node_basis(f, Circuit(**fit["fixed_circuit"]), fit["parameters"], signal_on, nphi) @ np.asarray(fit["quadrature_coefficients"])


def match_scan(f, y, single_fit, single_config, material_config, starts=6, seed=42):
    f, y = np.asarray(f, dtype=float), np.asarray(y, dtype=float)
    if f.shape != (500,) or y.shape != f.shape or not np.isfinite([f, y]).all() or np.any(np.diff(f) <= 0):
        raise ValueError("Need all 500 finite measured bins on their confirmed grid")
    if starts < 1:
        raise ValueError("Need at least one start")
    specs = {**single_config["fit_parameters"], **material_config["additional_parameters"]}
    low = np.array([specs[n]["bounds"][0] for n in NAMES])
    span = np.array([specs[n]["bounds"][1]-specs[n]["bounds"][0] for n in NAMES])
    initial = {**single_fit["parameters"], **{n:specs[n]["initial"] for n in NAMES[len(SINGLE_NAMES):]}}
    c = Circuit(**single_fit["fixed_circuit"])
    mask = np.ones(len(f), dtype=bool)
    scale = float(np.ptp(y))
    decode = lambda q:dict(zip(NAMES, low+span*q))

    def solve(q):
        basis = node_basis(f, c, decode(q))
        coefficients = project_readout(basis, y, mask)
        return basis@coefficients, coefficients

    def objective(q):
        return (solve(q)[0]-y)/scale

    rng = np.random.default_rng(seed)
    guesses = [np.array([initial[n] for n in NAMES])]
    for i in range(starts-1):
        p = dict(initial)
        p.update(od_split_ratio=rng.uniform(1.15,2.2), eta_od=rng.uniform(.03,.4),
                 od_fraction=rng.uniform(.015,.18))
        if i == starts-2:
            p["P_model"] *= -1
        guesses.append(np.array([p[n] for n in NAMES]))
    candidates, best = [], None
    for number, guess in enumerate(guesses, 1):
        q = np.clip((guess-low)/span, 1e-7, 1-1e-7)
        result = least_squares(objective, q, bounds=(0.,1.), jac="3-point", diff_step=1e-4,
                               max_nfev=material_config["max_nfev"], ftol=1e-10, xtol=1e-10, gtol=1e-10)
        sse = float(result.fun@result.fun)
        candidates.append({"parameters":decode(result.x), "range_normalized_sse":sse,
                           "success":bool(result.success), "evaluations":int(result.nfev),
                           "quadrature_coefficients":solve(result.x)[1].tolist()})
        print(f"  start {number}: SSE={sse:.8g}, P={decode(result.x)['P_model']:.6f}, success={result.success}", flush=True)
        if best is None or sse < float(best.fun@best.fun):
            best = result
    p = decode(best.x)
    prediction, coefficients = solve(best.x)
    baseline = node_basis(f, c, p, False)@coefficients
    lo, hi = single_config["initial_signal_window_mhz"]
    wing = (f < lo) | (f > hi)
    near = {n:"lower" if q <= 1e-4 else "upper" for n,q in zip(NAMES,best.x) if min(q,1-q) <= 1e-4}
    old_rms = single_fit["diagnostics"]["rms_recorded_units"]
    diagnostics = residual_diagnostics(y-prediction, f, wing)
    condition = float(np.linalg.cond(best.jac))
    fit = {"model_version":VERSION, "record_1based":single_fit["record_1based"],
           "fixed_circuit":single_fit["fixed_circuit"], "parameters":p,
           "quadrature_coefficients":coefficients.tolist(), "n_free":len(NAMES)+3,
           "nonlinear_parameters":list(NAMES), "profiled_readout_parameters":["a","b","d"],
           "all_500_bins_fitted":True, "success":bool(best.success), "near_bounds":near,
           "active_bounds":dict(zip(NAMES,best.active_mask.tolist())), "candidates":candidates,
           "scaled_profiled_jacobian_condition":condition if np.isfinite(condition) else None,
           "diagnostics":diagnostics, "single_site_rms_recorded_units":old_rms,
           "rms_reduction_percent":100*(1-diagnostics["rms_recorded_units"]/old_rms),
           "P_status":"Common site spin-temperature lineshape estimate; no TE prerequisite; not independent truth",
           "material_status":"Owner-confirmed butanol; fitted O-D weight is an effective signal fraction, not a measured chemical abundance"}
    return np.column_stack([f,y,prediction,baseline,prediction-baseline,y-prediction]), fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-site-report", type=Path, default=ROOT/"docs/assets/experimental-matching.json")
    parser.add_argument("--config", type=Path, default=ROOT/"configs/butanol-matching.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--starts", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan", type=int, help="Optional one-based scan for a development run")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a fresh output directory")
    single = json.loads(args.single_site_report.read_text())
    for name,expected in single["code_sha256"].items():
        if digest(ROOT/"tools"/name) != expected:
            parser.error(f"Stale single-site fit source: {name}")
    config = json.loads(args.config.read_text())
    rows, audit = load_signal_csv(ROOT/"examples/Sample_RawSignal.csv")
    if audit != single["signal_parsing"]:
        parser.error("Single-site report does not match the raw data")
    f = acquisition_grid(load_acquisition())/1e6
    fits = []
    args.output_dir.mkdir(parents=True)
    for row,seed_fit in zip(rows,single["fits"]):
        i = seed_fit["record_1based"]
        if args.scan is not None and args.scan != i:
            continue
        print(f"Butanol scan {i}",flush=True)
        trace,fit = match_scan(f,row[1:],seed_fit,single["matching_config"],config,args.starts,args.seed+i-1)
        fits.append(fit)
        np.savetxt(args.output_dir/f"scan_{i}.csv",trace,delimiter=",",comments="",
                   header="frequency_mhz,recorded,fitted,baseline,signal,residual")
        (args.output_dir/f"scan_{i}_fit.json").write_text(json.dumps(fit,indent=2,allow_nan=False)+"\n")
        print(f"RMS reduction: {fit['rms_reduction_percent']:.3f}%",flush=True)
    report = {"version":VERSION, "material":"butanol", "config":config, "config_sha256":digest(args.config),
              "single_site_report_sha256":digest(args.single_site_report), "acquisition":single["acquisition"],
              "signal_parsing":audit, "single_site_matching_config":single["matching_config"],
              "fits":fits, "starts_per_scan":args.starts, "seed":args.seed, "quadrature_nphi":128,
              "unit":"recorded units", "residual_sign":"recorded minus fitted",
              "comparison_scope":"Same measured bins and electronics bounds; 15 versus 12 fitted unknowns; in-sample residual comparison, not held-out accuracy",
              "code_sha256":{n:digest(ROOT/"tools"/n) for n in ("match_butanol.py","material_lineshapes.py","lineshape.py","circuit.py","experimental_data.py","match_experimental_signals.py")}}
    (args.output_dir/"butanol_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
