#!/usr/bin/env python3
"""Fit only declared unknowns using independent Q-meter tuning information.

Every active circuit/readout parameter must be explicitly fixed, bounded, or
analytically profiled (the three readout coordinates together).
Gaussian hardware constraints require a supplied measurement-noise scale.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from circuit import Circuit, cable_parameters, node_voltage, resonator_impedance
from baseline_data import load_baseline_csv, add_acquisition_arguments, resolve_acquisition

INACTIVE = {"filling_factor", "susceptibility_scale_cgs"}
READOUT_INTERNAL = {"detector_gain", "detector_phase_rad", "dc_offset_v"}
CIRCUIT_NAMES = set(asdict(Circuit()))-INACTIVE-READOUT_INTERNAL
READOUT_NAMES = {"readout_gain", "readout_phase_rad", "readout_offset"}
DEFAULT_SETUP = Path(__file__).resolve().parents[1]/"configs/deuteron-baseline-setup.json"


def calibrated_cable_lc(reference_hz, half_wave_m, impedance_ohm=50., resistance_ohm_per_m=3.43):
    """Derive L', C' for G'=0, sqrt(L'/C')=Z_lc and Im(gamma)=pi/h.

    The half-wave length is supplied tuning information; Z_lc and R' remain
    independent assumptions unless measured. Retain finite conductor loss.
    """
    if (not np.isfinite([reference_hz, half_wave_m, impedance_ohm, resistance_ohm_per_m]).all()
            or min(reference_hz, half_wave_m, impedance_ohm) <= 0 or resistance_ohm_per_m < 0):
        raise ValueError("Require positive frequency, half-wave length, impedance and nonnegative resistance")
    beta, omega = np.pi/half_wave_m, 2*np.pi*reference_hz
    capacitance = 2*beta**2/(omega*np.sqrt(resistance_ohm_per_m**2+4*beta**2*impedance_ohm**2))
    return impedance_ohm**2*capacitance, capacitance


def validate_setup(setup):
    """Reject silent defaults, conflicting length conventions, and invented weights."""
    if not isinstance(setup, dict) or not isinstance(setup.get("parameters"), dict):
        raise ValueError("Setup must be an object containing a parameters object")
    if set(setup)-{"description", "parameters", "cable_tuning", "noise_sigma_recorded"}:
        raise ValueError("Unknown setup field")
    params = setup.get("parameters", {})
    required = CIRCUIT_NAMES | READOUT_NAMES
    tuning = setup.get("cable_tuning")
    if tuning is not None:
        required_tuning = {"half_wave_multiple", "reference_hz", "source"}
        if not required_tuning <= set(tuning) or set(tuning)-required_tuning-{"half_wave_length_m"}:
            raise ValueError("cable_tuning needs half_wave_multiple, reference_hz, source")
        n = tuning["half_wave_multiple"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("The known half-wave multiple must be a positive integer")
        if not np.isfinite(tuning["reference_hz"]) or tuning["reference_hz"] <= 0 or not tuning["source"]:
            raise ValueError("Specify positive tuning frequency and its independent source")
        if "half_wave_length_m" in tuning:
            h = tuning["half_wave_length_m"]
            if not np.isfinite(h) or h <= 0:
                raise ValueError("The known half_wave_length_m must be positive")
        required = required-{"cable_length_m"} | {"cable_delta_length_m"}
    if set(params) != required:
        raise ValueError(f"Explicit parameter contract required; missing={sorted(required-set(params))}, "
                         f"unknown={sorted(set(params)-required)}")
    has_prior = False
    for name, spec in params.items():
        if not isinstance(spec, dict) or not isinstance(spec.get("source"), str) or not spec["source"].strip():
            raise ValueError(f"{name}: record the measurement source or explicit assumption")
        if "profile" in spec:
            if name not in READOUT_NAMES or set(spec) != {"profile", "source"} or spec["profile"] is not True:
                raise ValueError("Only the three readout parameters may use profile=true and source")
        elif "value" in spec:
            if set(spec) != {"value", "source"} or spec["value"] is None or not np.isfinite(spec["value"]):
                raise ValueError(f"{name}: fixed parameters require finite value and source only")
        else:
            if set(spec)-{"initial", "bounds", "prior", "source"} or not {"initial", "bounds"} <= set(spec):
                raise ValueError(f"{name}: unknown parameters need initial, bounds and source")
            bounds = spec["bounds"]
            if (not isinstance(bounds, list) or len(bounds) != 2 or spec["initial"] is None
                    or not np.isfinite([spec["initial"], *bounds]).all()
                    or not bounds[0] < bounds[1] or not bounds[0] <= spec["initial"] <= bounds[1]):
                raise ValueError(f"{name}: require finite lower < upper and initial inside bounds")
            if name == "reference_hz":
                raise ValueError("The reference frequency is scan metadata, not a baseline-fit unknown")
            if "prior" in spec:
                prior = spec["prior"]
                if (set(prior) != {"mean", "sigma"} or not np.isfinite(list(prior.values())).all()
                        or prior["sigma"] <= 0):
                    raise ValueError(f"{name}: prior needs a finite mean and positive standard uncertainty")
                has_prior = True
    profiled = {n for n, spec in params.items() if spec.get("profile")}
    if profiled and profiled != READOUT_NAMES:
        raise ValueError("Profile all three readout parameters together, or none")
    if "value" not in params["drive_v"] and "value" not in params["readout_gain"]:
        raise ValueError("Do not fit RF drive and free readout gain together: only their product is identifiable")
    if has_prior and "noise_sigma_recorded" not in setup:
        raise ValueError("Gaussian hardware constraints require noise_sigma_recorded to weight the data consistently")
    if "noise_sigma_recorded" in setup:
        sigma = np.asarray(setup["noise_sigma_recorded"], dtype=float)
        if sigma.ndim > 1 or (sigma.ndim == 1 and sigma.shape != (500,)) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("noise_sigma_recorded must be a positive scalar or 500 positive standard deviations")
    initial = {name: spec.get("value", spec.get("initial", 0.0)) for name, spec in params.items()}
    resolved_circuit(initial, setup)
    # Check passive parameter boundaries, not only the starting point.
    for name, spec in params.items():
        if "bounds" in spec:
            for bound in spec["bounds"]:
                resolved_circuit({**initial, name: bound}, setup)
    return initial


def resolved_circuit(parameters, setup):
    values = {name: parameters[name] for name in CIRCUIT_NAMES if name != "cable_length_m"}
    values["cable_length_m"] = parameters.get("cable_length_m", 0.0)
    if "cable_tuning" in setup:
        tuning = setup["cable_tuning"]
        beta = float(cable_parameters(tuning["reference_hz"], Circuit(**values))[1].imag)
        if "half_wave_length_m" in tuning and not np.isclose(np.pi/beta, tuning["half_wave_length_m"], rtol=1e-9, atol=0):
            raise ValueError("Cable RLGC contradict the known half-wave length; calibrate propagation before fitting")
        values["cable_length_m"] = tuning["half_wave_multiple"]*np.pi/beta + parameters["cable_delta_length_m"]
    return Circuit(**values)


def readout_basis(frequency_hz, parameters, setup):
    """Linear readout basis, retaining the known frequency-dependent phase."""
    c = resolved_circuit(parameters, setup)
    u = node_voltage(resonator_impedance(frequency_hz, c), c.drive_v,
                     c.drive_resistance_ohm, c.input_resistance_ohm)
    df = np.asarray(frequency_hz)-c.reference_hz
    phase = c.detector_phase_slope_rad_per_hz*df + c.detector_phase_curvature_rad_per_hz2*df**2
    rotated = u*np.exp(1j*phase)
    return np.column_stack([rotated.real, rotated.imag, np.ones_like(df)])


def setup_voltage(frequency_hz, parameters, setup):
    """Output in recorded units; no assumption about the DAQ-to-volts factor."""
    gain, phase = parameters["readout_gain"], parameters["readout_phase_rad"]
    coefficients = [gain*np.cos(phase), -gain*np.sin(phase), parameters["readout_offset"]]
    return readout_basis(frequency_hz, parameters, setup) @ coefficients


def fit_with_setup(frequency_hz, values, setup, starts=12, seed=42):
    initial = validate_setup(setup)
    frequency_hz, values = np.asarray(frequency_hz, dtype=float), np.asarray(values, dtype=float)
    if (frequency_hz.shape != (500,) or values.shape != (500,) or not np.isfinite(values).all()
            or not np.isfinite(frequency_hz).all() or np.any(frequency_hz <= 0)
            or np.any(np.diff(frequency_hz) <= 0) or not isinstance(starts, int) or starts < 1):
        raise ValueError("Need 500 finite values on a positive increasing frequency grid and positive starts")
    specs = setup["parameters"]
    free_names = [name for name, spec in specs.items() if "value" not in spec]
    profiled = [name for name in free_names if specs[name].get("profile")]
    names = [name for name in free_names if name not in profiled]
    lower = np.array([specs[n]["bounds"][0] for n in names])
    span = np.array([specs[n]["bounds"][1]-specs[n]["bounds"][0] for n in names])
    # Numerical coordinates are scaled to [0,1]; physical units stay in the report.
    sigma = np.asarray(setup.get("noise_sigma_recorded", max(float(np.ptp(values)), 1e-12)))

    def decode(q):
        parameters = {**initial, **dict(zip(names, lower+span*q))}
        if profiled:
            basis = readout_basis(frequency_hz, parameters, setup)
            weights = np.broadcast_to(sigma, values.shape)
            coefficients, _, rank, _ = np.linalg.lstsq(basis/weights[:, None], values/weights, rcond=None)
            if rank != 3:
                raise ValueError("Readout basis is rank deficient; fix independently known readout parameters")
            a, b, offset = coefficients
            parameters.update(readout_gain=float(np.hypot(a, b)), readout_phase_rad=float(np.arctan2(-b, a)),
                              readout_offset=float(offset))
        return parameters

    def residual_vector(q):
        parameters = decode(q)
        residual = (setup_voltage(frequency_hz, parameters, setup)-values)/sigma
        prior_residual = [(parameters[n]-specs[n]["prior"]["mean"])/specs[n]["prior"]["sigma"]
                          for n in names if "prior" in specs[n]]
        return np.concatenate([residual, prior_residual])

    candidates = []
    if names:
        rng = np.random.default_rng(seed)
        guesses = [(np.array([initial[n] for n in names])-lower)/span]
        guesses.extend(rng.uniform(0, 1, len(names)) for _ in range(starts-1))
        best = None
        for q in guesses:
            result = least_squares(residual_vector, q, bounds=(np.zeros(len(names)), np.ones(len(names))),
                                   jac="3-point", diff_step=1e-4,
                                   max_nfev=1500, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            cost = float(result.fun@result.fun)
            candidates.append({"parameters": decode(result.x), "objective": cost,
                               "success": bool(result.success), "evaluations": int(result.nfev)})
            if best is None or cost < float(best.fun@best.fun):
                best = result
        parameters = decode(best.x)
        objective_vector = best.fun
        singular = np.linalg.svd(best.jac, compute_uv=False)
        condition = float(singular[0]/max(singular[-1], 1e-30))
        active_bounds, success = best.active_mask.tolist(), bool(best.success)
    else:
        parameters, objective_vector = decode(np.array([])), residual_vector(np.array([]))
        condition, active_bounds, success = None, [], True
    prediction = setup_voltage(frequency_hz, parameters, setup)
    residual = values-prediction
    c = resolved_circuit(parameters, setup)
    rms = float(np.sqrt(np.mean(residual**2)))
    near_bounds = {n: "lower" if (parameters[n]-lower[i])/span[i] <= 1e-5 else "upper"
                   for i, n in enumerate(names)
                   if min(parameters[n]-lower[i], lower[i]+span[i]-parameters[n])/span[i] <= 1e-5}
    gain, phase, offset = [parameters[n] for n in ("readout_gain", "readout_phase_rad", "readout_offset")]
    return prediction, {
        "fit_mode": "tuning-informed", "circuit": asdict(c), "setup": setup,
        "resolved_parameters": parameters, "free_parameters": free_names, "n_free": len(free_names),
        "nonlinear_parameters": names, "profiled_parameters": profiled,
        "parameter_manifest": [{"name": n, "mode": "fixed" if "value" in spec else "profiled" if spec.get("profile") else "fitted",
                                "result": parameters[n], **spec} for n, spec in specs.items()],
        "quadrature_coefficients": [float(gain*np.cos(phase)), float(-gain*np.sin(phase)), float(offset)],
        "recorded_units_per_node_volt": float(gain), "detector_phase_rad": float(phase),
        "dc_offset_recorded_units": float(offset), "shape_fit_success": success,
        "rmse_recorded_units": rms, "rmse_over_trace_range": rms/max(float(np.ptp(values)), 1e-12),
        "residual_lag1": float(np.corrcoef(residual[:-1], residual[1:])[0, 1]) if residual.std() > 0 else None,
        "data_objective": float(objective_vector[:500]@objective_vector[:500]),
        "constraint_objective": float(objective_vector[500:]@objective_vector[500:]),
        "objective_kind": "diagonal-noise chi-square plus independent Gaussian constraints" if "noise_sigma_recorded" in setup
                          else "range-normalized unweighted SSE; no likelihood or parameter errors claimed",
        "scaled_shape_jacobian_condition": condition, "active_bounds": active_bounds,
        "active_bound_parameters": names,
        "near_bounds": near_bounds, "near_bound_fraction_of_search_interval": 1e-5,
        "numerical_derivative": "Three-point finite differences, relative step 1e-4, in unit-interval coordinates",
        "jacobian_scope": "Declared nonlinear parameters in unit-interval coordinates; readout profiled out; includes constraints" if profiled
                          else "All free parameters including readout and constraints, in unit-interval numerical coordinates",
        "multistart_candidates": candidates,
    }


def reconstruct_fit(frequency_hz, fit):
    return setup_voltage(frequency_hz, fit["resolved_parameters"], fit["setup"])


def main(default_setup=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("--setup", type=Path, default=default_setup, required=default_setup is None)
    parser.add_argument("--output-dir", type=Path, required=True)
    add_acquisition_arguments(parser)
    parser.add_argument("--starts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        f, acquisition = resolve_acquisition(args)
    except ValueError as error:
        parser.error(str(error))
    setup = json.loads(args.setup.read_text())
    validate_setup(setup)
    if args.output_dir.exists():
        parser.error("Choose a new output directory")
    if args.starts < 1:
        parser.error("Require positive starts")
    data, input_parsing = load_baseline_csv(args.data)
    _, first = np.unique(data, axis=0, return_index=True)
    args.output_dir.mkdir(parents=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    reports = []
    for i, row in enumerate(data[np.sort(first)]):
        prediction, fit = fit_with_setup(f, row[1:], setup, args.starts, args.seed)
        fit["timestamp"] = float(row[0])
        reports.append(fit)
        np.savetxt(args.output_dir/f"trace_{i}.csv", np.column_stack([f/1e6, row[1:], prediction, row[1:]-prediction]),
                   delimiter=",", header="frequency_mhz,recorded,fit,residual", comments="")
        axes[0].plot(f/1e6, row[1:], lw=1, label=f"Trace {i+1}")
        axes[0].plot(f/1e6, prediction, "k--", lw=.8)
        axes[1].plot(f/1e6, row[1:]-prediction, lw=.8)
        print(f"trace {i+1}: free={fit['n_free']} RMS={fit['rmse_recorded_units']:.6g} recorded units", flush=True)
    axes[0].legend()
    axes[0].set_ylabel("Recorded units")
    axes[1].set_ylabel("Recorded minus fitted")
    axes[1].set_xlabel("Frequency (MHz)")
    fig.tight_layout()
    fig.savefig(args.output_dir/"baseline_fits.png", dpi=160)
    import scipy
    report = {"fit_mode": "tuning-informed", "setup": setup, "nucleus": "deuteron",
              "reference_hz": setup["parameters"]["reference_hz"]["value"],
              "frequency_source": acquisition["frequency_source"], "acquisition": acquisition,
              "source_sha256": input_parsing["source_sha256"], "input_parsing": input_parsing,
              "setup_sha256": hashlib.sha256(args.setup.read_bytes()).hexdigest(),
              "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("fit_tuned_baseline.py", "circuit.py", "baseline_data.py")},
              "numpy_version": np.__version__, "scipy_version": scipy.__version__,
              "starts": args.starts, "seed": args.seed, "start_mhz": acquisition["start_mhz"], "step_mhz": acquisition["step_mhz"],
              "rows": len(data), "unique_rows": len(first), "fitted_bin_mask": "All 500 bins; no exclusions",
              "residual_sign": "recorded minus fitted", "fits": reports}
    (args.output_dir/"fit_report.json").write_text(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
