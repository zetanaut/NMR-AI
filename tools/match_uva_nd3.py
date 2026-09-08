#!/usr/bin/env python3
"""Fit UVA-ND3 reference-subtracted spectra with a conditional spin-1 lineshape.

Unknown hardware constants are not borrowed from the butanol apparatus. A
constant absorption/dispersion readout and joint cubic residual background are
explicit approximations for the already reference-subtracted data. No existing
DAQ polarization, area, fitted curve, or calibration value enters optimization.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from lineshape import powder_complex
from match_experimental_signals import digest, residual_diagnostics
from uva_nd3_data import load_nd3

ROOT = Path(__file__).resolve().parents[1]
VERSION = "uva-nd3-reference-subtracted-lineshape-v1"
NAMES = ("center_mhz","split_mhz","g","eta","P_model")


def normalized_shape(f, p, nphi=32):
    """Unit-area conditional shape, in inverse MHz; fitted amplitude includes P.

    Division of the spin-population weights by P is evaluated analytically,
    including its finite symmetric limit at P=0. This is a shape-only fitting
    coordinate, not a susceptibility generator with a nonzero zero-P signal.
    """
    polarization = p["P_model"]
    q_over_p = 3*polarization/(2+np.sqrt(4-3*polarization**2))
    x = (np.asarray(f)-p["center_mhz"])/p["split_mhz"]
    plus = powder_complex(x,1,p["eta"],p["g"],nphi)
    minus = powder_complex(x,-1,p["eta"],p["g"],nphi)
    return (.5*(1+q_over_p)*plus + .5*(1-q_over_p)*minus)/p["split_mhz"]


def basis(f, p, nphi=32):
    shape = normalized_shape(f,p,nphi)
    t = (np.asarray(f)-32.7)/.4
    return np.column_stack([-shape.imag,shape.real,np.ones_like(t),t,t*t,t*t*t])


def reconstruct(f, fit, background_only=False, nphi=32):
    design = basis(f,fit["parameters"],nphi)
    coefficients = np.asarray(fit["readout_background_coefficients"])
    if background_only:
        return design[:,2:]@coefficients[2:]
    return design@coefficients


def match_scan(f, y, config, starts=6, seed=42):
    f,y = np.asarray(f),np.asarray(y)
    if f.shape != (512,) or y.shape != f.shape or not np.isfinite([f,y]).all() or np.any(np.diff(f)<=0):
        raise ValueError("Require all 512 finite samples on the measured frequency grid")
    if starts < 1:
        raise ValueError("Need at least one start")
    specs = config["fit_parameters"]
    low = np.array([specs[n]["bounds"][0] for n in NAMES])
    span = np.array([specs[n]["bounds"][1]-specs[n]["bounds"][0] for n in NAMES])
    initial = np.array([specs[n]["initial"] for n in NAMES])
    decode = lambda q:dict(zip(NAMES,low+span*q))
    scale = float(np.ptp(y))

    def solve(q):
        design = basis(f,decode(q))
        coefficients,_,rank,_ = np.linalg.lstsq(design,y,rcond=None)
        if rank != 6:
            raise ValueError("Rank-deficient conditional readout/background")
        return design@coefficients,coefficients

    def objective(q):
        return (solve(q)[0]-y)/scale

    rng = np.random.default_rng(seed)
    guesses = [initial]
    for i in range(starts-1):
        guess = initial.copy()
        guess[0] += rng.uniform(-.003,.003)
        guess[1] *= rng.uniform(.94,1.06)
        guess[2] *= rng.uniform(.5,1.8)
        guess[3] = rng.uniform(.01,.15)
        guess[4] = .35 if i%2==0 else -.35
        guesses.append(guess)
    candidates,best = [],None
    for guess in guesses:
        q = np.clip((guess-low)/span,1e-7,1-1e-7)
        result = least_squares(objective,q,bounds=(0.,1.),jac="3-point",diff_step=1e-4,
                               max_nfev=config["max_nfev"],ftol=1e-11,xtol=1e-11,gtol=1e-11)
        sse = float(result.fun@result.fun)
        candidates.append({"parameters":decode(result.x),"range_normalized_sse":sse,
                           "success":bool(result.success),"evaluations":int(result.nfev)})
        if best is None or sse < float(best.fun@best.fun):
            best=result
    p=decode(best.x)
    prediction,coefficients=solve(best.x)
    background=basis(f,p)[:,2:]@coefficients[2:]
    lo,hi=config["noise_proxy_window_mhz"]
    wing=(f<lo)|(f>hi)
    condition=float(np.linalg.cond(best.jac))
    fit={"model_version":VERSION,"parameters":p,"readout_background_coefficients":coefficients.tolist(),
         "coefficient_order":["absorption_area","dispersion_area","b0","b1","b2","b3"],
         "n_free":11,"nonlinear_parameters":list(NAMES),"all_512_bins_fitted":True,
         "success":bool(best.success),"candidates":candidates,
         "near_bounds":{n:"lower" if q<=1e-4 else "upper" for n,q in zip(NAMES,best.x) if min(q,1-q)<=1e-4},
         "scaled_profiled_jacobian_condition":condition if np.isfinite(condition) else None,
         "diagnostics":residual_diagnostics(y-prediction,f,wing),
         "P_status":"Conditional spin-temperature lineshape estimate; independent of stored DAQ pol/cc; no calibrated uncertainty claim",
         "hardware_status":"Reference-subtracted response approximation; not a full-circuit fit or a calibration to the butanol apparatus"}
    return np.column_stack([f,y,prediction,background,prediction-background,y-prediction]),fit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,default=ROOT/"examples/uva-nd3.json")
    parser.add_argument("--config",type=Path,default=ROOT/"configs/uva-nd3-matching.json")
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--starts",type=int,default=6)
    parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error("Choose a fresh output directory")
    data,audit=load_nd3(args.data)
    config=json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True)
    fits=[]
    for r in data["records"]:
        i=r["source_record_1based"]
        trace,fit=match_scan(r["frequency_mhz"],r["basesub"],config,args.starts,args.seed+i)
        fit.update(source_record_1based=i,start_time=r["start_time"],sweeps=r["sweeps"])
        fits.append(fit)
        np.savetxt(args.output_dir/f"record_{i}.csv",trace,delimiter=",",comments="",
                   header="frequency_mhz,reference_subtracted,fitted,residual_background,signal,residual")
        print(f"UVA-ND3 record {i}: P={fit['parameters']['P_model']:.6f}, RMS={fit['diagnostics']['rms_recorded_units']:.6g}, bounds={fit['near_bounds']}",flush=True)
    report={"version":VERSION,"dataset":"UVA-ND3 data","material":"ND3","audit":audit,"config":config,
            "config_sha256":digest(args.config),"fits":fits,"starts_per_scan":args.starts,"seed":args.seed,
            "unit":"recorded units","residual_sign":"recorded reference-subtracted spectrum minus fitted",
            "code_sha256":{n:digest(ROOT/"tools"/n) for n in ("match_uva_nd3.py","uva_nd3_data.py","lineshape.py","match_experimental_signals.py")}}
    (args.output_dir/"uva_nd3_report.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":main()
