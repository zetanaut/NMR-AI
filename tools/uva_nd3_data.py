"""Audited 500-bin UVA-ND3 example and immutable measurement provenance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np

from baseline_data import acquisition_grid, load_acquisition

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "provenance/uva-nd3-measured.json"
SOURCE_SHA256 = "af5a4358bae2aa48cc3c7d16967816679ae15f9c863511c9c96daa729885e4bb"
VERSION = "uva-nd3-linear-resampling-500-v1"
ARRAYS = ("frequency_mhz", "phase", "baseline", "basesub")


def confirmed_frequency():
    return acquisition_grid(load_acquisition()) / 1e6


def load_nd3_source():
    """Read the original acquisition archive, never a working example input."""
    raw = SOURCE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
        raise ValueError("UVA-ND3 measurement archive hash changed")
    data = json.loads(raw)
    for row in data["records"]:
        f, phase, baseline, basesub = [np.asarray(row[k], dtype=float) for k in ARRAYS]
        if any(a.shape != (512,) or not np.isfinite(a).all() for a in (f, phase, baseline, basesub)):
            raise ValueError("Invalid original acquisition arrays in provenance archive")
        if np.any(np.diff(f) <= 0) or not np.array_equal(phase - baseline, basesub):
            raise ValueError("Invalid original frequency grid or recorded subtraction")
    return data


def linear_resample(source_frequency, values, target_frequency):
    """Piecewise linear interpolation in MHz; only roundoff outside endpoints.

    np.interp clamps the first target by 7.1e-15 MHz in this archive. Permit
    at most four floating-point spacings; reject substantive extrapolation.
    Columns of a model basis use the identical observation operator.
    """
    source = np.asarray(source_frequency, dtype=float)
    target = np.asarray(target_frequency, dtype=float)
    values = np.asarray(values)
    if (source.ndim != 1 or target.ndim != 1 or len(source) < 2 or len(target) < 1
            or values.ndim not in (1, 2) or values.shape[0] != len(source)
            or not all(np.isfinite(a).all() for a in (source, target, values))
            or np.any(np.diff(source) <= 0) or np.any(np.diff(target) <= 0)):
        raise ValueError("Require finite increasing grids and matching source values")
    tolerance = 4 * np.spacing(np.max(np.abs(source)))
    if target[0] < source[0] - tolerance or target[-1] > source[-1] + tolerance:
        raise ValueError("Resampling would extrapolate beyond the measured frequency support")
    if values.ndim == 1:
        return np.interp(target, source, values)
    return np.column_stack([np.interp(target, source, col) for col in values.T])


def sampling_contract(record_id):
    return {"version": VERSION, "source_record_1based": record_id,
            "source_excerpt_sha256": SOURCE_SHA256}


def sampling_frequency(sampling):
    if sampling != sampling_contract(sampling["source_record_1based"]):
        raise ValueError("Unrecognized UVA-ND3 sampling provenance")
    for row in load_nd3_source()["records"]:
        if row["source_record_1based"] == sampling["source_record_1based"]:
            return np.asarray(row["frequency_mhz"])
    raise ValueError("Unknown source record for UVA-ND3 sampling")


def prepare_nd3():
    """Derive the reproducible teaching artifact; preserve the source archive."""
    data = deepcopy(load_nd3_source())
    target = confirmed_frequency()
    data["grid"] = "Exact 500-bin teaching grid: f_j = 32.3 + 0.0015287*j MHz, j=0..499"
    data["export_policy"] = "Derived 500-bin arrays; float64 linear interpolation of raw phase and recorded baseline, then subtraction"
    data["resampling"] = {
        "version": VERSION, "source_archive": "provenance/uva-nd3-measured.json",
        "source_excerpt_sha256": SOURCE_SHA256,
        "acquisition_sha256": load_acquisition()["configuration_sha256"],
        "method": "Piecewise linear interpolation in frequency, applied independently to phase and baseline in float64; basesub = phase - baseline",
        "endpoint_policy": "Clamp only endpoint roundoff within four floating-point spacings; no physical extrapolation",
        "support_policy": "Use the confirmed target window; source frequencies beyond its upper endpoint are outside the working example",
        "noise_policy": "Interpolation changes noise correlation and resolution; no independent-bin noise or measured covariance claim",
        "model_policy": "Evaluate the conditional model on source frequencies and apply the same interpolation before fitting all 500 target bins",
    }
    for row in data["records"]:
        f = row["frequency_mhz"]
        phase = linear_resample(f, row["phase"], target)
        baseline = linear_resample(f, row["baseline"], target)
        row.update(frequency_mhz=target.tolist(), phase=phase.tolist(),
                   baseline=baseline.tolist(), basesub=(phase-baseline).tolist())
    return data


def load_nd3(path):
    """Accept only the exact 500-bin derived example and verify its provenance."""
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("dataset") != "UVA-ND3 data" or data.get("material") != "ND3":
        raise ValueError("Require the documented UVA-ND3 data example")
    rows = data.get("records", [])
    if not rows:
        raise ValueError("Need at least one record")
    target = confirmed_frequency()
    for row in rows:
        arrays = [np.asarray(row[k], dtype=float) for k in ARRAYS]
        if any(a.shape != (500,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Require 500 finite frequency/raw/reference/subtracted values")
        f, phase, baseline, basesub = arrays
        if not np.array_equal(f, target):
            raise ValueError("Require the exact confirmed 500-bin frequency grid")
        if not np.array_equal(phase-baseline, basesub):
            raise ValueError("Subtraction does not equal resampled phase minus its baseline")
    if data != prepare_nd3():
        raise ValueError("UVA-ND3 data differ from the documented source resampling")
    audit = {"source_sha256": hashlib.sha256(raw).hexdigest(),
             "parent_sha256": data["source_file_sha256"], "rows": len(rows), "bins": 500,
             "source_record_numbers_1based": [r["source_record_1based"] for r in rows],
             "frequency_policy": data["grid"], "resampling": data["resampling"],
             "subtraction_check": "resampled phase - resampled baseline equals basesub for every bin",
             "selection": data["selection"], "units": data["units"],
             "repair_count": 0, "records_sorted_or_removed": False}
    return data, audit
