"""Strict reader for timestamp + 500-amplitude baseline CSV records.

The supplied public example contains spaces between an integer part and its
decimal point. Repair only that unambiguous formatting pattern in memory,
report every repair, and leave the source file untouched. Do not guess a grid.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np

NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
DECIMAL_GAP = re.compile(r"(?<=\d)[ \t]+(?=\.)")
DEFAULT_ACQUISITION = Path(__file__).resolve().parents[1]/"configs/deuteron-acquisition.json"


def acquisition_grid(acquisition):
    """Construct exactly 500 samples from a documented acquisition contract."""
    if acquisition.get("nucleus") != "deuteron" or acquisition.get("bins") != 500:
        raise ValueError("Require the deuteron 500-bin acquisition contract")
    start, step, reference = [acquisition.get(k) for k in ("start_mhz", "step_mhz", "reference_hz")]
    if not np.isfinite([start, step, reference]).all() or min(start, step, reference) <= 0:
        raise ValueError("Require positive finite scan start, spacing and reference")
    if not isinstance(acquisition.get("frequency_source"), str) or not acquisition["frequency_source"].strip():
        raise ValueError("Record the source of the frequency mapping")
    frequency = (start + np.arange(500)*step)*1e6
    if not frequency[0] <= reference <= frequency[-1] or not frequency[0] <= 32.7e6 <= frequency[-1]:
        raise ValueError("The sweep must bracket the deuteron nominal reference near 32.7 MHz")
    return frequency


def load_acquisition(path=DEFAULT_ACQUISITION):
    raw = Path(path).read_bytes()
    acquisition = json.loads(raw)
    acquisition_grid(acquisition)
    return {**acquisition, "configuration_sha256": hashlib.sha256(raw).hexdigest()}


def add_acquisition_arguments(parser):
    parser.add_argument("--acquisition", type=Path, default=DEFAULT_ACQUISITION,
                        help="Default: the owner's confirmed 500-bin deuteron scan contract")
    parser.add_argument("--start-mhz", type=float, help="Explicit alternative start; requires step and source")
    parser.add_argument("--step-mhz", type=float, help="Explicit alternative spacing; requires start and source")
    parser.add_argument("--frequency-source", help="Independent source for an explicitly overridden grid")


def resolve_acquisition(args):
    overrides = (args.start_mhz, args.step_mhz, args.frequency_source)
    if any(x is not None for x in overrides):
        if not all(x is not None for x in overrides):
            raise ValueError("Override --start-mhz, --step-mhz and --frequency-source together")
        if args.acquisition != DEFAULT_ACQUISITION:
            raise ValueError("Choose an acquisition file or explicit grid overrides, not both")
        acquisition = {"nucleus": "deuteron", "bins": 500, "reference_hz": 32.7e6,
                       "start_mhz": args.start_mhz, "step_mhz": args.step_mhz,
                       "frequency_source": args.frequency_source}
    else:
        acquisition = load_acquisition(args.acquisition)
    return acquisition_grid(acquisition), acquisition


def load_baseline_csv(path):
    """Return all records and parsing provenance; no deduplication or smoothing."""
    raw = Path(path).read_bytes()
    rows, repairs = [], []
    for row_number, fields in enumerate(csv.reader(io.StringIO(raw.decode("ascii"))), start=1):
        if len(fields) != 501:
            raise ValueError(f"Row {row_number}: expected timestamp + 500 amplitudes, got {len(fields)} fields")
        row = []
        for field_index, field in enumerate(fields):
            stripped = field.strip()
            normalized = DECIMAL_GAP.sub("", stripped)
            if not NUMBER.fullmatch(normalized):
                raise ValueError(f"Row {row_number}, field {field_index}: invalid or ambiguous numeric field")
            value = float(normalized)
            if not np.isfinite(value):
                raise ValueError(f"Row {row_number}, field {field_index}: nonfinite value")
            if normalized != stripped:
                repairs.append({"row_1based": row_number, "field_0based": field_index,
                                "original": field, "normalized": normalized})
            row.append(value)
        rows.append(row)
    if not rows:
        raise ValueError("Expected at least one timestamp + 500-amplitude record")
    records = np.asarray(rows, dtype=float)
    return records, {
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(records), "samples_per_row": 500,
        "decimal_whitespace_repair_count": len(repairs), "decimal_whitespace_repairs": repairs,
        "normalization": "Strip outer whitespace; remove spaces/tabs only between a digit and decimal point; no other edits",
        "field_convention": "Field 0 is timestamp; fields 1..500 are sample indices 0..499",
        "frequency_grid": "Not encoded in this format; require separate acquisition metadata",
    }
