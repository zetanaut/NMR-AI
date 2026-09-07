"""Strict reader for timestamp + 500-amplitude baseline CSV records.

The supplied public example contains spaces between an integer part and its
decimal point. Repair only that unambiguous formatting pattern in memory,
report every repair, and leave the source file untouched. Do not guess a grid.
"""
import csv
import hashlib
import io
from pathlib import Path
import re

import numpy as np

NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\Z")
DECIMAL_GAP = re.compile(r"(?<=\d)[ \t]+(?=\.)")


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
