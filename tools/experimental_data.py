"""Audited UTF-8 reader for the public timestamp + 500-bin polarized scans.

Reuse the strict numeric grammar of baseline_data without changing that older
file's parsing contract. Normalize only documented record separators and gaps.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np
from baseline_data import NUMBER, DECIMAL_GAP


def load_signal_csv(path):
    raw = Path(path).read_bytes()
    text = raw.decode("utf-8")
    separators = [{"character_offset": i, "codepoint": "U+2028"}
                  for i, char in enumerate(text) if char == "\u2028"]
    # U+2028 is a record separator, never silently deleted inside a number.
    text = text.replace("\u2028", "\n").replace("\r\n", "\n")
    rows, blank_lines, repairs, record_lines = [], [], [], []
    lines = text.split("\n")
    for line_number, line in enumerate(lines, 1):
        if not line.strip(" \t"):
            if line_number < len(lines) or line:
                blank_lines.append(line_number)
            continue
        fields = next(csv.reader([line]))
        if len(fields) != 501:
            raise ValueError(f"Logical line {line_number}: expected timestamp + 500 samples, got {len(fields)}")
        values = []
        for index, field in enumerate(fields):
            stripped = field.strip(" \t")
            normalized = DECIMAL_GAP.sub("", stripped)
            if not NUMBER.fullmatch(normalized):
                raise ValueError(f"Logical line {line_number}, field {index}: invalid or ambiguous numeric value")
            value = float(normalized)
            if not np.isfinite(value):
                raise ValueError(f"Logical line {line_number}, field {index}: nonfinite numeric value")
            if normalized != stripped:
                repairs.append({"logical_line_1based": line_number, "field_0based": index,
                                "original": field, "normalized": normalized})
            values.append(value)
        rows.append(values)
        record_lines.append(line_number)
    if not rows:
        raise ValueError("Need at least one timestamp + 500-sample record")
    records = np.asarray(rows)
    duplicate_of = []
    for i, row in enumerate(records):
        matches = [j+1 for j, previous in enumerate(records[:i]) if np.array_equal(row, previous)]
        duplicate_of.append(matches[0] if matches else None)
    return records, {
        "source_sha256": hashlib.sha256(raw).hexdigest(), "encoding": "UTF-8", "rows": len(rows),
        "samples_per_row": 500, "record_logical_lines_1based": record_lines,
        "unicode_line_separators": separators, "blank_logical_lines_1based": blank_lines,
        "decimal_whitespace_repairs": repairs, "duplicate_of_record_1based": duplicate_of,
        "timestamp_order": "increasing" if np.all(np.diff(records[:, 0]) > 0) else "not strictly increasing; input order retained",
        "policy": "Preserve every record and all 500 values; audit U+2028 and blank separators; no smoothing, rescaling or deduplication",
        "frequency_grid": "Not stored in CSV; supplied separately as an acquisition contract",
    }
