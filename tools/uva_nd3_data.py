"""Audited reader for the standalone UVA-ND3 teaching excerpt."""
import hashlib
import json
from pathlib import Path

import numpy as np


def load_nd3(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("dataset") != "UVA-ND3 data" or data.get("material") != "ND3":
        raise ValueError("Require the documented UVA-ND3 data excerpt")
    rows = data.get("records", [])
    if not rows:
        raise ValueError("Need at least one measured record")
    seen = set()
    for r in rows:
        if r["source_record_1based"] in seen:
            raise ValueError("Duplicate source record identifier")
        seen.add(r["source_record_1based"])
        arrays = [np.asarray(r[k],dtype=float) for k in ("frequency_mhz","phase","baseline","basesub")]
        if any(a.shape != (512,) or not np.isfinite(a).all() for a in arrays):
            raise ValueError("Require 512 finite frequency/raw/reference/subtracted values")
        f,phase,baseline,basesub = arrays
        if np.any(np.diff(f) <= 0) or not f[0] < 32.7 < f[-1]:
            raise ValueError("Require the increasing measured deuteron grid")
        if not np.array_equal(phase-baseline,basesub):
            raise ValueError("Recorded subtraction does not equal raw phase minus its baseline")
    audit = {"source_sha256":hashlib.sha256(raw).hexdigest(),
             "parent_sha256":data["source_file_sha256"], "rows":len(rows), "bins":512,
             "source_record_numbers_1based":[r["source_record_1based"] for r in rows],
             "frequency_policy":"Use each supplied frequency array exactly; no interpolation or relabeling",
             "subtraction_check":"phase - baseline equals basesub for every bin",
             "selection":data["selection"], "units":data["units"],
             "repair_count":0, "records_sorted_or_removed":False}
    return data,audit
