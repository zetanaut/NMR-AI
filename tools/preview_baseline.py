#!/usr/bin/env python3
"""Plot the public measured baseline against sample index, without inventing Hz."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from baseline_data import load_baseline_csv

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"examples/deuteron-baseline.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/assets/deuteron-baseline-preview.svg")
    args = parser.parse_args()
    records, provenance = load_baseline_csv(args.data)
    if len(records) != 1:
        parser.error("This preview expects the single-record public example")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "deuteron-baseline-preview",
                         "font.family": "DejaVu Sans", "font.size": 12})
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    fig.patch.set_facecolor("#fafaf6")
    ax.set_facecolor("#fafaf6")
    ax.scatter(np.arange(500), records[0, 1:], color="#267b71", s=9, linewidths=0,
               label="Measured baseline · all 500 samples")
    ax.set(xlabel="Sample index (frequency grid not supplied)", ylabel="Recorded units", xlim=(-3, 502))
    ax.set_title("Deuteron baseline · measured data only, no fit", loc="left", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dce2d8", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Date": None, "Title": "Measured deuteron baseline by sample index",
                                     "Description": "All 500 recorded amplitudes; no fitted curve or assumed frequency mapping."})
    plt.close(fig)
    provenance.update({"nucleus": "deuteron", "approximate_reference_hz": 32.68e6,
                       "plot_kind": "Measured amplitudes versus sample index; not a circuit fit",
                       "voltage_unit": "Recorded units; conversion not supplied", "plotted_samples": 500,
                       "svg_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                       "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ("preview_baseline.py", "baseline_data.py")}})
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
    print(f"Previewed all 500 samples; {provenance['decimal_whitespace_repair_count']} documented decimal-spacing repairs; no fit")


if __name__ == "__main__":
    main()
