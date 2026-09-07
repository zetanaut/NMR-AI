#!/usr/bin/env python3
"""Plot all measured baseline samples on the owner's confirmed deuteron grid."""
import argparse
import hashlib
import json
from pathlib import Path

from baseline_data import load_baseline_csv, add_acquisition_arguments, resolve_acquisition

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT/"examples/deuteron-baseline.csv")
    parser.add_argument("--output", type=Path, default=ROOT/"docs/assets/deuteron-baseline-preview.svg")
    add_acquisition_arguments(parser)
    args = parser.parse_args()
    frequency, acquisition = resolve_acquisition(args)
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
    ax.scatter(frequency/1e6, records[0, 1:], color="#267b71", s=9, linewidths=0,
               label="Measured baseline · all 500 samples")
    ax.set(xlabel="Frequency (MHz)", ylabel="Recorded units", xlim=(frequency[0]/1e6-.004, frequency[-1]/1e6+.004))
    ax.set_title("Deuteron baseline · measured data only, no fit", loc="left", pad=14)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dce2d8", linewidth=.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower center")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={"Date": None, "Title": "Measured deuteron baseline on the confirmed frequency grid",
                                     "Description": "All 500 recorded amplitudes; the owner's acquisition mapping; no fitted curve."})
    args.output.write_text("\n".join(line.rstrip() for line in args.output.read_text().splitlines())+"\n")
    plt.close(fig)
    provenance.update({"nucleus": "deuteron", "reference_hz": acquisition["reference_hz"], "acquisition": acquisition,
                       "frequency_grid": "Owner-confirmed start + sample index * step; see acquisition",
                       "plot_kind": "Measured amplitudes versus confirmed frequency; not a circuit fit",
                       "voltage_unit": "Recorded units; conversion not supplied", "plotted_samples": 500,
                       "svg_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                       "code_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                       for name in ("preview_baseline.py", "baseline_data.py")}})
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2, allow_nan=False)+"\n")
    print(f"Previewed all 500 samples; {provenance['decimal_whitespace_repair_count']} documented decimal-spacing repairs; no fit")


if __name__ == "__main__":
    main()
