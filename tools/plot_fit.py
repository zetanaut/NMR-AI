#!/usr/bin/env python3
"""Inspect all saved raw-sweep fits and write a local figure for the walkthrough."""
import argparse
import json
from pathlib import Path

import numpy as np

from baseline_data import acquisition_grid, load_acquisition


def load_fits(folder):
    """Check the saved grid and decomposition without refitting or subtracting input."""
    reports = [folder / name for name in
               ("matching_report.json", "butanol_report.json", "uva_nd3_report.json")
               if (folder / name).is_file()]
    if len(reports) != 1:
        raise ValueError("Expected one matching, butanol, or UVA-ND3 report in --fit-dir")
    report = json.loads(reports[0].read_text())
    frequency = acquisition_grid(load_acquisition()) / 1e6
    records = []
    for fit in report["fits"]:
        nd3 = "source_record_1based" in fit
        identifier = fit["source_record_1based" if nd3 else "record_1based"]
        stem = f"record_{identifier}" if nd3 else f"scan_{identifier}"
        rows = np.genfromtxt(folder / f"{stem}.csv", delimiter=",", names=True)
        columns = {name: rows[name] for name in rows.dtype.names}
        raw = columns["raw_phase" if nd3 else "recorded"]
        if (rows.shape != (500,) or not np.isfinite(np.column_stack(list(columns.values()))).all()
                or not np.array_equal(columns["frequency_mhz"], frequency)):
            raise ValueError(f"{stem}: expected all 500 finite samples on the exact acquisition grid")
        if not np.allclose(raw - columns["fitted"], columns["residual"], rtol=1e-10, atol=1e-12):
            raise ValueError(f"{stem}: residual must equal raw minus full fit")
        if not np.allclose(columns["fitted"] - columns["baseline"], columns["signal"], rtol=1e-10, atol=1e-12):
            raise ValueError(f"{stem}: signal must equal full fit minus fitted baseline")
        records.append((stem, fit, columns, raw))
    if not records:
        raise ValueError("The report has no fits")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New .png or .svg figure path")
    args = parser.parse_args()
    if args.output.exists() or args.output.suffix not in (".png", ".svg"):
        parser.error("Choose a new .png or .svg output path")
    try:
        records = load_fits(args.fit_dir)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(records), 3, figsize=(14, 2.7 * len(records)), squeeze=False)
    for (stem, fit, columns, raw), row in zip(records, axes):
        frequency = columns["frequency_mhz"]
        row[0].plot(frequency, raw, color="#267b71", lw=1, label="Raw input (baseline present)")
        row[0].plot(frequency, columns["fitted"], color="#b44c30", lw=1, ls="--", label="Full fit")
        row[0].plot(frequency, columns["baseline"], color="#64716b", lw=1, ls=":", label="Fitted circuit baseline")
        row[1].plot(frequency, columns["signal"], color="#267b71", lw=1)
        row[2].plot(frequency, columns["residual"], color="#b44c30", lw=1)
        row[2].axhline(0, color="#64716b", lw=.5)
        row[0].set_title(f"{stem}: raw sweep and joint fit")
        row[1].set_title("Fitted response = full fit - baseline")
        row[2].set_title("Residual = raw - full fit")
        row[0].legend(fontsize=7)
        for ax in row:
            ax.set(xlabel="Frequency (MHz)", ylabel="Recorded units", xlim=(frequency[0], frequency[-1]))
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3), useOffset=False)
        rms = np.sqrt(np.mean(columns["residual"] ** 2))
        print(f"{stem}: 500 bins; P fit={100 * fit['parameters']['P_model']:.3f}%; "
              f"RMS={rms:.6g} recorded units; converged={fit['success']}; bounds={fit['near_bounds']}")
        if "rms_reduction_percent" in fit:
            print(f"  RMS reduction from the supplied single-site fit: {fit['rms_reduction_percent']:.3f}%")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=140)
    plt.close(fig)
    print(f"Saved {len(records)} raw-fit rows to {args.output}")
    print("P values are conditional fit estimates; convergence and RMS do not establish polarization accuracy.")


if __name__ == "__main__":
    main()
