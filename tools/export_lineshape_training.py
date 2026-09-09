#!/usr/bin/env python3
"""Verify and publish the 500-bin raw/reference network exercise."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from export_model_comparison import verify_comparison

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- LINESHAPE TRAINING START -->", "<!-- LINESHAPE TRAINING END -->"


def render_block(snapshot):
    run = snapshot["runs"][0]
    metric = run["metrics"]
    groups, rows = snapshot["group_values"], snapshot["rows"]
    errors = "".join(f"<td>{100*metric[key]:.4f}</td>" for key in ("bias", "width", "rmse", "p95_absolute_error"))
    return f'''{START}
<div id="lineshape-network" class="subsection"><h3>Train and predict on the 500-bin lineshapes</h3>
<p>The trainer selects raw and reference-subtracted recorded-unit channels, with no required TE calibration. It saves the exact frequency grid, channel mode, units, training-only scalers and model weights; prediction reuses and checks that contract. Simulator-only clean arrays are excluded from the inputs.</p>
<p>Follow <a href="walkthrough.html#train">walkthrough step 8</a> after generating and checking <code>lineshape-data.npz</code>. It gives the exact training command, the saved-file check, and the expected source split. Then <a href="walkthrough.html#predict">step 9 reloads your checkpoint, makes 400 held-out predictions, and explains the error report</a>. Keep the model and saved preprocessing together.</p>
<p>Grouping uses <code>source_scan_1based</code>: training sources {groups['train']}, validation source {groups['validation']}, and test source {groups['test']}. All simulated descendants remain with their measured seed. This gives {rows['train']:,}/{rows['validation']:,}/{rows['test']:,} training/validation/test spectra. New simulator truth P spans −0.6 to +0.6; the measured fits' P estimates are not labels.</p>
<div class="table-wrap"><table><caption>Saved multiscale CNN · synthetic source holdout · errors in percentage points</caption><thead><tr><th>Test spectra</th><th>Bias</th><th>Width</th><th>RMSE</th><th>95th absolute error</th></tr></thead><tbody><tr><td>{metric['n']}</td>{errors}</tr></tbody></table></div>
<p>This run selected epoch {run['best_epoch']} of {len(run['history'])} by validation MSE. Test error is measured on one held-out source's synthetic descendants. The {100*metric['rmse']:.4f} pp RMSE shows remaining recovery error under this coverage; it does not establish experimental polarization accuracy. The TE-area benchmark uses different input information and P coverage, so its scores are a separate exercise.</p>
<p>For ordinary prediction on measured inputs, supply <code>signals</code> and corresponding <code>baselines</code> arrays of shape <code>(events,500)</code>, the exact <code>frequency_mhz</code>, and strings <code>feature_mode="lineshape"</code> and <code>voltage_unit="recorded units"</code>. P labels and simulator/group identifiers are unnecessary. Inputs from different acquisition grids require their own model contract.</p>
<p class="provenance">The <a href="assets/lineshape-training.json">verified training record</a> retains every test prediction, source groups, preprocessing contract, history and artifact hashes. Reconstruction uses the saved checkpoint and training-only preprocessing.</p>
</div>
{END}'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/lineshape-training.json")
    parser.add_argument("--runs-dir", type=Path)
    args = parser.parse_args()
    snapshot = verify_comparison(args.protocol, args.runs_dir)
    if snapshot["input_contract"]["feature_mode"] != "lineshape" or len(snapshot["runs"]) != 1:
        raise ValueError("Require one verified raw/reference lineshape run")
    snapshot["publication_exporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT/"docs/assets/lineshape-training.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False)+"\n")
    page = ROOT/"docs/matching.html"
    text = page.read_text()
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError("Require a unique lineshape-training publication block")
    page.write_text(re.sub(re.escape(START)+r".*?"+re.escape(END), lambda _: render_block(snapshot), text, flags=re.S))
    print("Verified and published lineshape network; test RMSE (pp):", 100*snapshot["runs"][0]["metrics"]["rmse"])


if __name__ == "__main__":
    main()
