#!/usr/bin/env python3
"""Verify and publish the same-data MLP/DNN/CNN teaching comparison."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_predictions import summarize
from learning_data import SIMULATOR, load_dataset, validate_contract
from nmr_lab import FREQUENCY, build_model, group_split, make_features

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"mlp": "Basic MLP", "dnn": "Deep dense DNN", "physics_multiscale": "Multiscale CNN + summaries"}
COLORS = ("#b44c30", "#57786e", "#173c3a")
START, END = "<!-- MODEL COMPARISON START -->", "<!-- MODEL COMPARISON END -->"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_comparison(protocol_path, runs_dir=None):
    """Rebuild inputs, partitions, saved predictions and metrics independently."""
    protocol = json.loads(protocol_path.read_text())
    runs_dir = runs_dir or ROOT/protocol["run_directory"]
    dataset_path = ROOT/protocol["dataset"]
    dataset_hash = digest(dataset_path)
    if dataset_hash != protocol["dataset_sha256"]:
        raise ValueError("Dataset differs from the declared comparison protocol")
    if json.loads((runs_dir/"protocol.json").read_text()) != protocol:
        raise ValueError("Saved training protocol differs from the publication protocol")
    evaluation = json.loads((runs_dir/"evaluation.json").read_text())
    if evaluation["protocol_sha256"] != digest(protocol_path) or evaluation["predictor_sha256"] != digest(ROOT/"tools/predict.py"):
        raise ValueError("Evaluation protocol or prediction code differs from its saved record")
    data = load_dataset(dataset_path, training=True)
    contract = data["contract"]
    group_column = contract["group_column"]
    groups, labels, features = data["identifiers"][group_column], data["P"], data["features"]
    partitions = dict(zip(("train", "validation", "test"), group_split(groups, protocol["training"]["seed"])))
    mean = features[partitions["train"]].mean(axis=(0, 2), keepdims=True)
    std = features[partitions["train"]].std(axis=(0, 2), keepdims=True)
    std[std < 1e-8] = 1
    target_scale = max(float(labels[partitions["train"]].std()), 1e-6)
    inputs = ((features-mean)/std).astype(np.float32)
    runs = []
    torch.set_num_threads(1)
    for architecture in protocol["architectures"]:
        folder = runs_dir/architecture
        config = json.loads((folder/"config.json").read_text())
        if config["architecture"] != architecture or config["dataset_sha256"] != dataset_hash:
            raise ValueError("Run architecture or dataset mismatch")
        if any(config[key] != value for key, value in protocol["training"].items()):
            raise ValueError("Training settings differ between comparison runs")
        with np.load(folder/"partition.npz", allow_pickle=False) as saved:
            for name, indices in partitions.items():
                np.testing.assert_array_equal(saved[name], indices)
        with np.load(folder/"scaler.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["frequency_mhz"], FREQUENCY)
            if str(saved["feature_mode"].item()) != contract["feature_mode"] or str(saved["voltage_unit"].item()) != contract["voltage_unit"]:
                raise ValueError("Saved preprocessing mode/units mismatch")
            np.testing.assert_array_equal(saved["feature_mean"], mean)
            np.testing.assert_array_equal(saved["feature_std"], std)
            if float(saved["target_scale"][0]) != target_scale:
                raise ValueError("Target scaler was not fitted on the common training rows")
        provenance = json.loads((folder/"provenance.json").read_text())
        if provenance["dataset_sha256"] != dataset_hash or provenance["partition_sha256"] != digest(folder/"partition.npz"):
            raise ValueError("Run provenance does not match its data or partition")
        for name, expected in provenance["code_sha256"].items():
            if digest(ROOT/"tools"/name) != expected:
                raise ValueError(f"Training source changed since fitting: {name}")
        checkpoint = torch.load(folder/"model.pt", map_location="cpu", weights_only=True)
        if evaluation["checkpoint_sha256"][architecture] != digest(folder/"model.pt"):
            raise ValueError("Checkpoint changed after the comparison was fixed")
        if checkpoint["architecture"] != architecture or checkpoint["simulator"] != data["simulator"] or checkpoint["input_contract"] != contract:
            raise ValueError("Checkpoint contract mismatch")
        model = build_model(architecture).eval()
        model.load_state_dict(checkpoint["model_state"])
        counts = (sum(p.numel() for p in model.parameters()), sum(p.numel() for p in model.parameters() if p.requires_grad))
        # Restoring a frozen ridge head also restores its behavior; requires_grad is not in state_dict.
        if architecture == "physics_multiscale":
            counts = (counts[0], counts[0]-26)
        if counts != (provenance["parameters"], provenance["trainable_parameters"]):
            raise ValueError("Parameter counts disagree with the trained architecture")
        with (folder/"history.csv").open(newline="") as stream:
            history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        best = min(history, key=lambda row: row["val_loss"])
        if int(best["epoch"]) != provenance["best_epoch"]:
            raise ValueError("Saved best epoch differs from minimum validation MSE")
        results, max_differences, reconstruction_devices, cpu_differences = {}, {}, {}, {}
        for name in ("validation", "test"):
            indices = partitions[name]
            rows = np.atleast_1d(np.genfromtxt(folder/f"{name}_predictions.csv", delimiter=",", names=True))
            np.testing.assert_array_equal(rows["P_true"], labels[indices])
            np.testing.assert_array_equal(rows[group_column], groups[indices])
            def reconstruct(device):
                model.to(device)
                with torch.inference_mode():
                    return np.concatenate([
                        model(torch.from_numpy(inputs[indices[i:i+128]]).to(device)).cpu().numpy()[:, 0]*target_scale
                        for i in range(0, len(indices), 128)])

            reproduced = reconstruct("cpu")
            cpu_differences[name] = float(np.max(np.abs(rows["P_pred"]-reproduced)))
            reconstruction_devices[name] = "cpu"
            if not np.allclose(rows["P_pred"], reproduced, rtol=1e-6, atol=1e-7):
                # Training saves validation predictions on its training device; the
                # declared comparison evaluates test rows on CPU. Verify roundoff
                # sensitive validation results on the original device, retaining
                # the independent CPU difference as a separate diagnostic.
                if name != "validation" or provenance["device"] != "cuda":
                    raise ValueError(f"Saved {name} predictions do not reconstruct on CPU")
                if not torch.cuda.is_available():
                    raise ValueError("CUDA is required to verify these saved validation predictions at the declared tolerance")
                reproduced = reconstruct("cuda")
                reconstruction_devices[name] = "cuda"
            np.testing.assert_allclose(rows["P_pred"], reproduced, rtol=1e-6, atol=1e-7)
            max_differences[name] = float(np.max(np.abs(rows["P_pred"]-reproduced)))
            results[name] = rows
        rows = results["test"]
        metrics = summarize(rows["P_true"], rows["P_pred"], .05)
        validation = summarize(results["validation"]["P_true"], results["validation"]["P_pred"], .05)
        np.testing.assert_allclose(validation["rmse"], np.sqrt(best["val_loss"])*target_scale, rtol=1e-4, atol=1e-8)
        local = (rows["P_true"] >= .045) & (rows["P_true"] < .055)
        runs.append({
            "architecture": architecture, "label": LABELS[architecture], "parameters": counts[0],
            "trainable_parameters": counts[1], "metrics": metrics, "validation_metrics": validation,
            "near_five_percent": summarize(rows["P_true"][local], rows["P_pred"][local], .05),
            "history": history, "best_epoch": int(best["epoch"]), "target_scale": target_scale,
            "test_predictions": np.column_stack([rows["P_true"], rows["P_pred"], rows[group_column]]).tolist(),
            "prediction_columns": ["P_true", "P_pred", group_column],
            "provenance": provenance, "reconstruction_max_difference_fractional_p": max_differences,
            "reconstruction_device": reconstruction_devices,
            "cpu_reconstruction_max_difference_fractional_p": cpu_differences,
            "artifact_sha256": {name: digest(folder/name) for name in
                                ("model.pt", "scaler.npz", "partition.npz", "history.csv", "test_predictions.csv", "validation_predictions.csv")},
        })
    return {
        "version": protocol["version"], "simulator": data["simulator"], "protocol": protocol,
        "input_contract": contract, "group_column": group_column,
        "protocol_sha256": digest(protocol_path), "exporter_sha256": digest(__file__),
        "dataset_sha256": dataset_hash, "events": len(labels), "bins": len(FREQUENCY),
        "rows": {name: len(indices) for name, indices in partitions.items()},
        "group_values": {name: np.unique(groups[indices]).tolist() for name, indices in partitions.items()},
        "test_row_indices": partitions["test"].tolist(), "runs": runs, "evaluation": evaluation,
        "units": "P is fractional; displayed absolute errors multiply by 100 to give percentage points",
        "residual_definition": "P_true - P_pred", "width_definition": "Population standard deviation (ddof=0)",
        "reference_p0": .05,
        "limitations": [
            "One training seed and a common budget; no claim that depth or convolution always improves accuracy",
            "CNN includes a train-only ridge fit to 25 physical summaries; gains are not attributable to convolutions alone",
            ("Ideal TE calibration and independently noisy, stable references; no experimental accuracy or calibration claim"
             if contract["feature_mode"] == "te_area" else
             "Recorded-unit raw/reference inputs without TE calibration; source-scan grouped synthetic holdout, not experimental accuracy"),
            "Controlled sensitivity coverage; no measured material-specific accuracy",
        ],
    }


def draw_errors(snapshot, output):
    runs = snapshot["runs"]
    values = [100*run["metrics"]["rmse"] for run in runs]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none",
                         "svg.hashsalt": snapshot["version"],
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#d8dcd3", "text.color": "#173c3a",
                         "axes.labelcolor": "#173c3a", "xtick.color": "#5c6867", "ytick.color": "#5c6867"}):
        fig, ax = plt.subplots(figsize=(9.5, 4.2), layout="constrained", facecolor="#fffefa")
        ax.set_facecolor("#fffefa")
        ax.plot(range(3), values, color="#b4c5b5", linewidth=2, zorder=2)
        for i, (value, color) in enumerate(zip(values, COLORS)):
            ax.scatter(i, value, color=color, s=85, zorder=3)
            ax.annotate(f"{value:.4g} pp", (i, value), xytext=(0, 15), textcoords="offset points",
                        ha="center", color=color, weight="bold")
        ax.set_yscale("log")
        ax.set_ylim(min(values)/2.2, max(values)*2.6)
        ax.set_xlim(-.45, 2.45)
        ax.set_xticks(range(3), ["Basic MLP", "Deep dense DNN", "Multiscale CNN\n+ physical summaries"])
        ax.set_ylabel("Held-out RMSE (percentage points) · log scale")
        ax.grid(axis="y", which="major", color="#dce2d8", zorder=0)
        ax.set_title(f"Same {snapshot['rows']['test']} test spectra · lower error is better", loc="left", pad=28)
        fig.savefig(output, metadata={"Date": None, "Description": "All test events enter RMSE. Logarithmic error axis; no uncertainty intervals."})
        plt.close(fig)
    output = Path(output)
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines())+"\n")


def render_block(snapshot):
    runs = snapshot["runs"]
    row_html = "".join(
        f"<tr><td>{html.escape(run['label'])}</td>"
        f"<td>{100*run['validation_metrics']['rmse']:.5f}</td>"
        + "".join(f"<td>{100*run['metrics'][key]:.5f}</td>" for key in ("rmse", "bias", "width", "p95_absolute_error"))
        + "</tr>" for run in runs)
    reductions = [runs[i-1]["metrics"]["rmse"]/runs[i]["metrics"]["rmse"] for i in (1, 2)]
    if min(reductions) > 1:
        finding = (f"In this run, the DNN reduces test RMSE by a factor of <strong>{reductions[0]:.2f}</strong> "
                   f"relative to the basic MLP. The CNN with physical summaries reduces it by a further "
                   f"factor of <strong>{reductions[1]:.2f}</strong>.")
    else:
        finding = "The measured errors below show how these estimators compare under this training budget; complexity alone does not guarantee an improvement."
    options = "".join(f'<option value="{run["architecture"]}">{html.escape(run["label"])}</option>' for run in runs)
    return f'''{START}
  <div id="model-comparison" class="subsection">
  <h3>One dataset. Three measured errors.</h3>
  <p>All three models use the same 6,000 broad-variation spectra: {snapshot['rows']['train']:,} training,
  {snapshot['rows']['validation']:,} validation, and {snapshot['rows']['test']:,} test events. The same 240/30/30
  configuration groups, two input channels, training-only scalers, batch-order seed, AdamW settings,
  and 80-epoch cap are used throughout. Each checkpoint minimizes validation MSE; the test rows are
  evaluated after fixing the comparison.</p>
  <figure class="architecture"><div class="model-chart-scroll" tabindex="0" role="region" aria-label="Test error chart; scroll horizontally on small screens"><img class="chart" src="assets/model-comparison.svg"
  alt="Held-out RMSE for the basic MLP, deep dense DNN and multiscale CNN; exact values are in the following table."
  width="950" height="420"></div><figcaption>Absolute error in percentage points (pp), on a logarithmic axis.
  All 600 test events enter each value. One seed; points have no uncertainty intervals.</figcaption></figure>
  <p>{finding}</p>
  <div class="table-wrap"><table><caption>Same-data comparison · all errors in percentage points · residual = truth − prediction</caption>
  <thead><tr><th>Estimator</th><th>Validation RMSE</th><th>Test RMSE</th><th>Test bias</th><th>Test width</th><th>Test 95th |error|</th></tr></thead>
  <tbody id="model-comparison-table">{row_html}</tbody></table></div>
  <p>The CNN's result includes the contribution of its ridge estimate from physical summaries.
  This comparison measures the complete estimators. It does not isolate convolution, establish a universal
  ordering, or measure experimental polarization accuracy. The generator assumes ideal TE calibration
  and a stable, independently noisy reference.</p>
  <div class="results-panel"><div class="panel-heading"><div><p class="eyebrow">Watch learning on the same data</p>
  <label for="model-curve-select" class="select-label">Learning curves</label>
  <select id="model-curve-select"><option value="all">Compare all validation curves</option>{options}</select></div>
  <span class="badge">Saved training histories</span></div>
  <div class="model-chart-scroll" tabindex="0" role="region" aria-label="Learning curve chart; scroll horizontally on small screens"><svg id="model-curve-chart" class="chart" viewBox="0 0 850 330" role="img" aria-label="Saved model learning curves"></svg></div>
  <p id="model-curve-caption" class="chart-caption">Enable JavaScript to explore the saved learning curves. The error figure and table above remain available.</p></div>
  <p class="provenance">Verified against saved checkpoints, exact partition rows, training-only preprocessing,
  and prediction CSVs. Download the <a href="assets/model-comparison.json">comparison record</a> or
  <a href="assets/model-comparison.svg">error figure</a>. The record includes every synthetic test prediction,
  hashes, settings, histories, parameter counts, and numerical reconstruction checks.</p>
  </div>
{END}'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT/"configs/model-comparison.json")
    parser.add_argument("--runs-dir", type=Path)
    args = parser.parse_args()
    snapshot = verify_comparison(args.protocol, args.runs_dir)
    assets = ROOT/"docs/assets"
    draw_errors(snapshot, assets/"model-comparison.svg")
    snapshot["figure_sha256"] = digest(assets/"model-comparison.svg")
    (assets/"model-comparison.json").write_text(json.dumps(snapshot, indent=2, allow_nan=False)+"\n")
    (assets/"model-comparison.js").write_text("// Generated by tools/export_model_comparison.py.\nwindow.MODEL_COMPARISON = "
                                             +json.dumps(snapshot, separators=(",", ":"), allow_nan=False)+";\n")
    page = ROOT/"docs/index.html"
    updated, count = re.subn(re.escape(START)+r".*?"+re.escape(END), lambda _: render_block(snapshot), page.read_text(), flags=re.S)
    if count != 1:
        raise ValueError("Expected one model comparison block in the tutorial")
    page.write_text(updated)
    for run in snapshot["runs"]:
        print(run["label"], "test RMSE (pp):", 100*run["metrics"]["rmse"])
    print("Verified and exported the MLP/DNN/CNN comparison")


if __name__ == "__main__":
    main()
