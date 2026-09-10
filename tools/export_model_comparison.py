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
from polarization_metrics import polarization_profile

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


def draw_polarization_errors(snapshot, output):
    profile = snapshot['polarization_profile']
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'polarization-bands-v1', 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.edgecolor': '#d8dcd3',
                         'text.color': '#173c3a', 'axes.labelcolor': '#173c3a'}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.7), facecolor='#fffefa')
        fig.subplots_adjust(left=.085, right=.98, bottom=.21, top=.88, hspace=.42, wspace=.25)
        for row, sign in enumerate(('positive', 'negative')):
            for column, (key, label) in enumerate((('rmse_pp', 'Absolute RMSE (pp)'), ('relative_rms_percent', 'Relative RMS error (%)'))):
                ax = axes[row, column]
                ax.set_facecolor('#fffefa')
                for run, color in zip(profile['views'][sign], COLORS):
                    bands = [band for band in run['bands'] if band[key] is not None]
                    ax.plot([50*(band['low_inclusive']+band['high']) for band in bands],
                            [band[key] for band in bands], marker='o', ms=4, lw=1.5,
                            color=color, label=run['label'])
                ax.set_yscale('log')
                ax.set_xlim(0, 25)
                ax.set_xlabel('|True polarization| (%)')
                ax.set_ylabel(label+' · log scale')
                ax.set_title(('P ≥ 0' if sign == 'positive' else 'P < 0')+' · '+label, loc='left', fontsize=11)
                ax.grid(axis='y', color='#dce2d8')
        fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False)
        counts = []
        for sign, label in (('positive', '+'), ('negative', '−')):
            bands = profile['views'][sign][0]['bands']
            counts.append(f"P{label}: events "+', '.join(str(b['n']) for b in bands)+
                          '; configuration groups '+', '.join(str(b['n_groups']) for b in bands))
        fig.text(.085, .135, 'Band edges in |P| (%): 0, 1, 2.5, 5, 10, 15, 20, 25. Markers are band centers; lines guide the eye.', fontsize=9)
        fig.text(.085, .105, counts[0], fontsize=9)
        fig.text(.085, .08, counts[1], fontsize=9)
        fig.text(.085, .035, 'Same 600 saved test rows; no fixed-P experiment or uncertainty intervals. Bands share groups.\nRelative RMS uses each true P; omitted for bands touching zero. No denominator floor.', fontsize=9)
        fig.savefig(output, metadata={'Date': None, 'Description': profile['scope']})
        plt.close(fig)
    output = Path(output)
    output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines())+'\n')


def render_polarization_block(snapshot):
    profile = snapshot['polarization_profile']
    def number(value):
        return '—' if value is None else f'{value:.5g}'
    default_rows = ''.join(
        '<tr><td>'+html.escape(run['label'])+'</td>'+''.join(f'<td>{number(run["default_evaluation"][key])}</td>'
        for key in ('n', 'n_groups', 'bias_pp', 'rmse_pp', 'relative_rms_percent', 'p95_absolute_error_pp'))+'</tr>'
        for run in profile['views']['positive'])
    return f'''<div id="polarization-performance" class="subsection">
  <h3>How does model error change with polarization?</h3>
  <p>A pooled score can hide the weak-signal region you need to measure. These curves recompute errors from the same saved MLP, DNN and CNN test predictions in declared <strong>true-P bands</strong>. They do not rescale one pooled RMSE or simulate a new performance trend. Choose a sign and compare absolute with relative error.</p>
  <div class="results-panel">
    <div class="polarization-controls">
      <div><label for="polarization-sign">Polarization sign</label><select id="polarization-sign" disabled><option value="positive">Positive (P ≥ 0)</option><option value="negative">Negative (P &lt; 0)</option></select></div>
      <div><label for="polarization-metric">Error measure</label><select id="polarization-metric" disabled><option value="relative_rms_percent">Relative RMS error (%)</option><option value="rmse_pp">Absolute RMSE (pp)</option></select></div>
      <div class="controls"><label for="polarization-minimum">Lowest |P| scale of interest <output id="polarization-minimum-value" for="polarization-minimum">1.0%</output></label><input id="polarization-minimum" type="range" min="0" max="24.5" step="0.1" value="1" disabled></div>
    </div>
    <div id="polarization-live" hidden><svg id="polarization-chart" class="chart" viewBox="0 0 760 340" role="img" aria-label="Saved model errors versus true polarization magnitude"></svg>
    <p class="chart-caption" id="polarization-caption"></p></div>
    <p class="polarization-legend"><span>Rust: MLP</span><span>Sage: DNN</span><span>Dark green: CNN + summaries</span></p>
    <p id="polarization-band-label" aria-live="polite">Evaluate near +1%: 0.5% ≤ P &lt; 1.5%. This finite band is not a fixed-P experiment.</p>
    <div class="table-wrap" tabindex="0" role="region" aria-label="Selected polarization-band errors; scroll horizontally"><table><caption>Selected evaluation band · unchanged held-out predictions</caption><thead><tr><th>Model</th><th>Events</th><th>Configuration groups</th><th>Bias (pp)</th><th>RMSE (pp)</th><th>Relative RMS (%)</th><th>95th |error| (pp)</th></tr></thead><tbody id="polarization-band-table">{default_rows}</tbody></table></div>
    <p id="polarization-count-note">Inspect both event and independent configuration counts. Related rows are not independent acquisitions; these descriptive metrics have no uncertainty intervals.</p>
    <details id="polarization-static" open><summary>All polarization bands: printable figure and counts</summary><div class="details-body"><div class="model-chart-scroll" tabindex="0" role="region" aria-label="Polarization-dependent error figure; scroll horizontally"><img class="chart" src="assets/model-polarization.svg" width="1100" height="770" alt="Absolute and relative model errors in positive and negative polarization bands, with event and configuration counts. Trends differ between models and error measures."></div><p>Download the <a href="assets/model-polarization.svg">four-panel SVG</a> or the band metrics and all prediction rows in the <a href="assets/model-comparison.json">comparison record</a>. The absolute curves retain the 0–1% band. Relative curves omit bands touching zero; they do not floor denominators or discard rows from the absolute metrics.</p></div></details>
    <noscript><p>The default selected-band table and complete figure remain available above. Enable JavaScript to select another sign, metric or evaluation scale.</p></noscript>
  </div>
  <h4>Make the low end of your operating range an explicit evaluation point</h4>
  <p>At fixed absolute error, relative error increases as 1/|P|. At fixed noise and other physical settings, small |P| also weakens the spin-1 signal and its branch imbalance. Define the lowest nonzero |P| you need <strong>before choosing the model</strong>, collect enough validation cases near it, and report that band's error and counts. Repeat the check across the relevant signs, SNR and setup conditions, then evaluate the frozen design on a final holdout. The <a href="#error-lab">idealized scale demo</a> below isolates the 1/|P| effect.</p>
  <div id="polarization-diagnostic">
  <h4>A flat or rising relative-error curve needs investigation</h4>
  <p>With fixed additive noise, comparable setup conditions and a response that remains sensitive to P, increasing |P| should improve noise-limited relative precision. A persistent plateau or rise is therefore a <strong>warning of possible estimator underperformance or an unresolved limitation</strong>. The CNN's nearly flat relative-error curve here deserves that investigation; its small pooled RMSE does not establish that training is sufficient or that the available information has been fully used. Small fluctuations between sparsely sampled bands need not be significant.</p>
  <p>This saved benchmark uses <strong>fixed 10⁻⁹ V additive noise SD and ideal TE calibration</strong>; noise is not increased with P to hold SNR constant, and no calibration uncertainty was sampled. These assumptions make the missing improvement worth examining. The plotted bands still mix configurations and center shifts, so they do not identify the cause. More independent training coverage, better convergence, or changes to the model and training objective are candidates to test—not established cures.</p>
  <details><summary>How to diagnose the plateau and test improvements</summary><div class="details-body"><ol>
  <li>On a separate development set, compare fixed-P cases across both signs using the same setup distribution, fixed absolute noise and matched reference treatment. Inspect bias, width, RMSE and tails, with enough independent groups and repeated noise draws to distinguish a sustained trend from sampling variation.</li>
  <li>Use noise-free and repeated-noise simulations as diagnostics while keeping the normal model inputs. Error that persists without added noise requires checking approximation, nuisance ambiguity and preprocessing; it is not explained by additive detector noise alone.</li>
  <li>Hold validation arrays and groups fixed while increasing training rows and independent configurations. Refit all training-only preprocessing and ridge coefficients. Compare convergence, initialization seeds and model capacity; the <a href="https://github.com/zetanaut/NMR-AI/blob/main/docs/reference/training-design.md#measure-dataset-size-with-a-controlled-learning-curve">dataset-size study</a> explains how to determine whether more data help. The current trainer selects pooled absolute MSE; a local relative-error target also needs explicit validation criteria.</li>
  <li>Keep the lowest relevant |P| as a required evaluation point and verify the full operating range. Judge candidates by lower errors and the declared tolerances, not by forcing the plotted curve to fall. Use a fresh final holdout after development; these already-inspected test curves cannot be reused as an untouched final test.</li>
  </ol><p>A simple diagnostic model is <code>e = ε + δP</code>, with zero-mean additive error ε of fixed SD σₐ and constant fractional mismatch δ. At fixed nonzero P, <code>relative RMS = 100 × sqrt(σₐ²/P² + δ²)</code>. The additive term decreases, while an uncorrected scale error leaves a plateau. This is an illustrative derivation, not a fit to the CNN or a demonstrated noise floor. In measurements, independently established gain/calibration uncertainty could also leave a relative floor; more training alone would not remove missing calibration information. The saved benchmark does not include that calibration uncertainty.</p><p>These are proposed diagnostic comparisons, consistent with <a href="https://www.deeplearningbook.org/contents/guidelines.html">Deep Learning, chapter 11</a>. The current scores and curves remain unchanged; no data-size or noise-floor study has yet established their limiting cause.</p></div></details>
  </div>
  <details><summary>Definitions, band boundaries and limits</summary><div class="details-body"><p>For selected events, <code>e = P_true − P_pred</code>. Absolute RMSE is <code>100 × sqrt(mean(e²))</code> in pp. Relative RMS is <code>100 × sqrt(mean((e/P_true)²))</code> in percent: each event supplies its own denominator. This differs from dividing a band's RMSE by its midpoint, and from changing P₀ under a pooled score.</p><p>Curve edges in |P| are 0, 1, 2.5, 5, 10, 15, 20 and 25 percent, lower-inclusive and upper-exclusive except that 25% is included. Markers sit at band midpoints; their connecting lines do not imply fixed-P measurements or monotonic interpolation. The slider separately recomputes a band ±0.5 percentage points around its chosen magnitude, clipped to 0–25%, for the selected sign. The sign and magnitude inequalities define negative bands too.</p><p>Bands touching zero show absolute errors only. Near zero, relative errors can be dominated by tiny denominators; no hidden floor is used. Empty bands show unavailable metrics, not zero error. Sign and scale selections retain the original configuration IDs. Bands can share groups and differ in their noise/setup composition, so the curves do not isolate changing P with every nuisance fixed. Use paired fixed-P simulations for that controlled question, and independent acquisitions for experimental accuracy.</p></div></details>
  </div>'''


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
  {render_polarization_block(snapshot)}
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
    snapshot['polarization_profile'] = polarization_profile(snapshot)
    snapshot['polarization_metrics_sha256'] = digest(ROOT/'tools/polarization_metrics.py')
    assets = ROOT/"docs/assets"
    draw_errors(snapshot, assets/"model-comparison.svg")
    snapshot["figure_sha256"] = digest(assets/"model-comparison.svg")
    draw_polarization_errors(snapshot, assets/'model-polarization.svg')
    snapshot['polarization_figure_sha256'] = digest(assets/'model-polarization.svg')
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
