#!/usr/bin/env python3
"""Publish a verified measured-baseline overlay and residuals, without the raw CSV.

Select the median whole-scan RMS among distinct traces, rather than the best fit.
No refitting, bin removal, smoothing, or changes to the physical model occur.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from fit_tuned_baseline import reconstruct_fit
from baseline_parameters import parameter_table_html

ROOT = Path(__file__).resolve().parents[1]


def load_example(folder):
    report_file = folder/"fit_report.json"
    report = json.loads(report_file.read_text())
    if report.get("nucleus") != "deuteron" or not report.get("frequency_source", "").strip():
        raise ValueError("Require a fresh deuteron fit with an explicit acquisition frequency source; do not relabel an old fit")
    if not report["start_mhz"] <= 32.7 <= report["start_mhz"] + 499*report["step_mhz"]:
        raise ValueError("The saved sweep does not bracket the deuteron reference; check acquisition metadata")
    if report.get("fit_mode") != "tuning-informed" or not report.get("fits"):
        raise ValueError("Require a tuning-informed fit; unconstrained legacy reports must not be republished")
    for name, expected in report["code_sha256"].items():
        if hashlib.sha256((ROOT/"tools"/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"{name} changed since fitting; verify and regenerate the fit report")
    traces, summaries = [], []
    for index, fit in enumerate(report["fits"]):
        filename = folder/f"trace_{index}.csv"
        data = np.loadtxt(filename, delimiter=",", skiprows=1)
        if data.shape != (500, 4) or not np.isfinite(data).all():
            raise ValueError("Expected 500 finite frequency/recorded/fit/residual rows")
        frequency, measured, prediction, residual = data.T
        np.testing.assert_allclose(frequency, report["start_mhz"]+np.arange(500)*report["step_mhz"], atol=1e-10, rtol=0)
        np.testing.assert_allclose(residual, measured-prediction, atol=1e-14, rtol=1e-10)
        if fit.get("fit_mode") != "tuning-informed":
            raise ValueError("Require a tuning-informed fit; unconstrained legacy reports must not be republished")
        reconstructed = reconstruct_fit(frequency*1e6, fit)
        np.testing.assert_allclose(prediction, reconstructed, atol=1e-12, rtol=1e-10)
        rms = float(np.sqrt(np.mean(residual**2)))
        span = float(np.ptp(measured))
        np.testing.assert_allclose(rms, fit["rmse_recorded_units"], rtol=1e-10)
        summaries.append({"trace_number": index+1, "bins": len(data), "rms_recorded_units": rms,
                          "peak_to_peak_recorded_units": span, "rms_percent_of_peak_to_peak": 100*rms/span,
                          "max_absolute_residual_recorded_units": float(np.max(np.abs(residual))),
                          "trace_csv_sha256": hashlib.sha256(filename.read_bytes()).hexdigest()})
        traces.append(data)
    selected = int(np.argsort([s["rms_recorded_units"] for s in summaries], kind="stable")[len(summaries)//2])
    metadata = {
        "selection": "Only trace in the supplied baseline file; all 500 bins" if len(traces) == 1 else "Median whole-scan residual RMS among distinct traces (upper median for even counts)",
        "selected_trace_number": selected+1,
        "source_sha256": report["source_sha256"],
        "nucleus": report["nucleus"], "reference_hz": report["reference_hz"],
        "start_mhz": report["start_mhz"], "step_mhz": report["step_mhz"],
        "frequency_source": report["frequency_source"],
        "fit_report_sha256": hashlib.sha256(report_file.read_bytes()).hexdigest(),
        "fit_code_sha256": report["code_sha256"],
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "parameter_catalog_sha256": hashlib.sha256(Path(__file__).with_name("baseline_parameters.py").read_bytes()).hexdigest(),
        "fit_mode": report["fit_mode"],
        "source_rows": report["rows"], "unique_rows": report["unique_rows"],
        "voltage_unit": "recorded units; DAQ conversion not assumed",
        "residual_sign": "recorded minus fitted", "fitted_bin_mask": report["fitted_bin_mask"],
        "starts": report["starts"], "seed": report["seed"], "summaries": summaries,
        "selected_fit": {key: report["fits"][selected][key] for key in
                         ("circuit", "quadrature_coefficients", "scaled_shape_jacobian_condition", "active_bounds")},
        "note": "Measured-fit illustration, not a synthetic trace or held-out accuracy test. "
                "This metadata omits timestamps; source-CSV publication is handled separately. All 500 measured bins appear in the figure. "
                "Fit agreement does not establish unique hardware parameters or electronic-noise variance.",
    }
    for key in ("acquisition", "bounds_pf_m_pf", "loss", "input_parsing"):
        if key in report:
            metadata[key] = report[key]
    public_csv = ROOT/"examples/deuteron-baseline.csv"
    if public_csv.exists() and hashlib.sha256(public_csv.read_bytes()).hexdigest() == report["source_sha256"]:
        metadata["source_csv_url"] = "https://github.com/zetanaut/NMR-AI/blob/main/examples/deuteron-baseline.csv"
    chosen = report["fits"][selected]
    metadata["selected_fit"].update({key: chosen[key] for key in
                                    ("shape_fit_success", "multistart_candidates", "residual_lag1")})
    if chosen.get("fit_mode") == "tuning-informed":
        metadata["selected_fit"].update({key: chosen[key] for key in
                                         ("fit_mode", "resolved_parameters", "setup", "free_parameters", "parameter_manifest",
                                          "nonlinear_parameters", "profiled_parameters", "n_free", "jacobian_scope",
                                          "active_bound_parameters", "near_bounds", "near_bound_fraction_of_search_interval",
                                          "numerical_derivative", "data_objective", "constraint_objective", "objective_kind")})
    return traces[selected], metadata


def draw_example(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none",
                         "svg.hashsalt": "nmr-ai-baseline-example", "axes.labelcolor": "#173c3a",
                         "text.color": "#173c3a", "xtick.color": "#53645f", "ytick.color": "#53645f"})
    f, measured, prediction, residual = data.T
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1], "hspace": .33})
    fig.patch.set_facecolor("#fafaf6")
    for ax in axes:
        ax.set_facecolor("#fafaf6")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#bac8bf")
        ax.grid(axis="y", color="#dce2d8", linewidth=.7)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=11)
        ax.set_xlim(f[0]-.004, f[-1]+.004)
    axes[0].scatter(f, measured, s=8, color="#267b71", alpha=.65, linewidths=0,
                    label="Measured · 500 bins", zorder=3)
    axes[0].plot(f, prediction, color="#b44c30", linewidth=1.7, linestyle=(0, (5, 3)),
                 label="Physical circuit fit", zorder=4)
    axes[0].set_title("A   Measured baseline and fitted Q-curve", loc="left", fontsize=14, pad=15)
    axes[0].set_ylabel("Recorded units")
    axes[0].yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    axes[0].legend(loc="lower center", frameon=False, fontsize=11)
    axes[1].axhline(0, color="#b44c30", linewidth=1, linestyle="--")
    axes[1].plot(f, residual*1e6, color="#267b71", linewidth=.8, marker=".", markersize=2.2)
    axes[1].set_title("B   Residuals · recorded minus fitted", loc="left", fontsize=14, pad=15)
    axes[1].set_ylabel("Residual (10⁻⁶ recorded units)")
    axes[1].set_xlabel("Frequency (MHz)", labelpad=9)
    axes[1].xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[1].xaxis.set_major_locator(MaxNLocator(6))
    axes[1].margins(y=.15)
    fig.subplots_adjust(left=.145, right=.975, top=.92, bottom=.12)
    fig.savefig(output, metadata={"Date": None, "Title": "Measured baseline and physical Q-meter fit",
                                 "Description": "All 500 measured bins and the saved physical circuit fit; residuals shown separately."})
    output.write_text("\n".join(line.rstrip() for line in output.read_text().splitlines())+"\n")
    plt.close(fig)


def example_html(metadata):
    m = metadata["summaries"][metadata["selected_trace_number"]-1]
    rms_millionths = m["rms_recorded_units"]*1e6
    count = metadata["unique_rows"]
    selection = ("The supplied baseline file contains one measured scan; no selection among traces was made." if count == 1 else
                 "This example is selected by the median whole-scan residual RMS, not the minimum.")
    trace_label = "The supplied deuteron baseline" if count == 1 else f'Trace {m["trace_number"]} · median residual RMS among {count} distinct sweeps'
    duplicates = metadata["source_rows"] - count
    optimization = ("All inputs are fixed; no optimizer is run." if metadata["selected_fit"].get("free_parameters") == []
                    else f'The fit uses {metadata["starts"]} starting points.')
    source_link = ('<a href="'+metadata['source_csv_url']+'">Public test CSV</a> (published with the owner’s authorization).'
                   if "source_csv_url" in metadata else 'Source-CSV publication is separate from this figure export.')
    from html import escape
    bound_warning = ''
    near = metadata['selected_fit'].get('near_bounds', {})
    if near:
        boundaries = ', '.join(escape(name)+' ('+escape(side)+')' for name, side in near.items())
        bound_warning = ('<aside class="note"><h3>Constraint warning: this is not an accepted hardware calibration</h3>'
                         '<p>The fit reaches or approaches these declared limits: '+boundaries+'. '
                         'Do not widen the known cable branch/trim to absorb model mismatch. '
                         'Check independent capacitor, coil, cable-loss and detector records; '
                         'correlated residuals and compensating readout coefficients require investigation.</p></aside>')
    rows = "".join(f'<tr><td>Trace {s["trace_number"]}{" · shown" if s is m else ""}</td>'
                   f'<td>{s["rms_recorded_units"]*1e6:.3f}</td><td>{s["rms_percent_of_peak_to_peak"]:.4f}%</td></tr>'
                   for s in metadata["summaries"])
    return f'''<section id="example" class="chapter">
<p class="eyebrow">Measured example · the physical circuit in action</p>
<h2>Fit a real baseline with its known tuning constraints.</h2>
<p>The green points are an experimental sweep; the dashed rust curve is the saved physical Q-meter fit. Compare them in the upper panel; the lower panel expands the residual scale so the remaining differences stay visible.</p>
{bound_warning}
<div class="results-panel baseline-example">
<div class="panel-heading"><div><p class="eyebrow">{trace_label}</p><h3>Measured data → circuit fit → residuals</h3></div><span class="badge">All {m["bins"]} bins</span></div>
<figure>
<div class="baseline-figure-scroll" tabindex="0" role="region" aria-label="Measured baseline plot; scroll horizontally on narrow screens">
<img class="baseline-fit-image" src="assets/baseline-example.svg" width="950" height="720" alt="Five hundred measured points and the physical Q-curve fit. A separately scaled residual panel retains endpoint deviations. Residual RMS is {rms_millionths:.3f} millionths of one recorded unit.">
</div>
<figcaption class="chart-caption">Upper: measured points and fitted circuit response in the original recorded units. Lower: recorded minus fitted, in 10⁻⁶ recorded units—not microvolts. All bins, including both endpoints, enter the fit, plot, and statistics. On a narrow screen, scroll the plot or <a href="assets/baseline-example.svg">open it full-size</a>.</figcaption>
</figure>
<div class="baseline-fit-stats"><div><span>Residual RMS</span><strong>{rms_millionths:.3f} × 10⁻⁶</strong><small>recorded units</small></div><div><span>RMS / peak-to-peak range</span><strong>{m["rms_percent_of_peak_to_peak"]:.4f}%</strong><small>shape-fit error, not polarization error</small></div><div><span>Fitted samples</span><strong>{m["bins"]} / {m["bins"]}</strong><small>no smoothing or excluded bins</small></div></div>
<p>{selection} The largest absolute residual is {m["max_absolute_residual_recorded_units"]*1e6:.2f} × 10⁻⁶ recorded units; the endpoint deviations have not been cropped out. {optimization} It uses the same circuit implementation used by the generator.</p>
{parameter_table_html(metadata)}
<details><summary>Whole-scan fit statistics</summary><div class="details-body"><div class="table-wrap"><table><caption>Whole-scan residuals; {duplicates} exact duplicate records omitted</caption><thead><tr><th>Sweep</th><th>RMS (10⁻⁶ recorded units)</th><th>RMS / peak-to-peak</th></tr></thead><tbody>{rows}</tbody></table></div><p>The <a href="#diagnostics">diagnostics section</a> explains why residual structure must not automatically be treated as electronic noise.</p></div></details>
<p class="provenance">Computed directly from the saved measured-fit CSVs; the fitted curve is checked against its stored circuit and detector coefficients before export. <a href="assets/baseline-example.json">Download the fit settings and statistics</a>. {source_link} A close baseline fit does not uniquely measure each component or establish polarization accuracy.</p>
</div>
</section>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True, help="Fresh fit directory with confirmed deuteron acquisition mapping")
    args = parser.parse_args()
    data, metadata = load_example(args.fit_dir)
    page = ROOT/"docs/baseline.html"
    original = page.read_text()
    pattern = r"<!-- BEGIN BASELINE EXAMPLE -->.*?<!-- END BASELINE EXAMPLE -->"
    if len(re.findall(pattern, original, flags=re.S)) != 1:
        raise ValueError("Expected exactly one baseline-example marker pair")
    asset = ROOT/"docs/assets/baseline-example.svg"
    draw_example(data, asset)
    metadata["svg_sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()
    (ROOT/"docs/assets/baseline-example.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    page.write_text(re.sub(pattern, "<!-- BEGIN BASELINE EXAMPLE -->\n"+example_html(metadata)
                           +"\n<!-- END BASELINE EXAMPLE -->", original, flags=re.S))
    print(f"Published measured trace {metadata['selected_trace_number']}, all 500 bins; {metadata['selection']}")


if __name__ == "__main__":
    main()
