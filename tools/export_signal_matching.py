#!/usr/bin/env python3
"""Publish all five verified experimental matches and a generated example."""
import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re

import numpy as np

from experimental_data import load_signal_csv
from baseline_data import load_baseline_csv
from match_experimental_signals import ROOT, digest, reconstruct
from generate_matched_data import validated_report, GENERATOR
from lineshape import spin_weights, pake_susceptibility


def verify_products(folder, dataset):
    report = validated_report(folder/"matching_report.json")
    raw, parsing = load_signal_csv(ROOT/"examples/Sample_RawSignal.csv")
    if parsing != report["signal_parsing"]:
        raise ValueError("The public raw file does not match the fitting input")
    if digest(ROOT/report["matching_config"]["component_setup"]) != report["component_setup_sha256"]:
        raise ValueError("Component setup changed since fitting")
    if digest(ROOT/"configs/experimental-matching.json") != report["config_sha256"]:
        raise ValueError("Matching configuration changed since fitting")
    if len(raw) != len(report["fits"]):
        raise ValueError("Publish every supplied experimental record")
    traces = []
    for row, fit in zip(raw, report["fits"]):
        data = np.loadtxt(folder/f"scan_{fit['record_1based']}.csv", delimiter=",", skiprows=1)
        if data.shape != (500, 6):
            raise ValueError("Expected all 500 rows in each saved match")
        f, measured, prediction, baseline, signal, residual = data.T
        expected_f = report["acquisition"]["start_mhz"]+np.arange(500)*report["acquisition"]["step_mhz"]
        np.testing.assert_allclose(f, expected_f, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(measured, row[1:])
        np.testing.assert_allclose(prediction, reconstruct(f, fit), atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(baseline, reconstruct(f, fit, False), atol=1e-11, rtol=1e-10)
        np.testing.assert_allclose(signal, prediction-baseline, atol=1e-14, rtol=0)
        np.testing.assert_allclose(residual, measured-prediction, atol=1e-14, rtol=0)
        np.testing.assert_allclose(np.sqrt(np.mean(residual**2)), fit["diagnostics"]["rms_recorded_units"], rtol=1e-10)
        traces.append(data)
    generated = dict(np.load(dataset, allow_pickle=False))
    generation = json.loads(dataset.with_suffix('.json').read_text())
    if generation["dataset_sha256"] != digest(dataset) or generation["matching_report_sha256"] != digest(folder/"matching_report.json"):
        raise ValueError("Generated dataset provenance mismatch")
    if str(generated["simulator"]) != GENERATOR:
        raise ValueError("Do not publish the unrelated 512-bin benchmark here")
    return report, traces, generated, generation


def save_figure(fig, path, title):
    fig.savefig(path, metadata={"Date": None, "Title": title})
    path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')


def draw_figures(report, traces, generated, assets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'nmr-ai-experimental-matching', 'figure.facecolor': '#fafaf6',
                         'axes.facecolor': '#fafaf6', 'axes.labelcolor': '#173c3a', 'text.color': '#173c3a'})
    def style(ax):
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.2)
        ax.tick_params(labelsize=10)
    f = traces[0][:, 0]
    fig, ax = plt.subplots(figsize=(10.5, 5))
    for fit, trace in zip(report['fits'], traces):
        ax.plot(f, trace[:, 1], lw=1.2, label=f"Scan {fit['record_1based']}")
    base, _ = load_baseline_csv(ROOT/'examples/deuteron-baseline.csv')
    ax.plot(f, base[0, 1:], 'k--', lw=1, label='Earlier baseline')
    ax.set(xlabel='Frequency (MHz)', ylabel='Recorded units', title='Five raw spin-1 sweeps · original backgrounds retained')
    ax.legend(ncol=3, fontsize=10); style(ax); fig.tight_layout()
    save_figure(fig, assets/'experimental-raw-signals.svg', 'Five measured raw spin-1 sweeps'); plt.close(fig)

    fig, axes = plt.subplots(len(traces), 2, figsize=(12, 14), sharex=True)
    for row, (fit, trace) in enumerate(zip(report['fits'], traces)):
        signal_view = (trace[:, 1]-trace[:, 3])*1e3
        axes[row, 0].scatter(f, signal_view, s=5, color='#267b71', alpha=.65, linewidths=0,
                            label='Measured − fitted circuit baseline', gid=f'measured-scan-{row+1}')
        axes[row, 0].plot(f, trace[:, 4]*1e3, '--', color='#b44c30', lw=1.4, label='Circuit-coupled Pake fit')
        axes[row, 0].set_ylabel('10⁻³ recorded units')
        axes[row, 0].set_title(f"Scan {row+1} · lineshape P = {100*fit['parameters']['P_model']:.2f}%", loc='left', fontsize=12)
        axes[row, 1].plot(f, trace[:, 5]*1e6, color='#267b71', lw=.7, marker='.', markersize=1.6)
        axes[row, 1].axhline(0, color='#b44c30', ls='--', lw=.8)
        axes[row, 1].set_ylabel('10⁻⁶ recorded units')
        axes[row, 1].set_title(f"Full-sweep residual · RMS {fit['diagnostics']['rms_recorded_units']*1e6:.2f}", loc='left', fontsize=12)
        for ax in axes[row]: style(ax)
    axes[0, 0].legend(loc='upper right', fontsize=8, frameon=False)
    for ax in axes[-1]: ax.set_xlabel('Frequency (MHz)')
    fig.suptitle('All 2,500 measured samples · no excluded fit bins', fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, .975])
    save_figure(fig, assets/'experimental-signal-matches.svg', 'All five Pake matches and full-sweep residuals'); plt.close(fig)

    first = report['fits'][0]['parameters']
    x = (f-first['center_mhz'])/first['split_mhz']
    total, plus, minus = pake_susceptibility(x, first['P_model'], first['eta'], first['g'])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(f, -plus.imag, label='Positive-frequency branch')
    axes[0].plot(f, -minus.imag, label='Negative-frequency branch')
    axes[0].plot(f, -total.imag, color='#173c3a', lw=1.5, label='Total absorption')
    wp, wm = spin_weights(first['P_model'])
    ratio = wp/wm
    axes[0].set(title=f'Scan 1 intrinsic branches · area ratio r = {ratio:.4f}', ylabel='−Im(χ) / amplitude scale')
    axes[0].legend(fontsize=9, frameon=False)
    axes[1].plot(f, total.real, color='#b44c30')
    axes[1].set(title='Dispersion enters the same circuit', ylabel='Re(χ) / amplitude scale')
    for ax in axes: style(ax); ax.set_xlabel('Frequency (MHz)')
    fig.tight_layout()
    save_figure(fig, assets/'experimental-pake-branches.svg', 'Intrinsic absorption branches and complex dispersion'); plt.close(fig)

    # Selection is fixed: the first simulated event, never the best-looking one.
    f = generated['frequency_mhz']; i = 0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))
    axes[0].plot(f, generated['signals'][i], color='#267b71', lw=1, label='Generated raw sweep')
    axes[0].plot(f, generated['clean_baseline'][i], '--', color='#b44c30', lw=1, label='Physical circuit baseline')
    axes[0].set(title=f"New simulator label P = {100*generated['P'][i]:.2f}%", ylabel='Recorded units')
    axes[0].legend(fontsize=9, frameon=False)
    axes[1].plot(f, (generated['signals'][i]-generated['baselines'][i])*1e3, color='#267b71', lw=.7,
                 label='Raw − independently noisy reference')
    axes[1].plot(f, generated['clean_signal'][i]*1e3, '--', color='#b44c30', lw=1.3, label='Clean coupled Pake signal')
    axes[1].set(title='Signal and independently drawn noise', ylabel='10⁻³ recorded units')
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes: style(ax); ax.set_xlabel('Frequency (MHz)')
    fig.tight_layout()
    save_figure(fig, assets/'matched-generator-example.svg', 'First generated event from experimental model seeds'); plt.close(fig)


def results_html(report, generation):
    rows = []
    for fit in report['fits']:
        p = fit['parameters']; d = fit['diagnostics']
        rows.append(f'<tr><td>{fit["record_1based"]}</td><td>{100*p["P_model"]:.3f}%</td>'
                    f'<td>{p["center_mhz"]:.6f}</td><td>{1e3*p["split_mhz"]:.3f}</td>'
                    f'<td>{1e3*p["g"]*p["split_mhz"]:.3f}</td><td>{p["eta"]:.4f}</td>'
                    f'<td>{fit["resolved_tune_capacitance_pf"]:.2f}</td><td>{d["rms_recorded_units"]*1e6:.2f}</td></tr>')
    noise = ''.join(f'<tr><td>{fit["record_1based"]}</td>'
                    f'<td>{fit["diagnostics"]["wing_second_difference_sigma_proxy"]*1e6:.2f}</td>'
                    f'<td>{fit["diagnostics"]["lag1"]:.3f}</td>'
                    f'<td>{fit["diagnostics"]["max_absolute_recorded_units"]*1e6:.1f}</td></tr>' for fit in report['fits'])
    unsuccessful = sum(not candidate['success'] for fit in report['fits'] for candidate in fit['candidates'])
    return f'''<section id="results" class="chapter">
<p class="eyebrow">Measured results · every supplied scan</p><h2>The fitted lineshapes, with the residuals visible.</h2>
<p>Each curve is fitted directly to all 500 raw samples. The left panels subtract only that fit's physical χ = 0 circuit response for display. The right panels show recorded minus full fitted voltage; no endpoints are cropped from the fit or statistics. Polarization comes from the spin-1 shape, not TE area calibration.</p>
<div class="baseline-figure-scroll" tabindex="0" role="region" aria-label="All five measured matches; scroll horizontally on small screens"><img class="baseline-fit-image" src="assets/experimental-signal-matches.svg" width="1200" height="1400" alt="All five experimental Pake signals after fitted circuit-background subtraction, alongside full-sweep residuals."></div>
<div class="table-wrap"><table><caption>Best full-sweep fit for each experimental record; P is a fitted estimate, not an uncertainty claim</caption>
<thead><tr><th>Scan</th><th>P from shape</th><th>Center (MHz)</th><th>Split scale (kHz)</th><th>Lorentzian HWHM (kHz)</th><th>EFG η</th><th>C_tune (pF)</th><th>RMS (10⁻⁶ units)</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>All selected fits converged without an active near-bound warning. There were {report['starts_per_scan']} starting points per scan; {unsuccessful} alternative start(s) reached the evaluation limit and remain in the saved report. Agreement among successful starts checks the optimizer—it is not a polarization confidence interval. The <a href="assets/experimental-matching.json">complete fit record</a> retains all candidates, parameter sources, the numerical conditioning and code/data hashes.</p>
<h3>Check the branches before interpreting detector peaks</h3>
<div class="baseline-figure-scroll" tabindex="0" role="region" aria-label="Intrinsic absorption and dispersion; scroll horizontally on small screens"><img class="baseline-fit-image" src="assets/experimental-pake-branches.svg" width="1100" height="450" alt="Intrinsic absorption branches and dispersion for the first fitted scan."></div>
<p>These intrinsic absorption branches have the fitted spin-temperature area ratio. The observed detector trace additionally contains dispersion mixing and the circuit transfer function. Do not replace the branch-area ratio with the ratio of two observed peak heights.</p>
<h3>What remains in the residual?</h3>
<div class="table-wrap"><table><caption>Noise-scale diagnostic and full-residual structure; all amplitudes in recorded units</caption><thead><tr><th>Scan</th><th>High-frequency σ proxy (10⁻⁶)</th><th>Full residual lag-one correlation</th><th>Maximum |residual| (10⁻⁶)</th></tr></thead><tbody>{noise}</tbody></table></div>
<p>The high-frequency proxy is much smaller than the full RMS. Residuals contain smooth mismatch and endpoint structure, not just random noise. The first generator uses the measured high-frequency scale in a white-Gaussian reference model; it does not claim that white noise reproduces every observed residual. Supply a characterized 500 × 500 covariance for colored Gaussian noise, and model reproducible acquisition artifacts separately.</p>
</section>
<section id="generated-example" class="chapter"><p class="eyebrow">From fitted examples to new labeled sweeps</p><h2>A generated event—not a copied experimental label.</h2>
<p>This example is event 1 from an actual run of {generation['num_samples']:,} newly generated spectra and {len(generation['configurations'])} varied configurations. Its polarization was sampled by the simulator. The experimentally fitted P values are not copied into the training target array.</p>
<div class="baseline-figure-scroll" tabindex="0" role="region" aria-label="Generated example; scroll horizontally on small screens"><img class="baseline-fit-image" src="assets/matched-generator-example.svg" width="1100" height="470" alt="First newly generated raw spectrum, physical baseline, and circuit-coupled signal with independent reference noise."></div>
<p>Raw voltage, clean χ = 0 baseline, clean nuclear response, event noise and independently noisy reference are stored separately. This is a 500-bin, recorded-unit, lineshape-regression dataset. It is not silently converted into the existing 512-bin TE-area benchmark.</p>
</section>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit-dir', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args()
    report, traces, generated, generation = verify_products(args.fit_dir, args.dataset)
    assets = ROOT/'docs/assets'
    draw_figures(report, traces, generated, assets)
    metadata = {**report, "publication": {"source_csv_url": 'https://github.com/zetanaut/NMR-AI/blob/main/examples/Sample_RawSignal.csv',
                "matching_report_sha256": digest(args.fit_dir/'matching_report.json'), "exporter_sha256": digest(__file__),
                "generation": generation, "displayed_generated_event_0based": 0,
                "figure_sha256": {name: digest(assets/name) for name in ('experimental-raw-signals.svg', 'experimental-signal-matches.svg',
                                                                        'experimental-pake-branches.svg', 'matched-generator-example.svg')},
                "scan_csv_sha256": {str(i+1): digest(args.fit_dir/f'scan_{i+1}.csv') for i in range(len(traces))}}}
    (assets/'experimental-matching.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    page = ROOT/'docs/matching.html'
    original = page.read_text()
    pattern = r'<!-- BEGIN MATCHING RESULTS -->.*?<!-- END MATCHING RESULTS -->'
    if len(re.findall(pattern, original, flags=re.S)) != 1:
        raise ValueError("Need exactly one matching-results marker pair")
    page.write_text(re.sub(pattern, '<!-- BEGIN MATCHING RESULTS -->\n'+results_html(report, generation)
                           +'\n<!-- END MATCHING RESULTS -->', original, flags=re.S))
    print('Published all five verified matches and the first newly generated event')


if __name__ == '__main__':
    main()
