#!/usr/bin/env python3
"""Publish the butanol comparison and UVA-ND3 practical from verified local fits."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy

from baseline_data import acquisition_grid, load_acquisition
from experimental_data import load_signal_csv
from match_butanol import reconstruct as butanol_reconstruct, susceptibility
from match_experimental_signals import digest, reconstruct as single_reconstruct
from match_uva_nd3 import reconstruct as nd3_reconstruct
from uva_nd3_data import load_nd3

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/"docs/assets"
PREVIEW_DIR=None
REUSE_FIGURES=False
FIGURE_PROVENANCE={}


def single_site_reference(report):
    """Resolve the immutable snapshot used by the saved two-site comparison."""
    for name in ("experimental-matching.json", "experimental-matching-single-site-reference.json"):
        path=ASSETS/name
        if path.exists() and digest(path)==report["single_site_report_sha256"]:
            return path
    raise ValueError("Cannot find the pinned single-site comparison snapshot")


def verify_reused_figures(report, asset_name):
    if not REUSE_FIGURES:
        return
    previous=json.loads((ASSETS/asset_name).read_text())
    if {key:value for key,value in previous.items() if key!="publication"}!=report:
        raise ValueError("Existing figures describe a different fit report")
    publication=previous["publication"]
    for name,expected in publication["figure_sha256"].items():
        if digest(ASSETS/name)!=expected:
            raise ValueError(f"Existing figure changed: {name}")
        FIGURE_PROVENANCE[name]=publication.get("figure_environment",publication["validation_environment"])


def read_verified(path):
    report=json.loads(path.read_text())
    for name,expected in report["code_sha256"].items():
        if digest(ROOT/"tools"/name)!=expected:
            raise ValueError(f"Source changed since fitting: {name}")
    return report


def save(fig,name,title):
    if REUSE_FIGURES:
        if name not in FIGURE_PROVENANCE:
            raise ValueError(f"Unverified figure reuse: {name}")
        plt.close(fig)
        return
    fig.savefig(ASSETS/name,metadata={"Date":None,"Title":title})
    if PREVIEW_DIR is not None:
        fig.savefig(PREVIEW_DIR/Path(name).with_suffix(".png"),dpi=120)
    p=ASSETS/name
    p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines())+"\n")
    plt.close(fig)


def style(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",alpha=.2)
    ax.tick_params(labelsize=10)


def page(title,number,lede,content):
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="description" content="{title}"><meta name="theme-color" content="#173c3a"><title>NMR / AI — {title}</title><link rel="stylesheet" href="assets/style.css"></head>
<body><a class="skip-link" href="#main">Skip to the practical</a>
<header class="topbar"><a class="brand" href="index.html">NMR <span>/</span> AI</a><span class="topbar-note">Experimental example {number} of 3</span><a class="repo-link" href="https://github.com/zetanaut/NMR-AI">View on GitHub ↗</a></header>
<div class="layout"><aside class="sidebar"><p class="eyebrow">Three experimental examples</p><nav aria-label="Experimental examples"><a href="matching.html">1 · Single-site starting model</a><a href="butanol.html">2 · Butanol: C–D and O–D</a><a href="uva-nd3.html">3 · UVA-ND3 data</a><a href="#model">Model and assumptions</a><a href="#results">Fits and full residuals</a><a href="#reproduce">Reproduce the example</a><a href="#limits">Interpretation and next steps</a><a href="baseline.html">Physical-baseline practical</a><a href="index.html#network">500-bin learning models</a></nav><p class="sidebar-footer">All examples use 500 bins. Match the material and readout.</p></aside>
<main id="main"><section class="hero"><p class="eyebrow">Experimental example {number} of 3</p><h1>{title}</h1><p class="lede">{lede}</p></section>
{content}
<footer><a class="brand" href="index.html">NMR <span>/</span> AI</a><p>Material-specific lineshapes · measured data · reproducible comparisons</p></footer></main></div></body></html>
'''


def publish_butanol(folder):
    report=read_verified(folder/"butanol_report.json")
    reference=single_site_reference(report)
    single=json.loads(reference.read_text())
    verify_reused_figures(report,"butanol-matching.json")
    if digest(ROOT/"configs/butanol-matching.json")!=report["config_sha256"]:
        raise ValueError("Butanol fitting configuration changed")
    records,audit=load_signal_csv(ROOT/"examples/Sample_RawSignal.csv")
    if audit!=report["signal_parsing"] or len(report["fits"])!=5:
        raise ValueError("Publish all five supplied butanol scans")
    f=acquisition_grid(load_acquisition())/1e6
    fig,axes=plt.subplots(5,2,figsize=(12,14),sharex=True)
    rows=[]; convergence=[]
    for i,(fit,old) in enumerate(zip(report["fits"],single["fits"])):
        trace=np.loadtxt(folder/f"scan_{i+1}.csv",delimiter=",",skiprows=1)
        y=records[i,1:];pred=butanol_reconstruct(f,fit);base=butanol_reconstruct(f,fit,False)
        before=single_reconstruct(f,old)
        np.testing.assert_array_equal(trace[:,1],y)
        np.testing.assert_allclose(trace[:,0],f,atol=0,rtol=0)
        np.testing.assert_allclose(trace[:,2],pred,atol=1e-11,rtol=1e-10)
        residual=y-pred
        rms=float(np.sqrt(np.mean(residual**2)))
        np.testing.assert_allclose(rms,fit["diagnostics"]["rms_recorded_units"],rtol=1e-7)
        change=butanol_reconstruct(f,fit,nphi=256)-pred
        original_at_128=butanol_reconstruct(f,{**old,"parameters":{
            **old["parameters"],"od_fraction":0.,"eta_od":0.,"od_split_ratio":1.5}})
        convergence.append({"record_1based":i+1,"nphi_128_to_256_max_recorded_units":float(np.max(np.abs(change))),
                            "nphi_128_to_256_rms_recorded_units":float(np.sqrt(np.mean(change**2))),
                            "single_site_32_to_128_max_recorded_units":float(np.max(np.abs(original_at_128-before)))})
        axes[i,0].scatter(f,(y-base)*1e3,s=5,alpha=.65,color="#267b71",gid=f"butanol-scan-{i+1}",label="Measured − common fitted baseline")
        axes[i,0].plot(f,(before-base)*1e3,color="#ab7558",lw=1.1,label="Single-site full fit − same baseline")
        axes[i,0].plot(f,(pred-base)*1e3,"--",color="#173c3a",lw=1.3,label="Two-site butanol fit")
        axes[i,1].plot(f,(y-before)*1e6,color="#ab7558",lw=.8,label="Single site")
        axes[i,1].plot(f,residual*1e6,color="#267b71",lw=.8,label="Butanol two site")
        axes[i,1].axhline(0,color="black",lw=.5,alpha=.4)
        for ax in axes[i]:style(ax);ax.set_xlim(f[0],f[-1])
        axes[i,0].set_title(f"Scan {i+1} · P fit {fit['parameters']['P_model']*100:.2f}%",loc="left",fontsize=11)
        axes[i,1].set_title(f"Full residual · RMS reduction {fit['rms_reduction_percent']:.2f}%",loc="left",fontsize=11)
        axes[i,0].set_ylabel("10⁻³ recorded units");axes[i,1].set_ylabel("10⁻⁶ recorded units")
        p=fit["parameters"]
        rows.append(f'<tr><td>{i+1}</td><td>{old["parameters"]["P_model"]*100:.3f}%</td><td>{p["P_model"]*100:.3f}%</td><td>{old["diagnostics"]["rms_recorded_units"]*1e6:.2f}</td><td>{rms*1e6:.2f}</td><td>{fit["rms_reduction_percent"]:.2f}%</td><td>{p["od_fraction"]:.4f}</td></tr>')
    for ax in axes[0]:ax.legend(fontsize=7,loc="upper left")
    for ax in axes[-1]:ax.set_xlabel("Frequency (MHz)")
    fig.tight_layout();save(fig,"butanol-comparison.svg","Single-site and two-site butanol fits: all five measured scans")
    first=report["fits"][0]
    chi,cd,od=susceptibility(f,first["parameters"])
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,index,label in [(axes[0],0,"Absorption −Im χ"),(axes[1],1,"Dispersion Re χ")]:
        for values,name,color in [(cd,"C–D contribution","#267b71"),(od,"O–D contribution","#b44c30"),(chi,"Total","#173c3a")]:
            ax.plot(f,-values.imag if index==0 else values.real,label=name,color=color,lw=1.3)
        style(ax);ax.set(xlabel="Frequency (MHz)",ylabel="Effective fitted cgs susceptibility",title=label);ax.legend(fontsize=9)
    fig.tight_layout();save(fig,"butanol-sites.svg","Fitted C-D and O-D complex susceptibility contributions for scan 1")
    warnings=[f'Scan {f["record_1based"]}: {f["near_bounds"]}' for f in report["fits"] if f["near_bounds"]]
    warning_text="No selected fit lies within the declared near-bound threshold." if not warnings else "Near-bound flags: "+"; ".join(warnings)
    selected_status="All selected fits converged." if all(f["success"] for f in report["fits"]) else "Some selected fits did not converge; inspect the saved termination flags."
    failed=sum(not c["success"] for fit in report["fits"] for c in fit["candidates"])
    reductions=[fit["rms_reduction_percent"] for fit in report["fits"]]
    publication={"report_sha256":digest(folder/"butanol_report.json"),"exporter_sha256":digest(__file__),
                 "theory_source":json.loads((ROOT/"configs/butanol-theory-source.json").read_text()),
                 "theory_source_sha256":digest(ROOT/"configs/butanol-theory-source.json"),
                 "numerical_convergence":convergence,"figure_sha256":{n:digest(ASSETS/n) for n in ("butanol-comparison.svg","butanol-sites.svg")},
                 "validation_environment":{"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__}}
    publication["single_site_reference_asset"]=reference.name
    publication["figure_environment"]=FIGURE_PROVENANCE.get("butanol-sites.svg",publication["validation_environment"])
    (ASSETS/"butanol-matching.json").write_text(json.dumps({**report,"publication":publication},indent=2,allow_nan=False)+"\n")
    content=f'''
<section id="model" class="chapter"><p class="eyebrow">Same data · a material-specific model</p><h2>Two deuteron environments in butanol.</h2>
<p>The owner identified the five sweeps in <a href="https://github.com/zetanaut/NMR-AI/blob/main/examples/Sample_RawSignal.csv">Sample_RawSignal.csv</a> as butanol. <a href="matching.html">Example 1</a> keeps the useful single-site starting model. This second example fits the same raw samples with the C–D/O–D mixture in the supplied butanol theory. The scientific context is <a href="https://doi.org/10.1016/S0168-9002(97)00317-3">Dulya et al., A line-shape analysis for spin-1 NMR signals</a>.</p>
<div class="equation small">χ(f) = A_int [(1 − K) κ_CD(f)/Δ_CD + K κ_OD(f)/Δ_OD]<br>g_CD = Γ/Δ_CD, &nbsp; g_OD = Γ/Δ_OD<br>κ_site = w₊ K₊ + w₋ K₋, &nbsp; w₊ = (P + Q)/2, &nbsp; w₋ = (P − Q)/2</div>
<p>Both sites share vector polarization P, spin-temperature Q(P), and physical Lorentzian HWHM Γ. They have separate splitting scales Δ and EFG asymmetries. The inverse-splitting factors make K an integrated susceptibility weight. A_int has units cgs·MHz; the fitter retains the original amplitude coordinate with A_int = scale_cgs × Δ_CD. K is an effective fitted contribution, not an independently measured chemical abundance.</p>
<p>The supplied theory's weak-quadrupole intensity option is used. Complex absorption and dispersion use the existing orientation integral with 128-point Gauss–Legendre azimuth averaging. The source demonstration's frequency-dependent intensity option, synthetic parameter values, empirical gain tilt, and arbitrary frequency rescaling are not silently treated as measurements.</p>
<p>Both comparisons use all 500 confirmed bins, fⱼ = 32.3 + 0.0015287 j MHz. The cable remains n=1, 3.580 m at 32.7 MHz. The full single-capacitor circuit, finite-source loading, passive cable and fitted detector phase are retained. The same electronics and original lineshape bounds are used; three new nonlinear coordinates are the O–D splitting ratio, asymmetry and fraction. There are <strong>15 fitted unknowns versus 12</strong> in the starting model. At K=0 the original single-site model is recovered.</p>
<div class="baseline-figure-scroll" tabindex="0" role="region" aria-label="C-D and O-D site contributions"><img class="baseline-fit-image" src="assets/butanol-sites.svg" width="1100" height="440" alt="Intrinsic fitted absorption and dispersion for the C-D and O-D sites in scan 1."></div>
<p>These are intrinsic susceptibility components. The detector response is nonlinear, so separate site detector voltages should not be added as if they were independent measured signals.</p></section>
<section id="results" class="chapter"><p class="eyebrow">Measured comparison</p><h2>Better agreement, with the remaining mismatch visible.</h2>
<p>The selected two-site fits reduce full-sweep RMS by <strong>{min(reductions):.2f}% to {max(reductions):.2f}%</strong>. All bins and endpoints remain in the fit and statistics. The left panels use one common fitted baseline for the data and both curves; the right panels show each model's recorded-minus-fitted residual.</p>
<div class="table-wrap"><table><caption>Same five raw sweeps; RMS in 10⁻⁶ recorded units</caption><thead><tr><th>Scan</th><th>Single-site P fit</th><th>Butanol P fit</th><th>Single-site RMS</th><th>Butanol RMS</th><th>RMS reduction</th><th>O–D weight K</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<div class="baseline-figure-scroll" tabindex="0" role="region" aria-label="All five butanol fit comparisons and residuals"><img class="baseline-fit-image" src="assets/butanol-comparison.svg" width="1200" height="1400" alt="All five measured butanol signals compared with the single-site and two-site fits, with full residuals."></div>
<p>{selected_status} {warning_text} {failed} starting solutions did not converge within their evaluation limit; every candidate remains in the <a href="assets/butanol-matching.json">full fit and numerical-validation record</a>. Extra parameters can reduce fit residuals, and structured endpoint deviations remain. This comparison does not establish polarization accuracy or a calibrated uncertainty.</p></section>
<section id="reproduce" class="chapter"><h2>Reproduce the comparison.</h2><p>Run from the standalone repository with the declared requirements installed. Use fresh output directories. Limiting BLAS threads avoids overhead in these small repeated matrix operations.</p>
<pre><code>OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/match_butanol.py \\
  --output-dir local-results/butanol-matching --starts 6
python tools/export_material_examples.py \\
  --butanol-fit-dir local-results/butanol-matching</code></pre>
<p>The fitting contract is <a href="https://github.com/zetanaut/NMR-AI/blob/main/configs/butanol-matching.json">butanol-matching.json</a>. Shared parameters retain the original matching bounds; O–D ratio is 1.05–2.5, η_OD is 0–0.5, and K is 0–0.4. These are numerical search choices. The <a href="https://github.com/zetanaut/NMR-AI/blob/main/configs/butanol-theory-source.json">source record</a> identifies the supplied theory files and the adopted conventions.</p></section>
<section id="limits" class="chapter"><h2>What this example teaches.</h2><p>Material identity changes the appropriate lineshape model. Here it is known from the owner, and the added site improves the recorded fit. Common site polarization, common linewidth, fixed nominal circuit components, and the intensity approximation remain hypotheses to test. Near-best starting solutions and quadrature convergence are numerical diagnostics, not statistical confidence intervals.</p>
<p>The original 500-bin single-site generator remains a separate teaching scenario. This material comparison does not update its data or claim a newly trained network. Read the <a href="https://github.com/zetanaut/NMR-AI/blob/main/notes/material-examples.md">complete material-model record</a>, then compare a different acquisition in <a href="uva-nd3.html">Example 3: UVA-ND3 data</a>.</p></section>'''
    (ROOT/"docs/butanol.html").write_text(page("Butanol: fit both deuteron sites",2,"Use the known material to extend the lineshape, then compare both models on exactly the same measured sweeps.",content))


def publish_nd3(folder):
    report=read_verified(folder/"uva_nd3_report.json")
    verify_reused_figures(report,"uva-nd3-matching.json")
    data,audit=load_nd3(ROOT/"examples/uva-nd3.json")
    if report["audit"]!=audit or len(report["fits"])!=len(data["records"]):
        raise ValueError("UVA-ND3 report does not match the complete teaching excerpt")
    if digest(ROOT/"configs/uva-nd3-matching.json")!=report["config_sha256"]:
        raise ValueError("UVA-ND3 fitting configuration changed")
    fig,axes=plt.subplots(5,2,figsize=(12,14),sharex=True)
    rows=[]; convergence=[]
    for i,(r,fit) in enumerate(zip(data["records"],report["fits"])):
        f=np.asarray(r["frequency_mhz"]);y=np.asarray(r["basesub"])
        pred=nd3_reconstruct(f,fit);background=nd3_reconstruct(f,fit,True)
        residual=y-pred
        saved=np.loadtxt(folder/f'record_{r["source_record_1based"]}.csv',delimiter=",",skiprows=1)
        np.testing.assert_array_equal(saved[:,0],f);np.testing.assert_array_equal(saved[:,1],y)
        np.testing.assert_allclose(saved[:,2],pred,atol=1e-13,rtol=1e-10)
        np.testing.assert_allclose(np.sqrt(np.mean(residual**2)),fit["diagnostics"]["rms_recorded_units"],rtol=1e-8)
        change=nd3_reconstruct(f,fit,nphi=64)-pred
        convergence.append({"source_record_1based":r["source_record_1based"],"nphi_32_to_64_max_recorded_units":float(np.max(np.abs(change)))})
        axes[i,0].scatter(f,(y-background)*1e3,s=6,color="#267b71",alpha=.65,gid=f'nd3-record-{r["source_record_1based"]}',label="Resampled reference subtraction − fitted background")
        axes[i,0].plot(f,(pred-background)*1e3,"--",color="#b44c30",lw=1.3,label="Conditional spin-1 fit")
        axes[i,1].plot(f,residual*1e6,color="#267b71",lw=.8);axes[i,1].axhline(0,color="black",lw=.5,alpha=.4)
        for ax in axes[i]:style(ax);ax.set_xlim(f[0],f[-1])
        axes[i,0].set_title(f'Record {r["source_record_1based"]} · P fit {fit["parameters"]["P_model"]*100:.2f}%',loc="left",fontsize=11)
        axes[i,1].set_title("Full residual · reference-subtracted data minus fit",loc="left",fontsize=10)
        axes[i,0].set_ylabel("10⁻³ recorded units");axes[i,1].set_ylabel("10⁻⁶ recorded units")
        p=fit["parameters"];d=fit["diagnostics"]
        rows.append(f'<tr><td>{r["source_record_1based"]}</td><td>{p["P_model"]*100:.3f}%</td><td>{p["center_mhz"]:.6f}</td><td>{p["split_mhz"]*1e3:.3f}</td><td>{p["g"]*p["split_mhz"]*1e3:.3f}</td><td>{p["eta"]:.4f}</td><td>{d["rms_recorded_units"]*1e6:.2f}</td></tr>')
    axes[0,0].legend(fontsize=6.5,loc="upper right")
    for ax in axes[-1]:ax.set_xlabel("Frequency (MHz) · 500-bin grid")
    fig.tight_layout();save(fig,"uva-nd3-matches.svg","UVA-ND3 conditional lineshape fits and full residuals")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for r in data["records"]:
        f=np.asarray(r["frequency_mhz"])
        axes[0].plot(f,r["phase"],lw=1,label=f'Record {r["source_record_1based"]}')
        axes[1].plot(f,r["basesub"],lw=1)
    axes[0].plot(data["records"][0]["frequency_mhz"],data["records"][0]["baseline"],"k--",lw=1,label="Stored reference (first record)")
    for ax,title in zip(axes,["Resampled phase and recorded reference","Resampled phase − resampled baseline"]):
        style(ax);ax.set(xlabel="Frequency (MHz) · 500-bin grid",ylabel="Recorded units",title=title)
    axes[0].legend(fontsize=7);fig.tight_layout();save(fig,"uva-nd3-raw.svg","500-bin UVA-ND3 phase and reference subtraction")
    publication={"report_sha256":digest(folder/"uva_nd3_report.json"),"exporter_sha256":digest(__file__),
                 "numerical_convergence":convergence,"figure_sha256":{n:digest(ASSETS/n) for n in ("uva-nd3-matches.svg","uva-nd3-raw.svg")},
                 "validation_environment":{"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__}}
    publication["figure_environment"]=FIGURE_PROVENANCE.get("uva-nd3-raw.svg",publication["validation_environment"])
    (ASSETS/"uva-nd3-matching.json").write_text(json.dumps({**report,"publication":publication},indent=2,allow_nan=False)+"\n")
    all_success=all(f["success"] for f in report["fits"])
    warnings=[f'Record {f["source_record_1based"]}: {f["near_bounds"]}' for f in report["fits"] if f["near_bounds"]]
    status="All selected fits converged." if all_success else "Some selected fits did not converge; inspect the saved report."
    status+=" No selected fit has a near-bound flag." if not warnings else " Near-bound flags: "+"; ".join(warnings)
    content=f'''
<section id="model" class="chapter"><p class="eyebrow">Another material · another acquisition</p><h2>Fit UVA-ND3 on the common 500-bin grid.</h2>
<p>The <a href="https://github.com/zetanaut/NMR-AI/blob/main/examples/uva-nd3.json">500-bin UVA-ND3 working example</a> contains records 1, 126, 251, 376 and 501, selected at equal intervals through the source acquisition before fitting. Every phase, baseline and reference-subtracted array uses <strong>fⱼ = 32.3 + 0.0015287 j MHz, j = 0…499</strong>, ending at 33.0628213 MHz. All fits, figures and residuals below use these 500 bins.</p>
<p>These are derived data: raw phase and the recorded baseline are linearly interpolated in frequency onto the common grid in float64, then subtracted. The original measurement arrays remain in a <a href="https://github.com/zetanaut/NMR-AI/tree/main/provenance">source provenance archive</a>. The saved derivation records hashes, the target window and endpoint roundoff handling. Interpolation changes resolution and correlates errors; the 500 samples are not independent new measurements. The recorded 4,000 acquisition sweeps do not establish a noise covariance.</p>
<div class="baseline-figure-scroll" tabindex="0" role="region" aria-label="500-bin UVA-ND3 data and reference subtraction"><img class="baseline-fit-image" src="assets/uva-nd3-raw.svg" width="1100" height="450" alt="Five UVA-ND3 phase sweeps and recorded references resampled onto the exact 500-bin grid, with their subtraction."></div>
<h3>A conditional fit after resampling and reference subtraction</h3>
<div class="equation small">y_sub = L[phase] − L[recorded baseline]<br>y_fit = L[a_abs [−Im κ(f;P)] + a_disp Re κ(f;P)<br>+ b₀ + b₁t + b₂t² + b₃t³], &nbsp; t = (f_MHz − 32.7)/0.4</div>
<p>L applies the same frequency interpolation to the complete model evaluated on the original source coordinates. The fit minimizes unweighted residuals over all 500 target bins; interpolation is included in the response, without assuming an independently measured noise covariance.</p>
<p>κ is the unit-area single-site spin-temperature shape. Its branch weights are the physical weights divided by P, evaluated with their finite symmetric limit at zero; the fitted amplitude carries the overall signal strength. This parameterization fits a nonzero observed lineshape and is not a simulator that predicts finite nuclear signal at zero P. The five nonlinear unknowns are center, splitting, Lorentzian width, EFG asymmetry and P. Two constant absorption/dispersion coefficients and four residual-background coefficients are profiled jointly: <strong>11 fitted unknowns</strong>.</p>
<p>The cubic describes residual mismatch after subtraction of the recorded reference. It is fitted jointly on all bins, not frozen from a preliminary wing fit. The constant readout mixture and cubic residual are explicit approximations. Cable, coil and capacitor records for this acquisition have not been independently supplied, so this example does not borrow the butanol apparatus as a hardware calibration or claim a full-circuit fit.</p>
<p>No stored acquisition polarization, area calibration, or previous fitted curve enters the optimization. P is inferred from the intrinsic branch shapes and ratio under the spin-temperature hypothesis; TE calibration is not a prerequisite.</p></section>
<section id="results" class="chapter"><p class="eyebrow">Five predetermined records</p><h2>Fit the observed structure and show what remains.</h2>
<div class="table-wrap"><table><caption>UVA-ND3 conditional fit estimates; RMS in 10⁻⁶ recorded units</caption><thead><tr><th>Source record</th><th>P from shape</th><th>Center (MHz)</th><th>Split (kHz)</th><th>HWHM (kHz)</th><th>EFG η</th><th>RMS</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<div class="baseline-figure-scroll" tabindex="0" role="region" aria-label="UVA-ND3 fits with all 500 bins and residuals"><img class="baseline-fit-image" src="assets/uva-nd3-matches.svg" width="1200" height="1400" alt="Five UVA-ND3 lineshapes resampled onto 500 bins, fitted curves and full 500-bin residuals."></div>
<p>{status} The left panels remove the jointly fitted residual background for display; the right panels retain all 500 bins of resampled reference-subtracted data minus the complete resampled fitted model. Remaining structured residuals show the limits of the conditional model. The <a href="assets/uva-nd3-matching.json">complete fit record</a> contains every starting solution, source/code hashes and numerical convergence checks.</p></section>
<section id="reproduce" class="chapter"><h2>Run the UVA-ND3 example.</h2>
<pre><code>python tools/prepare_uva_nd3.py
python tools/match_uva_nd3.py \\
  --output-dir local-results/uva-nd3-matching-500-v2 --starts 6
python tools/export_material_examples.py \\
  --nd3-fit-dir local-results/uva-nd3-matching-500-v2</code></pre>
<p>The audited reader requires the exact 500-bin grid, reproduces the interpolation from the preserved source and checks <code>phase − baseline == basesub</code> for every bin. The fitter rejects the original acquisition arrays as working input. The <a href="https://github.com/zetanaut/NMR-AI/blob/main/configs/uva-nd3-matching.json">fit configuration</a> states bounds and readout/background assumptions. Everything needed to reproduce the example is in this repository.</p></section>
<section id="limits" class="chapter"><h2>Different setups need different validation.</h2><p>These are conditional fit estimates from five development records in one period. They do not establish calibrated polarization uncertainty or performance on independent acquisitions. Interpolation correlates neighboring errors, so an independent-bin noise proxy is not reported. Residual structure must not be treated as a measured covariance.</p>
<p>A common 500-bin grid establishes input dimensions. Applying a trained network also requires validated material and readout coverage, compatible units and references, and its saved preprocessing. Fitted P values are conditional estimates; this example does not establish network accuracy or the butanol hardware settings for ND3. Compare <a href="matching.html">the original single-site exercise</a> and <a href="butanol.html">the butanol two-site comparison</a>, and read the <a href="https://github.com/zetanaut/NMR-AI/blob/main/notes/material-examples.md">material-model record</a> before extending a simulator.</p></section>'''
    (ROOT/"docs/uva-nd3.html").write_text(page("UVA-ND3 data",3,"Five UVA-ND3 records on the exact 500-bin teaching grid, with documented resampling, recorded reference subtraction and conditional lineshape fits.",content))


def main():
    global PREVIEW_DIR,REUSE_FIGURES
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--butanol-fit-dir",type=Path)
    parser.add_argument("--nd3-fit-dir",type=Path)
    parser.add_argument("--preview-dir",type=Path,help="Optional local PNG copies for visual inspection")
    parser.add_argument("--reuse-figures",action="store_true",help="Keep hash-verified existing figures for unchanged fit reports")
    args=parser.parse_args()
    REUSE_FIGURES=args.reuse_figures
    if args.butanol_fit_dir is None and args.nd3_fit_dir is None:
        parser.error("Supply at least one material fit directory")
    if args.preview_dir is not None:
        args.preview_dir.mkdir(parents=True,exist_ok=True)
        PREVIEW_DIR=args.preview_dir
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11,"svg.fonttype":"none",
                         "svg.hashsalt":"nmr-ai-material-examples","figure.facecolor":"#fafaf6","axes.facecolor":"#fafaf6"})
    if args.butanol_fit_dir is not None:
        publish_butanol(args.butanol_fit_dir)
    if args.nd3_fit_dir is not None:
        publish_nd3(args.nd3_fit_dir)
    print("Published the selected verified material practicals")


if __name__=="__main__":main()
