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
    if report.get("input_channel") != "phase" or report.get("stored_reference_used_in_fit") is not False:
        raise ValueError("Only the joint raw-phase ND3 workflow can be published")
    verify_reused_figures(report,"uva-nd3-matching.json")
    data,audit=load_nd3(ROOT/"examples/uva-nd3.json")
    if report["audit"]!=audit or len(report["fits"])!=len(data["records"]):
        raise ValueError("UVA-ND3 report does not match the complete teaching excerpt")
    if digest(ROOT/"configs/uva-nd3-matching.json")!=report["config_sha256"]:
        raise ValueError("UVA-ND3 fitting configuration changed")
    fig,axes=plt.subplots(5,3,figsize=(15,14),sharex=True)
    rows=[]; convergence=[]
    for i,(r,fit) in enumerate(zip(data["records"],report["fits"])):
        f=np.asarray(r["frequency_mhz"]);y=np.asarray(r["phase"])
        pred=nd3_reconstruct(f,fit);background=nd3_reconstruct(f,fit,True)
        residual=y-pred
        saved=np.loadtxt(folder/f'record_{r["source_record_1based"]}.csv',delimiter=",",skiprows=1)
        np.testing.assert_array_equal(saved[:,0],f);np.testing.assert_array_equal(saved[:,1],y)
        np.testing.assert_allclose(saved[:,2],pred,atol=1e-13,rtol=1e-10)
        np.testing.assert_allclose(np.sqrt(np.mean(residual**2)),fit["diagnostics"]["rms_recorded_units"],rtol=1e-8)
        change=nd3_reconstruct(f,fit,nphi=64)-pred
        convergence.append({"source_record_1based":r["source_record_1based"],"nphi_32_to_64_max_recorded_units":float(np.max(np.abs(change)))})
        axes[i,0].scatter(f,y,s=6,color="#267b71",alpha=.65,gid=f'nd3-record-{r["source_record_1based"]}',label="Raw phase · baseline present")
        axes[i,0].plot(f,pred,"--",color="#b44c30",lw=1.3,label="Full ND3 circuit fit")
        axes[i,0].plot(f,background,color="#173c3a",lw=1,label="Jointly fitted circuit baseline")
        axes[i,1].plot(f,(pred-background)*1e3,color="#b44c30",lw=1.2,label="Fitted nuclear response")
        axes[i,2].plot(f,residual*1e6,color="#267b71",lw=.8)
        axes[i,2].axhline(0,color="black",lw=.5,alpha=.4)
        for ax in axes[i]:style(ax);ax.set_xlim(f[0],f[-1])
        axes[i,0].ticklabel_format(axis="y",useOffset=False)
        axes[i,0].set_title(f'Record {r["source_record_1based"]} · raw sweep and fit',loc="left",fontsize=10)
        axes[i,1].set_title(f'ND3 response · P fit {fit["parameters"]["P_model"]*100:.2f}%',loc="left",fontsize=10)
        axes[i,2].set_title("Full residual · raw phase minus fit",loc="left",fontsize=10)
        axes[i,0].set_ylabel("Recorded units")
        axes[i,1].set_ylabel("10⁻³ recorded units");axes[i,2].set_ylabel("10⁻⁶ recorded units")
        p=fit["parameters"];d=fit["diagnostics"]
        rows.append(f'<tr><td>{r["source_record_1based"]}</td><td>{p["P_model"]*100:.3f}%</td><td>{p["center_mhz"]:.6f}</td><td>{p["split_mhz"]*1e3:.3f}</td><td>{p["g"]*p["split_mhz"]*1e3:.3f}</td><td>{p["eta"]:.4f}</td><td>{fit["resolved_tune_capacitance_pf"]:.2f}</td><td>{d["rms_recorded_units"]*1e6:.2f}</td></tr>')
    axes[0,0].legend(fontsize=6,loc="lower left")
    for ax in axes[-1]:ax.set_xlabel("Frequency (MHz) · 500-bin grid")
    fig.tight_layout();save(fig,"uva-nd3-matches.svg","UVA-ND3 raw sweeps, jointly fitted circuit baselines, nuclear responses and full residuals")
    fig,ax=plt.subplots(figsize=(11,4.5))
    for r in data["records"]:
        f=np.asarray(r["frequency_mhz"])
        ax.plot(f,r["phase"],lw=1,label=f'Record {r["source_record_1based"]}')
    style(ax)
    ax.ticklabel_format(axis="y",useOffset=False)
    ax.set(xlabel="Frequency (MHz) · 500-bin grid",ylabel="Raw recorded phase units",
           title="Input to every fit: raw ND3 phase with its baseline present")
    ax.legend(fontsize=8)
    fig.tight_layout();save(fig,"uva-nd3-raw.svg","All five raw 500-bin UVA-ND3 sweeps with baseline present")
    publication={"report_sha256":digest(folder/"uva_nd3_report.json"),"exporter_sha256":digest(__file__),
                 "numerical_convergence":convergence,"figure_sha256":{n:digest(ASSETS/n) for n in ("uva-nd3-matches.svg","uva-nd3-raw.svg")},
                 "validation_environment":{"numpy":np.__version__,"scipy":scipy.__version__,"matplotlib":matplotlib.__version__}}
    publication["figure_environment"]=FIGURE_PROVENANCE.get("uva-nd3-raw.svg",publication["validation_environment"])
    (ASSETS/"uva-nd3-matching.json").write_text(json.dumps({**report,"publication":publication},indent=2,allow_nan=False)+"\n")
    all_success=all(f["success"] for f in report["fits"])
    warnings=[]
    for fit in report["fits"]:
        for name,side in fit["near_bounds"].items():
            label="tuning capacitance" if name=="log10_tune_capacitance_pf" else name
            value=f" ({fit['resolved_tune_capacitance_pf']:.2f} pF)" if name=="log10_tune_capacitance_pf" else ""
            warnings.append(f"Record {fit['source_record_1based']}: {label} reaches the {side} search bound{value}")
    status="All selected fits converged." if all_success else "Some selected fits did not converge; inspect the saved report."
    status+=" No selected fit has a near-bound flag." if not warnings else " Boundary diagnostic: "+"; ".join(warnings)
    content=f'''
<section id="model" class="chapter"><p class="eyebrow">Raw sweep → joint fit → ND3 response</p><h2>Fit the ND3 signal with its baseline present.</h2>
<p>The <a href="https://github.com/zetanaut/NMR-AI/blob/main/examples/uva-nd3.json">500-bin UVA-ND3 working example</a> supplies the raw <code>phase</code> channel for records 1, 126, 251, 376 and 501. The baseline is already present in these sweeps. The fit uses each complete raw sweep and jointly determines its circuit baseline, detector response and ND3 spin-1 lineshape. No additional baseline is needed.</p>
<p>Every working array uses <strong>fⱼ = 32.3 + 0.0015287 j MHz, j = 0…499</strong>, ending at 33.0628213 MHz. Raw phase is linearly resampled from the original frequency coordinates in float64. The stored reference and its subtraction remain available as provenance; neither enters this fit. Original measurements are preserved in the <a href="https://github.com/zetanaut/NMR-AI/tree/main/provenance">source archive</a>. Interpolation changes resolution and correlates errors.</p>
<div class="baseline-figure-scroll" tabindex="0" role="region" aria-label="Five raw UVA-ND3 input sweeps with baseline present"><img class="baseline-fit-image" src="assets/uva-nd3-raw.svg" width="1100" height="450" alt="Five raw 500-bin UVA-ND3 phase sweeps, retaining their existing baselines."></div>
<h3>The same full-circuit workflow, with the ND3 lineshape</h3>
<div class="equation small">χ_ND3 = A [w₊ K₊ + w₋ K₋]<br>w₊ = (P + Q)/2, &nbsp; w₋ = (P − Q)/2<br>y_fit = L[a Re(u(χ_ND3) exp(iφ)) + b Im(u(χ_ND3) exp(iφ)) + d]<br>B_fit = y_fit with χ = 0, &nbsp; S_fit = y_fit − B_fit</div>
<p>The single-site ND3 response uses the complex spin-temperature powder model, with its fitted splitting, broadening and EFG asymmetry. Q = 3P² / (2 + √(4 − 3P²)). The susceptibility vanishes at P=0. Its absorption and dispersion pass through the coil, passive RLGC cable, one tuning capacitor and finite source/input loading, following the same physical circuit as the other examples.</p>
<p>The detector phase is φ₁Δf + φ₂Δf²; its constant phase, gain and offset are represented by the jointly profiled coefficients a, b and d. The five ND3 shape parameters, effective susceptibility amplitude, tuning capacitance, phase slope and phase curvature give nine nonlinear coordinates. Including the three readout coefficients gives <strong>12 fitted unknowns</strong>. The baseline comes from the same fitted circuit with susceptibility set to zero.</p>
<p>L is the documented interpolation onto 500 bins. The complete circuit model is evaluated on source coordinates and resampled with the same operator as raw phase. Wings initialize the electronics; the final optimization fits all 500 raw values jointly. The stored reference is not subtracted or fixed as the model baseline.</p>
<p>The <a href="https://github.com/zetanaut/NMR-AI/blob/main/configs/uva-nd3-matching.json">configuration</a> lists all fixed circuit values and fit bounds. Fixed values use the standalone tutorial's nominal circuit, including its derived half-wave operating point. These are explicit modeling assumptions. They do not assign the measured butanol hardware to ND3 or establish ND3 component calibration. Polarization comes from the lineshape; no TE calibration or stored acquisition polarization is used.</p></section>
<section id="results" class="chapter"><p class="eyebrow">Five complete raw fits</p><h2>Keep the baseline visible and inspect the full residual.</h2>
<div class="table-wrap"><table><caption>Joint raw-phase ND3 fit estimates; RMS in 10⁻⁶ recorded units</caption><thead><tr><th>Source record</th><th>P from shape</th><th>Center (MHz)</th><th>Split (kHz)</th><th>HWHM (kHz)</th><th>EFG η</th><th>C_tune (pF)</th><th>Raw-fit RMS</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<div class="baseline-figure-scroll" tabindex="0" role="region" aria-label="Raw ND3 fits, circuit baselines, fitted signals and full residuals"><img class="baseline-fit-image" src="assets/uva-nd3-matches.svg" width="1500" height="1400" alt="Five rows showing raw ND3 phase with the complete fit and circuit baseline, the fitted nuclear response, and all 500 raw-minus-fit residuals."></div>
<p>{status} A search-bound solution is a diagnostic of the assumed electronics and does not measure a component setting. The left panels retain the original baseline level in the raw input. The middle panels show the fitted nuclear response, defined after fitting as the complete model minus its χ=0 baseline. The right panels show raw phase minus the complete fit across all 500 bins. The <a href="assets/uva-nd3-matching.json">complete fit record</a> saves every starting solution, nominal circuit assumption, fitted parameter, source hash and quadrature check.</p></section>
<section id="reproduce" class="chapter"><h2>Run the raw UVA-ND3 example.</h2>
<pre><code>python tools/prepare_uva_nd3.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/match_uva_nd3.py \\
  --output-dir local-results/uva-nd3-raw-matching-500-v3 --starts 6
python tools/export_material_examples.py \\
  --nd3-fit-dir local-results/uva-nd3-raw-matching-500-v3</code></pre>
<p>The audited reader verifies the exact grid and reproducible source resampling. The fitting command selects <code>phase</code>. Each saved fit CSV contains frequency, raw phase, full fit, fitted circuit baseline, fitted signal and raw-minus-fit residual. All columns contain 500 values. Everything needed is included in this standalone repository.</p></section>
<section id="limits" class="chapter"><h2>Interpret the joint fit as a conditional model.</h2><p>The circuit and ND3 lineshape are fitted together, so their assumptions affect both the baseline and inferred polarization. These five development records do not establish calibrated hardware, statistical parameter errors or independent polarization accuracy. Structured raw-fit residuals and interpolation-induced correlation remain visible; an independent-bin noise sigma is not inferred from them.</p>
<p>A common raw-sweep workflow and 500-bin grid do not establish material-specific network accuracy. Generator and training results remain tied to their existing contracts. Compare <a href="matching.html">the single-site full-circuit example</a> and <a href="butanol.html">the two-site butanol example</a>, and see the <a href="https://github.com/zetanaut/NMR-AI/blob/main/notes/material-examples.md">material-model record</a> for the ND3 assumptions and validation.</p></section>'''
    (ROOT/"docs/uva-nd3.html").write_text(page("UVA-ND3 data",3,"Fit all 500 raw samples with their baseline present, using the ND3 spin-1 response and the same physical circuit workflow as the other examples.",content))


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
