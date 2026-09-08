# Project status and session handoff

Updated 2026-09-08, including the material-example and model-comparison
publication, the matching-uncertainty tutorial addition, and records of available
`local-results/` artifacts. This is a
status record; the detailed scientific conventions remain in the
[physics notes](physics-electronics-theory.md),
[baseline record](baseline-fitting.md), and
[experimental-matching record](experimental-matching.md). The new
[material-model record](material-examples.md) covers the butanol and UVA-ND3 demonstrations.

## Latest tutorial addition — matching uncertainty and shift, 2026-09-08

The new `docs/index.html#distribution-shift` section and
[detailed protocol](distribution-shift.md) refine the owner's proposed
noise/baseline/signal covariance, shift and drift records. The three proposed
deliverables are measurement/joint fit uncertainty, residual-and-input-distribution
discrepancy, and a conditional robustness table with separate temporal monitoring.
The section retains systematic means and cross-covariances, distinguishes strict
covariate shift from general distribution shift, defines SNR/P conditioning and
propagation to P error, and requires independent validation after mitigation.

This is a documented protocol, not newly estimated experimental matrices or
new network accuracy. Neither generator nor the existing fit/training results
changed. The current five butanol scans do not identify the proposed conditional
covariances; the 500-bin neural-network workflow remains pending.

Numerical identity checks and browser evidence are saved under
`local-results/distribution-shift-review-1uz0m8hz/`. Constructed examples verified
the covariance/second-moment and propagation identities to a maximum discrepancy
of 5.33e-15, including independent whitened least-squares refits for residual
projection. Desktop/mobile checks covered all three expandable explanations,
JavaScript-disabled reading and absence of page overflow; an HTML audit checked
165 local/repository links on five pages. This documentation-only change did
not require another training run or a new full Python-suite result.

Published in commit
[`6dc7448`](https://github.com/zetanaut/NMR-AI/commit/6dc74486be8e88b8a63660defefaf91a88caef39),
with a successful [Pages deployment](https://github.com/zetanaut/NMR-AI/actions/runs/34251188019).
The live [matching uncertainty and shift section](https://zetanaut.github.io/NMR-AI/#distribution-shift)
was verified on 2026-09-08: its complete HTML matches the committed bytes,
and desktop/mobile Chromium checks passed for all three expandable explanations,
active navigation, absence of overflow and absence of JavaScript errors.
The same evidence directory contains `publication-report.json` and live screenshots.

## Model-building tutorial completed — 2026-09-08

The model-building task expanded the tutorial from a basic MLP through a deeper
dense DNN to a multiscale CNN, showing errors on the same training data. This tutorial
update is complete and published on GitHub Pages, including final validation. The
dedicated 500-bin workflow below remains a separate subsequent milestone.

The tutorial contains all three model explanations, runnable examples,
an error figure/table, and interactive saved learning curves in
`docs/index.html#network`. `configs/model-comparison.json` declares the protocol.
The completed runs are in `local-results/model-comparison-v1/{mlp,dnn,physics_multiscale}/`,
with checkpoints, preprocessing, exact partitions, histories and predictions;
the parent directory retains `protocol.json` and `evaluation.json`.
Public comparison assets are `docs/assets/model-comparison.{json,js,svg}`.
The [comparison record](model-comparison.md) documents the complete protocol,
architecture details, metrics and limitations. The README and tutorial status
summary now include the new comparison alongside the original CNN runs.

The 2026-09-08 catch-up review reconstructed the saved predictions and verified
the common dataset, partitions, training-only preprocessing, checkpoint hashes,
metrics and generated HTML. All models use the existing broad 512-bin TE-area
dataset: 4,788 training rows, 612 validation rows and the same 600 test rows,
with 240/30/30 disjoint configuration groups. Observed test RMSE is:

| Estimator | RMSE (percentage points) |
| --- | --- |
| Basic MLP | 0.06514504 |
| Deep dense DNN | 0.03840369 |
| Multiscale CNN with physical summaries | 0.00236318 |

This is one seed with a common 80-epoch cap. The dense models reach that cap;
the CNN stops after 27 epochs. Each checkpoint is selected by validation MSE.
The CNN includes a train-only ridge estimator, so the comparison measures the
complete estimators and does not isolate convolution. These remain synthetic
512-bin results under ideal TE calibration, not experimental or 500-bin accuracy.

Final verification on 2026-09-08 reconstructed the complete public comparison
from saved checkpoints, scalers, partitions and predictions, including the
current exporter and figure hashes. All 54 tests passed in 19.058 seconds.
Browser checks at 1440px and 390px verified all four learning-curve selections,
no JavaScript errors or page overflow, and the static figure/table without
JavaScript. The charts now retain readable labels on phones through keyboard-
accessible horizontal scrolling. Final desktop/mobile screenshots were inspected.
Browser evidence is in `local-results/model-comparison-final-review-8dl4_ha8/`.
HTML nesting, unique IDs and 162 local/repository links were checked across all
five tutorial pages, and `git diff --check` passed. The exporter also strips
trailing SVG whitespace; numeric content was independently checked unchanged,
both comparison-publication tests passed afterward, and the staged diff check
passed before committing.

## GitHub publication — 2026-09-08

The owner authorized publishing the completed tutorial. Commit
[`75a8699`](https://github.com/zetanaut/NMR-AI/commit/75a86995a6d86073804210d9ab52227fb4120266)
was pushed to `main`, and the
[Pages deployment](https://github.com/zetanaut/NMR-AI/actions/runs/34235972078)
succeeded. The live [model-building lesson](https://zetanaut.github.io/NMR-AI/#network)
contains the MLP/DNN/CNN progression, comparison results and learning curves,
with the supporting code and linked material examples in the repository.

All 18 changed pages/assets were retrieved from the public site and matched
the committed local bytes. A live Chromium check exercised all four learning-curve
selections with finite plotted coordinates and no JavaScript errors. The report
and live screenshot are in `local-results/github-publication-20260908-18scaq63/`.
The deployment confirms publication; the Python test results above are separate
validation evidence. Generated datasets, checkpoints and full local fit products
remain git-ignored reproduction artifacts.

## Quick start on the owner's workstation

Environment and full test suite verified **2026-09-08**. The existing Conda
environment has Python **3.11.16**, PyTorch **2.11.0+cu128**, and working CUDA
**12.8** on an **NVIDIA RTX A6000**. NumPy is 2.4.6, SciPy 1.17.1, and
Matplotlib 3.11.1. These are observed environment versions, not new dependency pins.

From a fresh tool shell:

```bash
cd /home/dustin/work/projects/NMR-AI
NMR_AI_PYTHON=/home/dustin/work/projects/gen-NMR/.conda-env/bin/python
"$NMR_AI_PYTHON" -c "import sys, torch; print(sys.executable); print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
git status --short --branch
```

Use this absolute interpreter, or redeclare `NMR_AI_PYTHON` in each tool shell;
activation and shell variables do not carry over automatically between calls.
The path records an existing machine-local Python installation. Run tutorial
code from NMR-AI with this repository's inputs and outputs. Fresh checkouts can
create their own environment from `requirements.txt` as described in the README.

The default `/usr/bin/python` is Python 3.12.3 without PyTorch. That caused the
earlier import error; no installation was needed. If the documented interpreter
has moved, inspect `conda env list --json` and probe available interpreters before
concluding that dependencies are missing or installing packages.

For a full validation run, this command works without shell activation:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /home/dustin/work/projects/gen-NMR/.conda-env/bin/python \
  -m unittest discover -s tests -v
```

Latest result: **54 tests passed in 19.058 seconds**, including `test_physics`,
model training/partition round trips and comparison publication checks.
This supersedes the earlier 49-test and partial checks. Recheck after relevant changes;
the saved result describes the tested working tree, not future edits.

The butanol/UVA-ND3 and MLP/DNN/CNN comparison code, examples, figures, tests and
documentation were published on `main` in `75a8699`; a following documentation
commit records the deployment evidence. Inspect the current Git status before editing. The
completed tutorial-comparison task is described above; the dedicated 500-bin
training/prediction workflow remains pending.

## Saved artifacts to resume from

These paths are relative to NMR-AI and are git-ignored local artifacts. Use
existing verified products for inspection and documentation exports; a new
fit, generation, or training run needs a fresh output path.

| Artifact | Location |
| --- | --- |
| Constrained measured baseline | `local-results/deuteron-tuned-baseline-fit/fit_report.json` |
| Original five single-site fits | `local-results/experimental-matching/matching_report.json` |
| 500-bin generated data and provenance | `local-results/experiment-anchored.npz` and `.json` |
| Completed butanol comparison | `local-results/butanol-comparison-final/butanol_report.json` |
| Completed UVA-ND3 fits | `local-results/uva-nd3-matching/uva_nd3_report.json` |
| Existing 512-bin datasets | `local-results/qmeter/narrow.npz` and `broad.npz` |
| Saved models and preprocessing | `local-results/qmeter/{narrow_multiscale,broad_multiscale,broad_compact}/` |
| New MLP/DNN/CNN comparison | `local-results/model-comparison-v1/{mlp,dnn,physics_multiscale}/` |
| Final comparison browser audit | `local-results/model-comparison-final-review-8dl4_ha8/browser-report.json` and adjacent screenshots |
| GitHub publication verification | `local-results/github-publication-20260908-18scaq63/publication-report.json` and live screenshot |
| Matching uncertainty and shift verification | `local-results/distribution-shift-review-1uz0m8hz/` contains numerical, browser and publication reports plus screenshots |

The butanol development and partial-run directories are not publication inputs.
On 2026-09-08 the matched dataset hash was checked against its JSON metadata,
its exact 500-bin grid was verified, and its 2,000 events, 100 configurations,
and five source-scan IDs were confirmed. All three benchmark model directories
contain `model.pt` and `scaler.npz`.

## Implemented and reproduced

| Workstream | Evidence and current scope |
| --- | --- |
| Physical forward model | Complex spin-1 susceptibility propagated through the shared single-capacitor Q-meter circuit; passive RLGC cable, finite-source loading, and phase-sensitive readout |
| Measured baseline | Audited public CSV with 35 reported decimal-spacing repairs; all 500 samples retained on the confirmed grid; constrained constant-phase fit remains a diagnostic |
| Experimental matching | Five public raw sweeps preserved byte-for-byte; all selected full-spectrum fits converge without near-bound flags; capacitor and detector phase are fitted |
| Butanol comparison | Owner-confirmed material; C–D/O–D theory fitted to the same five raw sweeps with the original circuit; all selected fits converge; full-sweep RMS falls by 0.0535–4.3525%, with three additional parameters |
| UVA-ND3 data | Public excerpt of five predetermined source records; measured 512-bin frequency arrays and recorded baseline preserved; all five conditional reference-subtracted lineshape fits converge |
| Experiment-anchored generation | `local-results/experiment-anchored.npz` contains 2,000 events, 100 configurations, and all five source-scan IDs; newly sampled simulator truth P and independent synthetic reference noise |
| Existing learning benchmark | Separate 512-bin TE-area generator, compact and multiscale CNNs, training/prediction tools, and three saved runs under `local-results/qmeter/` |
| MLP/DNN/CNN comparison | Three new runs on the same broad 512-bin dataset and exact partitions, saved under `local-results/model-comparison-v1/`, with verified errors and interactive learning curves |
| Tutorial | Main guide, baseline practical, and three experimental examples, with verified fit figures and saved benchmark aggregates |

The 500-bin contract is `f_j = 32.3 + 0.0015287*j MHz`, j=0..499, ending
at 33.0628213 MHz. The nominal center remains 32.7 MHz. The measured setup
confirms n=1 and a 3.580 m cable half-wave at that frequency. These contracts
are distinct from the 512-bin synthetic sweep spanning 32.3–33.1 MHz. The
UVA-ND3 acquisition has a third contract: 512 stored measured frequencies with
slightly varying digitized spacing, ending near 33.0999878 MHz. It does not
inherit the butanol hardware setup or either generated grid.

The five original single-site P estimates, in file order, are approximately 35.08%,
−5.74%, 36.85%, 41.07%, and −7.96%. Full-fit residual RMS is about
6.0–6.6 × 10⁻⁵ recorded units. These are lineshape fit estimates, not independently
known labels or polarization-accuracy measurements. TE calibration is not a
prerequisite for this spin-temperature lineshape extraction.

The butanol two-site P estimates are about 35.27%, −5.76%, 37.26%, 41.44%, and
−8.10%. The corresponding per-scan RMS reductions are 2.57%, 0.05%, 2.37%,
4.35%, and 0.28%. The small improvements for two scans remain visible, as do
all endpoints and residual structure. Fifteen unknowns replace twelve; this is
an in-sample model comparison, not a polarization-accuracy measurement. One
alternative start did not converge; all selected solutions did. The source
theory's weak-quadrupole option and common-site P/linewidth hypotheses are explicit.

UVA-ND3 uses source records 1, 126, 251, 376 and 501, selected before fitting.
Its conditional P estimates range from −25.03% to −34.42%, with full residual RMS
1.7200×10⁻⁵–3.2323×10⁻⁵ recorded units. The model jointly fits a constant complex
readout and cubic residual background after measured reference subtraction.
Its hardware calibration is unresolved. Stored acquisition polarization and
TE-area calibration are not inputs or truth labels. Both new demonstrations
leave the generators, generated datasets and trained networks unchanged.

The earlier constant-phase baseline fit has RMS 8.1195 × 10⁻⁴ recorded units
and reaches its +3% cable-trim and 400 pF stray-capacitance bounds. The separate
phase/capacitance comparison in the matching report has RMS 6.3643 × 10⁻⁵
recorded units and fitted C_tune about 544.84 pF. That model also fixes length
at 3.580 m and C_stray at zero; the comparison changes several assumptions and
does not isolate the effect of phase alone. Neither fit establishes independent
hardware calibration. Both use recorded-minus-fitted residuals.

## Next implementation milestone

The [distribution-shift protocol](distribution-shift.md) now defines the
experimental validation work alongside this implementation: independent repeats
and corresponding references, justified full joint fit uncertainty, residual
means/covariance, distribution diagnostics and conditional known-P stress tests.
The required empirical matrices must not be fabricated from the five development
scans or replaced by independent draws from numerical fit bounds.

The new generator version is `experiment-anchored-qmeter-pake-500-v1`.
`tools/train_model.py`, `tools/predict.py`, and `tools/nmr_lab.py` still implement
the `qmeter-complex-pake-v2` 512-bin contract, including TE-area preprocessing.
They do not yet provide a training/prediction path for the new dataset.

1. Implement and save preprocessing for the exact 500-bin grid and recorded-unit
   raw/reference inputs, without a required TE-area channel. Keep clean simulator
   arrays available for diagnostics, not as experimentally available inputs.
2. Train and save a compatible model, preprocessing, partition, and predictions.
   Fit preprocessing on training data only. When evaluating unseen measured seed
   configurations, group by `source_scan_1based` so their simulated descendants
   cannot cross partitions; grouping only by `configuration_id` is insufficient.
3. Evaluate known-P synthetic recovery, bias, residual width, RMSE, tails, and
   inference cost, then validate against independent experimental acquisitions.
   The five development scans do not supply a held-out experimental accuracy test.

The original three published network results belong to the 512-bin benchmark:
two datasets of 6,000 events and 300 configurations, with 600 test events per run.
The broad compact and broad multiscale runs share the same dataset and partition.
They assume ideal TE calibration and a stable, independently noisy reference.
No network performance has been established for the new 500-bin workflow.

## Scientific work still open

The experimental residuals retain correlation and endpoint structure. Their
high-frequency noise proxy supplies a white-Gaussian reference scenario, not a
measured 500×500 covariance. Controlled excursions around five fitted seeds are
robustness choices, not measured population distributions. Independent component
and detector characterization, model-sensitivity checks, and further acquisitions
remain necessary. A recorded-unit-to-volt conversion has not been supplied;
lineshape matching continues in recorded units.

## Local verification and documentation maintenance

The final 2026-09-08 discovery run passed all 54 tests using the interpreter and
command in the quick-start section. Coverage includes audited readers,
baseline fitting/publication, experimental fitting/reconstruction, matched
generation, independent two-site normalization/complex quadrature, conditional
synthetic recovery, both material publications, all 11 `test_physics` checks,
and the new training and model-comparison contracts.
The earlier 38 passing tests plus a module import error came from selecting
system Python; they were incomplete validation, not a missing PyTorch installation
on this workstation. CUDA availability was checked separately; the unit suite
does not establish GPU training performance or experimental instrument validity.

The earlier documentation audit verified the original generated HTML blocks
against their exporter templates and all three published bias/width/RMSE values
against hashed prediction CSVs. The matched dataset grid, units, array types,
and configuration/source counts were inspected. The material additions have
their own saved data/code/figure hashes, all-bin reconstruction checks and
quadrature refinements. Local PNG copies of all four new figures were inspected.
The final audit checked 226 local/repository links and balanced HTML element
nesting on all five tutorial pages, including links from local notes.

Local datasets and checkpoints are git-ignored reproduction artifacts; new
checkouts use the commands in the [README](../README.md).

The baseline and matching HTML result blocks are generated by
`tools/export_baseline_example.py` and `tools/export_signal_matching.py`.
Update their text templates when changing generated prose, then re-export from
the verified existing fit products. The two new pages are generated by
`tools/export_material_examples.py`; update that template before re-exporting.
Original verified SVG files, fit values, datasets and checkpoints are unchanged.
The completed butanol fits are in `local-results/butanol-comparison-final/`,
assembled from independent per-scan runs using the same six starts and per-scan
seed schedule as the sequential command. Development/partial runs are not
publication inputs. UVA-ND3 fits are in `local-results/uva-nd3-matching/`.
The static Pages workflow deploys `docs/`;
it does not run the Python test suite. Deployment success is not test evidence.
