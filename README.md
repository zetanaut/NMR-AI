# NMR / AI

A standalone student tutorial on extracting vector polarization from continuous-wave NMR spectra.

**[Start the step-by-step tutorial](https://zetanaut.github.io/NMR-AI/walkthrough.html)** · [Physics and network explanations](https://zetanaut.github.io/NMR-AI/)

Follow the nine steps below in order. The linked walkthrough supplies a **Run → Expected result → Next step** sequence, copy buttons, input checks, and explanations of every output. It starts with the supplied measurements and finishes with a saved network and held-out predictions. No other repository or pretrained checkpoint is required.

All three signal examples fit **raw sweeps with the baseline present**, using the physical circuit and the appropriate spin-1 response together. Every working spectrum, generated example, and network uses the exact **500-bin** grid:

```text
f_j = 32.3 + 0.0015287*j MHz, j = 0..499
first = 32.3000000 MHz; last = 33.0628213 MHz
```

The nominal deuteron center is 32.7 MHz. Preserve the offsets and spacing in [the acquisition contract](configs/deuteron-acquisition.json). UVA-ND3's working arrays are reproducibly resampled from its preserved source archive; its hardware calibration remains unresolved.

## Set up

**Step 1 — Install and check the environment.** Use Python 3.10 or newer and a Bash terminal on Linux, macOS, or Windows with WSL. If the repository already exists, open its directory and start with environment creation or activation.

```bash
git clone https://github.com/zetanaut/NMR-AI.git
cd NMR-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -c "import sys, numpy, scipy, torch, matplotlib; print(sys.executable); print('Imports OK'); print('CUDA:', torch.cuda.is_available())"
python -m unittest discover -s tests -v
```

**Check:** imports succeed and the tests end with `OK`. Either CUDA value is valid; CPU execution is supported. If an import fails, check the printed interpreter and activate the environment before installing requirements again.

Run all subsequent commands from `NMR-AI/`. Repeat activation and the two thread exports in each new terminal. All outputs below use `local-results/walkthrough/`. Use a fresh prefix consistently for a second run; existing fit, dataset, model, and prediction paths are protected.

## Inspect the inputs and fit the baseline

**Step 2 — Preview the measured input.**

```bash
python tools/preview_baseline.py \
  --output local-results/walkthrough/baseline-input.svg
```

**Check:** `Previewed all 500 samples; 35 documented decimal-spacing repairs; no fit`. Open the SVG in your browser. [Step 2 also provides the code to audit both signal datasets](https://zetanaut.github.io/NMR-AI/walkthrough.html#inspect): baseline `(1,500)`, butanol `(5,500)`, and raw UVA-ND3 `(5,500)`.

**Step 3 — Fit the baseline.**

```bash
python tools/fit_baseline.py examples/deuteron-baseline.csv \
  --starts 24 --output-dir local-results/walkthrough/baseline
```

**Check:** open `local-results/walkthrough/baseline/baseline_fits.png` and `fit_report.json` in that folder. The reference RMS is about `0.000811955` recorded units. The constrained fit reaches the +3% cable-trim and 400 pF stray-capacitance bounds. This is a diagnostic, not an accepted hardware calibration. [Step 3 explains the residual and provides the report-checking code](https://zetanaut.github.io/NMR-AI/walkthrough.html#baseline).

## Three experimental examples

The first two examples deliberately use the same five butanol sweeps. All three fit the raw input with its baseline present; plotting the fitted nuclear response afterward does not change the input.

**Step 4 — Example 1: fit the single-site starting model.**

```bash
python tools/match_experimental_signals.py \
  --output-dir local-results/walkthrough/single-site --starts 6
python tools/plot_fit.py --fit-dir local-results/walkthrough/single-site \
  --output local-results/walkthrough/single-site-fits.png
```

**Check:** five scan summaries and five rows in `single-site-fits.png`, showing raw data, full fit, fitted circuit baseline, fitted signal, and raw-minus-fit residuals. Approximate fitted P values are 35.08%, −5.74%, 36.85%, 41.07%, and −7.96%. Keep `single-site/matching_report.json` for steps 5 and 7. [Inspect the expected outputs](https://zetanaut.github.io/NMR-AI/walkthrough.html#single-site).

**Step 5 — Example 2: fit both butanol sites using your step 4 report.**

```bash
python tools/match_butanol.py \
  --single-site-report local-results/walkthrough/single-site/matching_report.json \
  --output-dir local-results/walkthrough/butanol --starts 6
python tools/plot_fit.py --fit-dir local-results/walkthrough/butanol \
  --output local-results/walkthrough/butanol-fits.png
```

**Check:** five RMS comparisons and `butanol-fits.png`. The reference two-site fit reduces RMS by about 2.57%, 0.05%, 2.37%, 4.35%, and 0.28% in scan order. It adds three parameters, so lower in-sample RMS alone does not establish more accurate P. [Compare the results and material assumptions](https://zetanaut.github.io/NMR-AI/walkthrough.html#butanol). This fit can take substantially longer than the single-site example.

**Step 6 — Example 3: fit raw UVA-ND3 with its existing baseline.**

```bash
python tools/match_uva_nd3.py \
  --output-dir local-results/walkthrough/nd3 --starts 6
python tools/plot_fit.py --fit-dir local-results/walkthrough/nd3 \
  --output local-results/walkthrough/nd3-fits.png
```

**Check:** five `record_N.csv` files, `nd3/uva_nd3_report.json`, and `nd3-fits.png`. The fitter selects raw `phase`; the recorded reference and stored subtraction are provenance. No baseline is added. The reference record 376 reaches the 2000 pF tuning-capacitance bound. [Inspect the five fits and their limitations](https://zetanaut.github.io/NMR-AI/walkthrough.html#nd3).

The butanol and UVA-ND3 material demonstrations do not change the single-site generator or establish material-specific network accuracy. See the [material record](notes/material-examples.md), [matching record](notes/experimental-matching.md), and [baseline record](notes/baseline-fitting.md) for sources and conventions.

## Generate experiment-anchored learning data

**Step 7 — Generate new spectra from the step 4 single-site configurations.**

```bash
python tools/generate_matched_data.py \
  --matching-report local-results/walkthrough/single-site/matching_report.json \
  --num-samples 2000 --num-configurations 100 --seed 42 \
  --output local-results/walkthrough/lineshape-data.npz
```

**Check:** the command reports 2,000 new sweeps with 500 bins. [Run the dataset check in step 7](https://zetanaut.github.io/NMR-AI/walkthrough.html#generate): raw and reference arrays have shape `(2000,500)`, network features have shape `(2000,2,500)`, and there is no required TE calibration array. The adjacent JSON documents coverage settings, units, seed configurations, and limitations.

P labels are newly sampled simulator truth in [−0.6,0.6], not the fitted experimental estimates. Raw sweeps contain the baseline; each simulated configuration has an independent noisy reference. Controlled parameter excursions and a high-frequency noise proxy do not establish measured parameter distributions or full noise covariance.

## Train on experiment-anchored lineshapes

**Step 8 — Train with validation selection and defer the test.**

```bash
python tools/train_model.py \
  --data local-results/walkthrough/lineshape-data.npz \
  --output-dir local-results/walkthrough/lineshape-model \
  --architecture physics_multiscale --epochs 80 --patience 12 \
  --learning-rate 0.0001 --seed 42 --batch-order-seed 42 --defer-test
```

**Check:** the trainer reports a saved checkpoint and `test_evaluated=False`. [Inspect `split.json` using the step 8 check](https://zetanaut.github.io/NMR-AI/walkthrough.html#train): sources 3/4/5 give 1,200 training events, source 2 gives 400 validation events, and source 1 gives 400 test events. Keep `model.pt`, `scaler.npz`, `partition.npz`, and the saved configuration together.

The two channels are raw and reference-subtracted recorded units. Subtraction occurs in float64 before conversion to float32. Scalers and ridge coefficients use training rows only; clean simulator arrays are excluded from inputs. All descendants of a measured source stay in the same partition.

**Step 9 — Predict the held-out rows and analyze the errors.**

```bash
python tools/predict.py \
  --model-dir local-results/walkthrough/lineshape-model \
  --data local-results/walkthrough/lineshape-data.npz --partition test \
  --output local-results/walkthrough/test-predictions.csv
python tools/analyze_predictions.py local-results/walkthrough/test-predictions.csv \
  --p0 0.05 --half-width 0.005
```

**Check:** 400 saved predictions and an error report with `pooled.n = 400`. Inspect bias, width, and RMSE in percentage points, plus the count in the signed 4.5%–5.5% band. The [published source-grouped run](docs/matching.html#lineshape-network) has RMSE about 6.37 percentage points. This is synthetic recovery, not measured experimental accuracy; stopping epochs and final decimals can vary by library/device. [Step 9 explains the metrics](https://zetanaut.github.io/NMR-AI/walkthrough.html#predict).

For measured inference, supply aligned `(events,500)` `signals` and corresponding `baselines`, the exact `frequency_mhz`, and scalar strings `feature_mode="lineshape"` and `voltage_unit="recorded units"`. Ordinary prediction needs no P labels or group identifiers. A fitted baseline from the same sweep is not an independent reference; inspect reference drift and the coverage of the trained model before interpreting predictions.

## Continue with the controlled TE-area benchmark

The following exercises have their own dataset and model directories. Both paths use 500 bins, but their inputs and assumptions differ:

| Learning path | Channels and units | Split unit |
| --- | --- | --- |
| Steps 7–9 above | Raw and reference-subtracted recorded units; no TE calibration | Measured source and all simulated descendants |
| TE-area benchmark below | Raw voltage and TE-calibrated reference-subtracted area contributions | Simulated circuit configuration |

The generators propagate complex Dulya/Pake spin-1 susceptibility through the physical Q-meter circuit. Baseline and signal use the same circuit at χ=0 and χ(P). The [physics/electronics record](notes/physics-electronics-theory.md) documents the source theory, single-capacitor topology, units, readout conventions, and unresolved experimental assumptions.

## Compare an MLP, DNN and CNN

The [network-design lesson](docs/index.html#network) builds a one-hidden-layer
MLP, a three-hidden-layer dense DNN, and a multiscale CNN with physical summaries.
It includes runnable code, architecture explanations, an error figure/table,
and interactive learning curves from the saved runs.

All three use the same broad 500-bin dataset, training-only preprocessing,
configuration partitions and training settings. The [comparison record](notes/model-comparison.md)
and the tutorial report measured errors and saved learning curves from the new
runs. These are one-seed results under an 80-epoch cap. The CNN includes a train-only
ridge estimate, so the comparison measures the complete estimators.

If `local-results/qmeter-500-v1/broad.npz` is absent, generate it once with the broad
dataset command in [Reproduce the published runs](#reproduce-the-published-runs).
Then use fresh model output paths:

```bash
python tools/run_model_comparison.py --runs-dir local-results/my-comparison
python tools/export_model_comparison.py --runs-dir local-results/my-comparison
```

The runner reads [the declared protocol](configs/model-comparison.json), trains
with deferred test evaluation, and evaluates the saved test rows after fixing
the comparison. Use `--stage train` and then `--stage evaluate` for a separate
validation review. Checkpoints, preprocessing, partitions and predictions are
saved together. The exporter verifies them before updating the tutorial assets.
See [the comparison record](notes/model-comparison.md) for the full metrics,
provenance and limitations. The additional 500-bin runs below use the same data contracts.

## 500-bin benchmark scope

The deuteron presets are controlled physical sensitivity studies, not experimentally fitted parameter distributions. They use a derived half-wave/tuned operating point near 32.68 MHz and a single-site spin-temperature powder response. Filling factor and susceptibility scale require independent signal calibration; a χ=0 baseline cannot determine them.

Default detector noise is a white-Gaussian reference scenario with σ=10⁻⁹ V. Supply a validated 500×500 output covariance in V² using `--noise-covariance development_covariance_v2.npy`; provide supported joint physical configurations using `--configurations configurations.json`. The generator checks covariance symmetry and positive semidefiniteness. Non-Gaussian pickup and circuit drift need separate characterized extensions.

Each configuration has an independently noisy baseline reference, with 16-sweep averaging by default. The benchmark assumes stable electronics between reference and signal and an ideal TE calibration. It does not include temperature/calibration uncertainty. These conditions make the area-based problem substantially easier than uncontrolled experimental extraction; a small simulated residual is not a complete polarimetry error budget.

## Reproduce the published runs

`docs/assets/results.js` contains three actual runs, their generation/training settings, histories, group counts, prediction hashes, residual histograms, and a reproducible spectrum. Each dataset has 6,000 events and 300 configurations. These are independent teaching networks, not the paper's training runs or accuracy claims.

```bash
python tools/generate_data.py --num-samples 6000 --num-configurations 300 \
  --coverage narrow --output local-results/qmeter-500-v1/narrow.npz
python tools/generate_data.py --num-samples 6000 --num-configurations 300 \
  --coverage broad --output local-results/qmeter-500-v1/broad.npz
python tools/train_model.py --data local-results/qmeter-500-v1/narrow.npz \
  --output-dir local-results/qmeter-500-v1/narrow_multiscale --architecture physics_multiscale \
  --epochs 40 --patience 8 --learning-rate 0.0001
python tools/train_model.py --data local-results/qmeter-500-v1/broad.npz \
  --output-dir local-results/qmeter-500-v1/broad_multiscale --architecture physics_multiscale \
  --epochs 40 --patience 8 --learning-rate 0.0001
python tools/train_model.py --data local-results/qmeter-500-v1/broad.npz \
  --output-dir local-results/qmeter-500-v1/broad_compact --architecture compact \
  --epochs 40 --patience 8 --learning-rate 0.0001
python tools/export_results.py
```

The recorded runs used PyTorch 2.11.0 and CUDA; library/hardware changes can alter results. The compact run reaches the 40-epoch cap while validation is still improving. This is a fixed-budget comparison, not proof of its best achievable accuracy or a capacity limit.

Residuals are **truth minus prediction**, matching the paper. Positive bias means underprediction. Width is population SD, so `RMSE² = bias² + width²`. Multiply fractional P by 100 for percentage points; relative percent at P0 is `100*error/abs(P0)`. Local signed bands and counts are reported separately from pooled conversions. Residual width is not a calibrated per-event confidence interval.

## Website and code

The tutorial also includes a [matching uncertainty and distribution-shift section](docs/index.html#distribution-shift).
It explains how to separate noise and joint baseline/signal fit uncertainty,
retain systematic residuals as well as covariance, compare experimental and
training input distributions, and measure robustness over signed P, SNR and
specified disturbances. The [detailed protocol](notes/distribution-shift.md)
distinguishes these proposed checks from the available experimental evidence;
it does not claim new measured covariance matrices or network accuracy.

Preview with `python3 -m http.server 8000 --bind 127.0.0.1 --directory docs`. The static Pages workflow deploys only `docs/`. There is no build system, external font, or CDN. Lessons and the summary table remain readable without JavaScript.

- `docs/walkthrough.html`, `tools/plot_fit.py`: numbered runnable workflow, expected results, and local raw-fit figures.
- `docs/index.html`, `docs/baseline.html`: theory, network explanations, and baseline practical.
- `docs/matching.html`, `notes/experimental-matching.md`: five measured matches and the 500-bin generator practical.
- `docs/butanol.html`, `docs/uva-nd3.html`, `notes/material-examples.md`: material comparisons, assumptions, the common 500-bin grid and provenance.
- `tools/material_lineshapes.py`, `tools/match_butanol.py`, `tools/match_uva_nd3.py`: standalone single-site ND3 and two-site butanol full-circuit fitting demos.
- `tools/uva_nd3_data.py`, `tools/export_material_examples.py`: audited UVA-ND3 excerpt reader and verified publication of both material demos.
- `tools/circuit.py`, `tools/lineshape.py`: physical electronics and complex nuclear response.
- `tools/fit_tuned_baseline.py`, `configs/baseline-setup.template.json`: fit only declared unknowns using independent tuning information.
- `tools/fit_baseline.py`, `configs/deuteron-baseline-setup.json`: constrained default fit for the supplied n=1, 3.580 m setup.
- `examples/`, `tools/baseline_data.py`, `tools/preview_baseline.py`: public baseline CSV, audited parser, and confirmed-frequency preview.
- `tools/experimental_data.py`, `tools/match_experimental_signals.py`: audited raw-signal reader and full-spectrum lineshape/circuit fitting.
- `tools/generate_matched_data.py`, `tools/export_signal_matching.py`: new 500-bin labeled examples and verified matching figures.
- `tools/baseline_parameters.py`, `tools/export_baseline_example.py`: physical parameter catalogue and verified measured-fit publication.
- `tools/learning_data.py`: exact 500-bin grid, raw/reference and TE-area preprocessing contracts.
- `tools/nmr_lab.py`: generation, calibration, group split and networks.
- `configs/lineshape-training.json`, `tools/export_lineshape_training.py`: source-grouped raw/reference network exercise and verified results.
- `configs/model-comparison.json`, `tools/run_model_comparison.py`, `tools/export_model_comparison.py`: declared same-data comparison, deferred test evaluation, and verified publication.
- `notes/model-comparison.md`: MLP/DNN/CNN results, common data, training protocol and limitations.
- `tools/generate_data.py`, `tools/train_model.py`, `tools/predict.py`: learning workflow.
- `tools/analyze_predictions.py`, `tools/export_results.py`, `tools/benchmark_network.py`: evaluation, published aggregates, timing.
- `tests/`: independent complex quadrature, circuit limits, fit reconstruction, calibration, split and metric checks.

Both authorized raw CSVs and the UVA-ND3 numeric excerpt live in `examples/`. Other measurements, generated datasets, full fit products, and checkpoints stay in git-ignored `local-results/`. The lessons publish selected verified fit records, figures, and benchmark aggregates in `docs/assets/`. The website is independent of the separate research pipeline.
