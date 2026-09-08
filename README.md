# NMR / AI

A standalone student tutorial on extracting vector polarization from continuous-wave NMR spectra.

**[Read the tutorial](https://zetanaut.github.io/NMR-AI/) · [Baseline practical](https://zetanaut.github.io/NMR-AI/baseline.html) · [Experimental lineshape matching](https://zetanaut.github.io/NMR-AI/matching.html)**

The three phases are a physically grounded training-data generator, a progression from MLP to DNN to CNN on the same data, and validation with bias, residual width, and relative errors at a stated polarization scale.

The generators couple complex Dulya/Pake spin-1 susceptibility to a passive Q-meter circuit: coil and stray capacitance, RLGC transmission line, one tuning capacitor, finite-source/input loading, and a phase-sensitive detector. Baseline and signal are computed from the same circuit at χ=0 and χ(P). The experimental-matching workflow extracts polarization from the spin-1 lineshape without TE calibration. The 500-bin learning benchmark separately uses TE-area features.

The scientific source is [Seay, Fernando, and Keller, arXiv:2603.10146v5](https://arxiv.org/abs/2603.10146v5). Read the [detailed physics/electronics notes](notes/physics-electronics-theory.md) and [baseline-fitting record](notes/baseline-fitting.md) for derivations, conventions, component assumptions, and measured-fit diagnostics.

## The 500-bin acquisition

All generated training examples and learning models use the confirmed grid:

```
f_j = 32.3 + 0.0015287*j MHz, j = 0..499
500 samples · first 32.3000000 MHz · last 33.0628213 MHz
```

The nominal deuteron center is 32.7 MHz. Preserve the offsets and spacing in
[deuteron-acquisition.json](configs/deuteron-acquisition.json); do not recenter
a symmetric linspace. Models save and validate the grid and preprocessing.

| Exercise | Inputs | Independent split unit |
| --- | --- | --- |
| Experiment-anchored lineshape regression | Raw and reference-subtracted recorded units; no TE calibration required | Measured source scan and all its simulated descendants |
| Controlled TE-area benchmark | Raw voltage and TE-calibrated, reference-subtracted area contributions | Simulated circuit configuration |

Both exercises have 500 bins. Their units and calibration assumptions are explicit.

## Set up

Use Python 3.10 or newer, from `NMR-AI/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

On Windows activate with `.venv\Scripts\activate`. A CPU supports all labs; training automatically uses CUDA when available. No other repository, research data checkout, or pretrained model is required.

The full test suite imports PyTorch. If that import fails, check that the selected
Python environment contains the packages in `requirements.txt` before rerunning.

## Fit a measured deuteron baseline

The public [deuteron-baseline.csv](examples/deuteron-baseline.csv) is the owner's
measured test example: one timestamp plus 500 amplitudes, with nominal center
**32.7 MHz**. This baseline and the five butanol sweeps use the confirmed mapping:
`f_j = 32.3 + 0.0015287*j MHz`, j=0…499; last sample 33.0628213 MHz.
See [the acquisition contract](configs/deuteron-acquisition.json) and
[data README](examples/README.md). The loader preserves all samples and explicitly
repairs 35 decimal-spacing artifacts in memory.

Reproduce the [measured fit](https://zetanaut.github.io/NMR-AI/baseline.html#example):

```bash
python tools/fit_baseline.py examples/deuteron-baseline.csv \
  --starts 24 --output-dir local-results/deuteron-tuned-baseline-fit
python tools/export_baseline_example.py --fit-dir local-results/deuteron-tuned-baseline-fit
```

The default fit now enforces the owner's **n = 1** branch and **3.580 m cable
half-wavelength at 32.7 MHz**, using consistent RLGC propagation (velocity factor
0.7809803). The [setup preset](configs/deuteron-baseline-setup.json) permits only
a provisional ±3% trim, an explicit working bound rather than a measured uncertainty.

The constrained result reaches the upper trim and stray-capacitance limits.
RMS is approximately 8.1195 × 10⁻⁴ recorded units (0.3296% of peak-to-peak range),
with strongly correlated residuals. All 500 bins remain visible. This is a
**diagnostic fit, not an accepted hardware calibration**: additional capacitor,
coil, cable-loss and detector records are needed. The tutorial explains the
warnings and lists every fixed, fitted, profiled and derived quantity.

Both fitting entry points use the same setup-driven fitter. To add independent
measurements for this setup, copy and refine `deuteron-baseline-setup.json`.
For another setup, complete `configs/baseline-setup.template.json`:

```bash
python tools/fit_tuned_baseline.py examples/deuteron-baseline.csv \
  --setup my-known-setup.json \
  --starts 12 --output-dir local-results/tuning-informed-baseline-fit
```

Known quantities are fixed; only declared unknowns vary within supported bounds.
A known cable half-wave multiple uses length = n*pi/beta + delta_length at the
recorded tuning frequency. Gaussian constraints require a data-noise scale.
Missing hardware records are not filled with invented measurements.

`--acquisition` selects another documented contract; alternative start, step,
and frequency-source overrides must be supplied together. Fit reports save the
acquisition hash, parsing record, fixed/fitted values, numerical bounds,
starting-point candidates, residual statistics, and reconstruction information.

The owner authorized this baseline CSV, the five butanol signal sweeps in
`examples/Sample_RawSignal.csv`, and the new UVA-ND3 teaching excerpt for publication.
Other measurements and full fit products stay local. Both learning exercises use the confirmed 500-bin grid; their input units and
calibration assumptions are saved separately.

## Three experimental examples

Experimental lineshapes depend on the material and setup. These examples cover
two materials and two acquisitions; the first two deliberately use the same data.

| Example | Data and purpose |
| --- | --- |
| [1. Single-site starting model](docs/matching.html) | Preserve the original full-circuit exercise and its 500-bin generator |
| [2. Butanol: C–D and O–D](docs/butanol.html) | Apply the supplied two-site theory to the same five butanol sweeps and compare every full residual |
| [3. UVA-ND3 data](docs/uva-nd3.html) | Fit five predetermined records with their measured 512-bin frequency arrays and recorded baseline subtraction |

The butanol comparison adds three fit parameters while retaining the original
electronics assumptions. The UVA-ND3 model uses a conditional complex lineshape
and joint residual background because its hardware calibration is unresolved.
Neither fit comparison is a measured polarization-accuracy claim. See the
[material-model and provenance record](notes/material-examples.md).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/match_butanol.py \
  --output-dir local-results/butanol-matching --starts 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/match_uva_nd3.py \
  --output-dir local-results/uva-nd3-matching --starts 6
python tools/export_material_examples.py \
  --butanol-fit-dir local-results/butanol-matching \
  --nd3-fit-dir local-results/uva-nd3-matching
```

All inputs needed for these commands are included here. Use fresh output paths.
The new material demonstrations do not change the existing generators or networks.

## Example 1: match the supplied spin-1 signals and make training data

[Sample_RawSignal.csv](examples/Sample_RawSignal.csv) contains five experimental
spin-1 sweeps, now identified by the owner as butanol, published byte-for-byte with permission. The grid and n=1 cable
setup are confirmed to match the baseline example. The reader audits Unicode
line separators and blank lines without changing samples or record order.

```bash
python tools/match_experimental_signals.py \
  --output-dir local-results/experimental-matching --starts 6
python tools/generate_matched_data.py \
  --matching-report local-results/experimental-matching/matching_report.json \
  --num-samples 2000 --num-configurations 100 --seed 42 \
  --output local-results/experiment-anchored-500-v2.npz
python tools/export_signal_matching.py \
  --fit-dir local-results/experimental-matching \
  --dataset local-results/experiment-anchored-500-v2.npz
```

This fits the tuning capacitor, frequency-dependent detector phase, readout and
complex Pake lineshape against all 500 samples of every scan. Polarizations are
determined from the intrinsic branch shapes/area ratio; no TE normalization is
needed. The selected P estimates are approximately 35.08%, −5.74%, 36.85%,
41.07%, and −7.96% in file order, with residual RMS about 6.0–6.6 × 10⁻⁵
recorded units. These are fit results, not calibrated uncertainty claims.

The generator draws **new simulator-known P labels**, with explicit robustness
excursions around the complete fitted seed configurations. Its first noise
model uses a high-frequency noise-scale proxy; correlated residual structure
and endpoint artifacts remain visible and need separate modeling. A supplied
500×500 covariance can replace the white-Gaussian reference. Five scans do not
define a measured population distribution or a large held-out test set.

The [new practical](https://zetanaut.github.io/NMR-AI/matching.html) shows all
fits, residuals, physical parameter meanings, and a generated example. The
[matching record](notes/experimental-matching.md) documents every step.
The trainer and predictor accept this 500-bin recorded-unit dataset directly.
They use raw and reference-subtracted channels without a required `calibration`
array, subtracting in float64 before float32 conversion. Clean simulator arrays
remain diagnostic products, never model inputs.

## Train on experiment-anchored lineshapes

```bash
python tools/train_model.py --data local-results/experiment-anchored-500-v2.npz \
  --output-dir local-results/my-lineshape-model --architecture physics_multiscale \
  --epochs 80 --patience 12 --learning-rate 0.0001 --batch-order-seed 42
python tools/predict.py --model-dir local-results/my-lineshape-model \
  --data local-results/experiment-anchored-500-v2.npz --partition test \
  --output local-results/my-lineshape-test.csv
```

All scalers and ridge coefficients use training rows only. Splits group by
`source_scan_1based`, so all descendants of a measured seed stay together.
Five sources give three training, one validation and one test source. The
[matching practical](docs/matching.html) includes the saved synthetic evaluation.
Reproduce its declared run with the commands below, using a matching fresh
`--runs-dir` for both commands when needed.

```bash
python tools/run_model_comparison.py --protocol configs/lineshape-training.json
python tools/export_lineshape_training.py
```

A held-out synthetic source group is not an independent experimental accuracy test.

For measured inference, provide NPZ arrays `signals` and corresponding `baselines`
of shape `(events,500)`, the exact `frequency_mhz`, and scalar strings
`feature_mode="lineshape"` and `voltage_unit="recorded units"`. Ordinary prediction
needs neither P labels nor simulator/group identifiers. The saved contract checks
units and preprocessing and predictions are fractional P.

## Run the 500-bin learning labs

```bash
python tools/generate_data.py --num-samples 2000 --num-configurations 100 \
  --output local-results/smoke.npz
python tools/train_model.py --data local-results/smoke.npz \
  --output-dir local-results/smoke_model --epochs 20 --patience 8 \
  --learning-rate 0.0001
python tools/analyze_predictions.py local-results/smoke_model/test_predictions.csv \
  --p0 0.05 --half-width 0.005
python tools/benchmark_network.py --architecture physics_multiscale --device cpu --batch-size 1
```

Choose `--architecture mlp` (64,129 parameters), `dnn` (297,473), `compact`
(3,889), or the default `physics_multiscale` (82,391, including 26 frozen
coefficients fitted to training-only physical summaries). Benchmarking measures
architecture forward passes, not trained accuracy or end-to-end latency.

Generated data contain raw/reference voltages in V, an independent TE calibration per event, fractional P, configuration IDs, the frequency grid, and `qmeter-complex-pake-500-v3` provenance. Voltage storage is float64; network inputs are float32 after subtraction/calibration. Generation metadata retain all physical configurations and dataset hashes. Existing outputs are protected; use fresh paths.

Training splits entire configurations 80/10/10. Scalers and the ridge estimate use training rows only. Validation selects the best checkpoint and early stopping; the test partition is evaluated afterward. Prediction uses the same saved preprocessing:

```bash
python tools/generate_data.py --num-samples 500 --configuration-seed 91 --seed 99 \
  --output local-results/new_spectra.npz
python tools/predict.py --model-dir local-results/smoke_model \
  --data local-results/new_spectra.npz --output local-results/new_predictions.csv
```

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

- `docs/index.html`, `docs/baseline.html`: three-phase guide and baseline practical.
- `docs/matching.html`, `notes/experimental-matching.md`: five measured matches and the 500-bin generator practical.
- `docs/butanol.html`, `docs/uva-nd3.html`, `notes/material-examples.md`: material comparisons, assumptions, measured grids and provenance.
- `tools/material_lineshapes.py`, `tools/match_butanol.py`, `tools/match_uva_nd3.py`: standalone two-site/full-circuit and reference-subtracted fitting demos.
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
