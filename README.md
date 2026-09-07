# NMR / AI

A standalone student tutorial on extracting vector polarization from continuous-wave NMR spectra.

**[Read the tutorial](https://zetanaut.github.io/NMR-AI/) · [Baseline practical](https://zetanaut.github.io/NMR-AI/baseline.html) · [Experimental lineshape matching](https://zetanaut.github.io/NMR-AI/matching.html)**

The three phases are a physically grounded training-data generator, an efficient DNN/CNN, and validation with bias, residual width, and relative errors at a stated polarization scale.

The generators couple complex Dulya/Pake spin-1 susceptibility to a passive Q-meter circuit: coil and stray capacitance, RLGC transmission line, one tuning capacitor, finite-source/input loading, and a phase-sensitive detector. Baseline and signal are computed from the same circuit at χ=0 and χ(P). The new experimental-matching workflow extracts polarization from the spin-1 lineshape without TE calibration. The existing 512-bin learning benchmark separately uses TE-area features.

The scientific source is [Seay, Fernando, and Keller, arXiv:2603.10146v5](https://arxiv.org/abs/2603.10146v5). Read the [detailed physics/electronics notes](notes/physics-electronics-theory.md) and [baseline-fitting record](notes/baseline-fitting.md) for derivations, conventions, component assumptions, and measured-fit diagnostics.

## Set up

Use Python 3.10 or newer, from `NMR-AI/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

On Windows activate with `.venv\Scripts\activate`. A CPU supports all labs; training automatically uses CUDA when available. No other repository, research data checkout, or pretrained model is required.

## Fit a measured deuteron baseline

The public [deuteron-baseline.csv](examples/deuteron-baseline.csv) is the owner's
measured test example: one timestamp plus 500 amplitudes, with nominal center
**32.7 MHz**. Every supplied baseline uses the same confirmed mapping:
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

Both commands below use the same setup-driven fitter. To add independent
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

The public CSV is the explicitly authorized raw-data exception. Other measurements
and full fit products stay local. The main 512-bin synthetic training benchmark is
unchanged; it is not silently relabeled as this 500-bin measured acquisition.

## Match the supplied spin-1 signals and make training data

[Sample_RawSignal.csv](examples/Sample_RawSignal.csv) contains five experimental
spin-1 sweeps, published byte-for-byte with permission. The grid and n=1 cable
setup are confirmed to match the baseline example. The reader audits Unicode
line separators and blank lines without changing samples or record order.

```bash
python tools/match_experimental_signals.py \
  --output-dir local-results/experimental-matching --starts 6
python tools/generate_matched_data.py \
  --matching-report local-results/experimental-matching/matching_report.json \
  --num-samples 2000 --num-configurations 100 --seed 42 \
  --output local-results/experiment-anchored.npz
python tools/export_signal_matching.py \
  --fit-dir local-results/experimental-matching \
  --dataset local-results/experiment-anchored.npz
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
This new 500-bin recorded-unit lineshape dataset is separate from the existing
512-bin TE-area learning benchmark below; it needs grid-aware preprocessing
without a required TE calibration channel. No new trained-network accuracy is
claimed here.

## Run the existing 512-bin learning labs

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

The compact CNN has 3,889 parameters; the multiscale CNN with train-only physical summaries has 82,391. Use `--architecture compact` for the smaller model. Benchmarking measures architecture forward passes, not trained accuracy or end-to-end latency.

Generated data contain raw/reference voltages in V, an independent TE calibration per event, fractional P, configuration IDs, the frequency grid, and `qmeter-complex-pake-v2` provenance. Voltage storage is float64; network inputs are float32 after subtraction/calibration. Generation metadata retain all physical configurations and dataset hashes. Existing outputs are protected; use fresh paths.

Training splits entire configurations 80/10/10. Scalers and the ridge estimate use training rows only. Validation selects the best checkpoint and early stopping; the test partition is evaluated afterward. Prediction uses the same saved preprocessing:

```bash
python tools/generate_data.py --num-samples 500 --configuration-seed 91 --seed 99 \
  --output local-results/new_spectra.npz
python tools/predict.py --model-dir local-results/smoke_model \
  --data local-results/new_spectra.npz --output local-results/new_predictions.csv
```

## Model scope

The deuteron presets are controlled physical sensitivity studies, not experimentally fitted parameter distributions. They use a derived half-wave/tuned operating point near 32.68 MHz and a single-site spin-temperature powder response. Filling factor and susceptibility scale require independent signal calibration; a χ=0 baseline cannot determine them.

Default detector noise is a white-Gaussian reference scenario with σ=10⁻⁹ V. Supply a validated 512×512 output covariance in V² using `--noise-covariance development_covariance_v2.npy`; provide supported joint physical configurations using `--configurations configurations.json`. The generator checks covariance symmetry and positive semidefiniteness. Non-Gaussian pickup and circuit drift need separate characterized extensions.

Each configuration has an independently noisy baseline reference, with 16-sweep averaging by default. The benchmark assumes stable electronics between reference and signal and an ideal TE calibration. It does not include temperature/calibration uncertainty. These conditions make the area-based problem substantially easier than uncontrolled experimental extraction; a small simulated residual is not a complete polarimetry error budget.

## Reproduce the published runs

`docs/assets/results.js` contains three actual runs, their generation/training settings, histories, group counts, prediction hashes, residual histograms, and a reproducible spectrum. Each dataset has 6,000 events and 300 configurations. These are independent teaching networks, not the paper's training runs or accuracy claims.

```bash
python tools/generate_data.py --num-samples 6000 --num-configurations 300 \
  --coverage narrow --output local-results/qmeter/narrow.npz
python tools/generate_data.py --num-samples 6000 --num-configurations 300 \
  --coverage broad --output local-results/qmeter/broad.npz
python tools/train_model.py --data local-results/qmeter/narrow.npz \
  --output-dir local-results/qmeter/narrow_multiscale --architecture physics_multiscale \
  --epochs 40 --patience 8 --learning-rate 0.0001
python tools/train_model.py --data local-results/qmeter/broad.npz \
  --output-dir local-results/qmeter/broad_multiscale --architecture physics_multiscale \
  --epochs 40 --patience 8 --learning-rate 0.0001
python tools/train_model.py --data local-results/qmeter/broad.npz \
  --output-dir local-results/qmeter/broad_compact --architecture compact \
  --epochs 40 --patience 8 --learning-rate 0.0001
python tools/export_results.py
```

The recorded runs used PyTorch 2.11.0 and CUDA; library/hardware changes can alter results. The compact run reaches the 40-epoch cap while validation is still improving. This is a fixed-budget comparison, not proof of its best achievable accuracy or a capacity limit.

Residuals are **truth minus prediction**, matching the paper. Positive bias means underprediction. Width is population SD, so `RMSE² = bias² + width²`. Multiply fractional P by 100 for percentage points; relative percent at P0 is `100*error/abs(P0)`. Local signed bands and counts are reported separately from pooled conversions. Residual width is not a calibrated per-event confidence interval.

## Website and code

Preview with `python3 -m http.server 8000 --bind 127.0.0.1 --directory docs`. The static Pages workflow deploys only `docs/`. There is no build system, external font, or CDN. Lessons and the summary table remain readable without JavaScript.

- `docs/index.html`, `docs/baseline.html`: three-phase guide and baseline practical.
- `tools/circuit.py`, `tools/lineshape.py`: physical electronics and complex nuclear response.
- `tools/fit_tuned_baseline.py`, `configs/baseline-setup.template.json`: fit only declared unknowns using independent tuning information.
- `tools/fit_baseline.py`, `configs/deuteron-baseline-setup.json`: constrained default fit for the supplied n=1, 3.580 m setup.
- `examples/`, `tools/baseline_data.py`, `tools/preview_baseline.py`: public measured CSV, audited parser, and sample-index preview.
- `tools/baseline_parameters.py`, `tools/export_baseline_example.py`: physical parameter catalogue and verified measured-fit publication.
- `tools/nmr_lab.py`: generation, calibration, features, group split, networks.
- `tools/generate_data.py`, `tools/train_model.py`, `tools/predict.py`: learning workflow.
- `tools/analyze_predictions.py`, `tools/export_results.py`, `tools/benchmark_network.py`: evaluation, published aggregates, timing.
- `tests/`: independent complex quadrature, circuit limits, fit reconstruction, calibration, split and metric checks.

The public test CSV lives in `examples/`. Other measurements, generated datasets, full fit products, and checkpoints stay in git-ignored `local-results/`. The baseline lesson publishes a verified measured fit only after acquisition metadata are established. The website is independent of the separate research pipeline.
