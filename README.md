# NMR / AI

A standalone student tutorial on extracting vector polarization from continuous-wave NMR spectra.

**[Read the tutorial](https://zetanaut.github.io/NMR-AI/) · [Baseline-fitting practical](https://zetanaut.github.io/NMR-AI/baseline.html)**

The three phases are a physically grounded training-data generator, an efficient DNN/CNN, and validation with bias, residual width, and relative errors at a stated polarization scale.

The generator couples a complex Dulya/Pake spin-1 susceptibility to a passive Q-meter circuit: coil and stray capacitance, RLGC transmission line, one tuning capacitor, finite-source/input loading, and a phase-sensitive detector. Baseline and signal are computed from the same circuit at χ=0 and χ(P). Independent reference subtraction and TE area calibration precede the networks.

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

## Fit a measured baseline

```bash
python tools/fit_baseline.py /path/to/single_event_data.csv \
  --start-mhz 212.6 --step-mhz 0.0015287 \
  --starts 24 --output-dir local-results/baseline-fit
```

The input is a headerless timestamp plus 500-bin CSV. The fitter preserves the first row and identifies exact duplicates. It fits bounded physical shape parameters and solves detector quadratures/DC offset by linear least squares. The output includes measured/fitted overlays, residual CSVs, fitted and fixed parameters, all multi-start candidates, identifiability diagnostics, hashes, and library versions.

Recorded values remain in **recorded units** until the DAQ voltage conversion is confirmed. Supply `--volts-per-unit` only with a known conversion. No reduced χ² or parameter uncertainties are fabricated without a measurement-error model. The page shows one explicitly exported measured-fit illustration; the original CSV, timestamps, and full local fit products remain local.

The [measured example](https://zetanaut.github.io/NMR-AI/baseline.html#example) contains a data/fit overlay, separately scaled residuals, and full-scan statistics. It selects the median-RMS trace among the five distinct sweeps, retaining all 500 bins. Regenerate it with `python tools/export_baseline_example.py --fit-dir local-results/baseline-fit`.

The supplied scans have five distinct Q-curves; the physical model closely follows their backgrounds. One has a localized residual near 212.91 MHz, and several have endpoint residuals. These features remain visible rather than being labeled white noise. A close fit does not uniquely determine every component or establish deuteron settings from proton-frequency scans.

## Run the learning labs

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
- `tools/fit_baseline.py`: measured Q-curve fitting and reproducible diagnostics.
- `tools/nmr_lab.py`: generation, calibration, features, group split, networks.
- `tools/generate_data.py`, `tools/train_model.py`, `tools/predict.py`: learning workflow.
- `tools/analyze_predictions.py`, `tools/export_results.py`, `tools/benchmark_network.py`: evaluation, published aggregates, timing.
- `tests/`: independent complex quadrature, circuit limits, fit reconstruction, calibration, split and metric checks.

Generated datasets, original measurements, full fit products, and checkpoints stay in git-ignored `local-results/`. The baseline lesson publishes an explicitly exported measured illustration and its provenance. The website is independent of the separate research pipeline.
