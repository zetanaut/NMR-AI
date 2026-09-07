# NMR / AI

A self-contained student tutorial on extracting vector polarization from NMR spectra.

**[Read the tutorial →](https://zetanaut.github.io/NMR-AI/)**

The three learning phases are:

1. Build a training-data generator: match experimental spectra, characterize noise, vary physical parameters, and generate enough independent configurations and realizations.
2. Design a DNN/CNN: build a compact CNN, explore a multiscale model with physical summary features, and measure inference cost.
3. Train and evaluate: grouped splits, validation and stopping, bias, residual width, RMSE, and relative error at a stated polarization scale.

The website includes an interactive bias/precision lab, training curves from its own teaching runs, residual histograms, local-band metrics near +5% P, runnable examples, and student assignments. It assumes basic Python and physics, with no prior neural-network experience. Everything needed to run the labs lives in this repository.

## Preview

From this repository:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open [the local tutorial](http://127.0.0.1:8000). No build, JavaScript package manager, external font, or CDN is required. Opening `docs/index.html` directly also works; clipboard support depends on the browser. The print button expands optional examples for PDF export.

## GitHub Pages

Repository: [zetanaut/NMR-AI](https://github.com/zetanaut/NMR-AI). Website: [NMR / AI](https://zetanaut.github.io/NMR-AI/).

The workflow in `.github/workflows/pages.yml` publishes **only `docs/`**, on a relevant push to `main` or a manual workflow run. GitHub Pages is configured with **GitHub Actions** as its source. For a fork, select that source in **Settings → Pages → Build and deployment**. See [GitHub's custom Pages workflow documentation](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages).

## Run the Python labs

Use Python 3.10 or newer. From `NMR-AI/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python tools/generate_data.py \
  --num-samples 2000 --num-configurations 100 \
  --output local-results/smoke.npz

python tools/train_model.py \
  --data local-results/smoke.npz --output-dir local-results/smoke_model \
  --epochs 20 --patience 8 --learning-rate 0.0001

python tools/analyze_predictions.py \
  local-results/smoke_model/test_predictions.csv \
  --p0 0.05 --half-width 0.005

python tools/benchmark_network.py \
  --architecture physics_multiscale --device cpu --batch-size 1
```

On Windows, activate with `.venv\Scripts\activate`. A CPU supports all labs; the trainer automatically uses CUDA when available, or accepts `--device cpu`. Use `--architecture compact` for the smaller CNN. Benchmarking times randomly initialized architecture forward passes, excluding preprocessing and transfers; it does not measure trained accuracy.

Generated datasets, configuration lists, and checkpoints live under git-ignored `local-results/`. The commands protect existing outputs, so choose a new output name for each experiment. No external source checkout, experimental data, or pretrained model is required.

## What the teaching simulator represents

`tools/nmr_lab.py` uses two Gaussian peaks with spin-1 Boltzmann transition weights, a linear gain curve, a cubic baseline, known amplitude calibration, and independent or AR(1) Gaussian noise. Default parameter ranges are synthetic assumptions. This is a learning model, not a validated instrument simulation.

The lessons explain how an experimental application would require fitting real scans, checking residuals and noise correlations, preserving parameter relationships, and validating against independent reference measurements. To experiment with configuration distributions, pass `--configurations configurations.json`; use the `configurations` list in a generated dataset's JSON metadata as the schema. Center, splitting, width, gain slope, four baseline coefficients, and nonzero `cc` are required.

`tools/train_model.py` splits by `configuration_id`, fits input and target scaling only on training rows, trains with absolute normalized MSE, restores the best validation weights, and records held-out predictions. `tools/predict.py` uses the same preprocessing and loads only this tutorial's model architectures.

## Data provenance and interpretation

`docs/assets/results.js` contains three actual runs of this repository's **simplified teaching simulator**: narrow variation with the multiscale model, broad variation with the same model, and the compact model on the same broad dataset. Each dataset has 6,000 spectra drawn from 300 synthetic configurations. The snapshot includes generation/training settings, split counts, histories, residual histograms, summary metrics, and prediction-CSV hashes. It also contains one simulated trace for the component viewer.

To regenerate all displayed examples, run these from this repository in an environment with its requirements installed. Use fresh output paths or preserve your existing runs elsewhere first:

```bash
python tools/generate_data.py --num-samples 6000 --num-configurations 300 \
  --coverage narrow --seed 42 --output local-results/narrow.npz
python tools/generate_data.py --num-samples 6000 --num-configurations 300 \
  --coverage broad --seed 42 --output local-results/broad.npz
python tools/train_model.py --data local-results/narrow.npz \
  --output-dir local-results/narrow_multiscale --architecture physics_multiscale \
  --epochs 25 --patience 8 --learning-rate 0.0001
python tools/train_model.py --data local-results/broad.npz \
  --output-dir local-results/broad_multiscale --architecture physics_multiscale \
  --epochs 25 --patience 8 --learning-rate 0.0001
python tools/train_model.py --data local-results/broad.npz \
  --output-dir local-results/broad_compact --architecture compact \
  --epochs 25 --patience 8 --learning-rate 0.0001
python tools/export_results.py
```

The snapshot recomputes bias, population residual SD (`ddof=0`), RMSE, MAE, and the 95th-percentile absolute error from the prediction CSVs in float64. The identity `RMSE² = bias² + width²` uses the same events and population SD. Relative values use a fixed nonzero reference `P0`; local-band metrics select signed truth values before computation. No uncertainty intervals are inferred from residual width.

The recorded examples used PyTorch 2.11.0 with CUDA. Hardware and library versions can change numerical training results. These measurements illustrate learning and generalization within a deliberately simple simulator. They do not establish experimental accuracy; `0.0005` fractional P is an example acceptance target for the lessons.

## Editing

- `docs/index.html`: the complete lesson text, examples, and semantic structure.
- `docs/assets/style.css`: responsive and print styles.
- `docs/assets/tutorial.js`: interactive plots, error conversion, navigation, and code copying.
- `tools/nmr_lab.py`: teaching simulator, preprocessing, grouped splitting, and both networks.
- `tools/generate_data.py`, `tools/train_model.py`, `tools/predict.py`: complete standalone lab workflow.
- `tools/analyze_predictions.py`, `tools/benchmark_network.py`, `tools/export_results.py`: evaluation, timing, and website snapshots.

All website asset paths are relative, so the site works at the `/NMR-AI/` project URL. The content remains readable without JavaScript; interactive plots require it.

## Validation

The included runs exercise generation, grouped training, saved checkpoints, and error reporting. Both network benchmarks run independently on CPU. Before publishing changes, check inference on a separate generated dataset, group separation, finite outputs, internal links, interactive controls, and the mobile layout. Keep simulator validation distinct from website and software checks.

Run the numerical and split checks with `python -m unittest discover -s tests -v`.
