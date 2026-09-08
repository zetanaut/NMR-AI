# MLP → DNN → CNN on the same training data

Implemented and evaluated 2026-09-08 for the tutorial's
[network-design lesson](../docs/index.html#network). This record covers a new
architecture comparison on the existing **512-bin TE-area benchmark**. Read the
[physics/electronics conventions](physics-electronics-theory.md) and
[baseline record](baseline-fitting.md) for the measurement assumptions.
The forward generator, its data, and the original three CNN runs are unchanged.

## Common data and evaluation protocol

All three estimators use `local-results/qmeter/broad.npz`: 6,000 spectra,
300 simulated configurations, and fractional P sampled in [-0.25, 0.25].
SHA-256: `5bddd6429b8b9c683a359a2b20f68f717a83302fa851f97a9723d2f6c37afdc1`.
This is the existing controlled broad-variation scenario, with ideal TE
calibration and a stable, independently noisy reference. It is separate from
the 500-bin experiment-anchored dataset and the measured material demonstrations.

The declared protocol is [model-comparison.json](../configs/model-comparison.json).
All models use identical raw/reference/calibration arrays and the same
`make_features()` preprocessing. Input shape is (events, 2, 512): raw voltage
and TE-calibrated per-bin reference-subtracted area contributions. Voltage
subtraction precedes float32 network conversion. Scalers and target normalization
are fitted only on training rows. Clean simulator arrays are not network inputs.

The seed-42 configuration split contains 240 training, 30 validation, and 30 test
groups, giving 4,788 / 612 / 600 rows. Each run saves exact row indices in
`partition.npz`, group identities in `split.json`, and its dataset hash.
The exporter reconstructs the split and scalers independently and verifies all
three runs against them. Equal group counts alone are not the comparison check.

All models use AdamW, learning rate 0.0001, weight decay 0.0001, batch size 128,
an 80-epoch cap, patience 12, and normalized absolute MSE. The scheduler halves
the learning rate after three stagnant validation epochs. Seed 42 initializes
each run; a separate seed-42 DataLoader generator gives the same training batch
order independent of architecture initialization and dropout. No test scores
enter the training loop, scheduler, checkpoint choice, or architecture selection.

The trainer first ran with `--defer-test`. All three declared architectures were
retained after inspecting validation results, with no architecture or setting
changes. Their checkpoint hashes were recorded in `evaluation.json` before
running `predict.py --partition test`. Prediction verifies the original dataset
hash before applying the saved row indices. All 600 test rows are retained.

## Models and measured results

| Estimator | Structure | Total / trainable parameters | Best / total epochs |
| --- | --- | --- | --- |
| Basic MLP | Flatten 2×512 → 64 GELU → 1 | 65,665 / 65,665 | 79 / 80 |
| Deep dense DNN | Flatten 2×512 → 256 GELU → 128 GELU → 64 GELU → 1 | 303,617 / 303,617 | 75 / 80 |
| Multiscale CNN + summaries | Width-5/15/31 convolutions, dilated residual blocks, pooling and correction head, plus a train-only ridge estimate from 25 summaries | 82,391 / 82,365 | 15 / 27 |

The dense models have no dropout. The CNN uses dropout 0.1 in its correction
head during training. Its 26 ridge coefficients are frozen after fitting to
training rows. This is a comparison of complete estimators; the large gain
cannot be attributed to convolutions alone. A summary-only ablation would be a
separate study. The original 3,889-parameter compact CNN remains available as
an additional control and in the earlier experiment notebook.

All errors below are **percentage points**, equal to 100 times fractional-P error.

| Estimator | Validation RMSE | Test RMSE | Test bias | Test width | Test 95th absolute error |
| --- | --- | --- | --- | --- | --- |
| Basic MLP | 0.09055280 | 0.06514504 | -0.00125380 | 0.06513297 | 0.12616456 |
| Deep dense DNN | 0.04030151 | 0.03840369 | -0.00227648 | 0.03833616 | 0.07554870 |
| Multiscale CNN + summaries | 0.00196303 | 0.00236318 | -0.00008144 | 0.00236178 | 0.00500701 |

Residual is `P_true - P_pred`; bias is its mean and width its population SD
(`ddof=0`). Independently recomputed metrics obey RMSE² = bias² + width².
The DNN's test RMSE is 1.6963 times smaller than the MLP's; the CNN's is a further
16.2509 times smaller, or 27.5667 times smaller than the MLP's overall.
No SNR is inferred from these regression errors.

These are one-seed, fixed-budget teaching results. The MLP and DNN reach the
80-epoch cap, so the comparison does not establish their best achievable error.
The CNN stops under the same patience rule after 27 epochs. Increasing depth or
using convolution does not guarantee this ordering on another problem. None of
these results establishes experimental accuracy, a calibrated uncertainty
interval, or performance on the 500-bin/material datasets.

## Multiscale explanation expanded — 2026-09-08

The tutorial's `#multiscale-cnn` explanation now separates weight sharing across
frequency positions from freezing the summary regression. Three independent
first-layer banks have 12 filters each at widths 5, 15 and 31. Each filter reads
both input channels and reuses its weights and bias at every position. These
banks contain 1,260 trainable parameters and produce 36 feature channels.
The later encoder produces 48 channels; mean, maximum and sample-SD pooling
concatenate them into 144 inputs to the correction head.

The 25 deterministic summaries are calculated from already standardized inputs.
Let r be the calibrated reference-subtracted channel, v the raw channel, and u
the fixed 512-point coordinate from -1 to 1. Define
`center = sum(abs(r)*u)/max(sum(abs(r)), 1e-6)` and `t = u-center`.
The vector, in code order, contains:

- 11 signed moments `mean(r*t**k)`, k=0..10.
- 7 absolute-amplitude moments `mean(abs(r)*t**k)`, k=0..6.
- Minimum, maximum and sample SD of r.
- Mean, sample SD, minimum and maximum of v.

These moments retain amplitude and use a dimensionless coordinate; they are
not normalized by total area or reported as independently fitted physical
parameters. The feature SD uses denominator 511. Prediction-error width still
uses population SD. A second standardization uses each summary's training-row
mean and population SD, replacing SDs below 1e-8 with 1.

Ridge regression fits 25 weights and one unpenalized intercept to training P
divided by its training-row population SD (floored at 1e-6). Its objective is
summed squared error plus `1e-3*sum(weights**2)`. These 26 coefficients are
then frozen with `requires_grad_(False)` while AdamW trains the other 82,365
parameters. The correction head starts at zero; the combined output is
`target_scale * (ridge_estimate + convolutional_correction)` in fractional P.
The coordinate and summary scaling buffers are saved but excluded from the
82,391 parameter count. Prediction reuses all saved preprocessing and weights.

This is an explanation of the existing implementation; generator, architecture,
training protocol and published results are unchanged.

## Reproduce and verify

Use the interpreter in the [session handoff](project-status.md). If the broad
dataset is already present, reuse it. For a new checkout, generate it with:

```bash
python tools/generate_data.py --num-samples 6000 --num-configurations 300 \
  --coverage broad --output local-results/qmeter/broad.npz
python tools/run_model_comparison.py
python tools/export_model_comparison.py
```

For a fresh output directory, supply the same `--runs-dir` to the runner and
exporter. The runner also supports `--stage train` followed by `--stage evaluate`.
It preserves existing outputs and saves a protocol snapshot, individual logs,
three model directories, and the evaluation decision. Each run directory contains
checkpoint, scaler, partition, group split, history, validation metrics and
predictions, test predictions after evaluation, and provenance. The training
provenance's `test_evaluated: false` records the deferred training stage;
`evaluation.json` records subsequent test evaluation.

Training used Python 3.11.16, PyTorch 2.11.0+cu128, and an NVIDIA RTX A6000.
Saved test predictions were produced on CPU, batch size 128 and two threads.
The exporter independently reconstructs predictions on CPU with one thread.
All test predictions reproduce exactly in this check; the largest validation
CPU/CUDA difference is 7.4506×10⁻⁸ fractional P. This is a numerical check, not a
statistical uncertainty. Library/hardware changes can alter training results.

The publication exporter checks data, code and checkpoint hashes, every saved
partition, training-only scalers, best epochs, parameter counts, and all saved
predictions before writing the public JSON/JS, SVG, and marked HTML block.
The JSON retains all synthetic test predictions and the histories, so metrics
can be checked independently without local checkpoints. Tests also recompute
metrics, verify identical test labels/groups, and check the public figure/table.

Update the exporter template for generated comparison prose. The learning-curve
control lives in `docs/assets/tutorial.js`; its common logarithmic axes display
validation curves together or training/validation curves for one estimator.
The static figure and numerical table remain readable without JavaScript.

The original notebook assets in `docs/assets/results.js` retain their three
40-epoch-cap runs. The new publication is `docs/assets/model-comparison.json`
and its companion JS/SVG, backed by `local-results/model-comparison-v1/`.

## Completion checks — 2026-09-08

The full suite passed all 54 tests in 19.058 seconds using the documented
interpreter. A separate reconstruction verified the entire public snapshot
against the saved models and data, including current exporter and figure hashes.
The README, tutorial status summary and physics notes now include this comparison.

Chromium checks at 1440px and 390px exercised all four learning-curve selections,
verified finite plotted coordinates, and found no JavaScript errors or page
overflow. Both charts preserve readable labels with horizontally scrollable,
keyboard-accessible regions on small screens. The numerical table and error
figure also remain available with JavaScript disabled. Final desktop and mobile
screenshots were inspected; the browser report and images are saved in
`local-results/model-comparison-final-review-8dl4_ha8/`. The browser check reused
existing temporary Playwright tooling; no repository runtime dependency was added.
An HTML audit checked nesting, unique IDs and 162 local/repository links across
the five tutorial pages. Before publication, the exporter was updated to strip
trailing SVG whitespace. The SVG's numeric content and comparison results were
verified unchanged, and both comparison-publication tests passed again.

The requested lesson update was published on 2026-09-08 in commit `75a8699`.
The [Pages deployment](https://github.com/zetanaut/NMR-AI/actions/runs/34235972078)
succeeded, and the [lesson is live](https://zetanaut.github.io/NMR-AI/#network).
All 18 changed pages/assets matched the committed local bytes when retrieved
from the hosted site. Live browser checks exercised all four learning-curve
selections without JavaScript errors. Publication evidence is saved in
`local-results/github-publication-20260908-18scaq63/`.
