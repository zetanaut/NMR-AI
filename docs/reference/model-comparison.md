# MLP → DNN → CNN on 500-bin spectra

The [network lesson](https://zetanaut.github.io/NMR-AI/index.html#network) compares three estimators on
newly generated 500-bin TE-area examples. The grid is
`32.3 + 0.0015287*j MHz`, j=0..499, ending at 33.0628213 MHz.
The nominal deuteron center is 32.7 MHz. The [physics notes](physics-electronics-theory.md)
record the full-circuit and calibration assumptions.

## Common data and protocol

The [declared protocol](../../configs/model-comparison.json) uses
`local-results/qmeter-500-v1/broad.npz`: 6,000 spectra from 300 controlled
simulated configurations, with new fractional P drawn in [-0.25,0.25].
Dataset SHA-256: `af70281eb8bd6efd52dcfe3b764d2bc12712259207342b937bb9e9f2aa157c37`.
The generator is `qmeter-complex-pake-500-v3`. These results come from regeneration
and retraining on the exact 500-bin grid.

Every estimator sees the same `(events,2,500)` inputs: raw voltage and
TE-calibrated, reference-subtracted area contributions. Subtraction takes place
in float64 before float32 feature conversion. The independent noisy reference
is shared within each configuration; TE calibration is ideal in this exercise.
Scalers, target normalization and ridge fitting use training rows only.

Seed 42 assigns 240/30/30 configuration groups to training/validation/test,
giving 4,788/612/600 rows. Exact indices, group IDs and the dataset hash are saved.
All three use AdamW at learning rate 0.0001, weight decay 0.0001, batch size 128,
an 80-epoch cap, patience 12 and normalized absolute MSE. The learning rate halves
after three stagnant validation epochs. A separate seed-42 batch-order generator
keeps the data order common across architectures. Validation selects each
checkpoint; test evaluation is deferred until the declared comparison is fixed.

## Measured results

All errors below are percentage points, equal to 100 times fractional-P error.

| Estimator | Total / trainable parameters | Best / total epochs | Validation RMSE | Test RMSE | Test bias | Test width | Test 95th absolute error |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Basic MLP | 64,129 / 64,129 | 80 / 80 | 0.05596309 | 0.06447080 | 0.01155447 | 0.06342695 | 0.13340659 |
| Deep dense DNN | 297,473 / 297,473 | 77 / 80 | 0.04028927 | 0.03405602 | -0.00505858 | 0.03367823 | 0.07102668 |
| Multiscale CNN + summaries | 82,391 / 82,365 | 20 / 32 | 0.00208873 | 0.00257513 | -0.00021951 | 0.00256576 | 0.00532694 |

Residual is `P_true - P_pred`. Bias is its mean and width is its population SD
(ddof=0), so RMSE² = bias² + width². A width is not a calibrated uncertainty
interval. The [public comparison data](../assets/model-comparison.json)
retain every test prediction and training history.

This is one seed and a common epoch cap. It does not establish the best achievable
error of an architecture, a universal ordering, or experimental accuracy. The
CNN includes a fitted ridge estimator, so its performance measures the complete
estimator rather than the isolated benefit of convolution. The separate
[raw/reference lineshape exercise](https://zetanaut.github.io/NMR-AI/matching.html#lineshape-network) uses
recorded units, no TE calibration, a different P range and source-scan holdouts;
its error belongs to that contract.

## Shared filters, summaries and frozen coefficients

The basic MLP flattens 2×500 inputs into a 1,000 → 64 → 1 network. The dense
DNN uses 1,000 → 256 → 128 → 64 → 1. Both use GELU hidden activations.

The multiscale CNN begins with three independent banks of 12 filters, at widths
5, 15 and 31. Each filter uses both channels and shares its weights across
frequency positions. The banks contain 1,260 trainable parameters and produce
36 channels. A 1×1 convolution mixes them into 48 channels, followed by dilated
residual blocks, group normalization and pooling. Mean, maximum and sample-SD
pooling give 144 inputs to a 64-unit correction head with dropout 0.1.

A second branch computes 25 deterministic summaries from standardized inputs.
Let r be the standardized TE-area channel, v the standardized raw channel,
and u the 500-point dimensionless coordinate from -1 to 1. Define
`center = sum(abs(r)*u)/max(sum(abs(r)),1e-6)` and `t = u-center`.
In code order the summaries are:

- Eleven signed moments `mean(r*t**k)`, k=0..10.
- Seven absolute-amplitude moments `mean(abs(r)*t**k)`, k=0..6.
- Minimum, maximum and sample SD of r.
- Mean, sample SD, minimum and maximum of v.

These moments retain amplitude; they are not divided by signal area or reported
in MHz. Feature sample SD uses denominator 499 inside the square root, while
prediction-error width uses population SD. Each of the 25 summaries is then
standardized by its own training-row mean and population SD; SDs below 1e-8
are replaced by 1.

Ridge regression fits 25 weights and an unpenalized intercept against P divided
by the training-row population SD (floored at 1e-6). Its objective is summed
squared error plus `0.001*sum(weights**2)`. These 26 coefficients are frozen
while AdamW trains the remaining 82,365 parameters. The correction head begins
at zero. Final fractional P is
`target_scale*(ridge_estimate + convolutional_correction)`.
The fixed coordinate and summary scaling statistics are saved buffers, excluded
from the 82,391 parameter count.

## Reproduce and verify

Create an environment using the [README](../../README.md#set-up). Generate the broad
dataset once and use fresh output paths:

```bash
python tools/generate_data.py --num-samples 6000 --num-configurations 300 \
  --coverage broad --output local-results/qmeter-500-v1/broad.npz
python tools/run_model_comparison.py
python tools/export_model_comparison.py
```

Use the same `--runs-dir` on the last two commands for a fresh comparison.
The runner also supports separate `--stage train` and `--stage evaluate` steps.
Runs are stored in `local-results/model-comparison-500-v2/` with checkpoints,
preprocessing, exact partitions, histories, predictions, and code/data hashes.
They are inference checkpoints, not optimizer-resume states.

The exporter reconstructs inputs, partitions, training-only scalers, parameter
counts, best epochs and saved predictions before publishing. Test predictions
are recomputed from the saved checkpoints on CPU; numerical reconstruction
differences are saved in the public record. Browser controls display the saved
learning curves; the static error figure/table remain readable without JavaScript.

The additional 40-epoch CNN comparison uses the same 500-bin contract and fresh
runs under `local-results/qmeter-500-v1/`. Its separate aggregate is
`docs/assets/results.js`; see the [benchmark guide](benchmarks.md#reproduce-the-additional-cnn-runs) for commands.
