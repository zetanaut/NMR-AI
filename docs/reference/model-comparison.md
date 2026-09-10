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

“TE-area” names the chosen calibration reference, not the allowed signal scale.
An area-to-polarization map can use a suitable known-P reference at enhanced
polarization as well, provided its response transfers to the target conditions.
Boltzmann spin populations need not be at the lattice temperature, but population
equilibrium alone does not establish the amplitude scale or circuit linearity.
See [the reference derivation](physics-electronics-theory.md#area-calibration-beyond-a-thermal-reference).
The saved data still use their original ideal TE reference; no inputs or scores
change with this explanation.

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

### Why a convolutional network for a one-dimensional spectrum?

A Conv1d filter slides over frequency; its output is a feature map of responses
at successive positions. The two input channels are aligned views of one sweep,
not two spatial dimensions. Local connections encode the expectation that
neighboring bins jointly describe useful patterns. Sharing each filter across
positions lets evidence about a similar horn, shoulder or slope update the same
weights even when it occurs at a different frequency. Several widths and later
layers combine local features over larger contexts.

This is a structural hypothesis appropriate to test on spectra with local
correlation and modest line shifts. It reduces parameters compared with a dense
map of the same input/output sizes. It does not prove a speed or accuracy
advantage over every alternative, nor assign physical meanings to learned
filters. The nonlinear detector response and finite scan window make arbitrary
frequency shifts physically consequential. Convolution's translation equivariance
means a pattern's feature map shifts with the input; the complete estimator is
not exactly invariant to those shifts.

The general concepts are described in [Deep Learning, chapter 9](https://www.deeplearningbook.org/contents/convnets.html).
The implementation below, rather than that general source, defines this model's
architecture. [PyTorch Conv1d](https://docs.pytorch.org/docs/2.11/generated/torch.nn.Conv1d.html)
uses cross-correlation: the sliding filter coefficients are not reversed.

The basic MLP flattens 2×500 inputs into a 1,000 → 64 → 1 network. The dense
DNN uses 1,000 → 256 → 128 → 64 → 1. Both use GELU hidden activations.

The multiscale CNN begins with three independent banks of 12 filters, at widths
5, 15 and 31. Each filter uses both channels and shares its weights across
frequency positions. The banks contain 1,260 trainable parameters and produce
36 channels. A 1×1 convolution mixes them into 48 channels, followed by dilated
residual blocks, group normalization and pooling. Mean, maximum and sample-SD
pooling give 144 inputs to a 64-unit correction head with dropout 0.1.

### Layer dimensions and the purpose of pooling

For one sweep, shapes below omit the batch dimension:

| Stage | Output | Role |
| --- | --- | --- |
| Three 2→12 Conv1d banks, widths 5/15/31, concatenated | 36×500 | Learn several local scales using zero padding |
| Width-1 convolution 36→48, GELU | 48×500 | Mix the banks at each position, then apply a smooth nonlinearity |
| Residual block, dilation 1 | 48×500 | Refine features with a skip connection |
| Average pool, kernel=stride=2 | 48×250 | Replace neighboring pairs with their means |
| Residual block, dilation 2 | 48×250 | Use taps spaced two feature positions apart |
| Average pool, kernel=stride=2 | 48×125 | Compress the position axis again |
| Residual block, dilation 4 | 48×125 | Use wider context on the reduced feature grid |
| Global mean, maximum, sample SD across 125 positions | 144 | Describe average, strongest and varying feature responses |
| Linear 144→64, GELU, dropout 0.1, linear 64→1 | 1 | Predict a correction in normalized target units |

At the exact 1.5287 kHz input spacing, the first filter windows span 6.1148,
21.4018 and 45.8610 kHz between their first and last samples, using `(k-1)*df`.
These are window extents, not prescribed physical linewidths. Each residual
block applies GroupNorm(6,48), GELU and a width-5 convolution twice, at its
stated dilation, then adds the input. Dilation expands context without increasing
the number of filter taps; the skip path supports refinement and gradient flow.
Group normalization operates per sample across each group's channels and positions,
so the complete block is not a strictly local filter. It has learned affine
coefficients and is distinct from saved input scaling. General sources are
[He et al.](https://arxiv.org/abs/1512.03385) for residual learning and
[Wu and He](https://arxiv.org/abs/1803.08494) for group normalization.

Local pooling applies `y[j]=(x[2j]+x[2j+1])/2` independently to each feature map.
It preserves the pair mean but loses the within-pair difference, reducing
the positions processed by later layers. This is internal feature compression,
not resampling or denoising the raw input file. It can suppress rapid variation
but also lose narrow structure; it does not guarantee alias suppression or
invariance to a one-bin shift. See the exact
[AvgPool1d operation](https://docs.pytorch.org/docs/2.11/generated/torch.nn.AvgPool1d.html).

Global pooling then supplies three different views of each final learned map:
mean activation, strongest activation, and sample SD with denominator 124.
These are not direct physical area/height/linewidth measurements. They lose
the positions of the activations, so useful local arrangements must be encoded
before this reduction. The resulting 144→64 linear layer has 9,280 parameters;
flattening 48×125 values into a 6,000→64 layer would require 384,064. This
arithmetic explains a capacity reduction, not a measured latency gain.

Pooling is optional architecture design. Less downsampling or learned strided
convolutions deserve comparisons on fixed validation groups, including narrow
features and branch imbalance. The exact widths, channel counts, dilation
schedule and pooling statistics are inspectable teaching settings; no saved
hyperparameter search establishes them as optimal. Dropout suppresses 10% of
correction-head activations during training with retained-value scaling; it is
disabled for evaluation and does not generate calibrated uncertainty intervals.

### Exact summary definitions

A second branch computes 25 deterministic summaries from standardized inputs.
Let r be the standardized second channel, v the standardized raw channel,
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

For this benchmark r contains standardized calibrated area contributions; in
the separate lineshape workflow it contains standardized raw-minus-reference
recorded units. The summary formulas are shared, while their meaning, fitted
scalers and weights depend on the input contract.

Ridge regression fits 25 weights and an unpenalized intercept against P divided
by the training-row population SD (floored at 1e-6). Its objective is summed
squared error plus `0.001*sum(weights**2)`. These 26 coefficients are frozen
while AdamW trains the remaining 82,365 parameters. The correction head begins
at zero. Final fractional P is
`target_scale*(ridge_estimate + convolutional_correction)`.
The fixed coordinate and summary scaling statistics are saved buffers, excluded
from the 82,391 parameter count.

### Why these summaries and how to choose them

The motivation is to give the estimator direct access to global amplitude,
shape and background descriptors. A small regression can fit relationships among
them before the convolutional branch learns a correction. This may help with
limited training data and supplies an inspectable reference estimate. It remains
a design hypothesis: the saved dense-versus-CNN comparison does not isolate
the benefit of the summary branch.

Signed zero-order information is particularly useful in area mode. If a_j is
the unstandardized calibrated contribution, and mu_2 and sigma_2 are the saved
second-channel mean and effective standardization divisor, then

```
r_j = (a_j - mu_2)/sigma_2
m_0 = mean(r)
P_area = sum(a_j) = 500 * (sigma_2*m_0 + mu_2).
```

Thus the first signed moment retains the conventional area estimate through a
known affine map, before rounding. In lineshape mode its counterpart retains
an uncalibrated sum of recorded-unit differences. No per-spectrum division by
area removes amplitude. The network's `calibrate()` method fits the ridge head;
that method name does not mean it performs TE or hardware calibration.

Signed odd moments weight the two sides with opposite signs; even powers
weight distance without reversing sign across the center. Absolute-amplitude
moments avoid cancellation between positive and negative amplitudes; their odd
powers can still have either sign and describe left–right imbalance. Neither
family is a direct branch-ratio measurement, variance or standardized skewness.
Extrema and SD complement integration, while raw-level statistics expose
background context correlated with nuisance conditions. Noise, detector phase,
offsets and broadening can all affect these features.

Centering describes shape relative to an absolute-amplitude centroid, avoiding
division by a potentially cancelling signed sum. It does not estimate the
resonance center independently; truncation, noise and input standardization
still affect it. For a center denominator above its floor, direct substitution
gives `mean(abs(r)*t)=0` up to floating-point error. The first absolute centered
moment is therefore redundant. A 25-entry vector is not 25 independent physical
measurements. Higher monomials can be correlated and sensitive to tails and
endpoints. This explanation preserves the existing feature vector and checkpoints.

The orders 0…10 and 0…6 are finite, inexpensive choices offering several views
of shape; neither the physics nor a saved search determines those cutoffs.
The signed family retains sign through more orders, while the shorter absolute
family supplies magnitude-based descriptors. That describes the design, not
proof of a unique optimum. Standardization puts their scales on comparable
footing; the ridge penalty discourages large weights for correlated features.
It does not make the features independent or coefficient magnitude a causal
importance measure. Freezing the ridge solution gives neural training a fixed
starting estimate; it does not prove joint training would be worse.

For a controlled selection study, hold validation arrays and source groups
fixed and compare area-only (where calibrated), ridge-only, CNN-only and the
combined estimator. Remove families, reduce orders, omit the redundant moment
or compare an orthogonal basis. Refit input/summary scaling and ridge weights
on training rows for every candidate; state parameter counts, updates and
runtime. Inspect signed-P/SNR/source bias, RMSE and tails across repeated seeds.
Choose on validation and evaluate the frozen choice once on the untouched test.
These ablations are proposed work, not completed results or new network scores.

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
