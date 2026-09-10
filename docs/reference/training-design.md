# Design the training population and measure how much data you need

Use this guide alongside [generation step 7](https://zetanaut.github.io/NMR-AI/walkthrough.html#generate)
and [training step 8](https://zetanaut.github.io/NMR-AI/walkthrough.html#train).
The 2,000-event exercise establishes a working pipeline. It is not a prescription
for the dataset size or sampling distribution needed by another experiment.

## Why small polarization can be harder

Here “low polarization” means small **absolute** polarization, |P|, on either
side of zero. In the single-site spin-temperature model, holding the electronics,
susceptibility scale, splitting, broadening, and noise fixed:

```text
Q(P) = 3 P² / (2 + sqrt(4 − 3 P²))
w_plus = (P + Q)/2; w_minus = (P − Q)/2
chi = A [P/2 (K_plus + K_minus) + Q/2 (K_plus − K_minus)]
```

For small |P|, Q ≈ 3P²/4. The leading response scales with P, while the
branch-imbalance term scales with P². For nonzero P, the signed fractional
branch imbalance is `(w_plus − w_minus)/(w_plus + w_minus) = Q/P ≈ 3P/4`.
Its magnitude is about 0.75% at |P| = 1% and 3.75% at |P| = 5%.
These are derived intrinsic branch-weight differences, not differences between
observed peak heights. The full complex susceptibility still passes through
the circuit and phase-sensitive detector.

Thus the signal becomes weaker against a fixed noise level and its branch
weights become more nearly equal. The model's splitting and linewidth do not
automatically change with P: reduced visibility is different from a change in
physical peak separation. If amplitude A is also unknown, the leading product
A·P creates a further ambiguity; the small shape asymmetry carries information
that helps distinguish amplitude from polarization. This matters particularly
for the lineshape exercise without TE calibration. The TE-area exercise has
additional calibration information and a different uncertainty problem.

This is an expansion of the [documented spin-1 model](physics-electronics-theory.md#4-complex-spin-1-powder-response),
not an experimental error law. At P = 0 the nuclear susceptibility vanishes.
More examples can improve the network's approximation and coverage, but cannot
make noisy, indistinguishable inputs uniquely informative. Evaluate absolute
errors near zero; relative error is undefined at zero. Detector polarity alone
does not determine the polarization sign without the readout convention.

Define the lowest nonzero |P| of interest before model selection, and make its
local validation error an explicit criterion. At fixed absolute RMSE, relative
RMS scales as 1/|P|; the actual model's absolute error may also change with P.
The [polarization-error demo](https://zetanaut.github.io/NMR-AI/index.html#polarization-performance)
shows both effects and band counts on saved predictions. A low-end criterion
must accompany checks throughout the operating range: higher polarization does
not guarantee smaller errors for every model, calibration or noise condition.

## Decide where the training examples should go

Start with broad coverage, then inspect validation errors in **signed** P bands
and in noise, linewidth, and setup strata. Uniform P does not imply uniform
difficulty. Conversely, a sparsely populated band is not automatically the
hardest one. Allocate extra training effort where evidence and the scientific
error target justify it. Continuous-target imbalance is a regression problem:
nearby P values share structure, so treating every bin as an unrelated class
can be misleading. [Yang et al. (2021)](https://proceedings.mlr.press/v139/yang21m.html)
study this distinction; their benchmarks do not establish an NMR sampling rule.

A simple **illustrative** training density is

```text
q(P) = (1 − alpha) Uniform(−0.6, 0.6)
       + alpha Uniform(−b, b)
```

The broad component retains both signs and the full teaching range. The narrow
component gives additional draws near zero. With alpha = 0.5 and b = 0.05,
the probability of |P| ≤ 0.05 is `0.5 + 0.5*(0.1/1.2) = 13/24`.
Among 1,200 training rows, that is **650 expected low-|P| examples**, compared
with 100 under uniform sampling. The remaining region receives 550 instead of
1,100. At a fixed total size, emphasis has a coverage cost. These are expected
counts, not observed data or an optimized allocation. Use the
[interactive allocation example](https://zetanaut.github.io/NMR-AI/index.html#sampling-lab)
to explore that tradeoff.

This standalone code illustrates the label draw only:

```python
import numpy as np

rng = np.random.default_rng(42)
n_train, alpha, b = 1200, 0.5, 0.05
P = rng.uniform(-0.6, 0.6, n_train)
focus = rng.random(n_train) < alpha
P[focus] = rng.uniform(-b, b, focus.sum())
print("Low-|P| draws:", np.count_nonzero(np.abs(P) <= b))
print("Expected:", n_train * (alpha + (1-alpha)*b/0.6))
```

The first number fluctuates around the expectation. This is not a training NPZ
and is not wired into the current generator. In a generator extension, each
drawn P must be used in a fresh full-circuit response, with appropriate event
noise and configuration-shared reference noise. Never replace labels on an
existing spectrum, multiply an existing raw sweep to “change P,” or normalize
every signal to the same height: those operations do not reproduce this model.

## Keep sampling, loss weighting, and evaluation separate

| Choice | What it changes | What to watch |
| --- | --- | --- |
| Generate new examples from q(P) | Coverage and the distribution of physical responses seen in training | Preserve broad support, supported nuisance relationships, and source grouping |
| Oversample existing training rows | How often a row contributes to optimizer updates | Repeats do not add new noise realizations, configurations, or measured sources |
| Weight the loss | The penalty assigned to errors in selected conditions | Weights express a target; they do not supply missing examples |

PyTorch's [WeightedRandomSampler](https://docs.pytorch.org/docs/stable/data#torch.utils.data.WeightedRandomSampler)
can draw training-row indices with specified probabilities and replacement.
It is an option for a trainer extension, not a feature enabled in this tutorial.
If used, build its weights from the training subset only. Combining oversampling
and extra loss weights compounds their emphasis; do that only deliberately.

For loss weight a(P), uncorrected sampling approximately minimizes
`E_q[a(P)*(h(X) − P)²]`. If the intended target is another known density p(P),
weights proportional to `p(P)/q(P)` recover that target expectation when q is
positive everywhere p is positive and the conditional simulator inputs given P
are the same. If nuisance
distributions also change, a P-only correction is insufficient. Large weights
make estimates unstable. A non-uniform design can instead be intentional: state
the desired low-|P| emphasis and report its cost elsewhere.

This expectation identity follows directly by substituting q(P) into the
integral. It is not evidence that the experimental P population is known.
For squared loss, the ideal predictor is the conditional mean under its
training objective. Changing the P population can therefore change predictions
for ambiguous weak spectra, even with unchanged forward physics. Check signed
bias as well as RMSE; more predictions near zero are not by themselves an
improvement in recovery.

Choose the objective before testing. Report both an evaluation matching the
intended operating population (when known) and a diagnostic evaluation with
enough rows in each important P/SNR/setup band. Keep both fixed while comparing
training designs. A pooled score under an oversampled test mixture answers a
different question from a score under the operating population. If a weighted
pooled score is reported, give the weights and the unweighted per-band scores.

Here simulation SNR means `max(abs(clean_signal))/max(abs(noise))`, excluding
the baseline. This differs from a peak-to-noise-standard-deviation ratio.
Preserve that definition and the raw/reference noise dependence. Experimental
P and signal estimates have their own uncertainty; do not label them simulator
truth for stratification or error measurement.

## What the current commands support

Both generators currently draw P uniformly between `--p-min` and `--p-max`.
`--num-samples` controls event count; `--num-configurations` controls simulated
configurations. For example, after walkthrough step 4:

```bash
python tools/generate_matched_data.py \
  --matching-report local-results/walkthrough/single-site/matching_report.json \
  --num-samples 4000 --num-configurations 200 \
  --p-min -0.1 --p-max 0.1 --seed 43 \
  --output local-results/low-p-study/data.npz
```

This makes a **separate uniform −10% to +10% study**, not the mixture above.
Inspect its saved P histogram and configuration/source counts. It is not a
replacement for a broad-range model. Changing the seed is also not a way to
preserve an existing comparison's validation rows.

The current trainer shuffles training rows, uses unweighted absolute MSE after
constant target scaling, and selects checkpoints by pooled validation MSE.
There is no mixture, weighted-loss, balanced-batch, or fixed-P option hidden in
the command examples. A study of those choices needs an explicit extension,
new data and retraining as appropriate, and its own saved results. Do not merge
NPZs naively: configuration IDs can collide and source descendants still belong
to one partition. The five measured sources remain five sources at any event
count.

The multiscale model also fits 25 ridge weights and an intercept before neural
training. A new objective must specify whether and how it applies to that fit,
feature scaling, and checkpoint selection. Changing only the minibatch sampler
does not change the ridge fit. Preserve all preprocessing and grouping metadata.

## How many examples for this many parameters?

There is **no dependable fixed ratio of examples to neural-network weights**.
A rule such as “ten spectra per weight” ignores the function being learned,
shared convolutional weights, regularization, noise, and dependence between
examples. Networks can fit datasets with more weights than labeled examples;
being able to fit or memorize them does not establish generalization.
[Zhang et al. (2017)](https://arxiv.org/abs/1611.03530) demonstrate why training
fit alone is insufficient. Dataset size for this task must be measured against
its held-out conditions and error target.

| Count | What it means in this repository |
| --- | --- |
| Spectral samples | 500 frequency bins per sweep; these are input features, not 500 independent labeled examples |
| Physical fit parameters | The single-site raw fit has 12 unknowns per spectrum; these describe the measurement, not the network's capacity |
| Network parameters | MLP: 64,129; dense DNN: 297,473; compact CNN: 3,889; multiscale: 82,391 total |
| Neural optimizer parameters | Multiscale: 82,365; its other 26 coefficients were fitted on training rows and then frozen |
| Hyperparameters | Architecture, learning rate, batch size, regularization, and stopping rules chosen using development data |
| Independent groups | Circuit configurations for the TE-area exercise; measured sources for the anchored exercise |

Counts come from [the implementation](../../tools/nmr_lab.py) and
[saved training records](model-comparison.md). An increasing number of physical
nuisance dimensions generally needs wider **joint** coverage. A Cartesian grid
with k values in each of d coordinates has k^d combinations: use this arithmetic
to see why sampling every bound combination quickly becomes impractical, not to
claim k^d examples are required. Sample physically supported combinations and
retain their correlations.

## Measure dataset size with a controlled learning curve

A learning curve here means error versus **training-set size**, distinct from
loss versus epoch in `history.csv`. Plotting train and validation performance
against sample count helps diagnose the benefit of additional data.
[The scikit-learn learning-curve guide](https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve)
explains that distinction; no scikit-learn dependency is needed for this project.

1. Declare signed P bands, the SNR convention, error tolerances, and the
   validation population. Reserve independent groups for the final test.
2. Make nested training subsets of a fixed training pool, for example 1,200,
   2,400, 4,800, and 9,600 rows when available. These are an illustrative doubling
   schedule, not minimum requirements. Hold validation arrays and groups fixed.
3. Separate two studies: add fresh P/noise events within existing training
   configurations; then add independent training configurations or acquisitions.
   More simulated variations of three training seeds do not add measured sources.
4. Refit scalers, target scaling, ridge coefficients and neural weights using
   only the selected training rows at each size. Never reuse a scaler fitted on
   a larger pool. Preserve group assignments and appropriate reference sharing.
5. Compare the same architectures and training settings with enough updates to
   assess convergence. Record both updates and elapsed training time: an epoch
   at twice the dataset size entails roughly twice as many optimizer updates at
   the same batch size. Repeat initialization and sampling seeds to expose variation.
6. Plot pooled and per-band validation bias, RMSE, and tails with row and group
   counts. Use independent-group resampling for uncertainty when enough groups
   exist; one held-out measured source cannot estimate between-source variability.
7. Select the smallest tested design meeting the declared validation criteria
   with adequate coverage, or continue investigating if none does. Evaluate the
   frozen choice once on the untouched final test.

The current CLI recomputes partitions when loading a dataset. Merely invoking
it on several regenerated NPZs does not implement the fixed-validation study
above. Use an explicit experiment driver that freezes holdout arrays and group
membership; hashing and storing that split is part of the study.

Doubling dataset size and diagnosing the training/validation gap are practical
methods discussed in [Goodfellow, Bengio and Courville, chapter 11](https://www.deeplearningbook.org/contents/guidelines.html).
Our grouping and error-stratification requirements adapt that methodology to
this simulator. A plateau alone does not identify an information limit: model
capacity, optimization, coverage and measurement information can each matter.

## Read the training process and choose the next experiment

The walkthrough uses batch size 128 by default. With 1,200 training rows and
no dropped final batch, an epoch has `ceil(1200/128) = 10` optimizer updates.
Each update predicts a batch, computes scaled MSE, clears old gradients,
backpropagates, clips the gradient norm at 5, and applies AdamW. Validation uses
evaluation mode without gradient updates. The ridge coefficients stay frozen.

The learning-rate scheduler halves the rate after its validation plateau
criterion is met. `--patience 12` is a separate early-stopping setting: it stops
after 12 successive epochs without a new best validation loss. `--epochs 80`
is a cap, not a required stopping point or a number of distinct datasets.
The trainer restores the best validation weights. These are current
[code behaviors](../../tools/train_model.py), not universal preferred settings.

| Validation finding | Useful next comparison |
| --- | --- |
| Low training error, worse validation error | Add supported independent training coverage; compare stronger regularization or a smaller network |
| Both errors remain high | Check units, targets and preprocessing; inspect optimization, model capacity, and whether the inputs contain the needed information |
| Pooled error improves but low-\|P\| error does not | Check counts, signed bias, noise and amplitude ambiguity; compare targeted coverage under the same validation distribution |
| Low-\|P\| error improves but larger-\|P\| error worsens | Increase total coverage or reduce the emphasis; quantify the tradeoff against the declared objective |
| Different sources or seeds give different conclusions | Add independent acquisition evidence and report the variation |
| More rows on the same sources stop helping | Test new supported configurations, model adequacy, and measurement information before increasing event count again |

Training loss can exceed validation loss when dropout is active only during
training. Compare on the same units and evaluation convention. Because each
run fits its own target scale, normalized losses from differently sampled
datasets are not directly comparable; use fractional-P or percentage-point
metrics. For the current unweighted loss,
`RMSE_fractional_P = sqrt(val_loss) * target_scale`.

Keep a compact experiment table: sampling density and seed, rows and independent
groups, P/SNR coverage, architecture and parameter counts, preprocessing, batch
size, learning rate, optimizer updates, selected epoch, and conditional validation
errors. Freeze the selected recipe before final testing. This guide proposes
comparisons; the published runs retain their original uniform sampling and scores.
