# Matching uncertainty, distribution shift and robustness

Added 2026-09-08 for the [tutorial section](https://zetanaut.github.io/NMR-AI/index.html#distribution-shift),
with a same-day revision emphasizing unfamiliar experimental structure at inference.
This is a proposed measurement and validation protocol. It does not report
new experimental covariance estimates, revise either generator, or establish
new network accuracy. The [physics](physics-electronics-theory.md),
[baseline](baseline-fitting.md), [matching](experimental-matching.md) and
[material](material-examples.md) records retain their measurement contracts.

## Start with the incoming experimental spectrum

An experimental input can contain a shoulder, ripple, reference mismatch or
noise correlation poorly represented or absent in training. Even with the
expected frequency grid and input channels, passing format checks does not
establish that the learned estimator handles this structure accurately. This
is the practical motivation for comparing inference inputs with training coverage.
Detection does not have to wait for an independently known P or an established
physical explanation of the new pattern.

As an illustrative scenario, consider a baseline ripple appearing in incoming
spectra after training on examples without that ripple. The saved preprocessing
can pass its effects to both the multiscale CNN's convolutional features and
the 25 physical summaries. The fixed estimator may respond with a changed P
estimate. Weight sharing and a frozen summary regression do not themselves
establish tolerance to that structure. This is a proposed failure mechanism
to test, not a diagnosed ripple or measured prediction error in the supplied data.

The operational sequence is to identify the unfamiliar structure, compare it
with training coverage, and evaluate its effect on polarization. An unusual
individual input can motivate an anomaly flag. Recurring differences across
independent acquisitions support a distribution comparison; chronological
changes motivate drift monitoring. One unusual sweep does not establish a
population-level shift. Input novelty and prediction error require separate
evidence, and either can occur without a large value of the other.

## What the three deliverables mean

Comparing noise, baseline and signal over SNR and polarization is a useful
starting point. Three deliverable packages make it
operational, with different mathematical objects in each:

| Package | Save | Question it answers |
| --- | --- | --- |
| Measurement and fit uncertainty | Noise/reference covariance; full joint parameter covariance or refit samples; joint baseline/signal curve covariance | How uncertain is the measurement and its fitted decomposition, conditional on the assumed model? |
| Matching and distribution discrepancy | Mean residual, residual covariance and second moment; input-distribution comparisons, support checks and their sampling uncertainty | What remains unexplained, and what differs between experimental inputs and training inputs? |
| Robustness and temporal monitoring | Conditional prediction-error tables under specified disturbances; chronological input diagnostics and later labeled checks | Which tested deviations can a fixed estimator tolerate, and are acquisition conditions changing? |

These records connect the new structure to a measured consequence: fit residuals
describe unexplained patterns, input comparisons describe coverage, and known-P
robustness tests quantify error. Systematic means and covariances contribute to
these records; neither alone provides the entire assessment.

### Terminology supports the investigation

New or underrepresented experimental structure can be a case of covariate shift.
The formal assumption is a change in the input distribution with the same
conditional relationship between P and the complete input X:

```
p_exp(X) != p_train(X), but p_exp(P | X) = p_train(P | X).
```

See [Sugiyama, Krauledat and Müller (2007)](https://www.jmlr.org/papers/v8/sugiyama07a.html).
Even under this assumption, a finite trained model can perform poorly in regions
of input space that it has learned inadequately. An unchanged conditional
relationship does not make its learned approximation correct everywhere.

For an inverse problem, changes in detector response, material physics or the
polarization population can also change P given X. The observed structure alone
does not establish which case applies. **Distribution shift** covers both;
**temporal drift** describes change over time. These distinctions guide methods
such as importance weighting; they are not prerequisites for detecting unfamiliar
inputs or testing prediction sensitivity. Holding true P fixed in a disturbance
experiment does not by itself prove equality of the two conditional distributions.

Where experimental inputs occupy a region absent from training, the training
sample provides no direct evidence about that region. Reweighting existing rows
cannot supply the missing structure. Additional characterization, supported
generation or new training acquisitions may be needed.

There is no universal “covariate shift matrix” or “covariate drift matrix” that
also measures network accuracy. A named residual second-moment matrix and a
table of measured robustness results are useful, but their axes, conditioning,
units and interpretation must be defined. Temporal drift is a change with time;
robustness is an estimator's performance under a declared change.

## 1. Establish which components can be identified

For a frequency-vector observation, write the diagnostic decomposition as

```
y = F(theta, P) + d + epsilon
F = B(theta) + S(theta, P)
B = F(chi=0); S = F(chi(P)) - F(chi=0)
r = y - F(theta_hat, P_hat).
```

This notation preserves the existing full-circuit definition of B and S. The
additive d denotes unexplained output discrepancy; it is not a new physical
voltage background or a generator component that has been calibrated by naming it.
Parameter uncertainty and model discrepancy are distinct, as emphasized by
[Kennedy and O'Hagan (2001)](https://doi.org/10.1111/1467-9868.00294).

**Noise:** collect independent repeats while the underlying signal and apparatus
remain stable. For equal-covariance, independent repeats, `(y1-y2)/sqrt(2)`
has the single-sweep noise covariance. Examine repeat means, correlations,
spectral density, tails and dependence on averaging. Shared coherent pickup
can cancel from differences and still contaminate measurements. Evolving P,
baseline or electronics also invalidates the pure-noise interpretation.

**Baseline:** use independently recorded signal-off references under the
corresponding operating conditions, with reference-to-signal drift characterized.
Neither a different baseline acquisition nor a chi=0 curve reconstructed from
the same fitted polarized sweep is an independent baseline truth measurement.

**Signal:** fit the physical signal and baseline jointly, retaining their
correlation. Reference-subtracted data still contain reference noise and any
reference mismatch. From one raw sweep, `y-B_hat-S_hat` is just the original
full-fit residual; it does not uniquely separate baseline, signal and noise errors.
Unknown component contributions must remain explicitly unidentified.

For raw and reference noise with cross-covariance C_y,ref, subtraction gives

```
C_(y-ref) = C_y + C_ref - C_y,ref - C_y,ref.T.
```

Keep the same noisy reference shared across simulated events when that matches
the acquisition. Its contribution does not average away as though each event
had a new independent reference.

## 2. Joint fit uncertainty, including baseline–signal correlations

Let alpha contain every identifiable fitted coordinate, including P, electronics,
and readout. A q-by-q parameter covariance is not a frequency-by-frequency
covariance. Under a locally linear, correct mean model, known observation
covariance Sigma, full column rank and an interior solution, generalized least
squares gives the local approximation

```
C_alpha ~= inverse(J.T @ inverse(Sigma) @ J),
J = derivative of the complete fitted observation with respect to alpha.
```

These conditions matter. Active bounds, competing minima, weak identifiability
and a fitted discrepancy model require more than an inverse curvature matrix.
Profiled linear readout parameters must be restored in the joint uncertainty;
the existing profiled Jacobian condition numbers are not that covariance.
Transform log/scaled coordinates to physical units before reporting parameter
uncertainties. Parameter-covariance entries carry products of parameter units.
Never use structured full-fit RMS as a measured white-noise scale.

Given suitable C_alpha, define J_B and J_S on the exact acquisition grid:

```
C_BB = J_B @ C_alpha @ J_B.T
C_SS = J_S @ C_alpha @ J_S.T
C_BS = J_B @ C_alpha @ J_S.T
C_fit = C_BB + C_SS + C_BS + C_BS.T.
```

The stacked covariance of `[B_hat, S_hat]` has these four blocks. Dropping
the cross blocks can misstate the uncertainty because both curves share fitted
parameters. C_fit describes fitted mean curves; predicting another noisy
observation also requires its observation uncertainty, with dependencies retained.
This follows the correlated-input propagation rule in
[JCGM 100, section 5](https://www.iso.org/sites/JCGM/GUM/JCGM100/C045315e-html/C045315e_FILES/MAIN_C045315e/05_e.html).

For nonlinear fits, repeat the full fitting procedure on independently justified
noise/reference realizations and specified discrepancy scenarios. Retain failures,
boundary solutions and multiple branches. Distribution propagation is described
in [JCGM 101](https://www.bipm.org/en/doi/10.59161/jcgm101-2008).
Such uncertainty is conditional on the supplied noise/discrepancy assumptions;
the simulation does not establish their experimental validity.

Fit uncertainty within one setup and the population spread of actual setups
are different. The observed spread of fitted parameters combines both. Do not
interpret each covariance ellipse, optimizer-start spread or numerical search
bound as a measured distribution from which all training configurations should
be sampled independently.

## 3. Retain the mean discrepancy as well as covariance

For N residual vectors on the same grid in a declared condition group z, save

```
mu_r(z) = sum_i r_i / N
C_r(z) = sum_i (r_i-mu_r)(r_i-mu_r).T / (N-1)
M_r(z) = sum_i r_i r_i.T / N
       = ((N-1)/N) C_r(z) + mu_r(z) mu_r(z).T.
```

Here C_r needs N >= 2; M_r is an empirical second moment, not a centered
covariance. A repeated identical residual shape has C_r = 0 but nonzero mu_r
and M_r. Thus covariance alone can miss a reproducible systematic error.
Curve means are in recorded units for matching; these matrices are in
recorded-units squared. The separate synthetic TE-area benchmark uses volts
and V² instead.

Residuals from refitting the same data are reduced by the fitting operation.
For the local linear model, with

```
H = J @ inverse(J.T @ inverse(Sigma) @ J) @ J.T @ inverse(Sigma),
Cov(r) = (I-H) @ Sigma @ (I-H).T
```

when discrepancy is absent. This is a direct linear-algebra derivation;
post-fit residual covariance is not the original observation covariance.
Discrepancy components that resemble changes in fitted parameters can be
absorbed into those parameters and leave a small residual, including a biased P.

Report in-sample diagnostics as such. Use independent validation acquisitions
for checking predictive adequacy. When a validation spectrum necessarily has
its own nuisance fit, apply the same complete fitting procedure to simulated
validation spectra and compare the resulting diagnostics. Fitting that spectrum
still does not make its P estimate independent truth.

In general C_r contains noise, fitted-parameter effects and discrepancy. Their
cross-covariances can matter. Subtracting estimated noise and fit matrices from
C_r is not automatically an identifiable or positive-semidefinite estimate of
discrepancy covariance. Fixing negative eigenvalues would add a modeling choice,
not recover missing information.

With N independent spectra, empirical centered covariance has rank at most
N-1. Even if the five butanol spectra could be pooled meaningfully, that rank
would be at most four; they are not repeats at one P and setup. Hundreds of
frequency bins are not hundreds of independent spectra. A low-rank, banded or
shrinkage estimate needs justified structure and validation, especially within
SNR/P strata. Save the sample count, rank, regularization choice and uncertainty.

## 4. Compare the actual input distributions

Compare experimental and generated raw/reference inputs using the same available
channels, exact grid, units and frozen training preprocessing. Also compare
interpretable summaries: baseline shape, line position/width, branch structure,
noise correlation and reference stability. Feature maps and thresholds must be
chosen on development data. Frequency bins are features; acquisitions are samples.

Inspect raw spectra as well as any compressed feature representation. For the
existing multiscale estimator, its 25 summaries and learned features provide
additional views of a compatible input; a compressed representation can hide a
new pattern. A distance or anomaly score fitted to a training-derived reference
can assess an individual input, with thresholds and false-alarm behavior checked
on separate development acquisitions. That individual score is different from
a two-sample test on batches or chronological windows.

Save a flagged input's raw/reference arrays, timestamp, acquisition information,
score and model/preprocessing version. Choose the response during validation,
such as routing unsupported conditions for review. A flag is evidence of a
coverage concern, not a calibrated P error bar; the absence of a flag is not an
accuracy guarantee. This record proposes these checks; the current predictor
does not implement such a novelty flag or response policy.

Save differences in feature means and covariances, marginal distributions,
tails, joint coverage and support gaps. Matching first and second moments does
not establish distributional equality. A predeclared two-sample diagnostic such
as [maximum mean discrepancy](https://www.jmlr.org/papers/v13/gretton12a.html)
can add evidence. Use independent acquisition groups for sampling uncertainty
and group/block resampling appropriate to the data. A domain classifier, if used,
also needs held-out groups, balanced evaluation and uncertainty on its score.
Near-chance classification or a nonsignificant test is not proof of agreement.

A covariance difference `C_exp-C_train` remains useful for comparing feature
variation: directions can gain or lose variance, so this difference may be
indefinite. To generate an added physical disturbance, specify that process's
mean, valid covariance and dependencies separately; the matrix difference is
not automatically its additive-noise covariance. A large input difference can also reflect a
different P population rather than faulty matching; compare both the overall
population and appropriately conditioned groups. Neither input distances nor
matching residual RMS directly measure polarization error.

Use z = (signed P band, stated SNR definition, material, setup, acquisition/time
group), retaining additional measured covariates where relevant. SNR and P alone
need not explain the variation. In simulations P is newly sampled truth. On
experimental data, a fitted P or signal amplitude is an estimate: label it and
propagate its uncertainty into grouping where possible. Near P=0 retain absolute
errors; relative errors are undefined there.

The paper-compatible simulation definition remains
`SNR_peak = max(abs(S))/max(abs(epsilon))`, excluding the baseline.
An experimental diagnostic `max(abs(S_hat))/sigma_noise_hat` is a different
peak-to-noise-SD proxy and must have a different name. Estimate its denominator
from appropriate repeat/noise evidence, not structured full-fit RMS. Match the
definition and averaging convention before comparing SNR strata.

## 5. Measure robustness and monitor temporal drift

For the illustrative ripple case, construct nominal and disturbed inputs at
the same simulator-known P, with a stated output disturbance supported by the
experimental evidence. Vary its amplitude and pattern; preserve justified
raw/reference sharing and dependencies. Run both inputs through the same frozen
preprocessing and network. Retain the nominal error, disturbed error and change
in prediction, with error sign `P_true - P_pred` throughout. This can distinguish
an input difference that the estimator tolerates from one that biases it.

Use the full forward response for changes to electronics or material physics.
An arbitrary nuclear peak added to a trace does not automatically preserve its
label. On an experimental trace with unknown true P, changing the trace under a
proposed disturbance and observing the output measures sensitivity conditional
on that scenario. It is not a measured accuracy result. Accuracy requires
appropriate known-P tests or independent polarization evidence.

Freeze the trained estimator h and its preprocessing T. Generate controlled
known-P stress cases that change specified baseline/electronic settings, noise
amplitude/correlation, reference-to-signal drift and physically supported signal
shapes. Include coupled changes and conditions outside the training coverage.
Preserve truth P and use the full circuit when changing its physical parameters;
arbitrary branch distortions can inadvertently change the label itself.

Define a robustness table R(z, mode, amplitude), with cells containing

```
N events and independent groups
bias = mean(P_true - P_pred)
width = population SD(P_true - P_pred)
RMSE, 95th percentile of absolute error, fraction beyond a declared tolerance
sampling uncertainty; interval coverage if the estimator supplies intervals.
```

Record the disturbance definition, train/test coverage boundary, seeds, data/model
hashes and untouched evaluation split. This is a conditional risk table, not a
covariance matrix. A finite set of stress tests establishes performance only
for its specified changes. Broader evidence that uncertainty calibration can
fail under shift comes from [Ovadia et al. (2019)](https://papers.neurips.cc/paper_files/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html);
their classification benchmark supplies no NMR accuracy guarantee.

Connect input mismatch to P explicitly. Let w include the available raw/reference
inputs and any required calibration, and let g be the gradient of `h(T(w))`
with respect to w. For a small additive perturbation delta_w at fixed true P,

```
delta_e ~= -g.T @ delta_w
mean(delta_e) ~= -g.T @ mu_delta
Var(delta_e) ~= g.T @ C_delta @ g.
```

These are local approximations; use full propagation for large or nonlinear
changes. If baseline prediction error is e0 and the change is delta_e,
`E[(e0+delta_e)^2] = E[e0^2] + 2 E[e0 delta_e] + E[delta_e^2]`.
Consequently one cannot simply add a matching-error width in quadrature to a
test RMSE, especially if its noise/reference contribution is already included.

Monitor temporal drift separately by comparing chronological acquisition
windows with the validated reference distribution. Choose alarm thresholds and
false-alarm expectations using stable development periods; check actual P error
when independent labels become available. The raw butanol file order is not
strictly chronological. Five development spectra do not establish a time-drift
process, and unlabeled temporal changes cannot by themselves identify error in P.

## 6. Mitigate, then evaluate again on untouched data

Correct supported mean-model errors and acquisition problems first. Extend
generation with physically supported joint nuisance variation, appropriate
reference sharing, and characterized noise/discrepancy scenarios. Extra coverage
can improve robustness but can also reduce precision or create ambiguities; it
does not compensate for missing physics or distinguish inputs that contain too
little information. Compare both nominal and shifted performance after changing
the training distribution. An importance-weighting strategy additionally needs
supported covariate-shift assumptions and adequate overlap of the input domains.

Partition by independent acquisitions before choosing models, discrepancy modes,
feature maps, covariance regularization or augmentation ranges. Keep all
`source_scan_1based` descendants together in the 500-bin workflow. Retain a final
independent acquisition/time holdout after mitigation; do not reuse the same
discrepancies for designing augmentation and claiming final performance.

If a full generator changes, regenerate data and retrain before changing results.
Where measurements have no independently established P, report matching and
input-distribution evidence without calling fit/network agreement an accuracy
test. Spin-1 lineshape inference still does not require TE calibration.

The existing five butanol sweeps and separate UVA-ND3 excerpt do not identify
the conditional matrices and robustness tables proposed here. The 500-bin
training and prediction tools are implemented and have a saved source-holdout
synthetic evaluation. Preserve each acquisition's grid, input units and hardware
assumptions when applying any checkpoint.

## Verification — 2026-09-08

Constructed linear-algebra examples independently checked the sample second-moment
identity, invariance of covariance to a common offset, the rank-four limit for
five centered vectors, joint baseline/signal propagation, correlated-reference
subtraction, post-fit noise projection, local polarization-error sign/variance,
and the MSE cross term. Direct whitened least-squares refits were used to check
the projection formula. The largest identity discrepancy was 5.33e-15 in the
arbitrary units of these examples. A separate example confirmed that a difference
of positive-definite covariances can have negative eigenvalues. These checks
establish the stated arithmetic, not experimental covariance or instrument validity.

The tutorial preserves these mathematical checks as distinct from experimental
validation. The current runnable learning exercises use 500-bin inputs.
