# Experimental spin-1 matching and an anchored generator

This reference accompanies [Practical 01B](https://zetanaut.github.io/NMR-AI/matching.html).
Read the [circuit and lineshape conventions](physics-electronics-theory.md)
and [baseline practical](baseline-fitting.md) first. The reference is
[Seay, Fernando and Keller, arXiv:2603.10146](https://arxiv.org/pdf/2603.10146):
Fig. 1 and Eqs. (1)–(14) for electronics; Eqs. (15)–(26) for the spin-1 model.
The [learning example](https://zetanaut.github.io/NMR-AI/matching.html#lineshape-network) includes generation,
training, saved preprocessing and prediction on this exact 500-bin acquisition.

**Example 1 uses a single-site starting model** for the five butanol sweeps. [Example 2](https://zetanaut.github.io/NMR-AI/butanol.html) applies
the supplied C–D/O–D theory to those exact samples; [Example 3](../uva-nd3.html)
uses raw phase from a separate UVA-ND3 acquisition and jointly fits its circuit
baseline and ND3 spin-1 response.
The [material-model record](material-examples.md) documents both additions.
The generator below uses that single-site model. The material comparisons
do not establish material-specific network accuracy.

## Experimental data and confirmed setup

`examples/Sample_RawSignal.csv` is explicitly authorized for GitHub publication.
It is byte-for-byte identical to the supplied file, including UTF-8 separators:
SHA-256 `cdbb7e3afa6531694a4b97848d295bbb5c7c03ef62d796b053e3b4e16fbaea5a`.
There are five records of timestamp plus 500 amplitudes. The audited reader
`tools/experimental_data.py` recognizes five U+2028 line separators and 18 blank
logical lines. Record logical line numbers are 1, 6, 11, 16 and 22. No numeric
repairs are needed, no records are duplicates, and timestamps are not strictly
increasing. Preserve input order, every sample, and the original file.

Experimental set up details confirm the same acquisition mapping and cable
setup as the earlier baseline:

```
f_j = 32.3 + 0.0015287*j MHz, j=0..499
n = 1; cable half-wave = 3.580 m at nominal 32.7 MHz
```

The raw backgrounds have different offsets and shapes from the earlier
baseline. The first practical shows why a direct subtraction or an unconstrained
cable-length fit is not enough. This practical fits the full raw spectra.

## Polarization comes from the lineshape, not TE calibration

No externally supplied polarization or TE area calibration is required for
this spin-1 lineshape fit. Under the single-site spin-temperature hypothesis:

```
Q(P) = 3P^2 / [2 + sqrt(4 - 3P^2)]
w_plus = (P+Q)/2; w_minus = (P-Q)/2
r = w_plus/w_minus
P = (r^2 - 1)/(r^2 + r + 1)
```

The fitted P is determined by both intrinsic branch shapes and their integrated
intensity ratio. Overall susceptibility amplitude is a fitted nuisance.
The intrinsic absorption ratio must not be replaced by the ratio of observed
detector peak heights after dispersion mixing and circuit transformation.
At P=0 the signal vanishes; the ratio formula has a limiting value but is not
an independently observable ratio of two nonzero signals there.

This removes the need for TE normalization in the matching/generator workflow.
It does not make a fitted estimate a ground-truth accuracy test or assign it a
statistical error bar. Tensor-enhanced populations, multiple sites, contaminants
or an inappropriate material model require corresponding physics extensions.

## Electronics and unknowns

`configs/experimental-matching.json` is the explicit fitting contract.
The cable uses the established propagation scale and n=1 nominal length
3.580 m. It is not allowed to migrate to a different half-wave branch.
Coil R/L, resistances, RF drive and filling factor retain the documented nominal
seeds. G'=0 and C_stray=0 are explicit first-model assumptions.
The actual series tuning capacitance is fitted, initialized near 50 pF; it is
not fixed from the old fit, a knob position, or an assumed resonance equation.

Crucially, this practical allows the frequency-dependent detector phase in
paper Eqs. (10)–(14): phi0 + phi1*df + phi2*df^2. This is a phase rotation of
complex node voltage, not a polynomial voltage baseline. The stored reference
circuit's capacitor value is only an initialization field; the fitted log-C
coordinate overrides it at every model evaluation and in reconstruction.

Nine nonlinear coordinates are fitted per scan:

| Coordinate | Numerical search interval | Physical meaning |
| --- | --- | --- |
| center_mhz | 32.64–32.71 MHz | Lineshape center |
| split_mhz | 0.045–0.085 MHz | 3 omega_Q/(2 pi); reduced-frequency scale |
| g | 0.004–0.20 | Lorentzian HWHM / splitting |
| eta | 0–0.25 | EFG asymmetry, not coil filling factor |
| P_model | −0.95–0.95 | Fractional vector polarization from shape |
| log10_signal_scale_cgs | −8 to −1 | Effective susceptibility amplitude |
| log10_tune_capacitance_pf | log10(10)–log10(2000) | Effective series capacitor, pF |
| phase_slope_rad_per_mhz | −1–1 rad/MHz | Slow detector-phase slope |
| phase_curvature_rad_per_mhz2 | −3–3 rad/MHz² | Slow detector-phase curvature |

The linear coefficients a,b,d are profiled at every nonlinear step:

```
u_rot = u * exp(i*(phi1*df + phi2*df^2))
y_fit = a*Re(u_rot) + b*Im(u_rot) + d
A = hypot(a,b); phi0 = atan2(-b,a)
```

They count as three additional unknowns: twelve fitted degrees of freedom,
not nine or fifteen. RF drive is held nominal; independent drive and gain
cannot be determined from their product. Filling factor and susceptibility
amplitude are not fitted as separately identifiable coupling measurements.
All numerical bounds are declared search choices, not measured uncertainties.

## Matching steps and observed results

```bash
python tools/match_experimental_signals.py \
  --output-dir local-results/experimental-matching --starts 6
```

1. Audit the data and construct the confirmed grid.
2. Initialize capacitance and phase from the wings outside 32.50–32.85 MHz.
   This mask is only for initialization; nuclear tails may extend into it.
3. Optimize the full coupled signal-on model against all 500 raw samples,
   profiling readout and trying six starts including both polarization signs.
4. Save every candidate and convergence status. Reconstruct chi=0 and chi(P)
   with the same parameters to separate the physical baseline and signal.
5. Inspect every full-sweep residual, bound status and numerical conditioning.

For the supplied scans, selected fits converge without near-bound flags:

| Scan (file order) | Fitted P (%) | Center (MHz) | Split scale (kHz) | g | eta | C_tune (pF) | RMS (recorded units) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 35.078 | 32.678999 | 62.458 | 0.019326 | 0.064345 | 635.16 | 6.0126e-5 |
| 2 | −5.738 | 32.678749 | 62.308 | 0.022184 | 0.063432 | 1153.70 | 6.6430e-5 |
| 3 | 36.846 | 32.680238 | 62.487 | 0.018476 | 0.065841 | 611.28 | 6.1363e-5 |
| 4 | 41.070 | 32.679916 | 62.425 | 0.015041 | 0.067655 | 649.28 | 6.2352e-5 |
| 5 | −7.965 | 32.678977 | 62.313 | 0.015614 | 0.067307 | 1128.65 | 6.5380e-5 |

Displayed digits document numerical fit results, not significant-figure claims
for experimental uncertainty. One alternative start for scan 2 reaches its
evaluation cap and has worse SSE. Successful near-best starts agree on P;
the saved near-best range uses a 1% SSE window and is an optimizer diagnostic,
not a confidence interval. The scaled profiled Jacobian conditions are roughly
965–3103, excluding the eliminated readout directions.

The same phase-capacitance model fits the earlier baseline with RMS about
6.3643e-5 recorded units and C_tune about 544.84 pF, while remaining at n=1.
This model also fixes cable length at 3.580 m and C_stray at zero, while the
constant-phase diagnostic fits a bounded cable trim and stray capacitance.
The comparison illustrates dependence of inferred capacitance on the complete
electronics model; it does not isolate phase alone. Do not mistake a fitted
capacitance for an independently measured knob setting or impose the previous
fit as a prior.

`scan_N.csv` contains frequency, recorded, fitted, baseline, signal, residual.
The residual is recorded minus fitted; the display subtraction is recorded
minus the fitted physical baseline, not a separately measured clean signal.
No bins, including endpoints, are excluded from the final fits or figures.
`scan_N_fit.json` and `matching_report.json` retain settings, source and code
hashes, acquisition, parsing audit, all candidates and reconstruction parameters.

## Noise is a separate modeling step

The [matching uncertainty and distribution-shift protocol](distribution-shift.md)
extends the validation plan: noise/reference covariance, full joint baseline/signal
fit uncertainty, residual means and covariance, input-distribution comparisons,
and conditional known-P robustness tests are distinct outputs. The present
matching reports do not yet provide these conditional matrices or an experimental
network-accuracy test.

The full residual RMS is about 60–66 millionths of a recorded unit. A robust
high-frequency proxy is about 19–23 millionths:

```
sigma_proxy = MAD(second differences of residual)/(0.67448975*sqrt(6))
```

Only consecutive triplets wholly within the declared wings enter this
diagnostic. Its assumptions are locally smooth mean response and independent
Gaussian equal-variance noise. The full residual lag-one correlations are
about 0.43–0.52, and endpoint deviations reach roughly 0.00084–0.00092 units.
Smooth mismatch and endpoint structure must not be relabeled as white noise.
Different polarized scans are not repeats of one unchanged signal; a general
500x500 covariance is not identified by these five examples.

The default generator uses the high-frequency proxy for a white-Gaussian
reference scenario. A characterized covariance in recorded-units squared can
be supplied; symmetry, shape and positive semidefiniteness are checked.
Reproducible settling, pickup and model discrepancy require separate modeling
and further validation. The generator does not silently replay fitted residuals
as clean signal or independent noise.

## Generate new labeled examples

```bash
python tools/generate_matched_data.py \
  --matching-report local-results/experimental-matching/matching_report.json \
  --num-samples 2000 --num-configurations 100 --seed 42 \
  --output local-results/experiment-anchored-500-v2.npz
python tools/export_signal_matching.py \
  --fit-dir local-results/experimental-matching \
  --dataset local-results/experiment-anchored-500-v2.npz
```

The displayed generator example is always event zero, not a selected favorable
case. The run produces 2,000 events from 100 configurations anchored in all five
jointly fitted seed configurations. Each event gets a NEW uniform P in
[-0.6,0.6] by default. Labels are simulator truth, not fitted experimental P.

`configs/matched-generator-coverage.json` records controlled half-ranges:
1% for drive/resistors/coil R-L/tuning C/cable R/readout gain; 5% susceptibility
amplitude; 5 mm cable trim; 1 kHz center; 1% splitting; 5% broadening; 0.005
EFG eta; 0.01 rad phase; 0.01 rad/MHz slope; 0.02 rad/MHz² curvature; 0.001
recorded-unit offset. These are robustness choices, not fitted population
uncertainties. The complete seed bundle preserves its fitted relationships
before these controlled excursions. Cable L/C are recalculated together when
R changes to keep the established half-wave propagation scale. Fixed quantities
and the coupling-amplitude convention are explicit in the coverage file.

The full signal-on circuit supplies the raw clean voltage. The nuclear signal
is the difference from its chi=0 response; it is included once. Independent
event noise is added, and an independent baseline reference is averaged over
16 synthetic sweeps and shared within each simulated configuration. Sixteen is
a teaching setting, not a newly inferred acquisition count.

NPZ outputs retain raw/noisy, raw/clean, baseline/clean, nuclear/clean, event
noise, and independent noisy references as float64 arrays in recorded units.
Newly drawn P is a float64 fraction; the exact 500-bin frequency grid is float64
in MHz. Configuration and source-scan identifiers are integer arrays, and
provenance fields are strings. There is no `calibration` array.
The adapter uses the circuit's legacy detector/DC field names for the effective
readout: detector gain is recorded-units per node volt, and `dc_offset_v` carries
the recorded-unit offset in these matched configurations, not a claimed DAQ
voltage conversion. The companion metadata state the output units explicitly.

## Training and prediction

`train_model.py` and `predict.py` accept the generated NPZ directly. The
`learning_data.py` contract uses raw and reference-subtracted recorded units
without a required TE calibration. Subtraction precedes float32 conversion.
Each model saves the exact 500-bin frequency array, feature mode, units,
training-only scalers and target normalization. Clean simulator arrays remain
available for diagnostics and are excluded from network inputs.

The declared [training protocol](../../configs/lineshape-training.json) uses a
multiscale CNN with an 80-epoch cap and validation-selected checkpoint. All
`source_scan_1based` descendants stay together: source scans 3, 4 and 5 supply
1,200 training examples, source 2 supplies 400 validation examples, and source 1
supplies 400 test examples. These labels are newly sampled simulator truth.
Splitting only by simulated configuration would leak measured-seed relationships.

The [published training record](../assets/lineshape-training.json) retains
all test predictions, metrics, history, source partitions, preprocessing and
hashes. Its source-holdout synthetic result is not experimental polarization
accuracy. The five development scans do not provide an independent experimental
holdout, and controlled excursions do not establish empirical distributions.

```bash
python tools/run_model_comparison.py --protocol configs/lineshape-training.json
python tools/export_lineshape_training.py
```

For fresh output directories pass the same `--runs-dir` to both commands.
Ordinary measured prediction accepts raw/reference arrays, the exact grid and
explicit `feature_mode="lineshape"` and `voltage_unit="recorded units"`; it does
not require P labels or simulator/group identifiers. The trained model requires
corresponding references; a fitted baseline from the same sweep is not an
independent reference. See the [README](https://zetanaut.github.io/NMR-AI/walkthrough.html#train)
for direct training and prediction commands.
