# Reproduce the measured baseline fit

This is the laboratory record for the [baseline-fitting practical](../docs/baseline.html). The model is freshly derived from [arXiv:2603.10146v5, Fig. 1 and Eqs. (1)–(14)](https://arxiv.org/pdf/2603.10146v5#page=4); see the [physics notes](physics-electronics-theory.md) for circuit conventions and fixed component assumptions.

## Inputs and preparation

The supplied CSV is headerless, with a timestamp followed by 500 values per row. There are six rows and five distinct records. Use all records on load; eliminate only exact duplicates for fitting and retain the count in metadata. Do not interpret two copies of one record as independent noise measurements.

The supplied grid is `212.6 + j*0.0015287 MHz`, j=0…499. Voltage conversion remains a separately confirmed acquisition input. The fitter therefore reports residuals in recorded units, not guessed V or mV. It preserves both source files and does not execute the supplied old script.

## Use independent tuning information first

The published illustration is an **exploratory** fit: approximate capacitance,
the intended half-wave integer, cable characterization, and tolerances have not
been independently supplied for these scans. The fitted length is about 6.7014
half-waves in the nominal cable model, not a known integer tuning setting.
Do not infer the branch from this fit and then claim it as independent evidence.

For a hardware-specific fit, complete `configs/baseline-setup.template.json` from
the tuning/component record and run:

```bash
python tools/fit_tuned_baseline.py /path/to/single_event_data.csv \
  --setup my-known-setup.json \
  --start-mhz 212.6 --step-mhz 0.0015287 \
  --starts 12 --output-dir local-results/tuned-baseline-fit
```

Every active parameter requires an explicit declaration and source/assumption:

- `value`: fixed, excluded from optimization.
- `initial` and `bounds`: unknown, fitted only within supported limits.
- Optional `prior` with `mean` and positive `sigma`: independent Gaussian
  measurement constraint, in the parameter's physical units. This requires
  `noise_sigma_recorded` so the spectrum and hardware constraints are weighted
  consistently. A starting guess alone carries no prior information.

The noise scale is a positive scalar or 500-bin array in recorded units. The
objective uses diagonal data errors and independent Gaussian constraints; no
full covariance or correlated hardware-constraint treatment is implied. Without
noise estimates, use fixed values and bounds with unweighted SSE. No parameter
uncertainties are fabricated from that objective.

For known half-wave integer n at f_tune, the optional `cable_tuning` block records
`half_wave_multiple`, `reference_hz`, and `source`. Replace `cable_length_m` with
`cable_delta_length_m` in the parameter declarations. At each evaluation,

```
beta0 = Im[gamma(2*pi*f_tune)]
length = n*pi/beta0 + delta_length.
```

The integer is fixed from the independent tuning record; only the supported
correction may vary. Alternatively supply physical length directly and omit the
branch block. Cable wavelength depends on propagation velocity, not vacuum c.
Known velocity factor and impedance imply consistent low-loss L_c/C_c values:
`L_c ≈ Z0/(VF*c)`, `C_c ≈ 1/(Z0*VF*c)`. Use measured RLGC when available.

Record the calibrated capacitor setting and its tolerance; `1/(omega²L0)` is
only the isolated-coil estimate. The full zero-reactance relation is
`C_tune = 1/[omega*Im(Z_line)]` when the denominator is positive. A detector
voltage maximum alone does not prove that this reactance condition holds.

The new report includes all parameter sources, fixed/fitted status, values,
bounds, Gaussian constraints, derived length, free-parameter names/count,
starting-point solutions and separated data/constraint objectives. Reconstruct
with `fit_tuned_baseline.reconstruct_fit(f_hz, fit)`. Both free RF amplitude and
free readout gain are rejected because their product is what the baseline sees.
Inactive susceptibility/filling parameters cannot be made baseline unknowns.

The template deliberately contains nulls for missing tuning records. Numerical
tests use explicitly synthetic hardware settings to verify fixed values,
single-unknown recovery, known-branch trim, and constraint weighting. They do not
supply missing experimental information for a new constrained fit.

## Reproduce the exploratory example from NMR-AI/

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python tools/fit_baseline.py /path/to/single_event_data.csv \
  --start-mhz 212.6 --step-mhz 0.0015287 \
  --starts 24 --output-dir local-results/baseline-fit
```

Use a new output directory for a rerun. A confirmed conversion can be supplied with `--volts-per-unit`; 0.001 is appropriate ONLY when the recorded unit has been confirmed as mV. The numerical shape fit is otherwise identical.

## Fitting sequence

1. Build the χ=0 coil with its parallel stray admittance.
2. Transform the load through a passive distributed RLGC cable.
3. Add the single series tuning capacitor and damping resistance.
4. Solve the finite-source/input-loaded node voltage from Fig. 1.
5. For candidate C_tune, length, and C_stray, solve readout quadratures and offset by linear least squares.
6. Optimize the three positive/bounded shape coordinates with 24 starts and seed 42.
7. Keep all candidates; choose the minimum whole-scan unweighted squared residual.
8. Reconstruct the chosen curve, calculate recorded-minus-fit residuals, and inspect structure and identifiability.

The search fits C_tune in pF, length in meters, and C_stray in pF with bounds `[0.2,3,0]` to `[600,5,400]`. RF resistance values, coil inductance, and cable RLGC are fixed nominal assumptions. Frequency-dependent detector phase is fixed at zero for this exercise. Constant phase, effective amplitude, and offset are fitted. Susceptibility parameters and filling factor are not baseline-fit variables.

## Saved outputs

| Output | Contents |
|---|---|
| `fit_report.json` | Source/code hashes, versions, scan mapping, unit status, record counts, seed/starts, bounds, fixed circuit, all starting-point solutions, coefficients, convergence flags and diagnostics |
| `baseline_fits.png` | Measured/fitted overlays and separately scaled residuals for the five distinct traces |
| `trace_0.csv` … `trace_4.csv` | Frequency MHz, recorded voltage, fitted voltage, recorded-minus-fit residual |

Each fit stores `quadrature_coefficients=[a,b,d]` and the physical circuit used to calculate u. The saved prediction is `a*Re(u)+b*Im(u)+d`. Recover it with `fit_baseline.fitted_curve(frequency_hz, report["fits"][i])`; no optimizer is needed. A known voltage conversion also produces `circuit_in_confirmed_volts` with the corresponding detector phase/gain/DC fields.

The shape Jacobian's columns are scaled by `[100 pF,1 m,100 pF]` before computing its condition number. This is a diagnostic under stated coordinates, not a physical uncertainty interval. The candidate table is important: weakly distinguishable cable branches and capacitor combinations can give similar curves.

## Observations from the supplied scans

With all bins included, four distinct scans have residual RMS approximately 1.4–1.5×10⁻⁵ recorded units (about 0.024–0.036% of their individual peak-to-peak ranges). One scan has RMS about 1.1×10⁻⁴ (0.174% of its range), with a conspicuous localized residual near 212.91 MHz. Several scans have endpoint residuals. No mask was applied and no bins were silently removed.

This establishes that the physical circuit can closely track the observed Q-curve shapes within the stated assumptions. It does not establish unique component measurements, prove the origin of the localized feature, or measure white-noise variance. The duplicate is not independent evidence. An electronic-noise study needs stable independent repeats, acquisition/filter metadata, and separation of drift/signal/model mismatch.

The next experiment should confirm frequency/voltage metadata, constrain weakly identified hardware fields, and provide independent baseline acquisitions. A deuteron generator additionally needs its own 32.7 MHz baseline/signal/calibration data; the supplied proton-region scans must not be transferred by relabeling the frequency.

## Publish the measured example

```bash
python tools/export_baseline_example.py --fit-dir local-results/baseline-fit
```

This explicit export selects the median whole-scan residual RMS among the five
distinct sweeps (trace 5 in the supplied set). It verifies the saved curves against
the circuit coefficients and the stored statistics, then publishes a vector
overlay with all 500 measured points and a separately scaled residual panel.
No refitting, smoothing, bin exclusion, or voltage-unit inference is performed.

The displayed RMS is 15.303 × 10⁻⁶ recorded units, or 0.0275% of the measured
peak-to-peak range. This is a baseline-fit metric, not polarization error. The
largest absolute residual is 237.38 × 10⁻⁶ recorded units; the endpoint deviations
remain visible. A comparison table includes all five whole-scan RMS values.

The exporter also lists every fitted and fixed parameter on the page with its
physical meaning and value. The three readout quantities A, phase, and offset
are derived from the saved quadratures rather than unused circuit defaults.
Linear coefficients a, b, d are the same three degrees of freedom, not additional
parameters. The actual exploratory solution and nominal fixed components are
explicitly distinguished from independent hardware measurements.

The exporter writes `docs/assets/baseline-example.svg` and a companion JSON with
selection criteria, source/code/figure hashes, circuit/readout coefficients, and
statistics. It synchronizes the example section in `docs/baseline.html`. The
figure and statistics work without JavaScript. The original CSV, timestamps,
and full local fit outputs remain outside version control.
