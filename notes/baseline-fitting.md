# Fit the measured deuteron baseline

This is the laboratory record for the [baseline practical](../docs/baseline.html).
The circuit follows [arXiv:2603.10146v5, Fig. 1 and Eqs. (1)–(14)](https://arxiv.org/pdf/2603.10146v5#page=4);
see the [physics notes](physics-electronics-theory.md) for topology and conventions.

## Establish acquisition metadata first

All work in this tutorial concerns **deuterons near 32.68 MHz**, including the
supplied `single_event_data.csv`. The user confirmed this operating frequency.
The CSV has six headerless timestamp-plus-500-bin rows, one exact duplicate, and
no frequency column. The old script's proton grid is invalid for these scans.
The former fit overlay, parameter estimates, and residual metrics have been
withdrawn, not relabeled as deuteron results. They must be recomputed.

The actual start frequency and bin spacing (or equivalent endpoints) are still
needed. The approximate reference frequency alone does not specify the width,
whether endpoints are included, or where the reference falls between bins.
Do not silently assume the synthetic generator's 512-bin window describes this
500-bin measurement. Preserve the raw CSV and record the acquisition source.

## Use independent tuning information

Complete `configs/baseline-setup.template.json` from component/tuning records.
Its reference is approximately 32.68 MHz; confirm the exact tuning frequency.
The cable half-wave integer, capacitance, component tolerances, and readout
information remain independent inputs, not quantities established by an
unconstrained curve match.

After establishing the grid, replace the uppercase placeholders below with
actual values; they are not runnable defaults:

```bash
python tools/fit_tuned_baseline.py /path/to/single_event_data.csv \
  --setup my-known-setup.json \
  --start-mhz START_MHZ --step-mhz STEP_MHZ \
  --frequency-source "Acquisition record identifying this scan and its grid" \
  --starts 12 --output-dir local-results/deuteron-baseline-fit
```

Every active parameter requires a source or explicit assumption:

- `value`: fixed, excluded from optimization.
- `initial` and `bounds`: unknown, fitted only within supported limits.
- Optional `prior` with `mean` and positive `sigma`: an independent Gaussian
  measurement constraint in physical units. It requires `noise_sigma_recorded`
  so data and hardware constraints have consistent statistical weights.

A starting point is not a prior; a bound is not automatically a standard
uncertainty. The noise scale is a positive scalar or 500-bin array in recorded
units. This implementation uses diagonal data errors and independent hardware
constraints. Without a measured noise scale, use fixed values/bounds and
unweighted least squares; no likelihood or parameter errors are claimed.

The template intentionally contains nulls for missing tuning inputs. Tests use
explicitly synthetic deuteron grids and components. They do not fill in missing
experimental records.

## Cable length and capacitance

For independently known half-wave integer n at f_tune, provide `cable_tuning`
with `half_wave_multiple`, `reference_hz`, and `source`. Declare
`cable_delta_length_m` instead of `cable_length_m`. At each evaluation:

```
beta0 = Im[gamma(2*pi*f_tune)]
length = n*pi/beta0 + delta_length
```

Use the supported trim/reference-plane tolerance. Alternatively supply a
measured physical length directly and omit the branch block. If several integer
branches remain possible, compare supported discrete hypotheses and retain the
ambiguity. Do not derive an alleged independent branch from the same fit.

Cable wavelength is set by propagation velocity, not vacuum c. Consistent
low-loss conversions from independently known velocity factor and impedance are
`L_c ≈ Z0/(VF*c)` and `C_c ≈ 1/(Z0*VF*c)`. Prefer measured RLGC.
All wavelength and tuning calculations must be evaluated at the deuteron
frequency, not transferred from proton-frequency results.

Use a calibrated capacitance setting and its tolerance. The isolated-coil
estimate `1/(omega²L0)` is only a check. The full zero-reactance condition gives
`C_tune = 1/[omega*Im(Z_line)]` when the denominator is positive.
A phase-sensitive voltage maximum need not mark zero resonator reactance.

## Fitting and validation

1. Read every CSV row and identify only exact duplicates.
2. Construct the actual frequency grid from confirmed acquisition metadata.
3. Build the χ=0 coil with parallel stray admittance, transform it through the
   passive RLGC cable, and add one series tuning capacitor and damping resistor.
4. Solve the finite-source/input-loaded node voltage and phase-sensitive readout.
5. Optimize only declared unknowns; retain every starting-point solution.
6. Plot recorded-minus-fitted residuals on their own scale, retaining all 500 bins.
7. Check bound hits, correlated/endpoint/localized residuals, competing solutions,
   and the scaled Jacobian before interpreting parameters.
8. Validate against independent acquisitions, not other bins from the fitted scan.

The baseline cannot identify filling factor or susceptibility amplitude at χ=0.
Free RF amplitude and free readout gain are rejected because only their product
is identifiable without independent information.

`fit_baseline.py` illustrates variable projection of readout quadratures. It
also requires explicit frequency metadata and uses a deuteron reference; its
built-in broad search bounds are numerical assumptions, not measured settings.
The setup-driven fitter is preferred for a hardware-specific result.

## Saved outputs and publication

Use a new output directory. The outputs are `baseline_fits.png`,
`trace_N.csv` (MHz, recorded, fitted, recorded-minus-fitted), and
`fit_report.json`. The report includes nucleus/reference, frequency source and
mapping, source/code/setup hashes, versions, seed, duplicate counts, all-bin
policy, fixed/fitted values, constraints, candidate solutions, and diagnostics.

Reconstruct the tuning-informed prediction with
`fit_tuned_baseline.reconstruct_fit(f_hz, report["fits"][i])`.
This retains any frequency-dependent phase. Readout coefficients a, b, d are an
alternative representation of A, phase, and offset, not three extra unknowns.

Recorded-unit conversion remains unknown. Never label residuals µV without
independent DAQ conversion. Residual RMS is not an electronic-noise measurement
or a polarization-error metric.

After a fresh fit on the correct grid, publish with:

```bash
python tools/export_baseline_example.py --fit-dir local-results/deuteron-baseline-fit
```

The exporter requires deuteron acquisition provenance, rejects a grid that does
not bracket the approximate deuteron reference, checks code hashes and exact
saved-curve reconstruction, and selects the median whole-scan RMS among distinct
traces. It publishes all measured bins, residuals, parameter meanings/values,
and provenance, without raw timestamps. No current measured-fit accuracy is
claimed while the acquisition grid remains unresolved.

The main training generator already uses a 32.68 MHz circuit reference. Its
synthetic configurations, noise, and TE calibration are not measurements derived
from the supplied baseline. The generator and trained benchmark are unchanged
by withdrawal of the invalid baseline illustration.
