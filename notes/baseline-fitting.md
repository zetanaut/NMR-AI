# Fit the measured deuteron baseline

This is the laboratory record for the [baseline practical](../docs/baseline.html).
The circuit follows [arXiv:2603.10146v5, Fig. 1 and Eqs. (1)–(14)](https://arxiv.org/pdf/2603.10146v5#page=4);
see the [physics notes](physics-electronics-theory.md) for topology and conventions.

## Establish acquisition metadata first

The public example is now [`examples/deuteron-baseline.csv`](../examples/deuteron-baseline.csv),
provided by the repository owner specifically for student use and publication.
It contains one headerless record: timestamp plus 500 amplitudes. This is a
deuteron baseline with nominal center **32.7 MHz**; no frequency column is present.

The CSV preserves every supplied field and space, with only a final newline
added. Published SHA-256: `dac7c4598c6ec32250ab763ab1bf99e2a7aa7346de4e452ce00e80f0e2b1eb28`.
`load_baseline_csv` in `tools/baseline_data.py` repairs 35 spaces-before-decimal
artifacts in memory and records every repaired token. Ambiguous numbers,
nonfinite values, and wrong field counts are rejected. All 500 samples and the
timestamp are preserved; neither fitter interprets the first row as a header.
See [the example README](../examples/README.md) for source provenance and loading.

The owner confirmed that every baseline has 500 bins and retains the original
proton sweep offsets and spacing, with nominal center changed from 213 MHz to
32.7 MHz. This gives:

```
f_j = (212.6 + 0.0015287*j) - (213 - 32.7) MHz
    = 32.3 + 0.0015287*j MHz, j = 0..499
first = 32.3000000 MHz; last = 33.0628213 MHz
spacing = 1.5287 kHz; endpoint midpoint = 32.68141065 MHz
```

The nominal center is not exactly the midpoint of these inherited offsets.
Keep the translated grid; do not substitute a symmetric linspace or count 500
intervals between 500 samples. The confirmed default is
`configs/deuteron-acquisition.json`; its hash and source are saved with the fit.
Both fitters and `preview_baseline.py` use it by default. Alternative grid
overrides require start, step, and frequency source together.

## Apply the known cable tuning BEFORE fitting

The owner confirms **n = 1**, with **λ/2 = 358.0 cm = 3.580 m** at nominal
32.7 MHz and installed length very close to this. This is independent hardware
information, not an integer inferred from the fitted curve. The paper explicitly
requires integer half-wave tuning and quotes approximately 0.78 velocity factor
and 360 cm half-wavelength for deuterons [Sec. 2, pp. 2–3](https://arxiv.org/pdf/2603.10146#page=2).

The working preset is [deuteron-baseline-setup.json](../configs/deuteron-baseline-setup.json).
It fixes the branch and propagation scale, allowing only δℓ = ±0.1074 m (±3%).
This is our explicit working interpretation of “a few percent,” not an independently
measured tolerance or Gaussian standard uncertainty. Do not increase it to absorb
other model errors. The allowed physical interval is 3.4726–3.6874 m.

### Make the cable constants consistent

For h = 3.580 m and f₀ = 32.7 MHz, β₀ = π/h and v_phase/c = 2f₀h/c = 0.7809802874.
Changing a length label while keeping incompatible RLGC constants does not
change the model's electrical wavelength.

Retain the nominal assumptions R′ = 3.43 Ω/m, G′ = 0 and Z_lc = sqrt(L′/C′) = 50 Ω.
These losses and impedance are not newly measured. With G′ = 0, derive both L′
and C′ from the exact lossy propagation relation:

```
ω₀ = 2πf₀; β₀ = π/h
C′ = 2β₀² / [ω₀ sqrt(R′² + 4β₀² Z_lc²)]
L′ = Z_lc² C′
γ = sqrt[(R′ + iωL′) iωC′]
```

This gives L′ = 2.13391803595×10⁻⁷ H/m and C′ = 8.53567214380×10⁻¹¹ F/m.
At f₀, Im(γ)h = π to numerical precision and attenuation is 0.03427383/m.
The exact complex characteristic impedance is approximately 50.0382−1.9543i Ω;
50 Ω denotes nominal sqrt(L′/C′), not a claim that the lossy Z₀ is purely real.

The derivation is implemented in `calibrated_cable_lc` in
`tools/fit_tuned_baseline.py`. It tends to the familiar low-loss conversions
L′ = Z_lc/v and C′ = 1/(Z_lc v) when R′ = 0.
The setup's `half_wave_length_m` field rejects RLGC constants that contradict
the known propagation scale. At each frequency use the full γ and Z₀ from
the same RLGC values. The physical length remains fixed across the sweep.

## Reproduce the constrained diagnostic example

```bash
python tools/fit_baseline.py examples/deuteron-baseline.csv \
  --starts 24 --output-dir local-results/deuteron-tuned-baseline-fit
python tools/export_baseline_example.py --fit-dir local-results/deuteron-tuned-baseline-fit
```

Use a fresh output directory. `fit_baseline.py` is a convenience entry point to
the setup-driven fitter with the supplied deuteron preset, not a separate
unrestricted search. The former unconstrained workflow and its public fit have
been replaced. The CSV and confirmed frequency grid are unchanged.

Twenty-four starts (seed 42) vary only C_tune, C_stray, and δℓ. Numerical
capacitance bounds are 10–2000 pF and 0–400 pF respectively; these are search
intervals, not measured component tolerances. The 800 pF initial tuning estimate
comes from 1/(ω₀²L₀) with nominal L₀ = 30 nH; it is not a prior.
At every circuit step the three unknown readout coefficients a,b,d are solved
by weighted linear least squares. They still count as three unknowns:
**three nonlinear coordinates plus three profiled readout coordinates = six**.

### What the constrained result tells us

This is a **diagnostic constrained fit, not an accepted hardware calibration**.
It reaches the +3% length limit (ℓ ≈ 3.6874 m, 1.0300 half-waves, δℓ ≈ +10.74 cm)
and the 400 pF stray-capacitance limit. Do not interpret the fitted trim as a
measurement of the installed length; it shows pressure against the allowed range.

Whole-scan RMS is approximately 8.1195×10⁻⁴ recorded units, or 0.3296% of the
measured 0.2463471 peak-to-peak range. The largest residual is approximately
3.5850×10⁻³ at sample 0. Lag-one residual correlation is approximately 0.9958.
All 500 bins and both endpoints remain fitted and plotted. This smooth mismatch
is not an estimate of electronic noise or polarization error.

The effective C_tune is about 49.92 pF. Readout gain is about 1.606×10⁴ recorded
units/V and offset about −194.6 recorded units, strongly compensating each other.
Those values, the boundary pressure, and correlated residuals challenge the
remaining nominal components and constant-phase approximation. They are not
validated settings to seed an experimental generator. In particular, a nominal
30 nH coil near half-wave would suggest an isolated-coil capacitance near 790 pF;
the fitted capacitance must be checked against the actual setting and full
zero-reactance condition, not accepted because the curve looks close.

The public JSON contains exact values, every candidate, convergence flags,
active-bound status and a separate near-bound warning. “Near” means within
10⁻⁵ of the declared search-interval width, so optimizer tolerance cannot hide
boundary pressure. The reported Jacobian is for the three scaled nonlinear
coordinates after profiling readout, not a full six-parameter covariance.
Derivatives use three-point finite differences with relative step 10⁻⁴;
no statistical parameter errors are inferred from this diagnostic.

## Use additional independent information

First obtain the approximate tuning capacitance or knob calibration, coil R/L,
cable impedance/loss characterization, and detector settings. Replace corresponding
nominal assumptions with independently known values or supported constraints.
Do not declare a fitted value to be an independent measurement.

For this same setup, copy `configs/deuteron-baseline-setup.json` and preserve
the known branch and propagation. For another setup, complete
`configs/baseline-setup.template.json`; nulls intentionally prevent silent
assumptions about missing cable/component records.

```bash
python tools/fit_tuned_baseline.py examples/deuteron-baseline.csv \
  --setup my-known-setup.json \
  --starts 24 --output-dir local-results/my-tuned-baseline-fit
```

Every active parameter requires a source or explicit assumption:

- `value`: fixed, excluded from optimization.
- `initial` and `bounds`: an unknown fitted only within the declared limits.
- `profile: true`: allowed only for all three readout parameters together;
  eliminate a,b,d analytically. If any readout value is independently constrained,
  use explicit fixed/bounded coordinates instead.
- Optional `prior` with `mean` and positive `sigma`: independent Gaussian
  measurement constraint in physical units. It requires `noise_sigma_recorded`
  so spectrum and hardware constraints have consistent statistical weights.

A starting point is not a prior; a hard bound is not a standard uncertainty.
The noise scale must be a positive scalar or 500-bin array in recorded units.
The implementation assumes diagonal data errors and independent Gaussian
constraints. Without a measured noise scale it uses range-normalized unweighted
least squares and makes no likelihood or parameter-error claim.

With known branch n, specify `cable_tuning` and `cable_delta_length_m`:
`length = n*pi/beta(f_tune) + delta_length`. Include `half_wave_length_m`
when independently known. A directly measured physical length is an alternative:
omit the branch block and supply `cable_length_m`. Never impose both length
definitions or optimize an integer as a continuous parameter.

The full zero-reactance condition gives C_tune = 1/[ω Im(Z_line)] when the
denominator is positive. The isolated-coil estimate is only a check. A
phase-sensitive voltage maximum is not necessarily zero resonator reactance.

## Fitting, validation, and saved evidence

1. Read every record with the audited CSV loader; remove only exact duplicates.
2. Construct the actual frequency grid and apply independent hardware constraints.
3. Build the χ=0 coil with shunt stray admittance and the passive RLGC line.
4. Add one series capacitor and damping resistor, then finite-source/input loading.
5. Fit only declared unknowns, retaining every starting-point solution.
6. Plot recorded-minus-fitted residuals on a separate scale, retaining all bins.
7. Inspect bound hits, correlation, localized/endpoint deviations and competing fits.
8. Validate against independent acquisitions, not other bins of the fitted scan.

Free RF amplitude and free readout gain together are rejected: only their
product is identifiable. Filling factor and susceptibility amplitude multiply
χ=0 and cannot be determined from a baseline.

Saved outputs are `baseline_fits.png`, `trace_N.csv` (MHz, recorded, fitted,
recorded-minus-fitted) and `fit_report.json`. The report includes data/code/setup
hashes, acquisition provenance, versions, seed, parsing repairs, fixed/fitted/
profiled status, all candidates, bounds and diagnostics.

Reconstruct without optimizing using
`fit_tuned_baseline.reconstruct_fit(f_hz, report["fits"][i])`.
Frequency-dependent detector phase is retained. Coefficients a,b,d and A,φ₀,d
are alternative representations, not six readout unknowns.
Recorded units are not volts without independent DAQ conversion.

The exporter requires a tuning-informed deuteron report, validates code hashes
and saved-curve reconstruction, and publishes all 500 points, residuals, physical
parameter meanings and provenance. For multiple distinct scans it selects the
median whole-scan RMS, not the best. The supplied CSV (including its timestamp)
is explicitly authorized for publication; other raw data are not published.

The existing 512-bin synthetic teaching benchmark is unchanged. It already
uses a derived n=1 operating point, but its nominal cable and component settings
are a separate controlled scenario, not this measured 3.580 m setup. Adopting
new hardware parameters in that benchmark requires regenerated data and retraining.
