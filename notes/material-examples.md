# Three experimental examples

Updated 2026-09-07. These are three worked examples across two materials and two
acquisitions, not three independent datasets. The owner identified the existing
five raw sweeps as butanol and requested the additional UVA-ND3 example.

| Example | Measurements | Model and purpose |
| --- | --- | --- |
| [1: Single-site starting model](../docs/matching.html) | All five butanol sweeps in `Sample_RawSignal.csv`, 500 bins each | Preserve the original complex spin-1/full-circuit exercise and its separate single-site generator |
| [2: Butanol C–D/O–D comparison](../docs/butanol.html) | Exactly the same five raw sweeps and frequency grid | Add the second deuteron environment from the supplied butanol theory; compare full residuals with Example 1 |
| [3: UVA-ND3 data](../docs/uva-nd3.html) | Five predetermined records from a different acquisition, 512 measured bins each | Fit a single-site conditional response after subtraction of the recorded reference; retain unresolved hardware assumptions |

The [physics](physics-electronics-theory.md), [baseline](baseline-fitting.md),
and [original matching](experimental-matching.md) records remain the authority
for the existing circuit and 500-bin measurement contracts. The new fit demos
do not change either generator, any generated dataset, or the trained 500-bin
TE-area benchmark. Extending a full generator would require new data and training
before changing its performance claims.

## Butanol theory: supplied statements and implementation choices

The owner supplied `fit_multisite.py` and `fit_multisite_test.py` in the
`butanol_theory` directory. Their byte counts and SHA-256 hashes are recorded in
[butanol-theory-source.json](../configs/butanol-theory-source.json). They were
read as sources, not executed or imported. The scientific context is
[Dulya et al., *A line-shape analysis for spin-1 NMR signals*, NIM A 398 (1997)
109–125](https://doi.org/10.1016/S0168-9002(97)00317-3). The supplied code is the
source of the particular two-site implementation choices below; the article's
existence does not independently validate the full instrument fit.

The supplied model combines C–D and O–D responses with a shared polarization,
a common physical Lorentzian half-width, separate splitting scales and EFG
asymmetries, and an O–D mixing weight. Its inverse-splitting factors are essential:
the weight describes integrated susceptibility, rather than the ratio of heights
on two differently scaled frequency axes. The local implementation is
[material_lineshapes.py](../tools/material_lineshapes.py):

```
χ(f) = A_int [(1−K) κ_CD((f−fc)/Δ_CD)/Δ_CD
                  + K κ_OD((f−fc)/Δ_OD)/Δ_OD]
κ_site = w_plus K_plus + w_minus K_minus
Γ = g_CD Δ_CD = g_OD Δ_OD
```

Each kernel's absorption integral over its infinite reduced-frequency axis is
P. Each site density therefore integrates to P over MHz; the weighted areas
are `(1−K)P` and `KP`. No finite-window normalization is performed. `A_int` has
units cgs·MHz. To retain the original fit amplitude coordinate,
`A_int = 10**log10_signal_scale_cgs * Δ_CD`. K is an effective signal weight,
not an independently measured chemical abundance. P is common to the two sites
under spin temperature; site-specific populations would be a different model.

The fit uses the supplied source's **weak-quadrupole intensity option**, consistent
with the original spin-temperature branch weights and complex kernel. It does
not implement the source's optional frequency-dependent intensity correction.
The source's absorption kernel with its azimuth weight is proportional to the
existing normalized orientation integral by the common factor `2 sqrt(3)`;
the fitted amplitude absorbs that common convention. Here a uniform
Gauss–Legendre powder average replaces the supplied endpoint-inclusive sample
mean. Independent polar/azimuth quadrature tests check both complex components.
The supplied synthetic example parameters, axis rescaling, and empirical gain
tilt are not apparatus measurements. The source's polynomial voltage background
is not substituted for the physical circuit in this comparison.

[match_butanol.py](../tools/match_butanol.py) propagates the summed complex
susceptibility through the same physical single-capacitor circuit as Example 1.
It retains n=1, the 3.580 m cable at nominal 32.7 MHz, consistent passive RLGC,
finite-source loading, zero stray capacitance, and the explicit nominal coil,
resistance and filling-factor assumptions. Capacitance and detector phase are
fitted again. All 500 measured samples remain on
`32.3 + 0.0015287*j MHz`, including both endpoints.

Shared parameters keep the original bounds. The three extra nonlinear coordinates
are `Δ_OD/Δ_CD` in 1.05–2.5, `η_OD` in 0–0.5, and K in 0–0.4. These are search
bounds, not empirical distributions or uncertainty intervals. There are 12
nonlinear coordinates plus three profiled readout coefficients, versus nine plus
three in Example 1. Setting K=0 recovers the original single-site circuit model
at identical quadrature order. The new fits use 128 azimuth nodes; publication
also evaluates each saved solution at 256 nodes without refitting.

Six starts per scan include an opposite-sign P seed. Every candidate, unsuccessful
termination, near-bound flag and scaled profiled-Jacobian condition is retained.
The minimum residual solution is reported with its actual convergence status.
Different near-best solutions reflect optimization/model sensitivity; their
spread is not a statistical confidence interval. The old report remains the
unaltered comparison snapshot, identified by its hash.

See [the complete comparison record](../docs/assets/butanol-matching.json) and
[all five fits](../docs/butanol.html#results) for the measured RMS reductions,
P estimates and remaining residual structure. These are in-sample comparisons
with three added unknowns; smaller residuals do not establish greater P accuracy.

The reproduced RMS reductions in scan order are 2.57125%, 0.05350%, 2.37152%,
4.35254%, and 0.27578%. Common-site P estimates are approximately 35.274%,
−5.763%, 37.264%, 41.442%, and −8.098%. All selected solutions converge and
have no near-bound flags; one alternative start reaches the evaluation limit.
Scaled profiled-Jacobian condition numbers range from about 3,855 to 22,247,
so numerical convergence does not imply well-determined independent parameters.
The largest change from 128 to 256 azimuth nodes is 1.3411×10⁻⁷ recorded units.
Re-evaluating the original single-site solutions at 128 instead of 32 nodes
changes any sample by at most 2.2624×10⁻⁸ recorded units. These checks resolve
the numerical integration more finely than the observed residual improvement.

The figure removes one common two-site fitted χ=0 baseline from the measured
data and both predictions. Its residual panels always show recorded minus each
complete fitted sweep. Intrinsic site contributions are plotted separately;
separate detector voltages are not added through the nonlinear circuit.

## UVA-ND3 acquisition and provenance

The owner requested publication of the earlier ND3 example as **UVA-ND3 data**.
The standalone [teaching excerpt](../examples/uva-nd3.json) selects source records
1, 126, 251, 376 and 501 at equal intervals through the 501-record acquisition,
before fitting. The source is the 2022-09-23 acquisition file named in the excerpt,
48,637,894 bytes, SHA-256
`57b5755108b05109d54bc5e1b40c11f07a6fefa59e4e35ecad05e5d9a7a77afc`.
The excerpt hash is
`af5a4358bae2aa48cc3c7d16967816679ae15f9c863511c9c96daa729885e4bb`.
Each original record's hash, timestamp and reported sweep count are preserved.
This is a documented numeric excerpt, not a byte-for-byte copy of the source
file. No external checkout or pipeline is needed to load, fit or display it.

Each selected record preserves the parsed values of `frequency_mhz`, `phase`,
`baseline`, and `basesub`. The audited [reader](../tools/uva_nd3_data.py) checks
512 finite bins, strictly increasing frequency and exact floating-point equality
of `phase − baseline == basesub`. It does not interpolate, smooth or repair them.
All selected records report 4,000 acquisition sweeps; that alone does not
establish independence or a noise covariance.

The measured grid starts at about 32.3000000 MHz and ends at 33.0999878 MHz.
Digitized frequency intervals range from about 1.550293 to 1.574707 kHz.
Use the stored arrays. The fact that they contain 512 bins does not make them
the 500-bin grid used for generated learning examples, and they do not inherit
the butanol hardware contract. Hardware calibration and the recorded-unit-to-volt conversion for
this acquisition remain unresolved. In particular, the butanol cable and tuning
assumptions are not assigned to this dataset.

## UVA-ND3 conditional lineshape fit

The input is the measured `basesub`, with its recorded reference retained for
inspection. Existing acquisition polarization, area-calibration coefficients,
and previous fitted curves are neither inputs nor truth labels. TE calibration
is not required for spin-temperature lineshape inference.

[match_uva_nd3.py](../tools/match_uva_nd3.py) fits the complex single-site shape
with a constant absorption/dispersion mixture and a joint cubic residual:

```
y_sub = phase − baseline
y_fit = a_abs (−Im κ) + a_disp Re κ + b0 + b1 t + b2 t² + b3 t³
t = (f_MHz − 32.7)/0.4
```

Here κ is a unit-area shape in MHz⁻¹. The physical branch weights are divided by
P analytically, using `Q/P = 3P/(2+sqrt(4−3P²))`, which has a finite limit at
P=0. The free amplitude carries the overall signal strength. This conditional
coordinate fits a nonzero observed lineshape; it is not a physical generator
predicting a nonzero zero-P signal. The full susceptibility generator continues
to vanish at P=0.

There are five nonlinear parameters (center, splitting, width, asymmetry and P)
plus six profiled linear coefficients, for 11 unknowns. Absorption/dispersion
coefficients have recorded-unit·MHz units and polynomial coefficients have
recorded units. The polynomial represents residual background after measured
reference subtraction. It is optimized jointly over all 512 bins, not frozen
from a preliminary wing-only fit. This explicit response approximation does
not claim a full-circuit reconstruction of unknown apparatus.

The [configuration](../configs/uva-nd3-matching.json) gives every bound and
starting value. Six starts cover both P signs; all selected fits converge and
have no near-bound flags. They give P estimates approximately −25.03%, −34.42%,
−33.51%, −33.20%, and −32.65%, in excerpt order. Whole-sweep residual RMS ranges
from 1.7200×10⁻⁵ to 3.2323×10⁻⁵ recorded units. These conditional estimates
are not calibration/accuracy claims. The [full record](../docs/assets/uva-nd3-matching.json)
saves the candidate solutions, data/code hashes, diagnostics, and 32-to-64-node
quadrature checks (maximum change 3.3224×10⁻¹⁰ recorded units). Figures retain all measured bins, with residual sign
`recorded reference-subtracted spectrum minus complete fit`.

## Noise, validation and reproduction

Both new demonstrations reuse the original residual diagnostics: MAD of the
second residual difference divided by `0.6744897501960817*sqrt(6)`, conditional on
independent Gaussian noise and a locally smooth underlying trace. The declared
wing masks serve only that diagnostic. Structured residuals are not a measured
covariance. The pages report full RMS, not an unstated SNR. If an SNR is added,
its signal and noise definitions must be saved explicitly.

Numerical checks include independent complex orientation integrals, infinite
frequency site-area normalization, the zero-site/full-circuit limit, zero-P
susceptibility, preserved measured arrays, known-P conditional synthetic recovery,
and reconstruction of every published fit. These tests establish the stated
mathematical/numeric behavior, not instrument validity. Validation of polarization
accuracy requires independent acquisitions and appropriate known-P tests.

From the repository root, with requirements installed and fresh output paths:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/match_butanol.py \
  --output-dir local-results/butanol-matching --starts 6
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/match_uva_nd3.py \
  --output-dir local-results/uva-nd3-matching --starts 6
python tools/export_material_examples.py \
  --butanol-fit-dir local-results/butanol-matching \
  --nd3-fit-dir local-results/uva-nd3-matching
python -m unittest discover -s tests -p 'test_material_examples.py' -v
```

The butanol command uses the checked-in single-site comparison report. An
experimental refit must update that report before starting a new comparison.
The exporter verifies source hashes and reconstruction before writing the two
new practicals, four SVG figures and two public fit records. Its text templates
are the source for generated prose. Library versions used for verification are
saved in each publication record. Existing benchmark artifacts are unchanged.

The butanol comparison retains its byte-identical original
[single-site reference snapshot](../docs/assets/experimental-matching-single-site-reference.json).
The current matching publication has updated generation metadata; the fitted
single-site parameters and residuals agree exactly with this reference.
