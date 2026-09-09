# Three experimental examples

These are three worked examples across two materials and two acquisitions.
The first two compare models on the same butanol sweeps; UVA-ND3 supplies
the separate acquisition.

For the student execution sequence, follow [walkthrough steps 4–6](https://zetanaut.github.io/NMR-AI/walkthrough.html#single-site).
Each step provides commands, expected results and local raw-fit figures through
`tools/plot_fit.py`. The two-site step explicitly reads the single-site report
from the preceding step. The commands later in this record rebuild the public
reference artifacts and retain their pinned comparison provenance.

| Example | Measurements | Model and purpose |
| --- | --- | --- |
| [1: Single-site starting model](https://zetanaut.github.io/NMR-AI/matching.html) | All five butanol sweeps in `Sample_RawSignal.csv`, 500 bins each | Preserve the original complex spin-1/full-circuit exercise and its separate single-site generator |
| [2: Butanol C–D/O–D comparison](https://zetanaut.github.io/NMR-AI/butanol.html) | Exactly the same five raw sweeps and frequency grid | Add the second deuteron environment from the supplied butanol theory; compare full residuals with Example 1 |
| [3: UVA-ND3 data](../uva-nd3.html) | Five predetermined records resampled onto the exact 500-bin teaching grid | Jointly fit raw phase, circuit baseline and the ND3 spin-1 response; retain explicit nominal hardware assumptions |

The [physics](physics-electronics-theory.md), [baseline](baseline-fitting.md),
and [original matching](experimental-matching.md) records remain the authority
for the existing circuit and 500-bin measurement contracts. The new fit demos
do not change either generator, any generated dataset, or the trained 500-bin
TE-area benchmark. Extending a full generator would require new data and training
before changing its performance claims.

## Butanol theory: supplied statements and implementation choices

The owner supplied `fit_multisite.py` and `fit_multisite_test.py` in the
`butanol_theory` directory. Their byte counts and SHA-256 hashes are recorded in
[butanol-theory-source.json](../../configs/butanol-theory-source.json). They were
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
[material_lineshapes.py](../../tools/material_lineshapes.py):

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

[match_butanol.py](../../tools/match_butanol.py) propagates the summed complex
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

See [the complete comparison record](../assets/butanol-matching.json) and
[all five fits](https://zetanaut.github.io/NMR-AI/butanol.html#results) for the measured RMS reductions,
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

The public [working example](../../examples/uva-nd3.json) contains five records on
the exact **500-bin** grid `32.3 + 0.0015287*j MHz`, j=0..499, ending at
33.0628213 MHz. Source records 1, 126, 251, 376 and 501 were selected at equal
intervals before fitting. All current fit inputs, residuals and figures use
these 500 samples.

The original numeric excerpt is preserved byte-for-byte in
[the reader’s original measurement resource](../../tools/_data/uva-nd3-measured.json), with SHA-256
`af5a4358bae2aa48cc3c7d16967816679ae15f9c863511c9c96daa729885e4bb`.
The parent acquisition is the 2022-09-23 file named in the artifact, 48,637,894
bytes, SHA-256 `57b5755108b05109d54bc5e1b40c11f07a6fefa59e4e35ecad05e5d9a7a77afc`.
Its original frequency coordinates and phase/reference values are source
provenance, rather than an alternate working input contract.

[prepare_uva_nd3.py](../../tools/prepare_uva_nd3.py) linearly interpolates phase and
recorded baseline separately in frequency using float64. Their subtraction is
retained as a provenance check; the active fit uses raw `phase` with its baseline
present. No baseline is added to these already raw sweeps.
The target window is contained in the source support except for a 7.1e-15 MHz
first-endpoint roundoff difference. The interpolation clamps only endpoint
roundoff within four floating-point spacings and rejects physical extrapolation.
Source frequencies above the target endpoint are outside the working window.
The original arrays are preserved; the derived arrays are explicitly labeled.

The [reader](../../tools/uva_nd3_data.py) requires exact equality to the confirmed
500-bin frequency array and to the reproducible source derivation. It checks
`phase - baseline == basesub` in every bin, source/configuration hashes and record
identity. The source archive is rejected as a working example input. Each
record retains its timestamp and reported 4,000 acquisition sweeps. Interpolation
changes resolution and correlates errors; no independent-bin noise covariance
or new independent measurements are inferred. ND3 hardware and voltage
calibration remain unresolved; the butanol hardware contract does not follow
from sharing its teaching grid. No external checkout or pipeline is required.

## UVA-ND3 joint raw-phase and circuit fit

The active input is the resampled raw `phase`. Its baseline is present. Stored
`baseline` and `basesub` remain provenance and are excluded from fitting.
Acquisition polarization, area-calibration coefficients and previous fitted
curves are also excluded. No added baseline or TE calibration is required.

[match_uva_nd3.py](../../tools/match_uva_nd3.py) now uses the same physical single-
capacitor circuit and full complex susceptibility propagation as the other raw
examples, with the single-site ND3 spin-temperature model:

```
Q(P) = 3P² / (2 + sqrt(4 - 3P²))
chi_ND3 = A_cgs [(P+Q)/2 K_plus + (P-Q)/2 K_minus]
u = node_voltage(resonator_impedance(f, chi_ND3))
phi(f) = phi1*(f-f_ref) + phi2*(f-f_ref)²
y_fit = L[a Re(u exp(i phi)) + b Im(u exp(i phi)) + d]
B_fit = same fitted response with chi=0
S_fit = y_fit - B_fit
residual = resampled raw phase - y_fit
```

L is the existing linear interpolation from source frequencies onto the exact
500-bin grid. The complete circuit model uses the same observation operator as
raw phase. The susceptibility vanishes at P=0; it is not divided by P. Kernel
normalization, cgs convention, passive RLGC propagation, finite-source loading
and one series tuning capacitor follow the [physics record](physics-electronics-theory.md).
No polynomial voltage baseline is used. The signal is included once through
susceptibility; its difference from the chi=0 response is formed after fitting.

Nine nonlinear coordinates are center, splitting, g, EFG eta, P, log10 effective
susceptibility scale, log10 tuning capacitance in pF, phase slope in rad/MHz and
phase curvature in rad/MHz². Three profiled coefficients a,b,d specify gain,
constant detector phase and offset. There are **12 fitted unknowns**. Gain is
hypot(a,b) recorded-units per node volt, constant phase is atan2(-b,a), and d is
in recorded units. This parameterization does not establish a DAQ conversion.

The [configuration](../../configs/uva-nd3-matching.json) records all fixed component
values, fitted bounds and their source. Fixed components use the existing
standalone tutorial `circuit.py:nominal_circuit`, with its derived half-wave
operating point at 32.68 MHz. These are declared assumptions for a conditional
fit, not ND3 component measurements or the known butanol cable setup. The
capacitor is fitted from a nominal resonance initialization, not fixed at a
previous estimate. No cable-length search or empirical hardware population is
inferred. The same phase rotation and component topology apply to both chi=0
and signal-on cases.

Electronics are initialized on wings outside 32.48–32.88 MHz. Every final fit
uses all 500 raw samples and six starts, including both P signs; no reference
subtraction or frozen wing polynomial is used. Interpolation correlates errors,
so this unweighted diagnostic fit does not claim a measured noise covariance
or statistical parameter errors.

All five selected fits converge. In source record order, fitted P estimates are
approximately -25.9103%, -34.8012%, -34.1851%, -33.5660%, and -33.2714%.
Whole-sweep raw residual RMS ranges from 1.63207e-5 to 2.99509e-5 recorded units.
Record 376 reaches the 2000 pF upper tuning-capacitance search bound; that bound
is retained and shown, not widened. Other selected capacitances are about
681.83, 1056.75, 383.65 and 495.07 pF. Scaled profiled-Jacobian conditions range
from about 492 to 82,015. Numerical convergence and fitted settings do not
establish identifiable hardware or accurate experimental polarization.
The [complete fit record](../assets/uva-nd3-matching.json) retains every
candidate, fixed circuit value, initialization mask, source/code hash and
32-to-64-node quadrature check.

The figures lead with the complete raw sweeps and preserve their baseline
level. Each results row shows raw phase with the full prediction and fitted
circuit baseline, the fitted nuclear response S_fit, and raw-minus-fit residuals.
The response decomposition is a fit result; it is not the input channel.

## Noise, validation and reproduction

The butanol demonstration retains its MAD second-difference noise proxy,
conditional on independent Gaussian noise and a locally smooth mean. The ND3
resampling correlates neighboring errors, so its working fit reports RMS, maximum
residual and lag-one correlation without an iid-noise sigma proxy. Structured
residuals are not a measured covariance. The pages report full RMS, not an unstated SNR. If an SNR is added,
its signal and noise definitions must be saved explicitly.

Numerical checks include independent complex orientation integrals, infinite
frequency site-area normalization, the zero-site/full-circuit limit, zero-P
susceptibility, preserved source arrays, independently checked linear resampling,
known-P raw full-circuit recovery after resampling, explicit raw-channel selection,
and reconstruction of every published fit. These tests establish the stated
mathematical/numeric behavior, not instrument validity. Validation of polarization
accuracy requires independent acquisitions and appropriate known-P tests.

From the repository root, with requirements installed and fresh output paths:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/match_butanol.py \
  --output-dir local-results/butanol-matching --starts 6
python tools/prepare_uva_nd3.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python tools/match_uva_nd3.py \
  --output-dir local-results/uva-nd3-raw-matching-500-v3 --starts 6
python tools/export_material_examples.py \
  --butanol-fit-dir local-results/butanol-matching \
  --nd3-fit-dir local-results/uva-nd3-raw-matching-500-v3
python -m unittest discover -s tests -p 'test_material_examples.py' -v
```

The butanol command above defaults to the checked-in single-site comparison report.
For a local comparison against a fresh fit, pass that report explicitly with
`--single-site-report`, as in the walkthrough, and use `plot_fit.py` to view the
local result. Publishing a new comparison also requires its matching single-site
snapshot in the public assets so the exporter can verify the pinned report hash.
The exporter verifies source hashes and reconstruction before writing the two
new practicals, four SVG figures and two public fit records. Its text templates
are the source for generated prose. Library versions used for verification are
saved in each publication record. Existing benchmark artifacts are unchanged.

The butanol comparison retains its byte-identical original
[single-site reference snapshot](../assets/experimental-matching-single-site-reference.json).
The current matching publication has updated generation metadata; the fitted
single-site parameters and residuals agree exactly with this reference.
