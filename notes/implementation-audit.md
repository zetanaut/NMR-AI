# Implementation audit and unresolved physics conventions

Prepared 2026-09-07 against [arXiv:2603.10146v5](https://arxiv.org/pdf/2603.10146v5), after reading the full 28-page paper. Start with the [physics/electronics notes](physics-electronics-theory.md).

Status: **the current NMR-AI full-spectrum generator is not an accepted physical Q-meter generator**. This audit does not replace it with another assumed circuit. Existing data and trained examples are retained as labeled software prototypes, not validated physics results or reproductions of the paper.

## 1. Scope and evidence

The review distinguishes four types of evidence:

- **Paper:** equations, definitions, and claims in the pinned v5 publication.
- **Code:** behavior of the supplied circuit/lineshape implementations or the current tutorial.
- **Derivation/check:** a mathematical consequence or a numerical test performed during this review.
- **Unresolved:** an interpretation requiring confirmation before implementation or quantitative use.

Supplied source fingerprints, recorded without a runtime dependency or link to another project:

| Implementation inspected | SHA-256 |
|---|---|
| Circuit `Baseline` implementation | `148582d6d38269aec1edcec7547c3044572cdde00d3cf5ed3c8581a9d85f6c0e` |
| Supplied lineshape module | `41ae2c01c8d47099f0fedb7363b1cb07740a813a0d725e67d82245d3e7b43f1b` |

The supplied files were inspected read-only. No experimental records, fit pools, checkpoints, or research workflow are imported into this tutorial by this review.

## 2. What is wrong or incomplete in the current tutorial

| Current implementation | Finding | Required disposition |
|---|---|---|
| [Cubic baseline in `simulate`](../tools/nmr_lab.py) | Random polynomial coefficients do not describe coil/cable/tuning/loading physics | Replace only after the physical circuit contract is resolved |
| `gain_slope` multiplier | Independent real linear gain is not a derived phase-sensitive susceptibility response | Derive or experimentally justify the signal distortion; do not rename it “Q-meter theory” |
| `sample_configurations` ranges | Synthetic assumptions, not fitted experimental distributions | Keep provenance explicit; establish physical joint ranges from development measurements |
| Pake branch and powder average | Replaced the erroneous Gaussian peaks; convolution checks support the supplied branch implementation | Preserve those checks; resolve printed notation separately |
| `sum(S)=P/cc` | Fixed-grid discrete normalization | Establish its relation to detector units and the paper's calibrated integral before experimental use |
| `2.7e-5` noise and optional AR(1) | Not a measured noise covariance in the paper's units | Obtain/document noise scale, covariance, structure, and averaging conditions |
| Cubic wing subtraction | Potentially legitimate preprocessing, but not the physical baseline itself | Assess signal-tail bias and subtraction mismatch; do not conflate it with generation |
| Two-channel ridge/multiscale CNN | An independent teaching architecture | Do not identify it as the paper's Inception/SE CNN |
| 512-bin, 6,000-event, ≤25-epoch examples | Actual prototype runs, not paper reproduction settings | Do not transfer paper performance claims to these runs |
| `dulya-pake-v1` artifact identifier | Describes the lineshape version, not a validated complete instrument | A future full circuit needs distinct provenance and newly generated/trained artifacts |

The earlier replacement of Gaussian peaks corrected **one component only**. It did not validate the baseline, gain, scale, noise, or full generator. Passing a regression test that reproduces a plot establishes software consistency, not agreement with an experiment.

The current doublet's branch-weight ratio also needs a finite-window qualification: multiplying two kernels by the spin-temperature coefficients gives the intended integrated ratio only when their normalization/areas are treated consistently. An off-center finite grid truncates the branches differently, and frequency-dependent gain changes their areas further. The symmetric-grid ratio test is not proof that the post-gain, finite-window peak or area ratio equals the intrinsic population ratio.

## 3. Supplied circuit code: exact conventions that need preserving or resolving

### 3.1 Frequency and voltage units

**Code:** input frequencies are MHz; the code converts them with `omega=2*pi*f*1e6`. It computes drive current as `I=U*1000/R`, explicitly labeled mA, then multiplies by impedance.

**Derivation:** if `U` is volts and `R` is ohms, the resulting pre-gain voltage is mV. With `U=0.5` and `R=619`, current is `0.0008077544 A`, or `0.8077544 mA`.

**Unresolved:** the detector gain, scale factors, data-acquisition units, and DC-offset units must be traced together. A code variable called voltage cannot safely be relabeled V. The paper's scale `C_E` is experiment dependent.

### 3.2 Tuning-capacitor control

**Code:** `Cmain=20e-12*Cknob`, plus a frequency-dependent trim term. Thus a knob value of `0.4` produces `8 pF` before that trim.

**Paper:** Eq. (9) gives a knob-to-capacitance factor, while prose also describes capacitance ranges in pF.

**Required:** distinguish a dimensionless knob position from physical pF; publish the conversion constant. The numerical knob value is not automatically a capacitance in farads or pF.

### 3.3 Cable trim is not automatically “number of half-waves”

**Code:** `beta=beta1*omega` and `length=trim/(beta1*omega0)+delta_l`.

**Derivation:** at `delta_l=0`, `beta0*length=trim`. In this implementation, `trim` is an electrical phase in radians, not `n/2`. A half-wave requires `trim=pi`; a value `0.5` means 0.5 rad. Using its nominal `beta1=4.752e-9` and `f0=32.68 MHz`, `trim=0.5` gives approximately `0.51243 m`.

**Unresolved:** determine whether fitted trim values are intended as absolute electrical length, a residual detuning, or an effective fit coordinate. Do not silently interpret them as a physical `n*lambda/2` length.

### 3.4 Detector phase units

**Code:** the circuit exponent uses `exp(i*phi*pi/180)`, so `phi_const`, `delta_phi`, and the phase-polynomial values are degree-based. For example, a value of 5 becomes `0.0872665 rad`.

**Paper:** Eqs. (10)–(14) describe phase in radians.

**Required:** use explicit `_deg`/`_rad` names and a tested conversion. The EFG azimuth in the Pake formula is a different angle entirely.

### 3.5 Zero-susceptibility baseline and filling factor

**Code:** the supplied circuit baseline sets `chi` to zero. Its effective inductance therefore becomes `L0` independently of `eta`.

**Check:** changing the supplied baseline's `eta` from 0.01 to 0.8 changed the 512-bin baseline by exactly zero in the tested configuration.

**Consequence:** this parameter cannot be inferred from that circuit-only trace. It becomes relevant when susceptibility is included. Randomizing an inactive fill factor does not add physical variation, and a returned fit value does not make it identifiable.

### 3.6 Stray-capacitance zero limit

**Code:** the supplied `Zstray` helper assigns zero impedance when `Cstray==0`, which is then combined in parallel with the coil.

**Derivation:** zero capacitance means no shunt capacitive admittance, not a short circuit. The correct limiting behavior is `Z_terminal -> Z_coil`. An admittance formulation handles that limit directly.

**Disposition:** record this as a boundary-condition issue, not silently copy it into a new physical implementation. No supplied research code was changed during this audit.

### 3.7 Baseline offset and polarity

**Code:** the raw response is multiplied by −1; its real-part minimum over the selected grid is subtracted; `DC_offset` is then added.

**Consequence:** the offset is a minimum-referencing convention, not merely a physical detector DC level. It depends on the selected grid. A phase or sign change before this operation is not equivalent to negating the final shifted array.

**Required:** preserve the distinction between raw detector voltage and baseline-referenced display/DAQ values. Confirm the convention against a representative recorded baseline.

### 3.8 Passive parameters and fit-pool independence

Some supplied fit material uses a reference-fit reuse setting, and some effective fitted component values are negative. An unconstrained fit can produce a good curve while giving component values that cannot be interpreted literally as passive R, L, C, or physical cable length.

Those values must not be copied into a student table as measured hardware components without qualification. Likewise, repeated records sharing one reference fit are not independent baseline configurations. A future generator must distinguish effective fit coefficients from physical component measurements and count independent configurations honestly.

## 4. Printed equations needing confirmation

### 4.1 Resonator topology and naming

**Paper, p. 4:** Eq. (3) includes `R_D + Z_cap + transformed_load` inside `Z_T`. Eq. (6) then uses `r + Z_cap + Z_T`. A literal substitution therefore includes two tuning-capacitor terms.

**Code:** the supplied `ZT` is only the transformed cable/load; the capacitor is inserted once in `Zleg1=r+ZC+ZT`.

**Question to resolve:** is the Eq. (6) `Z_T` intended to mean the line-input impedance rather than the full Eq. (3) resonant impedance? Figure 1 and the component definitions should determine the intended topology, not a visual fit to a baseline.

Eq. (2) uses `R0` for current limitation and `R_i` for input loading. Eq. (6) describes `R1` as a source resistance, whereas the supplied code's `R1` is algebraically in parallel with the resonant leg and a separate `R` sets current. This naming must be mapped explicitly before combining formulas.

### 4.2 Pake square-root notation

**Paper, p. 7:** Eq. (17) prints `X=sqrt(A**2+b**2)` and Eq. (19) prints `cos(alpha)=b/X`; Eq. (16), however, uses `1/X`, `X**2`, and `Y*X` in places where the supplied closed form uses the additional square root.

**Code/independent check:** define `rho=sqrt(A**2+b**2)` and `c=sqrt(rho)`. The supplied branch uses `cos(alpha)=b/rho`, denominator `2*pi*c`, and `Y**2 +/- rho` terms. That expression agrees with direct numerical integration of the broadened orientation distribution.

**Disposition:** retain the two distinct variables in implementation notes. Seek confirmation of the printed notation rather than silently editing the paper or implementing a conflicting definition. Existing Pake tests do not rely on guessing that notation.

### 4.3 Susceptibility normalization

**Paper:** Eq. (1) includes `4*pi*eta`; Eq. (4) uses `eta_L` with susceptibility without explicitly repeating `4*pi`.

**Question to resolve:** is the numerical coupling in Eq. (4) defined to absorb `4*pi`? Confirm the susceptibility units and scale before computing calibrated signal amplitudes.

### 4.4 Sweep width in phase slope

**Paper, p. 5:** the example describes a “4 MHz” sweep covering 209–217 MHz, whose full width is 8 MHz.

**Required:** label half-span versus full span explicitly when converting a phase excursion to a phase slope. Do not infer that distinction from the word “length.”

### 4.5 Table 5 area-to-polarization conversion

**Paper, p. 22:** the text discusses converting area residuals to P residuals with calibration uncertainty set to zero. Under one fixed linear conversion, bias and width must be scaled by the same factor (width uses its absolute value).

**Check:** the displayed low-noise rows imply factors of `0.010/(4e-5)=250` from bias but `0.011/(3e-6)=3666.67` from width. The high-noise rows likewise give approximately `433.33` versus `3833.33`.

**Unresolved:** these rows may refer to different estimators, normalization conventions, or a transcription issue. Do not infer a calibration constant from this table until clarified. This observation concerns that conversion, not the validity of the entire study.

## 5. Paper-to-tutorial metric translations

| Quantity | Paper | Current tutorial | Translation |
|---|---|---|---|
| Residual | `P_true - P_pred` | `P_pred - P_true` | Reverse bias sign when comparing |
| Polarization table units | Polarization percent | Fractional P in saved metrics | Multiply tutorial bias/width by 100 to obtain percentage points |
| Residual width | Population SD, Eq. (40) | Population SD | Same definition after unit conversion |
| SNR | Peak signal / peak noise, Eq. (45) | Lesson also discusses peak signal / noise SD | Different denominators; do not compare values directly |
| Area | Calibrated frequency integral | Fixed-grid unweighted signal sum convention | Needs grid, gain, and calibration mapping |
| DAE bands | Input-error propagation, full output covariance | No DAE in current lab | Not a feature supplied by the tutorial |
| CNN | Residual/multiscale/SE architecture | Ridge summaries plus a separate multiscale CNN | Separate models; no equality of performance implied |

## 6. Checks performed and remaining work

Checks already performed during this review:

- Full v5 reading, including Appendix A; visual inspection of key typeset equations and tables.
- Source inspection of the supplied circuit, signal assembly, fitting conventions, and current tutorial code.
- Numerical confirmation of the inactive fill-factor parameter in the zero-susceptibility baseline.
- Dimensional checks for knob-to-pF conversion, degree-based phase, current units, and electrical cable trim.
- Independent complex-admittance derivation/check of Eq. (2).
- Boltzmann-population calculation reproducing the paper's rounded TE examples.
- Existing Pake convolution, signed-area, endpoint, powder, and reflection tests remain applicable to the lineshape component.

The [reference-math tests](../tests/test_reference_math.py) preserve standalone checks for the loaded-voltage identity, transmission-line limits, phase detection, TE values, and metric conventions. They do **not** validate a complete replacement circuit or experimental parameter distribution.

Still required before a physically grounded replacement baseline can be claimed:

1. An authoritative circuit/topology and unit mapping resolving the issues above.
2. At least one reference instrument configuration and baseline trace with grid, detector channel, sign, gain, and offset convention identified.
3. Baseline and noise characterization supporting realistic parameter variation and covariance.
4. A self-contained implementation checked against that reference, then new data/training runs with full provenance.

No new physical baseline, noise covariance, calibration, or experimental agreement is claimed by this audit. The existing toy baseline has not been relabeled as real theory.
