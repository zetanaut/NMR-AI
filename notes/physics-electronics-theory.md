# Physics, electronics, and the measurement contract

Reference: D. Seay, I. P. Fernando, and D. Keller, *Polarized target nuclear magnetic resonance measurements with deep neural networks*, [arXiv:2603.10146v5](https://arxiv.org/abs/2603.10146v5), [EPJ A 62, 153 (2026)](https://doi.org/10.1140/epja/s10050-026-01937-x). Prepared 2026-09-07 after reading all 28 pages, including Appendix A. Page numbers below refer to that version. Key equations and the circuit diagram were also inspected visually. PDF SHA-256: `14977413a028f93120029729bfb4b895aa5eee2ad3e12e0560a814847db76331`.

These notes distinguish the paper's physics, explicit implementation conventions, and evidence from the supplied measurements. Equations below are restated/derived with unambiguous names. This independent educational implementation is not a reproduction of the paper's training runs or an experimentally calibrated deuteron instrument. The [baseline practical](../docs/baseline.html) explains the measured-fit workflow.

## 1. Continuous-wave measurement and conventions

This is swept-frequency, continuous-wave, phase-sensitive NMR, not pulsed FID spectroscopy. An approximately constant-current RF source excites a coil coupled to the sample. The complex magnetic susceptibility changes the coil impedance; the tuning components, transmission line, input loading, and detector phase determine the recorded voltage. Analysis and calibration act on that response.

Internal units are Hz, rad/s, V, A, Ω, H, F, m, and radians. Input MHz is multiplied by 10⁶ once. Fit coordinates in pF are multiplied by 10⁻¹² once. The detector phase is distinct from the EFG azimuth; `filling_factor` is distinct from quadrupole `eta`. Vector and tensor polarization are fractions, not percent. Output polarity depends on the detector, not just the spin label.

With time dependence exp(+iωt), use the paper's cgs convention:

```
χ = χ′ − iχ″
L_eff = L0 (1 + 4π η_fill χ)
Z_coil = R_coil + iω L_eff
       = R_coil + 4π η_fill ω L0 χ″ + iω L0(1 + 4π η_fill χ′).
```

Positive absorptive susceptibility increases real loss; dispersion changes reactance. Do not add a second `4π` if susceptibility has already been converted to SI. This implementation uses cgs susceptibility explicitly, with a named setup-dependent amplitude coefficient. [Paper Eqs. (1), (4), pp. 3–4](https://arxiv.org/pdf/2603.10146v5#page=3).

## 2. Physical circuit implemented in circuit.py

The topology follows Fig. 1 and the loaded-voltage relation in Eq. (2). The resonant leg contains ONE damping resistance, ONE tuning capacitor, and the line-transformed coil. The source resistance and finite amplifier input load the same node.

The coil is shunted by stray capacitance in both the baseline and susceptibility-on cases:

```
Z_load = 1 / (1/Z_coil + iω C_stray).
```

This admittance form correctly tends to `Z_coil` when stray capacitance vanishes. Zero capacitance is an open capacitive branch, not a short. Retaining the same terminal topology for signal-on and signal-off avoids an artificial difference unrelated to nuclear response.

For distributed cable R/L/G/C per meter, define

```
γ = sqrt[(Rc + iωLc)(Gc + iωCc)]
Z0 = sqrt[(Rc + iωLc)/(Gc + iωCc)]
Z_line = Z0 (Z_load + Z0 tanh(γℓ)) / (Z0 + Z_load tanh(γℓ)).
```

The passive square-root branch has nonnegative attenuation. Compute `γ` and `Z0` from the same RLGC parameters, not from inconsistent independent approximations. At zero length the line input equals the load. In the lossless half-wave limit, `βℓ=nπ` also reproduces the load; a quarter wave gives `Z0²/Z_load`. A fixed physical cable is not an exact half wave at every frequency in a sweep. Loss and frequency-dependent electrical length produce the Q-curve. [Paper Eqs. (3), (7), (8), p. 4](https://arxiv.org/pdf/2603.10146v5#page=4).

The circuit's remaining equations are

```
Z_leg = R_D + 1/(iω C_tune) + Z_line
I0 = U/R0
Y = 1/R0 + 1/R_input
u = I0 / (1/Z_leg + Y).
```

Taking the real part yields Eq. (2):

```
Re(u) = I0 [Re(Z_leg) + Y |Z_leg|²]
              / [(1 + Y Re(Z_leg))² + (Y Im(Z_leg))²].
```

This exact nodal form includes finite-source loading. The ideal-current approximation drops `1/R0` from the loading admittance while retaining `I0`; the tutorial does not make that approximation.

**Printed-notation resolution:** Eq. (3) defines `ZT` including a capacitor, while Eq. (6) places a capacitor outside its own `ZT`. Literal substitution counts two capacitors. Here `line_input` denotes only the transformed load; `resonator_impedance` adds the single capacitor and damping resistor shown in Fig. 1. We derive the final loading from Eq. (2), rather than cascading Eq. (6) with another loading correction. The implementation is tied to the diagram, not to a claim that those overlapping printed names are internally consistent.

The detector measures

```
V_det = G Re[u exp(iφ(f))] + V_DC
φ(f) = φ0 + φ1(f−f_ref) + φ2(f−f_ref)².
```

`φ1` is in rad/Hz and `φ2` in rad/Hz². This centered polynomial parameterizes the electronics PHASE described in [Eqs. (10)–(14), p. 5](https://arxiv.org/pdf/2603.10146v5#page=5), not the baseline voltage. The first fitting exercise keeps the frequency-dependent phase terms fixed at zero. The detector gain may be negative to represent inversion. No grid-dependent minimum is subtracted. Magnitude/diode readout is a different observable and must not be substituted for a phase-sensitive trace.

## 3. Deuteron baseline: frequency provenance and fit status

The current public example is `examples/deuteron-baseline.csv`, supplied by the
repository owner for student use and explicit publication. It contains one
headerless timestamp-plus-500-amplitude record and no frequency column. The
nucleus is deuteron, with approximate reference 32.68 MHz. Published SHA-256:
`dac7c4598c6ec32250ab763ab1bf99e2a7aa7346de4e452ce00e80f0e2b1eb28`.

The raw fields are preserved; only a final newline was added. The shared loader
repairs 35 spaces-before-decimal formatting artifacts in memory, records each
repair, and rejects ambiguous or nonfinite numbers. No smoothing, missing-bin
interpolation, or voltage conversion is performed. The website preview plots
all 500 amplitudes against sample index, not an assumed frequency axis.

The actual start and spacing/endpoints are still needed for a physical fit.
An approximate reference alone does not establish scan width. DAQ conversion
to volts also remains unknown. No previous fit on another file or an invalid
frequency mapping supplies these missing inputs. The parameter guide and
physical circuit remain usable; measured-fit metrics await acquisition metadata.

The circuit implementation does not execute or import the supplied old script.
Its positive component entries supply nominal engineering seeds: coil 30 nH and
0.35 Ω; source 619 Ω; input 50 Ω; damping 10 Ω; cable L = 2.542×10⁻⁷ H/m,
C = 1.027×10⁻¹⁰ F/m. Cable R = 3.43 Ω/m follows the low-loss seed
Rc≈2 Z_nom α using 50 Ω and 0.0343/m; Gc=0 is an explicit approximation.
These are not independently confirmed deuteron hardware measurements.
Exact RLGC propagation replaces an independently specified propagation slope.

Use `fit_tuned_baseline.py` with the actual acquisition grid and independent
capacitance, cable branch/length, component and readout records. Fixed inputs are
excluded from optimization. Bounded unknowns remain within supported ranges.
Independent Gaussian constraints require a measured data-noise scale.
A known half-wave branch gives `length = n*pi/beta(f_tune) + delta_length`,
evaluated at the deuteron tuning frequency. Do not use vacuum wavelength,
infer a branch from the fit and call it independent, or mistake a starting guess
for a measurement constraint.

The pedagogical variable-projection fitter `fit_baseline.py` uses a 32.68 MHz
reference and requires explicit start/spacing and their source. Its built-in
bounds are only numerical search assumptions. For known hardware, use the
setup-driven fitter instead. Detector coefficients (a,b,d) correspond to
`atan2(-b,a)` phase and `hypot(a,b)` recorded-unit gain. Unknown RF voltage and
gain cannot both be identified from their product. Filling factor and
susceptibility scale are inactive at χ=0.

The reports retain acquisition provenance, code/source hashes, versions, seeds,
all candidates, fixed/fitted status, supported constraints, duplicate count,
all-bin policy, reconstruction information, residual metrics and scaled-Jacobian
diagnostics. A corrected fit must regenerate these quantities, inspect structure
and parameter degeneracies, and be checked on independent acquisitions.
See [the fitting record](baseline-fitting.md).

## 4. Complex spin-1 powder response

The single-site, weak-quadrupole, spin-temperature model follows [Dulya et al., NIM A 398 (1997) 109–125](https://doi.org/10.1016/S0168-9002(97)00317-3) and [paper §3.1, pp. 7–8](https://arxiv.org/pdf/2603.10146v5#page=7).

For normalized populations,

```
P = n_plus − n_minus; Q = 1 − 3 n_zero
n_plus = (2+Q+3P)/6; n_zero = (1−Q)/3; n_minus = (2+Q−3P)/6
Q = 2 − sqrt(4−3P²) = 3P² / [2+sqrt(4−3P²)]
w_plus = (P+Q)/2; w_minus = (P−Q)/2.
```

The rational expression avoids cancellation near zero. Weights are finite at P=0 and ±1 without a polarization floor. Their sum is P. The branch-area ratio obeys `P=(r²−1)/(r²+r+1)` under spin temperature; it is not automatically the ratio of two peak heights after circuit distortion. Independent tensor polarization would require a different population contract.

Let `x=(f−f_center)/split`, where `split=3ωQ/(2π)`, and `g` is Lorentzian HWHM divided by `split`. For EFG asymmetry η and azimuth φ,

```
Y = sqrt(3 − η cos(2φ))
x_res(t) = eps [1 − η cos(2φ) − Y² t²],  0 ≤ t ≤ 1
K_eps(x,φ) = (1/π) integral_0^1 dt / [x_res(t) − x + i g].
```

The orientation variable t is the absolute polar cosine. Absorption is `−Im(K)` and integrates to one over the infinite x axis. The analytic integral uses `z=1−eps*x−η*cos(2φ)+i*g` and `atanh(Y/sqrt(z))/sqrt(z)`. For the positive branch divide by `πY`; for the negative branch take the negative complex conjugate before that division. This ensures the physically correct dispersion reflection as well as positive absorption. Independent complex quadrature tests verify both real and imaginary parts.

The azimuthal average uses Gauss–Legendre quadrature on `[0,π/2]`. This is a convergent uniform powder average rather than an endpoint-biased unweighted sample mean. The total normalized kernel is `w_plus*K_plus + w_minus*K_minus`. At η=0 and small g, one transition spans −2…+1 with a horn at +1; its partner is reflected. Broadening gives extended tails, not two isolated Gaussian peaks.

The real absorption integral is equivalent to the Dulya broadened branch with distinct `rho=sqrt(g²+b²)` and `c=sqrt(rho)`. This avoids ambiguous square-root naming in the printed Eqs. (16)–(19); the orientation integral, not a guessed interpretation of `X`, defines the code.

`susceptibility_scale_cgs` multiplies this dimensionless normalized kernel. The 0.11133 nominal coefficient is a supplied setup-scale seed, not a universal material susceptibility or an experimental calibration. A real application must fit/calibrate the signal scale with the filling factor and detector response. Normalization is never forced to a discrete finite-scan sum; truncated tails remain truncated.

## 5. Full-spectrum generation and calibration

Define the baseline and nuclear detector signal by

```
B(f;θ) = V_det(f,χ=0;θ)
S(f;P,θ) = V_det(f,χ(P);θ) − B(f;θ)
V_measured = B + S + noise.
```

This is full complex-susceptibility propagation. The paper also uses an additive circuit-baseline benchmark; that does not justify independently adding the nuclear response a second time here. A single-site kernel does not cover multi-site materials, contaminants, or RF-modified populations. Spin-1/2 and unresolved-quadrupole Voigt problems in §3.2 require their own response model.

The deuteron demonstration uses a fixed 512-bin, 32.3–33.1 MHz sweep. Starting from the explicit nominal passive components, `nominal_circuit` derives a one-half-wave cable at 32.68 MHz and chooses the tuning capacitance to cancel the line's imaginary impedance there. Zero stray admittance and zero phase slope/curvature are reference idealizations. They are not inferred from the supplied baseline data, whose exact acquisition grid still needs confirmation.

For temperature T, with `t=tanh(h*f0/(2*kB*T))`, spin-1 TE polarization is `4t/(3+t²)`; spin-1/2 is `t`. At 1.5 K, 32.7 MHz gives 0.06974899% for spin 1 for this deuteron-scale example. A 5% training example is not a 0.05% TE example. [Paper §5.1, Eqs. (28)–(34), p. 10](https://arxiv.org/pdf/2603.10146v5#page=10).

Area calibration is

```
C_cal = P_TE / integral [S_TE(f)/f] df
P_area = C_cal * integral [S(f)/f] df.
```

Changing from angular frequency gives `dω/ω=df/f`, with no extra `2π`. `integration_weights` supplies trapezoidal `df/f` weights on the actual grid. `make_features` uses raw voltage plus per-bin calibrated contributions from an independent reference-subtracted sweep. Their sum is the conventional area estimate. Circuit nonlinearity and shape-dependent weighting can make that estimate differ from P away from TE; the generator never normalizes it to the unknown label.

Each configuration has one independent noisy baseline reference, averaged over 16 sweeps by default and shared by its events. The TE calibration is ideal/noiseless in this benchmark. Voltage storage is float64 to retain tiny signals during subtraction; network inputs are float32 after preprocessing. A real model requires measured references and independently established calibration, including their uncertainty and any reference-to-signal drift.

## 6. Noise and physical parameter coverage

[Paper §2.2, §6.4, and Eq. (45)](https://arxiv.org/pdf/2603.10146v5#page=5) distinguish Gaussian electronic noise, coherent pickup, microphonics, and tuning drift. A full noise model is more than a histogram width. Characterize repeat differences, covariance, autocorrelation, spectral density, tails, and sweep-to-sweep coherence using development measurements.

The generator supports a finite symmetric positive-semidefinite 512×512 covariance in V² and draws `noise=L*z`, where `LLᵀ=Σ` and z is standard normal. Singular positive-semidefinite covariances are supported. A covariance measured on 500 bins is not silently relabeled as a 512-bin covariance. The default diagonal case has σ=10⁻⁹ V, corresponding to the nominal 10⁻⁶ mV scale discussed in the paper. It is a reference noise scenario, not the noise inferred from the supplied traces. Circuit filtering can be represented by a measured output covariance; thermal resistances and amplifier/filter bandwidths are not fabricated to claim a first-principles absolute noise level.

For independent repeats of an unchanged signal, `(scan1−scan2)/sqrt(2)` has single-scan noise variance. Shared drift, repeated duplicate rows, or evolving signals violate that interpretation. Coherent pickup does not generally disappear as `1/sqrt(N)`. Non-Gaussian structured processes need a validated extension beyond Gaussian covariance.

Use `SNR_peak=max(abs(S))/max(abs(noise))`, excluding B, to match Eq. (45). Peak-noise and noise-SD denominators are not equivalent. Keep residual-background mismatch separate from additive noise.

The narrow/broad presets are controlled physical sensitivity studies, not empirical distributions. Broad fractional half-ranges are 10% for RF voltage, coil resistance, cable resistance, filling factor, and susceptibility coefficient; 3% for tuning capacitance and coil inductance; 5% for damping; 1% for cable length. Detector phase varies ±0.12 rad, center ±0.025 MHz, splitting ±0.01 MHz around 0.08, g ±0.02 around 0.08, and EFG η ±0.02 around 0.03. Narrow excursions are one quarter of these. Independent center jitter is ±0.002 MHz by default. Other circuit fields are held fixed explicitly.

These distributions are intentionally inspectable numerical experiments. For experimental training, infer supported joint ranges from independent hardware constraints and development fits. Varying every fit coordinate independently can violate correlations; increasing event count cannot fix an unidentifiable calibration. A low baseline RMS alone cannot validate nuclear amplitude, noise, or a family of operating configurations.

## 7. Networks, validation, and metric conventions

The compact CNN (3,889 parameters) and ridge-summary-plus-multiscale CNN (82,391) are independent teaching architectures, not the paper's residual/Inception/SE model. Both see the same two-channel preprocessing. The multiscale model combines filters of widths 5/15/31, dilated residual blocks, pooling, and a correction head with a train-only frozen ridge estimate from 25 summaries. Network size is not a substitute for measured inference latency.

Configuration IDs define disjoint 80/10/10 train/validation/test groups. Scalers, target normalization, and ridge coefficients are fitted only on training rows. AdamW optimizes normalized absolute MSE; validation selects the checkpoint and early stopping. Test scores must not be reused to optimize the final estimator. Forward-pass benchmarking excludes preprocessing, IO, and transfers; measure end-to-end cost before deployment. The tutorial saves inference checkpoints, not a complete optimizer-resume state.

Following [Eqs. (38)–(45), pp. 13–14](https://arxiv.org/pdf/2603.10146v5#page=13):

```
error = P_true − P_pred
bias = mean(error)
width = population SD(error), ddof=0
RMSE² = bias² + width²
percentage-point error = 100 * fractional-P error
relative error percent at P0 = 100 * fractional-P error / abs(P0).
```

Positive bias means underprediction. A fractional error 0.0005 is 0.05 percentage points and 1% relative at P0=0.05. Pooled metrics converted at P0 are not conditional performance at that P. Use signed local bands or fixed-P pseudo-experiments, state counts, and evaluate tails. Residual width is neither the standard error on mean bias nor automatically a calibrated per-event confidence interval.

The paper constructs conditional intervals by inversion of fixed-estimator residual quantiles. Its DAE uncertainty study propagates replicas through a fixed network and downstream analysis, retaining the full output covariance. Pointwise one-SD bands are not simultaneous coverage bands; treating denoised bins as independent can understate fit uncertainty. Appendix A uses sample SD across replicas, whereas the scalar RMSE identity uses population SD.

Extraction error is only one term in polarimetry uncertainty. TE temperature/calibration, gain drift, coil coupling, field homogeneity, nonlinearity, and coil-weighted versus beam-weighted polarization remain. The paper's area-to-P numerical table cannot safely define a universal calibration: displayed bias/width conversion factors are not uniformly consistent with one scalar. Derive and measure the actual calibration instead. No paper accuracy claim is inherited by these smaller teaching runs.
