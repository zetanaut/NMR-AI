# Physics, electronics, and inference: working reference

Read and prepared on 2026-09-07. Primary source: D. Seay, I. P. Fernando, and D. Keller, *Polarized target nuclear magnetic resonance measurements with deep neural networks*, [arXiv:2603.10146v5](https://arxiv.org/abs/2603.10146v5), [EPJ A 62, 153 (2026)](https://doi.org/10.1140/epja/s10050-026-01937-x). Page references below use the **28-page v5 PDF**, not the earlier preprint versions.

The complete paper, including Appendix A and references, was read. The circuit diagram and the typeset equations on pp. 4, 7, and 10 were inspected visually; extraction of PDF text alone is not reliable for these equations. The PDF SHA-256 is `14977413a028f93120029729bfb4b895aa5eee2ad3e12e0560a814847db76331`.

These are explanatory notes, independent derivations, and an implementation review—not a reproduction of the paper's experiment. The published article identifies its text as [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) on p. 26. Attribution is given above; notation has been disambiguated and commentary added. No paper figures or experimental datasets are bundled here.

Use alongside the [implementation audit and unresolved questions](implementation-audit.md). The existing tutorial's polynomial-baseline simulator is **not** an implementation of the paper's Q-meter model. Its results must not be described as a reproduction of this work.

## 1. What is being measured

Reference: [§§2–2.1, pp. 2–5; §5.1, p. 10](https://arxiv.org/pdf/2603.10146v5#page=2).

The instrument is a swept-frequency, continuous-wave NMR Q-meter used for polarized targets. It is not a pulsed-NMR free-induction-decay experiment. The RF coil couples to the target's magnetic susceptibility. The approximately constant-current drive converts the impedance response into a voltage, and phase-sensitive detection selects the intended channel.

Keep three different objects separate:

1. **Intrinsic nuclear response:** the complex susceptibility and the spin-dependent absorption lineshape.
2. **Instrument response:** coil, parasitic capacitance, tuning components, transmission line, loading, gain, and detection phase.
3. **Analysis:** baseline subtraction, residual-background fitting, integration or lineshape estimation, and calibration.

The Q-curve is the instrument response without the NMR susceptibility contribution. A polynomial fitted to spectral wings is an estimator for a remaining background, not an equivalent circuit for that Q-curve.

The spectrum can change at fixed polarization because instrument parameters change. Conversely, similar raw voltages need not imply identical polarizations if calibration or coupling changes. Experimental units and operating conditions are part of the inference problem, not optional metadata.

## 2. Notation that must not be conflated

| Symbol or field | Meaning in these notes | Important distinction |
|---|---|---|
| `f`, `omega = 2*pi*f` | Frequency in Hz, angular frequency in rad/s | MHz needs a factor of `1e6`; do not insert an extra `2*pi` into a frequency ratio |
| `eta_fill` | Coil–sample filling/coupling factor | Not the quadrupole asymmetry |
| `eta_efg` | Electric-field-gradient asymmetry in the Pake kernel | Not an electronic gain or phase |
| `phi_efg` | Azimuthal orientation for the powder lineshape | Not the detector phase |
| `phi_det` | Electronic phase relative to the RF reference | Units must be radians or explicitly converted from degrees |
| `P`, `Q_tensor` | Vector and tensor polarization fractions | `0.05` is 5%; tensor polarization is not circuit quality factor |
| `Q_cable` | Cable quality factor in the low-loss approximation | Not `Q_tensor` or the baseline Q-curve |
| `R_detuning` | Dimensionless spectral detuning | Not a resistance |
| `r_population` | Ratio of spin-transition branch areas | Not the series resistance named `r` in circuit code |
| `A` or `g_broadening` | Dimensionless dipolar broadening | Not signal area or a nuclear g-factor |
| `C_tune`, `C_stray` | Physical capacitances | Not polarization calibration `C_cal` or voltage scale `C_E` |
| `C_E`, `C_cal`, `cc` | Setup/implementation-dependent scales | They cannot be interchanged without an explicit conversion |

## 3. Susceptibility and the sensing coil

Reference: [Eqs. (1), (4), pp. 3–4](https://arxiv.org/pdf/2603.10146v5#page=3).

With the paper's convention,

```text
chi(omega) = chi_prime(omega) - i*chi_double_prime(omega)
L_eff(omega) = L0 * [1 + 4*pi*eta_fill*chi(omega)]
```

An explicit algebraic consequence, using that convention, is

```text
Z_coil = R_coil + i*omega*L_eff
       = [R_coil + 4*pi*eta_fill*omega*L0*chi_double_prime]
         + i*omega*L0*[1 + 4*pi*eta_fill*chi_prime].
```

Thus absorption changes the real loss term; dispersion changes the reactive term. The sign of the plotted voltage also depends on detector phase, amplifier inversion, and calibration. Downward peaks by themselves do not determine the sign of P.

The `4*pi` susceptibility normalization in Eq. (1) and the coupling notation in Eq. (4) require an explicit convention when translating to code. They should not be silently mixed with an SI susceptibility definition.

For a coil branch with parallel stray capacitance, the convenient terminal-impedance expression is

```text
Z_terminal = 1 / [1/Z_coil + i*omega*C_stray].
```

This has the correct `C_stray -> 0` limit: `Z_terminal -> Z_coil`. If susceptibility is included, compare signal-on and signal-off cases using the **same physical topology**. Changing the topology between those calculations would create an artificial signal.

For `chi = 0`, a fill factor that acts only through susceptibility cannot change the circuit-only baseline. This is an identifiability issue: fitting that parameter from a zero-susceptibility baseline alone provides no information about it.

## 4. Transmission line and the physical Q-curve

Reference: [§2, pp. 2–3; Fig. 1 and Eqs. (2)–(8), p. 4](https://arxiv.org/pdf/2603.10146v5#page=4).

### Distributed cable model

For per-unit-length cable resistance, inductance, leakage conductance, and capacitance,

```text
gamma(omega) = sqrt[(R_c + i*omega*L_c)*(G_c + i*omega*C_c)]
Z_char(omega) = sqrt[(R_c + i*omega*L_c)/(G_c + i*omega*C_c)]

Z_line_in = Z_char * [Z_terminal + Z_char*tanh(gamma*length)]
                      / [Z_char + Z_terminal*tanh(gamma*length)]
```

Use the passive propagation convention, with attenuation nonnegative. Resistance and conductance are per length, not lumped values. The square-root branch, units, and voltage convention belong in the implementation contract.

In the ideal lossless limit, `gamma = i*beta` and `beta = omega/v_phase`. At the reference frequency,

```text
length = n*lambda0/2  -> beta0*length = n*pi
                     -> Z_line_in = Z_terminal.
```

At a quarter wave, the independent check is `Z_line_in = Z_char**2/Z_terminal`. At zero length, the input equals the load. These are useful tests of any circuit implementation.

The half-wave identity is not exact throughout a sweep: the physical cable length stays fixed while `beta(omega)` changes. Loss and detuning therefore reshape the measured background. A fitted cubic cannot enforce these transmission-line relationships.

### Tuning and electrical loading

The tuning capacitor contributes `Z_cap = 1/(i*omega*C_tune)`. Series loss, the transformed coil impedance, and the tuning capacitor form a resonant leg. The source and amplifier load that leg.

For the source/loading representation of Fig. 1 and Eq. (2), an independently useful derivation is

```text
I0 = U/R0
Y_load = 1/R0 + 1/R_input
u = I0 * Z_resonator/(1 + Y_load*Z_resonator).
```

Taking the real part for real `Y_load` yields Eq. (2):

```text
Re(u) = (U/R0) * [Re(Z) + Y_load*abs(Z)**2]
                  / [(1 + Y_load*Re(Z))**2 + (Y_load*Im(Z))**2].
```

This derivation specifies where source and amplifier loading enter. Do not cascade a loaded-impedance formula with another loading factor without checking whether it counts the same component twice.

**Unresolved transcription issue:** Eq. (3) includes a tuning-capacitor term inside `Z_T`; Eq. (6) includes a tuning-capacitor term again outside `Z_T`. The supplied circuit implementation instead uses its `ZT` name for the transformed cable load alone. These definitions need reconciliation before claiming a literal implementation of both printed equations. See the audit; no circuit topology is silently selected here.

### Detection phase and output

Reference: [Eqs. (9)–(14), p. 5](https://arxiv.org/pdf/2603.10146v5#page=5).

The observed phase-sensitive output has the form

```text
V_detector = G_detector * Re[I0*Z_effective*exp(i*phi_det)] + V_DC.
```

Here `G_detector` and `V_DC` are explicit instrumental/conversion terms, not universal constants. The paper uses a slowly varying phase plus a trim-dependent contribution. A numerically convenient, explicitly centered representation is

```text
phi_det(omega) = phi0 + phi1*(omega-omega0) + phi2*(omega-omega0)**2.
```

That centered form is a reparameterization, not an additional claim from the paper. Coefficients must have the corresponding inverse-frequency units. Phase tuning acts on the **complex response**; it can mix absorption and dispersion. It is not generally represented by multiplying an absorption curve by an independently sampled real straight line.

Keep phase-sensitive and diode/magnitude readout distinct. Figure 5 uses diode-mode examples; these do not define the same observable as `Re(V)`. Document the recorded channel before reproducing a figure or fitting data.

## 5. Building a spectrum without double-counting its physics

Reference: [§2.1, Eq. (20), and Eqs. (32)–(34)](https://arxiv.org/pdf/2603.10146v5#page=10).

For a full susceptibility-coupled response, use the conceptual decomposition

```text
B(f; theta) = detector_output(chi=0, theta)
S(f; P, theta) = detector_output(chi(P), theta) - B(f; theta)
V_measured(f) = B(f; theta) + S(f; P, theta) + noise(f).
```

The same instrument configuration `theta` appears in both terms. This gives a clear definition of baseline, nuclear response, and readout channel.

The paper also describes an additive benchmark generator: a circuit-derived Q-curve plus a scaled physical lineshape and noise. That is distinct from propagating the full complex susceptibility through every component. For that construction, the signal scale and distortions still need experimental support.

Do not both insert susceptibility into the circuit and add the same nuclear signal again. Conversely, do not describe an absorption-only additive model as a full complex circuit simulation. When using a first-order approximation, state it as such and verify it over the intended operating range.

## 6. Spin-1 populations and the Pake doublet

Reference: [§3.1, Eqs. (15)–(26), pp. 7–8](https://arxiv.org/pdf/2603.10146v5#page=7); [Dulya et al. (1997)](https://doi.org/10.1016/S0168-9002(97)00317-3) is the paper's underlying lineshape reference.

For normalized populations,

```text
n_plus + n_zero + n_minus = 1
P = n_plus - n_minus
Q_tensor = 1 - 3*n_zero

n_plus  = (2 + Q_tensor + 3*P)/6
n_zero  = (1 - Q_tensor)/3
n_minus = (2 + Q_tensor - 3*P)/6.
```

The last three expressions are algebraic inversions, useful for checking physical labels. Nonnegative populations imply `-2 <= Q_tensor <= 1` and `3*abs(P) <= 2 + Q_tensor`.

For the spin-temperature/Boltzmann model used for vector extraction,

```text
Q_tensor = 2 - sqrt(4 - 3*P**2)
r_population = I_plus_area/I_minus_area
P = (r_population**2 - 1)/(r_population**2 + r_population + 1).
```

The ratio refers to transition-family **areas**, not automatically to the heights of two overlapping extrema. Instrumental false asymmetry can imitate a change in that ratio. Independent tensor enhancement breaks the one-parameter Boltzmann restriction and needs a different labeling/modeling problem.

### Powder kernel and broadening

Use `R_detuning = (omega - omega0)/(3*omega_Q)`. The two transition branches are indexed by `eps = +1, -1`. The branch shape comes from the distribution of quadrupolar orientations and broadening, not from two freely chosen Gaussian peaks.

In the axial, unbroadened limit, the `+1` branch has support from −2 to +1 and a horn at +1; the other is reflected. Finite broadening rounds the singularities and extends the tails. Nonzero EFG asymmetry requires treatment of azimuthal orientation.

To avoid ambiguous square-root names, the supplied kernel can be written using

```text
b = 1 - eps*R_detuning - eta_efg*cos(2*phi_efg)
rho = sqrt(A**2 + b**2)
c = sqrt(rho)
Y = sqrt(3 - eta_efg*cos(2*phi_efg))
alpha = atan2(A, b)

F = {2*cos(alpha/2)*[pi/2 + atan((Y**2-rho)/(2*Y*c*sin(alpha/2)))]
     + sin(alpha/2)*log[(Y**2+rho+2*Y*c*cos(alpha/2))
                       /(Y**2+rho-2*Y*c*cos(alpha/2))]} / (2*pi*c).
```

This expression is independently checked by the convolution integral

```text
F = (2*A/pi) * integral_0^Y dy / [(y**2 - b)**2 + A**2].
```

The tutorial's existing branch function includes a factor `1/10`; that overall factor cancels in its normalized doublet. Its powder average uses a weighted azimuthal mean with weight `sqrt(3/(3-eta_efg*cos(2*phi_efg)))`. Those are implementation conventions, not additional fitted experimental inputs.

**Do not silently copy the printed X notation:** the typeset v5 Eqs. (16)–(19) use X in a way that does not map consistently onto `rho` and `c` above. The audit records the issue. The existing kernel is supported by the supplied code and independent integration; this is not a claim that the inconsistency in the printed definitions has been resolved with the authors.

### Single-site versus material-specific spectra

A single-site model is the benchmark starting point. Multiple chemically inequivalent sites, different splittings and broadenings, contaminant resonances, and RF-modified populations require explicit components and supported weights. A d-butanol spectrum with O–D structure cannot be validated solely by matching a single ideal doublet.

For cubic or approximately cubic environments with unresolved quadrupole structure, do not force a Pake shape onto the data. The paper's [§3.2](https://arxiv.org/pdf/2603.10146v5#page=8) uses Voigt-type single-line models for the area-oriented problem.

## 7. Thermal equilibrium, calibration, and units

Reference: [§5.1, Eqs. (28)–(34), p. 10](https://arxiv.org/pdf/2603.10146v5#page=10).

An unambiguous way to evaluate the Boltzmann expressions is to use the Zeeman spacing `DeltaE = h*f0` and define `x = DeltaE/(k_B*T)`. Then

```text
P_TE(spin 1/2) = tanh(x/2)
P_TE(spin 1)   = 2*sinh(x)/(1 + 2*cosh(x))
              = 4*tanh(x/2)/(3 + tanh(x/2)**2).
```

Using `f0` avoids inadvertently multiplying by a nuclear g-factor twice when a variable named magnetic moment already includes it. This is a derivation from level populations, not a replacement calibration measurement.

At 1.5 K, evaluating these expressions at 213 MHz for spin 1/2 and 32.7 MHz for spin 1 gives approximately **0.340745%** and **0.069749%**, respectively, consistent with the rounded examples in the paper. These values depend on frequency/field and temperature. A **5%** tutorial example is not a **0.05%** TE example.

The paper's exact area convention is

```text
P = C_cal * integral S(omega)/omega d(omega)
C_cal = P_TE / integral S_TE(omega)/omega d(omega)
S = Re[V(omega, chi) - V(omega, 0)].
```

On changing variable from angular frequency to ordinary frequency, `d(omega)/omega = df/f`. A numerical implementation therefore needs its frequency grid and quadrature weights. For a narrow scan, replacing `1/f` by `1/f0` can be an explicit approximation absorbed into an area calibration; it is not an identity across arbitrary grids and windows.

The current tutorial's `sum(S) = P/cc` is a **discrete fixed-grid normalization**. It has not been shown equivalent to the paper's calibrated voltage-area convention. Changing bin count, sweep width, detector gain, or output units requires checking the conversion. Never label `cc` universally as frequency calibration or amplitude calibration based only on its name.

Calibration uncertainty remains part of the measurement. Improving extraction does not independently determine the TE temperature, coil filling, spatial polarization, or a drifting electronics gain.

## 8. Noise is a measured stochastic process

Reference: [§2.2, pp. 5–7; §6.4, pp. 17–18; Eq. (45)](https://arxiv.org/pdf/2603.10146v5#page=5).

Separate at least these mechanisms:

| Mechanism | How it enters the measurement | Required evidence |
|---|---|---|
| Thermal/electronic noise | Additive fluctuations shaped by filtering and the circuit | Marginal variance and frequency-bin covariance |
| RF pickup, supply ripple, interference | Coherent or quasi-coherent structure | Frequencies, amplitudes, phases, stability across repeated sweeps |
| Cable motion, contacts, microphonics | Changes to impedance and phase | Correlated drift or jump distributions, not merely a wider white-noise term |
| Tuning/temperature drift | Changed circuit response | Joint distributions and time/period dependence of physical parameters |
| Baseline subtraction error | An analysis-induced residual | Matched subtraction tests; not automatically statistical noise |

For a Gaussian part, use `epsilon ~ Normal(0, Sigma_NMR)`. The white-noise special case is `Sigma_NMR = sigma**2 * I`. AR(1) noise is one possible demonstration, not evidence that a measured instrument has AR(1) covariance.

The paper defines

```text
SNR_peak = max(abs(clean nuclear signal))/max(abs(noise realization)).
```

It excludes the baseline. The tutorial previously discussed `max(abs(signal))/noise_SD`, a different quantity. Record both names explicitly if both are needed. Peak-noise SNR depends on the realization and the number/correlation of bins; it cannot be converted by a universal constant.

The paper gives a nominal Gaussian scale of order `1e-6 mV` in its convention. That is `1e-9 V` before any additional calibration/gain scaling. The tutorial's `2.7e-5` generator-unit noise is **not** established as that same noise level. Figure axes involving `C_E` must not be compared as absolute voltages without identifying `C_E`.

Independent repeated-sweep noise averages down as `1/sqrt(N)`. This statement does not apply unchanged to correlated drift or a coherent sinusoid that has the same phase in every sweep: such a sinusoid survives averaging. Repetition statistics and phase coherence must be checked; bin-to-bin covariance and sweep-to-sweep covariance are different objects.

For two independent repeats of the same underlying signal, `(scan1-scan2)/sqrt(2)` estimates single-scan fluctuations. Drift, signal evolution, or shared noise invalidate that simple interpretation. Estimate noise from development measurements, not from a final evaluation set repeatedly consulted during design.

## 9. What training data must demonstrate

Reference: [§5.3 and §6.4](https://arxiv.org/pdf/2603.10146v5#page=11).

The paper's augmentation is parameter-based: vary the circuit and lineshape within experimentally constrained conditions. Merely drawing unrelated random polynomial coefficients, gains, or widths does not establish that coverage.

For a defensible standalone dataset, retain:

- The circuit version, topology, detection channel, frequency grid, units, and sign convention.
- Physical parameters, their joint sampling distribution, provenance, bounds, and any fixed values.
- Lineshape model, material/site assumptions, vector/tensor constraints, and signal normalization.
- Measured noise model and covariance, structured-noise parameters, and averaging conditions.
- Configuration/period identifiers, labels, independent seeds, and partition assignments.
- Overlay and residual checks against development data, plus independent measurements reserved for evaluation.

Vary every relevant encoded parameter, but do not mistake an inactive parameter for useful augmentation. Do not treat repeated copies of a fitted reference configuration as independent configuration coverage. A small fit residual also does not prove that fitted component values describe a passive physical circuit.

The paper reports 500-bin spectra and an 80/10/10 split. The current tutorial uses 512 bins and grouped configurations. These can be separate design choices, but they must not be represented as exact reproduction settings. Noise realizations alone do not protect against leakage across shared physical configurations.

## 10. Networks, stopping, and task boundaries

Reference: [§§5.3, 6–6.4, pp. 12–18](https://arxiv.org/pdf/2603.10146v5#page=12).

The paper separates high-polarization regression, low-polarization regression, area prediction, and denoising. These are different estimands and may need different input scaling and capacity.

- The polarization CNN includes residual connections, multiscale convolutions, and squeeze-and-excitation. The current tutorial's ridge-plus-CNN estimator is a separate architecture, not that paper model.
- The area study motivates simpler estimators when global area is sufficient; adding a large CNN is not automatically useful.
- Low-P extraction relies on small signals and weak asymmetry. A tightly matched voltage scale can improve inference while reducing transfer to other operating conditions.
- A DAE predicts a clean spectrum. It does not directly output calibrated polarization or automatically supply uncertainty.

The paper reports AdamW, hyperparameter searches, cosine warm restarts, and validation-selected checkpoints. The tutorial's scheduler, epoch budget, sample count, and architectures differ. Paper settings are recorded experimental choices, not guaranteed optimal defaults for a new lab.

Preserve optimizer and scheduler state for a true training resume. Select preprocessing, architecture, stopping, and hyperparameters with development/validation data. Then freeze the entire estimator and use independent pseudo-experiments for the final extraction-error study. Do not continue tuning against final test metrics.

A sample-count-to-parameter-count heuristic is not a proof of sufficient training data. Test learning curves versus both event count and independent configuration coverage. Measure end-to-end latency, including preprocessing and transfers, on the intended hardware.

## 11. Bias, precision, relative error, and confidence intervals

Reference: [§5.4, Eqs. (38)–(45), pp. 13–14](https://arxiv.org/pdf/2603.10146v5#page=13).

The paper defines `Delta = P_true - P_pred`. The current tutorial defines `e = P_pred - P_true`. Consequently,

```text
Delta = -e
mu_paper = -bias_tutorial
sigma_paper = width_tutorial
RMSE = sqrt(bias**2 + width**2).
```

The last identity uses population SD (`ddof=0`) and the same events. A positive paper residual means underprediction; a positive tutorial residual means overprediction. Do not reverse only the prose while leaving stored residual columns unchanged.

For fractional-P error `dP`:

```text
absolute percentage-point error = 100*dP
relative error percent at P0    = 100*dP/abs(P0).
```

For example, `dP=0.0005` is `0.05` percentage points and 1% relative at `P0=0.05`. The same absolute error would be 100% relative at `P0=0.0005`. These are unit conversions, not measured performance claims.

A pooled residual width over a wide P range does not establish performance at a chosen P. Use fixed-P trials or a sufficiently narrow signed P bin with relevant operating conditions. Quote event counts and finite-Monte-Carlo uncertainty. Width is not the standard error on mean bias and is not automatically a confidence interval for one prediction.

The paper constructs intervals by inverting conditional residual quantiles for a fixed estimator. In its residual convention, candidate P is accepted when

```text
q_alpha/2(P) <= P - P_pred_observed <= q_1-alpha/2(P).
```

Locally Gaussian residuals give `P_pred + mu_paper +/- z*sigma`, equivalently `P_pred - bias_tutorial +/- z*width`. This is a local approximation requiring validated calibration; heteroscedastic or non-Gaussian residuals require conditional quantiles and coverage checks.

## 12. Denoising uncertainty and what improved fits do not establish

Reference: [§7.4, pp. 22–23; Appendix A, pp. 26–27](https://arxiv.org/pdf/2603.10146v5#page=26).

Hold the DAE fixed and propagate replicas drawn from the experimental input-error model. Each replica yields a reconstructed spectrum. The ensemble provides a mean, pointwise SD, and a full output covariance matrix. Appendix A uses sample SD (`N_MC-1`), unlike the population residual width used for the scalar benchmark identity.

A pointwise one-SD band is not a simultaneous confidence band over all frequency bins. Smoothing also induces output-bin correlations; fitting a denoised curve as if its bins had independent original noise can make fit uncertainties misleadingly small.

Pass each reconstructed replica through the fixed downstream estimator to propagate the same input uncertainty through the whole chain. Reconstruction bias and simulation mismatch remain separate contributions. Resampling around an already noisy observation is an uncertainty-propagation prescription, not evidence that the unknown clean curve has been recovered without bias.

The experimental comparisons in the paper are useful consistency checks, but do not provide exact truth or independently validate a much smaller extraction uncertainty. Displaced-coil data are explicitly a different coupling/noise condition requiring dedicated quantitative validation. The paper's concluding deployment proposal includes prospective comparisons and coverage checks; the tutorial must not imply those have already been completed here.

## 13. Error-budget boundary and implementation gate

Reference: [§7.5 and §9, pp. 23–25](https://arxiv.org/pdf/2603.10146v5#page=23).

Extraction error is only part of total polarimetry uncertainty. TE calibration, temperature, field homogeneity, Q-meter nonlinearity, coil coupling, target polarization gradients, and beam-weighted versus coil-weighted sampling do not disappear when a synthetic-data residual becomes small. The quoted hardware accuracy is specific to the instrument context, not a mathematical lower bound for every NMR apparatus.

Before replacing the tutorial generator or publishing new physics-performance results:

1. Resolve the circuit topology and unit issues in the audit with an authoritative reference configuration.
2. Implement a self-contained physical circuit with documented units and detector channel.
3. Verify passive-circuit limits, phase behavior, normalization, and source-code comparisons.
4. Fit/check representative measured baselines and signal-to-baseline scales; establish noise statistics separately.
5. Define supported joint parameter variation and keep independent evaluation conditions aside.
6. Regenerate data, retrain, and republish only results carrying the new full-model provenance.

Do not fill a missing calibration, component value, covariance, or convention with a plausible-looking number and call it experimentally grounded.
