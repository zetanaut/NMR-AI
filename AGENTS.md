# Scientific-source requirements

Before changing or describing the physics, electronics, simulator, preprocessing,
or performance claims, read these files completely:

- `notes/physics-electronics-theory.md`
- `notes/baseline-fitting.md`

The user's requirement is to follow the real physics and supplied theory, not
invent plausible-looking substitutes. Keep paper statements, mathematical
derivations, observed implementation behavior, and unresolved conventions distinct.

The current generator uses complex spin-1 susceptibility and the physical
Fig. 1 / Eq. (2) circuit. Preserve its single-capacitor topology, explicit units,
finite-source loading, passive cable, and phase-sensitive detector convention.
Do not invent component measurements, voltage conversions, calibration, noise
covariance, or empirical parameter distributions. Keep controlled sensitivity
presets distinct from experimental fits. The 213 MHz baseline data do not
establish 32.7 MHz deuteron calibration or parameter distributions.

NMR-AI is a standalone educational repository. Do not add runtime dependencies,
imports, links, workflow integrations, or automatic data/checkpoint transfers to
the separate research pipeline. Do not modify that pipeline for tutorial work.

When the full generator changes, regenerate data and retrain before updating
results. A lineshape version tag or passing plot regression test does not establish
full instrument validity. Record units, residual sign, SNR definition, calibration,
source provenance, limitations, and independently verified numerical checks.
