# Scientific-source requirements

Before changing or describing the physics, electronics, simulator, preprocessing,
or performance claims, read these files completely:

- `notes/physics-electronics-theory.md`
- `notes/implementation-audit.md`

The user's requirement is to follow the real physics and supplied theory, not
invent plausible-looking substitutes. Keep paper statements, mathematical
derivations, observed implementation behavior, and unresolved conventions distinct.

The current polynomial-baseline generator is an unvalidated software prototype,
not the Q-meter model in arXiv:2603.10146v5. A polynomial wing-subtraction model
must not be presented as the physical baseline generator. Do not invent component
values, units, phase conventions, calibrations, noise covariance, or empirical
parameter distributions. Resolve the audit's relevant open questions before
implementing or publishing replacement physics and new performance claims.

NMR-AI is a standalone educational repository. Do not add runtime dependencies,
imports, links, workflow integrations, or automatic data/checkpoint transfers to
the separate research pipeline. Do not modify that pipeline for tutorial work.

When the full generator changes, regenerate data and retrain before updating
results. A lineshape version tag or passing plot regression test does not establish
full instrument validity. Record units, residual sign, SNR definition, calibration,
source provenance, limitations, and independently verified numerical checks.
