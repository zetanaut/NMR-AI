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
presets distinct from experimental fits. All tutorial work concerns deuterons
near 32.7 MHz, including the supplied baseline CSV. The owner confirmed the
500-bin acquisition: preserve the original 0.0015287 MHz spacing and offsets
but replace the nominal 213 MHz center with 32.7 MHz. Thus f_j = 32.3 +
0.0015287*j MHz, j=0..499, ending at 33.0628213 MHz. This is recorded in
`configs/deuteron-acquisition.json`; do not recenter a new linspace on 32.7 MHz.
Keep this measured 500-bin contract distinct from the existing 512-bin synthetic
training benchmark; changing that generator requires new data and training.
A baseline fit alone does not establish signal calibration or parameter distributions.
The owner explicitly authorized publishing `examples/deuteron-baseline.csv`
as a student test example. Use the shared `baseline_data.py` loader to report its
decimal-spacing repairs; do not silently rewrite values or the confirmed grid.

NMR-AI is a standalone educational repository. Do not add runtime dependencies,
imports, links, workflow integrations, or automatic data/checkpoint transfers to
the separate research pipeline. Do not modify that pipeline for tutorial work.

When the full generator changes, regenerate data and retrain before updating
results. A lineshape version tag or passing plot regression test does not establish
full instrument validity. Record units, residual sign, SNR definition, calibration,
source provenance, limitations, and independently verified numerical checks.
