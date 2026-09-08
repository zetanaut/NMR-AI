# Session startup

Read `notes/project-status.md` first for the handoff, verified local Python
interpreter, quick checks, saved artifacts, and next implementation milestone.
Use that interpreter for local work. Check the selected environment before
reporting missing dependencies or installing packages; the default system
Python does not have PyTorch, but the documented Conda environment does.
Shell activation does not persist between tool calls: use the absolute
interpreter path or redeclare the documented variable in each shell.
Check `git status --short --branch` and preserve existing uncommitted work.
Keep `notes/project-status.md` current when the environment, validation results,
authoritative artifact paths, or next milestone changes; date the evidence.

# Scientific-source requirements

Before changing or describing the physics, electronics, simulator, preprocessing,
or performance claims, read these files completely:

- `notes/physics-electronics-theory.md`
- `notes/baseline-fitting.md`
- `notes/experimental-matching.md` when working on experimental signal matching
  or the new 500-bin experiment-anchored generator.
- `notes/material-examples.md` when working on the butanol or UVA-ND3 demonstrations.

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
For the supplied baseline the owner confirmed n=1 and cable half-wavelength
3.580 m at nominal 32.7 MHz. Use configs/deuteron-baseline-setup.json: consistent
RLGC propagation (phase velocity factor 0.7809803) and only a +/-3% working trim
bound, explicitly not a measured standard uncertainty. Never widen the branch or
silently reuse incompatible cable L/C to improve the residual. Boundary pressure
and other nominal component assumptions must remain visible. This measured setup
is distinct from the existing synthetic teaching benchmark.
The owner explicitly authorized publishing `examples/deuteron-baseline.csv`
as a student test example. Use the shared `baseline_data.py` loader to report its
decimal-spacing repairs; do not silently rewrite values or the confirmed grid.
The five raw spin-1 sweeps in examples/Sample_RawSignal.csv are also authorized
for publication, byte-for-byte including UTF-8 record separators. Their grid and
n=1, 3.580 m half-wave setup are confirmed to match the baseline. Use the audited
experimental_data.py reader. Polarization is inferred by spin-1 lineshape/branch
ratio fitting; TE calibration or externally supplied P is NOT a prerequisite.
Fit the tuning capacitor and detector phase rather than treating earlier fit
estimates as independently known measurements. Keep experimental P fit estimates
distinct from newly sampled simulator truth labels and from accuracy claims.
The 500-bin lineshape generator has no TE-area input requirement and is separate
from the existing 512-bin TE-area benchmark. Do not conflate their preprocessing.

The owner identified Sample_RawSignal.csv as butanol. Preserve the original
single-site example and compare the supplied C-D/O-D theory on the same data.
The authorized UVA-ND3 excerpt in examples/uva-nd3.json is a different acquisition:
retain its 512 measured frequency values, raw phase and recorded baseline.
The 500-bin and n=1, 3.580 m setup statements above apply to the original baseline
and butanol sweeps, not this ND3 acquisition. ND3 hardware calibration is unresolved;
do not borrow the butanol setup or treat stored acquisition polarization as truth.
Use only the public name UVA-ND3 data. The new material fit demos do not by
themselves revise either generator or establish material-specific network accuracy.

NMR-AI is a standalone educational repository. Do not add runtime dependencies,
imports, links, workflow integrations, or automatic data/checkpoint transfers to
the separate research pipeline. Do not modify that pipeline for tutorial work.

When the full generator changes, regenerate data and retrain before updating
results. A lineshape version tag or passing plot regression test does not establish
full instrument validity. Record units, residual sign, SNR definition, calibration,
source provenance, limitations, and independently verified numerical checks.
