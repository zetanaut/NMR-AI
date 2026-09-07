# Public deuteron teaching examples

[deuteron-baseline.csv](deuteron-baseline.csv) is the measured baseline supplied
by the repository owner for public student use. It is independent of any research
software checkout. The original fields and spacing are preserved; only a final
newline was added to the supplied file.

- One headerless record: timestamp, followed by 500 recorded amplitudes.
- Nucleus: deuteron; owner-confirmed nominal center: 32.7 MHz.
- Frequency: `32.3 + 0.0015287*j MHz`, j = 0…499, preserving the original
  proton sweep offsets and spacing with the owner's corrected center.
- First/last sample: 32.3000000 / 33.0628213 MHz. There are 499 intervals
  between 500 samples, each 1.5287 kHz. Do not substitute a symmetric linspace.
- Contract: [`deuteron-acquisition.json`](../configs/deuteron-acquisition.json).
- Recorded-unit to volts conversion: **not supplied**.
- 35 numeric fields contain spaces immediately before their decimal point.
  The loader repairs that specific formatting pattern in memory and reports
  each repair. It does not smooth, discard, interpolate, or rescale samples.

Published CSV SHA-256:
`dac7c4598c6ec32250ab763ab1bf99e2a7aa7346de4e452ce00e80f0e2b1eb28`.
Original file SHA-256, before the final newline:
`cc24ece801180ac3113052610042bf8228a97fabbfe8350fdc7aabbb5b4c48e8`.

## Load it from the repository root

```python
import sys
sys.path.insert(0, "tools")
from baseline_data import load_baseline_csv, load_acquisition, acquisition_grid

records, parsing = load_baseline_csv("examples/deuteron-baseline.csv")
timestamp = records[0, 0]
amplitudes = records[0, 1:]
frequency_hz = acquisition_grid(load_acquisition())
assert amplitudes.shape == (500,)
assert parsing["decimal_whitespace_repair_count"] == 35
```

Ordinary `numpy.loadtxt` rejects the embedded spaces; use the shared loader in
both fitting programs. Ambiguous whitespace such as `1 2` is rejected, not
silently converted to `12`. The CSV is never rewritten by the loader.

## Preview and fit

Run `python tools/preview_baseline.py` to reproduce the website's measured-data
preview against the confirmed frequency axis. It is a measured-only plot.

To reproduce the displayed physical-circuit comparison:

```bash
python tools/fit_baseline.py examples/deuteron-baseline.csv \
  --starts 24 --output-dir local-results/deuteron-tuned-baseline-fit
```

It uses the confirmed acquisition and hardware preset automatically:
[deuteron-baseline-setup.json](../configs/deuteron-baseline-setup.json) fixes n=1
and λ/2=3.580 m at 32.7 MHz. The provisional ±3% length trim is a working bound,
not a measured uncertainty. Cable L/C are derived consistently from that known
propagation scale with explicit nominal impedance/loss assumptions.

The constrained fit reaches trim and stray-capacitance limits and has correlated
residuals; it is not an accepted hardware calibration. Add actual capacitor,
coil, loss and detector records before treating fitted values as physical settings.
For this setup, copy and refine the preset; the blank template is for other setups.
See the [fitting procedure](../notes/baseline-fitting.md) for diagnostics, parameter
meanings, and publication.

## Polarized spin-1 raw signals

[Sample_RawSignal.csv](Sample_RawSignal.csv) is also explicitly authorized for
public student use. Its bytes are preserved exactly, including five UTF-8
U+2028 line separators and blank lines. SHA-256:
`cdbb7e3afa6531694a4b97848d295bbb5c7c03ef62d796b053e3b4e16fbaea5a`.

There are five distinct timestamp-plus-500-sample records. They use the same
confirmed frequency grid and n=1, 3.580 m cable half-wave setup as the baseline.
Timestamps are not in increasing order; file order is retained. The file has
no polarization or frequency columns. Polarization is determined by matching
the spin-1 lineshape; TE calibration is not needed for that extraction.

Use the UTF-8-aware, audited reader:

```python
from experimental_data import load_signal_csv

signals, audit = load_signal_csv("examples/Sample_RawSignal.csv")
assert signals.shape == (5, 501)
assert len(audit["unicode_line_separators"]) == 5
assert len(audit["blank_logical_lines_1based"]) == 18
```

It reuses the strict numeric grammar from `baseline_data.py`, but handles this
file's record separators explicitly. It never removes a separator inside an
otherwise invalid number to invent a new value. No records are deduplicated,
sorted or smoothed. The original single-baseline parser remains unchanged.

See [Practical 01B](../docs/matching.html) and the
[experimental matching record](../notes/experimental-matching.md) for full
signal/circuit fits, noise diagnostics and a 500-bin generator that samples new
simulator-known polarizations around fitted experimental configurations.
