# Public deuteron baseline example

[deuteron-baseline.csv](deuteron-baseline.csv) is the measured baseline supplied
by the repository owner for public student use. It is independent of any research
software checkout. The original fields and spacing are preserved; only a final
newline was added to the supplied file.

- One headerless record: timestamp, followed by 500 recorded amplitudes.
- Nucleus: deuteron; approximate operating frequency: 32.68 MHz.
- Actual first/last sample frequencies and bin spacing: **not supplied**.
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
from baseline_data import load_baseline_csv

records, parsing = load_baseline_csv("examples/deuteron-baseline.csv")
timestamp = records[0, 0]
amplitudes = records[0, 1:]
assert amplitudes.shape == (500,)
assert parsing["decimal_whitespace_repair_count"] == 35
```

Ordinary `numpy.loadtxt` rejects the embedded spaces; use the shared loader in
both fitting programs. Ambiguous whitespace such as `1 2` is rejected, not
silently converted to `12`. The CSV is never rewritten by the loader.

## Preview and fit

Run `python tools/preview_baseline.py` to reproduce the website's measured-data
preview against sample index. It is not a fit and assumes no frequency mapping.

For circuit fitting, first obtain the actual acquisition grid and complete
`configs/baseline-setup.template.json` with independently known deuteron tuning
information. The approximate center frequency does not establish the width.
See the [baseline-fitting procedure](../notes/baseline-fitting.md).
