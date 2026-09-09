# Example data

Use these files with the commands in the
[step-by-step tutorial](https://zetanaut.github.io/NMR-AI/walkthrough.html).
They are included and ready to load.

| File | Contains | Reader | Tutorial step |
| --- | --- | --- | --- |
| [deuteron-baseline.csv](deuteron-baseline.csv) | One baseline sweep | `baseline_data.load_baseline_csv` | [3: baseline fit](https://zetanaut.github.io/NMR-AI/walkthrough.html#baseline) |
| [Sample_RawSignal.csv](Sample_RawSignal.csv) | Five raw butanol sweeps | `experimental_data.load_signal_csv` | [4–5: single-site and two-site fits](https://zetanaut.github.io/NMR-AI/walkthrough.html#single-site) |
| [uva-nd3.json](uva-nd3.json) | Five raw UVA-ND3 sweeps | `uva_nd3_data.load_nd3` | [6: ND3 fit](https://zetanaut.github.io/NMR-AI/walkthrough.html#nd3) |

Every working sweep has 500 bins on this frequency grid:

```text
frequency_mhz[j] = 32.3 + 0.0015287*j, j = 0..499
first = 32.3000000 MHz; last = 33.0628213 MHz
```

The nominal deuteron frequency is 32.7 MHz. The actual grid is defined in
[deuteron-acquisition.json](../configs/deuteron-acquisition.json); use its spacing
and offsets. Amplitudes are in recorded units. A conversion to volts has not
been supplied.

## Load the examples

Run this Python code from the repository root. The fitting commands use these
same readers automatically.

```python
import sys
sys.path.insert(0, "tools")
from baseline_data import load_baseline_csv, load_acquisition, acquisition_grid
from experimental_data import load_signal_csv
from uva_nd3_data import load_nd3

baseline, baseline_audit = load_baseline_csv("examples/deuteron-baseline.csv")
butanol, butanol_audit = load_signal_csv("examples/Sample_RawSignal.csv")
nd3, nd3_audit = load_nd3("examples/uva-nd3.json")
frequency_mhz = acquisition_grid(load_acquisition()) / 1e6

print(baseline[:, 1:].shape)        # (1, 500); column 0 is the timestamp
print(butanol[:, 1:].shape)         # (5, 500); column 0 is the timestamp
print(len(nd3["records"]))          # 5; each record has 500 phase values
print(baseline_audit["decimal_whitespace_repair_count"])  # 35
```

The baseline CSV contains spaces before some decimal points. Its reader repairs
that formatting in memory and reports every repair. The butanol reader handles
UTF-8 record separators and preserves the original record order. Use these
readers instead of a generic CSV importer.

For UVA-ND3, fit `record["phase"]`: it already includes the baseline. The stored
`baseline` and `basesub` fields describe the recorded reference and its
subtraction; they are excluded from the raw-sweep fit. The working arrays were
linearly interpolated onto the teaching grid. The reader verifies that
derivation, and the fitter applies the same interpolation to its model.
Interpolation correlates neighboring errors. ND3 hardware values are explicit
tutorial assumptions, separate from the confirmed butanol setup.

## From fits to training data

The supplied signal files have no independently established polarization labels.
Their fits estimate polarization from the spin-1 lineshape. Tutorial step 7
uses the single-site fitted configurations to generate **new** spectra with
simulator-known labels, which are used for training and evaluation.

Source details and numerical checks are in the reference pages for the
[baseline](../docs/reference/baseline-fitting.md),
[butanol sweeps](../docs/reference/experimental-matching.md), and
[material models](../docs/reference/material-examples.md).
Save your generated files in `local-results/`.
