# Reference and further exercises

Start with the [guided tutorial](https://zetanaut.github.io/NMR-AI/walkthrough.html)
for commands and expected outputs. These pages supply the detailed mathematics,
assumptions, and sources behind the lessons.

| Topic | Read when you want to… |
| --- | --- |
| [Physics and electronics](physics-electronics-theory.md) | Derive the circuit, susceptibility, calibration, and metric conventions |
| [Baseline fitting](baseline-fitting.md) | Understand the cable constraint, fitted parameters, and residuals |
| [Experimental matching](experimental-matching.md) | Understand lineshape fitting and generation from measured configurations |
| [Butanol and UVA-ND3](material-examples.md) | Compare material models and inspect the ND3 interpolation and circuit assumptions |
| [Training-data design](training-design.md) | Plan low-polarization coverage, non-uniform sampling, dataset size, and controlled training comparisons |
| [Model comparison](model-comparison.md) | Inspect the MLP, DNN, CNN, and their saved results |
| [Optional benchmarks](benchmarks.md) | Reproduce the controlled TE-area learning exercises |
| [Uncertainty and distribution shift](distribution-shift.md) | Study the proposed validation work beyond synthetic test scores |

The experiment-based tutorial and the optional TE-area benchmark have distinct
input channels and calibration assumptions. Both use the exact 500-bin grid.
Keep each dataset with its own saved preprocessing and model.

For repository tests or regenerating the website's figures, use the
[contributor guide](../../CONTRIBUTING.md).
