# NMR / AI

Learn **simulation-based inference** of vector polarization from NMR spectra
with Python. Fit measured data, generate labeled spectra, train a neural network,
and inspect its errors.
Basic Python and familiarity with arrays are enough to begin.

**[Start the step-by-step tutorial →](https://zetanaut.github.io/NMR-AI/walkthrough.html)**

The tutorial gives commands in order, expected outputs, and plots to check at
each step. All data needed for the examples are included. You train your own
model; no pretrained download is required.

## What kind of model is this?

These are **polarization regression models**: each spectrum produces one signed
value, the vector polarization P. The main walkthrough uses a multiscale
convolutional neural network (CNN); simpler dense networks provide comparisons.
A denoising autoencoder (DAE) instead reconstructs a cleaner spectrum for further
analysis. Read [how the 1D CNN works](https://zetanaut.github.io/NMR-AI/#multiscale-cnn):
shared filters, pooling, the exact architecture, and why it includes spectral moments.

Experimental polarization estimates can have roughly 5% relative uncertainty in
the motivating measurements. We therefore use measured spectra and known setup
information to constrain a physical simulator, then train on new spectra with
simulator-known P labels. Experimental fit estimates are not treated as exact
training truth. The simulator carries the supplied spin and circuit constraints;
its fidelity still needs experimental validation.

The aim is to reduce extraction error compared with conventional fitting.
Calibration and instrument uncertainties remain, including the approximately
0.8% relative Q-meter design limit specified for the systems motivating this
tutorial. Read the [model introduction](https://zetanaut.github.io/NMR-AI/#inference-method)
for the evidence, uncertainty scope, and distinction between the tutorial's
results and the reference paper's comparisons.

## Set up

Use Python 3.10 or newer and a Bash terminal on Linux, macOS, or Windows with WSL.
A CPU is supported; a compatible GPU can speed up training.

The requirements command below installs **PyTorch** (`torch`), NumPy, SciPy,
and Matplotlib. For a CPU-only or GPU-specific PyTorch build, use the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
Choose **Stable**, **Pip**, **Python**, your OS (**Linux** inside WSL), and your
compute platform. Run its command after activating `.venv` and before installing
the requirements below, replacing `pip` or `pip3` with `python -m pip` to use
the active environment.

```bash
git clone https://github.com/zetanaut/NMR-AI.git
cd NMR-AI
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -c "import numpy, scipy, torch, matplotlib; print('Imports OK'); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

You should see `Imports OK` and the installed PyTorch version. `CUDA: False`
is valid for CPU use; if you intend to use an NVIDIA GPU, check the official
installation instructions until `CUDA: True` is reported. Run subsequent commands
from the repository root. In each new terminal, activate the environment and
repeat the two thread exports.

## Make your first plot

```bash
python tools/preview_baseline.py --output local-results/baseline-input.svg
```

Open `local-results/baseline-input.svg` in your browser. The command reports
500 samples and 35 decimal-spacing repairs made while reading the supplied CSV.
The file itself is preserved.

**Next: [fit the baseline in tutorial step 3](https://zetanaut.github.io/NMR-AI/walkthrough.html#baseline).**
Keep the tutorial's output paths when following its remaining commands.

## What you will learn

| Steps | Exercise | What you will produce |
| --- | --- | --- |
| [1–3](https://zetanaut.github.io/NMR-AI/walkthrough.html#setup) | Set up, inspect data, fit the circuit baseline | A measured-data plot and a baseline fit |
| [4–6](https://zetanaut.github.io/NMR-AI/walkthrough.html#single-site) | Fit butanol with one and two sites, then fit UVA-ND3 | Raw sweeps, fitted responses, and residual plots |
| [7](https://zetanaut.github.io/NMR-AI/walkthrough.html#generate) | Generate new spectra from the single-site fits | 2,000 examples with simulator-known polarization |
| [8–9](https://zetanaut.github.io/NMR-AI/walkthrough.html#train) | Train and evaluate a network | A saved model and 400 held-out predictions |

The two butanol fits use the same measurements to compare models. UVA-ND3 is a
separate acquisition. All three fit raw sweeps with their baseline present.
Every working example and network uses the same 500-bin frequency grid.
See the [data guide](examples/README.md) for the files and readers.

Fitting these spin-1 lineshapes does not require thermal-equilibrium (TE)
calibration. The network lesson uses raw sweeps and corresponding independent
baseline references. Its reported errors measure recovery of simulated
polarization; experimental polarization accuracy remains to be established.

The optional TE-area benchmark uses an area-to-polarization scale set by an ideal
TE reference. That map can apply at enhanced polarization, and a suitable known-P
reference need not be at lattice thermal equilibrium. Boltzmann spin populations
and a validated calibration response are distinct requirements. See the
[calibration explanation](https://zetanaut.github.io/NMR-AI/#area-calibration).

For your own training study, see the [training-data design guide](docs/reference/training-design.md):
low-polarization coverage, non-uniform sampling, and choosing dataset size from
validation evidence rather than a fixed examples-per-weight rule.

## Find your way around

| Location | Use it for |
| --- | --- |
| [Tutorial website](https://zetanaut.github.io/NMR-AI/) | Physics, network explanations, and worked results |
| [examples/](examples/README.md) | The three input files used in the tutorial |
| [tools/](tools/README.md) | Python commands and the models behind them |
| [configs/](configs/README.md) | Frequency grid, circuit assumptions, and exercise settings |
| [docs/reference/](docs/reference/README.md) | Detailed derivations, sources, and optional benchmark exercises |
| `local-results/` | Your generated plots, data, and models; created as you work |

For a new run, choose fresh output paths consistently. Existing results are
protected from accidental overwrite. If imports fail, activate the environment
and check `python -c "import sys; print(sys.executable)"` before reinstalling.

To browse the lessons offline after cloning:

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

Open [the local walkthrough](http://127.0.0.1:8000/walkthrough.html).
For tests and website updates, see the [contributor guide](CONTRIBUTING.md).
