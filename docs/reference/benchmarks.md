# Optional TE-area benchmarks

These exercises compare networks on controlled simulated spectra. Complete the
[guided tutorial](https://zetanaut.github.io/NMR-AI/walkthrough.html) first if
you want to begin with measured inputs.

| Learning path | Network channels | Groups kept together |
| --- | --- | --- |
| Guided tutorial | Raw and reference-subtracted recorded units; no TE calibration | Each measured source and its simulated descendants |
| TE-area benchmarks here | Raw voltage and TE-calibrated reference-subtracted area contributions | Each simulated circuit configuration |

Both use the exact 500-bin grid. Their units, calibration assumptions, and saved
preprocessing differ. The [physics reference](physics-electronics-theory.md)
derives the full circuit and the TE area calibration.

TE names the ideal reference used in these saved runs; their training spectra
already span enhanced polarization. More generally, a suitable known-P reference
can set the area scale at enhanced polarization too. Spin Boltzmann equilibrium
is distinct from equilibrium with the lattice, and does not by itself validate
calibration transfer through a changing or nonlinear detector response. Read the
[area-calibration explanation](physics-electronics-theory.md#area-calibration-beyond-a-thermal-reference)
before changing a reference procedure or applying a saved checkpoint.

## Compare an MLP, DNN and CNN

Use the [README setup](../../README.md#set-up), then run these commands from the
repository root. Generate the broad dataset once; use a fresh model directory
for each comparison.

```bash
python tools/generate_data.py --num-samples 6000 --num-configurations 300 \
  --coverage broad --output local-results/qmeter-500-v1/broad.npz
python tools/run_model_comparison.py --runs-dir local-results/my-comparison
```

The runner reads [model-comparison.json](../../configs/model-comparison.json).
It trains all three models on the same dataset and partition, selects checkpoints
on validation data, then evaluates the held-out test rows. Use `--stage train`
and later `--stage evaluate` with the same output directory to review validation
before testing.

Check the `mlp/`, `dnn/`, and `physics_multiscale/` folders for histories,
checkpoints, preprocessing, partitions, and predictions. The
[model-comparison reference](model-comparison.md) explains the architectures and
saved scores. The published test RMSEs are about 0.06447, 0.03406, and 0.00258
percentage points, respectively. These are one-seed results with an 80-epoch
cap; the CNN also includes a fitted ridge estimate.

## Reproduce the additional CNN runs

The [benchmark lesson](https://zetanaut.github.io/NMR-AI/#results) also compares
three runs with a 40-epoch cap. Generate the narrow dataset below. If you have
not generated the broad dataset, use the command in the preceding section.

```bash
python tools/generate_data.py --num-samples 6000 --num-configurations 300 \
  --coverage narrow --output local-results/qmeter-500-v1/narrow.npz
python tools/train_model.py --data local-results/qmeter-500-v1/narrow.npz \
  --output-dir local-results/qmeter-500-v1/narrow_multiscale --architecture physics_multiscale \
  --epochs 40 --patience 8 --learning-rate 0.0001
python tools/train_model.py --data local-results/qmeter-500-v1/broad.npz \
  --output-dir local-results/qmeter-500-v1/broad_multiscale --architecture physics_multiscale \
  --epochs 40 --patience 8 --learning-rate 0.0001
python tools/train_model.py --data local-results/qmeter-500-v1/broad.npz \
  --output-dir local-results/qmeter-500-v1/broad_compact --architecture compact \
  --epochs 40 --patience 8 --learning-rate 0.0001
```

Each dataset has 6,000 events and 300 configurations. Inspect each run's
`history.csv`, `split.json`, and `test_predictions.csv`. The saved reference
runs used PyTorch 2.11.0 and CUDA; library and device differences can affect
results. The compact run reaches its epoch cap while validation is improving.

## Interpret the results

The presets are controlled sensitivity exercises. They use nominal circuit
settings and a single-site spin-temperature response. Their parameter ranges
are not experimentally fitted populations. The reference noise is white
Gaussian with standard deviation 10⁻⁹ V, and each configuration has an
independent baseline reference averaged over 16 simulated sweeps.

TE calibration is ideal, and the electronics are stable between reference and
signal in these exercises. Temperature/calibration uncertainty and unmodeled
drift remain outside their scores. A supplied noise covariance must be on the
exact 500-bin grid in V²; see `generate_data.py --help` and the physics reference.

Prediction error is `P_true - P_pred`. Positive bias means underprediction.
Width is population standard deviation, giving `RMSE² = bias² + width²`.
Multiply fractional-P errors by 100 for percentage points. Report counts when
restricting to a signed polarization band. Residual width alone is not a
calibrated confidence interval.

To update the website with a reproduced result, follow the
[contributor guide](../../CONTRIBUTING.md#update-reference-results).
