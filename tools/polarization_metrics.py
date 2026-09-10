"""Descriptive error metrics in truth-polarization bands of saved predictions."""
import numpy as np


def band_metrics(predictions, low, high, sign, *, include_high=False):
    """Select by |P_true| and sign; retain shared-group counts and zero semantics.

    Columns are fractional P_true, fractional P_pred and an independent-group ID.
    Relative RMS uses each event's true P. It is deliberately omitted for any
    band touching zero, without dropping rows or introducing a denominator floor.
    """
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Need finite (events,3) truth, prediction and group columns")
    if np.any(abs(values[:, 0]) > 1) or np.any(values[:, 2] < 0) or np.any(values[:, 2] != np.floor(values[:, 2])):
        raise ValueError("Truth must be fractional P and group IDs nonnegative integers")
    if not np.isfinite([low, high]).all() or not 0 <= low < high <= 1:
        raise ValueError("Need 0 <= low < high <= 1 in fractional polarization")
    if sign not in ("positive", "negative"):
        raise ValueError("Choose positive (P >= 0) or negative (P < 0)")
    magnitude = abs(values[:, 0])
    selected = ((values[:, 0] >= 0) if sign == "positive" else (values[:, 0] < 0))
    selected &= (magnitude >= low) & ((magnitude <= high) if include_high else (magnitude < high))
    truth, prediction, groups = values[selected].T
    error = truth - prediction
    return {
        "low_inclusive": float(low), "high": float(high), "high_inclusive": bool(include_high),
        "sign": sign, "n": len(error), "n_groups": len(np.unique(groups)),
        "bias_pp": float(100 * error.mean()) if len(error) else None,
        "width_pp": float(100 * error.std(ddof=0)) if len(error) else None,
        "rmse_pp": float(100 * np.sqrt(np.mean(error**2))) if len(error) else None,
        "p95_absolute_error_pp": float(100 * np.quantile(abs(error), .95)) if len(error) else None,
        "relative_rms_percent": float(100 * np.sqrt(np.mean((error / truth)**2))) if len(error) and low > 0 else None,
    }


def polarization_profile(snapshot):
    """Use declared descriptive bands for the unchanged ±25% TE-area benchmark."""
    if snapshot['input_contract']['feature_mode'] != 'te_area':
        raise ValueError("This publication profile is for the TE-area comparison")
    edges = [0, .01, .025, .05, .10, .15, .20, .25]
    views = {}
    reference = np.asarray(snapshot['runs'][0]['test_predictions'])[:, [0, 2]]
    if np.any(abs(reference[:, 0]) > edges[-1]):
        raise ValueError("Declared bands do not cover the test polarization range")
    for sign in ('positive', 'negative'):
        views[sign] = []
        for run in snapshot['runs']:
            np.testing.assert_array_equal(np.asarray(run['test_predictions'])[:, [0, 2]], reference)
            views[sign].append({
                'architecture': run['architecture'], 'label': run['label'],
                'bands': [band_metrics(run['test_predictions'], lo, hi, sign, include_high=(hi == edges[-1]))
                          for lo, hi in zip(edges[:-1], edges[1:])],
                'default_evaluation': band_metrics(run['test_predictions'], .005, .015, sign),
            })
    return {
        'version': 'polarization-bands-v1', 'edges_fractional_p': edges,
        'group_column': snapshot['group_column'], 'views': views,
        'default_scale_percent': 1, 'evaluation_half_width_pp': .5,
        'absolute_definition': '100 * sqrt(mean((P_true-P_pred)^2)), in percentage points',
        'relative_definition': '100 * sqrt(mean(((P_true-P_pred)/P_true)^2)), in percent',
        'zero_policy': 'Relative RMS is omitted for bands touching zero; all rows remain in absolute metrics. No denominator floor.',
        'scope': 'Descriptive reanalysis of unchanged saved test predictions; no fixed-P experiment, retraining or experimental accuracy claim. Bands share configuration groups; no uncertainty intervals.',
    }
