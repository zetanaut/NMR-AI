"""Check units, boundaries and published conditional errors independently."""
import hashlib
import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from polarization_metrics import band_metrics, polarization_profile


class PolarizationMetrics(unittest.TestCase):
    def test_eventwise_relative_units_and_error_decomposition(self):
        result = band_metrics([[.1, .09, 1], [.2, .18, 2]], .05, .25, 'positive')
        self.assertAlmostEqual(result['relative_rms_percent'], 10)
        self.assertAlmostEqual(result['rmse_pp'], np.sqrt(2.5))
        self.assertAlmostEqual(result['p95_absolute_error_pp'], 1.95)
        self.assertAlmostEqual(result['rmse_pp']**2, result['bias_pp']**2 + result['width_pp']**2)
        self.assertNotAlmostEqual(result['relative_rms_percent'], result['rmse_pp']/.15)

    def test_sign_edges_zero_and_empty_policy(self):
        rows = [[0, .001, 0], [.01, .009, 1], [.02, .021, 1], [-.01, -.009, 2], [-.02, -.019, 3]]
        self.assertEqual(band_metrics(rows, .01, .02, 'positive')['n'], 1)
        inclusive = band_metrics(rows, .01, .02, 'positive', include_high=True)
        self.assertEqual((inclusive['n'], inclusive['n_groups']), (2, 1))
        negative = band_metrics(rows, .01, .02, 'negative', include_high=True)
        self.assertEqual((negative['n'], negative['n_groups']), (2, 2))
        zero = band_metrics(rows, 0, .01, 'positive')
        self.assertEqual(zero['n'], 1)
        self.assertAlmostEqual(zero['rmse_pp'], .1)
        self.assertIsNone(zero['relative_rms_percent'])
        touching = band_metrics(rows, 0, .02, 'negative')
        self.assertEqual(touching['n'], 1)
        self.assertIsNone(touching['relative_rms_percent'])
        empty = band_metrics(rows, .03, .04, 'positive')
        self.assertEqual((empty['n'], empty['n_groups']), (0, 0))
        for key in ('rmse_pp', 'bias_pp', 'width_pp', 'p95_absolute_error_pp', 'relative_rms_percent'):
            self.assertIsNone(empty[key])

    def test_invalid_inputs_are_rejected(self):
        for rows in ([[.1, float('nan'), 1]], [[1.1, .1, 1]], [[.1, .1, -.1]], [[.1, .1, .5]], [[.1, .1]]):
            with self.assertRaises(ValueError):
                band_metrics(rows, 0, .2, 'positive')
        for lo, hi, sign in ((0, 0, 'positive'), (-.1, .1, 'positive'), (0, 1.1, 'positive'), (0, .1, 'both')):
            with self.assertRaises(ValueError):
                band_metrics([[0, 0, 0]], lo, hi, sign)

    def test_published_bands_cover_all_rows_and_match_independent_calculation(self):
        snapshot = json.loads((ROOT/'docs/assets/model-comparison.json').read_text())
        profile = snapshot['polarization_profile']
        self.assertEqual(profile, polarization_profile(snapshot))
        for i, run in enumerate(snapshot['runs']):
            values = np.asarray(run['test_predictions'])
            total = 0
            for sign in ('positive', 'negative'):
                view = profile['views'][sign][i]
                for band in view['bands'] + [view['default_evaluation']]:
                    selected = []
                    for truth, pred, group in values:
                        upper = abs(truth) <= band['high'] if band['high_inclusive'] else abs(truth) < band['high']
                        if (truth >= 0) == (sign == 'positive') and abs(truth) >= band['low_inclusive'] and upper:
                            selected.append((truth, pred, group))
                    self.assertEqual(len(selected), band['n'])
                    self.assertEqual(len({row[2] for row in selected}), band['n_groups'])
                    truth, prediction, _ = np.asarray(selected).T
                    error = truth - prediction
                    self.assertAlmostEqual(band['rmse_pp'], 100*np.linalg.norm(error)/np.sqrt(len(error)), places=13)
                    self.assertAlmostEqual(band['bias_pp'], 100*np.mean(error), places=13)
                    self.assertAlmostEqual(band['width_pp'], 100*np.std(error), places=13)
                    self.assertAlmostEqual(band['p95_absolute_error_pp'], 100*np.percentile(abs(error), 95), places=13)
                    if band['low_inclusive'] > 0:
                        self.assertAlmostEqual(band['relative_rms_percent'], 100*np.linalg.norm(error/truth)/np.sqrt(len(error)), places=13)
                    else:
                        self.assertIsNone(band['relative_rms_percent'])
                total += sum(b['n'] for b in view['bands'])
            self.assertEqual(total, len(values))
        for path, key in (('docs/assets/model-polarization.svg', 'polarization_figure_sha256'),
                          ('tools/polarization_metrics.py', 'polarization_metrics_sha256')):
            self.assertEqual(hashlib.sha256((ROOT/path).read_bytes()).hexdigest(), snapshot[key])


if __name__ == '__main__':
    unittest.main()
