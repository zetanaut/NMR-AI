"""Independent identities for the paper review; NOT a full circuit validation.

See notes/physics-electronics-theory.md and notes/implementation-audit.md.
These calculations do not supply fitted component values or experimental noise.
"""

import unittest

import numpy as np


class ReferenceMath(unittest.TestCase):
    def test_loaded_real_voltage_matches_complex_admittance(self):
        impedance = np.array([12 + 9j, 50 - 30j, 1 + 100j, 20 + 0j])
        voltage, source_r, input_r = 0.5, 500.0, 50.0
        loading = 1/source_r + 1/input_r
        current = voltage/source_r
        complex_result = current / (1/impedance + loading)
        eq2 = current * (impedance.real + loading*abs(impedance)**2) / (
            (1 + loading*impedance.real)**2 + (loading*impedance.imag)**2)
        np.testing.assert_allclose(eq2, complex_result.real, rtol=1e-14)

    def test_transmission_line_zero_half_and_quarter_wave_limits(self):
        load, characteristic = 12 + 9j, 50.0

        def transformed(electrical_length):
            t = np.tanh(1j*electrical_length)
            return characteristic * (load + characteristic*t)/(characteristic + load*t)

        self.assertAlmostEqual(abs(transformed(0) - load), 0, places=12)
        for n in (1, 2, 3):
            self.assertAlmostEqual(abs(transformed(n*np.pi) - load), 0, places=12)
        self.assertAlmostEqual(abs(transformed(np.pi/2) - characteristic**2/load), 0, places=11)

    def test_absent_stray_capacitance_leaves_coil_connected(self):
        omega, coil = 2*np.pi*32.7e6, 0.35 + 6j
        terminal = 1/(1/coil + 1j*omega*0)
        self.assertAlmostEqual(abs(terminal-coil), 0, places=14)

    def test_phase_rotation_is_not_voltage_magnitude(self):
        value = 3 + 4j
        self.assertAlmostEqual((value*np.exp(0j)).real, 3)
        self.assertAlmostEqual((value*np.exp(1j*np.pi/2)).real, -4)
        self.assertAlmostEqual(abs(value), 5)

    def test_te_formulas_follow_populations_and_match_rounded_examples(self):
        h, kb, temperature = 6.62607015e-34, 1.380649e-23, 1.5
        deuteron_x = h*32.7e6/(kb*temperature)
        weights = np.exp(deuteron_x*np.array([1, 0, -1]))
        populations = weights/weights.sum()
        p_deuteron = populations[0] - populations[2]
        closed = 4*np.tanh(deuteron_x/2)/(3 + np.tanh(deuteron_x/2)**2)
        self.assertAlmostEqual(p_deuteron, closed, places=14)
        self.assertAlmostEqual(100*p_deuteron, 0.069748986608294, places=11)
        p_proton = np.tanh(h*213e6/(2*kb*temperature))
        self.assertAlmostEqual(100*p_proton, 0.3407449394360998, places=11)

    def test_metric_sign_and_percent_conversions(self):
        truth = np.array([0.049, 0.05, 0.051])
        prediction = np.array([0.048, 0.0495, 0.0515])
        tutorial = prediction-truth
        paper = truth-prediction
        self.assertAlmostEqual(paper.mean(), -tutorial.mean())
        self.assertAlmostEqual(paper.std(), tutorial.std())
        self.assertAlmostEqual(np.mean(paper**2), paper.mean()**2 + paper.std()**2)
        error = 0.0005
        self.assertAlmostEqual(100*error, 0.05)  # percentage points
        self.assertAlmostEqual(100*error/0.05, 1.0)  # relative percent at 5% P
        self.assertAlmostEqual(100*error/0.0005, 100.0)  # relative percent at 0.05% P


if __name__ == "__main__":
    unittest.main()
