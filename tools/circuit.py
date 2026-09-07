"""CW Q-meter: passive RLGC cable, resonator, finite source, phase detector.

Topology: arXiv:2603.10146v5 Fig. 1 and Eq. (2), ONE series tuning capacitor.
Component seeds: supplied minuit.py, not a fitted trace. Units: Hz, V, ohm,
H, F, m, radians. No minimum subtraction or arbitrary baseline polynomial.
"""
from dataclasses import asdict, dataclass
import numpy as np


@dataclass(frozen=True)
class Circuit:
    reference_hz: float = 32.68e6
    drive_v: float = 0.382652652
    drive_resistance_ohm: float = 619.0
    input_resistance_ohm: float = 50.0
    damping_resistance_ohm: float = 10.0
    coil_resistance_ohm: float = 0.35
    coil_inductance_h: float = 3e-8
    tune_capacitance_f: float = 1e-10
    stray_capacitance_f: float = 0.0
    cable_resistance_ohm_per_m: float = 3.43
    cable_inductance_h_per_m: float = 2.542e-7
    cable_conductance_s_per_m: float = 0.0
    cable_capacitance_f_per_m: float = 1.027e-10
    cable_length_m: float = 3.0
    filling_factor: float = 0.098563235
    susceptibility_scale_cgs: float = 0.11133
    detector_gain: float = -1.0
    detector_phase_rad: float = 0.0
    detector_phase_slope_rad_per_hz: float = 0.0
    detector_phase_curvature_rad_per_hz2: float = 0.0
    dc_offset_v: float = 0.0

    def __post_init__(self):
        values = asdict(self)
        if not np.isfinite(list(values.values())).all():
            raise ValueError("Circuit parameters must be finite")
        for name in ("reference_hz", "drive_resistance_ohm", "input_resistance_ohm",
                     "coil_resistance_ohm", "coil_inductance_h", "tune_capacitance_f",
                     "cable_inductance_h_per_m", "cable_capacitance_f_per_m", "susceptibility_scale_cgs"):
            if values[name] <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("damping_resistance_ohm", "stray_capacitance_f", "cable_resistance_ohm_per_m",
                     "cable_conductance_s_per_m", "cable_length_m"):
            if values[name] < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.filling_factor <= 1 or self.drive_v == 0 or self.detector_gain == 0:
            raise ValueError("Require 0 <= filling_factor <= 1 and nonzero drive/gain")


def cable_parameters(frequency_hz, c):
    omega = 2*np.pi*np.asarray(frequency_hz, dtype=float)
    if not np.isfinite(omega).all() or np.any(omega <= 0):
        raise ValueError("Frequency must be positive and finite")
    series = c.cable_resistance_ohm_per_m + 1j*omega*c.cable_inductance_h_per_m
    shunt = c.cable_conductance_s_per_m + 1j*omega*c.cable_capacitance_f_per_m
    return np.sqrt(series/shunt), np.sqrt(series*shunt)


def shunted_coil(frequency_hz, c, chi=0):
    omega = 2*np.pi*np.asarray(frequency_hz)
    inductance = c.coil_inductance_h*(1 + 4*np.pi*c.filling_factor*chi)
    coil = c.coil_resistance_ohm + 1j*omega*inductance
    return 1/(1/coil + 1j*omega*c.stray_capacitance_f)


def line_input(frequency_hz, c, load):
    z0, gamma = cable_parameters(frequency_hz, c)
    t = np.tanh(gamma*c.cable_length_m)
    return z0*(load + z0*t)/(z0 + load*t)


def resonator_impedance(frequency_hz, c, chi=0):
    omega = 2*np.pi*np.asarray(frequency_hz)
    return (c.damping_resistance_ohm + 1/(1j*omega*c.tune_capacitance_f)
            + line_input(frequency_hz, c, shunted_coil(frequency_hz, c, chi)))


def node_voltage(impedance, drive_v, drive_resistance, input_resistance):
    """Exact Fig. 1 nodal solution, including finite source resistance."""
    return (drive_v/drive_resistance)/(1/impedance + 1/drive_resistance + 1/input_resistance)


def detector_voltage(frequency_hz, c, chi=0):
    z = resonator_impedance(frequency_hz, c, chi)
    u = node_voltage(z, c.drive_v, c.drive_resistance_ohm, c.input_resistance_ohm)
    df = np.asarray(frequency_hz) - c.reference_hz
    phase = c.detector_phase_rad + c.detector_phase_slope_rad_per_hz*df + c.detector_phase_curvature_rad_per_hz2*df**2
    return c.detector_gain*np.real(u*np.exp(1j*phase)) + c.dc_offset_v


def nominal_circuit():
    """One-half-wave deuteron benchmark, tuned at f0 using supplied R/L/C.

    Zero stray admittance is an explicit idealization, not a unit guess for
    the supplied script's ambiguous Cstray=400 initial value.
    """
    values = asdict(Circuit())
    c = Circuit(**values)
    _, gamma = cable_parameters(c.reference_hz, c)
    values["cable_length_m"] = float(np.pi/gamma.imag)
    c = Circuit(**values)
    zline = line_input(c.reference_hz, c, shunted_coil(c.reference_hz, c))
    if zline.imag <= 0:
        raise ValueError("Operating point cannot be tuned with a series capacitor")
    values["tune_capacitance_f"] = float(1/(2*np.pi*c.reference_hz*zline.imag))
    return Circuit(**values)
