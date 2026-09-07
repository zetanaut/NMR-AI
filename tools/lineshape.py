"""Self-contained spin-1 Dulya/Pake powder lineshape (NumPy only).

Theory: C. Dulya et al., NIM A 398 (1997) 109–125,
https://doi.org/10.1016/S0168-9002(97)00317-3.

This is the supplied broadened branch formula and weighted azimuthal average,
with weak-quadrupole, spin-temperature transition weights. It represents one
quadrupolar site, not a fitted mixture of sites or a complete Q-meter model.
"""

import numpy as np

SIMULATOR = "dulya-pake-v1"


def branch(x, eps, eta, phi, g):
    """Dulya branch at fixed azimuth phi (radians); x and g are dimensionless.

    In the axial, zero-broadening limit, eps=+1 has support -2 < x < +1
    and its powder horn at +1. The eps=-1 branch is its frequency reflection.
    The conventional factor 1/10 cancels in the final doublet normalization.
    """
    x = np.asarray(x, dtype=float)
    phi = np.asarray(phi, dtype=float)
    if eps not in (-1, 1) or not np.isfinite([eta, g]).all() or not 0 <= eta <= 1 or g <= 0:
        raise ValueError("Require eps = ±1, 0 <= eta <= 1, and finite g > 0")
    if not np.isfinite(x).all() or not np.isfinite(phi).all():
        raise ValueError("Frequency and azimuth must be finite")
    y = np.sqrt(3.0 - eta * np.cos(2.0 * phi))
    b = 1.0 - eps * x - eta * np.cos(2.0 * phi)
    radius = np.hypot(g, b)
    # atan2 half-angles avoid subtracting nearly equal numbers in the wings.
    alpha = np.arctan2(g, b)
    cosine, sine = np.cos(alpha / 2.0), np.sin(alpha / 2.0)
    root = np.sqrt(radius)
    angle = np.pi / 2.0 + np.arctan2(y*y - radius, 2.0*y*root*sine)
    numerator = y*y + radius + 2.0*y*root*cosine
    denominator = (y - root*cosine)**2 + (root*sine)**2
    logarithm = np.log(numerator / denominator)
    return (2.0*cosine*angle + sine*logarithm) / (20.0*np.pi*root)


def powder_branch(x, eps, eta=0.03, g=0.08, nphi=32):
    """Weighted phi average, using the supplied 33-point convention by default."""
    x = np.asarray(x, dtype=float)
    if not isinstance(nphi, (int, np.integer)) or nphi < 1:
        raise ValueError("nphi must be a positive integer")
    phis = np.linspace(0.0, np.pi / 2.0, nphi + 1)
    # branch validates eta before it is used in the powder weights.
    kernels = branch(x.reshape(-1, 1), eps, eta, phis[None, :], g)
    weights = np.sqrt(3.0 / (3.0 - eta * np.cos(2.0 * phis)))
    return (kernels * weights).mean(axis=1).reshape(x.shape)


def pake_doublet(x, p, cc, eta=0.03, g=0.08, nphi=32):
    """Return total and both transition contributions; sum(total) = p / cc.

    x = (frequency - center) / split_mhz; g is Lorentzian HWHM / split_mhz.
    The calibration uses a DISCRETE sum on the supplied frequency grid, before
    gain, baseline, and noise. It is not a frequency integral.

    Spin temperature fixes Q = 2 - sqrt(4 - 3P²). The normalized branch weights
    are (1 ± Q/P)/2, equivalent to the supplied r:1 prescription. Computing
    Q/P as 3P/(2 + sqrt(4 - 3P²)) avoids cancellation, a near-zero P floor,
    and the singular r at |P|=1. Negative P reverses polarity and horn weights.
    """
    x = np.asarray(x, dtype=float)
    if not np.isfinite([p, cc]).all() or not -1 <= p <= 1 or cc == 0:
        raise ValueError("Require finite -1 <= P <= 1 and nonzero finite cc")
    plus = powder_branch(x, +1, eta, g, nphi)
    minus = powder_branch(x, -1, eta, g, nphi)
    q_over_p = 3.0 * p / (2.0 + np.sqrt(4.0 - 3.0*p*p))
    plus *= 0.5 * (1.0 + q_over_p)
    minus *= 0.5 * (1.0 - q_over_p)
    area = np.sum(plus + minus)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("The sampled lineshape must have positive finite area")
    scale = (p / cc) / area
    plus *= scale
    minus *= scale
    return plus + minus, plus, minus
