"""Material extensions of the complex spin-1 kernel; see notes/material-examples.md.

The supplied butanol theory mixes C-D and O-D sites with a common physical
Lorentzian HWHM and polarization. This implementation uses its weak-quadrupole
option and the existing uniformly averaged complex kernels. Site densities
include 1/splitting in MHz, so site fractions weight integrated susceptibility.
"""
import numpy as np

from lineshape import pake_susceptibility


def butanol_components(frequency_mhz, polarization, center_mhz, split_cd_mhz,
                       split_od_mhz, hwhm_mhz, eta_cd, eta_od, od_fraction,
                       nphi=32):
    """Return total, C-D and O-D complex densities, in inverse MHz.

    Each unweighted site's absorption integral over MHz equals P. No finite-grid
    normalization is applied. The two sites share spin temperature and HWHM.
    Multiply by a susceptibility amplitude in cgs*MHz before circuit propagation.
    """
    f = np.asarray(frequency_mhz, dtype=float)
    values = [center_mhz, split_cd_mhz, split_od_mhz, hwhm_mhz, od_fraction]
    if not np.isfinite(f).all() or not np.isfinite(values).all():
        raise ValueError("Require finite frequencies and material parameters")
    if min(split_cd_mhz, split_od_mhz, hwhm_mhz) <= 0 or not 0 <= od_fraction <= 1:
        raise ValueError("Require positive splittings/HWHM and 0 <= O-D fraction <= 1")
    cd = (1-od_fraction)/split_cd_mhz * pake_susceptibility(
        (f-center_mhz)/split_cd_mhz, polarization, eta_cd, hwhm_mhz/split_cd_mhz, nphi)[0]
    od = od_fraction/split_od_mhz * pake_susceptibility(
        (f-center_mhz)/split_od_mhz, polarization, eta_od, hwhm_mhz/split_od_mhz, nphi)[0]
    return cd+od, cd, od
