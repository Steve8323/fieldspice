"""Physical constants and unit helpers (strict SI everywhere).

fieldspice uses SI internally with **no** scaling or normalisation: metres,
seconds, volts, amperes, kelvin.  Convenience multipliers are provided so that
user scripts can write ``2 * um`` instead of ``2e-6``.

Rationale for strict SI: circuit problems span ~9 decades of length (1 nm gate
oxide to 1 m cable) and ~12 decades of time (1 ps edge to 1 s settling).  Any
"natural units" choice that helps one end hurts the other, and silent unit
scaling is a classic source of wrong answers in mixed field/circuit tools.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# Fundamental constants (CODATA 2018)
# --------------------------------------------------------------------------
c0 = 299_792_458.0
"""Speed of light in vacuum [m/s] (exact)."""

mu0 = 4e-7 * math.pi * 1.0000000000552
"""Vacuum magnetic permeability [H/m] (CODATA 2018; no longer exactly 4pi e-7)."""

eps0 = 1.0 / (mu0 * c0 * c0)
"""Vacuum electric permittivity [F/m]."""

eta0 = math.sqrt(mu0 / eps0)
"""Vacuum wave impedance [ohm], ~376.73."""

q = 1.602_176_634e-19
"""Elementary charge [C] (exact)."""

kB = 1.380_649e-23
"""Boltzmann constant [J/K] (exact)."""

hbar = 1.054_571_817e-34
"""Reduced Planck constant [J s]."""

h_planck = hbar * 2.0 * math.pi
"""Planck constant [J s]."""

m_e = 9.109_383_7015e-31
"""Electron rest mass [kg]."""

T_ROOM = 300.0
"""Default lattice temperature [K].  Note this is 300 K, not 293.15 K --- the
semiconductor convention, chosen so that kT/q = 25.852 mV."""


def thermal_voltage(T: float = T_ROOM) -> float:
    """Thermal voltage kT/q [V].  25.852 mV at 300 K."""
    return kB * T / q


# --------------------------------------------------------------------------
# Length
# --------------------------------------------------------------------------
m = 1.0
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9
angstrom = 1e-10
inch = 25.4e-3
mil = 25.4e-6  # thousandth of an inch; the PCB unit

# --------------------------------------------------------------------------
# Time / frequency
# --------------------------------------------------------------------------
s = 1.0
ms = 1e-3
us = 1e-6
ns = 1e-9
ps = 1e-12
fs = 1e-15

Hz = 1.0
kHz = 1e3
MHz = 1e6
GHz = 1e9
THz = 1e12

# --------------------------------------------------------------------------
# Electrical
# --------------------------------------------------------------------------
V = 1.0
mV = 1e-3
uV = 1e-6

A = 1.0
mA = 1e-3
uA = 1e-6
nA = 1e-9
pA = 1e-12
fA = 1e-15

ohm = 1.0
kohm = 1e3
Mohm = 1e6

S = 1.0
mS = 1e-3
uS = 1e-6

F = 1.0
mF = 1e-3
uF = 1e-6
nF = 1e-9
pF = 1e-12
fF = 1e-15
aF = 1e-18

H = 1.0
mH = 1e-3
uH = 1e-6
nH = 1e-9
pH = 1e-12

W = 1.0
mW = 1e-3
uW = 1e-6

# --------------------------------------------------------------------------
# Doping / density
# --------------------------------------------------------------------------
per_cm3 = 1e6
"""Multiply a cm^-3 number by this to get m^-3.  ``1e17 * per_cm3`` is the
idiomatic way to write a 1e17 cm^-3 doping in fieldspice."""

per_cm2 = 1e4
cm2_per_Vs = 1e-4
"""Mobility unit conversion: cm^2/(V s) -> m^2/(V s)."""


__all__ = [
    "c0", "mu0", "eps0", "eta0", "q", "kB", "hbar", "h_planck", "m_e",
    "T_ROOM", "thermal_voltage",
    "m", "cm", "mm", "um", "nm", "angstrom", "inch", "mil",
    "s", "ms", "us", "ns", "ps", "fs",
    "Hz", "kHz", "MHz", "GHz", "THz",
    "V", "mV", "uV", "A", "mA", "uA", "nA", "pA", "fA",
    "ohm", "kohm", "Mohm", "S", "mS", "uS",
    "F", "mF", "uF", "nF", "pF", "fF", "aF",
    "H", "mH", "uH", "nH", "pH", "W", "mW", "uW",
    "per_cm3", "per_cm2", "cm2_per_Vs",
]
