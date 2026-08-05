"""fieldspice --- electromagnetic field simulation for analog and digital circuits.

A from-first-principles field solver built the way MEEP is built --- a geometry
description, a staggered grid, sources, monitors --- but aimed at the regime
MEEP is not: structures far smaller than a wavelength, filled with conductors
and semiconductors, driven for microseconds rather than picoseconds.

Quick start
-----------
>>> import numpy as np
>>> import fieldspice as fs
>>> g = fs.RectilinearGrid.uniform([(0, 2e-6), (0, 5e-6), (0, 4e-6)], [24, 10, 8])
>>> mats = fs.MaterialMap(g, background="sio2")
>>> C = fs.extraction.capacitance_matrix(g, mats.eps(), terminals)   # doctest: +SKIP

Why this exists
---------------
Explicit full-wave FDTD is bound by the Courant condition, so its time step is
set by how long light takes to cross the smallest cell. For circuits that is
catastrophic: a 2 nm gate oxide forces a 1e-18 s step, and 1 ns of switching
then needs a billion of them. Circuits, however, are almost always *electrically
small* --- the structure is far shorter than a wavelength --- so the radiative
coupling between E and B contributes nothing to the answer. Dropping it removes
the wave, removes the CFL limit, and makes implicit time stepping
unconditionally stable, at which point the step size is set by the signal rather
than by the speed of light. That is a 1e4 to 1e8 reduction in step count.

Every approximation involved is written down in ``docs/ASSUMPTIONS.md`` with a
tag, a validity condition, and what it costs you. Solvers record which tags were
active in ``Result.meta["assumptions"]``, and :mod:`fieldspice.validate` checks
the validity conditions against your actual geometry before you waste an hour on
a run whose physics does not apply.

Choosing a solver
-----------------
Compute ``fieldspice.reference.electrical_length(size, t_rise=...)`` first.

===================  ==========================================================
``L/lambda < 0.01``  ``eqs`` (R and C), ``mqs`` (L and eddy), ``darwin`` (all)
``0.01 - 0.3``       ``darwin``, cross-checked against ``fdtd``
``> 0.3``            ``fdtd`` --- the wave is real, quasi-static will lie to you
semiconductors       ``dd`` (drift-diffusion, Scharfetter-Gummel)
frequency domain     ``ac`` --- one complex solve per frequency
lumped netlist       ``circuit.mna``, or ``circuit.coupling`` to join the two
===================  ==========================================================
"""

from __future__ import annotations

__version__ = "0.1.0"

# -- core (frozen, validated) ---------------------------------------------
from .grid import RectilinearGrid, auto_mesh_1d, graded_1d
from .operators import Operators
from .units import (
    A, F, GHz, H, Hz, MHz, S, V, W, aF, c0, cm, cm2_per_Vs, eps0, fF, fs, kB,
    kHz, kohm, m, mA, mF, mH, mV, mil, mm, mu0, nA, nF, nH, nm, ns, ohm, pA,
    pF, pH, per_cm3, ps, q, s, thermal_voltage, uA, uF, uH, um, us,
)

__all__ = [
    "__version__",
    # grid / operators
    "RectilinearGrid", "graded_1d", "auto_mesh_1d", "Operators",
    # units and constants
    "c0", "eps0", "mu0", "q", "kB", "thermal_voltage",
    "m", "cm", "mm", "um", "nm", "mil", "s", "ms", "us", "ns", "ps", "fs",
    "Hz", "kHz", "MHz", "GHz", "V", "mV", "A", "mA", "uA", "nA", "pA",
    "ohm", "kohm", "S", "F", "mF", "uF", "nF", "pF", "fF", "aF",
    "H", "mH", "uH", "nH", "pH", "W", "per_cm3", "cm2_per_Vs",
    # submodules
    "reference", "validate", "grid", "operators", "units",
]

# ``ms`` is re-exported by name below to keep the import list above readable.
from .units import ms  # noqa: E402

# -- lazily-exposed submodules --------------------------------------------
# Importing every solver eagerly would drag in scipy.sparse.linalg, optional
# accelerators and matplotlib on ``import fieldspice``, which makes the common
# case (a short script that uses one solver) noticeably slower to start and
# makes a missing optional dependency fatal rather than local. PEP 562 module
# __getattr__ gives the ergonomics of eager import with the cost of lazy.
_LAZY = {
    "materials", "geometry", "boundaries", "sources", "monitors", "io", "viz",
    "validate", "reference", "solvers", "circuit", "extraction",
}

_LAZY_ATTRS = {
    # convenience re-exports, resolved on first use
    "Material": ("materials", "Material"),
    "SemiconductorParams": ("materials", "SemiconductorParams"),
    "MaterialMap": ("materials", "MaterialMap"),
    "LIBRARY": ("materials", "LIBRARY"),
    "Box": ("geometry", "Box"),
    "Cylinder": ("geometry", "Cylinder"),
    "Sphere": ("geometry", "Sphere"),
    "Prism": ("geometry", "Prism"),
    "HalfSpace": ("geometry", "HalfSpace"),
    "LayerStack": ("geometry", "LayerStack"),
    "voxelize": ("geometry", "voxelize"),
    "BoundarySpec": ("boundaries", "BoundarySpec"),
    "Dirichlet": ("boundaries", "Dirichlet"),
    "Neumann": ("boundaries", "Neumann"),
    "Terminal": ("solvers.base", "Terminal"),
    "Result": ("solvers.base", "Result"),
    "SolverConfig": ("solvers.base", "SolverConfig"),
}


def __getattr__(name: str):
    import importlib

    if name in _LAZY:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod
    if name in _LAZY_ATTRS:
        modname, attr = _LAZY_ATTRS[name]
        obj = getattr(importlib.import_module(f".{modname}", __name__), attr)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__) | _LAZY | set(_LAZY_ATTRS))
