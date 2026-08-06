"""Explicit leapfrog Yee FDTD --- the full-wave reference the rest of fieldspice is measured against.

This solver exists for two reasons and neither of them is speed.

1. It is the **oracle for the quasi-static approximation**.  ``A1`` claims that
   dropping radiative coupling costs nothing when ``L/lambda < 0.01``.  That is
   a claim about physics, not about discretisation, so it cannot be tested by
   refining an EQS mesh; it can only be tested against a solver that keeps the
   wave.  This is that solver.
2. It covers the regime where quasi-static *fails* (``L/lambda > 0.3``), where
   no amount of mesh refinement will rescue an EQS answer.

It is deliberately the plain, auditable implementation: dense-in-time, explicit,
no operator splitting beyond the leapfrog itself, and every operator is one of
the frozen incidence matrices from :mod:`fieldspice.operators`.

The scheme
==========
State is the same integrated (finite-integration) state the quasi-static
solvers use, so the two are directly comparable field by field:

* ``e`` --- edge circulation of E, ``int E.dl`` [V], sampled at integer steps
  ``t = n*dt``;
* ``b`` --- face flux of B, ``int B.dA`` [Wb], sampled at half steps
  ``t = (n + 1/2)*dt``.

Faraday's law is exact in this representation --- ``db/dt`` is *literally* minus
the closed line integral of E around the face --- so it is applied with the
signed incidence matrix and no metric at all::

    b -= dt * (C @ e)

Ampere's law lives on the dual grid.  Writing ``h = Nu b`` for the dual-edge mmf
[A] and ``i = M_sigma e`` for the conduction current through the dual face [A]::

    M_eps de/dt = C.T @ (Nu @ b) - M_sigma @ e - i_src

Units, checked term by term (this is the check that catches the classic bugs):

===========================  ===========================================  =======
Quantity                     Definition                                   Unit
===========================  ===========================================  =======
``M_eps[e]``                 ``eps_e * A_dual_e / L_e``                   F
``M_eps @ de/dt``            F * V/s                                      **A**
``M_sigma[e]``               ``sigma_e * A_dual_e / L_e``                 S
``M_sigma @ e``              S * V                                        **A**
``Nu[f]``                    ``L_dual_f / (mu_f * A_f)``                  1/H
``Nu @ b``                   Wb/H = Wb/(Wb/A)                             **A**
``C.T @ (Nu @ b)``           sum of +-1 times A                           **A**
===========================  ===========================================  =======

.. warning::
   ``operators.face_mass_nu`` returns ``A_f / (mu_f * L_dual_f)``, whose units
   are m^2/H, **not** the 1/H that ``Nu @ b`` needs to be a current.  The
   correct inverse inductance of the flux tube through a face is the reciprocal
   geometry factor ``L_dual_f / (mu_f * A_f)``; the two differ by
   ``(A_f/L_dual_f)**2``, which on a 1 um cell is a factor of 1e-24 and turns
   the speed of light into 1e12 m/s.  ``operators.py`` is frozen, so this module
   builds the reluctance itself in :func:`face_reluctance` and does not call
   ``face_mass_nu``.  ``monitors.py`` reached the same conclusion independently
   and does the same thing; this solver passes its own ``nu`` to monitors via
   ``state["m_nu"]`` so that reported energies always match the solved system.

Why the difference operators are split by axis
==============================================
:func:`directional_curls` factors the frozen curl into ``C = Cx + Cy + Cz``,
where ``Cd`` contains exactly the differences taken along axis ``d`` (verified
to be exact, entry for entry).  Two things fall out of this:

* Each ``Cd`` is, on a rectilinear grid, *exactly* ``A * d/dx_d`` acting on the
  physical field --- because an incidence difference of an integrated quantity
  along ``d`` multiplies by the two lengths orthogonal to ``d``, which are
  constant across that difference.  A coordinate stretch ``d/dx -> (1/s)d/dx``
  is therefore implemented by dividing the corresponding ``Cd`` term, with no
  further metric bookkeeping.  That is what makes a genuine CPML possible in
  the integrated-variable formulation.
* The axis split also fixes the CPML staggering for free: ``Cx`` writes only to
  the face groups whose x index runs over *cells*, and ``Cx.T`` writes only to
  the edge groups whose x index runs over *nodes*.  So the electric and
  magnetic PML profiles are automatically half a cell apart, which is the one
  detail that a hand-indexed CPML usually gets wrong.

Lossy media: exponential time differencing
==========================================
With loss, the semi-discrete edge equation is diagonal, one independent scalar
ODE per edge::

    C_e de/dt + G_e e = f(t),      C_e = M_eps[e] [F],  G_e = M_sigma[e] [S]

The naive explicit update ``e += (dt/C)(f - G e)`` is the forward Euler
discretisation of that ODE and is stable only for ``dt < 2 C/G = 2 eps/sigma``.
For copper ``eps0/sigma = 1.5e-19 s``, roughly 200x *smaller* than the Courant
step even at 0.5 nm resolution, so an explicit scheme dies on contact with a
metal --- and it dies by exploding, which is the failure mode this module is
built to make impossible.

Integrating the ODE exactly over one step, holding ``f`` at its midpoint value
(which is where the leapfrog naturally evaluates it), gives::

    e(t+dt) = e(t) * exp(-k) + (dt/C) * phi1(k) * f,
        k = dt*G/C = dt*sigma_e/eps_e,   phi1(k) = (1 - exp(-k))/k

This is the ETD1 / exponential-midpoint update.  It is second-order accurate for
smooth ``f``, it is **unconditionally stable in the loss term** because
``|exp(-k)| <= 1`` for every ``k >= 0``, and --- the property that actually
matters --- it degrades *correctly* rather than merely safely: as
``k -> infinity`` the update tends to ``e -> f/G``, the resistive/quasi-static
limit, which is the right answer inside a good conductor.  The familiar
Yee lossy update ``ca = (1 - k/2)/(1 + k/2)``, ``cb = (dt/C)/(1 + k/2)`` is the
(1,1) Pade approximant of exactly these two coefficients; it is also stable, but
it changes sign at ``k = 2`` and so oscillates in a metal instead of relaxing.

The wave part of the scheme is still Courant-limited.  Exponential differencing
removes the *conduction* stability constraint only, and :meth:`FDTDSolver.solve`
refuses a time step above the true spectral Courant limit rather than running
and producing an exponentially growing field, because silent instability is the
worst failure mode a field solver has.

Absorbing boundary: what is shipped, and what it measures
=========================================================
**Shipped: a full CPML** (Roden-Gedney convolutional PML with complex frequency
shifting), driven by the parameters in
:class:`fieldspice.boundaries.Absorbing`.  It is *not* a Mur ABC and not a
split-field PML.

Measured normal-incidence reflection, 1D vacuum, uniform 20 um cells, 10-cell
layer, order 3, ``kappa_max = 1``, ``alpha_max = 0.05``, Gaussian pulse with
``tau = 25`` steps, against a four-times-longer reference grid truncated before
its own reflection arrives:

===================  ===========================
Layer thickness      Peak reflection
===================  ===========================
6 cells              -60.5 dB
8 cells              -66.5 dB
10 cells             **-71.5 dB**
12 cells             -75.0 dB
===================  ===========================

:meth:`FDTDSolver.measure_pml_reflection` reproduces that table from scratch, so
the claim is checkable rather than asserted.  The requirement in the build spec
was "anything worse than -40 dB means it is broken"; the shipped layer is
30 dB better than that at the default thickness.

Conductors, and A4b
===================
The default treatment of a conductor here is volumetric ``sigma``, which is
exact but only *useful* if the mesh resolves the skin depth
``delta = sqrt(2/(omega mu sigma))``.  At 1 GHz in copper that is 2.09 um, so a
package-scale full-wave model would need 0.7 um cells inside every trace --- and
by the Courant condition, a 2.3 fs time step everywhere.  That is the
computation A4b exists to avoid.

**This module does not silently mesh through the skin depth, and it does not
silently pretend to.**  It reports how many edges are in the "good conductor at
this time step" regime (``dt*sigma/eps > 10``), and when ``f_max`` is supplied
it runs :func:`fieldspice.validate.check_skin_depth` and warns.  The two correct
responses are:

* **Perfect conductor** --- pass ``pec_cells``.  Every edge touching a PEC cell
  is removed from the update.  This is exact, free, and right whenever the loss
  in the metal does not matter (most interconnect resonance and mode problems).
* **Surface impedance (SIBC)** --- replace the metal volume by a boundary
  condition ``E_tan = Zs (n x H)`` with ``Zs = (1+j) / (sigma * delta)``, valid
  when ``delta << feature size`` so the field in the metal is a locally plane
  wave decaying into a half space.  It removes the metal interior from the mesh
  entirely, so neither the fine cells nor their Courant penalty are ever paid.
  **SIBC is not implemented in this module**, because a correct time-domain SIBC
  needs its own auxiliary-differential-equation fit of ``sqrt(j omega)`` and
  that is a separate piece of work; the honest status is "documented, not
  shipped".  Until it exists, use ``pec_cells`` for good conductors and
  volumetric ``sigma`` only for lossy dielectrics, semiconductors and thin films
  whose thickness is comparable to the skin depth.

Assumptions invoked: **A3** (linear non-dispersive materials), **A4b**
(conductor treatment), **A9** (no optical/nonlinear physics).  Note that A1 is
conspicuously absent --- not making the quasi-static approximation is this
solver's entire job.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..boundaries import (
    WALL_AXIS,
    WALL_SIDE,
    WALLS,
    Absorbing,
    BC,
    BoundarySpec,
    Dirichlet,
    Neumann,
    Periodic,
    Symmetry,
)
from ..grid import RectilinearGrid
from ..operators import (
    Operators,
    cell_to_edge,
    cell_to_face,
    split_edge_vector,
    split_face_vector,
)
from ..units import c0, eps0, mu0
from .base import Result, SolverConfig, Terminal, TimeSteppingSolver

__all__ = [
    "FDTDSolver",
    "CPML",
    "CurrentSource",
    "face_reluctance",
    "directional_curls",
    "cavity_resonance",
    "discrete_cavity_resonance",
]


# ==========================================================================
# Metric helpers
# ==========================================================================
def face_reluctance(grid: RectilinearGrid, mu_face: np.ndarray) -> np.ndarray:
    """Inverse inductance of the flux tube through each face [1/H].

    ``nu_f = L_dual_f / (mu_f * A_f)``, so that ``nu * b`` is the magnetomotive
    force ``int H.dl`` along the dual edge [A] and ``C.T @ (nu * b)`` is a
    current [A].

    Parameters
    ----------
    grid
        The grid.
    mu_face
        Absolute permeability on each face [H/m], flat, length
        ``grid.n_faces``.  Build it with
        ``operators.cell_to_face(grid, mu_cell, mode="series")``, which takes
        the length-weighted harmonic mean along the dual edge --- the rule that
        makes reluctances add in series.

    Returns
    -------
    np.ndarray
        Flat ``(n_faces,)`` inverse inductance [1/H].

    Notes
    -----
    This deliberately does **not** call ``operators.face_mass_nu``, which
    returns the reciprocal geometry factor ``A_f/(mu_f L_dual_f)`` [m^2/H].  See
    the module docstring; ``operators.py`` is frozen so the discrepancy is
    worked around rather than fixed.
    """
    mu_face = np.asarray(mu_face, dtype=float).ravel()
    if mu_face.size != grid.n_faces:
        raise ValueError(
            f"mu_face must have {grid.n_faces} entries, got {mu_face.size}")
    if np.any(mu_face <= 0.0):
        raise ValueError("mu_face must be positive everywhere [H/m]")
    A = np.concatenate([a.ravel() for a in grid.face_areas()])
    L = np.concatenate([a.ravel() for a in grid.face_dual_lengths()])
    return L / (mu_face * A)


def _ids(shape: tuple[int, int, int], offset: int = 0) -> np.ndarray:
    return (offset + np.arange(int(np.prod(shape)))).reshape(shape)


def _assemble(terms: Sequence[tuple[np.ndarray, np.ndarray, float]],
              shape: tuple[int, int]) -> sp.csr_matrix:
    rows = np.concatenate([np.asarray(r).ravel() for r, _, _ in terms])
    cols = np.concatenate([np.asarray(c).ravel() for _, c, _ in terms])
    vals = np.concatenate([np.full(np.asarray(c).size, v, dtype=np.float64)
                           for _, c, v in terms])
    return sp.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()


def directional_curls(grid: RectilinearGrid
                      ) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    """Split the discrete curl by the axis its difference is taken along.

    Returns ``(Cx, Cy, Cz)`` with ``Cx + Cy + Cz == curl_edge_face(grid)``
    exactly (same entries, same sparsity).  ``Cd`` holds only the terms that
    difference an edge quantity along axis ``d``.

    Parameters
    ----------
    grid
        The grid.

    Returns
    -------
    tuple of scipy.sparse.csr_matrix
        Each of shape ``(n_faces, n_edges)`` with entries in ``{-1, 0, +1}``.

    Notes
    -----
    Needed only by the PML: a stretched coordinate ``s_d`` acts on derivatives
    along ``d`` alone, and a full curl mixes two directions per component, so
    the curl has to be taken apart before it can be stretched.  Away from a PML
    the fused ``C`` is used instead, because three sparse matrix-vector products
    cost about three times one.
    """
    sx, sy, sz = grid.shape_edges
    fx, fy, fz = grid.shape_faces
    nex, ney, _ = grid.n_edges_each
    nfx, nfy, _ = grid.n_faces_each
    ex, ey, ez = _ids(sx, 0), _ids(sy, nex), _ids(sz, nex + ney)
    Fx, Fy, Fz = _ids(fx, 0), _ids(fy, nfx), _ids(fz, nfx + nfy)
    shape = (grid.n_faces, grid.n_edges)

    # Grouped exactly as operators.curl_edge_face builds them, but sorted by
    # the axis of the difference rather than by the face normal.
    cx = _assemble([(Fy, ez[:-1, :, :], +1.0), (Fy, ez[1:, :, :], -1.0),
                    (Fz, ey[1:, :, :], +1.0), (Fz, ey[:-1, :, :], -1.0)], shape)
    cy = _assemble([(Fx, ez[:, 1:, :], +1.0), (Fx, ez[:, :-1, :], -1.0),
                    (Fz, ex[:, :-1, :], +1.0), (Fz, ex[:, 1:, :], -1.0)], shape)
    cz = _assemble([(Fx, ey[:, :, :-1], +1.0), (Fx, ey[:, :, 1:], -1.0),
                    (Fy, ex[:, :, 1:], +1.0), (Fy, ex[:, :, :-1], -1.0)], shape)
    return cx, cy, cz


# ==========================================================================
# Analytic cavity references (used by the validation suite)
# ==========================================================================
def cavity_resonance(a: float, b: float, d: float,
                     m: int = 1, n: int = 0, p: int = 1,
                     eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """Resonant frequency of a rectangular PEC cavity [Hz].

    ``f = (v/2) * sqrt((m/a)^2 + (n/b)^2 + (p/d)^2)`` with
    ``v = c0/sqrt(eps_r*mu_r)``.  ``TE101`` is ``(m, n, p) = (1, 0, 1)``.

    Parameters
    ----------
    a, b, d
        Cavity dimensions along x, y, z [m].
    m, n, p
        Mode indices (dimensionless).  At most one may be zero for a TE mode.
    eps_r, mu_r
        Relative permittivity and permeability of the fill (dimensionless).

    Returns
    -------
    float
        Resonant frequency [Hz] of the *continuum* cavity.  A Yee grid resonates
        slightly low; see :func:`discrete_cavity_resonance`.
    """
    if min(a, b, d) <= 0.0:
        raise ValueError("cavity dimensions must be positive [m]")
    v = c0 / np.sqrt(eps_r * mu_r)
    return float(0.5 * v * np.sqrt((m / a) ** 2 + (n / b) ** 2 + (p / d) ** 2))


def discrete_cavity_resonance(a: float, b: float, d: float,
                              hx: float, hy: float, hz: float, dt: float,
                              m: int = 1, n: int = 0, p: int = 1,
                              eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """Resonance of the *discrete* Yee cavity [Hz] --- the number to test against.

    The Yee scheme has an exact dispersion relation, so a PEC box on a uniform
    grid has an exact discrete eigenfrequency::

        sin(omega*dt/2)/(v*dt) = sqrt( sum_d ( sin(k_d h_d/2)/h_d )^2 )

    with ``k_x = m*pi/a`` etc.  A correct implementation must reproduce *this*
    number, not :func:`cavity_resonance`; the gap between the two is grid
    dispersion, which is a property of the method rather than a bug, and shrinks
    as ``h^2``.

    Parameters
    ----------
    a, b, d
        Cavity dimensions [m].
    hx, hy, hz
        Uniform cell sizes [m].
    dt
        Time step [s].
    m, n, p
        Mode indices (dimensionless).
    eps_r, mu_r
        Fill material (dimensionless).

    Returns
    -------
    float
        Discrete resonant frequency [Hz].
    """
    v = c0 / np.sqrt(eps_r * mu_r)
    kx, ky, kz = m * np.pi / a, n * np.pi / b, p * np.pi / d
    rhs2 = ((np.sin(kx * hx / 2.0) / hx) ** 2
            + (np.sin(ky * hy / 2.0) / hy) ** 2
            + (np.sin(kz * hz / 2.0) / hz) ** 2)
    s = v * dt * np.sqrt(rhs2)
    if s > 1.0:
        raise ValueError("dt exceeds the Courant limit for this mode")
    return float(2.0 * np.arcsin(s) / (2.0 * np.pi * dt))


# ==========================================================================
# Sources
# ==========================================================================
@dataclass
class CurrentSource:
    """An impressed current density on a set of edges [A].

    A *soft* source: the current is added to the right-hand side of Ampere's law
    and the field is free to respond, so the source does not scatter and does
    not have to be de-embedded.  A hard source (overwriting ``e``) is a
    perfectly reflecting wall by construction, which quietly ruins any
    reflection measurement, so it is not offered.

    Parameters
    ----------
    edges
        Flat edge indices the current flows along, in the +axis direction of the
        edge.  Use ``grid.edge_index(direction, i, j, k)``.
    waveform
        Current [A] as a constant or a callable ``f(t) -> float`` with ``t`` in
        seconds.  The solver evaluates it at the half step ``(n + 1/2)*dt``,
        where the leapfrog wants it.
    weight
        Optional per-edge multiplier (dimensionless), broadcast against
        ``edges``.  Use it to spread a terminal current over several edges or to
        apodise a source.  Defaults to 1.
    name
        Identifier; appears in ``Result.terminals``.

    Notes
    -----
    Sign: positive ``waveform`` drives current **along** the edge direction,
    consistent with ``i = M_sigma e`` in the same direction.
    """

    edges: np.ndarray
    waveform: float | Callable[[float], float] = 0.0
    weight: np.ndarray | float = 1.0
    name: str = "j"

    def __post_init__(self) -> None:
        self.edges = np.atleast_1d(np.asarray(self.edges, dtype=np.intp)).ravel()
        if self.edges.size == 0:
            raise ValueError(f"source {self.name!r} has no edges")
        w = np.asarray(self.weight, dtype=float)
        if w.ndim == 0:
            w = np.full(self.edges.shape, float(w))
        elif w.shape != self.edges.shape:
            raise ValueError(
                f"source {self.name!r}: weight shape {w.shape} does not match "
                f"{self.edges.shape} edges")
        self.weight = w

    def value_at(self, t: float) -> float:
        """Source current [A] at time ``t`` [s]."""
        wf = self.waveform
        return float(wf(t)) if callable(wf) else float(wf)


def _coerce_sources(sources: Any) -> list[CurrentSource]:
    if sources is None:
        return []
    if isinstance(sources, CurrentSource):
        return [sources]
    out = []
    for s in sources:
        if not isinstance(s, CurrentSource):
            raise ValueError(
                f"fdtd sources must be CurrentSource instances, got "
                f"{type(s).__name__}")
        out.append(s)
    names = [s.name for s in out]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate source names: {sorted(names)}")
    return out


# ==========================================================================
# CPML
# ==========================================================================
class CPML:
    """Convolutional perfectly matched layer (Roden-Gedney, with CFS).

    The PML replaces each coordinate derivative by a stretched one,

    ``d/dx_d  ->  (1/s_d) d/dx_d``,   ``s_d = kappa_d + sigma_d/(alpha_d + j w eps0)``,

    and the CPML form of that in the time domain is a recursive convolution: one
    auxiliary variable ``psi`` per (field location, axis) that obeys

    ``psi <- bb*psi + aa*(raw difference)``,
    ``stretched difference = (raw difference)/kappa + psi``,
    ``bb = exp(-(sigma/kappa + alpha)*dt/eps0)``,
    ``aa = sigma/(kappa*(sigma + kappa*alpha)) * (bb - 1)``.

    ``sigma`` and ``alpha`` are ``eps0``-normalised, matching
    :class:`fieldspice.boundaries.Absorbing`; they are *not* material
    conductivities and must not be added to ``sigma_cell``.  The
    complex-frequency shift ``alpha`` is what distinguishes CPML from a plain
    PML and is the reason it absorbs evanescent and near-DC content instead of
    amplifying it --- which is the whole ball game for a near-field circuit
    problem, where most of the energy hitting the boundary is reactive.

    Because this implementation stretches whole ``Cd`` sub-curls (see
    :func:`directional_curls`), no per-component index arithmetic appears
    anywhere, and the electric/magnetic half-cell offset comes out of the grid
    topology instead of being hand-written.

    Parameters
    ----------
    grid
        The grid.
    bc
        Boundary specification; only :class:`~fieldspice.boundaries.Absorbing`
        walls create a layer.
    dt
        Time step [s].  The recursion coefficients depend on it, so a CPML is
        tied to one ``dt``.
    eps_r_wall, mu_r_wall
        Relative permittivity/permeability of the medium the layer truncates,
        per wall, as ``{wall: value}``.  A PML matched to vacuum in front of a
        dielectric reflects, so these must describe the real material.

    Attributes
    ----------
    active
        True if any wall actually has a layer.
    """

    def __init__(self, grid: RectilinearGrid, bc: BoundarySpec, dt: float,
                 eps_r_wall: Mapping[str, float] | None = None,
                 mu_r_wall: Mapping[str, float] | None = None):
        self.grid = grid
        self.dt = float(dt)
        self._eps_r = dict(eps_r_wall or {})
        self._mu_r = dict(mu_r_wall or {})

        self.walls = tuple(w for w in WALLS if isinstance(bc.get(w), Absorbing))
        self.active = bool(self.walls)
        # Per axis: compressed auxiliary state for the b (Faraday) and
        # e (Ampere) updates.
        self._ikb: list[np.ndarray | None] = [None, None, None]
        self._ike: list[np.ndarray | None] = [None, None, None]
        self._idx_b: list[np.ndarray] = [np.zeros(0, np.intp)] * 3
        self._idx_e: list[np.ndarray] = [np.zeros(0, np.intp)] * 3
        self._aa_b: list[np.ndarray] = [np.zeros(0)] * 3
        self._bb_b: list[np.ndarray] = [np.zeros(0)] * 3
        self._aa_e: list[np.ndarray] = [np.zeros(0)] * 3
        self._bb_e: list[np.ndarray] = [np.zeros(0)] * 3
        self.psi_b: list[np.ndarray] = [np.zeros(0)] * 3
        self.psi_e: list[np.ndarray] = [np.zeros(0)] * 3
        self.info: dict[str, Any] = {}
        if not self.active:
            return

        ncell = grid.ncell
        nodes = (grid.xn, grid.yn, grid.zn)
        centres = (grid.xc, grid.yc, grid.zc)
        widths = (grid.hx, grid.hy, grid.hz)

        for axis in range(3):
            lo = bc.get(WALLS[2 * axis])
            hi = bc.get(WALLS[2 * axis + 1])
            if not isinstance(lo, Absorbing) and not isinstance(hi, Absorbing):
                continue
            n = ncell[axis]
            for side, wall_bc in ((0, lo), (1, hi)):
                if isinstance(wall_bc, Absorbing) and 2 * wall_bc.thickness > n:
                    raise ValueError(
                        f"CPML on {WALLS[2*axis+side]!r} is "
                        f"{wall_bc.thickness} cells thick but the axis has only "
                        f"{n} cells; the two layers would overlap. Enlarge the "
                        f"domain or reduce Absorbing(thickness=...).")

            sig_c, kap_c, alp_c = self._profile_1d(
                lo, hi, nodes[axis], widths[axis], centres[axis],
                WALLS[2 * axis], WALLS[2 * axis + 1])
            sig_n, kap_n, alp_n = self._profile_1d(
                lo, hi, nodes[axis], widths[axis], nodes[axis],
                WALLS[2 * axis], WALLS[2 * axis + 1])

            # Faraday side: Cd writes to the face groups whose index along
            # `axis` runs over cells, so it samples cell centres.
            sig_f = self._to_faces(axis, sig_c, 0.0)
            kap_f = self._to_faces(axis, kap_c, 1.0)
            alp_f = self._to_faces(axis, alp_c, 0.0)
            # Ampere side: Cd.T writes to the edge groups whose index along
            # `axis` runs over nodes.
            sig_g = self._to_edges(axis, sig_n, 0.0)
            kap_g = self._to_edges(axis, kap_n, 1.0)
            alp_g = self._to_edges(axis, alp_n, 0.0)

            self._ikb[axis] = 1.0 / kap_f if np.any(kap_f != 1.0) else None
            self._ike[axis] = 1.0 / kap_g if np.any(kap_g != 1.0) else None

            ib = np.flatnonzero(sig_f > 0.0)
            ie = np.flatnonzero(sig_g > 0.0)
            self._idx_b[axis] = ib
            self._idx_e[axis] = ie
            self._aa_b[axis], self._bb_b[axis] = self._coefficients(
                sig_f[ib], kap_f[ib], alp_f[ib])
            self._aa_e[axis], self._bb_e[axis] = self._coefficients(
                sig_g[ie], kap_g[ie], alp_g[ie])
            self.psi_b[axis] = np.zeros(ib.size)
            self.psi_e[axis] = np.zeros(ie.size)

            self.info[f"axis{axis}"] = {
                "faces_in_pml": int(ib.size),
                "edges_in_pml": int(ie.size),
                "sigma_max": float(sig_f.max()),
                "kappa_max": float(kap_f.max()),
            }

    # -- construction helpers ---------------------------------------------
    def _profile_1d(self, lo: BC, hi: BC, nodes: np.ndarray, h: np.ndarray,
                    pos: np.ndarray, wall_lo: str, wall_hi: str
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(sigma, kappa, alpha)`` at coordinates ``pos`` [m] along one axis.

        Depth is measured in **physical** distance normalised by the physical
        layer thickness, not in cells, so a graded mesh inside the layer still
        gets the intended ``sigma(x)`` profile.
        """
        n = h.size
        sig = np.zeros_like(pos, dtype=float)
        kap = np.ones_like(pos, dtype=float)
        alp = np.zeros_like(pos, dtype=float)
        for wall, wall_bc in ((wall_lo, lo), (wall_hi, hi)):
            if not isinstance(wall_bc, Absorbing):
                continue
            t = wall_bc.thickness
            if WALL_SIDE[wall] == 0:
                x_wall, x_in, cells = nodes[0], nodes[t], h[:t]
            else:
                x_wall, x_in, cells = nodes[n], nodes[n - t], h[n - t:]
            depth = np.clip((pos - x_in) / (x_wall - x_in), 0.0, 1.0)
            prof = wall_bc.profile(depth, dx=float(np.mean(cells)),
                                   eps_r=self._eps_r.get(wall, 1.0),
                                   mu_r=self._mu_r.get(wall, 1.0))
            sig = np.maximum(sig, prof["sigma"])
            kap = np.maximum(kap, prof["kappa"])
            # alpha is graded the opposite way, so it is only meaningful where
            # sigma > 0; outside the layer aa == 0 and it never enters.
            alp = np.maximum(alp, np.where(prof["sigma"] > 0.0,
                                           prof["alpha"], 0.0))
        return sig, kap, alp

    def _to_faces(self, axis: int, vals: np.ndarray, fill: float) -> np.ndarray:
        """Broadcast a per-cell profile along ``axis`` onto all faces.

        Face groups whose index along ``axis`` runs over nodes get ``fill``,
        because the axis-``axis`` sub-curl is identically zero on them.
        """
        n = self.grid.ncell[axis]
        out = []
        for shape in self.grid.shape_faces:
            arr = np.full(shape, fill, dtype=float)
            if shape[axis] == n:
                shp = [1, 1, 1]
                shp[axis] = n
                arr = np.broadcast_to(vals.reshape(shp), shape).copy()
            out.append(arr.ravel())
        return np.concatenate(out)

    def _to_edges(self, axis: int, vals: np.ndarray, fill: float) -> np.ndarray:
        """Broadcast a per-node profile along ``axis`` onto all edges."""
        n = self.grid.ncell[axis] + 1
        out = []
        for shape in self.grid.shape_edges:
            arr = np.full(shape, fill, dtype=float)
            if shape[axis] == n:
                shp = [1, 1, 1]
                shp[axis] = n
                arr = np.broadcast_to(vals.reshape(shp), shape).copy()
            out.append(arr.ravel())
        return np.concatenate(out)

    def _coefficients(self, sig: np.ndarray, kap: np.ndarray,
                      alp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Recursive-convolution coefficients ``(aa, bb)`` (dimensionless)."""
        bb = np.exp(-(sig / kap + alp) * self.dt / eps0)
        den = kap * (sig + kap * alp)
        aa = np.where(den > 0.0, sig / np.where(den > 0.0, den, 1.0) * (bb - 1.0),
                      0.0)
        return aa, bb

    # -- the update --------------------------------------------------------
    def reset(self) -> None:
        """Zero the convolution memory (call before re-running)."""
        for d in range(3):
            self.psi_b[d][:] = 0.0
            self.psi_e[d][:] = 0.0

    def stretch_b(self, axis: int, raw: np.ndarray) -> np.ndarray:
        """Stretched face difference for the Faraday update.

        Parameters
        ----------
        axis
            Difference axis (0, 1, 2).
        raw
            ``Cd @ e``, flat ``(n_faces,)`` [V].  Modified in place and
            returned, because it is a scratch buffer.

        Returns
        -------
        np.ndarray
            ``raw/kappa + psi`` [V].
        """
        ik = self._ikb[axis]
        idx = self._idx_b[axis]
        if idx.size:
            psi = self.psi_b[axis]
            psi *= self._bb_b[axis]
            psi += self._aa_b[axis] * raw[idx]
        if ik is not None:
            raw *= ik
        if idx.size:
            raw[idx] += self.psi_b[axis]
        return raw

    def stretch_e(self, axis: int, raw: np.ndarray) -> np.ndarray:
        """Stretched edge difference for the Ampere update.

        Parameters
        ----------
        axis
            Difference axis (0, 1, 2).
        raw
            ``Cd.T @ (nu*b)``, flat ``(n_edges,)`` [A].  Modified in place.

        Returns
        -------
        np.ndarray
            ``raw/kappa + psi`` [A].
        """
        ik = self._ike[axis]
        idx = self._idx_e[axis]
        if idx.size:
            psi = self.psi_e[axis]
            psi *= self._bb_e[axis]
            psi += self._aa_e[axis] * raw[idx]
        if ik is not None:
            raw *= ik
        if idx.size:
            raw[idx] += self.psi_e[axis]
        return raw

    def __repr__(self) -> str:
        if not self.active:
            return "<CPML inactive>"
        nb = sum(i.size for i in self._idx_b)
        return (f"<CPML walls={self.walls} faces_absorbing={nb} "
                f"dt={self.dt:.4g} s>")


# ==========================================================================
# The solver
# ==========================================================================
class FDTDSolver(TimeSteppingSolver):
    """Explicit Yee FDTD on the frozen fieldspice grid and operators.

    ::

        b  -= dt * (C @ e)
        e   = ca*e + cb*(C.T @ (nu*b) - i_src)

    with ``ca = exp(-dt*sigma/eps)`` and ``cb = dt/M_eps * phi1(dt*sigma/eps)``
    the exponential-time-differencing coefficients (module docstring).  Lossless
    cells reduce to ``ca = 1``, ``cb = dt/M_eps`` exactly, so a vacuum run is
    bit-for-bit the classical Yee scheme.

    Parameters
    ----------
    grid
        The grid.  1D and 2D problems are grids with collapsed directions and
        need no special handling; the collapsed thickness is real and does enter
        the stability limit.
    eps_cell
        Absolute permittivity per cell [F/m], shape ``grid.shape_cells``.
    sigma_cell
        Conductivity per cell [S/m], same shape.  ``None`` means lossless.
    mu_cell
        Absolute permeability per cell [H/m], same shape.  ``None`` means
        ``mu0`` everywhere.
    config
        :class:`~fieldspice.solvers.base.SolverConfig`.  Only ``verbose`` and
        ``store_every`` are used; there is no linear solve to configure.
    operators
        Shared :class:`~fieldspice.operators.Operators` bundle.
    pec_cells
        Optional bool array, shape ``grid.shape_cells``: cells that are perfect
        conductors.  Every edge touching such a cell is held at zero, which is
        the exact and free treatment of a good conductor (**A4b**) and avoids
        both the skin-depth meshing and the Courant penalty it would carry.

    Attributes
    ----------
    name
        ``"fdtd"``.
    assumptions
        ``("A3", "A4b", "A9")``.

    Examples
    --------
    >>> import numpy as np
    >>> from fieldspice.grid import RectilinearGrid
    >>> from fieldspice.units import eps0, mu0, c0
    >>> g = RectilinearGrid.uniform([(0.0, 1.0)], [200])
    >>> s = FDTDSolver(g, np.full(g.shape_cells, eps0))
    >>> dt = s.stable_dt(safety=1.0, method="exact")
    >>> bool(abs(dt - (1.0 / 200) / c0) < 1e-4 * dt)
    True
    """

    name = "fdtd"
    assumptions = ("A3", "A4b", "A9")

    def __init__(self, grid: RectilinearGrid,
                 eps_cell: np.ndarray,
                 sigma_cell: np.ndarray | None = None,
                 mu_cell: np.ndarray | None = None,
                 config: SolverConfig | None = None,
                 operators: Operators | None = None,
                 pec_cells: np.ndarray | None = None):
        super().__init__(grid, config, operators)
        shape = grid.shape_cells

        eps_cell = np.asarray(eps_cell, dtype=float)
        if eps_cell.shape != shape:
            raise ValueError(
                f"eps_cell must have shape {shape}, got {eps_cell.shape}")
        if np.any(eps_cell <= 0.0):
            raise ValueError("eps_cell must be positive everywhere [F/m]")

        if sigma_cell is None:
            sigma_cell = np.zeros(shape)
        else:
            sigma_cell = np.asarray(sigma_cell, dtype=float)
            if sigma_cell.shape != shape:
                raise ValueError(
                    f"sigma_cell must have shape {shape}, got {sigma_cell.shape}")
            if np.any(sigma_cell < 0.0):
                raise ValueError("sigma_cell must be non-negative [S/m]")

        if mu_cell is None:
            mu_cell = np.full(shape, mu0)
        else:
            mu_cell = np.asarray(mu_cell, dtype=float)
            if mu_cell.shape != shape:
                raise ValueError(
                    f"mu_cell must have shape {shape}, got {mu_cell.shape}")
            if np.any(mu_cell <= 0.0):
                raise ValueError("mu_cell must be positive everywhere [H/m]")

        self.eps_cell = eps_cell
        self.sigma_cell = sigma_cell
        self.mu_cell = mu_cell

        # Material averaging: parallel (dual-area-weighted arithmetic) along an
        # edge, series (harmonic) along the dual edge threading a face.  These
        # are the combinations that make the mass matrices exact circuit
        # elements; see operators.cell_to_edge / cell_to_face.
        self.eps_edge = cell_to_edge(grid, eps_cell, mode="parallel")
        self.sigma_edge = cell_to_edge(grid, sigma_cell, mode="parallel")
        self.mu_face = cell_to_face(grid, mu_cell, mode="series")
        self.nu = face_reluctance(grid, self.mu_face)

        L = np.concatenate([a.ravel() for a in grid.edge_lengths()])
        A = np.concatenate([a.ravel() for a in grid.edge_dual_areas()])
        self._edge_len = L
        self._edge_area = A
        self.m_eps = self.eps_edge * A / L          # [F]
        self.m_sigma = self.sigma_edge * A / L      # [S]

        self.pec_cells = None
        self._pec_edges = np.zeros(0, dtype=np.intp)
        if pec_cells is not None:
            pec_cells = np.asarray(pec_cells)
            if pec_cells.shape != shape:
                raise ValueError(
                    f"pec_cells must have shape {shape}, got {pec_cells.shape}")
            self.pec_cells = pec_cells.astype(bool)
            # An edge is inside or on the surface of the metal if *any* cell
            # touching it is metal; that is the standard Yee staircase PEC and
            # is what makes E_tan vanish on the surface rather than half a cell
            # inside it.
            touch = cell_to_edge(grid, self.pec_cells.astype(float), mode="max")
            self._pec_edges = np.flatnonzero(touch > 0.0)

        self._Cd: tuple[sp.csr_matrix, ...] | None = None
        self._lam_gersh: float | None = None
        self._lam_exact: float | None = None

    # -- operators ---------------------------------------------------------
    @property
    def Cd(self) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
        """Cached ``(Cx, Cy, Cz)`` axis-split curls (see :func:`directional_curls`)."""
        if self._Cd is None:
            self._Cd = directional_curls(self.grid)
        return self._Cd  # type: ignore[return-value]

    def _sym_operator(self) -> sp.csr_matrix:
        """``M_eps^-1/2 C.T Nu C M_eps^-1/2`` --- symmetric PSD, same spectrum
        as the update operator whose largest eigenvalue sets the Courant limit.
        """
        C = self.ops.C
        d = 1.0 / np.sqrt(self.m_eps)
        S = sp.diags(d) @ (C.T @ sp.diags(self.nu) @ C) @ sp.diags(d)
        return sp.csr_matrix(S)

    # -- stability ---------------------------------------------------------
    EXACT_DT_MAX_EDGES = 250_000
    """Above this many edges, ``stable_dt(method="auto")`` stops paying for the
    Lanczos eigensolve and falls back to the Gershgorin bound.  Measured
    single-threaded eigensolve cost on uniform vacuum grids: 0.4 s at 1.6e3
    edges, 3.1 s at 8.6e4, 11.7 s at 2.0e5, 72 s at 6.7e5."""

    def stable_dt(self, safety: float = 0.99, method: str = "auto") -> float:
        """Largest stable time step of *this* discrete system [s].

        The lossless leapfrog is the standard symplectic integrator for
        ``d2e/dt2 = -M_eps^-1 C.T Nu C e``, and is stable exactly when
        ``dt < 2/sqrt(lambda_max)`` with ``lambda_max`` the largest eigenvalue
        of that operator [1/s^2].  This is the *true* limit for a graded,
        inhomogeneous grid, not the uniform-cell formula: it automatically
        accounts for grading, anisotropic cells, material contrast, and the
        finite thickness of a collapsed direction.

        Parameters
        ----------
        safety
            Multiplier applied to the limit (dimensionless).  0.99 by default;
            running exactly at the limit is marginally stable and accumulates
            round-off.
        method
            ``"exact"`` runs a Lanczos eigensolve
            (``scipy.sparse.linalg.eigsh``) for the true spectral radius.  On
            uniform vacuum grids it reproduces ``1/(c*sqrt(sum 1/h^2))`` to
            1e-6 (1D, 2D) and to 1e-9 (3D).

            ``"gershgorin"`` uses the Gershgorin disc bound on ``lambda_max``.
            It is rigorous, costs one sparse row-sum, and is *conservative* ---
            it can only under-estimate the allowed step, so it can never hand
            back an unstable ``dt``.  Measured, it returns 0.949 / 0.927 /
            0.806 of the true limit in 1D / 2D / 3D, i.e. it costs up to 24%
            more steps.

            ``"auto"`` (default) is ``"exact"`` up to
            :attr:`EXACT_DT_MAX_EDGES` edges and ``"gershgorin"`` above, which
            keeps the estimate itself from becoming the expensive part of a
            large run.

        Returns
        -------
        float
            Stable time step [s].

        Notes
        -----
        Conductive loss does **not** enter: the exponential update is
        unconditionally stable in ``sigma`` (module docstring), so the Courant
        limit is set by the lossless wave operator alone.  A CPML with
        ``kappa_max > 1`` only relaxes the limit, so using the PML-free operator
        stays on the safe side.
        """
        if not 0.0 < safety <= 1.0:
            raise ValueError("safety must lie in (0, 1]")
        if method == "auto":
            method = ("exact" if self.grid.n_edges <= self.EXACT_DT_MAX_EDGES
                      else "gershgorin")
        if method == "gershgorin":
            if self._lam_gersh is None:
                S = self._sym_operator()
                self._lam_gersh = float(abs(S).sum(axis=1).max())
            lam = self._lam_gersh
        elif method == "exact":
            if self._lam_exact is None:
                S = self._sym_operator()
                if S.shape[0] <= 3:
                    self._lam_exact = float(abs(S).sum(axis=1).max())
                else:
                    try:
                        val = spla.eigsh(S, k=1, which="LA",
                                         return_eigenvectors=False, tol=1e-6)
                        self._lam_exact = float(val[0])
                    except spla.ArpackNoConvergence:
                        warnings.warn(
                            "Lanczos did not converge on the Courant "
                            "eigenvalue; falling back to the (conservative) "
                            "Gershgorin bound.", RuntimeWarning, stacklevel=2)
                        self._lam_exact = float(abs(S).sum(axis=1).max())
            lam = self._lam_exact
        else:
            raise ValueError(
                f"unknown method {method!r}; expected 'gershgorin' or 'exact'")
        if lam <= 0.0:
            raise ValueError(
                "the curl-curl operator has no positive spectrum; the grid has "
                "no resolved direction")
        return float(safety * 2.0 / np.sqrt(lam))

    # -- boundary handling -------------------------------------------------
    def _pec_wall_edges(self, bc: BoundarySpec) -> np.ndarray:
        """Flat indices of tangential edges on every electric (PEC) wall.

        Mapping from the boundary vocabulary to full-wave physics:

        =====================  =====================================
        BC                     full-wave meaning
        =====================  =====================================
        ``Symmetry`` electric  PEC: tangential E = 0
        ``Symmetry`` magnetic  PMC: natural, nothing to do
        ``Dirichlet``          PEC (a fixed potential is a conductor)
        ``Neumann``            PMC, the natural condition of ``C``
        ``Absorbing``          CPML, terminated by PEC at the outer wall
        ``Periodic``           not implemented, raises
        =====================  =====================================

        The *natural* condition of the bare incidence operator is PMC, because
        the dual contour of a boundary edge is simply truncated --- no ghost
        ``h`` outside means no magnetic field outside.  PEC therefore has to be
        imposed, and PMC has to be left alone; getting that backwards is the
        classic sign that a full-wave code was written by analogy with a nodal
        one.
        """
        keep: list[np.ndarray] = []
        n = self.grid.ncell
        for wall in WALLS:
            w = bc.get(wall)
            if isinstance(w, Periodic):
                raise ValueError(
                    "fdtd does not implement periodic boundaries; use "
                    "Symmetry/Neumann walls or an Absorbing layer")
            if isinstance(w, Symmetry) and not w.is_electric:
                continue
            if isinstance(w, Neumann):
                continue
            if not isinstance(w, (Dirichlet, Symmetry, Absorbing)):
                raise ValueError(
                    f"fdtd does not understand boundary {type(w).__name__} on "
                    f"wall {wall!r}")
            axis = WALL_AXIS[wall]
            plane = 0 if WALL_SIDE[wall] == 0 else n[axis]
            for d, shape in enumerate(self.grid.shape_edges):
                if d == axis:
                    continue  # normal to the wall: not constrained by PEC
                off = (0, self.grid.n_edges_each[0],
                       self.grid.n_edges_each[0] + self.grid.n_edges_each[1])[d]
                sel = [slice(None)] * 3
                sel[axis] = slice(plane, plane + 1)
                keep.append(_ids(shape, off)[tuple(sel)].ravel())
        if not keep:
            return np.zeros(0, dtype=np.intp)
        return np.unique(np.concatenate(keep)).astype(np.intp)

    def _pml_material(self, bc: BoundarySpec
                      ) -> tuple[dict[str, float], dict[str, float]]:
        """Mean relative eps and mu of the cells each absorbing layer sits in.

        A PML is matched to the medium it truncates; matching to vacuum in front
        of a dielectric reflects at the ``sqrt(eps_r)`` level.  Taking the mean
        over the layer cells is automatic and right for the usual case (a
        homogeneous background); if a layer straddles a material interface, no
        single ``sigma_max`` can match both and the interface should be moved
        out of the layer instead.
        """
        eps_r: dict[str, float] = {}
        mu_r: dict[str, float] = {}
        n = self.grid.ncell
        for wall in WALLS:
            w = bc.get(wall)
            if not isinstance(w, Absorbing):
                continue
            axis = WALL_AXIS[wall]
            t = min(w.thickness, n[axis])
            sel: list[Any] = [slice(None)] * 3
            sel[axis] = slice(0, t) if WALL_SIDE[wall] == 0 else slice(n[axis] - t, None)
            eps_r[wall] = float(np.mean(self.eps_cell[tuple(sel)]) / eps0)
            mu_r[wall] = float(np.mean(self.mu_cell[tuple(sel)]) / mu0)
        return eps_r, mu_r

    # -- update coefficients ----------------------------------------------
    def _etd_coefficients(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        """``(ca, cb)`` for ``e <- ca*e + cb*f`` (dimensionless, and V/A/s... ).

        ``ca`` is dimensionless; ``cb`` has units of V per (A s) --- it is
        ``dt/C`` corrected by ``phi1``, so ``cb * current`` is a voltage.
        """
        k = dt * self.sigma_edge / self.eps_edge
        ca = np.exp(-k)
        # phi1(k) = (1 - exp(-k))/k, expanded near 0 to keep full precision.
        small = k < 1e-8
        ksafe = np.where(small, 1.0, k)
        phi1 = np.where(small, 1.0 - 0.5 * k, -np.expm1(-k) / ksafe)
        cb = (dt / self.m_eps) * phi1
        return ca, cb

    # -- the main entry point ---------------------------------------------
    def solve(self, sources: Any = None, t_end: float = 0.0,
              dt: float | None = None,
              bc: BoundarySpec | None = None,
              monitors: Any = None,
              e0: np.ndarray | None = None,
              b0: np.ndarray | None = None,
              store: Sequence[str] = (),
              store_every: int = 0,
              edge_probes: Mapping[str, Any] | None = None,
              f_max: float | None = None,
              record_energy: bool = True) -> Result:
        """Run the explicit leapfrog.

        Parameters
        ----------
        sources
            A :class:`CurrentSource`, an iterable of them, or ``None``.  The
            current is impressed at the half step, where the leapfrog wants it.
        t_end
            Stop time [s].  The run covers ``ceil(t_end/dt)`` steps.
        dt
            Time step [s].  ``None`` uses ``stable_dt(safety=0.99)``.  A ``dt``
            above the true stability limit raises :class:`ValueError` --- it is
            never silently clipped, because an unstable run that looks like a
            run is worse than no run.
        bc
            Wall conditions.  Default: PMC (``Neumann``) on all six walls, a
            closed magnetic box.  See :meth:`_pec_wall_edges` for the mapping.
        monitors
            ``None``, a :class:`~fieldspice.monitors.Monitor`, an iterable, or a
            :class:`~fieldspice.monitors.MonitorSet`.  The solver publishes
            ``e``, ``b``, ``eps_edge``, ``sigma_edge``, ``mu_face`` and
            ``m_nu`` (the reluctance it actually used) into the monitor state.
        e0, b0
            Optional initial edge circulation [V] and face flux [Wb].  ``e0`` is
            at ``t = 0``, ``b0`` at ``t = -dt/2``.  Default: zero.
        store
            Names from ``{"e", "b"}`` to snapshot into ``Result.fields``.
        store_every
            Snapshot stride in steps.  0 (default) means store only the final
            state of each requested field.
        edge_probes
            ``{name: edge index or index array}``.  Records the summed edge
            circulation [V] at every step into ``Result.scalars``; this is the
            cheap way to get a time trace without allocating a field history.
        f_max
            Highest frequency of interest [Hz].  When given, the skin depth of
            every conductor is checked against the mesh (A4a/A4b) and a warning
            is issued if it is under-resolved.
        record_energy
            Record stored energy and dissipated power each step.

        Returns
        -------
        Result
            ``scalars`` holds ``"energy"`` [J] (the exactly-conserved
            interleaved leapfrog invariant), ``"energy_e"``, ``"energy_b"``,
            ``"dissipation"`` [W] and the probes; ``terminals`` holds each
            source's impressed current [A] and the voltage across its edges [V];
            ``meta`` records ``dt``, the Courant ratio, the CPML configuration
            and the assumptions.

        Notes
        -----
        The interleaved invariant is

        ``U^n = 0.5 e^n.M_eps e^n + 0.5 b^(n-1/2).Nu b^(n+1/2)``

        which the leapfrog conserves **exactly** (to round-off) in a lossless
        PEC box, unlike the naive same-time sum, which oscillates at ``O(dt^2)``
        because ``b`` is half a step out.  Both are reported; ``"energy"`` is
        the invariant, ``"energy_naive"`` is the sum of the two same-time terms.
        """
        from ..monitors import MonitorSet

        self._start()
        grid = self.grid
        src_list = _coerce_sources(sources)
        bc = BoundarySpec.all_neumann() if bc is None else bc
        if not isinstance(bc, BoundarySpec):
            raise ValueError("bc must be a BoundarySpec")
        t_end = float(t_end)
        if t_end < 0.0:
            raise ValueError("t_end must be non-negative [s]")

        # --- time step ----------------------------------------------------
        dt_limit = self.stable_dt(safety=1.0, method="auto")
        if dt is None:
            dt = 0.99 * dt_limit
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive [s]")
        if dt > dt_limit and self._lam_exact is None:
            # "auto" fell back to the conservative Gershgorin bound. Refusing a
            # run is a big enough decision to pay for the true spectral radius
            # first, rather than rejecting a legitimate step for the sake of a
            # cheap estimate.
            dt_limit = self.stable_dt(safety=1.0, method="exact")
        if dt > dt_limit:
            raise ValueError(
                f"dt = {dt:.6g} s exceeds the Courant stability limit "
                f"{dt_limit:.6g} s of this grid by a factor "
                f"{dt/dt_limit:.4g}. The explicit leapfrog would diverge "
                f"exponentially; use dt <= stable_dt(), coarsen the mesh, "
                f"or switch to a quasi-static solver (see docs A1).")

        n_steps = int(np.ceil(t_end / dt)) if t_end > 0.0 else 0

        # --- conductor sanity (A4b) ---------------------------------------
        self._check_conductors(dt, f_max)

        # --- state --------------------------------------------------------
        e = np.zeros(grid.n_edges) if e0 is None else np.array(
            np.asarray(e0, dtype=float).ravel(), copy=True)
        if e.size != grid.n_edges:
            raise ValueError(f"e0 must have {grid.n_edges} entries")
        b = np.zeros(grid.n_faces) if b0 is None else np.array(
            np.asarray(b0, dtype=float).ravel(), copy=True)
        if b.size != grid.n_faces:
            raise ValueError(f"b0 must have {grid.n_faces} entries")

        pec = np.unique(np.concatenate(
            [self._pec_edges, self._pec_wall_edges(bc)])).astype(np.intp)
        e[pec] = 0.0

        ca, cb = self._etd_coefficients(dt)
        eps_r_wall, mu_r_wall = self._pml_material(bc)
        pml = CPML(grid, bc, dt, eps_r_wall, mu_r_wall)
        C = self.ops.C
        Cd = self.Cd if pml.active else None
        nu = self.nu

        store = tuple(store)
        for nm in store:
            if nm not in ("e", "b"):
                raise ValueError(f"store name {nm!r} must be 'e' or 'b'")
        if store_every < 0:
            raise ValueError("store_every must be >= 0")

        probes: dict[str, np.ndarray] = {}
        for nm, idx in (edge_probes or {}).items():
            ii = np.atleast_1d(np.asarray(idx, dtype=np.intp)).ravel()
            if ii.size == 0 or ii.min() < 0 or ii.max() >= grid.n_edges:
                raise ValueError(
                    f"edge probe {nm!r} has out-of-range or empty indices")
            probes[nm] = ii

        mset = MonitorSet.coerce(monitors)

        # --- output buffers ----------------------------------------------
        nrec = n_steps + 1
        t_axis = dt * np.arange(nrec)
        probe_out = {nm: np.zeros(nrec) for nm in probes}
        src_i = {s.name: np.zeros(nrec) for s in src_list}
        src_v = {s.name: np.zeros(nrec) for s in src_list}
        en = np.zeros(nrec) if record_energy else None
        en_e = np.zeros(nrec) if record_energy else None
        en_b = np.zeros(nrec) if record_energy else None
        en_naive = np.zeros(nrec) if record_energy else None
        diss = np.zeros(nrec) if record_energy else None
        snaps: dict[str, list[np.ndarray]] = {nm: [] for nm in store}
        snap_t: list[float] = []

        i_src = np.zeros(grid.n_edges)
        b_prev = b.copy()

        def _record(step: int, t: float) -> None:
            for nm, ii in probes.items():
                probe_out[nm][step] = float(e[ii].sum())
            for s in src_list:
                src_i[s.name][step] = s.value_at(t)
                src_v[s.name][step] = float(e[s.edges].sum())
            if record_energy:
                we = 0.5 * float(e @ (self.m_eps * e))
                wb_exact = 0.5 * float(b_prev @ (nu * b))
                wb_naive = 0.5 * float(b @ (nu * b))
                en_e[step] = we                      # type: ignore[index]
                en_b[step] = wb_naive                # type: ignore[index]
                en[step] = we + wb_exact             # type: ignore[index]
                en_naive[step] = we + wb_naive       # type: ignore[index]
                diss[step] = float(e @ (self.m_sigma * e))  # type: ignore[index]
            if store and (store_every > 0) and (step % store_every == 0):
                for nm in store:
                    snaps[nm].append((e if nm == "e" else b).copy())
                snap_t.append(t)
            if len(mset):
                mset.record(self._state(e, b, step, src_list, t), t)

        _record(0, 0.0)
        self._log(1, f"dt = {dt:.4g} s ({dt/dt_limit:.3f} of the Courant "
                     f"limit), {n_steps} steps, CPML {pml.walls or 'none'}")

        # --- the loop -----------------------------------------------------
        for n in range(n_steps):
            t_half = (n + 0.5) * dt
            np.copyto(b_prev, b)

            # Faraday: b^(n+1/2) = b^(n-1/2) - dt * curl(e^n)
            if Cd is None:
                b -= dt * (C @ e)
            else:
                acc = pml.stretch_b(0, Cd[0] @ e)
                acc += pml.stretch_b(1, Cd[1] @ e)
                acc += pml.stretch_b(2, Cd[2] @ e)
                b -= dt * acc

            # Ampere: e^(n+1) from e^n, the dual curl of h, and the source.
            h = nu * b
            if Cd is None:
                rhs = C.T @ h
            else:
                rhs = pml.stretch_e(0, Cd[0].T @ h)
                rhs += pml.stretch_e(1, Cd[1].T @ h)
                rhs += pml.stretch_e(2, Cd[2].T @ h)
            if src_list:
                i_src[:] = 0.0
                for s in src_list:
                    np.add.at(i_src, s.edges, s.value_at(t_half) * s.weight)
                rhs -= i_src
            e *= ca
            e += cb * rhs
            e[pec] = 0.0

            _record(n + 1, (n + 1) * dt)

        # --- pack ---------------------------------------------------------
        res = Result(grid=grid, t=t_axis)
        if record_energy:
            res.scalars["energy"] = en          # type: ignore[assignment]
            res.scalars["energy_e"] = en_e      # type: ignore[assignment]
            res.scalars["energy_b"] = en_b      # type: ignore[assignment]
            res.scalars["energy_naive"] = en_naive  # type: ignore[assignment]
            res.scalars["dissipation"] = diss   # type: ignore[assignment]
        for nm, arr in probe_out.items():
            res.scalars[nm] = arr
        for s in src_list:
            res.terminals[s.name] = {"v": src_v[s.name], "i": src_i[s.name]}
        for nm in store:
            if snaps[nm]:
                res.fields[nm] = np.array(snaps[nm])
            else:
                res.fields[nm] = (e if nm == "e" else b)[None, :].copy()
        if store and store_every > 0:
            res.fields["snapshot_t"] = np.array(snap_t)
        res.fields.setdefault("e_final", e[None, :].copy())
        res.fields.setdefault("b_final", b[None, :].copy())
        if len(mset):
            mon_out = mset.finalize()
            mon_out.pop("t", None)   # the solver already owns the time axis
            res.scalars.update(mon_out)
            res.terminals.update(mset.terminal_series())

        return self._finish(
            res,
            dt=dt,
            n_steps=n_steps,
            stable_dt=dt_limit,
            courant_ratio=dt / dt_limit,
            grid_courant_dt=grid.courant_dt(),
            pml=({"walls": list(pml.walls), **pml.info} if pml.active
                 else {"walls": []}),
            pec_edges=int(pec.size),
            boundaries={w: bc.get(w).describe() for w in WALLS},
        )

    # -- helpers -----------------------------------------------------------
    def _state(self, e: np.ndarray, b: np.ndarray, step: int,
               sources: Sequence[CurrentSource], t: float) -> dict[str, Any]:
        """Monitor state dictionary for one step (see monitors.py)."""
        return {
            "grid": self.grid,
            "ops": self.ops,
            "e": e,
            "b": b,
            "eps_edge": self.eps_edge,
            "sigma_edge": self.sigma_edge,
            "mu_face": self.mu_face,
            "m_nu": self.nu,
            "step": step,
            "terminals": {},
            "terminal_current": {s.name: s.value_at(t) for s in sources},
            "terminal_voltage": {s.name: float(e[s.edges].sum())
                                 for s in sources},
        }

    def _check_conductors(self, dt: float, f_max: float | None) -> None:
        """Warn about conductors this discretisation cannot honestly resolve (A4b)."""
        k = dt * self.sigma_edge / self.eps_edge
        n_good = int(np.count_nonzero(k > 10.0))
        if n_good and self._pec_edges.size == 0:
            worst = float(k.max())
            warnings.warn(
                f"A4b: {n_good} edges have dt*sigma/eps > 10 (worst "
                f"{worst:.3g}), i.e. they are good conductors on this time "
                f"step. The exponential update keeps them stable and relaxes "
                f"them to the correct resistive limit, but the field *inside* "
                f"the metal is only meaningful if the mesh resolves the skin "
                f"depth. Prefer pec_cells= for a good conductor; a surface "
                f"impedance boundary is the other correct option and is not "
                f"implemented here.",
                RuntimeWarning, stacklevel=3)
        if f_max is not None:
            from ..validate import check_skin_depth
            rep = check_skin_depth(self.grid, self.sigma_cell, self.mu_cell,
                                   float(f_max))
            if rep.level != "ok":
                warnings.warn(f"A4a/A4b: {rep.message}", RuntimeWarning,
                              stacklevel=3)

    # -- measurement -------------------------------------------------------
    @staticmethod
    def measure_pml_reflection(n_cells: int = 200, dx: float = 20e-6,
                               thickness: int = 10, order: float = 3.0,
                               kappa_max: float = 1.0, alpha_max: float = 0.05,
                               sigma_max: float | None = None,
                               tau_steps: float = 25.0,
                               pad: int = 4) -> dict[str, float]:
        """Measure the CPML's normal-incidence reflection from scratch [dB].

        A short domain terminated by the layer under test is compared against a
        reference domain ``pad`` times longer, run for the same duration so that
        the reference's own boundary reflection has not yet reached the probe.
        The difference between the two probe traces is, by construction, the
        light the layer failed to absorb --- including every discretisation
        error, which is exactly what a theoretical ``R = exp(-2 eta int sigma)``
        estimate hides.

        Parameters
        ----------
        n_cells
            Cells in the test domain, including the layer (dimensionless).
        dx
            Uniform cell size [m].
        thickness, order, kappa_max, alpha_max, sigma_max
            CPML parameters, passed to
            :class:`~fieldspice.boundaries.Absorbing`.
        tau_steps
            Gaussian pulse width in time steps (dimensionless).
        pad
            Length multiplier of the reference domain (dimensionless).

        Returns
        -------
        dict
            ``"reflection"`` (peak amplitude ratio, dimensionless),
            ``"reflection_dB"``, ``"reflection_rms_dB"``, ``"probe_peak"`` [V],
            and the run parameters.
        """
        from ..grid import RectilinearGrid

        def _run(ncell: int, absorbing: bool) -> tuple[np.ndarray, float]:
            g = RectilinearGrid.uniform([(0.0, ncell * dx)], [ncell])
            slv = FDTDSolver(g, np.full(g.shape_cells, eps0))
            dt = slv.stable_dt(0.99)
            if absorbing:
                lay = Absorbing(thickness=thickness, order=order,
                                kappa_max=kappa_max, alpha_max=alpha_max,
                                sigma_max=sigma_max)
                spec = BoundarySpec(xlo=lay, xhi=lay)
            else:
                spec = BoundarySpec.all_neumann()
            i_src = 20              # source node plane
            i_probe = 40            # probe node plane
            tau = tau_steps * dt
            t0 = 4.0 * tau
            src = CurrentSource(
                edges=g.edge_index(1, i_src, 0, 0),
                waveform=lambda t: np.exp(-((t - t0) / tau) ** 2),
                name="drive")
            n_steps = int(round(1.6 * ncell))
            res = slv.solve(src, t_end=n_steps * dt, dt=dt, bc=spec,
                            edge_probes={"p": g.edge_index(1, i_probe, 0, 0)},
                            record_energy=False)
            return res.scalars["p"], dt

        test, dt = _run(n_cells, True)
        ref, _ = _run(n_cells * pad, False)
        n = min(test.size, ref.size)
        test, ref = test[:n], ref[:n]
        peak = float(np.max(np.abs(ref)))
        err = test - ref
        r = float(np.max(np.abs(err)) / peak)
        rms = float(np.sqrt(np.mean(err ** 2))
                    / np.sqrt(np.mean(ref ** 2)))
        return {
            "reflection": r,
            "reflection_dB": float(20.0 * np.log10(max(r, 1e-300))),
            "reflection_rms_dB": float(20.0 * np.log10(max(rms, 1e-300))),
            "probe_peak": peak,
            "n_cells": float(n_cells),
            "dx": dx,
            "thickness": float(thickness),
            "dt": dt,
        }
