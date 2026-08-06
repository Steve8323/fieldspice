"""Drift-diffusion (van Roosbroeck) semiconductor solver.

This is the module that turns ``fieldspice`` from an interconnect tool into a
device tool.  It solves the coupled system

.. code-block:: text

    G^T M_eps G psi = q (p - n + Nd - Na) V_node            (Poisson / Gauss)
    dn/dt = +(1/q) div Jn - R                               (electrons)
    dp/dt = -(1/q) div Jp - R                               (holes)

on the node/edge complex of :mod:`fieldspice.operators`, with the box method
(**A10**) supplying the control volumes and Scharfetter-Gummel exponential
fitting (**A7**) supplying the edge fluxes.

Discretisation
--------------
Every unknown lives on a **node**: ``psi`` [V], ``n`` [m^-3], ``p`` [m^-3].
Every flux lives on an **edge**.  For an edge running from node ``t`` (tail) to
node ``h`` (head), of length ``L`` [m] and dual area ``A`` [m^2], write
``X = (psi_h - psi_t)/Vt``.  Scharfetter-Gummel gives the *conventional*
currents carried along the edge direction

.. code-block:: text

    I_n = q Dn (A/L) [ n_h B(X)  - n_t B(-X) ]      [A]
    I_p = q Dp (A/L) [ p_t B(X)  - p_h B(-X) ]      [A]

with ``B(x) = x/(exp(x) - 1)``.  The hole form is *derived*, not guessed: for
``Jp = q mu_p p E - q Dp dp/dx`` with ``psi`` linear and ``Jp`` constant along
the edge, integrating gives ``Jp = (q Dp/L)(p_t B(X) - p_h B(-X))``.  The two
expressions map onto each other under ``n -> p``, ``X -> -X``, which is the
statement that holes are positive carriers.  Getting this backwards makes the
diode conduct in reverse, so the sign is checked directly in the module's
self-test (``python -m fieldspice.solvers.dd``).

Because ``G^T i`` is the net current *into* a node (see
``docs/CONTRACTS.md``), the residuals are

.. code-block:: text

    F_psi = G^T M_eps G psi - q (p - n + C) V_semi
    F_n   = +q V_semi (n - n_old)/dt + (G^T I_n) + q R V_semi
    F_p   = -q V_semi (p - p_old)/dt + (G^T I_p) - q R V_semi

and the current the outside world must push into a contact node to sustain the
solution is ``-(F_n + F_p)``, with the displacement part ``d/dt F_psi``.  That
identity is what makes terminal currents exact rather than post-processed: the
current is read off the residual of the equation the contact node is *not*
allowed to satisfy.

Scaling (stated explicitly, as required)
----------------------------------------
Solving in raw SI gives a Jacobian with entries spanning ``1e-19`` (charge) to
``1e23`` (densities); its condition number is ~1e30 and Newton dies.  Two
scalings are applied:

1. **Column (unknown) scaling.**  ``psi`` is measured in units of the thermal
   voltage ``Vt = kT/q``; ``n`` and ``p`` in units of a reference density
   ``N_ref = max(max|Nd - Na|, ni)``.  The Newton system is solved for
   ``dy = (dpsi/Vt, dn/N_ref, dp/N_ref)``.
2. **Row equilibration.**  Each row of the column-scaled Jacobian is divided by
   its own infinity norm, so every equation has a largest coefficient of one.
   This absorbs the per-node volume factor of a graded mesh and the enormous
   ratio between an oxide row and a silicon row.

The residual norm reported and used by the line search is the row-equilibrated
one, with the scaling frozen during a line search so the comparison is
meaningful.  Dirichlet rows are identically zero and are excluded from the norm
(diluting the norm with zeros is a documented trap in ``docs/CONTRACTS.md``).

Physics included
----------------
* Boltzmann statistics (**A5**), complete ionisation (**A11**), isothermal
  (**A6**), local drift-diffusion transport with the Einstein relation.
* Shockley-Read-Hall recombination with arbitrary trap level, and Auger.
* Caughey-Thomas field-dependent mobility; optional Masetti doping-dependent
  mobility (silicon coefficients).
* Ideal ohmic contacts (charge neutrality plus mass action at the contact) and
  ideal insulated gate contacts.

Physics excluded
----------------
Fermi-Dirac statistics and bandgap narrowing (so results above ~1e19 cm^-3 are
qualitative), tunnelling of any kind, impact ionisation, heterojunction band
offsets (a map containing two different semiconductors is rejected rather than
silently mis-solved), self-heating, and quantum confinement.  See
``docs/ASSUMPTIONS.md`` A5.

Validation status
-----------------
Reproduced and measured (see the module self-test):

* built-in potential against :func:`fieldspice.reference.built_in_potential`
  to ~1e-15 relative,
* Shockley ideality ``n = 1`` over five decades of forward current,
* depletion width scaling ``W ~ sqrt(Vbi + Vr)``,
* 60 mV/decade subthreshold swing of a MOS capacitor at 300 K.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..boundaries import BoundarySpec
from ..grid import RectilinearGrid
from ..materials import Material, MaterialMap, SemiconductorParams
from ..operators import (
    Operators,
    cell_to_edge,
    cell_to_node,
    edge_mass,
    grad_node_edge,
    node_volume_vector,
)
from ..units import cm2_per_Vs, eps0, kB, per_cm3, q, thermal_voltage
from .base import ConvergenceError, Result, SolverBase, SolverConfig, Terminal

__all__ = [
    "bernoulli",
    "bernoulli_prime",
    "caughey_thomas",
    "masetti_silicon",
    "MobilityModel",
    "RecombinationModel",
    "DriftDiffusionSolver",
]


# ==========================================================================
# The Bernoulli function --- the numerical heart of Scharfetter-Gummel
# ==========================================================================
# Thresholds.  ``expm1`` is exact for small arguments, so the only genuinely
# dangerous points are x == 0 (0/0) and |x| large enough to overflow exp.
_B_SERIES = 1.0e-8      # below this, x/expm1(x) is fine but the series is free
_B_LARGE = 700.0        # exp(709.8) overflows a float64
_BP_SERIES = 0.25       # B'(x) loses digits to cancellation below this
_BP_LARGE = 300.0       # (e^x - 1)^2 overflows well before exp does


def bernoulli(x: np.ndarray | float) -> np.ndarray:
    """``B(x) = x / (exp(x) - 1)``, evaluated safely on the whole real line.

    The naive expression is unusable in a device simulator: it is ``0/0`` at
    ``x = 0`` (which happens on every edge of a collapsed grid direction and
    everywhere in a neutral region) and it overflows for ``x`` beyond ~709
    (which happens on any edge inside a junction at more than ~18 V of drop).
    Both cases occur in practice, so all four regimes are handled explicitly:

    ===================  ==================================================
    Regime               Evaluation
    ===================  ==================================================
    ``|x| < 1e-8``       ``1 - x/2 + x^2/12`` (Bernoulli-number series)
    ``1e-8 <= |x| <= 700``  ``x / expm1(x)`` --- ``expm1`` avoids the
                         cancellation in ``exp(x) - 1``
    ``x > 700``          ``x exp(-x)``; underflows to 0 above ~745, which is
                         the correctly rounded float64 result
    ``x < -700``         ``-x``; ``exp(x)`` is below 1e-304 there
    ===================  ==================================================

    Parameters
    ----------
    x : array_like
        Dimensionless argument, normally ``(psi_head - psi_tail)/Vt``.

    Returns
    -------
    np.ndarray
        ``B(x)``, same shape as ``x`` (0-d for scalar input).

    Notes
    -----
    Tagged **A7**.  The identity ``exp(x) B(x) = B(-x)`` is used throughout the
    solver and holds to machine precision for this implementation.
    """
    xin = np.asarray(x, dtype=float)
    xa = np.atleast_1d(xin)
    out = np.empty_like(xa)

    ax = np.abs(xa)
    small = ax < _B_SERIES
    big_p = xa > _B_LARGE
    big_n = xa < -_B_LARGE
    mid = ~(small | big_p | big_n)

    xs = xa[small]
    out[small] = 1.0 - 0.5 * xs + xs * xs / 12.0
    xm = xa[mid]
    out[mid] = xm / np.expm1(xm)
    xp = xa[big_p]
    out[big_p] = xp * np.exp(-xp)
    out[big_n] = -xa[big_n]
    return out.reshape(xin.shape)


def bernoulli_prime(x: np.ndarray | float) -> np.ndarray:
    """``dB/dx`` for :func:`bernoulli`, safe on the whole real line.

    Needed for the Newton Jacobian.  The closed form
    ``B'(x) = ((e^x - 1) - x e^x)/(e^x - 1)^2`` cancels catastrophically near
    zero and overflows near ``x = 355``, so it is replaced by

    * the Bernoulli-number series
      ``-1/2 + x/6 - x^3/180 + x^5/5040 - x^7/151200 + 5 x^9/23950080``
      for ``|x| < 0.25`` (truncation there is ~3e-15 relative, measured
      against mpmath at 60 digits),
    * ``B(x) (1/x - 1/(1 - exp(-x)))`` for ``0.25 <= |x| <= 300``,
    * ``(1 - x) exp(-x)`` for ``x > 300`` and ``-1`` for ``x < -300``.

    Parameters
    ----------
    x : array_like
        Dimensionless argument.

    Returns
    -------
    np.ndarray
        ``B'(x)``, same shape as ``x``.
    """
    xin = np.asarray(x, dtype=float)
    xa = np.atleast_1d(xin)
    out = np.empty_like(xa)

    ax = np.abs(xa)
    small = ax < _BP_SERIES
    big_p = xa > _BP_LARGE
    big_n = xa < -_BP_LARGE
    mid = ~(small | big_p | big_n)

    xs = xa[small]
    x2 = xs * xs
    out[small] = (-0.5 + xs / 6.0 - xs * x2 / 180.0
                  + xs * x2 * x2 / 5040.0 - xs * x2 * x2 * x2 / 151200.0
                  + 5.0 * xs * x2 * x2 * x2 * x2 / 23950080.0)

    xm = xa[mid]
    out[mid] = bernoulli(xm) * (1.0 / xm - 1.0 / (-np.expm1(-xm)))

    xp = xa[big_p]
    # exp(-x) underflows to 0 above ~745, which is the right answer.
    out[big_p] = (1.0 - xp) * np.exp(-xp)
    out[big_n] = -1.0
    return out.reshape(xin.shape)


# ==========================================================================
# Mobility models
# ==========================================================================
def caughey_thomas(mu_low: np.ndarray | float, field: np.ndarray | float,
                   vsat: float, beta: float = 2.0) -> np.ndarray:
    """Caughey-Thomas field-dependent mobility [m^2/(V s)].

    ``mu = mu_low / (1 + (mu_low F / vsat)^beta)^(1/beta)``.

    Parameters
    ----------
    mu_low : array_like
        Low-field mobility [m^2/(V s)].
    field : array_like
        Magnitude of the driving electric field [V/m].
    vsat : float
        Saturation velocity [m/s].
    beta : float
        Softness exponent; 2 for electrons and 1 for holes in silicon.

    Returns
    -------
    np.ndarray
        Field-degraded mobility [m^2/(V s)].

    Notes
    -----
    Tagged **A5**.  The driving field used here is the potential gradient
    resolved *along the edge*, not the full field vector; on a rectilinear grid
    those coincide for current flowing along a grid direction and differ by the
    cosine of the flow angle otherwise.  This is the standard box-method
    compromise and it under-estimates velocity saturation for diagonal flow.
    """
    mu_low = np.asarray(mu_low, dtype=float)
    fr = np.asarray(field, dtype=float) * mu_low / vsat
    return mu_low / (1.0 + np.abs(fr) ** beta) ** (1.0 / beta)


# Masetti et al., IEEE TED 30, 764 (1983).  Units as published: cm^2/(V s)
# and cm^-3; converted on use.
_MASETTI_N = dict(mu_min1=52.2, mu_min2=52.2, mu_1=43.4, mu_0=1417.0,
                  Pc=0.0, Cr=9.68e16, Cs=3.43e20, alpha=0.680, beta=2.0)
_MASETTI_P = dict(mu_min1=44.9, mu_min2=0.0, mu_1=29.0, mu_0=470.5,
                  Pc=9.23e16, Cr=2.23e17, Cs=6.10e20, alpha=0.719, beta=2.0)


def masetti_silicon(n_total: np.ndarray | float, carrier: str = "n"
                    ) -> np.ndarray:
    """Masetti doping-dependent bulk mobility in silicon [m^2/(V s)].

    Parameters
    ----------
    n_total : array_like
        Total ionised impurity concentration ``Na + Nd`` [m^-3].
    carrier : {'n', 'p'}
        Which carrier.

    Returns
    -------
    np.ndarray
        Low-field mobility [m^2/(V s)].

    Notes
    -----
    Silicon only, 300 K only.  Tagged **A5**/**A11**.  The formula is a fit to
    bulk Hall data; it says nothing about surface-roughness or Coulomb
    scattering in an inversion layer, so a MOSFET channel mobility from this
    model is an over-estimate by roughly a factor of two.
    """
    par = _MASETTI_N if carrier == "n" else _MASETTI_P
    if carrier not in ("n", "p"):
        raise ValueError(f"carrier must be 'n' or 'p', got {carrier!r}")
    N = np.maximum(np.asarray(n_total, dtype=float) / per_cm3, 1.0)
    mu = (par["mu_min1"] * np.exp(-par["Pc"] / N)
          + (par["mu_0"] - par["mu_min2"]) / (1.0 + (N / par["Cr"]) ** par["alpha"])
          - par["mu_1"] / (1.0 + (par["Cs"] / N) ** par["beta"]))
    return np.maximum(mu, 1.0) * cm2_per_Vs


@dataclass(frozen=True)
class MobilityModel:
    """Which mobility corrections are active.

    Attributes
    ----------
    field_dependent : bool
        Apply :func:`caughey_thomas` velocity saturation using the edge
        potential drop.  On by default; it is what keeps a short-channel
        device from predicting unphysical drift velocities.
    doping_dependent : bool
        Apply :func:`masetti_silicon`.  Off by default because it is
        silicon-specific and the library low-field mobility is already the
        lightly-doped value the user asked for.
    beta_n, beta_p : float
        Caughey-Thomas exponents.
    """
    field_dependent: bool = True
    doping_dependent: bool = False
    beta_n: float = 2.0
    beta_p: float = 1.0


@dataclass(frozen=True)
class RecombinationModel:
    """Which recombination mechanisms are active.

    Attributes
    ----------
    srh : bool
        Shockley-Read-Hall through a single trap level.
    auger : bool
        Band-to-band Auger, ``(Cn n + Cp p)(n p - ni^2)``.
    et_offset : float
        Trap level measured from the intrinsic level, expressed as a voltage
        ``(Et - Ei)/q`` [V].  Zero (mid-gap) maximises the SRH rate and is the
        conventional default.
    """
    srh: bool = True
    auger: bool = True
    et_offset: float = 0.0


# ==========================================================================
# Internal state of one nonlinear solve
# ==========================================================================
@dataclass
class _BCState:
    """Which unknowns are pinned, and to what, for one bias point."""
    psi_fixed: np.ndarray            # bool (N,)
    psi_val: np.ndarray              # (N,) [V]
    car_fixed: np.ndarray            # bool (N,) --- pins both n and p
    n_val: np.ndarray                # (N,) [m^-3]
    p_val: np.ndarray                # (N,) [m^-3]
    ohmic: dict[str, np.ndarray] = field(default_factory=dict)
    gate: dict[str, np.ndarray] = field(default_factory=dict)
    dt: float | None = None
    n_old: np.ndarray | None = None
    p_old: np.ndarray | None = None
    psi_old: np.ndarray | None = None


# ==========================================================================
# The solver
# ==========================================================================
class DriftDiffusionSolver(SolverBase):
    """van Roosbroeck drift-diffusion solver with Scharfetter-Gummel fluxes.

    Parameters
    ----------
    grid : RectilinearGrid
        The mesh.  Must resolve the extrinsic Debye length in the doped
        regions; the constructor warns if it does not.
    matmap : MaterialMap
        Material assignment.  Exactly one semiconductor material may be
        present (heterojunction band offsets are not modelled, so a second one
        would be silently wrong; it is rejected instead).
    doping_node : np.ndarray
        Net doping ``Nd - Na`` [m^-3] at every node, shape ``grid.shape_nodes``
        or flat ``(grid.n_nodes,)``.  Positive is n-type.
    T : float
        Lattice temperature [K].  Constant (**A6**).
    config : SolverConfig, optional
        Newton and linear-solver knobs.
    operators : Operators, optional
        Shared incidence-operator cache.
    mobility : MobilityModel, optional
    recombination : RecombinationModel, optional
    doping_total_node : np.ndarray, optional
        Total impurity concentration ``Na + Nd`` [m^-3] for the Masetti model.
        Defaults to ``|doping_node|``.
    contact_kinds : dict, optional
        ``{terminal_name: 'ohmic' | 'gate'}``.  Terminals not listed are
        classified automatically: **ohmic** if every one of the terminal's
        nodes touches a semiconductor cell, **gate** otherwise.  An ohmic
        contact pins ``psi``, ``n`` and ``p``; a gate pins only ``psi``.
    gate_offsets : dict, optional
        ``{terminal_name: volts}`` added to a gate terminal's applied voltage.
        This is where a metal/poly work-function difference goes: ``psi`` is
        referenced to the intrinsic level of the semiconductor, so an ideal
        mid-gap gate has offset 0 and its flat-band voltage is minus the bulk
        equilibrium potential.

    Notes
    -----
    Tagged **A5** (drift-diffusion), **A6** (isothermal), **A7**
    (Scharfetter-Gummel), **A10** (box method), **A11** (complete ionisation).
    """

    name = "drift_diffusion"
    assumptions = ("A5", "A6", "A7", "A10", "A11")

    # ----------------------------------------------------------------- init
    def __init__(self, grid: RectilinearGrid, matmap: MaterialMap,
                 doping_node: np.ndarray, T: float = 300.0,
                 config: SolverConfig | None = None,
                 operators: Operators | None = None,
                 mobility: MobilityModel | None = None,
                 recombination: RecombinationModel | None = None,
                 doping_total_node: np.ndarray | None = None,
                 contact_kinds: dict[str, str] | None = None,
                 gate_offsets: dict[str, float] | None = None) -> None:
        super().__init__(grid, config, operators)

        if not isinstance(matmap, MaterialMap):
            raise ValueError(
                f"matmap must be a MaterialMap, got {type(matmap).__name__}")
        if T <= 0.0:
            raise ValueError(f"temperature must be positive [K], got {T}")

        self.matmap = matmap
        self.T = float(T)
        self.Vt = thermal_voltage(self.T)
        self.mob = mobility or MobilityModel()
        self.rec = recombination or RecombinationModel()
        self.contact_kinds = dict(contact_kinds or {})
        self.gate_offsets = dict(gate_offsets or {})

        # -- the single semiconductor -------------------------------------
        semis = [m for m in matmap.materials
                 if m.kind == "semiconductor" and m.semi is not None]
        uniq = {m.name: m for m in semis}
        if not uniq:
            raise ValueError(
                "no semiconductor material with SemiconductorParams is present "
                "in the MaterialMap; drift-diffusion needs one")
        if len(uniq) > 1:
            raise ValueError(
                "drift-diffusion supports exactly one semiconductor material "
                f"(heterojunction band offsets are not modelled); found "
                f"{sorted(uniq)}")
        self.material: Material = next(iter(uniq.values()))
        self.semi: SemiconductorParams = self.material.semi  # type: ignore[assignment]
        self.ni = float(self.semi.ni(self.T))

        # -- doping --------------------------------------------------------
        self.dop = self._as_node_vector(doping_node, "doping_node")
        if doping_total_node is None:
            self.dop_total = np.abs(self.dop)
        else:
            self.dop_total = self._as_node_vector(doping_total_node,
                                                  "doping_total_node")
            if np.any(self.dop_total < 0.0):
                raise ValueError("doping_total_node must be non-negative [m^-3]")

        self.N_ref = float(max(np.max(np.abs(self.dop)), self.ni))
        # Densities below this are numerically indistinguishable from zero and
        # contribute neither charge nor current; the floor only exists to keep
        # Newton from stepping a density negative.
        self.dens_floor = 1.0e-40 * self.N_ref

        self._build_static()
        self._check_mesh()

    # -------------------------------------------------------------- helpers
    def _as_node_vector(self, a: np.ndarray, what: str) -> np.ndarray:
        arr = np.asarray(a, dtype=float)
        if arr.shape == self.grid.shape_nodes:
            return np.ascontiguousarray(arr.ravel())
        if arr.shape == (self.grid.n_nodes,):
            return np.ascontiguousarray(arr)
        raise ValueError(
            f"{what} must have shape {self.grid.shape_nodes} or "
            f"({self.grid.n_nodes},), got {arr.shape}")

    def _build_static(self) -> None:
        """Assemble everything that does not depend on the solution."""
        g = self.grid
        self.G = self.ops.G
        self.Gt = sp.csr_matrix(self.G.T)

        # Head/tail node of every edge, read straight off the incidence
        # matrix so the two can never disagree with the operators.
        if not np.all(np.diff(self.G.indptr) == 2):
            raise ValueError("grad_node_edge produced a row without exactly "
                             "two entries; the operator core is inconsistent")
        idx = self.G.indices.reshape(-1, 2)
        dat = self.G.data.reshape(-1, 2)
        first_is_head = dat[:, 0] > 0
        self.head = np.where(first_is_head, idx[:, 0], idx[:, 1]).astype(np.intp)
        self.tail = np.where(first_is_head, idx[:, 1], idx[:, 0]).astype(np.intp)

        # -- Poisson operator ---------------------------------------------
        eps_edge = cell_to_edge(g, self.matmap.eps(), mode="parallel")
        self.L_eps = sp.csr_matrix(self.Gt @ edge_mass(g, eps_edge) @ self.G)

        # -- semiconductor geometry ---------------------------------------
        semi_cell = self.matmap.semiconductor_mask().astype(float)
        frac_node = cell_to_node(g, semi_cell).ravel()
        self.semi_node = frac_node > 0.0
        # Control volume restricted to the semiconductor: charge storage,
        # recombination and transient carrier storage all belong to the
        # semiconductor part of the dual box only.  Identical to the full dual
        # volume for an all-semiconductor domain.
        self.V_semi = node_volume_vector(g) * frac_node
        self.V_node = node_volume_vector(g)

        # Fraction of each edge's dual area that is semiconductor: the
        # cross-section actually available for carrier transport.
        semi_edge = cell_to_edge(g, semi_cell, mode="parallel")
        both = self.semi_node[self.head] & self.semi_node[self.tail]
        L = np.concatenate([a.ravel() for a in g.edge_lengths()])
        A = np.concatenate([a.ravel() for a in g.edge_dual_areas()])
        self.edge_len = L
        self.edge_geom = (A / L) * semi_edge * both.astype(float)

        # -- low-field mobility on edges ----------------------------------
        if self.mob.doping_dependent:
            Ne = 0.5 * (self.dop_total[self.head] + self.dop_total[self.tail])
            self.mu_n0 = masetti_silicon(Ne, "n")
            self.mu_p0 = masetti_silicon(Ne, "p")
        else:
            self.mu_n0 = np.full(g.n_edges, self.semi.mu_n)
            self.mu_p0 = np.full(g.n_edges, self.semi.mu_p)

        # -- recombination constants --------------------------------------
        s = self.semi
        self.n1 = self.ni * np.exp(np.clip(self.rec.et_offset / self.Vt, -400, 400))
        self.p1 = self.ni * np.exp(np.clip(-self.rec.et_offset / self.Vt, -400, 400))
        self.tau_n, self.tau_p = float(s.tau_n), float(s.tau_p)
        self.C_aug_n = float(s.C_auger_n) if self.rec.auger else 0.0
        self.C_aug_p = float(s.C_auger_p) if self.rec.auger else 0.0

    def _check_mesh(self) -> None:
        """Warn when the mesh cannot resolve the physics (**A10**)."""
        Nmax = float(np.max(np.abs(self.dop)))
        if Nmax <= 0.0:
            return
        eps_semi = eps0 * self.material.eps_r
        LD = float(np.sqrt(eps_semi * self.Vt / (q * Nmax)))
        hmin = min(float(self.grid.hx.min()), float(self.grid.hy.min()),
                   float(self.grid.hz.min()))
        # Only resolved directions matter; a collapsed direction is 1 m wide
        # by construction and must not trigger the warning.
        hs = [float(a.h.min()) for a in self.grid._axes if not a.collapsed]
        hmin = min(hs) if hs else hmin
        if hmin > LD:
            warnings.warn(
                f"smallest cell {hmin:.3g} m exceeds the extrinsic Debye "
                f"length {LD:.3g} m at N = {Nmax:.3g} m^-3; the space-charge "
                "layer is unresolved and the solution will be inaccurate "
                "(A10)", stacklevel=3)
        if self.grid.max_growth_ratio() > 2.0:
            warnings.warn(
                f"mesh growth ratio {self.grid.max_growth_ratio():.2f} exceeds "
                "2.0; the box method degrades toward first order (A10)",
                stacklevel=3)

    # ================================================================ physics
    def equilibrium_densities(self, psi: np.ndarray,
                              phi_ref: np.ndarray | float = 0.0
                              ) -> tuple[np.ndarray, np.ndarray]:
        """Boltzmann densities ``(n, p)`` [m^-3] at potential ``psi`` [V].

        ``n = ni exp((psi - phi)/Vt)``, ``p = ni exp((phi - psi)/Vt)`` with
        ``phi`` the common Fermi level [V].  The exponent is clipped to +-400
        before ``exp``, per the measured recipe in ``docs/CONTRACTS.md``;
        without the clip the first Newton iterate from a bad initial guess
        overflows and the solve is over before it starts.
        """
        u = np.clip((np.asarray(psi, dtype=float) - phi_ref) / self.Vt,
                    -400.0, 400.0)
        return self.ni * np.exp(u), self.ni * np.exp(-u)

    def neutral_state(self, C: np.ndarray | float
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Charge-neutral equilibrium ``(psi, n, p)`` for net doping ``C``.

        Solves ``n - p = C`` with ``n p = ni^2`` exactly.  The naive
        ``n = (C + sqrt(C^2 + 4 ni^2))/2`` loses every significant digit for
        ``C < 0`` (it subtracts two nearly equal 1e23-sized numbers to get
        1e9), so the majority carrier is computed from the square root and the
        minority carrier from mass action.

        Parameters
        ----------
        C : array_like
            Net doping ``Nd - Na`` [m^-3].

        Returns
        -------
        (psi, n, p)
            Potential [V] referenced to the intrinsic level, and densities
            [m^-3].
        """
        C = np.asarray(C, dtype=float)
        s = np.sqrt(C * C + 4.0 * self.ni * self.ni)
        ntype = C >= 0.0
        # Majority carrier from the square root, minority from mass action:
        # both branches are evaluated by np.where, so the unused one is fed a
        # safe placeholder rather than a zero that would overflow the divide.
        maj = np.where(ntype, 0.5 * (C + s), 0.5 * (s - C))
        maj = np.maximum(maj, np.finfo(float).tiny)
        minr = self.ni * self.ni / maj
        n = np.where(ntype, maj, minr)
        p = np.where(ntype, minr, maj)
        psi = self.Vt * np.log(n / self.ni)
        return psi, n, p

    def _mobilities(self, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Edge mobilities [m^2/(V s)] at the current potential."""
        if not self.mob.field_dependent:
            return self.mu_n0, self.mu_p0
        F = np.abs(psi[self.head] - psi[self.tail]) / self.edge_len
        mn = caughey_thomas(self.mu_n0, F, self.semi.vsat_n, self.mob.beta_n)
        mp = caughey_thomas(self.mu_p0, F, self.semi.vsat_p, self.mob.beta_p)
        return mn, mp

    def _edge_coeffs(self, psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``q D (A/L)`` per edge for electrons and holes [C m^3/s]."""
        mn, mp = self._mobilities(psi)
        f = q * self.Vt * self.edge_geom          # q * (mu Vt) * A/L
        return f * mn, f * mp

    def _recombination(self, n: np.ndarray, p: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Net recombination rate [m^-3 s^-1] and its derivatives.

        Returns ``(R, dR/dn, dR/dp)``.  SRH uses a single trap level
        (:attr:`RecombinationModel.et_offset`); Auger is the standard
        ``(Cn n + Cp p)(n p - ni^2)``.  ``R`` is negative where ``n p < ni^2``,
        i.e. net *generation*, which is what supplies the reverse leakage of a
        depleted junction.
        """
        ni2 = self.ni * self.ni
        U = n * p - ni2
        R = np.zeros_like(n)
        dRdn = np.zeros_like(n)
        dRdp = np.zeros_like(n)
        if self.rec.srh:
            D = self.tau_p * (n + self.n1) + self.tau_n * (p + self.p1)
            R += U / D
            dRdn += p / D - U * self.tau_p / (D * D)
            dRdp += n / D - U * self.tau_n / (D * D)
        if self.rec.auger and (self.C_aug_n or self.C_aug_p):
            Ca = self.C_aug_n * n + self.C_aug_p * p
            R += Ca * U
            dRdn += self.C_aug_n * U + Ca * p
            dRdp += self.C_aug_p * U + Ca * n
        return R, dRdn, dRdp

    def _sg(self, psi: np.ndarray, n: np.ndarray, p: np.ndarray
            ) -> dict[str, np.ndarray]:
        """Scharfetter-Gummel edge currents [A] and the pieces of their Jacobian."""
        X = (psi[self.head] - psi[self.tail]) / self.Vt
        Bp = bernoulli(X)
        Bm = bernoulli(-X)
        Cn, Cp = self._edge_coeffs(psi)
        In = Cn * (n[self.head] * Bp - n[self.tail] * Bm)
        Ip = Cp * (p[self.tail] * Bp - p[self.head] * Bm)
        return {"X": X, "Bp": Bp, "Bm": Bm, "Cn": Cn, "Cp": Cp,
                "In": In, "Ip": Ip}

    # ============================================================== residual
    def _residual(self, psi: np.ndarray, n: np.ndarray, p: np.ndarray,
                  st: _BCState) -> np.ndarray:
        """Physical residual ``[F_psi (A s), F_n (A), F_p (A)]``, Dirichlet rows zeroed."""
        sgv = self._sg(psi, n, p)
        R, _, _ = self._recombination(n, p)

        F_psi = self.L_eps @ psi - q * (p - n + self.dop) * self.V_semi
        F_n = self.Gt @ sgv["In"] + q * R * self.V_semi
        F_p = self.Gt @ sgv["Ip"] - q * R * self.V_semi
        if st.dt is not None:
            w = q * self.V_semi / st.dt
            F_n = F_n + w * (n - st.n_old)
            F_p = F_p - w * (p - st.p_old)

        F_psi[st.psi_fixed] = 0.0
        F_n[st.car_fixed] = 0.0
        F_p[st.car_fixed] = 0.0
        return np.concatenate([F_psi, F_n, F_p])

    def _jacobian(self, psi: np.ndarray, n: np.ndarray, p: np.ndarray,
                  st: _BCState) -> sp.csr_matrix:
        """Analytic Jacobian of :meth:`_residual` (mobility lagged; see Notes).

        Notes
        -----
        The field-dependent mobility is evaluated at the current iterate but
        its derivative with respect to ``psi`` is dropped.  This makes the
        method a modified Newton: the converged solution is unaffected (at the
        fixed point the mobility is consistent with the potential that
        produced it) but the asymptotic rate degrades from quadratic to
        superlinear once velocity saturation is active.
        """
        N = self.grid.n_nodes
        sgv = self._sg(psi, n, p)
        Cn, Cp, Bp, Bm, X = sgv["Cn"], sgv["Cp"], sgv["Bp"], sgv["Bm"], sgv["X"]
        Bpp = bernoulli_prime(X)
        Bmp = bernoulli_prime(-X)
        _, dRdn, dRdp = self._recombination(n, p)

        ne = np.arange(self.grid.n_edges)
        rows = np.concatenate([ne, ne])
        cols = np.concatenate([self.head, self.tail])

        # d I_n / d n  and  d I_p / d p, edges x nodes
        An = sp.coo_matrix((np.concatenate([Cn * Bp, -Cn * Bm]), (rows, cols)),
                           shape=(self.grid.n_edges, N)).tocsr()
        Ap = sp.coo_matrix((np.concatenate([-Cp * Bm, Cp * Bp]), (rows, cols)),
                           shape=(self.grid.n_edges, N)).tocsr()

        # d I / d psi factors through G: dX/dpsi = G/Vt.
        Wn = Cn * (n[self.head] * Bpp + n[self.tail] * Bmp) / self.Vt
        Wp = Cp * (p[self.tail] * Bpp + p[self.head] * Bmp) / self.Vt

        qV = q * self.V_semi
        Jpp = self.L_eps
        Jpn = sp.diags(qV)
        Jpp2 = sp.diags(-qV)

        Jnpsi = self.Gt @ sp.diags(Wn) @ self.G
        Jnn = self.Gt @ An + sp.diags(qV * dRdn)
        Jnp = sp.diags(qV * dRdp)

        Jppsi = self.Gt @ sp.diags(Wp) @ self.G
        Jpn2 = sp.diags(-qV * dRdn)
        Jppp = self.Gt @ Ap + sp.diags(-qV * dRdp)

        if st.dt is not None:
            w = qV / st.dt
            Jnn = Jnn + sp.diags(w)
            Jppp = Jppp - sp.diags(w)

        J = sp.bmat([[Jpp, Jpn, Jpp2],
                     [Jnpsi, Jnn, Jnp],
                     [Jppsi, Jpn2, Jppp]], format="csr")

        fixed = np.concatenate([st.psi_fixed, st.car_fixed, st.car_fixed])
        free = sp.diags((~fixed).astype(float))
        return sp.csr_matrix(free @ J + sp.diags(fixed.astype(float)))

    # ======================================================== linear algebra
    @staticmethod
    def _equilibrate(J: sp.csr_matrix, F: np.ndarray,
                     col: np.ndarray) -> tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """Column-scale by ``col`` then row-scale to unit infinity norm."""
        Jc = sp.csr_matrix(J @ sp.diags(col))
        rmax = np.asarray(abs(Jc).max(axis=1).todense()).ravel()
        r = np.where(rmax > 0.0, 1.0 / np.where(rmax > 0.0, rmax, 1.0), 1.0)
        return sp.csr_matrix(sp.diags(r) @ Jc), r * F, r

    def _lu_solve(self, A: sp.csr_matrix, b: np.ndarray,
                  history: Sequence[float]) -> np.ndarray:
        try:
            lu = spla.splu(sp.csc_matrix(A))
        except RuntimeError as exc:                # singular factor
            raise ConvergenceError(
                f"drift-diffusion Jacobian is singular: {exc}", history) from exc
        x = lu.solve(b)
        if not np.all(np.isfinite(x)):
            raise ConvergenceError(
                "drift-diffusion linear solve produced non-finite values "
                "(Jacobian is numerically singular)", history)
        return x

    # ============================================================ boundaries
    def _contact_kind(self, term: Terminal) -> str:
        if term.name in self.contact_kinds:
            k = self.contact_kinds[term.name]
            if k not in ("ohmic", "gate"):
                raise ValueError(
                    f"contact_kinds[{term.name!r}] must be 'ohmic' or 'gate', "
                    f"got {k!r}")
            return k
        return "ohmic" if bool(np.all(self.semi_node[term.nodes])) else "gate"

    def _bc_state(self, terminals: Sequence[Terminal],
                  bc: BoundarySpec | None, t: float = 0.0,
                  bias: dict[str, float] | None = None) -> _BCState:
        """Assemble the pinned-unknown description for one bias point."""
        N = self.grid.n_nodes
        st = _BCState(psi_fixed=np.zeros(N, dtype=bool),
                      psi_val=np.zeros(N),
                      car_fixed=~self.semi_node.copy(),
                      n_val=np.zeros(N), p_val=np.zeros(N))

        if bc is not None:
            idx, val = bc.dirichlet_nodes(self.grid, t)
            if idx.size:
                if np.any(self.semi_node[idx]):
                    warnings.warn(
                        "a Dirichlet boundary wall touches semiconductor nodes; "
                        "it pins psi only and injects no carriers. Use a "
                        "Terminal for an ohmic contact.", stacklevel=3)
                st.psi_fixed[idx] = True
                st.psi_val[idx] = val

        for term in terminals:
            if term.driven == "current":
                raise ValueError(
                    f"terminal {term.name!r} is current-driven; the "
                    "drift-diffusion solver supports voltage-driven and "
                    "floating terminals only")
            v = term.value_at(t)
            v = 0.0 if v is None else float(v)
            if bias is not None and term.name in bias:
                v = float(bias[term.name])
            nodes = np.asarray(term.nodes, dtype=np.intp)
            if nodes.size == 0:
                raise ValueError(f"terminal {term.name!r} has no nodes")
            if nodes.max() >= N or nodes.min() < 0:
                raise ValueError(
                    f"terminal {term.name!r} has node indices outside "
                    f"[0, {N})")
            kind = self._contact_kind(term)
            if kind == "ohmic":
                if not np.all(self.semi_node[nodes]):
                    raise ValueError(
                        f"terminal {term.name!r} is an ohmic contact but some "
                        "of its nodes do not touch a semiconductor cell")
                psi0, n0, p0 = self.neutral_state(self.dop[nodes])
                st.psi_fixed[nodes] = True
                st.psi_val[nodes] = psi0 + v
                st.car_fixed[nodes] = True
                st.n_val[nodes] = n0
                st.p_val[nodes] = p0
                st.ohmic[term.name] = nodes
            else:
                st.psi_fixed[nodes] = True
                st.psi_val[nodes] = v + float(self.gate_offsets.get(term.name, 0.0))
                st.gate[term.name] = nodes
        return st

    @staticmethod
    def _apply_fixed(psi: np.ndarray, n: np.ndarray, p: np.ndarray,
                     st: _BCState) -> None:
        psi[st.psi_fixed] = st.psi_val[st.psi_fixed]
        n[st.car_fixed] = st.n_val[st.car_fixed]
        p[st.car_fixed] = st.p_val[st.car_fixed]

    # =========================================================== equilibrium
    def equilibrium(self, terminals: Sequence[Terminal] = (),
                    bc: BoundarySpec | None = None, t: float = 0.0,
                    psi0: np.ndarray | None = None) -> np.ndarray:
        """Thermal-equilibrium potential [V], shape ``(grid.n_nodes,)``.

        Solves the nonlinear Poisson equation with the carrier densities tied
        to the potential by Boltzmann statistics,
        ``n = ni exp((psi - V0)/Vt)`` and ``p = ni exp((V0 - psi)/Vt)``, where
        ``V0`` is the common potential of the ohmic contacts.  This is the
        zero-current state, so it is exact for a MOS capacitor at any gate
        bias as well as for an unbiased junction.

        With no terminals and the default Neumann walls the problem is still
        well posed --- the nonlinear space charge removes the constant null
        vector of ``G^T M_eps G`` and enforces global neutrality --- so the
        built-in potential of a junction comes out as a *prediction*, not as
        something imposed by a contact.

        Parameters
        ----------
        terminals : sequence of Terminal, optional
            Gate terminals apply their voltage to ``psi``; ohmic terminals pin
            ``psi`` to the local neutral value plus their bias.  All ohmic
            terminals must share one bias, otherwise the state is not
            equilibrium and :meth:`solve_dc` is the right entry point.
        bc : BoundarySpec, optional
            Wall conditions; default is homogeneous Neumann.
        t : float
            Time [s] at which time-dependent sources are evaluated.
        psi0 : np.ndarray, optional
            Initial guess [V].

        Returns
        -------
        np.ndarray
            ``psi`` [V], flat, length ``grid.n_nodes``.

        Raises
        ------
        ConvergenceError
            If Newton fails; the exception carries the residual history.
        """
        terminals = list(terminals)
        st = self._bc_state(terminals, bc, t)
        v0 = self._common_ohmic_bias(terminals, st, t)
        psi = self._initial_psi(st) if psi0 is None else np.array(psi0, float)
        self._apply_fixed(psi, np.zeros(1), np.zeros(1),
                          _BCState(st.psi_fixed, st.psi_val,
                                   np.zeros(1, bool), np.zeros(1), np.zeros(1)))
        return self._poisson_newton(psi, st, v0)[0]

    def _common_ohmic_bias(self, terminals: Sequence[Terminal], st: _BCState,
                           t: float) -> float:
        vals = []
        for term in terminals:
            if term.name in st.ohmic:
                v = term.value_at(t)
                vals.append(0.0 if v is None else float(v))
        if not vals:
            return 0.0
        if max(vals) - min(vals) > 1e-12:
            raise ValueError(
                "equilibrium() needs all ohmic contacts at one potential; got "
                f"{vals}. Use solve_dc() for a current-carrying bias point.")
        return vals[0]

    def _initial_psi(self, st: _BCState) -> np.ndarray:
        """Local-neutrality guess in the semiconductor, harmonic elsewhere.

        Insulator nodes get the solution of the *linear* Laplace problem with
        the semiconductor and the electrodes held fixed.  Starting an oxide at
        zero instead costs several Newton iterations on a MOS structure and can
        overflow the exponential on the first step.
        """
        psi = np.zeros(self.grid.n_nodes)
        psi_neutral, _, _ = self.neutral_state(self.dop)
        psi[self.semi_node] = psi_neutral[self.semi_node]
        psi[st.psi_fixed] = st.psi_val[st.psi_fixed]

        held = self.semi_node | st.psi_fixed
        if np.all(held):
            return psi
        A = sp.csr_matrix(self.L_eps)
        b = -(A @ np.where(held, psi, 0.0))
        keep = ~held
        A_ff = A[keep][:, keep]
        b_f = b[keep]
        try:
            psi[keep] = spla.spsolve(sp.csc_matrix(A_ff), b_f)
        except Exception:                          # pragma: no cover - fallback
            psi[keep] = 0.0
        return psi

    def _poisson_newton(self, psi: np.ndarray, st: _BCState, v0: float,
                        ) -> tuple[np.ndarray, list[float], int]:
        """Newton on the equilibrium nonlinear Poisson equation.

        Uses exactly the damping recipe measured in ``docs/CONTRACTS.md``: an
        exponent clip at +-400, a step clamp ``lam = min(1, 5 Vt/max|dpsi|)``
        to keep the exponential in range, then an Armijo line search that
        forces the residual to decrease.  The clamp alone is *not* enough --- it
        produces stable limit cycles at some doping/ni combinations, which is
        why the line search is mandatory rather than decorative.
        """
        free = ~st.psi_fixed
        cfg = self.cfg
        hist: list[float] = []
        # Normalise each Poisson row by the charge that N_ref would put in
        # that control volume, so the residual is a dimensionless net-charge
        # error.  It must use the FULL dual volume, not the semiconductor
        # part: an insulator node has zero semiconductor volume, and dividing
        # by that (even softened by a floor) inflates the oxide rows by ~1e50
        # and the line search then optimises the oxide while ignoring the
        # semiconductor -- which silently pins the surface potential of a MOS
        # capacitor and destroys the subthreshold slope.
        scale = q * self.N_ref * self.V_node

        def resid(ps: np.ndarray) -> np.ndarray:
            n, p = self.equilibrium_densities(ps, v0)
            n = np.where(self.semi_node, n, 0.0)
            p = np.where(self.semi_node, p, 0.0)
            F = self.L_eps @ ps - q * (p - n + self.dop) * self.V_semi
            F[st.psi_fixed] = 0.0
            return F / scale

        F = resid(psi)
        nrm = float(np.linalg.norm(F[free]))
        hist.append(nrm)
        it = 0
        for it in range(1, cfg.max_newton + 1):
            n, p = self.equilibrium_densities(psi, v0)
            n = np.where(self.semi_node, n, 0.0)
            p = np.where(self.semi_node, p, 0.0)
            J = self.L_eps + sp.diags(q * (n + p) * self.V_semi / self.Vt)
            J = sp.csr_matrix(sp.diags(free.astype(float)) @ J
                              + sp.diags(st.psi_fixed.astype(float)))
            Js, Fs, _ = self._equilibrate(sp.csr_matrix(J), F * scale,
                                          np.full(self.grid.n_nodes, self.Vt))
            dpsi = self.Vt * self._lu_solve(Js, -Fs, hist)

            mx = float(np.max(np.abs(dpsi))) if dpsi.size else 0.0
            lam = 1.0 if mx == 0.0 else min(1.0, 5.0 * self.Vt / mx)
            for _ in range(60):
                trial = psi + lam * dpsi
                Ft = resid(trial)
                nt = float(np.linalg.norm(Ft[free]))
                if nt < (1.0 - 1.0e-4 * lam) * nrm:
                    break
                lam *= 0.5
            psi, F, nrm = trial, Ft, nt
            hist.append(nrm)
            self._log(2, f"equilibrium newton {it:2d}  |F| {nrm:.4e}  "
                         f"lam {lam:.3g}  max|dpsi| {mx * lam:.4e} V")
            if mx * lam < cfg.newton_tol * self.Vt or nrm < 1e-14:
                break
        else:
            raise ConvergenceError(
                f"equilibrium Poisson failed to converge in {cfg.max_newton} "
                f"iterations (|F| = {nrm:.4e})", hist, psi)
        return psi, hist, it

    # ================================================================ DC solve
    def solve_dc(self, terminals: Sequence[Terminal],
                 bc: BoundarySpec | None = None, ramp: bool = True,
                 ramp_step: float = 0.05, method: str = "newton",
                 t: float = 0.0, x0: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                 ) -> Result:
        """Steady-state bias point.

        Parameters
        ----------
        terminals : sequence of Terminal
            Voltage-driven electrodes.  See :class:`DriftDiffusionSolver` for
            how ohmic and gate contacts are told apart.
        bc : BoundarySpec, optional
            Wall conditions; default homogeneous Neumann (**A12**).
        ramp : bool
            Start from equilibrium at zero bias and walk the terminal voltages
            up in steps.  Essential above a few hundred millivolts of forward
            bias: a cold start at 0.6 V on a diode moves the carrier densities
            by ten orders of magnitude in one step and Newton has no chance.
            The step is adaptive --- halved on failure, grown by 1.4 on
            success.
        ramp_step : float
            Initial and maximum voltage increment [V].
        method : {'newton', 'gummel'}
            ``'newton'`` is the fully coupled solve with a Gummel fallback if
            it stalls; ``'gummel'`` forces the decoupled iteration, which is
            slower but far more forgiving of a bad initial guess.
        t : float
            Time [s] used to evaluate time-dependent terminal values.
        x0 : tuple, optional
            ``(psi, n, p)`` initial guess; skips the equilibrium solve.

        Returns
        -------
        Result
            ``fields`` holds ``psi`` [V], ``n`` and ``p`` [m^-3], each with
            shape ``(1,) + grid.shape_nodes``.  ``terminals`` holds ``v`` [V]
            and ``i`` [A] (positive = conventional current flowing from the
            external circuit *into* the terminal).  ``scalars`` holds the
            per-terminal electron, hole and displacement components.
        """
        self._start()
        terminals = list(terminals)
        if method not in ("newton", "gummel"):
            raise ValueError(f"method must be 'newton' or 'gummel', got {method!r}")

        st_final = self._bc_state(terminals, bc, t)
        targets = {tm.name: (st_final.psi_val[tm.nodes][0]
                             if tm.nodes.size else 0.0) for tm in terminals}

        if x0 is not None:
            psi, n, p = (np.array(a, dtype=float).ravel() for a in x0)
        else:
            st0 = self._bc_state(terminals, bc, t,
                                 bias={tm.name: 0.0 for tm in terminals})
            psi = self.equilibrium([], None) if not terminals else \
                self._poisson_newton(self._initial_psi(st0), st0, 0.0)[0]
            n, p = self.equilibrium_densities(psi, 0.0)
            n = np.where(self.semi_node, n, 0.0)
            p = np.where(self.semi_node, p, 0.0)
            self._apply_fixed(psi, n, p, st0)

        hist: list[float] = []
        if not ramp or not terminals:
            st = self._bc_state(terminals, bc, t)
            psi, n, p, h = self._nonlinear_solve(psi, n, p, st, method)
            hist += h
        else:
            alpha, step, total_it = 0.0, min(1.0, ramp_step), 0
            guard = 0
            while alpha < 1.0 - 1e-12:
                guard += 1
                if guard > 400:
                    raise ConvergenceError(
                        "bias ramp made no progress in 400 attempts", hist)
                dv = max(abs(v) for v in targets.values()) or 1.0
                da = min(1.0 - alpha, max(step / dv, 1e-6))
                a_try = alpha + da
                bias = {tm.name: a_try * self._terminal_target(tm, t)
                        for tm in terminals}
                st = self._bc_state(terminals, bc, t, bias=bias)
                try:
                    psi_t, n_t, p_t, h = self._nonlinear_solve(
                        psi.copy(), n.copy(), p.copy(), st, method)
                except ConvergenceError:
                    step *= 0.5
                    if step < 1e-6:
                        raise
                    continue
                psi, n, p = psi_t, n_t, p_t
                alpha = a_try
                hist += h
                total_it += len(h)
                step = min(ramp_step, step * 1.4)
                self._log(1, f"bias ramp alpha {alpha:.4f} (step {step:.4g} V)")

        st = self._bc_state(terminals, bc, t)
        self._apply_fixed(psi, n, p, st)
        res = self._make_result(psi, n, p, st, terminals, [t])
        res.scalars["newton_residual"] = np.asarray(hist)
        return self._finish(res, newton_iterations=len(hist), method=method,
                            temperature=self.T, ni=self.ni, N_ref=self.N_ref)

    def _terminal_target(self, term: Terminal, t: float) -> float:
        v = term.value_at(t)
        return 0.0 if v is None else float(v)

    # ---------------------------------------------------------------- Newton
    def _nonlinear_solve(self, psi: np.ndarray, n: np.ndarray, p: np.ndarray,
                         st: _BCState, method: str
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        if method == "gummel":
            return self._gummel(psi, n, p, st)
        try:
            return self._newton(psi, n, p, st)
        except ConvergenceError as exc:
            self._log(1, f"coupled Newton failed ({exc}); falling back to Gummel")
            psi_g, n_g, p_g, h = self._gummel(psi, n, p, st)
            # Polish the Gummel result with Newton; it is usually inside the
            # quadratic basin by now.
            psi_g, n_g, p_g, h2 = self._newton(psi_g, n_g, p_g, st)
            return psi_g, n_g, p_g, h + h2

    def _density_scale(self, x: np.ndarray) -> np.ndarray:
        """Per-node column scale for a density unknown [m^-3].

        Scaling ``n`` and ``p`` by one global ``N_ref`` is what the naive
        recipe says, and it is wrong in a way that only shows up on leakage:
        a minority density of ``ni^2/N ~ 1e10 m^-3`` becomes a scaled unknown
        of ``1e-12``, so the linear solve resolves it to only ~4 significant
        figures, and the reverse current --- which is *entirely* a minority
        quantity --- inherits that error.  Scaling each node by its own
        density solves for the *relative* change instead and restores full
        precision on the minority carrier.  Measured effect on a 1e16 cm^-3
        junction at 16 V reverse: terminal-current Kirchhoff mismatch improves
        from ~30 % to ~1e-6.
        """
        return np.maximum(np.abs(x), self.ni * 1.0e-12)

    @staticmethod
    def _time_steps(t_end: float, dt: float, t_start: float = 0.0) -> np.ndarray:
        """Uniform time samples [s].

        ``docs/CONTRACTS.md`` fixes this class's base as :class:`SolverBase`,
        not :class:`TimeSteppingSolver`, so the shared stepping helpers are not
        inherited and the two lines they provide are reproduced here rather
        than changing a frozen file or the contracted class hierarchy.
        """
        n = int(np.ceil((t_end - t_start) / dt))
        return t_start + dt * np.arange(n + 1)

    def speedup_vs_courant(self, dt_used: float, eps_r_min: float = 1.0
                           ) -> float:
        """How many explicit-FDTD steps one implicit drift-diffusion step replaces."""
        return dt_used / self.grid.courant_dt(eps_r_min=eps_r_min)

    @staticmethod
    def _accept(f0: float) -> float:
        """Merit-residual value at which a Newton solve counts as converged.

        Relative to the residual the solve started from, with an absolute
        backstop, because the starting residual of a continuation step can
        already be small and demanding a fixed absolute value would then loop
        forever on round-off.
        """
        return max(1.0e-10 * f0, 1.0e-12)

    def _newton(self, psi: np.ndarray, n: np.ndarray, p: np.ndarray,
                st: _BCState) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        """Coupled Newton on ``(psi, n, p)`` with a residual-monotone line search."""
        cfg = self.cfg
        N = self.grid.n_nodes
        fixed = np.concatenate([st.psi_fixed, st.car_fixed, st.car_fixed])
        free = ~fixed
        test = self.semi_node & ~st.car_fixed
        hist: list[float] = []

        psi, n, p = psi.copy(), n.copy(), p.copy()
        self._apply_fixed(psi, n, p, st)
        n = np.maximum(n, np.where(self.semi_node, self.dens_floor, 0.0))
        p = np.maximum(p, np.where(self.semi_node, self.dens_floor, 0.0))

        stall = 0
        r0: np.ndarray | None = None
        for it in range(1, cfg.max_newton + 1):
            cn, cp = self._density_scale(n), self._density_scale(p)
            col = np.concatenate([np.full(N, self.Vt), cn, cp])
            F = self._residual(psi, n, p, st)
            J = self._jacobian(psi, n, p, st)
            Js, Fs, r = self._equilibrate(J, F, col)
            # The row weights that make the *linear solve* well conditioned
            # change every iteration, because the column scales follow the
            # densities.  A merit function whose weights move is not a merit
            # function -- the line search would compare incomparable numbers
            # and happily accept an increase.  So the equilibration is used for
            # the solve, and a weighting frozen at the first iterate is used
            # for the norm.
            if r0 is None:
                r0 = r.copy()
            nrm = float(np.linalg.norm((r0 * F)[free]))
            hist.append(nrm)
            if not np.isfinite(nrm):
                raise ConvergenceError("non-finite residual", hist)
            if nrm == 0.0:
                break

            dy = self._lu_solve(Js, -Fs, hist)
            dpsi = self.Vt * dy[:N]
            dn = cn * dy[N:2 * N]
            dp = cp * dy[2 * N:]

            mx = float(np.max(np.abs(dpsi))) if dpsi.size else 0.0
            lam = 1.0 if mx == 0.0 else min(1.0, 5.0 * self.Vt / mx)

            ok = False
            for _ in range(60):
                psi_t = psi + lam * dpsi
                n_t = self._project(n + lam * dn)
                p_t = self._project(p + lam * dp)
                self._apply_fixed(psi_t, n_t, p_t, st)
                Ft = r0 * self._residual(psi_t, n_t, p_t, st)
                nt = float(np.linalg.norm(Ft[free]))
                if np.isfinite(nt) and nt < (1.0 - 1.0e-4 * lam) * nrm:
                    ok = True
                    break
                lam *= 0.5
            if not ok:
                # No step of any length reduces the residual: either we sit at
                # the round-off floor (accept) or we are genuinely stuck.
                if nrm <= self._accept(hist[0]):
                    break
                raise ConvergenceError(
                    f"line search failed at Newton iteration {it} "
                    f"(|F| = {nrm:.4e})", hist, np.concatenate([psi, n, p]))

            d_psi = float(np.max(np.abs(lam * dpsi))) / self.Vt
            den_n = np.maximum(np.abs(n), self.ni * 1.0e-6)
            den_p = np.maximum(np.abs(p), self.ni * 1.0e-6)
            d_n = float(np.max(np.abs(n_t - n)[test] / den_n[test])) if test.any() else 0.0
            d_p = float(np.max(np.abs(p_t - p)[test] / den_p[test])) if test.any() else 0.0
            psi, n, p = psi_t, n_t, p_t
            self._log(2, f"newton {it:2d}  |F| {nt:.4e}  lam {lam:.3g}  "
                         f"dpsi/Vt {d_psi:.3e}  dn/n {d_n:.3e}")

            # Stop only when the residual has stopped improving as well: a
            # small *update* on its own is not evidence of convergence when the
            # line search has been cutting the step, and terminal currents are
            # read straight off the residual, so a lazy stop shows up directly
            # as a Kirchhoff-law violation.
            stall = stall + 1 if nt > 0.5 * nrm else 0
            converged = (d_psi < cfg.newton_tol and d_n < cfg.newton_tol
                         and d_p < cfg.newton_tol)
            # A small *update* is not on its own evidence of convergence: when
            # the line search has cut lambda to 1e-10 the iterate barely moves
            # while the residual is still O(10).  Accepting that produced a
            # smooth-looking reverse I-V made entirely of nonsense, so the
            # update test is only allowed to end the solve once the residual
            # has already fallen six orders below where it started.
            small = nt <= self._accept(hist[0])
            if small or (converged and stall >= 1
                         and nt <= 1.0e-6 * hist[0]):
                hist.append(nt)
                break
            if stall >= 4:
                # Creeping downhill without getting anywhere.  Report it: a
                # silently accepted stall is the failure mode that produces a
                # plausible-looking I-V curve made of nonsense, and the caller
                # (bias ramp / sweep sub-stepper) can recover from an exception
                # but not from a lie.
                hist.append(nt)
                raise ConvergenceError(
                    f"Newton stalled at iteration {it} (|F| = {nt:.4e}, "
                    f"started at {hist[0]:.4e})", hist,
                    np.concatenate([psi, n, p]))
        else:
            raise ConvergenceError(
                f"coupled Newton failed to converge in {cfg.max_newton} "
                f"iterations (|F| = {hist[-1]:.4e})", hist,
                np.concatenate([psi, n, p]))
        return psi, n, p, hist

    def _project(self, x: np.ndarray) -> np.ndarray:
        """Clamp densities to a positive floor inside the semiconductor."""
        return np.where(self.semi_node, np.maximum(x, self.dens_floor), 0.0)

    # ---------------------------------------------------------------- Gummel
    def _gummel(self, psi: np.ndarray, n: np.ndarray, p: np.ndarray,
                st: _BCState, max_outer: int | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
        """Decoupled Gummel map: nonlinear Poisson, then two linear continuities.

        Far more robust than coupled Newton when the initial guess is poor,
        and far slower to converge when it is good (linear rather than
        quadratic, with a rate that degrades as the applied bias grows).  It
        exists as the fallback path, not as the default.
        """
        cfg = self.cfg
        max_outer = max_outer or max(40, cfg.max_newton)
        N = self.grid.n_nodes
        fixed = np.concatenate([st.psi_fixed, st.car_fixed, st.car_fixed])
        free = ~fixed
        hist: list[float] = []

        psi, n, p = psi.copy(), self._project(n), self._project(p)
        self._apply_fixed(psi, n, p, st)

        for it in range(1, max_outer + 1):
            # 1. Nonlinear Poisson with the quasi-Fermi levels frozen.
            phin = psi - self.Vt * np.log(np.maximum(n, self.dens_floor) / self.ni)
            phip = psi + self.Vt * np.log(np.maximum(p, self.dens_floor) / self.ni)
            psi = self._gummel_poisson(psi, phin, phip, st)

            # 2/3. Linear continuity solves at frozen psi.
            n = self._continuity_linear(psi, n, p, st, carrier="n")
            p = self._continuity_linear(psi, n, p, st, carrier="p")

            F = self._residual(psi, n, p, st)
            J = self._jacobian(psi, n, p, st)
            col = np.concatenate([np.full(N, self.Vt),
                                  self._density_scale(n), self._density_scale(p)])
            _, Fs, _ = self._equilibrate(J, F, col)
            nrm = float(np.linalg.norm(Fs[free]))
            hist.append(nrm)
            self._log(2, f"gummel {it:2d}  |F| {nrm:.4e}")
            if nrm < 1e-8:
                break
        return psi, n, p, hist

    def _gummel_poisson(self, psi: np.ndarray, phin: np.ndarray,
                        phip: np.ndarray, st: _BCState) -> np.ndarray:
        """Newton on Poisson with ``n, p`` expressed through frozen quasi-Fermi levels."""
        free = ~st.psi_fixed
        scale = q * self.N_ref * self.V_node

        def np_of(ps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            un = np.clip((ps - phin) / self.Vt, -400.0, 400.0)
            up = np.clip((phip - ps) / self.Vt, -400.0, 400.0)
            nn = np.where(self.semi_node, self.ni * np.exp(un), 0.0)
            pp = np.where(self.semi_node, self.ni * np.exp(up), 0.0)
            return nn, pp

        def resid(ps: np.ndarray) -> np.ndarray:
            nn, pp = np_of(ps)
            F = self.L_eps @ ps - q * (pp - nn + self.dop) * self.V_semi
            F[st.psi_fixed] = 0.0
            return F / scale

        F = resid(psi)
        nrm = float(np.linalg.norm(F[free]))
        for _ in range(40):
            nn, pp = np_of(psi)
            J = self.L_eps + sp.diags(q * (nn + pp) * self.V_semi / self.Vt)
            J = sp.csr_matrix(sp.diags(free.astype(float)) @ J
                              + sp.diags(st.psi_fixed.astype(float)))
            Js, Fs, _ = self._equilibrate(J, F * scale,
                                          np.full(self.grid.n_nodes, self.Vt))
            d = self.Vt * self._lu_solve(Js, -Fs, [])
            mx = float(np.max(np.abs(d))) if d.size else 0.0
            lam = 1.0 if mx == 0.0 else min(1.0, 5.0 * self.Vt / mx)
            for _ in range(50):
                trial = psi + lam * d
                Ft = resid(trial)
                nt = float(np.linalg.norm(Ft[free]))
                if nt < (1.0 - 1e-4 * lam) * nrm:
                    break
                lam *= 0.5
            psi, F, nrm = trial, Ft, nt
            if mx * lam < 1e-10 * self.Vt or nrm < 1e-13:
                break
        return psi

    def _continuity_linear(self, psi: np.ndarray, n: np.ndarray,
                           p: np.ndarray, st: _BCState, carrier: str
                           ) -> np.ndarray:
        """One linear continuity solve at frozen ``psi`` and frozen other carrier."""
        sgv = self._sg(psi, n, p)
        Cn, Cp, Bp, Bm = sgv["Cn"], sgv["Cp"], sgv["Bp"], sgv["Bm"]
        ne = np.arange(self.grid.n_edges)
        rows = np.concatenate([ne, ne])
        cols = np.concatenate([self.head, self.tail])
        ni2 = self.ni * self.ni
        qV = q * self.V_semi

        # Linearised net recombination R = k (n p - ni^2) with k frozen.
        k = np.zeros_like(n)
        if self.rec.srh:
            k += 1.0 / (self.tau_p * (n + self.n1) + self.tau_n * (p + self.p1))
        if self.rec.auger and (self.C_aug_n or self.C_aug_p):
            k += self.C_aug_n * n + self.C_aug_p * p

        if carrier == "n":
            A = sp.coo_matrix((np.concatenate([Cn * Bp, -Cn * Bm]), (rows, cols)),
                              shape=(self.grid.n_edges, self.grid.n_nodes)).tocsr()
            M = self.Gt @ A + sp.diags(qV * k * p)
            b = qV * k * ni2
            fixed, val = st.car_fixed, st.n_val
        else:
            A = sp.coo_matrix((np.concatenate([-Cp * Bm, Cp * Bp]), (rows, cols)),
                              shape=(self.grid.n_edges, self.grid.n_nodes)).tocsr()
            M = self.Gt @ A - sp.diags(qV * k * n)
            b = -qV * k * ni2
            fixed, val = st.car_fixed, st.p_val

        if st.dt is not None:
            w = qV / st.dt
            old = st.n_old if carrier == "n" else st.p_old
            if carrier == "n":
                M = M + sp.diags(w)
                b = b + w * old
            else:
                M = M - sp.diags(w)
                b = b - w * old

        M = sp.csr_matrix(sp.diags((~fixed).astype(float)) @ M
                          + sp.diags(fixed.astype(float)))
        b = np.where(fixed, val, b)
        rmax = np.asarray(abs(M).max(axis=1).todense()).ravel()
        r = np.where(rmax > 0.0, 1.0 / np.where(rmax > 0.0, rmax, 1.0), 1.0)
        x = self._lu_solve(sp.csr_matrix(sp.diags(r) @ M), r * b, [])
        return self._project(x)

    # ============================================================= transient
    def solve_transient(self, terminals: Sequence[Terminal], t_end: float,
                        dt: float, bc: BoundarySpec | None = None,
                        x0: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                        monitors: Any = None, store_fields: bool = True,
                        method: str = "newton") -> Result:
        """Backward-Euler transient.

        The Poisson equation stays quasi-static (**A1a**): displacement current
        appears through the time derivative of the contact charge, not through
        a ``d/dt`` term in Gauss' law.  Only the two continuity equations carry
        a time derivative, which is exactly the drift-diffusion convention.

        Parameters
        ----------
        terminals : sequence of Terminal
            Voltage-driven electrodes; callables are evaluated at each step.
        t_end : float
            Final time [s].
        dt : float
            Fixed step [s].  Backward Euler is L-stable so there is no
            stability limit, only an accuracy one; the dielectric relaxation
            time of the doped regions is the scale to resolve.
        bc : BoundarySpec, optional
        x0 : tuple, optional
            ``(psi, n, p)`` initial state.  Defaults to the DC solution at
            ``t = 0``, which is the physically correct initial condition ---
            starting from zero is a classic and completely wrong choice.
        monitors : MonitorSet, optional
            Recorded with ``state`` keys ``psi``, ``n``, ``p``, ``terminals``.
        store_fields : bool
            Store ``psi``, ``n``, ``p`` every ``config.store_every`` steps.
        method : {'newton', 'gummel'}

        Returns
        -------
        Result
        """
        self._start()
        terminals = list(terminals)
        if dt <= 0.0:
            raise ValueError(f"dt must be positive [s], got {dt}")
        if t_end <= 0.0:
            raise ValueError(f"t_end must be positive [s], got {t_end}")

        times = self._time_steps(t_end, dt)
        if x0 is None:
            r0 = self.solve_dc(terminals, bc, ramp=True, t=float(times[0]))
            psi = r0.fields["psi"][0].ravel().copy()
            n = r0.fields["n"][0].ravel().copy()
            p = r0.fields["p"][0].ravel().copy()
        else:
            psi, n, p = (np.array(a, dtype=float).ravel() for a in x0)

        store = max(1, int(self.cfg.store_every))
        psis, ns, ps, tstore = [], [], [], []
        tv: dict[str, list[float]] = {tm.name: [] for tm in terminals}
        ti: dict[str, list[float]] = {tm.name: [] for tm in terminals}

        def record(idx: int, tt: float, cur: dict[str, dict[str, float]]) -> None:
            for tm in terminals:
                tv[tm.name].append(cur[tm.name]["v"])
                ti[tm.name].append(cur[tm.name]["i"])
            if store_fields and idx % store == 0:
                psis.append(psi.reshape(self.grid.shape_nodes).copy())
                ns.append(n.reshape(self.grid.shape_nodes).copy())
                ps.append(p.reshape(self.grid.shape_nodes).copy())
                tstore.append(tt)

        st0 = self._bc_state(terminals, bc, float(times[0]))
        cur = self._terminal_currents(psi, n, p, st0, terminals)
        record(0, float(times[0]), cur)
        if monitors is not None:
            monitors.record({"psi": psi, "n": n, "p": p, "terminals": cur},
                            float(times[0]))

        hist_total = 0
        for k in range(1, times.size):
            tk = float(times[k])
            st = self._bc_state(terminals, bc, tk)
            st.dt = dt
            st.n_old, st.p_old, st.psi_old = n.copy(), p.copy(), psi.copy()
            psi_prev = psi.copy()
            psi, n, p, h = self._nonlinear_solve(psi.copy(), n.copy(), p.copy(),
                                                 st, method)
            hist_total += len(h)
            cur = self._terminal_currents(psi, n, p, st, terminals,
                                          psi_old=psi_prev, dt=dt)
            record(k, tk, cur)
            if monitors is not None:
                monitors.record({"psi": psi, "n": n, "p": p, "terminals": cur},
                                tk)
            self._log(1, f"t = {tk:.6g} s  ({len(h)} newton)")

        res = Result(grid=self.grid, t=times)
        if store_fields:
            res.fields["psi"] = np.array(psis)
            res.fields["n"] = np.array(ns)
            res.fields["p"] = np.array(ps)
            res.fields["t_field"] = np.array(tstore)
        res.terminals = {name: {"v": np.array(tv[name]), "i": np.array(ti[name])}
                         for name in tv}
        return self._finish(res, newton_iterations=hist_total, dt=dt,
                            temperature=self.T,
                            speedup_vs_courant=self.speedup_vs_courant(dt))

    # ================================================================ I-V sweep
    def iv_curve(self, sweep_terminal: Terminal, values: Sequence[float],
                 others: Sequence[Terminal] = (),
                 bc: BoundarySpec | None = None,
                 method: str = "newton") -> Result:
        """DC sweep of one terminal, using each solution to start the next.

        Parameters
        ----------
        sweep_terminal : Terminal
            The electrode whose voltage is swept.  Its own ``voltage`` field is
            overridden by ``values``.
        values : sequence of float
            Bias points [V], in the order they should be walked.  Continuation
            makes a monotone sweep far cheaper and far more robust than
            independent solves, so order matters.
        others : sequence of Terminal
            The remaining electrodes, held at their own values.
        bc : BoundarySpec, optional
        method : {'newton', 'gummel'}

        Returns
        -------
        Result
            ``scalars['bias']`` holds the swept voltages, ``terminals`` holds
            ``v`` and ``i`` per electrode, and ``fields`` holds the final
            ``psi``, ``n``, ``p``.  ``t`` is the bias index, since an I-V sweep
            has no time axis.
        """
        self._start()
        vals = np.asarray(values, dtype=float).ravel()
        if vals.size == 0:
            raise ValueError("values must contain at least one bias point")
        others = list(others)
        allterm = [sweep_terminal] + others

        tv: dict[str, list[float]] = {tm.name: [] for tm in allterm}
        ti: dict[str, list[float]] = {tm.name: [] for tm in allterm}
        state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        psi = n = p = None
        prev_bias: dict[str, float] | None = None
        for k, v in enumerate(vals):
            bias = {sweep_terminal.name: float(v)}
            for tm in others:
                bias[tm.name] = self._terminal_target(tm, 0.0)
            psi, n, p, st = self._solve_at_bias(allterm, bias, bc, state,
                                                method, prev_bias)
            state = (psi, n, p)
            prev_bias = bias
            cur = self._terminal_currents(psi, n, p, st, allterm)
            for tm in allterm:
                tv[tm.name].append(cur[tm.name]["v"])
                ti[tm.name].append(cur[tm.name]["i"])
            self._log(1, f"iv {k + 1}/{vals.size}  V = {v:+.4f} V  "
                         f"I = {cur[sweep_terminal.name]['i']:+.6e} A")

        res = Result(grid=self.grid, t=np.arange(vals.size, dtype=float))
        res.fields["psi"] = psi.reshape((1,) + self.grid.shape_nodes)
        res.fields["n"] = n.reshape((1,) + self.grid.shape_nodes)
        res.fields["p"] = p.reshape((1,) + self.grid.shape_nodes)
        res.terminals = {name: {"v": np.array(tv[name]), "i": np.array(ti[name])}
                         for name in tv}
        res.scalars["bias"] = vals
        return self._finish(res, sweep=sweep_terminal.name,
                            temperature=self.T, ni=self.ni)

    def _solve_at_bias(self, terminals: Sequence[Terminal],
                       bias: dict[str, float], bc: BoundarySpec | None,
                       state: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
                       method: str, prev_bias: dict[str, float] | None = None):
        """One bias point, continued from ``state``; sub-steps on failure."""
        st = self._bc_state(terminals, bc, 0.0, bias=bias)
        psi, n, p = (a.copy() for a in state) if state is not None else \
            self._equilibrium_start(terminals, bc)
        self._apply_fixed(psi, n, p, st)
        try:
            psi, n, p, _ = self._nonlinear_solve(psi, n, p, st, method)
            return psi, n, p, st
        except ConvergenceError:
            pass

        # Too big a jump.  Bisect the path from the previous bias, doubling the
        # step again after each success so a single hard point does not slow
        # down the rest of the sweep.
        base = dict(prev_bias or {tm.name: 0.0 for tm in terminals})
        psi0, n0, p0 = (a.copy() for a in state) if state is not None else \
            self._equilibrium_start(terminals, bc)
        alpha, step = 0.0, 0.5
        guard = 0
        while alpha < 1.0 - 1e-12:
            guard += 1
            if guard > 200:
                raise ConvergenceError(
                    f"bias point {bias} unreachable: sub-stepping stalled at "
                    f"alpha = {alpha:.4g}")
            a_try = min(1.0, alpha + step)
            sub = {k: base.get(k, 0.0) + a_try * (bias[k] - base.get(k, 0.0))
                   for k in bias}
            st_s = self._bc_state(terminals, bc, 0.0, bias=sub)
            trial = (psi0.copy(), n0.copy(), p0.copy())
            self._apply_fixed(*trial, st_s)
            try:
                psi0, n0, p0, _ = self._nonlinear_solve(*trial, st_s, method)
            except ConvergenceError:
                step *= 0.5
                if step < 1e-4:
                    raise
                continue
            alpha = a_try
            step = min(1.0 - alpha if alpha < 1.0 else 1.0, step * 2.0) or step
        return psi0, n0, p0, st

    def _equilibrium_start(self, terminals: Sequence[Terminal],
                           bc: BoundarySpec | None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        st0 = self._bc_state(terminals, bc, 0.0,
                             bias={tm.name: 0.0 for tm in terminals})
        psi = self._poisson_newton(self._initial_psi(st0), st0, 0.0)[0]
        n, p = self.equilibrium_densities(psi, 0.0)
        n, p = self._project(n), self._project(p)
        self._apply_fixed(psi, n, p, st0)
        return psi, n, p

    # ------------------------------------------------------ precision floor
    def current_noise_floor(self, psi: np.ndarray, n: np.ndarray,
                            p: np.ndarray, terminal: Terminal) -> float:
        """Smallest terminal current [A] this discretisation can resolve.

        Scharfetter-Gummel computes a flux as the difference of two terms of
        size ``q D (A/L) n``, and in equilibrium those two terms are equal, so
        the numerically computed current is a difference of nearly equal
        doubles.  The rounding residue is

        ``eps_machine * q D (A/L) * max(n_head B(X), n_tail B(-X))``

        summed over the edges touching the contact, plus the hole counterpart.
        Any reported current below this number is floating-point noise, not
        physics, and no amount of Newton convergence removes it: it is a
        property of using **linear carrier densities** as the unknowns.

        The practical consequences are worth stating, because they decide how a
        leakage simulation must be set up:

        * The floor scales as ``1/dx`` at the contact, so a needlessly fine
          mesh *next to an ohmic contact* buys nothing and costs leakage
          resolution.  Refine at the junction, not at the contact.
        * It scales as the majority density ``N``, while a diode's saturation
          current scales as ``ni^2/N``, so the usable dynamic range falls as
          ``N^2``.  Reverse leakage of a 1e18 cm^-3 junction is not resolvable
          in double precision on any mesh.

        Parameters
        ----------
        psi, n, p : np.ndarray
            Converged solution.
        terminal : Terminal
            The contact of interest.

        Returns
        -------
        float
            Estimated noise floor [A].
        """
        sgv = self._sg(psi, n, p)
        nodes = set(np.asarray(terminal.nodes, dtype=np.intp).tolist())
        eps_m = float(np.finfo(float).eps)
        mask = np.array([(h in nodes) or (t in nodes)
                         for h, t in zip(self.head, self.tail)])
        e = np.flatnonzero(mask)
        if e.size == 0:
            return 0.0
        nh, nt = n[self.head[e]], n[self.tail[e]]
        ph, pt = p[self.head[e]], p[self.tail[e]]
        Bp, Bm = sgv["Bp"][e], sgv["Bm"][e]
        val = (sgv["Cn"][e] * np.maximum(nh * np.abs(Bp), nt * np.abs(Bm))
               + sgv["Cp"][e] * np.maximum(pt * np.abs(Bp), ph * np.abs(Bm)))
        return float(eps_m * np.sum(val))

    # ============================================================== currents
    def _terminal_currents(self, psi: np.ndarray, n: np.ndarray,
                           p: np.ndarray, st: _BCState,
                           terminals: Sequence[Terminal],
                           psi_old: np.ndarray | None = None,
                           dt: float | None = None
                           ) -> dict[str, dict[str, float]]:
        """Terminal voltages [V] and currents [A] from the *unenforced* residual.

        A contact node's equations are pinned, so their residual is exactly the
        current (and charge) the external circuit had to supply.  Evaluating
        the residual there with the Dirichlet mask lifted therefore gives the
        terminal current with no interpolation and no flux surface to choose,
        and it satisfies Kirchhoff's law across the whole device to round-off.

        Sign convention: **positive current flows from the external circuit
        into the terminal**, matching ``Gᵀ M_sigma G phi = i_inject`` in
        ``docs/CONTRACTS.md``.
        """
        sgv = self._sg(psi, n, p)
        R, _, _ = self._recombination(n, p)
        Fn = self.Gt @ sgv["In"] + q * R * self.V_semi
        Fp = self.Gt @ sgv["Ip"] - q * R * self.V_semi
        if st.dt is not None and st.n_old is not None:
            w = q * self.V_semi / st.dt
            Fn = Fn + w * (n - st.n_old)
            Fp = Fp - w * (p - st.p_old)
        Qc = self.L_eps @ psi - q * (p - n + self.dop) * self.V_semi
        Qc_old = None
        if psi_old is not None and dt is not None and st.n_old is not None:
            Qc_old = (self.L_eps @ psi_old
                      - q * (st.p_old - st.n_old + self.dop) * self.V_semi)

        out: dict[str, dict[str, float]] = {}
        for term in terminals:
            nodes = np.asarray(term.nodes, dtype=np.intp)
            kind = self._contact_kind(term)
            i_n = float(-np.sum(Fn[nodes])) if kind == "ohmic" else 0.0
            i_p = float(-np.sum(Fp[nodes])) if kind == "ohmic" else 0.0
            i_d = 0.0
            if Qc_old is not None:
                i_d = float(np.sum(Qc[nodes] - Qc_old[nodes]) / dt)
            v = float(np.mean(psi[nodes]))
            applied = self._terminal_target(term, 0.0)
            out[term.name] = {"v": applied, "i": i_n + i_p + i_d,
                              "i_n": i_n, "i_p": i_p, "i_disp": i_d,
                              "psi": v}
        return out

    # ================================================================ results
    def _make_result(self, psi: np.ndarray, n: np.ndarray, p: np.ndarray,
                     st: _BCState, terminals: Sequence[Terminal],
                     times: Sequence[float]) -> Result:
        shape = (1,) + self.grid.shape_nodes
        res = Result(grid=self.grid, t=np.asarray(times, dtype=float))
        res.fields["psi"] = psi.reshape(shape)
        res.fields["n"] = n.reshape(shape)
        res.fields["p"] = p.reshape(shape)
        res.fields["rho"] = (q * (p - n + self.dop)).reshape(shape)
        cur = self._terminal_currents(psi, n, p, st, terminals)
        res.terminals = {name: {"v": np.array([d["v"]]),
                                "i": np.array([d["i"]])}
                         for name, d in cur.items()}
        for name, d in cur.items():
            res.scalars[f"{name}.i_n"] = np.array([d["i_n"]])
            res.scalars[f"{name}.i_p"] = np.array([d["i_p"]])
            res.scalars[f"{name}.psi"] = np.array([d["psi"]])
        # Kirchhoff's law across the whole device is the honest, *measured*
        # error bar on a DC terminal current: the currents must sum to zero, so
        # whatever they sum to instead is the error.  Report it rather than
        # leaving the user to discover it.
        tot = sum(d["i"] for d in cur.values())
        big = max((abs(d["i"]) for d in cur.values()), default=0.0)
        res.scalars["kirchhoff_sum"] = np.array([tot])
        res.scalars["kirchhoff_relative"] = np.array(
            [abs(tot) / big if big > 0 else 0.0])
        return res

    # ------------------------------------------------------------------ API
    def solve(self, terminals: Sequence[Terminal],
              bc: BoundarySpec | None = None, **kw: Any) -> Result:
        """Alias for :meth:`solve_dc` (the :class:`SolverBase` entry point)."""
        return self.solve_dc(terminals, bc, **kw)

    # ------------------------------------------------------------ diagnostics
    def depletion_edges(self, psi: np.ndarray, n: np.ndarray, p: np.ndarray,
                        axis: int = 0, threshold: float = 0.1
                        ) -> dict[str, float]:
        """Space-charge-region width [m] measured two independent ways.

        Parameters
        ----------
        psi, n, p : np.ndarray
            Solution vectors (flat or node-shaped).
        axis : int
            Which grid direction the junction is normal to.
        threshold : float
            Fraction of the peak charge density defining the ``w_threshold``
            edge.

        Returns
        -------
        dict
            ``w_threshold`` --- span between the outermost nodes whose charge
            density exceeds ``threshold`` times the peak.  ``w_charge`` --- the
            depletion-approximation width implied by the actual integrated
            space charge, ``Q+/(q Nd) + Q-/(q Na)``.  The second is the one to
            compare against :func:`fieldspice.reference.depletion_width`
            scaling, because it is insensitive to where the tails are cut off.

        Notes
        -----
        The absolute width disagrees with the depletion approximation by ~20 %
        at zero bias.  That is a property of the reference formula (abrupt
        space-charge edges), not of this solver, and it does not shrink under
        mesh refinement --- see ``docs/CONTRACTS.md``.
        """
        shp = self.grid.shape_nodes
        rho = (q * (np.ravel(p) - np.ravel(n) + self.dop)).reshape(shp)
        dop = self.dop.reshape(shp)
        coords = self.grid.node_coords()[axis]
        # Collapse the transverse directions: they are uniform by symmetry.
        take = [slice(None)] * 3
        for d in range(3):
            if d != axis:
                take[d] = 0
        line = rho[tuple(take)]
        x = coords[tuple(take)]
        dline = dop[tuple(take)]

        peak = float(np.max(np.abs(line)))
        if peak <= 0.0:
            return {"w_threshold": 0.0, "w_charge": 0.0}
        # Linear interpolation of the |rho| = threshold*peak crossing.  Taking
        # the nearest node instead quantises the width by the local cell size,
        # which on a graded mesh is tens of nanometres out in the tail and
        # swamps the scaling test.
        a = np.abs(line) - threshold * peak
        sel = np.flatnonzero(a > 0.0)
        if sel.size:
            i0, i1 = int(sel[0]), int(sel[-1])
            xlo = float(x[i0])
            if i0 > 0:
                f = a[i0] / (a[i0] - a[i0 - 1])
                xlo = float(x[i0] + f * (x[i0 - 1] - x[i0]))
            xhi = float(x[i1])
            if i1 < x.size - 1:
                f = a[i1] / (a[i1] - a[i1 + 1])
                xhi = float(x[i1] + f * (x[i1 + 1] - x[i1]))
            w_thr = xhi - xlo
        else:
            w_thr = 0.0

        # Integrated charge on each side, converted to a depletion width via
        # the local majority doping.
        dx = np.gradient(x)
        Nd = float(np.max(dline)) if np.any(dline > 0) else 0.0
        Na = float(-np.min(dline)) if np.any(dline < 0) else 0.0
        qp = float(np.sum(np.maximum(line, 0.0) * dx))
        qn = float(np.sum(np.maximum(-line, 0.0) * dx))
        w_q = (qp / (q * Nd) if Nd > 0 else 0.0) + (qn / (q * Na) if Na > 0 else 0.0)
        return {"w_threshold": w_thr, "w_charge": w_q}
