"""Lattice heat conduction --- the same machinery, with heat instead of charge.

The heat equation

    rho c_p dT/dt = div(kappa grad T) + q

is structurally identical to the electroquasistatic equation this package is
built around. Discretised on the same grid with the same operators it becomes

    C_th dT/dt + G^T M_kappa G T = Q

where ``M_kappa = edge_mass(grid, kappa_edge)`` is a **thermal conductance**
[W/K] and ``C_th = diag(rho c_p V_node)`` a **heat capacity** [J/K]. In other
words a thermal problem is an RC network too, and every piece of infrastructure
built for the electrical solver --- the incidence operators, the mass matrices,
the cached factorisation, the Dirichlet machinery --- carries over unchanged.
That structural identity is why adding self-heating was cheap: it needed a new
coefficient, not a new solver.

Boundary conditions
-------------------
Three kinds, and the third matters more than people expect:

* **Dirichlet** --- a fixed temperature. A perfect heat sink.
* **Neumann** --- zero normal flux, i.e. adiabatic. The natural condition, so
  it needs no action at all.
* **Robin / convective** --- ``-kappa dT/dn = h (T - T_inf)``. A finite heat
  sink. It adds ``h * A_dual`` to the diagonal and ``h * A_dual * T_inf`` to the
  right-hand side, preserving symmetry.

A purely Dirichlet model exaggerates cooling (the sink is infinitely good) and
a purely adiabatic one has no steady state under any heat load at all --- the
temperature rises without bound, which is correct for the stated problem and
almost never what was meant. Robin is usually the honest choice, and
:meth:`ThermalSolver.steady` warns when neither Dirichlet nor convection is
present.

Assumptions
-----------
Fourier conduction only: no radiation, no convective transport inside the
domain, no ballistic phonon effects (which matter below roughly the phonon mean
free path, ~100 nm in silicon at 300 K, where this model over-predicts
conduction). Anisotropic conductivity is not supported, for the same reason the
electrical solver is isotropic: the mass matrices are diagonal.
"""

from __future__ import annotations

import warnings
from typing import Callable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from ..grid import RectilinearGrid
from ..operators import (Operators, apply_dirichlet, cell_to_edge, edge_mass,
                         nodal_laplacian, node_volume_vector)
from .base import Result, SolverConfig, TimeSteppingSolver
from .poisson import LinearSystem

__all__ = ["ThermalSolver", "joule_heating_nodes", "WALLS"]

WALLS = ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")


def joule_heating_nodes(grid: RectilinearGrid, sigma_edge: np.ndarray,
                        phi: np.ndarray, ops: Operators | None = None
                        ) -> np.ndarray:
    """Nodal Joule heat source [W] from an electrical potential.

    Power dissipated in edge ``e`` is ``G_e (dphi_e)^2`` with ``G_e`` the edge
    conductance, and half is attributed to each end node.

    The distribution is exactly conservative: ``sum(q_node)`` equals
    ``phi^T (G^T M_sigma G) phi``, the total dissipation, to machine precision.
    That is worth checking in any coupled run, because a heat source that does
    not match the electrical power is the classic way an electro-thermal
    simulation goes quietly wrong.
    """
    ops = ops or Operators(grid)
    G = ops.G
    dphi = G @ phi
    p_edge = edge_mass(grid, sigma_edge).diagonal() * dphi ** 2
    # abs(G) has exactly two unit entries per edge, so abs(G).T @ p sums each
    # edge's power into both endpoints; half of that is the correct split.
    absG = sp.csr_matrix((np.abs(G.data), G.indices, G.indptr), shape=G.shape)
    return 0.5 * (absG.T @ p_edge)


class ThermalSolver(TimeSteppingSolver):
    """Fourier heat conduction on the fieldspice grid.

    Parameters
    ----------
    grid
        The mesh.
    kappa_cell
        Thermal conductivity per cell [W/(m K)].
    rho_cp_cell
        Volumetric heat capacity per cell [J/(m^3 K)]. Required for transient
        solves, ignored for steady ones.
    """

    name = "thermal"
    assumptions = ("A2", "A10")

    def __init__(self, grid: RectilinearGrid, kappa_cell: np.ndarray,
                 rho_cp_cell: np.ndarray | None = None,
                 config: SolverConfig | None = None,
                 operators: Operators | None = None):
        super().__init__(grid, config, operators)
        kappa_cell = np.asarray(kappa_cell, dtype=float)
        if kappa_cell.shape != grid.shape_cells:
            raise ValueError(
                f"kappa_cell must have shape {grid.shape_cells}, "
                f"got {kappa_cell.shape}")
        if np.any(kappa_cell <= 0.0):
            raise ValueError(
                "kappa_cell must be strictly positive [W/(m K)]; a zero makes "
                "the heat equation singular in that region. Use a small "
                "positive floor for vacuum.")
        self.kappa_cell = kappa_cell
        self.kappa_edge = cell_to_edge(grid, kappa_cell)
        self.K = nodal_laplacian(grid, self.kappa_edge, G=self.ops.G)  # [W/K]

        self.node_vol = node_volume_vector(grid)
        if rho_cp_cell is not None:
            rho_cp_cell = np.asarray(rho_cp_cell, dtype=float)
            if rho_cp_cell.shape != grid.shape_cells:
                raise ValueError(
                    f"rho_cp_cell must have shape {grid.shape_cells}")
            if np.any(rho_cp_cell <= 0.0):
                raise ValueError("rho_cp_cell must be strictly positive")
            from ..operators import cell_to_node
            self.C_th = cell_to_node(grid, rho_cp_cell).ravel() * self.node_vol
        else:
            self.C_th = None

    # ------------------------------------------------------------------
    def _wall_terms(self, convection: Mapping[str, tuple[float, float]] | None
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Diagonal and RHS contributions of convective walls."""
        diag = np.zeros(self.grid.n_nodes)
        rhs = np.zeros(self.grid.n_nodes)
        if not convection:
            return diag, rhs
        from ..boundaries import wall_dual_areas, wall_nodes
        for wall, spec in convection.items():
            if wall not in WALLS:
                raise ValueError(f"unknown wall {wall!r}; expected one of {WALLS}")
            h, T_inf = float(spec[0]), float(spec[1])
            if h < 0.0:
                raise ValueError("heat transfer coefficient must be >= 0")
            nodes = wall_nodes(self.grid, wall)
            area = np.asarray(wall_dual_areas(self.grid, wall), dtype=float).ravel()
            diag[nodes] += h * area
            rhs[nodes] += h * area * T_inf
        return diag, rhs

    def _dirichlet(self, bc, t: float) -> tuple[np.ndarray, np.ndarray]:
        if bc is None:
            return np.zeros(0, dtype=np.intp), np.zeros(0)
        idx, val = bc.dirichlet_nodes(self.grid, t)
        return np.asarray(idx, dtype=np.intp), np.asarray(val, dtype=float)

    # ------------------------------------------------------------------
    def solve(self, q_node: np.ndarray | float = 0.0, t_end: float | None = None,
              dt: float | None = None, **kw) -> Result:
        """Dispatch to :meth:`steady` or :meth:`transient`.

        Present because :class:`SolverBase` requires it; the named methods are
        clearer at a call site and are what the documentation uses.
        """
        if t_end is None or dt is None:
            return self.steady(q_node, **kw)
        return self.transient(q_node, t_end, dt, **kw)

    def steady(self, q_node: np.ndarray | float = 0.0, bc=None,
               convection: Mapping[str, tuple[float, float]] | None = None,
               ) -> Result:
        """Steady-state temperature field [K].

        Parameters
        ----------
        q_node
            Heat source per node [W]. Use :func:`joule_heating_nodes` to build
            it from an electrical solve.
        bc
            :class:`~fieldspice.boundaries.BoundarySpec`; ``Dirichlet`` walls
            become fixed-temperature heat sinks.
        convection
            ``{wall: (h, T_inf)}`` with ``h`` in W/(m^2 K).
        """
        self._start()
        n = self.grid.n_nodes
        q = (np.full(n, float(q_node)) if np.isscalar(q_node)
             else np.asarray(q_node, dtype=float).ravel())
        if q.size != n:
            raise ValueError(f"q_node must have {n} entries, got {q.size}")

        diag, rhs = self._wall_terms(convection)
        A = (self.K + sp.diags(diag)).tocsr()
        b = q + rhs
        fixed, vals = self._dirichlet(bc, 0.0)

        if fixed.size == 0 and not np.any(diag):
            warnings.warn(
                "thermal problem has neither a fixed-temperature boundary nor "
                "convection: it is fully adiabatic, so with any net heat input "
                "there is no steady state and the solution is defined only up "
                "to an additive constant. Add a Dirichlet wall (a perfect heat "
                "sink) or a convection coefficient (a finite one).",
                RuntimeWarning, stacklevel=2)

        if fixed.size:
            A, b = apply_dirichlet(A, b, fixed, vals)
        ls = LinearSystem(A, self.cfg)
        T = ls.solve(b)

        res = Result(grid=self.grid, t=np.zeros(1), fields={"T": T[None, :]})
        res.scalars["power_in"] = np.array([float(q.sum())])
        res.scalars["T_max"] = np.array([float(T.max())])
        res.scalars["T_rise"] = np.array([float(T.max() - T.min())])
        return self._finish(res, mode="steady",
                            linear=dict(getattr(ls, "info", {}) or {}))

    # ------------------------------------------------------------------
    def transient(self, q_node: np.ndarray | Callable[[float], np.ndarray],
                  t_end: float, dt: float, T0: np.ndarray | float = 300.0,
                  bc=None,
                  convection: Mapping[str, tuple[float, float]] | None = None,
                  theta: float = 1.0, store: bool = False,
                  store_every: int = 1) -> Result:
        """Transient temperature [K], backward Euler by default.

        Unconditionally stable, so ``dt`` is an accuracy choice. The natural
        scale is the thermal time constant ``R_th C_th``, which for on-chip
        structures is microseconds to milliseconds --- many orders of magnitude
        slower than the electrical transients driving it. That separation is
        why the coupled solver can hold temperature fixed within an electrical
        step (see :mod:`fieldspice.solvers.electrothermal`).
        """
        if self.C_th is None:
            raise ValueError(
                "transient thermal solve needs rho_cp_cell; pass it to the "
                "constructor (rho * c_p per cell, J/(m^3 K))")
        if dt <= 0:
            raise ValueError("dt must be positive")
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")
        self._start()

        n = self.grid.n_nodes
        T = (np.full(n, float(T0)) if np.isscalar(T0)
             else np.asarray(T0, dtype=float).copy())
        if T.shape != (n,):
            raise ValueError(f"T0 must have shape ({n},)")

        diag, wall_rhs = self._wall_terms(convection)
        K = (self.K + sp.diags(diag)).tocsr()
        C = sp.diags(self.C_th / dt)
        A_imp = (C + theta * K).tocsr()
        A_exp = (C - (1.0 - theta) * K).tocsr()

        times = self._time_points(t_end, dt)
        q_fn = q_node if callable(q_node) else (lambda t: q_node)

        Ts = [T.copy()]
        stored = [T.copy()] if store else []
        prepared: dict[bytes, LinearSystem] = {}
        for k, t in enumerate(times[1:], start=1):
            q = np.asarray(q_fn(t), dtype=float).ravel()
            if q.size == 1:
                q = np.full(n, float(q))
            b = A_exp @ T + q + wall_rhs
            fixed, vals = self._dirichlet(bc, t)
            A, bb = (apply_dirichlet(A_imp, b, fixed, vals) if fixed.size
                     else (A_imp, b))
            key = fixed.tobytes()
            if key not in prepared:
                prepared[key] = LinearSystem(A, self.cfg)
            T = prepared[key].solve(bb)
            Ts.append(T.copy())
            if store and k % max(1, store_every) == 0:
                stored.append(T.copy())

        res = Result(grid=self.grid, t=times)
        if store:
            res.fields["T"] = np.array(stored)
        res.scalars["T_max"] = np.array([float(x.max()) for x in Ts])
        res.scalars["T_mean"] = np.array([float(x.mean()) for x in Ts])
        return self._finish(res, mode="transient", theta=theta, dt=dt,
                            n_factorisations=len(prepared))

    # ------------------------------------------------------------------
    def thermal_resistance(self, source_nodes: np.ndarray, bc=None,
                           convection=None, power: float = 1.0) -> float:
        """Thermal resistance [K/W] from a node set to the boundary.

        Injects ``power`` spread over ``source_nodes`` and returns the mean
        temperature rise per watt. This is the number a package datasheet
        quotes, and it is what makes a distributed thermal solve comparable
        with a lumped network.
        """
        q = np.zeros(self.grid.n_nodes)
        nodes = np.asarray(source_nodes, dtype=np.intp).ravel()
        # Dual-volume weighting so the injected power DENSITY is uniform;
        # equal-per-node biases heat into low-volume corner nodes.
        w = self.node_vol[nodes]
        w = w / w.sum() if w.sum() > 0 else np.full(nodes.size, 1.0 / nodes.size)
        q[nodes] = power * w
        res = self.steady(q, bc=bc, convection=convection)
        T = res.fields["T"][0]
        T_ref = 0.0
        if bc is not None:
            fixed, vals = self._dirichlet(bc, 0.0)
            if fixed.size:
                T_ref = float(np.mean(vals))
        return float((float(np.dot(w, T[nodes])) - T_ref) / power)
