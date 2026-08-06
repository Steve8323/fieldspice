"""Electroquasistatic transient solver --- the workhorse of the package.

Solves, on the node potentials,

    G^T (M_sigma + M_eps d/dt) G phi = i_inject          (A1a)

which is ``div[(sigma + eps d/dt) grad phi] = 0`` in continuum form.  Because
``G^T M_sigma G`` is the nodal conductance matrix of a resistor mesh and
``G^T M_eps G`` the capacitance matrix of a capacitor mesh, this *is* nodal
analysis of a distributed RC network: the same equation SPICE solves, with the
network read off the grid instead of a netlist.

What it captures: resistance, capacitance, RC delay, capacitive crosstalk,
IR drop in power grids, substrate coupling, dielectric relaxation, charge
redistribution.  What it drops: inductance, magnetic energy, radiation --- for
those use :mod:`fieldspice.solvers.mqs` or :mod:`fieldspice.solvers.darwin`.

Why this is fast
----------------
The system is elliptic in space and parabolic in time, so implicit stepping is
unconditionally stable and the step size is set by *how fast the signal
changes*, not by how long light takes to cross a cell.  On the on-chip problem
in ``examples/00_why_quasistatic.py`` that is 2.1e4 explicit steps replaced by
one implicit step.  :meth:`speedup_vs_courant` reports the ratio for any run.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

from ..grid import RectilinearGrid
from ..operators import Operators, cell_to_edge, edge_mass, nodal_laplacian
from .base import Result, SolverConfig, Terminal, TimeSteppingSolver
from .poisson import LinearSystem

__all__ = ["EQSSolver"]


class EQSSolver(TimeSteppingSolver):
    """Time-domain electroquasistatic solver.

    Parameters
    ----------
    grid
        The mesh.
    eps_cell
        Absolute permittivity per cell [F/m], shape ``grid.shape_cells``.
        Use :meth:`fieldspice.materials.MaterialMap.eps`.
    sigma_cell
        Conductivity per cell [S/m], same shape.
    config
        Solver options; see :class:`~fieldspice.solvers.base.SolverConfig`.

    Notes
    -----
    Assumptions invoked: **A1a** (electroquasistatic --- no inductance, no
    radiation), **A2** (staircased rectilinear geometry), **A3** (linear
    isotropic non-dispersive materials), **A10** (box-method discretisation),
    **A12** (Neumann open boundaries underestimate fringing).
    """

    name = "eqs"
    assumptions = ("A1a", "A2", "A3", "A10", "A12")

    def __init__(self, grid: RectilinearGrid, eps_cell: np.ndarray,
                 sigma_cell: np.ndarray, config: SolverConfig | None = None,
                 operators: Operators | None = None, *,
                 eps_mode: str = "parallel", sigma_mode: str = "parallel"):
        super().__init__(grid, config, operators)
        eps_cell = np.asarray(eps_cell, dtype=float)
        sigma_cell = np.asarray(sigma_cell, dtype=float)
        for arr, nm in ((eps_cell, "eps_cell"), (sigma_cell, "sigma_cell")):
            if arr.shape != grid.shape_cells:
                raise ValueError(
                    f"{nm} must have shape {grid.shape_cells}, got {arr.shape}")
        if np.any(eps_cell <= 0.0):
            raise ValueError(
                "eps_cell must be positive and ABSOLUTE [F/m]; multiply a "
                "relative permittivity by fieldspice.units.eps0")
        # Absolute permittivity is ~1e-11 F/m. Anything above 1e-6 F/m would be
        # a relative permittivity of 1e5, which no material has, so it is far
        # more likely the caller passed eps_r by mistake. Catching it here turns
        # a silently wrong answer (every capacitance off by 1.1e11) into an
        # error at construction.
        if float(np.min(eps_cell)) > 1e-6:
            raise ValueError(
                f"eps_cell looks like a RELATIVE permittivity (min "
                f"{float(np.min(eps_cell)):.4g}); it must be absolute [F/m]. "
                f"Multiply by fieldspice.units.eps0, or use "
                f"MaterialMap.eps() which already returns absolute values.")
        if np.any(sigma_cell < 0.0):
            raise ValueError("sigma_cell must be non-negative [S/m]")

        self.eps_edge = cell_to_edge(grid, eps_cell, mode=eps_mode)
        self.sigma_edge = cell_to_edge(grid, sigma_cell, mode=sigma_mode)
        G = self.ops.G
        self.L_eps = nodal_laplacian(grid, self.eps_edge, G=G)      # [F]
        self.L_sigma = nodal_laplacian(grid, self.sigma_edge, G=G)  # [S]
        self._cache: dict[tuple[float, float], LinearSystem] = {}

    # ------------------------------------------------------------------
    # Boundary / terminal bookkeeping
    # ------------------------------------------------------------------
    def _dirichlet(self, terminals: Sequence[Terminal],
                   bc, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Collect fixed node indices and values from terminals + boundary."""
        idx: list[np.ndarray] = []
        val: list[np.ndarray] = []
        for term in terminals:
            if term.driven == "voltage":
                v = term.value_at(t)
                idx.append(term.nodes)
                val.append(np.full(term.nodes.size, float(v)))
        if bc is not None:
            bidx, bval = bc.dirichlet_nodes(self.grid, t)
            if np.size(bidx):
                idx.append(np.asarray(bidx, dtype=np.intp))
                val.append(np.asarray(bval, dtype=float))
        if not idx:
            return np.zeros(0, dtype=np.intp), np.zeros(0)
        fixed = np.concatenate(idx)
        values = np.concatenate(val)
        # Later entries win on a clash, matching boundaries.py precedence.
        uniq, last = np.unique(fixed[::-1], return_index=True)
        return uniq, values[::-1][last]

    def _inject(self, terminals: Sequence[Terminal], t: float) -> np.ndarray:
        """Current injected into each node by current-driven terminals [A]."""
        rhs = np.zeros(self.grid.n_nodes)
        for term in terminals:
            if term.driven == "current":
                i_tot = float(term.value_at(t))
                # Spread the terminal current over its nodes by dual area so
                # that refining the electrode mesh does not change the answer.
                w = np.ones(term.nodes.size)
                rhs[term.nodes] += i_tot * w / w.sum()
        return rhs

    def _system(self, dt: float, theta: float) -> LinearSystem:
        key = (float(dt), float(theta))
        if key not in self._cache:
            K = (self.L_eps / dt + theta * self.L_sigma).tocsr()
            self._cache[key] = None  # placeholder; filled after BC in solve()
            self._K = K
        return self._cache[key]

    # ------------------------------------------------------------------
    # Initial condition
    # ------------------------------------------------------------------
    def consistent_initial_condition(self, terminals: Sequence[Terminal],
                                     bc=None, t: float = 0.0) -> np.ndarray:
        """Electrostatic potential at ``t``: the correct start for a transient.

        At ``t = 0+`` no charge has moved, so the potential distribution is the
        one set by permittivity alone --- a *capacitive divider*, not zero.  A
        lossy two-layer stack driven to 1 V starts at ``eps1/(eps1+eps2)``,
        which for the case in ``docs/CONTRACTS.md`` is exactly 2/3.

        Starting a transient from zero is a physically wrong initial condition.
        It is also the most common way to conclude that a working EQS solver is
        broken, so this is the default and you have to opt out of it.
        """
        fixed, vals = self._dirichlet(terminals, bc, t)
        if fixed.size == 0:
            return np.zeros(self.grid.n_nodes)
        from ..operators import apply_dirichlet
        A, b = apply_dirichlet(self.L_eps, np.zeros(self.grid.n_nodes),
                               fixed, vals)
        return LinearSystem(A, self.cfg).solve(b)

    # ------------------------------------------------------------------
    # Steady state
    # ------------------------------------------------------------------
    def steady_state(self, terminals: Sequence[Terminal], bc=None,
                     t: float = 0.0) -> Result:
        """DC solve: the pure resistive problem, with ``eps`` dropped entirely.

        Nodes inside a perfectly insulating island (``sigma == 0``) are
        genuinely undetermined at DC --- no current can reach them, so their
        potential carries no information.  :class:`LinearSystem` detects those
        decoupled rows and pins them, which is the only well-posed choice; the
        pinned count is reported in ``Result.meta['linear']``.
        """
        self._start()
        from ..operators import apply_dirichlet
        fixed, vals = self._dirichlet(terminals, bc, t)
        rhs = self._inject(terminals, t)
        A, b = apply_dirichlet(self.L_sigma, rhs, fixed, vals)
        info: dict[str, Any] = {}
        ls = LinearSystem(A, self.cfg)
        phi = ls.solve(b)
        info = dict(getattr(ls, "info", {}) or {})

        res = Result(grid=self.grid, t=np.zeros(1),
                     fields={"phi": phi[None, :]})
        self._record_terminals(res, terminals, [phi], [t], dt=None)
        res.scalars["dissipation"] = np.array([self.dissipation(phi)])
        return self._finish(res, mode="steady_state", linear=info)

    # ------------------------------------------------------------------
    # Transient
    # ------------------------------------------------------------------
    def solve(self, terminals: Sequence[Terminal], t_end: float, dt: float,
              bc=None, theta: float = 1.0, monitors=None,
              phi0: np.ndarray | None = None, store: bool = False,
              store_every: int = 1, t_start: float = 0.0) -> Result:
        """Step the electroquasistatic system from ``t_start`` to ``t_end``.

        Parameters
        ----------
        terminals
            Electrodes.  Voltage-driven ones become Dirichlet constraints;
            current-driven ones inject into the right-hand side.
        t_end, dt
            End time and step [s].  ``dt`` is limited by *accuracy*, not
            stability: the scheme is unconditionally stable.  20-100 steps per
            signal rise time is the usual range.
        theta
            1.0 = backward Euler (default, L-stable). 0.5 = trapezoidal, which
            is second order but **rings on a step edge** --- the classic SPICE
            trapezoidal artifact.  Circuit inputs have discontinuous
            derivatives, so backward Euler is the safe default.
        phi0
            Initial potential.  Defaults to
            :meth:`consistent_initial_condition`, which is almost always what
            you want.

        Returns
        -------
        Result
            ``terminals`` always populated; ``fields['phi']`` present when
            ``store=True`` (shape ``(n_stored, n_nodes)``).
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if t_end <= t_start:
            raise ValueError("t_end must exceed t_start")
        if not 0.0 <= theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")
        self._start()
        from ..operators import apply_dirichlet

        n = self.grid.n_nodes
        phi = (self.consistent_initial_condition(terminals, bc, t_start)
               if phi0 is None else np.asarray(phi0, dtype=float).copy())
        if phi.shape != (n,):
            raise ValueError(f"phi0 must have shape ({n},)")

        times = self._time_points(t_end, dt, t_start)
        # Constant dt: build and factorise the system matrix exactly once.
        # This is the difference between a usable solver and an unusable one;
        # a refactorisation per step costs 10-100x more than the back-solve.
        K = (self.L_eps / dt + theta * self.L_sigma).tocsr()
        Kexp = (self.L_eps / dt - (1.0 - theta) * self.L_sigma).tocsr()
        prepared: dict[bytes, LinearSystem] = {}

        phis = [phi.copy()]
        stored = [phi.copy()] if store else []
        stored_t = [times[0]] if store else []

        mset = monitors
        if mset is not None:
            self._record_monitors(mset, phi, times[0], dt)

        for k, t in enumerate(times[1:], start=1):
            fixed, vals = self._dirichlet(terminals, bc, t)
            rhs = Kexp @ phi + self._inject(terminals, t)
            A, b = apply_dirichlet(K, rhs, fixed, vals)
            # The Dirichlet pattern only changes if the *set* of fixed nodes
            # changes, which for a fixed electrode layout never happens; keying
            # the cache on that pattern keeps one factorisation for the run.
            key = fixed.tobytes()
            if key not in prepared:
                prepared[key] = LinearSystem(A, self.cfg)
            phi = prepared[key].solve(b)
            phis.append(phi.copy())
            if store and (k % max(1, store_every) == 0):
                stored.append(phi.copy())
                stored_t.append(t)
            if mset is not None:
                self._record_monitors(mset, phi, t, dt)

        res = Result(grid=self.grid, t=times)
        if store:
            res.fields["phi"] = np.array(stored)
            res.scalars["t_stored"] = np.array(stored_t)
        self._record_terminals(res, terminals, phis, times, dt=dt)
        res.scalars["energy_electric"] = np.array(
            [self.stored_energy(p) for p in phis])
        res.scalars["dissipation"] = np.array(
            [self.dissipation(p) for p in phis])
        if mset is not None:
            res.meta["monitors"] = mset.finalize()
        return self._finish(
            res, mode="transient", theta=theta, dt=dt,
            n_factorisations=len(prepared),
            speedup_vs_courant=self.speedup_vs_courant(dt))

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    def terminal_current(self, phi_new: np.ndarray, phi_old: np.ndarray,
                         dt: float, nodes: np.ndarray,
                         theta: float = 1.0) -> float:
        """Current flowing *into* the mesh from an electrode [A].

        Evaluated with the **pre-boundary-condition** operators, because the
        constrained rows of the modified matrix no longer carry the physical
        balance; the unconstrained residual on those rows is exactly the
        injected current.
        """
        resid = (self.L_eps @ (phi_new - phi_old) / dt
                 + self.L_sigma @ (theta * phi_new + (1 - theta) * phi_old))
        return float(resid[nodes].sum())

    def stored_energy(self, phi: np.ndarray) -> float:
        """Electric energy ``0.5 phi^T L_eps phi`` [J]."""
        return 0.5 * float(phi @ (self.L_eps @ phi))

    def dissipation(self, phi: np.ndarray) -> float:
        """Ohmic power ``phi^T L_sigma phi`` [W]."""
        return float(phi @ (self.L_sigma @ phi))

    def capacitance_matrix(self, terminals: Sequence[Terminal], bc=None):
        """Maxwell capacitance matrix [F] of the same structure."""
        from .poisson import PoissonSolver
        p = PoissonSolver.__new__(PoissonSolver)
        TimeSteppingSolver.__init__(p, self.grid, self.cfg, self.ops)
        p.eps_edge = self.eps_edge
        p.L = self.L_eps
        return PoissonSolver.capacitance_matrix(p, terminals, bc)

    # ------------------------------------------------------------------
    def _record_terminals(self, res: Result, terminals: Sequence[Terminal],
                          phis: Sequence[np.ndarray], times, dt) -> None:
        for term in terminals:
            v = np.array([float(np.mean(p[term.nodes])) for p in phis])
            if dt is None:
                i = np.array([float((self.L_sigma @ phis[0])[term.nodes].sum())])
            else:
                i = np.zeros(len(phis))
                for k in range(1, len(phis)):
                    i[k] = self.terminal_current(phis[k], phis[k - 1], dt,
                                                 term.nodes)
                i[0] = i[1] if len(i) > 1 else 0.0
            res.terminals[term.name] = {"v": v, "i": i}

    def _record_monitors(self, mset, phi: np.ndarray, t: float, dt: float):
        state = {
            "phi": phi, "grid": self.grid, "ops": self.ops,
            "eps_edge": self.eps_edge, "sigma_edge": self.sigma_edge,
            "step": 0, "dt": dt,
        }
        try:
            mset.record(state, t)
        except Exception as exc:  # a monitor must never kill a solve
            self._log(1, f"monitor error at t={t:.4g}: {exc}")
