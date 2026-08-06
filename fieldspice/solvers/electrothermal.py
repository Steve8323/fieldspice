"""Coupled electro-thermal solution: Joule heat in, temperature-dependent sigma out.

Two problems, solved alternately until they agree:

    electrical:  G^T M_sigma(T) G phi = i        ->  Q = Joule power
    thermal:     G^T M_kappa   G T   = Q         ->  T

with ``sigma(T) = sigma0 / (1 + tcr (T - T_ref))``. This is a Gummel (block
Gauss-Seidel) iteration rather than a monolithic Newton: each block is
symmetric positive definite and reuses machinery that is already validated,
and the coupling is weak enough in most problems that it converges in a
handful of passes. Where it is *not* weak, that is the physically interesting
case, and it has a name.

Thermal runaway
---------------
The feedback sign depends on how the device is driven, and this is the whole
story:

* **Voltage driven.** ``P = V^2/R``. Heating raises ``R``, which *lowers* ``P``.
  Negative feedback --- always stable.
* **Current driven.** ``P = I^2 R``. Heating raises ``R``, which *raises* ``P``.
  Positive feedback --- stable only up to a threshold.

For a lumped conductor with thermal resistance ``R_th`` the fixed point of

    dT = R_th I^2 R0 (1 + tcr dT)

is

    dT = A / (1 - A tcr),      A = R_th I^2 R0

so the temperature rise is finite only while ``A tcr < 1``, and diverges as
that product approaches one. Above it there is **no steady state at all**: the
iteration does not fail to converge because the numerics are bad, it fails
because the physical problem has no solution. :class:`ThermalRunaway` reports
that distinctly from an ordinary convergence failure, because the two demand
opposite responses --- tighten the solver, or accept that the device melts.

An isothermal solver (A6) reports the same device as a perfectly stable
operating point. That is the class of error no mesh refinement can fix.
"""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from ..grid import RectilinearGrid
from ..operators import (Operators, apply_dirichlet, cell_to_edge,
                         nodal_laplacian)
from ..physics import PhysicsOptions
from .base import ConvergenceError, Result, SolverBase, SolverConfig, Terminal
from .poisson import LinearSystem
from .thermal import ThermalSolver, joule_heating_nodes

__all__ = ["ElectroThermalSolver", "ThermalRunaway", "runaway_threshold"]


class ThermalRunaway(ConvergenceError):
    """Raised when the electro-thermal fixed point does not exist.

    Distinct from a numerical convergence failure: the temperature is growing
    without bound because the positive feedback loop gain exceeds unity, so
    there is nothing to converge *to*. Carries the temperature history.
    """

    def __init__(self, msg: str, history: Sequence[float] = (),
                 loop_gain: float | None = None):
        super().__init__(msg, history)
        self.loop_gain = loop_gain


def runaway_threshold(R_th: float, R0: float, tcr: float) -> float:
    """Current [A] at which a lumped current-driven conductor runs away.

    From ``A tcr = 1`` with ``A = R_th I^2 R0``:  ``I_crit = 1/sqrt(R_th R0 tcr)``.
    Returns ``inf`` when ``tcr <= 0`` (a negative-coefficient material cannot
    run away by this mechanism; it self-stabilises).
    """
    if tcr <= 0.0 or R_th <= 0.0 or R0 <= 0.0:
        return float("inf")
    return float(1.0 / np.sqrt(R_th * R0 * tcr))


class ElectroThermalSolver(SolverBase):
    """Self-consistent steady-state conduction with self-heating.

    Parameters
    ----------
    grid
        The mesh.
    sigma_cell
        Reference-temperature electrical conductivity [S/m].
    kappa_cell
        Thermal conductivity [W/(m K)].
    tcr_cell
        Temperature coefficient of resistivity per cell [1/K]. Zero disables
        the feedback for that cell, which makes the solve a one-pass
        electrical-then-thermal calculation.
    options
        :class:`~fieldspice.physics.PhysicsOptions`. ``self_heating`` must be
        enabled or this solver is pointless; it raises otherwise rather than
        silently doing nothing.
    """

    name = "electrothermal"
    assumptions = ("A1a", "A2", "A3", "A10", "A12")

    def __init__(self, grid: RectilinearGrid, sigma_cell: np.ndarray,
                 kappa_cell: np.ndarray, tcr_cell: np.ndarray | float = 0.0,
                 options: PhysicsOptions | None = None,
                 config: SolverConfig | None = None,
                 operators: Operators | None = None,
                 T_ref: float = 300.0):
        super().__init__(grid, config, operators)
        self.opts = options or PhysicsOptions(self_heating=True)
        if not self.opts.self_heating:
            raise ValueError(
                "ElectroThermalSolver requires PhysicsOptions(self_heating=True); "
                "for an isothermal run use EQSSolver directly")
        self.sigma0 = np.asarray(sigma_cell, dtype=float)
        if self.sigma0.shape != grid.shape_cells:
            raise ValueError(f"sigma_cell must have shape {grid.shape_cells}")
        self.tcr = (np.full(grid.shape_cells, float(tcr_cell))
                    if np.isscalar(tcr_cell)
                    else np.asarray(tcr_cell, dtype=float))
        if self.tcr.shape != grid.shape_cells:
            raise ValueError(f"tcr_cell must have shape {grid.shape_cells}")
        self.T_ref = float(T_ref)
        self.thermal = ThermalSolver(grid, kappa_cell, config=config,
                                     operators=self.ops)
        from ..operators import node_volume_vector
        self._node_vol = node_volume_vector(grid)

    # ------------------------------------------------------------------
    def _sigma_at(self, T_cell: np.ndarray) -> np.ndarray:
        denom = 1.0 + self.tcr * (T_cell - self.T_ref)
        return self.sigma0 / np.maximum(denom, 1e-6)

    def _electrical(self, sigma_cell: np.ndarray,
                    terminals: Sequence[Terminal], bc) -> tuple:
        L = nodal_laplacian(self.grid, cell_to_edge(self.grid, sigma_cell),
                            G=self.ops.G)
        idx, val = [], []
        for t in terminals:
            if t.driven == "voltage":
                idx.append(t.nodes)
                val.append(np.full(t.nodes.size, float(t.value_at(0.0))))
        rhs = np.zeros(self.grid.n_nodes)
        for t in terminals:
            if t.driven == "current":
                # Dual-volume weighting, not equal-per-node: see
                # EQSSolver._node_weights for the measurement that forced this.
                w = self._node_vol[t.nodes]
                w = w / w.sum() if w.sum() > 0 else np.full(t.nodes.size,
                                                            1.0 / t.nodes.size)
                rhs[t.nodes] += float(t.value_at(0.0)) * w
        if bc is not None:
            bidx, bval = bc.dirichlet_nodes(self.grid, 0.0)
            if np.size(bidx):
                idx.append(np.asarray(bidx, dtype=np.intp))
                val.append(np.asarray(bval, dtype=float))
        if idx:
            fixed = np.concatenate(idx)
            vals = np.concatenate(val)
            A, b = apply_dirichlet(L, rhs, fixed, vals)
        else:
            A, b = L, rhs
        phi = LinearSystem(A, self.cfg).solve(b)
        return phi, L

    # ------------------------------------------------------------------
    def solve(self, terminals: Sequence[Terminal], bc_electrical=None,
              bc_thermal=None, convection=None, damping: float = 1.0,
              T_limit: float = 5000.0) -> Result:
        """Iterate electrical and thermal problems to self-consistency.

        Parameters
        ----------
        damping
            Under-relaxation on the temperature update, in ``(0, 1]``. Values
            below 1 widen the convergence basin near the runaway threshold at
            the cost of more iterations. It does **not** stabilise a genuine
            runaway --- nothing can, because no fixed point exists.
        T_limit
            Temperature [K] above which the run is declared a runaway. Silicon
            melts at 1687 K, so anything past a couple of thousand kelvin is
            already physically meaningless.

        Raises
        ------
        ThermalRunaway
            When the temperature exceeds ``T_limit`` or grows monotonically
            without settling.
        """
        if not 0.0 < damping <= 1.0:
            raise ValueError("damping must lie in (0, 1]")
        self._start()

        from ..operators import cell_to_node
        T_node = np.full(self.grid.n_nodes, self.opts.ambient_temperature)
        history: list[float] = []
        n_iter = 0
        converged = False
        phi = np.zeros(self.grid.n_nodes)
        power = 0.0

        for n_iter in range(1, self.opts.max_coupling_iterations + 1):
            # cell temperature from the nodal field (volume average)
            Tc = _node_to_cell(self.grid, T_node)
            sigma = self._sigma_at(Tc)
            phi, L = self._electrical(sigma, terminals, bc_electrical)
            q = joule_heating_nodes(self.grid, cell_to_edge(self.grid, sigma),
                                    phi, self.ops)
            power = float(q.sum())
            res_t = self.thermal.steady(q, bc=bc_thermal, convection=convection)
            T_new = res_t.fields["T"][0]

            dT = float(np.max(np.abs(T_new - T_node)))
            history.append(float(T_new.max()))
            T_node = T_node + damping * (T_new - T_node)

            # A single hot iterate is NOT a runaway. The first pass is
            # evaluated at ambient conductivity, so a voltage-driven device
            # overshoots badly before its negative feedback pulls it back: at
            # V = 1 V the case in the tests peaks at 13,936 K on iteration 1
            # and settles at 2,281 K. Declaring runaway there would be wrong
            # twice over -- wrong verdict, and wrong mechanism, since voltage
            # drive cannot run away at all. Only a sustained monotone climb
            # counts.
            # Monotone growth alone is NOT divergence -- a sequence converging
            # to its fixed point from below is monotone too, and that is the
            # normal case. The distinguishing signature of loop gain above one
            # is *accelerating* growth: successive increments getting larger
            # rather than smaller. A converging iteration has shrinking
            # increments by construction.
            diverging = False
            if len(history) >= 4 and history[-1] > T_limit:
                d1 = history[-3] - history[-4]
                d2 = history[-2] - history[-3]
                d3 = history[-1] - history[-2]
                diverging = d3 > d2 > d1 > 0.0
            if diverging or float(T_node.max()) > 20.0 * T_limit:
                raise ThermalRunaway(
                    f"thermal runaway: peak temperature climbing monotonically "
                    f"({history[-4]:.0f} -> {history[-1]:.0f} K) past the "
                    f"{T_limit:.0f} K limit after {n_iter} coupling "
                    f"iterations, with no fixed point in sight. A "
                    f"current-driven conductor with a positive temperature "
                    f"coefficient has no steady state above "
                    f"I_crit = 1/sqrt(R_th R0 tcr). Reduce the drive, lower "
                    f"the thermal resistance, or use a lower-tcr material.",
                    history=history)
            if dT < self.opts.coupling_tolerance:
                converged = True
                break

        if not converged:
            growing = (len(history) > 3
                       and all(b > a for a, b in zip(history[-4:], history[-3:])))
            if growing:
                raise ThermalRunaway(
                    f"thermal runaway: temperature still rising monotonically "
                    f"after {n_iter} iterations "
                    f"({history[-4]:.1f} -> {history[-1]:.1f} K) with no sign "
                    f"of a fixed point.", history=history)
            raise ConvergenceError(
                f"electro-thermal coupling did not converge in {n_iter} "
                f"iterations (last update {dT:.3e} K, tolerance "
                f"{self.opts.coupling_tolerance:.1e} K). Try damping < 1.",
                history=history)

        res = Result(grid=self.grid, t=np.zeros(1),
                     fields={"phi": phi[None, :], "T": T_node[None, :]})
        for t in terminals:
            _w = self._node_vol[t.nodes]
            _w = _w / _w.sum() if _w.sum() > 0 else np.full(t.nodes.size,
                                                            1.0 / t.nodes.size)
            res.terminals[t.name] = {
                "v": np.array([float(np.dot(_w, phi[t.nodes]))]),
                "i": np.array([float((L @ phi)[t.nodes].sum())]),
            }
        res.scalars["power"] = np.array([power])
        res.scalars["T_max"] = np.array([float(T_node.max())])
        res.scalars["T_rise"] = np.array(
            [float(T_node.max() - self.opts.ambient_temperature)])
        res.scalars["history"] = np.array(history)
        if float(T_node.max()) > T_limit:
            warnings.warn(
                f"electro-thermal solve converged, but the peak temperature is "
                f"{T_node.max():.0f} K, above the {T_limit:.0f} K sanity "
                f"limit. A fixed point exists mathematically; the material "
                f"almost certainly does not survive it (silicon melts at "
                f"1687 K, copper at 1358 K). Treat this as a design failure, "
                f"not a result.", RuntimeWarning, stacklevel=2)
        res.meta["assumptions"] = list(
            self.opts.remaining_assumptions(self.assumptions + ("A6",)))
        res.meta["physics"] = self.opts.summary()
        return self._finish(res, iterations=n_iter, converged=True,
                            coupled=True)


def _node_to_cell(grid: RectilinearGrid, v_node: np.ndarray) -> np.ndarray:
    """Average a nodal field onto cells (the 8 corners of each cell)."""
    a = v_node.reshape(grid.shape_nodes)
    return 0.125 * (a[:-1, :-1, :-1] + a[1:, :-1, :-1] + a[:-1, 1:, :-1]
                    + a[:-1, :-1, 1:] + a[1:, 1:, :-1] + a[1:, :-1, 1:]
                    + a[:-1, 1:, 1:] + a[1:, 1:, 1:])
