"""Solver base classes and result containers --- the contract every solver obeys.

A fieldspice *problem* is always the same five things:

1. a :class:`~fieldspice.grid.RectilinearGrid`,
2. per-cell material property arrays (from :mod:`fieldspice.materials` and
   :mod:`fieldspice.geometry`),
3. boundary conditions on the six domain walls,
4. terminals (electrodes) that carry either a driven voltage or a driven
   current, and optionally connect to a lumped netlist,
5. monitors that record what you care about.

Every solver consumes that and produces a :class:`Result`.  Keeping the
container uniform is what lets the same post-processing, plotting and
extraction code serve the electrostatic, quasi-static, drift-diffusion and
full-wave paths.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ..grid import RectilinearGrid
from ..operators import Operators

__all__ = ["Result", "SolverBase", "TimeSteppingSolver", "ConvergenceError",
           "SolverConfig", "Terminal"]


class ConvergenceError(RuntimeError):
    """Raised when a nonlinear or iterative solve fails to converge.

    Carries the residual history so that a failure is diagnosable rather than
    merely fatal --- the single most common support question for any nonlinear
    device simulator is "why did Newton stop", and the answer is always in the
    residual trace.
    """

    def __init__(self, msg: str, history: Sequence[float] = (),
                 last_state: np.ndarray | None = None):
        super().__init__(msg)
        self.history = list(history)
        self.last_state = last_state


@dataclass
class SolverConfig:
    """Knobs shared by all solvers.

    Attributes
    ----------
    linear_solver
        ``"auto"``, ``"direct"`` (SuperLU/CHOLMOD), ``"cg"``, ``"amg"``.
        ``"auto"`` picks direct below ~2e5 unknowns and AMG-preconditioned CG
        above, which is where the crossover sits on a typical workstation.
    tol
        Relative residual tolerance for the linear solve.
    newton_tol
        Absolute update tolerance for nonlinear solves.  For drift-diffusion
        this is measured in units of the thermal voltage on the potential and
        as a relative change on the carrier densities.
    max_newton
        Newton iteration cap before :class:`ConvergenceError`.
    damping
        Initial Newton damping factor (1.0 = full step).  Drift-diffusion uses
        Bank-Rose damping on top of this.
    verbose
        0 silent, 1 progress, 2 per-iteration residuals.
    """
    linear_solver: str = "auto"
    tol: float = 1e-10
    newton_tol: float = 1e-8
    max_newton: int = 60
    damping: float = 1.0
    verbose: int = 0
    store_every: int = 1
    dtype: type = np.float64


@dataclass
class Terminal:
    """An electrode: a set of nodes held at a common potential.

    A terminal is the *only* place where a field region touches the outside
    world.  It is either voltage-driven (``voltage`` set) or current-driven
    (``current`` set), and in either case the solver reports the conjugate
    quantity, so terminals are also the natural probe points.

    Parameters
    ----------
    name
        Identifier used in results and netlists.
    nodes
        Flat node indices belonging to the electrode.
    voltage, current
        Either a constant or a callable ``f(t) -> float``.  Exactly one of the
        two should be given; if both are ``None`` the terminal floats (it is
        an equipotential island with zero net injected current, which is the
        correct model for an unconnected metal plate).
    circuit_node
        Optional name of a node in an attached lumped netlist.  When set, the
        terminal's voltage becomes an unknown solved jointly with the circuit.
    """
    name: str
    nodes: np.ndarray
    voltage: float | Callable[[float], float] | None = None
    current: float | Callable[[float], float] | None = None
    circuit_node: str | None = None

    def __post_init__(self):
        self.nodes = np.unique(np.asarray(self.nodes, dtype=np.intp).ravel())
        if self.voltage is not None and self.current is not None:
            raise ValueError(
                f"terminal {self.name!r}: set voltage or current, not both")

    def value_at(self, t: float) -> float | None:
        src = self.voltage if self.voltage is not None else self.current
        if src is None:
            return None
        return float(src(t)) if callable(src) else float(src)

    @property
    def driven(self) -> str:
        if self.voltage is not None:
            return "voltage"
        if self.current is not None:
            return "current"
        return "float"


@dataclass
class Result:
    """Uniform output container.

    Attributes
    ----------
    t
        Time samples [s], shape ``(nt,)``.  Empty for steady-state solves.
    fields
        Named arrays of stored field history.  A stored scalar node field has
        shape ``(nt,) + grid.shape_nodes``; an edge field ``(nt, n_edges)``.
        Which fields are stored is solver- and request-dependent, because
        storing every step of a 3D field is usually the memory bottleneck.
    terminals
        ``{name: {"v": (nt,), "i": (nt,)}}`` --- voltage and current at every
        electrode, always recorded because they are cheap and are what circuit
        people actually want.
    scalars
        Named time series of derived quantities (stored energy, dissipated
        power, total charge).
    meta
        Free-form provenance: solver name, config, wall time, assumptions
        invoked, convergence statistics.  ``meta["assumptions"]`` is a list of
        the ``docs/ASSUMPTIONS.md`` tags that were active for this run, so a
        result can always be traced back to the physics that produced it.
    grid
        The grid the result lives on.
    """
    grid: RectilinearGrid
    t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    fields: dict[str, np.ndarray] = field(default_factory=dict)
    terminals: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    scalars: dict[str, np.ndarray] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    # -- convenience accessors --------------------------------------------
    def v(self, terminal: str) -> np.ndarray:
        return self.terminals[terminal]["v"]

    def i(self, terminal: str) -> np.ndarray:
        return self.terminals[terminal]["i"]

    def field(self, name: str, step: int = -1) -> np.ndarray:
        return self.fields[name][step]

    def final(self, name: str) -> np.ndarray:
        return self.fields[name][-1]

    def summary(self) -> str:
        lines = [f"Result from {self.meta.get('solver', '?')}"]
        if self.t.size:
            lines.append(f"  {self.t.size} time samples, "
                         f"t = {self.t[0]:.4g} .. {self.t[-1]:.4g} s")
        if self.terminals:
            lines.append("  terminals: " + ", ".join(self.terminals))
        if self.fields:
            lines.append("  fields: " + ", ".join(
                f"{k}{v.shape}" for k, v in self.fields.items()))
        if "wall_time" in self.meta:
            lines.append(f"  wall time {self.meta['wall_time']:.3f} s")
        if "assumptions" in self.meta:
            lines.append("  assumptions: " + ", ".join(self.meta["assumptions"]))
        return "\n".join(lines)

    def save(self, path: str) -> None:
        from ..io import save_result
        save_result(self, path)


class SolverBase(ABC):
    """Abstract solver.

    Subclasses must set :attr:`name` and :attr:`assumptions` (a list of tags
    from ``docs/ASSUMPTIONS.md``) and implement :meth:`solve`.
    """

    name: str = "base"
    assumptions: tuple[str, ...] = ()

    def __init__(self, grid: RectilinearGrid,
                 config: SolverConfig | None = None,
                 operators: Operators | None = None):
        self.grid = grid
        self.cfg = config or SolverConfig()
        self.ops = operators or Operators(grid)
        self._t0 = 0.0

    @abstractmethod
    def solve(self, *args, **kwargs) -> Result:
        """Run the solve and return a :class:`Result`."""

    # -- helpers for subclasses -------------------------------------------
    def _start(self) -> None:
        self._t0 = time.perf_counter()

    def _finish(self, res: Result, **extra) -> Result:
        res.meta.setdefault("solver", self.name)
        res.meta.setdefault("assumptions", list(self.assumptions))
        res.meta.setdefault("wall_time", time.perf_counter() - self._t0)
        res.meta.setdefault("grid_cells", self.grid.ncell)
        res.meta.update(extra)
        return res

    def _log(self, level: int, msg: str) -> None:
        if self.cfg.verbose >= level:
            print(f"[{self.name}] {msg}", flush=True)


class TimeSteppingSolver(SolverBase):
    """Base for transient solvers, with a shared adaptive-stepping loop.

    The stability story differs sharply between the two families and is the
    core computational claim of this project:

    * **Quasi-static solvers are unconditionally stable** (backward Euler /
      trapezoidal on an elliptic-in-space system).  Their step size is set by
      how fast the *signal* changes, not by how fast light crosses a cell.
    * **The full-wave solver is explicit and Courant-limited**, so its step is
      set by the smallest cell divided by c.

    On a 10 nm-resolved on-chip structure the Courant step is ~2e-17 s while
    the physics of interest evolves over ~1e-10 s, a ratio of 5e6 steps.  That
    ratio *is* the speedup the quasi-static approximation buys, and
    :meth:`speedup_vs_courant` reports it for any completed run.
    """

    def speedup_vs_courant(self, dt_used: float,
                           eps_r_min: float = 1.0) -> float:
        """How many explicit-FDTD steps one quasi-static step replaces."""
        return dt_used / self.grid.courant_dt(eps_r_min=eps_r_min)

    @staticmethod
    def _time_points(t_end: float, dt: float,
                     t_start: float = 0.0) -> np.ndarray:
        n = int(np.ceil((t_end - t_start) / dt))
        return t_start + dt * np.arange(n + 1)
