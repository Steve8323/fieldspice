"""Monitors --- the recording layer between a running solver and a ``Result``.

A monitor is a small stateful object that a solver calls once per time step
with the current solution.  It extracts one scalar, one time series or one
strided sequence of field snapshots, and at the end hands back plain NumPy
arrays with a leading time axis.

Why this is a separate layer at all: a field solve produces far more data than
anyone wants to keep.  A 200x200x50 grid stores 2e6 node potentials per step,
which is 16 MB; ten thousand time steps of that is 160 GB.  Deciding *at
declaration time* what will be kept --- a handful of scalars, terminal
currents, and every hundredth frame --- is the difference between a run that
fits in memory and one that does not.  Monitors make that decision explicit and
auditable instead of implicit in a solver flag.

The state contract
==================
Each step the solver calls ``monitor.record(state, t)`` with ``t`` the
simulation time [s] and ``state`` a mapping.  The **required** vocabulary is
deliberately small; every key is optional from the mapping's point of view, and
a monitor that needs a key it cannot find raises :class:`MonitorStateError`
(a ``ValueError``) on the *first* record with a message naming the key, the
monitor and the keys that were actually supplied.  It never raises a bare
``KeyError`` once per step.

=================  =========================  ================================
Key                Shape                      Meaning / units
=================  =========================  ================================
``"grid"``         ---                        :class:`~fieldspice.grid.RectilinearGrid`
``"ops"``          ---                        :class:`~fieldspice.operators.Operators`
``"phi"``          ``(n_nodes,)``             node electric potential [V]
``"e"``            ``(n_edges,)``             edge circulation of E,
                                              ``int E.dl`` [V]
``"b"``            ``(n_faces,)``             face magnetic flux,
                                              ``int B.dA`` [Wb]
``"a"``            ``(n_edges,)``             edge circulation of A [Wb]
``"n"``, ``"p"``   ``(n_nodes,)``             carrier densities [m^-3]
``"eps_edge"``     ``(n_edges,)``             permittivity on edges [F/m]
``"sigma_edge"``   ``(n_edges,)``             conductivity on edges [S/m]
``"mu_face"``      ``(n_faces,)``             permeability on faces [H/m]
``"terminals"``    ---                        sequence or ``{name: Terminal}``
``"step"``         ---                        integer step counter
=================  =========================  ================================

Optional keys, used when present and reconstructed by finite differences when
absent:

=====================  ============================================================
``"dphidt"``           ``(n_nodes,)`` dphi/dt [V/s]
``"dedt"``             ``(n_edges,)`` de/dt [V/s]
``"m_nu"``             ``(n_faces,)`` array or diagonal sparse matrix, the
                       face reluctance [1/H]; overrides ``mu_face``
``"terminal_voltage"`` ``{name: V}`` authoritative terminal voltages [V]
``"terminal_current"`` ``{name: I}`` authoritative terminal currents [A],
                       positive **into** the field region
``"doping"``           ``(n_nodes,)`` net donor density ``Nd - Na`` [m^-3]
=====================  ============================================================

``phi``, ``e``, ``b``, ``a``, ``n``, ``p`` may be passed either flat or in
their natural grid shape; monitors ravel them.  Solvers are free to reuse the
same buffer every step, so any monitor that *stores* an array copies it.

Sign conventions (inherited from :mod:`fieldspice.operators`, get these right)
=============================================================================
* ``(G phi)_e = phi_head - phi_tail`` is the potential **rise** along the edge.
* The field circulation is ``e = -G phi``.
* The current **along the edge direction** is ``i_e = -(M_sigma G phi)_e``,
  equivalently ``i_e = +(M_sigma e)_e``.
* ``G^T i`` is the net current flowing **into** each node through the mesh, so
  the current injected into a node from outside is
  ``I_inject = G^T M_sigma G phi`` --- note there is no minus sign on that
  form.  :class:`TerminalProbe` reports ``I_inject`` summed over the electrode,
  i.e. **positive current flows into the field region**.

Energy expressions
==================
All three are exact quadratic forms of the discrete state, not quadrature
approximations, because the mass matrices are diagonal circuit elements:

* stored electric energy ``W_e = 0.5 * e^T M_eps e`` [J].  With ``e = -G phi``
  this is identically ``0.5 * phi^T (G^T M_eps G) phi``, and
  ``G^T M_eps G`` is the nodal capacitance matrix, so for a two-terminal
  structure it reduces to ``0.5 C V^2`` exactly (verified below to 1e-15).
* stored magnetic energy ``W_m = 0.5 * b^T M_nu b`` [J], with ``M_nu`` the
  face **reluctance** ``L_dual / (mu A_face)`` [1/H].
* instantaneous dissipation ``P = e^T M_sigma e`` [W].  There is **no** factor
  of one half here: ``0.5`` appears in the two *stored* energies because they
  are integrals of ``x dx``, while Joule heating is ``V I = G V^2`` outright.
  Halving the dissipation, or forgetting to halve the energies, is the single
  most common bug in this kind of code.

.. warning::
   :func:`fieldspice.operators.face_mass_nu` currently returns
   ``A_face / (mu * L_dual)``, whose units are m^2/H, not the 1/H needed for
   ``M_nu b`` to be an mmf in amps.  The correct reluctance is the reciprocal
   geometry factor ``L_dual / (mu * A_face)``.  ``operators.py`` is frozen, so
   :class:`EnergyMonitor` builds the reluctance itself and does **not** call
   ``face_mass_nu``.  Pass ``state["m_nu"]`` to override with whatever the
   solver actually used, so the reported energy always matches the solved
   system even while the two disagree.

Output convention
=================
``Monitor.finalize()`` returns ``{key: array}`` where every array has a leading
time axis of length equal to the number of *accepted* records.  A monitor that
publishes a single quantity uses its own ``name`` as the key; one that
publishes several uses ``f"{name}.{quantity}"``.  A monitor whose time axis
differs from the solver's (only :class:`FieldSnapshot`, which strides) also
emits ``f"{name}.t"``.  :class:`MonitorSet` merges all of them and adds the
master ``"t"`` axis.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from .grid import RectilinearGrid
from .operators import grad_node_edge
from .solvers.base import Terminal
from .units import q as _q_electron

__all__ = [
    "MonitorStateError",
    "Monitor",
    "NodeProbe",
    "TerminalProbe",
    "FieldSnapshot",
    "EnergyMonitor",
    "FluxMonitor",
    "ChargeMonitor",
    "MonitorSet",
]

_AXIS_NAMES = {"x": 0, "y": 1, "z": 2, 0: 0, 1: 1, 2: 2}


# ==========================================================================
# Errors and small state helpers
# ==========================================================================
class MonitorStateError(ValueError):
    """The solver state lacks something a monitor needs.

    Subclasses :class:`ValueError` so that the eager-validation rule in
    ``docs/CONTRACTS.md`` is satisfied, while still being catchable
    specifically by a solver that wants to skip an unsupported monitor.
    """


def _present(state: Mapping[str, Any], key: str) -> bool:
    """True if ``key`` is in ``state`` with a value that is not ``None``."""
    return key in state and state[key] is not None


def _require(state: Mapping[str, Any], key: str, who: str,
             why: str = "") -> Any:
    """Fetch ``state[key]`` or raise a diagnosable :class:`MonitorStateError`."""
    if not _present(state, key):
        have = ", ".join(sorted(str(k) for k in state)) or "(nothing)"
        tail = f" ({why})" if why else ""
        raise MonitorStateError(
            f"{who} needs state[{key!r}]{tail}, but the solver did not supply "
            f"it. State contains: {have}. See the state contract at the top of "
            f"fieldspice/monitors.py."
        )
    return state[key]


def _flat(v: Any, n: int, who: str, key: str) -> np.ndarray:
    """Ravel a state entry and check its length."""
    arr = np.asarray(v)
    if arr.ndim != 1:
        arr = arr.ravel()
    if arr.size != n:
        raise ValueError(
            f"{who}: state[{key!r}] has {arr.size} entries, expected {n}")
    return arr


def _is_terminal(obj: Any) -> bool:
    """Duck-typed :class:`~fieldspice.solvers.base.Terminal` test.

    Avoids a hard ``isinstance`` so that a solver may pass any object exposing
    ``.name`` and ``.nodes`` (a mock, a subclass, a namedtuple in a test).
    """
    return hasattr(obj, "nodes") and hasattr(obj, "name")


def _find_terminal(state: Mapping[str, Any], name: str, who: str) -> Terminal:
    """Look ``name`` up in ``state['terminals']`` (dict or sequence)."""
    terms = _require(state, "terminals", who,
                     f"to resolve terminal {name!r} by name")
    if isinstance(terms, Mapping):
        if name not in terms:
            raise MonitorStateError(
                f"{who}: no terminal named {name!r}; solver supplied "
                f"{sorted(terms)}")
        return terms[name]
    for term in terms:
        if getattr(term, "name", None) == name:
            return term
    known = [getattr(x, "name", "?") for x in terms]
    raise MonitorStateError(
        f"{who}: no terminal named {name!r}; solver supplied {known}")


def _resolve_nodes(sel: Any, state: Mapping[str, Any], who: str,
                   n_nodes: int) -> np.ndarray:
    """Turn a node selector into a sorted array of flat node indices.

    Accepts a :class:`~fieldspice.solvers.base.Terminal`, a terminal name, an
    integer, a sequence of integers, or a boolean mask of length ``n_nodes``.
    """
    if _is_terminal(sel):
        idx = np.asarray(sel.nodes, dtype=np.intp).ravel()
    elif isinstance(sel, str):
        idx = np.asarray(_find_terminal(state, sel, who).nodes,
                         dtype=np.intp).ravel()
    else:
        arr = np.asarray(sel)
        if arr.dtype == bool:
            if arr.size != n_nodes:
                raise ValueError(
                    f"{who}: boolean node mask has {arr.size} entries, "
                    f"expected {n_nodes}")
            idx = np.flatnonzero(arr.ravel()).astype(np.intp)
        else:
            if not np.issubdtype(arr.dtype, np.integer):
                raise ValueError(
                    f"{who}: node selector must be integer indices, a boolean "
                    f"mask, a Terminal or a terminal name, got dtype "
                    f"{arr.dtype}")
            idx = arr.astype(np.intp).ravel()
    if idx.size == 0:
        raise ValueError(f"{who}: node selection is empty")
    if idx.min() < 0 or idx.max() >= n_nodes:
        raise ValueError(
            f"{who}: node index out of range [0, {n_nodes}), got "
            f"[{idx.min()}, {idx.max()}]")
    return np.unique(idx)


def _stack(vals: Sequence[Any]) -> np.ndarray:
    """Stack a list of per-step samples into one array with a leading time axis."""
    if len(vals) == 0:
        return np.zeros(0)
    if isinstance(vals[0], np.ndarray):
        return np.stack(vals)  # raises on a shape mismatch, which is correct
    return np.asarray(vals, dtype=float)


# ==========================================================================
# Per-instance geometry cache
# ==========================================================================
class _GridCache:
    """Grid-derived quantities cached on the monitor that owns them.

    Rebuilt whenever the state hands over a different grid object.  Kept
    per-instance rather than module-level because ``docs/CONTRACTS.md`` forbids
    module-level mutable state, and because a monitor is exactly the right
    lifetime for the cache: one solve, one grid.
    """

    __slots__ = ("_grid", "_G", "_edge_ratio", "_face_ratio", "_node_vol")

    def __init__(self) -> None:
        self._grid: RectilinearGrid | None = None
        self._G: sp.csr_matrix | None = None
        self._edge_ratio: np.ndarray | None = None
        self._face_ratio: np.ndarray | None = None
        self._node_vol: np.ndarray | None = None

    def grid(self, state: Mapping[str, Any], who: str) -> RectilinearGrid:
        g = _require(state, "grid", who, "monitors need the grid metric")
        if g is not self._grid:
            self._grid = g
            self._G = None
            self._edge_ratio = None
            self._face_ratio = None
            self._node_vol = None
        return g

    def G(self, state: Mapping[str, Any], who: str) -> sp.csr_matrix:
        """Discrete gradient, preferring the solver's cached ``Operators``."""
        grid = self.grid(state, who)
        ops = state.get("ops")
        if ops is not None and hasattr(ops, "G"):
            return ops.G
        if self._G is None:
            self._G = grad_node_edge(grid)
        return self._G

    def edge_ratio(self, state: Mapping[str, Any], who: str) -> np.ndarray:
        """``A_dual / L`` per edge [m].

        Multiplying by eps [F/m] gives the edge capacitance [F]; by sigma [S/m]
        the edge conductance [S].  This is exactly the diagonal that
        :func:`fieldspice.operators.edge_mass` builds, recomputed here as a
        vector so that no sparse matrix has to be reassembled per step.
        """
        grid = self.grid(state, who)
        if self._edge_ratio is None:
            L = np.concatenate([a.ravel() for a in grid.edge_lengths()])
            A = np.concatenate([a.ravel() for a in grid.edge_dual_areas()])
            self._edge_ratio = A / L
        return self._edge_ratio

    def face_ratio(self, state: Mapping[str, Any], who: str) -> np.ndarray:
        """``L_dual / A_face`` per face [1/m].

        Dividing by mu [H/m] gives the reluctance of the flux tube through the
        face [1/H].  See the module-level warning: this is *not* what the
        frozen :func:`fieldspice.operators.face_mass_nu` returns.
        """
        grid = self.grid(state, who)
        if self._face_ratio is None:
            A = np.concatenate([a.ravel() for a in grid.face_areas()])
            L = np.concatenate([a.ravel() for a in grid.face_dual_lengths()])
            self._face_ratio = L / A
        return self._face_ratio

    def node_volumes(self, state: Mapping[str, Any], who: str) -> np.ndarray:
        """Flat dual (box) volume per node [m^3]."""
        grid = self.grid(state, who)
        if self._node_vol is None:
            self._node_vol = grid.node_volumes().ravel()
        return self._node_vol


def _edge_circulation(state: Mapping[str, Any], cache: _GridCache,
                      who: str) -> np.ndarray:
    """Edge circulation of E [V], preferring ``state['e']``.

    Falls back to ``-G phi``, which is exact in the electroquasistatic model
    (A1a) where ``curl E = 0`` by construction.  In a magnetoquasistatic or
    full-wave run the induced part ``-da/dt`` is *not* recoverable from ``phi``
    alone, so those solvers must supply ``'e'``; the fallback would silently
    drop the inductive term.
    """
    grid = cache.grid(state, who)
    if _present(state, "e"):
        return _flat(state["e"], grid.n_edges, who, "e")
    if _present(state, "phi"):
        phi = _flat(state["phi"], grid.n_nodes, who, "phi")
        return -(cache.G(state, who) @ phi)
    have = ", ".join(sorted(str(k) for k in state)) or "(nothing)"
    raise MonitorStateError(
        f"{who} needs state['e'] (edge circulation of E [V]) or state['phi'] "
        f"(node potential [V]) to form it; state contains: {have}.")


# ==========================================================================
# Monitor base class
# ==========================================================================
class Monitor(ABC):
    """Abstract recorder.

    Subclasses implement :meth:`_record`, which is called once per accepted
    step and must ``self._emit(key, value)`` exactly one sample for each key it
    publishes.  The base class owns the time axis, the stacking and the
    ``name.key`` naming, so subclasses never touch bookkeeping.

    Parameters
    ----------
    name
        Non-empty identifier; becomes the key (or key prefix) in
        :meth:`finalize`.

    Attributes
    ----------
    name : str
        Identifier.
    n_seen : int
        How many times :meth:`record` was called.
    n_records : int
        How many of those were kept (differs only for strided monitors).
    """

    #: Subclasses whose time axis differs from the solver's set this True so
    #: that ``finalize`` also publishes ``f"{name}.t"``.
    _emit_time: bool = False

    def __init__(self, name: str):
        if not isinstance(name, str) or not name:
            raise ValueError("monitor name must be a non-empty string")
        if "/" in name:
            raise ValueError("monitor name must not contain '/'")
        self.name = name
        self._t: list[float] = []
        self._series: dict[str, list[Any]] = {}
        self._n_seen = 0
        self._warned: set[str] = set()
        self._cache = _GridCache()

    # -- public API --------------------------------------------------------
    def record(self, state: Mapping[str, Any], t: float) -> None:
        """Sample the solver state.

        Parameters
        ----------
        state
            Solver state mapping; see the module docstring for the key
            contract.
        t
            Simulation time [s].
        """
        if not isinstance(state, Mapping):
            raise ValueError(
                f"{self!r}: state must be a mapping, got {type(state).__name__}")
        tf = float(t)
        self._n_seen += 1
        if not self._accept(state, tf):
            return
        self._record(state, tf)
        self._t.append(tf)

    def finalize(self) -> dict[str, np.ndarray]:
        """Collected series, each with a leading time axis.

        Returns
        -------
        dict of str to numpy.ndarray
            Keys are ``name`` for a single-quantity monitor and
            ``f"{name}.{quantity}"`` otherwise.  Strided monitors additionally
            return ``f"{name}.t"`` [s].
        """
        out: dict[str, np.ndarray] = {}
        for key, vals in self._series.items():
            full = self.name if key == "" else f"{self.name}.{key}"
            out[full] = _stack(vals)
        if self._emit_time:
            out[f"{self.name}.t"] = np.asarray(self._t, dtype=float)
        return out

    def reset(self) -> None:
        """Discard all recorded data so the monitor can be reused."""
        self._t.clear()
        self._series.clear()
        self._n_seen = 0
        self._warned.clear()

    @property
    def times(self) -> np.ndarray:
        """Times of the accepted records [s]."""
        return np.asarray(self._t, dtype=float)

    @property
    def n_seen(self) -> int:
        return self._n_seen

    @property
    def n_records(self) -> int:
        return len(self._t)

    # -- hooks for subclasses ---------------------------------------------
    @abstractmethod
    def _record(self, state: Mapping[str, Any], t: float) -> None:
        """Emit one sample per published key."""

    def _accept(self, state: Mapping[str, Any], t: float) -> bool:
        """Whether this call should be kept.  Overridden by strided monitors."""
        return True

    def _emit(self, key: str, value: Any) -> None:
        self._series.setdefault(key, []).append(value)

    def _warn_once(self, tag: str, msg: str) -> None:
        if tag not in self._warned:
            self._warned.add(tag)
            warnings.warn(f"{self!r}: {msg}", RuntimeWarning, stacklevel=3)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r} n={len(self._t)}>"


# ==========================================================================
# NodeProbe
# ==========================================================================
class NodeProbe(Monitor):
    """Sample a nodal field at one point, one node index, or a set of nodes.

    The oscilloscope of the field solver: cheap, scalar, and the thing you
    actually plot.

    Parameters
    ----------
    name
        Series name.
    node
        Flat node index, sequence of flat node indices, or boolean node mask.
        Mutually exclusive with ``point``.
    point
        Physical coordinate ``(x, y, z)`` [m]; the nearest node is used and is
        resolved on the first :meth:`record` (the grid arrives with the state).
        Mutually exclusive with ``node``.
    quantity
        State key to sample: ``"phi"`` [V] (default), ``"n"`` or ``"p"``
        [m^-3], or any other nodal state entry.
    reduce
        ``"none"`` stores one value per selected node (shape
        ``(nt, n_selected)``), ``"mean"``, ``"sum"``, ``"min"``, ``"max"``
        reduce to a scalar per step.  A single node index always yields a
        scalar series of shape ``(nt,)``.

    Notes
    -----
    Nearest-node snapping means the probe reports the potential at a grid node,
    not at the requested coordinate; on a graded mesh those can differ by half a
    cell.  :attr:`node_index` and :attr:`node_coords` expose where the probe
    actually landed, which is the honest thing to put in a figure caption.
    """

    def __init__(self, name: str, node: Any = None,
                 point: Sequence[float] | None = None,
                 quantity: str = "phi", reduce: str = "mean"):
        super().__init__(name)
        if (node is None) == (point is None):
            raise ValueError(
                f"NodeProbe {name!r}: give exactly one of node= or point=")
        if reduce not in ("none", "mean", "sum", "min", "max"):
            raise ValueError(
                f"NodeProbe {name!r}: unknown reduce={reduce!r}")
        if point is not None:
            pt = np.asarray(point, dtype=float).ravel()
            if pt.size != 3:
                raise ValueError(
                    f"NodeProbe {name!r}: point must be (x, y, z) in metres, "
                    f"got {pt.size} components")
            self._point: np.ndarray | None = pt
        else:
            self._point = None
        self._sel = node
        self.quantity = str(quantity)
        self.reduce = reduce
        self._scalar = isinstance(node, (int, np.integer))
        self._nodes: np.ndarray | None = None
        self.node_coords: tuple[float, float, float] | None = None

    @property
    def node_index(self) -> np.ndarray | None:
        """Flat node indices actually sampled, once resolved."""
        return self._nodes

    def _resolve(self, state: Mapping[str, Any]) -> np.ndarray:
        if self._nodes is not None:
            return self._nodes
        who = repr(self)
        grid = self._cache.grid(state, who)
        if self._point is not None:
            ijk = grid.nearest_node(self._point)
            self._nodes = np.atleast_1d(
                np.asarray(grid.node_index(*ijk), dtype=np.intp))
            self.node_coords = (float(grid.xn[ijk[0]]), float(grid.yn[ijk[1]]),
                                float(grid.zn[ijk[2]]))
            self._scalar = True
        else:
            self._nodes = _resolve_nodes(self._sel, state, who, grid.n_nodes)
        return self._nodes

    def _record(self, state: Mapping[str, Any], t: float) -> None:
        who = repr(self)
        grid = self._cache.grid(state, who)
        idx = self._resolve(state)
        raw = _require(state, self.quantity, who,
                       f"NodeProbe samples the nodal field "
                       f"{self.quantity!r}")
        vals = _flat(raw, grid.n_nodes, who, self.quantity)[idx]
        if self._scalar or idx.size == 1:
            self._emit("", float(vals[0] if self.reduce == "none"
                                 else _reduce(vals, self.reduce)))
        elif self.reduce == "none":
            self._emit("", np.array(vals, dtype=float))
        else:
            self._emit("", float(_reduce(vals, self.reduce)))


def _reduce(vals: np.ndarray, how: str) -> float:
    if how == "mean":
        return float(np.mean(vals))
    if how == "sum":
        return float(np.sum(vals))
    if how == "min":
        return float(np.min(vals))
    if how == "max":
        return float(np.max(vals))
    return float(vals[0])


# ==========================================================================
# TerminalProbe
# ==========================================================================
class TerminalProbe(Monitor):
    """Voltage and current at an electrode --- what a circuit engineer wants.

    Publishes ``f"{name}.v"`` [V] and ``f"{name}.i"`` [A], and when the
    displacement term is active also ``f"{name}.i_cond"`` and
    ``f"{name}.i_disp"`` [A].

    **Current sign: positive flows into the field region.**  The current is the
    KCL residual of the electrode,

    .. code-block:: text

        I = sum_(nodes in terminal) [ G^T M_sigma G phi
                                    + G^T M_eps   G dphi/dt ]

    which is exactly the right-hand side the electroquasistatic solver would
    need to sustain the observed potential (see the sign discussion in the
    module docstring).  For a resistor held at ``V`` across ``R`` this returns
    ``+V/R`` at the high electrode and ``-V/R`` at the ground electrode; the
    two sum to zero, which is a useful runtime check on any solve.

    Parameters
    ----------
    terminal
        A :class:`~fieldspice.solvers.base.Terminal`, or its name, in which
        case it is looked up in ``state["terminals"]``.
    name
        Series prefix; defaults to the terminal name.
    displacement
        ``True`` always include the ``eps dphi/dt`` term, ``False`` never,
        ``"auto"`` (default) include it when ``state["eps_edge"]`` is present.
        Resolved once, on the first record, so all series stay the same length.
    voltage_from
        ``"auto"`` (default) prefers ``state["terminal_voltage"]``, then the
        mean of ``phi`` over the electrode nodes, then the terminal's own
        declared voltage.  ``"phi"`` forces the field value and errors if
        ``phi`` is absent, ``"declared"`` forces ``Terminal.value_at(t)`` and
        errors if the terminal is not voltage-driven.

    Notes
    -----
    ``dphi/dt`` is taken from ``state["dphidt"]`` when the solver supplies it.
    Otherwise it is a first-order **backward difference** between successive
    records, so the very first sample carries zero displacement current --- an
    unavoidable consequence of having seen only one frame, not a bug, but it
    does mean the current at ``t = 0`` of a capacitive structure is
    understated.  A solver that cares should pass ``dphidt`` or, better,
    ``terminal_current``, which is used verbatim when present because the
    solver knows it exactly.
    """

    def __init__(self, terminal: Terminal | str, name: str | None = None,
                 displacement: bool | str = "auto",
                 voltage_from: str = "auto"):
        if isinstance(terminal, str):
            tname = terminal
        elif _is_terminal(terminal):
            tname = str(terminal.name)
        else:
            raise ValueError(
                "TerminalProbe: terminal must be a Terminal or a terminal name")
        if displacement not in (True, False, "auto"):
            raise ValueError(
                "TerminalProbe: displacement must be True, False or 'auto'")
        if voltage_from not in ("auto", "phi", "declared"):
            raise ValueError(
                "TerminalProbe: voltage_from must be 'auto', 'phi' or 'declared'")
        super().__init__(name if name is not None else tname)
        self.terminal_name = tname
        self._terminal: Terminal | None = (
            terminal if _is_terminal(terminal) else None)
        self._disp_req = displacement
        self._disp: bool | None = None
        self.voltage_from = voltage_from
        self._nodes: np.ndarray | None = None
        self._phi_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    def _resolve(self, state: Mapping[str, Any], who: str) -> None:
        if self._terminal is None:
            self._terminal = _find_terminal(state, self.terminal_name, who)
        if self._nodes is None:
            grid = self._cache.grid(state, who)
            self._nodes = _resolve_nodes(self._terminal, state, who,
                                         grid.n_nodes)
        if self._disp is None:
            if self._disp_req == "auto":
                self._disp = _present(state, "eps_edge")
            else:
                self._disp = bool(self._disp_req)
            if self._disp:
                _require(state, "eps_edge", who,
                         "the displacement current term eps dphi/dt")

    def _record(self, state: Mapping[str, Any], t: float) -> None:
        who = repr(self)
        grid = self._cache.grid(state, who)
        self._resolve(state, who)
        assert self._nodes is not None and self._terminal is not None

        # -- voltage -------------------------------------------------------
        v: float | None = None
        tv = state.get("terminal_voltage")
        if self.voltage_from in ("auto",) and isinstance(tv, Mapping) \
                and self.terminal_name in tv:
            v = float(tv[self.terminal_name])
        if v is None and self.voltage_from in ("auto", "phi") \
                and _present(state, "phi"):
            phi = _flat(state["phi"], grid.n_nodes, who, "phi")
            v = float(np.mean(phi[self._nodes]))
        if v is None:
            declared = self._terminal.value_at(t)
            if declared is None or getattr(self._terminal, "driven", "") \
                    == "current":
                if self.voltage_from == "declared":
                    raise MonitorStateError(
                        f"{who}: voltage_from='declared' but terminal "
                        f"{self.terminal_name!r} is not voltage-driven")
                v = float("nan")
            else:
                v = float(declared)
        self._emit("v", v)

        # -- current -------------------------------------------------------
        ti = state.get("terminal_current")
        if isinstance(ti, Mapping) and self.terminal_name in ti:
            i_total = float(ti[self.terminal_name])
            self._emit("i", i_total)
            if self._disp:
                # The solver's number is authoritative but unsplit; do not
                # invent a decomposition it did not provide.
                self._emit("i_cond", float("nan"))
                self._emit("i_disp", float("nan"))
            self._phi_prev = None
            self._t_prev = t
            return

        phi = _flat(_require(state, "phi", who,
                             "reconstructing the terminal current from the "
                             "field needs the node potential"),
                    grid.n_nodes, who, "phi")
        G = self._cache.G(state, who)
        ratio = self._cache.edge_ratio(state, who)

        i_cond = 0.0
        if _present(state, "sigma_edge"):
            sig = _flat(state["sigma_edge"], grid.n_edges, who, "sigma_edge")
            i_cond = float(np.sum(
                (G.T @ ((sig * ratio) * (G @ phi)))[self._nodes]))
        elif not self._disp:
            raise MonitorStateError(
                f"{who}: needs state['sigma_edge'] or state['eps_edge'] to "
                f"reconstruct a terminal current, or state['terminal_current'] "
                f"supplied directly by the solver.")

        i_disp = 0.0
        if self._disp:
            eps = _flat(state["eps_edge"], grid.n_edges, who, "eps_edge")
            dphi = self._dphidt(state, phi, t, grid, who)
            if dphi is not None:
                i_disp = float(np.sum(
                    (G.T @ ((eps * ratio) * (G @ dphi)))[self._nodes]))
            self._emit("i_cond", i_cond)
            self._emit("i_disp", i_disp)
        self._emit("i", i_cond + i_disp)

        self._phi_prev = np.array(phi, dtype=float, copy=True)
        self._t_prev = t

    def _dphidt(self, state: Mapping[str, Any], phi: np.ndarray, t: float,
                grid: RectilinearGrid, who: str) -> np.ndarray | None:
        if _present(state, "dphidt"):
            return _flat(state["dphidt"], grid.n_nodes, who, "dphidt")
        if self._phi_prev is None or self._t_prev is None:
            return None
        dt = t - self._t_prev
        if dt <= 0.0:
            self._warn_once(
                "dt", "non-increasing time between records; displacement "
                      "current set to zero for this sample")
            return None
        return (phi - self._phi_prev) / dt


# ==========================================================================
# FieldSnapshot
# ==========================================================================
class FieldSnapshot(Monitor):
    """Store whole fields every ``every`` steps.

    This is the memory bottleneck of any transient field run, so the stride is
    a required design decision rather than an afterthought.  One frame costs

    .. code-block:: text

        bytes_per_frame = n_elements * itemsize

    For a 200 x 200 x 50 grid that is 2.05e6 nodes, 16.4 MB per frame in
    float64.  Ten thousand steps at ``every=1`` is 164 GB; at the default
    ``every=10`` it is 16 GB, which is still too much, so pick ``every`` from
    the frame budget you actually have.  :attr:`bytes_per_frame` reports the
    real number once the first frame has been seen, and ``max_frames`` turns a
    slow swap-death into an immediate, explicit error.

    Parameters
    ----------
    fields
        State key or sequence of keys to store (``"phi"``, ``"e"``, ``"b"``,
        ``"a"``, ``"n"``, ``"p"``, ...).
    every
        Stride in solver steps.  ``1`` stores every step.  Default ``10``.
    name
        Series prefix.  Defaults to the field name for a single field, else
        ``"snapshot"``.
    dtype
        Storage dtype.  ``numpy.float32`` halves the memory at the cost of
        ~7 significant digits, which is almost always enough for a field that
        is only going to be plotted.  The solve itself stays float64.
    reshape
        Reshape node-sized arrays to ``grid.shape_nodes`` and cell-sized ones
        to ``grid.shape_cells``, matching the ``Result.fields`` convention.
        Edge and face vectors stay flat, as ``Result`` specifies.
    max_frames
        Raise :class:`ValueError` rather than store more than this many frames.

    Notes
    -----
    Frames are appended to a Python list and stacked once in :meth:`finalize`.
    That costs one extra full copy at the end but does not require knowing the
    number of steps in advance, and appending is O(1) amortised, so the run
    itself never pays a reallocation.  Every frame is **copied** out of the
    state, because solvers legitimately reuse their work buffers.
    """

    _emit_time = True

    def __init__(self, fields: str | Sequence[str] = "phi", every: int = 10,
                 name: str | None = None, dtype: Any = np.float64,
                 reshape: bool = True, max_frames: int | None = None):
        if isinstance(fields, str):
            keys = [fields]
        else:
            keys = [str(f) for f in fields]
        if not keys:
            raise ValueError("FieldSnapshot: at least one field is required")
        every = int(every)
        if every < 1:
            raise ValueError("FieldSnapshot: every must be >= 1")
        if max_frames is not None and int(max_frames) < 1:
            raise ValueError("FieldSnapshot: max_frames must be >= 1")
        auto = keys[0] if len(keys) == 1 else "snapshot"
        super().__init__(name if name is not None else auto)
        self.fields = tuple(keys)
        self.every = every
        self.dtype = np.dtype(dtype)
        self.reshape = bool(reshape)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.bytes_per_frame: int = 0

    def _accept(self, state: Mapping[str, Any], t: float) -> bool:
        step = state.get("step")
        idx = int(step) if step is not None else self._n_seen - 1
        return idx % self.every == 0

    def _record(self, state: Mapping[str, Any], t: float) -> None:
        who = repr(self)
        if self.max_frames is not None and len(self._t) >= self.max_frames:
            raise ValueError(
                f"{who}: max_frames={self.max_frames} exceeded after "
                f"{self._n_seen} steps. Increase every= (currently "
                f"{self.every}), reduce the stored fields, or raise "
                f"max_frames.")
        grid = self._cache.grid(state, who)
        total = 0
        for key in self.fields:
            raw = _require(state, key, who,
                           f"FieldSnapshot stores the field {key!r}")
            arr = np.array(raw, dtype=self.dtype, copy=True)
            if self.reshape:
                if arr.size == grid.n_nodes:
                    arr = arr.reshape(grid.shape_nodes)
                elif arr.size == grid.n_cells:
                    arr = arr.reshape(grid.shape_cells)
                else:
                    arr = arr.ravel()
            else:
                arr = arr.ravel()
            self._emit(key if len(self.fields) > 1 else "", arr)
            total += arr.nbytes
        self.bytes_per_frame = total


# ==========================================================================
# EnergyMonitor
# ==========================================================================
class EnergyMonitor(Monitor):
    """Stored electric and magnetic energy, and instantaneous dissipation.

    Publishes, for whichever components are active,

    ==========================  =====  ============================================
    Key                         Unit   Expression
    ==========================  =====  ============================================
    ``f"{name}.electric"``      J      ``0.5 * e^T M_eps e``
    ``f"{name}.magnetic"``      J      ``0.5 * b^T M_nu b``
    ``f"{name}.dissipation"``   W      ``e^T M_sigma e``
    ``f"{name}.total"``         J      electric + magnetic
    ==========================  =====  ============================================

    with ``M_eps = diag(eps_e A_dual/L)`` the edge capacitance [F],
    ``M_sigma = diag(sigma_e A_dual/L)`` the edge conductance [S], and
    ``M_nu = diag(L_dual/(mu_f A_f))`` the face reluctance [1/H].

    Because ``e = -G phi`` in the electroquasistatic model, the electric term
    is identically ``0.5 * phi^T (G^T M_eps G) phi``, i.e. one half of the
    quadratic form of the nodal capacitance matrix.  Verified against a
    parallel-plate capacitor to 1e-15 relative error, and against a resistor
    for the dissipation.

    The factor of two is the trap.  Both *stored* energies carry ``0.5``
    because they are ``integral x dx``; the dissipation does **not**, because
    Joule heating is ``V I = G V^2`` with no integral.  If a reported
    dissipation is exactly half the expected ``V^2/R``, this is why.

    Parameters
    ----------
    name
        Series prefix.  Default ``"energy"``.
    components
        ``"auto"`` (default) enables every component the first state supports
        and then holds that choice fixed for the run, so all series stay the
        same length.  Otherwise a subset of ``("electric", "magnetic",
        "dissipation")``; any requested component whose inputs are missing
        raises on the first record.

    Notes
    -----
    ``state["m_nu"]`` overrides the reluctance with whatever the solver
    actually used (an array of length ``n_faces`` [1/H] or a diagonal sparse
    matrix).  Prefer it: see the module-level warning about
    :func:`fieldspice.operators.face_mass_nu`, which returns the reciprocal
    geometry factor and therefore does not have units of 1/H.
    """

    _ALL = ("electric", "magnetic", "dissipation")

    def __init__(self, name: str = "energy",
                 components: str | Sequence[str] = "auto"):
        super().__init__(name)
        if components == "auto":
            self._requested: tuple[str, ...] | None = None
        else:
            if isinstance(components, str):
                components = (components,)
            comps = tuple(str(c) for c in components)
            bad = [c for c in comps if c not in self._ALL]
            if bad:
                raise ValueError(
                    f"EnergyMonitor {name!r}: unknown components {bad}, "
                    f"expected a subset of {self._ALL}")
            if not comps:
                raise ValueError(
                    f"EnergyMonitor {name!r}: components must not be empty")
            self._requested = comps
        self.components_used: tuple[str, ...] = ()

    def _resolve(self, state: Mapping[str, Any], who: str) -> None:
        if self.components_used:
            return
        have_e = _present(state, "e") or _present(state, "phi")
        if self._requested is None:
            comps = []
            if have_e and _present(state, "eps_edge"):
                comps.append("electric")
            if _present(state, "b") and (_present(state, "mu_face")
                                         or _present(state, "m_nu")):
                comps.append("magnetic")
            if have_e and _present(state, "sigma_edge"):
                comps.append("dissipation")
            if not comps:
                have = ", ".join(sorted(str(k) for k in state)) or "(nothing)"
                raise MonitorStateError(
                    f"{who}: components='auto' found nothing computable. "
                    f"Electric energy needs ('e' or 'phi') and 'eps_edge'; "
                    f"magnetic needs 'b' and ('mu_face' or 'm_nu'); "
                    f"dissipation needs ('e' or 'phi') and 'sigma_edge'. "
                    f"State contains: {have}.")
            self.components_used = tuple(comps)
        else:
            for c in self._requested:
                if c in ("electric", "dissipation"):
                    if not have_e:
                        raise MonitorStateError(
                            f"{who}: component {c!r} needs state['e'] or "
                            f"state['phi']")
                    _require(state, "eps_edge" if c == "electric"
                             else "sigma_edge", who, f"component {c!r}")
                else:
                    _require(state, "b", who, "component 'magnetic'")
                    if not (_present(state, "mu_face")
                            or _present(state, "m_nu")):
                        raise MonitorStateError(
                            f"{who}: component 'magnetic' needs "
                            f"state['mu_face'] [H/m] or state['m_nu'] [1/H]")
            self.components_used = self._requested

    def _record(self, state: Mapping[str, Any], t: float) -> None:
        who = repr(self)
        grid = self._cache.grid(state, who)
        self._resolve(state, who)
        comps = self.components_used

        w_e = w_m = None
        if "electric" in comps or "dissipation" in comps:
            e = _edge_circulation(state, self._cache, who)
            ratio = self._cache.edge_ratio(state, who)
            e2 = e * e
            if "electric" in comps:
                eps = _flat(state["eps_edge"], grid.n_edges, who, "eps_edge")
                w_e = 0.5 * float(np.dot(eps * ratio, e2))
                self._emit("electric", w_e)
            if "dissipation" in comps:
                sig = _flat(state["sigma_edge"], grid.n_edges, who,
                            "sigma_edge")
                self._emit("dissipation", float(np.dot(sig * ratio, e2)))

        if "magnetic" in comps:
            b = _flat(state["b"], grid.n_faces, who, "b")
            nu = self._reluctance(state, grid, who)
            w_m = 0.5 * float(np.dot(nu, b * b))
            self._emit("magnetic", w_m)

        if w_e is not None or w_m is not None:
            self._emit("total", (w_e or 0.0) + (w_m or 0.0))

    def _reluctance(self, state: Mapping[str, Any], grid: RectilinearGrid,
                    who: str) -> np.ndarray:
        """Face reluctance [1/H], from ``m_nu`` if given else from ``mu_face``."""
        if _present(state, "m_nu"):
            m_nu = state["m_nu"]
            diag = m_nu.diagonal() if sp.issparse(m_nu) else np.asarray(m_nu)
            return _flat(diag, grid.n_faces, who, "m_nu")
        mu = _flat(state["mu_face"], grid.n_faces, who, "mu_face")
        if np.any(mu <= 0.0):
            raise ValueError(f"{who}: mu_face must be strictly positive [H/m]")
        return self._cache.face_ratio(state, who) / mu


# ==========================================================================
# FluxMonitor
# ==========================================================================
class FluxMonitor(Monitor):
    """Net current through an oriented surface [A].

    The surface is a set of primal edges, all pointing the same way, and the
    reported quantity is the signed sum of the edge currents crossing it.  On
    the Yee grid a coordinate plane cuts exactly the edges normal to it, so
    "current through a plane" is an exact discrete statement, not an
    interpolation: the number this monitor returns for a closed cross-section
    of a wire equals the terminal current to machine precision.

    Sign convention, matching :mod:`fieldspice.operators`: the current along an
    edge is ``i_e = -(M_sigma G phi)_e = (M_sigma e)_e``, positive along the
    edge's own ``+x``/``+y``/``+z`` direction.  The surface normal is therefore
    ``+axis`` by default; pass ``sign=-1`` to flip it.

    Publishes ``name`` [A], or with ``include_displacement=True`` also
    ``f"{name}.cond"`` and ``f"{name}.disp"`` [A] (in which case the total is
    under ``f"{name}.total"``).

    Parameters
    ----------
    name
        Series name.
    axis, position
        Coordinate plane: ``axis`` is ``"x"``/``"y"``/``"z"`` (or 0/1/2) and
        ``position`` is the coordinate [m].  The plane snaps to the layer of
        edges spanning it.  Mutually exclusive with ``edges``.
    bounds
        Optional 3-sequence of ``(lo, hi)`` [m] or ``None`` restricting the
        transverse extent of the plane, e.g. to cut one conductor out of a
        cross-section.  The entry for ``axis`` itself is ignored.
    edges
        Explicit flat edge indices into the concatenated ``[ex, ey, ez]``
        vector.  Mutually exclusive with ``axis``/``position``.
    signs
        Optional per-edge signs (``+1``/``-1``) matching ``edges``.
    sign
        Overall orientation multiplier, ``+1`` (default) or ``-1``.
    include_displacement
        Add ``eps de/dt`` to the conduction current.  Needs ``eps_edge``; with
        it on, ``sigma_edge`` becomes optional (a missing one is treated as a
        perfect insulator, with a warning).  ``de/dt`` comes from
        ``state["dedt"]`` when supplied, otherwise from a backward difference
        between records, which is first order and gives exactly zero on the
        first sample.  If the monitor is called at a stride, that difference
        spans the stride, so keep ``every=1`` on any solver-side down-sampling
        when displacement current matters.

    Notes
    -----
    A plane at a domain wall selects the edges of the outermost cell layer,
    which is the correct "current entering the domain" surface.  An empty
    selection (bounds that miss every edge) is a :class:`ValueError`, not a
    silent zero.
    """

    def __init__(self, name: str, axis: str | int | None = None,
                 position: float | None = None,
                 bounds: Sequence[Sequence[float] | None] | None = None,
                 edges: Any = None, signs: Any = None, sign: int = 1,
                 include_displacement: bool = False):
        super().__init__(name)
        by_plane = axis is not None or position is not None
        if by_plane == (edges is not None):
            raise ValueError(
                f"FluxMonitor {name!r}: give either axis= and position=, or "
                f"edges=, but not both")
        if by_plane and (axis is None or position is None):
            raise ValueError(
                f"FluxMonitor {name!r}: a plane needs both axis= and position=")
        if sign not in (1, -1, 1.0, -1.0):
            raise ValueError(f"FluxMonitor {name!r}: sign must be +1 or -1")
        self.axis: int | None = None
        if by_plane:
            if axis not in _AXIS_NAMES:
                raise ValueError(
                    f"FluxMonitor {name!r}: axis must be one of "
                    f"'x', 'y', 'z', 0, 1, 2; got {axis!r}")
            self.axis = _AXIS_NAMES[axis]
            self.position = float(position)  # type: ignore[arg-type]
        if bounds is not None:
            bl = list(bounds)
            if len(bl) != 3:
                raise ValueError(
                    f"FluxMonitor {name!r}: bounds must have 3 entries, one "
                    f"per axis (use None to leave an axis unrestricted)")
            self.bounds: list[Any] | None = bl
        else:
            self.bounds = None
        self._edge_sel = None if edges is None else np.asarray(edges).ravel()
        self._edge_signs_in = None if signs is None else np.asarray(
            signs, dtype=float).ravel()
        self.sign = float(sign)
        self.include_displacement = bool(include_displacement)
        self._edges: np.ndarray | None = None
        self._signs: np.ndarray | None = None
        self._e_prev: np.ndarray | None = None
        self._t_prev: float | None = None

    # -- surface resolution -----------------------------------------------
    def _resolve(self, state: Mapping[str, Any], who: str) -> None:
        if self._edges is not None:
            return
        grid = self._cache.grid(state, who)
        if self._edge_sel is not None:
            idx = self._edge_sel
            if idx.dtype == bool:
                if idx.size != grid.n_edges:
                    raise ValueError(
                        f"{who}: boolean edge mask has {idx.size} entries, "
                        f"expected {grid.n_edges}")
                idx = np.flatnonzero(idx)
            if not np.issubdtype(idx.dtype, np.integer):
                raise ValueError(f"{who}: edges must be integer indices")
            idx = idx.astype(np.intp)
            if idx.size == 0:
                raise ValueError(f"{who}: edge selection is empty")
            if idx.min() < 0 or idx.max() >= grid.n_edges:
                raise ValueError(
                    f"{who}: edge index out of range [0, {grid.n_edges})")
            sgn = (np.ones(idx.size) if self._edge_signs_in is None
                   else self._edge_signs_in)
            if sgn.size != idx.size:
                raise ValueError(
                    f"{who}: signs has {sgn.size} entries but edges has "
                    f"{idx.size}")
            self._edges, self._signs = idx, self.sign * sgn
            return

        d = self.axis
        assert d is not None
        nodes_d = (grid.xn, grid.yn, grid.zn)[d]
        ncell_d = grid.ncell[d]
        lo, hi = float(nodes_d[0]), float(nodes_d[-1])
        if self.position < lo - 1e-12 or self.position > hi + 1e-12:
            raise ValueError(
                f"{who}: plane at {self.position:g} m is outside the domain "
                f"extent [{lo:g}, {hi:g}] m along axis {'xyz'[d]}")
        layer = int(np.clip(np.searchsorted(nodes_d, self.position,
                                            side="right") - 1, 0, ncell_d - 1))
        shape = grid.shape_edges[d]
        off = (0, grid.n_edges_each[0],
               grid.n_edges_each[0] + grid.n_edges_each[1])[d]
        ids = (off + np.arange(int(np.prod(shape)))).reshape(shape)
        sl: list[Any] = [slice(None)] * 3
        sl[d] = layer
        sel = ids[tuple(sl)]

        keep = np.ones(sel.shape, dtype=bool)
        if self.bounds is not None:
            trans = [a for a in (0, 1, 2) if a != d]
            all_nodes = (grid.xn, grid.yn, grid.zn)
            for pos, a in enumerate(trans):
                lim = self.bounds[a]
                if lim is None:
                    continue
                blo, bhi = float(lim[0]), float(lim[1])
                if bhi < blo:
                    raise ValueError(
                        f"{who}: bounds along {'xyz'[a]} must be increasing")
                coord = all_nodes[a]
                if coord.size != sel.shape[pos]:
                    raise ValueError(
                        f"{who}: internal indexing error resolving bounds")
                ok = (coord >= blo - 1e-12) & (coord <= bhi + 1e-12)
                keep &= ok.reshape((-1, 1) if pos == 0 else (1, -1))
        idx = sel[keep].ravel().astype(np.intp)
        if idx.size == 0:
            raise ValueError(
                f"{who}: the plane axis={'xyz'[d]} position={self.position:g} "
                f"with bounds={self.bounds} selects no edges")
        self._edges = idx
        self._signs = np.full(idx.size, self.sign)
        self.n_edges_cut = int(idx.size)

    # -- recording ---------------------------------------------------------
    def _record(self, state: Mapping[str, Any], t: float) -> None:
        who = repr(self)
        grid = self._cache.grid(state, who)
        self._resolve(state, who)
        assert self._edges is not None and self._signs is not None

        e = _edge_circulation(state, self._cache, who)
        ratio = self._cache.edge_ratio(state, who)
        sel, sgn = self._edges, self._signs

        i_cond = 0.0
        if _present(state, "sigma_edge"):
            sig = _flat(state["sigma_edge"], grid.n_edges, who, "sigma_edge")
            i_cond = float(np.dot(sgn, (sig * ratio)[sel] * e[sel]))
        elif self.include_displacement:
            self._warn_once(
                "sigma", "no state['sigma_edge']; conduction current taken as "
                         "zero (perfect insulator)")
        else:
            raise MonitorStateError(
                f"{who} needs state['sigma_edge'] [S/m] to form a conduction "
                f"current, or include_displacement=True with "
                f"state['eps_edge'].")

        if not self.include_displacement:
            self._emit("", i_cond)
            self._e_prev = np.array(e, dtype=float, copy=True)
            self._t_prev = t
            return

        eps = _flat(_require(state, "eps_edge", who,
                             "the displacement current term eps de/dt"),
                    grid.n_edges, who, "eps_edge")
        dedt = self._dedt(state, e, t, grid, who)
        i_disp = 0.0 if dedt is None else float(
            np.dot(sgn, (eps * ratio)[sel] * dedt[sel]))
        self._emit("cond", i_cond)
        self._emit("disp", i_disp)
        self._emit("total", i_cond + i_disp)
        self._e_prev = np.array(e, dtype=float, copy=True)
        self._t_prev = t

    def _dedt(self, state: Mapping[str, Any], e: np.ndarray, t: float,
              grid: RectilinearGrid, who: str) -> np.ndarray | None:
        if _present(state, "dedt"):
            return _flat(state["dedt"], grid.n_edges, who, "dedt")
        if self._e_prev is None or self._t_prev is None:
            return None
        dt = t - self._t_prev
        if dt <= 0.0:
            self._warn_once(
                "dt", "non-increasing time between records; displacement "
                      "current set to zero for this sample")
            return None
        return (e - self._e_prev) / dt


# ==========================================================================
# ChargeMonitor
# ==========================================================================
class ChargeMonitor(Monitor):
    """Total charge on a set of nodes [C].

    Two independent routes, because they answer different questions:

    ``mode="gauss"`` (default)
        ``Q = sum_nodes (G^T M_eps G phi)``.  This is the discrete Gauss law
        with ``G^T M_eps G`` the nodal capacitance matrix, so summed over an
        electrode it gives the **free charge on that electrode**, and
        ``Q / V`` is its self-capacitance.  Summed over *all* nodes it gives
        the net free charge in the domain, which is a machine-precision zero
        for any isolated system --- a cheap and very sharp correctness check
        on a solve.

    ``mode="carriers"``
        ``Q = q * sum_nodes (p - n + doping) * V_node``, the semiconductor
        space charge, using the dual box volumes so the box method conserves
        charge exactly.  ``doping`` is the net ``Nd - Na`` [m^-3] and is taken
        as zero if the state omits it.

    Parameters
    ----------
    name
        Series name.
    nodes
        Node selector: a :class:`~fieldspice.solvers.base.Terminal`, a terminal
        name, flat indices, or a boolean node mask.  ``None`` (default) means
        every node.
    mode
        ``"gauss"`` or ``"carriers"``.
    """

    def __init__(self, name: str = "charge", nodes: Any = None,
                 mode: str = "gauss"):
        super().__init__(name)
        if mode not in ("gauss", "carriers"):
            raise ValueError(
                f"ChargeMonitor {name!r}: mode must be 'gauss' or 'carriers', "
                f"got {mode!r}")
        self.mode = mode
        self._sel = nodes
        self._nodes: np.ndarray | None = None

    def _resolve(self, state: Mapping[str, Any], who: str) -> np.ndarray | None:
        """``None`` means "all nodes", which lets us skip the fancy indexing."""
        if self._sel is None:
            return None
        if self._nodes is None:
            grid = self._cache.grid(state, who)
            self._nodes = _resolve_nodes(self._sel, state, who, grid.n_nodes)
        return self._nodes

    def _record(self, state: Mapping[str, Any], t: float) -> None:
        who = repr(self)
        grid = self._cache.grid(state, who)
        idx = self._resolve(state, who)

        if self.mode == "gauss":
            phi = _flat(_require(state, "phi", who,
                                 "Gauss-law charge needs the node potential"),
                        grid.n_nodes, who, "phi")
            eps = _flat(_require(state, "eps_edge", who,
                                 "Gauss-law charge needs the edge permittivity"),
                        grid.n_edges, who, "eps_edge")
            G = self._cache.G(state, who)
            ratio = self._cache.edge_ratio(state, who)
            qn = G.T @ ((eps * ratio) * (G @ phi))
        else:
            n = _flat(_require(state, "n", who,
                               "carrier charge needs the electron density"),
                      grid.n_nodes, who, "n")
            p = _flat(_require(state, "p", who,
                               "carrier charge needs the hole density"),
                      grid.n_nodes, who, "p")
            if _present(state, "doping"):
                dop = _flat(state["doping"], grid.n_nodes, who, "doping")
            else:
                dop = 0.0
            vol = self._cache.node_volumes(state, who)
            qn = _q_electron * (p - n + dop) * vol

        self._emit("", float(qn.sum() if idx is None else qn[idx].sum()))


# ==========================================================================
# MonitorSet
# ==========================================================================
class MonitorSet:
    """A named collection of monitors that records and finalizes as one.

    Solvers take a ``monitors=`` argument and should immediately wrap it with
    :meth:`coerce`, which accepts ``None``, a single :class:`Monitor`, an
    iterable of monitors, or an existing :class:`MonitorSet`.  After that the
    solver only ever calls :meth:`record` and :meth:`finalize`, so adding a
    monitor type never touches a solver.

    Parameters
    ----------
    monitors
        Initial monitors.  Names must be unique.
    """

    def __init__(self, monitors: Iterable[Monitor] | None = None):
        self._monitors: dict[str, Monitor] = {}
        self._t: list[float] = []
        if monitors is not None:
            for mon in monitors:
                self.add(mon)

    # -- construction ------------------------------------------------------
    @classmethod
    def coerce(cls, monitors: Any) -> "MonitorSet":
        """Normalise whatever a caller passed into a :class:`MonitorSet`."""
        if monitors is None:
            return cls()
        if isinstance(monitors, MonitorSet):
            return monitors
        if isinstance(monitors, Monitor):
            return cls([monitors])
        return cls(monitors)

    def add(self, monitor: Monitor) -> Monitor:
        """Register a monitor and return it, so it can be added inline."""
        if not isinstance(monitor, Monitor):
            raise ValueError(
                f"MonitorSet.add expects a Monitor, got "
                f"{type(monitor).__name__}")
        if monitor.name in self._monitors:
            raise ValueError(
                f"MonitorSet already has a monitor named {monitor.name!r}")
        self._monitors[monitor.name] = monitor
        return monitor

    # -- the loop ----------------------------------------------------------
    def record(self, state: Mapping[str, Any], t: float) -> None:
        """Forward one sample to every monitor."""
        tf = float(t)
        for mon in self._monitors.values():
            mon.record(state, tf)
        self._t.append(tf)

    def finalize(self) -> dict[str, np.ndarray]:
        """Merge every monitor's output, plus the master time axis ``"t"`` [s].

        Raises
        ------
        ValueError
            If two monitors produce the same key, which would silently lose
            one of them.
        """
        out: dict[str, np.ndarray] = {"t": np.asarray(self._t, dtype=float)}
        for mon in self._monitors.values():
            for key, arr in mon.finalize().items():
                if key in out:
                    raise ValueError(
                        f"monitor key collision on {key!r}: rename one of the "
                        f"monitors")
                out[key] = arr
        return out

    def terminal_series(self) -> dict[str, dict[str, np.ndarray]]:
        """``{terminal: {"v": (nt,), "i": (nt,)}}`` from every TerminalProbe.

        The shape :attr:`fieldspice.solvers.base.Result.terminals` wants, so a
        solver can assign it straight across.
        """
        out: dict[str, dict[str, np.ndarray]] = {}
        for mon in self._monitors.values():
            if isinstance(mon, TerminalProbe):
                data = mon.finalize()
                out[mon.terminal_name] = {
                    "v": data[f"{mon.name}.v"],
                    "i": data[f"{mon.name}.i"],
                }
        return out

    def reset(self) -> None:
        """Clear every monitor's data and the master time axis."""
        self._t.clear()
        for mon in self._monitors.values():
            mon.reset()

    # -- container protocol ------------------------------------------------
    @property
    def times(self) -> np.ndarray:
        return np.asarray(self._t, dtype=float)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._monitors)

    def __len__(self) -> int:
        return len(self._monitors)

    def __iter__(self):
        return iter(self._monitors.values())

    def __contains__(self, key: object) -> bool:
        return key in self._monitors

    def __getitem__(self, key: str) -> Monitor:
        return self._monitors[key]

    def __repr__(self) -> str:
        return (f"<MonitorSet {len(self._monitors)} monitors "
                f"[{', '.join(self._monitors)}], {len(self._t)} samples>")
