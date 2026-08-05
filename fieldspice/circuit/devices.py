"""Compact device models --- the lumped half of a mixed field/circuit solve.

Every class here is a *stamp*: given the modified-nodal-analysis (MNA) matrix
``G``, the right-hand side ``I`` and the present Newton iterate ``x``, a device
adds its own contribution and nothing else.  That is the entire interface, and
it is what lets :mod:`fieldspice.circuit.mna` drive an arbitrary mix of linear
elements, transistors and field-region Schur complements without knowing what
any of them are.

Conventions (all SI, no scaling anywhere)
-----------------------------------------
Unknown vector ``x`` holds node potentials [V] followed by the branch currents
[A] of the elements that need them (voltage sources, inductors, VCVS).  The
system solved is ``G x = I`` with ``I`` the current *injected into* each node
[A] --- the same sign convention the field solvers use for
``G_op^T M_sigma G_op phi = i_inject``.

Nonlinear devices stamp the **Newton companion**: ``G`` receives the analytic
Jacobian ``di/dv`` at the present iterate and ``I`` receives
``J x_k - F(x_k)``, so that one linear solve advances Newton by one step and a
linear element is just the degenerate case with a constant Jacobian.  Writing it
this way means the caller's loop is identical for linear and nonlinear
netlists::

    for k in range(max_newton):
        G, I = stamp_all(x)
        x_new = spsolve(G, I)
        if converged(x_new, x): break
        x = x_new

Reactive elements use the **charge/flux based** backward-Euler companion
(``i = (q(v) - q_prev)/dt``) rather than a capacitance-based one, because the
charge form conserves charge exactly under a varying capacitance, while
``C(v) dv/dt`` does not.

Node names
----------
Nodes are strings.  ``"0"``, ``"gnd"``, ``"GND"`` and ``"ground"`` are the
reference node and are stamped nowhere; every other name must appear in the
``nmap`` dict the caller passes in.  A device that needs its own unknown
(voltage source, inductor, VCVS) reports the name through
:meth:`Device.extra_unknowns`; the canonical spelling is ``"i(<devname>)"``, and
a small set of aliases is accepted so the interface survives a different naming
choice in :mod:`fieldspice.circuit.mna`.

Numerics: why this file is mostly about limiting, not about physics
-------------------------------------------------------------------
The models below are textbook.  What makes them usable is the limiting, and
that deserves the emphasis it gets here:

* :func:`limexp` replaces ``exp`` above a threshold argument with its tangent
  line.  ``exp(710)`` overflows a float64, and a Newton iterate that overshoots
  a junction by 1 V asks for ``exp(38)`` on the next line and ``exp(1500)`` two
  iterations later.  With ``limexp`` the model is monotone, positive and
  continuously differentiable everywhere, so an overshoot costs an iteration
  instead of an ``inf``.
* :func:`pnjlim` is the standard SPICE junction limiter: it damps the *change*
  in a junction voltage between Newton iterations, logarithmically once the
  junction is forward biased.  Without it, Newton on a diode is a divergent map
  for almost any starting point --- the tangent to an exponential at a
  forward-biased point crosses zero far below the root, so the undamped step
  overshoots backwards, and the next tangent is astronomically steep.
* :func:`fetlim` and :func:`limvds` are the MOSFET equivalents.

Every nonlinear device supplies an **analytic** Jacobian.  Finite-difference
Jacobians on an exponential device lose all their significant digits at exactly
the bias where the device is interesting.

Assumptions invoked (``docs/ASSUMPTIONS.md``)
---------------------------------------------
* **A3** --- device parameters are constant in time; no self-heating feedback
  and no aging.
* **A6** --- isothermal.  Temperature is a per-device constant ``T`` [K] used
  only to compute the thermal voltage; there is no thermal network.
* **A14** --- mismatch lives here rather than in the field solve.  The FET
  classes carry ``vth_sigma`` [V] and ``beta_sigma`` (relative, dimensionless)
  and expose :meth:`_MismatchMixin.from_seed` /
  :meth:`_MismatchMixin.apply_mismatch`, so a Monte Carlo loop over device
  instances costs one circuit solve per sample instead of one field solve.

Compact models are *fits*, not physics.  Where a model has a known qualitative
defect (the level-1 subthreshold junction is C0 but not C1; level-1 with
channel-length modulation is discontinuous at ``Vds = Vov`` in the reference
convention) the docstring says so and points at the class that does it
properly, which is nearly always :class:`EKV`.
"""

from __future__ import annotations

import copy
import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..units import T_ROOM, thermal_voltage

__all__ = [
    # helpers and limiting
    "GROUND_NAMES", "LIMEXP_ARG", "limexp", "dlimexp", "pnjlim", "fetlim",
    "limvds", "resolve_node", "stamp_g", "stamp_i", "pelgrom_sigma",
    # devices
    "Device", "Resistor", "Capacitor", "Inductor", "VSource", "ISource",
    "VCVS", "VCCS", "Diode", "MOSFETL1", "EKV", "SubthresholdTFT", "Switch",
    "SPICE_PREFIX",
]


# ==========================================================================
# Node / matrix plumbing
# ==========================================================================
GROUND_NAMES = frozenset({"0", "gnd", "GND", "ground", "Ground", "GROUND", ""})
"""Node names that denote the reference node.  Stamps touching it are dropped,
which is exactly the row/column deletion that makes the MNA matrix nonsingular."""

_BRANCH_ALIASES = ("i({name})", "{name}:i", "i_{name}", "{name}")
"""Accepted spellings for a device's own branch-current unknown, most canonical
first.  :meth:`Device.extra_unknowns` returns the first form; the others are
accepted on lookup so that a different naming choice in ``mna.py`` does not
break every voltage source in the netlist."""


def _node_name(n: Any) -> str:
    """Coerce a user node label to the canonical string form."""
    return n if isinstance(n, str) else str(n)


def resolve_node(nmap: Mapping[str, int], node: str) -> int:
    """Map a node name to its row index in the MNA system.

    Parameters
    ----------
    nmap
        ``{node name: row index}``.  Supplied by the netlist.
    node
        Node label.

    Returns
    -------
    int
        Row/column index, or ``-1`` for the reference node (ground), which
        signals "do not stamp".

    Notes
    -----
    ``nmap`` is consulted *before* the ground-name table, so a caller that
    deliberately keeps an explicit ground row (some formulations pin it instead
    of deleting it) still gets the behaviour it asked for.
    """
    idx = nmap.get(node)
    if idx is not None:
        return int(idx)
    if node in GROUND_NAMES:
        return -1
    raise KeyError(
        f"node {node!r} is not in the node map and is not a ground alias "
        f"{sorted(GROUND_NAMES)}; known nodes: {sorted(nmap)[:20]}")


def _branch_index(nmap: Mapping[str, int], name: str) -> int:
    """Row index of the branch-current unknown belonging to device ``name``."""
    for pat in _BRANCH_ALIASES:
        key = pat.format(name=name)
        if key in nmap:
            return int(nmap[key])
    raise KeyError(
        f"device {name!r} needs a branch-current unknown; none of "
        f"{[p.format(name=name) for p in _BRANCH_ALIASES]} is in the node map. "
        f"The netlist must allocate the names returned by extra_unknowns().")


def stamp_g(G: Any, i: int, j: int, value: float) -> None:
    """Accumulate ``value`` into matrix entry ``(i, j)``; ground rows dropped.

    ``G`` may be a dense ``ndarray``, a ``scipy.sparse.lil_matrix`` or
    ``dok_matrix``, or any object exposing ``add(i, j, v)`` (a triplet
    accumulator).  ``csr_matrix`` works but warns about sparsity changes and is
    O(nnz) per write --- build in ``lil``/``dok``/triplet form and convert once.
    """
    if i < 0 or j < 0:
        return
    add = getattr(G, "add", None)
    if add is not None:
        add(i, j, value)
    else:
        G[i, j] += value


def stamp_i(I: Any, i: int, value: float) -> None:
    """Accumulate ``value`` [A] into RHS entry ``i``; ground row dropped."""
    if i < 0:
        return
    I[i] += value


def _v(x: np.ndarray | None, i: int) -> float:
    """Read unknown ``i`` from iterate ``x``; 0 for ground or a missing ``x``.

    A missing ``x`` is the legitimate first-iteration state, not an error: the
    zero vector is the standard MNA starting guess.
    """
    if x is None or i < 0:
        return 0.0
    return float(x[i])


# ==========================================================================
# Limiting and safe exponentials
# ==========================================================================
LIMEXP_ARG = 80.0
"""Argument above which :func:`limexp` switches to its tangent line.

``exp(80) = 5.5e34``.  A junction current of 5e34 A is nonsense in any circuit,
while ``exp(710)`` is ``inf`` and destroys the whole matrix, so the entire
interval between them is pure numerical hazard with no physics in it.  For a
diode with ``n = 1`` at 300 K this threshold sits at 2.07 V of forward bias,
roughly 1.3 V beyond where any real junction is still described by the ideal
diode equation.
"""


def limexp(x: np.ndarray | float) -> np.ndarray | float:
    """Overflow-proof exponential: ``exp(x)`` below :data:`LIMEXP_ARG`, tangent above.

    Parameters
    ----------
    x
        Dimensionless exponent (typically ``V / (n*Vt)``).

    Returns
    -------
    float or ndarray
        ``exp(x)`` for ``x <= LIMEXP_ARG``; ``exp(A)*(1 + x - A)`` above, where
        ``A = LIMEXP_ARG``.

    Notes
    -----
    Value and first derivative are continuous at the switch point, the result
    is strictly positive and strictly increasing everywhere, and it cannot
    overflow.  Those four properties are exactly what Newton needs: the
    linearised model retains a unique root and a well-signed conductance no
    matter how far an iterate wanders.
    """
    scalar = np.ndim(x) == 0
    xa = np.asarray(x, dtype=float)
    xs = np.minimum(xa, LIMEXP_ARG)
    e = np.exp(xs)
    out = np.where(xa > LIMEXP_ARG, e * (1.0 + (xa - LIMEXP_ARG)), e)
    return float(out) if scalar else out


def dlimexp(x: np.ndarray | float) -> np.ndarray | float:
    """Derivative of :func:`limexp`: ``exp(min(x, LIMEXP_ARG))``."""
    scalar = np.ndim(x) == 0
    xa = np.asarray(x, dtype=float)
    out = np.exp(np.minimum(xa, LIMEXP_ARG))
    return float(out) if scalar else out


def pnjlim(vnew: float, vold: float, vt: float, vcrit: float) -> float:
    """SPICE junction limiter: damp the change in a pn-junction voltage.

    Parameters
    ----------
    vnew, vold
        Proposed and previous junction voltage [V].
    vt
        Effective thermal voltage of the junction, ``n*kT/q`` [V].
    vcrit
        Critical voltage [V], ``vt*ln(vt/(sqrt(2)*Is))`` --- the bias at which
        the exponential's curvature is such that an undamped Newton step
        starts to diverge.

    Returns
    -------
    float
        Limited junction voltage [V].

    Notes
    -----
    Above ``vcrit`` the update is compressed logarithmically, which is the
    inverse of the exponential the model is about to evaluate: a proposed jump
    of many volts becomes a jump of a few ``vt``.  This function is the single
    reason Newton converges on a diode; a plain step clamp does not work,
    because it does not adapt to how far up the exponential the iterate sits.
    """
    if vnew > vcrit and abs(vnew - vold) > 2.0 * vt:
        if vold > 0.0:
            arg = 1.0 + (vnew - vold) / vt
            if arg > 0.0:
                return vold + vt * math.log(arg)
            return vcrit
        if vnew > 0.0:
            # Coming from reverse bias: land on the log of the requested value
            # rather than tracking a huge relative jump.
            return vt * math.log(vnew / vt)
    return vnew


def fetlim(vnew: float, vold: float, vto: float) -> float:
    """SPICE MOSFET gate-voltage limiter (``DEVfetlim``).

    Parameters
    ----------
    vnew, vold
        Proposed and previous gate-source voltage [V].
    vto
        Threshold voltage [V].

    Returns
    -------
    float
        Limited gate-source voltage [V].

    Notes
    -----
    Unlike :func:`pnjlim` this is tuned to a square-law device, so its step
    limits are volts rather than millivolts.  It is therefore *not* adequate on
    its own for a device whose current is exponential in ``Vgs``; see
    :class:`SubthresholdTFT`, which uses :func:`pnjlim` on the gate instead.
    """
    vtsthi = abs(2.0 * (vold - vto)) + 2.0
    vtstlo = 0.5 * vtsthi + 2.0
    vtox = vto + 3.5
    delv = vnew - vold
    if vold >= vto:
        if vold >= vtox:
            if delv <= 0.0:
                if vnew >= vtox:
                    if -delv > vtstlo:
                        vnew = vold - vtstlo
                else:
                    vnew = max(vnew, vto + 2.0)
            else:
                if delv >= vtsthi:
                    vnew = vold + vtsthi
        else:
            if delv <= 0.0:
                vnew = max(vnew, vto - 0.5)
            else:
                vnew = min(vnew, vto + 4.0)
    else:
        if delv <= 0.0:
            if -delv > vtsthi:
                vnew = vold - vtsthi
        else:
            vtemp = vto + 0.5
            if vnew <= vtemp:
                if delv > vtstlo:
                    vnew = vold + vtstlo
            else:
                vnew = vtemp
    return vnew


def limvds(vnew: float, vold: float) -> float:
    """SPICE drain-source limiter (``DEVlimvds``).  Voltages in [V]."""
    if vold >= 3.5:
        if vnew > vold:
            return min(vnew, 3.0 * vold + 2.0)
        if vnew < 3.5:
            return max(vnew, 2.0)
        return vnew
    if vnew > vold:
        return min(vnew, 4.0)
    return max(vnew, -0.5)


def pelgrom_sigma(a_coeff: float, w: float, l: float) -> float:
    """Pelgrom mismatch sigma from an area coefficient.

    Parameters
    ----------
    a_coeff
        Pelgrom coefficient, [V m] for threshold mismatch (e.g. 5e-9 V m
        = 5 mV um for a mature CMOS node) or dimensionless-times-metres for a
        relative parameter.
    w, l
        Device width and length [m].

    Returns
    -------
    float
        ``a_coeff / sqrt(w*l)``, in [V] for a threshold coefficient.

    Notes
    -----
    The 1/sqrt(area) law is an averaging argument over independent microscopic
    fluctuations, so it holds for anything that averages: dopant count, oxide
    thickness, grain statistics in a thin film.  It fails when a single defect
    dominates, which is the normal situation in large-grain or amorphous-oxide
    TFTs --- treat the number as a lower bound there.
    """
    if w <= 0.0 or l <= 0.0:
        raise ValueError("w and l must be positive [m]")
    return float(a_coeff / math.sqrt(w * l))


def _sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    """Logistic function, computed as ``0.5*(1+tanh(z/2))`` for stability."""
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(z, dtype=float)))


# ==========================================================================
# Device base class
# ==========================================================================
class Device(ABC):
    """Abstract compact model.

    Attributes
    ----------
    name
        Unique instance name; also the key under which any branch-current
        unknown is registered.
    nodes
        Node labels in a device-specific order, documented per subclass.
    linear
        ``True`` if the stamp does not depend on ``x`` (no Newton needed).
    dynamic
        ``True`` if the device stores charge or flux (transient stamp differs
        from the DC stamp).
    assumptions
        Tags from ``docs/ASSUMPTIONS.md`` active for this model.
    t
        Present simulation time [s], used to evaluate time-dependent sources
        when the caller does not pass ``t`` explicitly to the stamp.
    limited
        Set by the most recent stamp: ``True`` if a limiter changed a device
        voltage, meaning the model was **not** evaluated at the voltage implied
        by ``x``.

    Notes
    -----
    **The caller must not declare Newton converged while any device reports
    ``limited``.**  This is not a refinement, it is a correctness requirement,
    and it is measured rather than asserted: with a stock SPICE ``limvds``, two
    successive iterates can be clamped to the *same* pair of limit values, which
    produces an identical stamp, hence an exactly zero Newton step, at a state
    whose KCL residual is 1e9 A.  A step-size-only convergence test accepts that
    as a solution.  The rule ``converged = small_step and not any(d.limited)``
    costs one boolean and removes the failure mode --- it is what SPICE's
    ``CKTnoncon`` counter is for.  Note also that ``G x - I`` is *not* the true
    residual on an iteration where limiting occurred, since the companion was
    linearised about the limited voltage, so a residual test needs the same
    guard.
    """

    linear: bool = True
    dynamic: bool = False
    assumptions: tuple[str, ...] = ("A3",)

    def __init__(self, name: str, nodes: Sequence[Any]):
        if not isinstance(name, str) or not name:
            raise ValueError("device name must be a non-empty string")
        self.name = name
        self.nodes: tuple[str, ...] = tuple(_node_name(n) for n in nodes)
        self.t: float = 0.0
        self.limited: bool = False

    # -- interface ---------------------------------------------------------
    def extra_unknowns(self) -> tuple[str, ...]:
        """Names of unknowns this device adds beyond the node voltages.

        The netlist must allocate a row and column for each returned name
        before stamping.  Default: none.
        """
        return ()

    def set_time(self, t: float) -> None:
        """Set the simulation time [s] used by time-dependent sources."""
        self.t = float(t)

    @abstractmethod
    def stamp_dc(self, G: Any, I: Any, x: np.ndarray | None,
                 nmap: Mapping[str, int], t: float | None = None) -> None:
        """Add this device's DC contribution to ``G`` [S] and ``I`` [A].

        Parameters
        ----------
        G
            MNA matrix being accumulated (see :func:`stamp_g` for accepted
            types).  Units are siemens in the node-voltage block and
            dimensionless in the branch-constraint rows.
        I
            Right-hand side [A] (or [V] in a branch-constraint row).
        x
            Present Newton iterate, or ``None`` for the zero starting guess.
        nmap
            ``{name: row index}`` for nodes and extra unknowns.
        t
            Simulation time [s].  Optional: the positional signature in
            ``docs/CONTRACTS.md`` has no time argument, so a caller that does
            not pass it gets ``self.t`` (set via :meth:`set_time`).  Both
            driving styles work.
        """

    def stamp_tran(self, G: Any, I: Any, x: np.ndarray | None,
                   x_prev: np.ndarray | None, dt: float,
                   nmap: Mapping[str, int], t: float | None = None) -> None:
        """Add the transient contribution over a step of size ``dt`` [s].

        Default implementation is the DC stamp, which is correct for every
        memoryless device.
        """
        self.stamp_dc(G, I, x, nmap, t=t)

    def stamp_ac(self, Y: Any, J: Any, x_op: np.ndarray | None,
                 omega: float, nmap: Mapping[str, int]) -> None:
        """Add the small-signal contribution at ``omega`` [rad/s].

        ``Y`` is complex-valued; ``x_op`` is the DC operating point about which
        nonlinear devices are linearised.  ``J`` receives only the AC
        excitation of independent sources.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement an AC stamp")

    def accept_timestep(self, x: np.ndarray, x_prev: np.ndarray | None,
                        dt: float, nmap: Mapping[str, int]) -> None:
        """Commit internal state after a converged time step.

        Only devices with state that cannot be recovered from ``x`` need this
        (trapezoidal integration, hysteretic switches, a diode with series
        resistance).  Every model here degrades gracefully to backward Euler if
        the caller never calls it.
        """
        return None

    def reset_state(self) -> None:
        """Clear iteration history (junction voltages, switch state)."""
        return None

    def op(self, x: np.ndarray | None,
           nmap: Mapping[str, int]) -> dict[str, float]:
        """Operating-point summary: currents [A], voltages [V], conductances [S]."""
        return {}

    # -- helpers for subclasses -------------------------------------------
    def _n(self, nmap: Mapping[str, int], k: int) -> int:
        return resolve_node(nmap, self.nodes[k])

    def _time(self, t: float | None) -> float:
        return self.t if t is None else float(t)

    def __repr__(self) -> str:
        return (f"{type(self).__name__}({self.name!r}, "
                f"nodes={list(self.nodes)})")


def _stamp_nl2(G: Any, I: Any, a: int, b: int, g: float, ieq: float) -> None:
    """Two-terminal nonlinear companion.

    ``g`` [S] is ``di/dv`` at the present iterate and ``ieq = i(v0) - g*v0``
    [A] is the companion current source, with ``i`` positive flowing from node
    ``a`` to node ``b`` inside the device.
    """
    stamp_g(G, a, a, g)
    stamp_g(G, a, b, -g)
    stamp_g(G, b, a, -g)
    stamp_g(G, b, b, g)
    stamp_i(I, a, -ieq)
    stamp_i(I, b, ieq)


def _stamp_fet(G: Any, I: Any, d: int, g_: int, s: int, b: int,
               idr: float, gm: float, gds: float, gmb: float,
               vg: float, vd: float, vs: float, vb: float) -> None:
    """Four-terminal transconductor companion.

    ``idr`` [A] flows from ``d`` to ``s``; ``gm``, ``gds``, ``gmb`` [S] are the
    partial derivatives of ``idr`` with respect to the *node* potentials of
    gate, drain and bulk.  The source partial is fixed by translation
    invariance, ``dI/dVs = -(gm + gds + gmb)``, which is also a useful runtime
    check on a hand-derived Jacobian: the four partials must sum to zero.
    """
    gtot = gm + gds + gmb
    ieq = idr - gm * (vg - vs) - gds * (vd - vs) - gmb * (vb - vs)
    stamp_g(G, d, g_, gm)
    stamp_g(G, d, d, gds)
    stamp_g(G, d, b, gmb)
    stamp_g(G, d, s, -gtot)
    stamp_g(G, s, g_, -gm)
    stamp_g(G, s, d, -gds)
    stamp_g(G, s, b, -gmb)
    stamp_g(G, s, s, gtot)
    stamp_i(I, d, -ieq)
    stamp_i(I, s, ieq)


def _stamp_lin_cap(G: Any, I: Any, a: int, b: int, cap: float, dt: float,
                   va_prev: float, vb_prev: float) -> None:
    """Backward-Euler companion of a constant capacitance ``cap`` [F]."""
    if cap == 0.0:
        return
    geq = cap / dt
    _stamp_nl2(G, I, a, b, geq, -geq * (va_prev - vb_prev))


# ==========================================================================
# Linear two-terminal elements
# ==========================================================================
class Resistor(Device):
    """Linear resistor.

    Parameters
    ----------
    name
        Instance name.
    n1, n2
        Node labels.
    r
        Resistance [ohm], strictly positive.

    Notes
    -----
    Zero resistance is rejected rather than silently regularised: a true short
    is a topological statement and belongs in the netlist as a 0 V source, which
    MNA handles exactly with a branch current.  Replacing it with a 1e-9 ohm
    resistor instead poisons the matrix condition number by 1e9 for no reason.
    """

    def __init__(self, name: str, n1: Any, n2: Any, r: float):
        super().__init__(name, (n1, n2))
        r = float(r)
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError(f"resistor {name!r}: r must be finite and > 0 "
                             f"[ohm], got {r!r}")
        self.r = r

    @property
    def g(self) -> float:
        """Conductance [S]."""
        return 1.0 / self.r

    def stamp_dc(self, G, I, x, nmap, t=None) -> None:
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        gg = self.g
        stamp_g(G, a, a, gg)
        stamp_g(G, a, b, -gg)
        stamp_g(G, b, a, -gg)
        stamp_g(G, b, b, gg)

    def stamp_ac(self, Y, J, x_op, omega, nmap) -> None:
        self.stamp_dc(Y, J, x_op, nmap)

    def op(self, x, nmap):
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        v = _v(x, a) - _v(x, b)
        return {"v": v, "i": v * self.g, "g": self.g, "p": v * v * self.g}


class Capacitor(Device):
    """Linear capacitor with a backward-Euler or trapezoidal companion.

    Parameters
    ----------
    name
        Instance name.
    n1, n2
        Node labels; current is defined positive from ``n1`` to ``n2``.
    c
        Capacitance [F], strictly positive.
    ic
        Initial voltage [V] used when no previous state is available.

    Notes
    -----
    In DC the element is an open circuit and stamps nothing.  That is correct
    physics and a classic source of singular matrices: a node whose only
    connections are capacitors has no DC path to ground, and the netlist must
    supply one (a large resistor, or the caller's gmin).  This class does not
    add a hidden leakage conductance, because a silent 1e-12 S is a silent 1 pA
    error in an analog cell that may be running at 10 pA.

    Trapezoidal integration is available (``integration="trap"``) but needs the
    branch current from the previous step, so the caller must invoke
    :meth:`accept_timestep`.  Backward Euler is the default: it is L-stable,
    while trapezoidal rings on a discontinuous input derivative, which is the
    classic SPICE step-edge artifact.
    """

    dynamic = True

    def __init__(self, name: str, n1: Any, n2: Any, c: float,
                 ic: float = 0.0, integration: str = "be"):
        super().__init__(name, (n1, n2))
        c = float(c)
        if not np.isfinite(c) or c <= 0.0:
            raise ValueError(f"capacitor {name!r}: c must be finite and > 0 "
                             f"[F], got {c!r}")
        if integration not in ("be", "trap"):
            raise ValueError("integration must be 'be' or 'trap'")
        self.c = c
        self.ic = float(ic)
        self.integration = integration
        self._i_prev: float | None = None

    def reset_state(self) -> None:
        self._i_prev = None

    def stamp_dc(self, G, I, x, nmap, t=None) -> None:
        return None  # open circuit

    def stamp_tran(self, G, I, x, x_prev, dt, nmap, t=None) -> None:
        if dt <= 0.0:
            raise ValueError(f"capacitor {self.name!r}: dt must be > 0 [s]")
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        if x_prev is None:
            vprev = self.ic
        else:
            vprev = _v(x_prev, a) - _v(x_prev, b)
        if self.integration == "trap" and self._i_prev is not None:
            geq = 2.0 * self.c / dt
            ieq = -geq * vprev - self._i_prev
        else:
            geq = self.c / dt
            ieq = -geq * vprev
        _stamp_nl2(G, I, a, b, geq, ieq)

    def stamp_ac(self, Y, J, x_op, omega, nmap) -> None:
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        y = 1j * omega * self.c
        stamp_g(Y, a, a, y)
        stamp_g(Y, a, b, -y)
        stamp_g(Y, b, a, -y)
        stamp_g(Y, b, b, y)

    def accept_timestep(self, x, x_prev, dt, nmap) -> None:
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        v = _v(x, a) - _v(x, b)
        vprev = self.ic if x_prev is None else _v(x_prev, a) - _v(x_prev, b)
        if self.integration == "trap" and self._i_prev is not None:
            self._i_prev = 2.0 * self.c * (v - vprev) / dt - self._i_prev
        else:
            self._i_prev = self.c * (v - vprev) / dt

    def op(self, x, nmap):
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        v = _v(x, a) - _v(x, b)
        return {"v": v, "q": self.c * v, "c": self.c,
                "energy": 0.5 * self.c * v * v}


class Inductor(Device):
    """Linear inductor; adds one branch-current unknown.

    Parameters
    ----------
    name
        Instance name.
    n1, n2
        Node labels; the branch current is positive flowing ``n1`` to ``n2``.
    l
        Inductance [H], strictly positive.
    ic
        Initial current [A] used when no previous state is available.

    Notes
    -----
    The branch current is a genuine unknown rather than an eliminated quantity,
    which is what makes the DC case (a short: ``V(n1) - V(n2) = 0``) and the
    transient case (``V(n1) - V(n2) - (L/dt) i = -(L/dt) i_prev``) the same row
    with a different diagonal.  Because the current is in ``x``, no internal
    state is needed for backward Euler.
    """

    dynamic = True

    def __init__(self, name: str, n1: Any, n2: Any, l: float,
                 ic: float = 0.0, integration: str = "be"):
        super().__init__(name, (n1, n2))
        l = float(l)
        if not np.isfinite(l) or l <= 0.0:
            raise ValueError(f"inductor {name!r}: l must be finite and > 0 "
                             f"[H], got {l!r}")
        if integration not in ("be", "trap"):
            raise ValueError("integration must be 'be' or 'trap'")
        self.l = l
        self.ic = float(ic)
        self.integration = integration
        self._v_prev: float | None = None

    def extra_unknowns(self) -> tuple[str, ...]:
        return (f"i({self.name})",)

    def reset_state(self) -> None:
        self._v_prev = None

    def _rows(self, nmap):
        return (self._n(nmap, 0), self._n(nmap, 1),
                _branch_index(nmap, self.name))

    def stamp_dc(self, G, I, x, nmap, t=None) -> None:
        a, b, k = self._rows(nmap)
        stamp_g(G, a, k, 1.0)
        stamp_g(G, b, k, -1.0)
        stamp_g(G, k, a, 1.0)
        stamp_g(G, k, b, -1.0)

    def stamp_tran(self, G, I, x, x_prev, dt, nmap, t=None) -> None:
        if dt <= 0.0:
            raise ValueError(f"inductor {self.name!r}: dt must be > 0 [s]")
        a, b, k = self._rows(nmap)
        i_prev = self.ic if x_prev is None else _v(x_prev, k)
        stamp_g(G, a, k, 1.0)
        stamp_g(G, b, k, -1.0)
        stamp_g(G, k, a, 1.0)
        stamp_g(G, k, b, -1.0)
        if self.integration == "trap" and self._v_prev is not None:
            # v = (2L/dt)*(i - i_prev) - v_prev
            stamp_g(G, k, k, -2.0 * self.l / dt)
            stamp_i(I, k, -(2.0 * self.l / dt) * i_prev - self._v_prev)
        else:
            stamp_g(G, k, k, -self.l / dt)
            stamp_i(I, k, -(self.l / dt) * i_prev)

    def stamp_ac(self, Y, J, x_op, omega, nmap) -> None:
        a, b, k = self._rows(nmap)
        stamp_g(Y, a, k, 1.0)
        stamp_g(Y, b, k, -1.0)
        stamp_g(Y, k, a, 1.0)
        stamp_g(Y, k, b, -1.0)
        stamp_g(Y, k, k, -1j * omega * self.l)

    def accept_timestep(self, x, x_prev, dt, nmap) -> None:
        a, b, _ = self._rows(nmap)
        self._v_prev = _v(x, a) - _v(x, b)

    def op(self, x, nmap):
        a, b, k = self._rows(nmap)
        i = _v(x, k)
        return {"v": _v(x, a) - _v(x, b), "i": i, "l": self.l,
                "energy": 0.5 * self.l * i * i}


# ==========================================================================
# Independent and controlled sources
# ==========================================================================
class VSource(Device):
    """Ideal independent voltage source; adds one branch-current unknown.

    Parameters
    ----------
    name
        Instance name.
    n1, n2
        Positive and negative node labels.
    value
        Source value [V]: a constant, or a callable ``f(t) -> float`` (anything
        from :mod:`fieldspice.sources` qualifies).
    ac_mag, ac_phase
        Small-signal amplitude [V] and phase [rad] used by :meth:`stamp_ac`.

    Notes
    -----
    The branch current is positive flowing *into* ``n1``, through the source,
    and out of ``n2`` --- the SPICE convention, in which a source delivering
    power reports a negative current.
    """

    def __init__(self, name: str, n1: Any, n2: Any,
                 value: float | Callable[[float], float],
                 ac_mag: float = 0.0, ac_phase: float = 0.0):
        super().__init__(name, (n1, n2))
        self.value = value
        self.ac_mag = float(ac_mag)
        self.ac_phase = float(ac_phase)

    def extra_unknowns(self) -> tuple[str, ...]:
        return (f"i({self.name})",)

    def value_at(self, t: float | None = None) -> float:
        """Source voltage [V] at time ``t`` [s]."""
        tt = self._time(t)
        return float(self.value(tt)) if callable(self.value) else float(self.value)

    def stamp_dc(self, G, I, x, nmap, t=None) -> None:
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        k = _branch_index(nmap, self.name)
        stamp_g(G, a, k, 1.0)
        stamp_g(G, b, k, -1.0)
        stamp_g(G, k, a, 1.0)
        stamp_g(G, k, b, -1.0)
        stamp_i(I, k, self.value_at(t))

    def stamp_ac(self, Y, J, x_op, omega, nmap) -> None:
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        k = _branch_index(nmap, self.name)
        stamp_g(Y, a, k, 1.0)
        stamp_g(Y, b, k, -1.0)
        stamp_g(Y, k, a, 1.0)
        stamp_g(Y, k, b, -1.0)
        stamp_i(J, k, self.ac_mag * np.exp(1j * self.ac_phase))

    def op(self, x, nmap):
        k = _branch_index(nmap, self.name)
        return {"v": self.value_at(None), "i": _v(x, k)}


class ISource(Device):
    """Ideal independent current source.

    Parameters
    ----------
    name
        Instance name.
    n1, n2
        Node labels.  Positive ``value`` drives current *out of* ``n1``, through
        the source, and *into* ``n2`` --- the SPICE convention.
    value
        Source value [A]: constant or callable ``f(t) -> float``.
    ac_mag, ac_phase
        Small-signal amplitude [A] and phase [rad].
    """

    def __init__(self, name: str, n1: Any, n2: Any,
                 value: float | Callable[[float], float],
                 ac_mag: float = 0.0, ac_phase: float = 0.0):
        super().__init__(name, (n1, n2))
        self.value = value
        self.ac_mag = float(ac_mag)
        self.ac_phase = float(ac_phase)

    def value_at(self, t: float | None = None) -> float:
        """Source current [A] at time ``t`` [s]."""
        tt = self._time(t)
        return float(self.value(tt)) if callable(self.value) else float(self.value)

    def stamp_dc(self, G, I, x, nmap, t=None) -> None:
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        i = self.value_at(t)
        stamp_i(I, a, -i)
        stamp_i(I, b, i)

    def stamp_ac(self, Y, J, x_op, omega, nmap) -> None:
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        i = self.ac_mag * np.exp(1j * self.ac_phase)
        stamp_i(J, a, -i)
        stamp_i(J, b, i)

    def op(self, x, nmap):
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        return {"i": self.value_at(None), "v": _v(x, a) - _v(x, b)}


class VCVS(Device):
    """Voltage-controlled voltage source (SPICE ``E``); one branch unknown.

    Parameters
    ----------
    name
        Instance name.
    np_, nn
        Output nodes.  ``V(np_) - V(nn) = gain * (V(cp) - V(cn))``.
    cp, cn
        Control nodes (drawing no current).
    gain
        Voltage gain [V/V].
    """

    def __init__(self, name: str, np_: Any, nn: Any, cp: Any, cn: Any,
                 gain: float):
        super().__init__(name, (np_, nn, cp, cn))
        self.gain = float(gain)

    def extra_unknowns(self) -> tuple[str, ...]:
        return (f"i({self.name})",)

    def _stamp(self, M, rhs, nmap) -> None:
        p, n, cp, cn = (self._n(nmap, i) for i in range(4))
        k = _branch_index(nmap, self.name)
        stamp_g(M, p, k, 1.0)
        stamp_g(M, n, k, -1.0)
        stamp_g(M, k, p, 1.0)
        stamp_g(M, k, n, -1.0)
        stamp_g(M, k, cp, -self.gain)
        stamp_g(M, k, cn, self.gain)

    def stamp_dc(self, G, I, x, nmap, t=None) -> None:
        self._stamp(G, I, nmap)

    def stamp_ac(self, Y, J, x_op, omega, nmap) -> None:
        self._stamp(Y, J, nmap)

    def op(self, x, nmap):
        k = _branch_index(nmap, self.name)
        p, n = self._n(nmap, 0), self._n(nmap, 1)
        return {"v": _v(x, p) - _v(x, n), "i": _v(x, k), "gain": self.gain}


class VCCS(Device):
    """Voltage-controlled current source (SPICE ``G``); no extra unknown.

    Parameters
    ----------
    name
        Instance name.
    np_, nn
        Output nodes.  Current ``gm*(V(cp) - V(cn))`` [A] flows from ``np_``
        through the device to ``nn``.
    cp, cn
        Control nodes.
    gm
        Transconductance [S].
    """

    def __init__(self, name: str, np_: Any, nn: Any, cp: Any, cn: Any,
                 gm: float):
        super().__init__(name, (np_, nn, cp, cn))
        self.gm = float(gm)

    def _stamp(self, M, nmap) -> None:
        p, n, cp, cn = (self._n(nmap, i) for i in range(4))
        stamp_g(M, p, cp, self.gm)
        stamp_g(M, p, cn, -self.gm)
        stamp_g(M, n, cp, -self.gm)
        stamp_g(M, n, cn, self.gm)

    def stamp_dc(self, G, I, x, nmap, t=None) -> None:
        self._stamp(G, nmap)

    def stamp_ac(self, Y, J, x_op, omega, nmap) -> None:
        self._stamp(Y, nmap)

    def op(self, x, nmap):
        cp, cn = self._n(nmap, 2), self._n(nmap, 3)
        vc = _v(x, cp) - _v(x, cn)
        return {"vc": vc, "i": self.gm * vc, "gm": self.gm}


# ==========================================================================
# Diode
# ==========================================================================
class Diode(Device):
    """Shockley junction diode with series resistance and charge storage.

    Parameters
    ----------
    name
        Instance name.
    anode, cathode
        Node labels.  Forward current flows anode to cathode.
    isat
        Saturation current [A].
    n
        Emission (ideality) factor [-].
    rs
        Series resistance [ohm].  Handled by eliminating the internal junction
        node locally (see Notes), so it costs no extra MNA unknown.
    cj0
        Zero-bias junction capacitance [F].
    vj
        Junction (built-in) potential [V].
    m
        Grading coefficient [-]; 0.5 abrupt, 0.33 linearly graded.
    fc
        Forward-bias depletion-capacitance coefficient [-]; above ``fc*vj`` the
        capacitance is linearly extrapolated, because the depletion formula
        diverges at ``vj``.
    tt
        Transit time [s]; gives the diffusion capacitance ``tt*gd``.
    bv
        Reverse breakdown voltage [V] (positive), or ``None`` for no breakdown.
    ibv
        Current magnitude [A] at ``-bv``.
    nbv
        Breakdown emission factor [-].
    area
        Area multiplier [-] scaling ``isat``, ``cj0`` and ``1/rs``.
    gmin
        Parallel conductance [S] added across the junction.  The SPICE default
        of 1e-12 S is kept: it guarantees a nonsingular Jacobian in deep
        reverse bias, at the cost of a 1 pA floor. Set to 0 when simulating
        femtoampere-scale circuits and let the caller do gmin stepping instead.
    T
        Temperature [K].

    Notes
    -----
    **Series resistance without an extra node.**  The textbook treatment adds an
    internal node, which requires the netlist to allocate an unknown this class
    would have to name.  Instead the scalar equation
    ``v = vd + rs*i_total(vd)`` is solved locally by a damped Newton iteration
    (a handful of steps, each using the same limiting as the outer loop) and the
    composite two-terminal conductance ``1/(1/gd + rs)`` is stamped.  This is
    exact at convergence, keeps the interface free of node-naming conventions,
    and makes ``rs`` cost nothing in matrix size.

    **Charge, not capacitance.**  The transient companion integrates the charge
    ``q(vd) = tt*i(vd) + qj(vd)``, so the amount of charge delivered over a step
    is exactly ``q(v) - q(v_prev)`` regardless of how strongly ``C`` varies
    across the step.  Using ``C(v)*dv/dt`` instead leaks charge whenever the
    junction swings through the ``fc*vj`` knee.

    Assumptions: A3, A6.
    """

    linear = False
    dynamic = True
    assumptions = ("A3", "A6")

    def __init__(self, name: str, anode: Any, cathode: Any,
                 isat: float = 1e-14, n: float = 1.0, rs: float = 0.0,
                 cj0: float = 0.0, vj: float = 1.0, m: float = 0.5,
                 fc: float = 0.5, tt: float = 0.0,
                 bv: float | None = None, ibv: float = 1e-3,
                 nbv: float = 1.0, area: float = 1.0,
                 gmin: float = 1e-12, T: float = T_ROOM):
        super().__init__(name, (anode, cathode))
        if isat <= 0.0:
            raise ValueError(f"diode {name!r}: isat must be > 0 [A]")
        if n <= 0.0:
            raise ValueError(f"diode {name!r}: n must be > 0")
        if rs < 0.0:
            raise ValueError(f"diode {name!r}: rs must be >= 0 [ohm]")
        if not 0.0 < fc < 1.0:
            raise ValueError(f"diode {name!r}: fc must lie in (0, 1)")
        if vj <= 0.0:
            raise ValueError(f"diode {name!r}: vj must be > 0 [V]")
        if area <= 0.0:
            raise ValueError(f"diode {name!r}: area must be > 0")
        if bv is not None and bv <= 0.0:
            raise ValueError(f"diode {name!r}: bv must be > 0 [V] or None")
        self.isat = float(isat) * float(area)
        self.n = float(n)
        self.rs = float(rs) / float(area)
        self.cj0 = float(cj0) * float(area)
        self.vj = float(vj)
        self.m = float(m)
        self.fc = float(fc)
        self.tt = float(tt)
        self.bv = None if bv is None else float(bv)
        self.ibv = float(ibv) * float(area)
        self.nbv = float(nbv)
        self.area = float(area)
        self.gmin = float(gmin)
        self.T = float(T)
        self.vt = thermal_voltage(self.T)
        self._vd_old: float | None = None
        self._vd_prev: float | None = None

    # -- model -------------------------------------------------------------
    @property
    def vcrit(self) -> float:
        """Critical voltage [V] for :func:`pnjlim`."""
        vte = self.n * self.vt
        return float(vte * math.log(vte / (math.sqrt(2.0) * self.isat)))

    def current(self, vd: np.ndarray | float) -> np.ndarray | float:
        """DC junction current [A] at junction voltage ``vd`` [V]."""
        i = self.isat * (limexp(np.asarray(vd, float) / (self.n * self.vt)) - 1.0)
        i = i + self.gmin * np.asarray(vd, float)
        if self.bv is not None:
            i = i - self.ibv * limexp(-(np.asarray(vd, float) + self.bv)
                                      / (self.nbv * self.vt))
        return float(i) if np.ndim(vd) == 0 else i

    def conductance(self, vd: np.ndarray | float) -> np.ndarray | float:
        """Junction conductance ``di/dv`` [S] at ``vd`` [V]."""
        vte = self.n * self.vt
        g = self.isat * dlimexp(np.asarray(vd, float) / vte) / vte + self.gmin
        if self.bv is not None:
            g = g + (self.ibv / (self.nbv * self.vt)) * dlimexp(
                -(np.asarray(vd, float) + self.bv) / (self.nbv * self.vt))
        return float(g) if np.ndim(vd) == 0 else g

    def charge(self, vd: float) -> tuple[float, float]:
        """Stored charge [C] and its derivative, the capacitance [F].

        Depletion charge uses the standard SPICE linearisation above
        ``fc*vj``: the physical formula ``cj0*(1 - v/vj)**-m`` diverges at
        ``v = vj``, which a forward-biased junction routinely exceeds during a
        Newton iteration.
        """
        cj, qj = 0.0, 0.0
        if self.cj0 > 0.0:
            if vd < self.fc * self.vj:
                arg = 1.0 - vd / self.vj
                cj = self.cj0 * arg ** (-self.m)
                qj = (self.cj0 * self.vj * (1.0 - arg ** (1.0 - self.m))
                      / (1.0 - self.m))
            else:
                f1 = (self.vj * (1.0 - (1.0 - self.fc) ** (1.0 - self.m))
                      / (1.0 - self.m))
                f2 = (1.0 - self.fc) ** (1.0 + self.m)
                f3 = 1.0 - self.fc * (1.0 + self.m)
                cj = self.cj0 * (f3 + self.m * vd / self.vj) / f2
                qj = self.cj0 * (f1 + (1.0 / f2) * (
                    f3 * (vd - self.fc * self.vj)
                    + (self.m / (2.0 * self.vj))
                    * (vd * vd - (self.fc * self.vj) ** 2)))
        if self.tt > 0.0:
            qj += self.tt * float(self.current(vd))
            cj += self.tt * float(self.conductance(vd))
        return qj, cj

    # -- local elimination of rs ------------------------------------------
    def _solve_junction(self, v: float, vd_guess: float,
                        dt: float | None = None,
                        q_prev: float = 0.0) -> float:
        """Invert ``v = vd + rs*i_total(vd)`` for the junction voltage [V]."""
        if self.rs == 0.0:
            return v
        vd = vd_guess
        vte = self.n * self.vt
        vcrit = self.vcrit
        for _ in range(80):
            i = float(self.current(vd))
            g = float(self.conductance(vd))
            if dt is not None:
                qq, cc = self.charge(vd)
                i += (qq - q_prev) / dt
                g += cc / dt
            f = vd + self.rs * i - v
            scale = 1e-12 * max(1.0, abs(v)) + 1e-15
            if abs(f) < scale:
                break
            dv = -f / (1.0 + self.rs * g)
            vd = pnjlim(vd + dv, vd, vte, vcrit)
        return vd

    def _junction_voltage(self, v: float, dt: float | None = None,
                          q_prev: float = 0.0) -> float:
        """Limited junction voltage [V] for the present iterate."""
        guess = self._vd_old if self._vd_old is not None else min(v, self.vcrit)
        vd_raw = self._solve_junction(v, guess, dt, q_prev)
        vd = pnjlim(vd_raw, guess, self.n * self.vt, self.vcrit)
        self.limited = vd != vd_raw
        self._vd_old = vd
        return vd

    def reset_state(self) -> None:
        self._vd_old = None
        self._vd_prev = None

    # -- stamps ------------------------------------------------------------
    def stamp_dc(self, G, I, x, nmap, t=None) -> None:
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        v = _v(x, a) - _v(x, b)
        vd = self._junction_voltage(v)
        i = float(self.current(vd))
        g = float(self.conductance(vd))
        if self.rs > 0.0:
            geq = g / (1.0 + self.rs * g)      # series combination, exact
            v_eff = vd + self.rs * i
        else:
            geq, v_eff = g, vd
        _stamp_nl2(G, I, a, b, geq, i - geq * v_eff)

    def stamp_tran(self, G, I, x, x_prev, dt, nmap, t=None) -> None:
        if dt <= 0.0:
            raise ValueError(f"diode {self.name!r}: dt must be > 0 [s]")
        if self.cj0 == 0.0 and self.tt == 0.0:
            self.stamp_dc(G, I, x, nmap, t=t)
            return
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        v = _v(x, a) - _v(x, b)
        v_old = 0.0 if x_prev is None else _v(x_prev, a) - _v(x_prev, b)
        if self._vd_prev is not None:
            vd_prev = self._vd_prev
        else:
            # Recover the previous junction voltage from the previous terminal
            # voltage.  Exact when rs == 0; when rs > 0 it neglects the
            # capacitive part of the previous drop across rs, an O(rs*C*dv/dt)
            # error that the caller removes by calling accept_timestep.
            vd_prev = self._solve_junction(v_old, v_old)
        q_prev = self.charge(vd_prev)[0]
        vd = self._junction_voltage(v, dt, q_prev)
        i = float(self.current(vd))
        g = float(self.conductance(vd))
        qq, cc = self.charge(vd)
        i_tot = i + (qq - q_prev) / dt
        g_tot = g + cc / dt
        if self.rs > 0.0:
            geq = g_tot / (1.0 + self.rs * g_tot)
            v_eff = vd + self.rs * i_tot
        else:
            geq, v_eff = g_tot, vd
        _stamp_nl2(G, I, a, b, geq, i_tot - geq * v_eff)

    def stamp_ac(self, Y, J, x_op, omega, nmap) -> None:
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        v = _v(x_op, a) - _v(x_op, b)
        vd = self._solve_junction(v, self._vd_old if self._vd_old is not None
                                  else min(v, self.vcrit))
        g = float(self.conductance(vd))
        y = g + 1j * omega * self.charge(vd)[1]
        if self.rs > 0.0:
            y = y / (1.0 + self.rs * y)
        stamp_g(Y, a, a, y)
        stamp_g(Y, a, b, -y)
        stamp_g(Y, b, a, -y)
        stamp_g(Y, b, b, y)

    def accept_timestep(self, x, x_prev, dt, nmap) -> None:
        self._vd_prev = self._vd_old

    def op(self, x, nmap):
        a, b = self._n(nmap, 0), self._n(nmap, 1)
        v = _v(x, a) - _v(x, b)
        vd = self._vd_old if self._vd_old is not None else v
        return {"v": v, "vd": vd, "i": float(self.current(vd)),
                "gd": float(self.conductance(vd)), "cj": self.charge(vd)[1]}


# ==========================================================================
# FET family
# ==========================================================================
class _MismatchMixin:
    """Per-instance process mismatch for compact models (assumption A14).

    Two knobs, both standard: ``vth_sigma`` [V] is an additive threshold offset
    and ``beta_sigma`` [-] a relative current-factor spread.  Both default to
    zero, so a device is deterministic unless mismatch is explicitly requested;
    ``docs/ASSUMPTIONS.md`` A14 requires that statistics never appear by
    accident.

    The nominal parameters are stored at construction and mismatch is always
    applied *from nominal*, so repeated sampling of the same instance does not
    accumulate a random walk --- a bug that is easy to write and hard to see,
    since the resulting distribution is still plausibly bell-shaped.
    """

    vth_sigma: float
    beta_sigma: float
    _vth_nom: float
    beta_factor: float

    def _init_mismatch(self, vth_nom: float, vth_sigma: float,
                       beta_sigma: float) -> None:
        if vth_sigma < 0.0 or beta_sigma < 0.0:
            raise ValueError("mismatch sigmas must be >= 0")
        self._vth_nom = float(vth_nom)
        self.vth_sigma = float(vth_sigma)
        self.beta_sigma = float(beta_sigma)
        self.beta_factor = 1.0

    def apply_mismatch(self, rng: np.random.Generator | int | None = None
                       ) -> "_MismatchMixin":
        """Draw one mismatch sample in place and return ``self``.

        Parameters
        ----------
        rng
            ``numpy.random.Generator``, an integer seed, or ``None`` for fresh
            entropy.

        Notes
        -----
        The threshold offset is Gaussian; the current factor is log-normal with
        unit mean and log-sigma ``beta_sigma``, which agrees with the usual
        ``1 + N(0, sigma)`` form to first order but cannot produce a negative
        (or zero) current factor, and a negative beta is a device that Newton
        will chase to infinity.
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        self.vth = self._vth_nom + self.vth_sigma * float(rng.standard_normal())
        s = self.beta_sigma
        self.beta_factor = float(np.exp(s * rng.standard_normal() - 0.5 * s * s))
        return self

    def with_mismatch(self, rng: np.random.Generator | int | None = None
                      ) -> "_MismatchMixin":
        """Return a mismatched *copy*, leaving this instance untouched."""
        return copy.deepcopy(self).apply_mismatch(rng)

    def from_seed(self, seed: int) -> "_MismatchMixin":
        """Return a mismatched copy drawn from ``seed`` (reproducible)."""
        return self.with_mismatch(np.random.default_rng(seed))

    def clear_mismatch(self) -> "_MismatchMixin":
        """Restore nominal parameters."""
        self.vth = self._vth_nom
        self.beta_factor = 1.0
        return self


class _FETBase(Device, _MismatchMixin):
    """Shared plumbing for four-terminal FET compact models.

    Nodes are ordered ``(drain, gate, source, bulk)``.  A three-terminal call
    ties the bulk to the source.  ``polarity`` is ``"n"`` or ``"p"``; a
    p-channel device is evaluated by negating all terminal voltages and the
    resulting current, which leaves the conductance stamp *identical* --- the
    two sign flips cancel in every partial derivative.
    """

    linear = False
    assumptions = ("A3", "A6", "A14")

    def __init__(self, name: str, d: Any, g: Any, s: Any, b: Any | None = None,
                 polarity: str = "n"):
        if b is None:
            b = s
        super().__init__(name, (d, g, s, b))
        if polarity not in ("n", "p"):
            raise ValueError("polarity must be 'n' or 'p'")
        self.polarity = polarity
        self._sign = 1.0 if polarity == "n" else -1.0
        self._vgs_old: float | None = None
        self._vds_old: float | None = None

    def reset_state(self) -> None:
        self._vgs_old = None
        self._vds_old = None

    def _terminal_voltages(self, x, nmap):
        d, g, s, b = (self._n(nmap, i) for i in range(4))
        return d, g, s, b, _v(x, d), _v(x, g), _v(x, s), _v(x, b)

    def drain_current(self, vgs: float, vds: float, vbs: float = 0.0
                      ) -> float:
        """Drain current [A] for internal (polarity-corrected) voltages [V]."""
        return self._eval(vgs, vds, vbs)[0]

    def _eval(self, vgs: float, vds: float, vbs: float
              ) -> tuple[float, float, float, float]:
        """Return ``(id, gm, gds, gmb)`` in the internal, forward frame.

        Units: current [A], conductances [S].  Subclasses implement this and
        need only handle ``vds >= 0``; :meth:`_eval_symmetric` performs the
        drain/source exchange.
        """
        raise NotImplementedError

    def _eval_symmetric(self, vg: float, vd: float, vs: float, vb: float
                        ) -> tuple[float, float, float, float]:
        """Evaluate with drain/source exchange, returning node-referred partials.

        Returns ``(id, dId/dVg, dId/dVd, dId/dVb)`` with ``id`` [A] flowing from
        the *nominal* drain node to the source node, and all derivatives taken
        with respect to node potentials [V].  ``dId/dVs`` is not returned
        because it equals ``-(dId/dVg + dId/dVd + dId/dVb)`` identically.
        """
        sg = self._sign
        vgs, vds, vbs = sg * (vg - vs), sg * (vd - vs), sg * (vb - vs)
        if vds >= 0.0:
            idr, gm, gds, gmb = self._eval(vgs, vds, vbs)
            return sg * idr, gm, gds, gmb
        # Reverse: the physical source is the nominal drain.  Evaluate in that
        # frame and map the partials back; the chain rule gives
        # dId/dVd = f_g + f_d + f_b, which is what keeps the four partials
        # summing to zero across the Vds = 0 crossing.
        idr, fg, fd, fb = self._eval(vgs - vds, -vds, vbs - vds)
        return -sg * idr, -fg, fg + fd + fb, -fb

    def _limit(self, vg: float, vd: float, vs: float) -> tuple[float, float]:
        """Apply SPICE limiting to the internal ``(vgs, vds)`` [V]."""
        sg = self._sign
        vgs, vds = sg * (vg - vs), sg * (vd - vs)
        if self._vgs_old is not None:
            new_gs = fetlim(vgs, self._vgs_old, self.vth)
            new_ds = limvds(vds, self._vds_old)
            self.limited = (new_gs != vgs) or (new_ds != vds)
            vgs, vds = new_gs, new_ds
        self._vgs_old, self._vds_old = vgs, vds
        return vgs, vds

    def _capacitances(self) -> tuple[float, float, float, float]:
        """``(cgs, cgd, cgb, cds)`` [F].  Constant overlap caps only."""
        return (getattr(self, "cgs", 0.0), getattr(self, "cgd", 0.0),
                getattr(self, "cgb", 0.0), getattr(self, "cds", 0.0))

    def _stamp_core(self, G, I, x, nmap, limiting: bool = True) -> None:
        self.limited = False
        d, g, s, b, vd, vg, vs, vb = self._terminal_voltages(x, nmap)
        if limiting and getattr(self, "limiting", True):
            vgs_l, vds_l = self._limit(vg, vd, vs)
            # Rebuild absolute potentials consistent with the limited
            # differences, keeping the source as the reference.
            vg = vs + self._sign * vgs_l
            vd = vs + self._sign * vds_l
        idr, dg, dd, db = self._eval_symmetric(vg, vd, vs, vb)
        idr += self.gmin * (vd - vs)
        dd += self.gmin
        _stamp_fet(G, I, d, g, s, b, idr, dg, dd, db, vg, vd, vs, vb)

    def stamp_dc(self, G, I, x, nmap, t=None) -> None:
        self._stamp_core(G, I, x, nmap)

    def stamp_tran(self, G, I, x, x_prev, dt, nmap, t=None) -> None:
        if dt <= 0.0:
            raise ValueError(f"{self.name!r}: dt must be > 0 [s]")
        self._stamp_core(G, I, x, nmap)
        cgs, cgd, cgb, cds = self._capacitances()
        if cgs == cgd == cgb == cds == 0.0:
            return
        d, g, s, b = (self._n(nmap, i) for i in range(4))
        vp = (lambda i: 0.0) if x_prev is None else (lambda i: _v(x_prev, i))
        _stamp_lin_cap(G, I, g, s, cgs, dt, vp(g), vp(s))
        _stamp_lin_cap(G, I, g, d, cgd, dt, vp(g), vp(d))
        _stamp_lin_cap(G, I, g, b, cgb, dt, vp(g), vp(b))
        _stamp_lin_cap(G, I, d, s, cds, dt, vp(d), vp(s))

    def stamp_ac(self, Y, J, x_op, omega, nmap) -> None:
        d, g, s, b, vd, vg, vs, vb = self._terminal_voltages(x_op, nmap)
        idr, dg, dd, db = self._eval_symmetric(vg, vd, vs, vb)
        dd += self.gmin
        gtot = dg + dd + db
        stamp_g(Y, d, g, dg)
        stamp_g(Y, d, d, dd)
        stamp_g(Y, d, b, db)
        stamp_g(Y, d, s, -gtot)
        stamp_g(Y, s, g, -dg)
        stamp_g(Y, s, d, -dd)
        stamp_g(Y, s, b, -db)
        stamp_g(Y, s, s, gtot)
        cgs, cgd, cgb, cds = self._capacitances()
        for (a, bb, cc) in ((g, s, cgs), (g, d, cgd), (g, b, cgb), (d, s, cds)):
            if cc:
                y = 1j * omega * cc
                stamp_g(Y, a, a, y)
                stamp_g(Y, a, bb, -y)
                stamp_g(Y, bb, a, -y)
                stamp_g(Y, bb, bb, y)

    def op(self, x, nmap):
        d, g, s, b, vd, vg, vs, vb = self._terminal_voltages(x, nmap)
        idr, dg, dd, db = self._eval_symmetric(vg, vd, vs, vb)
        sg = self._sign
        return {"id": idr, "gm": dg, "gds": dd, "gmb": db,
                "vgs": sg * (vg - vs), "vds": sg * (vd - vs),
                "vbs": sg * (vb - vs),
                "gm_over_id": dg / idr if idr != 0.0 else float("nan")}


class MOSFETL1(_FETBase, _MismatchMixin):
    """Shockley level-1 MOSFET with body effect, CLM and a subthreshold tail.

    Parameters
    ----------
    name
        Instance name.
    d, g, s, b
        Drain, gate, source and bulk node labels; ``b`` defaults to ``s``.
    vth
        Zero-bias threshold voltage [V] (magnitude for a p-channel device:
        pass ``vth=0.7, polarity="p"``).
    kp
        Transconductance parameter ``mu*Cox`` [A/V^2].
    w, l
        Channel width and length [m]; the model uses ``k = kp*w/l``.
    lam
        Channel-length modulation [1/V].
    gamma
        Body-effect coefficient [sqrt(V)].
    phi
        Surface potential [V], ``2*phi_F``.
    n_sub
        Subthreshold slope factor [-]; the swing is ``n_sub*ln(10)*kT/q``,
        59.5 mV/decade at 300 K for ``n_sub = 1``.
    subthreshold
        Enable the exponential tail below ``Vth + n_sub*Vt``.
    clm_mode
        ``"spice"`` applies ``(1 + lam*Vds)`` in both triode and saturation,
        which is continuous.  ``"reference"`` applies it only in saturation, to
        match :func:`fieldspice.reference.mosfet_square_law` exactly.
    cgs, cgd, cgb, cds
        Constant overlap capacitances [F].
    gmin
        Drain-source parallel conductance [S].
    vth_sigma, beta_sigma
        Mismatch parameters: threshold offset [V] and relative current-factor
        spread [-] (assumption A14).
    T
        Temperature [K].

    Notes
    -----
    **Two known discontinuities, both inherited from the model, not from this
    implementation.**

    1. With ``clm_mode="reference"`` and ``lam > 0`` the current jumps by a
       factor ``(1 + lam*Vov)`` at ``Vds = Vov``.  That is what the level-1
       equations as usually written actually do; ``clm_mode="spice"`` (the
       default here) removes it.
    2. The subthreshold junction is continuous in value but not in slope:
       ``gm`` halves as ``Vgs`` crosses ``Von`` from above.  That is the SPICE2
       ``NFS`` construction.  It is good enough for digital timing and wrong
       for any analog cell that biases near threshold, which is precisely why
       :class:`EKV` exists.

    Validated against :func:`fieldspice.reference.mosfet_square_law`.

    Assumptions: A3, A6, A14.
    """

    def __init__(self, name: str, d: Any, g: Any, s: Any, b: Any | None = None,
                 vth: float = 0.7, kp: float = 2e-5, w: float = 1e-6,
                 l: float = 1e-6, lam: float = 0.0, gamma: float = 0.0,
                 phi: float = 0.7, n_sub: float = 1.0,
                 subthreshold: bool = True, clm_mode: str = "spice",
                 polarity: str = "n", cgs: float = 0.0, cgd: float = 0.0,
                 cgb: float = 0.0, cds: float = 0.0, gmin: float = 0.0,
                 vth_sigma: float = 0.0, beta_sigma: float = 0.0,
                 limiting: bool = True, T: float = T_ROOM):
        super().__init__(name, d, g, s, b, polarity=polarity)
        if kp <= 0.0 or w <= 0.0 or l <= 0.0:
            raise ValueError(f"MOSFETL1 {name!r}: kp, w, l must be > 0")
        if n_sub <= 0.0:
            raise ValueError(f"MOSFETL1 {name!r}: n_sub must be > 0")
        if clm_mode not in ("spice", "reference"):
            raise ValueError("clm_mode must be 'spice' or 'reference'")
        if phi <= 0.0:
            raise ValueError(f"MOSFETL1 {name!r}: phi must be > 0 [V]")
        self.vth = float(vth)
        self.kp, self.w, self.l = float(kp), float(w), float(l)
        self.lam, self.gamma, self.phi = float(lam), float(gamma), float(phi)
        self.n_sub = float(n_sub)
        self.subthreshold = bool(subthreshold)
        self.clm_mode = clm_mode
        self.cgs, self.cgd, self.cgb, self.cds = (float(cgs), float(cgd),
                                                  float(cgb), float(cds))
        self.gmin = float(gmin)
        self.limiting = bool(limiting)
        self.T = float(T)
        self.vt = thermal_voltage(self.T)
        self._init_mismatch(self.vth, vth_sigma, beta_sigma)
        self.dynamic = any((cgs, cgd, cgb, cds))

    @property
    def k(self) -> float:
        """Current factor ``kp*W/L`` [A/V^2], including mismatch."""
        return self.kp * self.w / self.l * self.beta_factor

    def _vth_eff(self, vbs: float) -> tuple[float, float]:
        """Body-effect threshold [V] and ``dVth/dVbs`` [-]."""
        if self.gamma == 0.0:
            return self.vth, 0.0
        arg = self.phi - vbs
        if arg < 1e-6:            # forward-biased body: freeze the square root
            arg = 1e-6
            dvth = 0.0
        else:
            dvth = -self.gamma * 0.5 / math.sqrt(arg)
        return self.vth + self.gamma * (math.sqrt(arg)
                                        - math.sqrt(self.phi)), dvth

    def _square_law(self, vgs: float, vds: float, vth: float
                    ) -> tuple[float, float, float]:
        """Level-1 current [A] and ``(dI/dVgs, dI/dVds)`` [S] for ``vds >= 0``."""
        k = self.k
        vov = vgs - vth
        if vov <= 0.0:
            return 0.0, 0.0, 0.0
        clm_tri = self.clm_mode == "spice"
        if vds < vov:
            base = k * (vov * vds - 0.5 * vds * vds)
            dbdg = k * vds
            dbdd = k * (vov - vds)
            if clm_tri and self.lam != 0.0:
                f = 1.0 + self.lam * vds
                return base * f, dbdg * f, dbdd * f + base * self.lam
            return base, dbdg, dbdd
        base = 0.5 * k * vov * vov
        f = 1.0 + self.lam * vds
        return base * f, k * vov * f, base * self.lam

    def _eval(self, vgs: float, vds: float, vbs: float):
        vth, dvth_dvbs = self._vth_eff(vbs)
        if self.subthreshold:
            von = vth + self.n_sub * self.vt
            if vgs < von:
                # SPICE2 construction: evaluate the square law frozen at Von
                # and scale by the Boltzmann factor.  Continuous in value at
                # Von, with a factor-of-two slope step in gm (see class docs).
                i0, _, gds0 = self._square_law(von, vds, vth)
                arg = (vgs - von) / (self.n_sub * self.vt)
                e = float(limexp(arg))
                de = float(dlimexp(arg))
                idr = i0 * e
                gm = i0 * de / (self.n_sub * self.vt)
                gds = gds0 * e
                gmb = -gm * dvth_dvbs
                return idr, gm, gds, gmb
        idr, gm, gds = self._square_law(vgs, vds, vth)
        return idr, gm, gds, -gm * dvth_dvbs


class EKV(_FETBase, _MismatchMixin):
    """EKV-style charge-based MOSFET, continuous from weak to strong inversion.

    The simplified EKV v2.6 core::

        Vp   = (Vg - Vth)/n                       (pinch-off voltage [V])
        i_f  = ln^2(1 + exp((Vp - Vs)/(2*Vt)))    (forward normalised current)
        i_r  = ln^2(1 + exp((Vp - Vd)/(2*Vt)))
        Id   = Ispec * (i_f - i_r),  Ispec = 2*n*mu*Cox*(W/L)*Vt^2

    all voltages referred to the bulk.

    Parameters
    ----------
    name
        Instance name.
    d, g, s, b
        Node labels; ``b`` defaults to ``s``.
    vth
        Threshold voltage [V].
    kp
        ``mu*Cox`` [A/V^2].
    w, l
        Width and length [m].
    n
        Slope factor [-]; subthreshold swing ``n*ln(10)*kT/q``.
    lam
        Channel-length modulation [1/V], applied through a smoothed ``|Vds|``
        so the model stays differentiable at ``Vds = 0`` (see Notes).
    cgs, cgd, cgb, cds
        Constant overlap capacitances [F].
    gmin
        Drain-source parallel conductance [S].
    vth_sigma, beta_sigma
        Mismatch parameters (assumption A14).
    T
        Temperature [K].

    Notes
    -----
    **Why this model rather than a level-1 with a patched tail.**  The
    interpolation function ``ln^2(1 + exp(x))`` is analytic everywhere.  In weak
    inversion ``ln(1+e^x) -> e^x`` so ``Id -> Ispec*exp((Vg-Vth)/(n*Vt))``, the
    exponential law; in strong inversion ``ln(1+e^x) -> x`` so
    ``Id -> (k/(2n))*(Vg-Vth)^2``, the square law.  There is no branch, no
    ``Von``, and therefore no kink: ``gm/Id`` slides smoothly from its ceiling
    ``1/(n*Vt)`` down the ``2/Vov`` strong-inversion branch.  Since ``gm/Id`` is
    *the* design variable in low-power analog, a model with a discontinuity
    there produces optimisers that converge onto the discontinuity.

    **No overflow path exists.**  ``ln(1+exp(x))`` is evaluated as
    ``logaddexp(0, x)``, which is exact and bounded for every float64 input, so
    unlike the exponential models here EKV needs no ``limexp``.  Terminal
    limiting is still applied by default, purely to speed up Newton.

    **Smoothed CLM.**  ``lam`` multiplies ``1 + lam*(sqrt(Vds^2 + (4Vt)^2) -
    4Vt)``, which equals ``lam*|Vds|`` away from the origin and is smooth
    through it.  The plain ``1 + lam*Vds`` used by level-1 would make the model
    asymmetric and drive the current negative at large reverse ``Vds``.

    Assumptions: A3, A6, A14.
    """

    def __init__(self, name: str, d: Any, g: Any, s: Any, b: Any | None = None,
                 vth: float = 0.5, kp: float = 2e-5, w: float = 1e-6,
                 l: float = 1e-6, n: float = 1.3, lam: float = 0.0,
                 polarity: str = "n", cgs: float = 0.0, cgd: float = 0.0,
                 cgb: float = 0.0, cds: float = 0.0, gmin: float = 0.0,
                 vth_sigma: float = 0.0, beta_sigma: float = 0.0,
                 limiting: bool = True, T: float = T_ROOM):
        super().__init__(name, d, g, s, b, polarity=polarity)
        if kp <= 0.0 or w <= 0.0 or l <= 0.0:
            raise ValueError(f"EKV {name!r}: kp, w, l must be > 0")
        if n <= 0.0:
            raise ValueError(f"EKV {name!r}: n must be > 0")
        self.vth = float(vth)
        self.kp, self.w, self.l = float(kp), float(w), float(l)
        self.n, self.lam = float(n), float(lam)
        self.cgs, self.cgd, self.cgb, self.cds = (float(cgs), float(cgd),
                                                  float(cgb), float(cds))
        self.gmin = float(gmin)
        self.limiting = bool(limiting)
        self.T = float(T)
        self.vt = thermal_voltage(self.T)
        self._init_mismatch(self.vth, vth_sigma, beta_sigma)
        self.dynamic = any((cgs, cgd, cgb, cds))

    @property
    def ispec(self) -> float:
        """Specific (normalisation) current [A]: ``2*n*kp*(W/L)*Vt^2``."""
        return (2.0 * self.n * self.kp * self.w / self.l * self.vt * self.vt
                * self.beta_factor)

    @staticmethod
    def _F(z: float) -> tuple[float, float]:
        """``ln^2(1+e^z)`` and its derivative, overflow-free."""
        u = float(np.logaddexp(0.0, z))
        return u * u, 2.0 * u * float(_sigmoid(z))

    def _eval_symmetric(self, vg: float, vd: float, vs: float, vb: float):
        # EKV is symmetric by construction, so the drain/source exchange of the
        # base class is unnecessary and would only add a redundant branch.
        sg = self._sign
        vgb, vdb, vsb = sg * (vg - vb), sg * (vd - vb), sg * (vs - vb)
        vt2 = 2.0 * self.vt
        vp = (vgb - self.vth) / self.n
        zf, zr = (vp - vsb) / vt2, (vp - vdb) / vt2
        ff, dff = self._F(zf)
        fr, dfr = self._F(zr)
        ispec = self.ispec
        idr = ispec * (ff - fr)
        # Partials with respect to internal, bulk-referred terminal voltages.
        dg = ispec * (dff - dfr) / (2.0 * self.n * self.vt)
        dd = ispec * dfr / vt2
        db = ispec * (dff - dfr) * (1.0 - 1.0 / self.n) / vt2
        if self.lam != 0.0:
            vds = vdb - vsb
            eps = 4.0 * self.vt
            r = math.sqrt(vds * vds + eps * eps)
            clm = 1.0 + self.lam * (r - eps)
            dclm = self.lam * vds / r
            ds = -(dg + dd + db)            # dId/dVsb before CLM
            dg, db = dg * clm, db * clm
            dd = dd * clm + idr * dclm
            ds = ds * clm - idr * dclm
            idr = idr * clm
            # Recompute db from the sum rule to keep the four partials exactly
            # consistent (dclm cancels between drain and source).
            db = -(dg + dd + ds)
        # Node-referred partials equal the internal ones: the internal voltages
        # are sg*(V - Vref), so the two factors of sg cancel in every
        # derivative.  Only the current itself carries the polarity sign.
        return sg * idr, dg, dd, db

    def _eval(self, vgs: float, vds: float, vbs: float):
        # Provided for the base-class API and for direct model queries; the
        # symmetric path above is what the stamps use.  Node potentials that
        # realise the requested internal voltages with the source at zero.
        sg = self._sign
        idr, dg, dd, db = self._eval_symmetric(sg * vgs, sg * vds, 0.0,
                                               sg * vbs)
        return sg * idr, dg, dd, db

    def drain_current(self, vgs: float, vds: float, vbs: float = 0.0) -> float:
        """Drain current [A] at internal ``(vgs, vds, vbs)`` [V]."""
        return self._eval(vgs, vds, vbs)[0]

    def gm_over_id(self, vgs: float, vds: float, vbs: float = 0.0) -> float:
        """``gm/Id`` [1/V] --- the analog design figure of merit.

        Bounded above by ``1/(n*Vt)`` in weak inversion and falling as
        ``2/Vov`` in strong inversion, with a smooth transition.  A kink here
        is the classic symptom of a broken weak-to-strong interpolation, so
        this accessor exists to be swept and differentiated.
        """
        idr, gm, _, _ = self._eval(vgs, vds, vbs)
        return gm / idr if idr != 0.0 else float("nan")


class SubthresholdTFT(_FETBase, _MismatchMixin):
    """Weak-inversion thin-film transistor --- the analog exponential cell.

    ``Id = I0 * (W/L) * exp((Vgs - Vth)/(n*Vt)) * (1 - exp(-Vds/Vt))``

    Parameters
    ----------
    name
        Instance name.
    d, g, s, b
        Node labels; ``b`` defaults to ``s`` (a TFT has no independent body).
    i0
        Prefactor current at ``Vgs = Vth`` and ``W/L = 1`` [A].
    w, l
        Width and length [m].
    vth
        Threshold voltage [V].
    n
        Subthreshold slope factor [-]; swing ``n*ln(10)*kT/q``.
    lam
        Output-conductance coefficient [1/V], applied as ``1 + lam*Vds``
        (default 0, which reproduces the equation above exactly).
    symmetric
        Exchange drain and source when ``Vds < 0`` so the device is
        antisymmetric about ``Vds = 0``.  Leave on; see Notes.
    gmin
        Drain-source parallel conductance [S].  **Default 0**, unlike the
        diode: this device is used at 1 fA to 1 uA, where a 1 pS shunt is a
        100% error at the bottom of the range.  A caller that needs a
        nonsingular Jacobian in the fully-off state should use gmin stepping in
        the outer solver rather than a permanent shunt.
    vth_sigma, beta_sigma
        Mismatch parameters [V] and [-] (assumption A14).  For a subthreshold
        device these are not a refinement: a 20 mV threshold offset is a factor
        ``exp(0.02/(n*Vt))`` = 2.2x current error at ``n = 1``, so a mismatch
        run is the only honest way to predict an analog exponential array.
    T
        Temperature [K].

    Notes
    -----
    **Accuracy over decades is the whole specification.**  These cells
    implement analog ``exp()`` and softmax by construction, so what matters is
    not the value at one bias but the fidelity of the exponential over the
    entire usable current range.  The implementation therefore avoids every
    operation that loses relative precision at small currents: no additive
    floor (``gmin`` defaults to 0), no cancellation (the ``1 - exp(-Vds/Vt)``
    factor is evaluated with :func:`numpy.expm1` so that the small-``Vds`` limit
    stays accurate to full precision instead of cancelling to zero), and the
    exponential itself is exact below :data:`LIMEXP_ARG`.

    **Why the drain factor uses ``Vt`` and the gate factor ``n*Vt``.**  It is
    not a typo, and it is what makes the model asymmetric for ``n != 1``.  The
    gate couples to the channel barrier through the capacitive divider
    ``1/n``, while the drain lowers the barrier for back-injection by the full
    ``Vds``.  Any "symmetrised" version with ``n*Vt`` in both places gets the
    output conductance wrong by a factor ``n``.

    **Reverse operation.**  With ``symmetric=True`` the roles of drain and
    source swap for ``Vds < 0``, which keeps the model bounded (the raw
    formula's ``exp(-Vds/Vt)`` diverges) and is continuous in both value and
    slope at ``Vds = 0``.

    **Limiting.**  This device uses :func:`pnjlim` on ``Vgs``, not
    :func:`fetlim`: the current is exponential in ``Vgs`` with a ~40 mV decade,
    and ``fetlim``'s volt-scale steps allow ``exp(50)`` swings between
    iterations.

    Assumptions: A3, A6, A14.
    """

    def __init__(self, name: str, d: Any, g: Any, s: Any, b: Any | None = None,
                 i0: float = 1e-12, w: float = 1e-6, l: float = 1e-6,
                 vth: float = 0.0, n: float = 1.0, lam: float = 0.0,
                 polarity: str = "n", symmetric: bool = True,
                 cgs: float = 0.0, cgd: float = 0.0, cgb: float = 0.0,
                 cds: float = 0.0, gmin: float = 0.0,
                 vth_sigma: float = 0.0, beta_sigma: float = 0.0,
                 limiting: bool = True, T: float = T_ROOM):
        super().__init__(name, d, g, s, b, polarity=polarity)
        if i0 <= 0.0:
            raise ValueError(f"SubthresholdTFT {name!r}: i0 must be > 0 [A]")
        if w <= 0.0 or l <= 0.0:
            raise ValueError(f"SubthresholdTFT {name!r}: w, l must be > 0 [m]")
        if n <= 0.0:
            raise ValueError(f"SubthresholdTFT {name!r}: n must be > 0")
        self.i0 = float(i0)
        self.w, self.l = float(w), float(l)
        self.vth = float(vth)
        self.n, self.lam = float(n), float(lam)
        self.symmetric = bool(symmetric)
        self.cgs, self.cgd, self.cgb, self.cds = (float(cgs), float(cgd),
                                                  float(cgb), float(cds))
        self.gmin = float(gmin)
        self.limiting = bool(limiting)
        self.T = float(T)
        self.vt = thermal_voltage(self.T)
        self._init_mismatch(self.vth, vth_sigma, beta_sigma)
        self.dynamic = any((cgs, cgd, cgb, cds))

    @property
    def i_pref(self) -> float:
        """Prefactor ``I0*(W/L)`` [A], including current-factor mismatch."""
        return self.i0 * self.w / self.l * self.beta_factor

    @property
    def swing(self) -> float:
        """Subthreshold swing [V/decade]: ``n*ln(10)*kT/q``."""
        return self.n * math.log(10.0) * self.vt

    def _forward(self, vgs: float, vds: float
                 ) -> tuple[float, float, float]:
        """Forward-mode current [A] and ``(dI/dVgs, dI/dVds)`` [S], ``vds >= 0``."""
        nvt = self.n * self.vt
        arg = (vgs - self.vth) / nvt
        e = float(limexp(arg))
        de = float(dlimexp(arg))
        pref = self.i_pref
        z = -vds / self.vt
        if z <= 0.0:
            # -expm1(-x) = 1 - exp(-x), accurate to full relative precision at
            # small x, where the naive difference cancels catastrophically.
            sat = -math.expm1(z)
            dsat = math.exp(z) / self.vt
        else:
            # Only reachable with symmetric=False; limexp keeps it finite.
            sat = 1.0 - float(limexp(z))
            dsat = float(dlimexp(z)) / self.vt
        base = pref * e * sat
        dgs = pref * de * sat / nvt
        dds = pref * e * dsat
        if self.lam != 0.0:
            f = 1.0 + self.lam * vds
            dds = dds * f + base * self.lam
            base, dgs = base * f, dgs * f
        return base, dgs, dds

    def _eval(self, vgs: float, vds: float, vbs: float = 0.0):
        if vds >= 0.0 or not self.symmetric:
            i, gm, gds = self._forward(vgs, vds)
            return i, gm, gds, 0.0
        i, fgs, fds = self._forward(vgs - vds, -vds)
        # Reverse frame: gm_actual = -f_gs, gds_actual = f_gs + f_ds.
        return -i, -fgs, fgs + fds, 0.0

    def drain_current(self, vgs: np.ndarray | float,
                      vds: np.ndarray | float,
                      vbs: float = 0.0) -> np.ndarray | float:
        """Vectorised drain current [A] for internal ``(vgs, vds)`` [V].

        ``vbs`` is accepted for interface compatibility and ignored: a TFT has
        no independent body terminal.  Provided separately from the stamp path
        so that a decade sweep costs one array operation rather than one Python
        call per point.
        """
        vgs = np.asarray(vgs, dtype=float)
        vds = np.asarray(vds, dtype=float)
        fwd = (vds >= 0.0) | (not self.symmetric)
        vgs_e = np.where(fwd, vgs, vgs - vds)
        vds_e = np.where(fwd, vds, -vds)
        i = (self.i_pref * limexp((vgs_e - self.vth) / (self.n * self.vt))
             * -np.expm1(-vds_e / self.vt))
        if self.lam != 0.0:
            i = i * (1.0 + self.lam * vds_e)
        out = np.where(fwd, i, -i)
        return float(out) if np.ndim(vgs) == 0 and np.ndim(vds) == 0 else out

    @property
    def _vcrit(self) -> float:
        """Gate overdrive [V] above which an undamped Newton step diverges."""
        nvt = self.n * self.vt
        ratio = nvt / (math.sqrt(2.0) * self.i_pref)
        # The floor keeps pnjlim's logarithmic branch well defined for an
        # unphysically large prefactor current.
        return max(nvt * math.log(ratio) if ratio > 1.0 else 0.0, 2.0 * nvt)

    def _limit(self, vg: float, vd: float, vs: float) -> tuple[float, float]:
        """Junction-style limiting on *both* barrier voltages, Vgs and Vgd.

        The current is exponential in the gate overdrive at whichever terminal
        is acting as the source, so both ``Vgs - Vth`` and ``Vgd - Vth`` are
        junction-like and both get :func:`pnjlim`.  ``Vds`` is then reconstructed
        from the two limited barriers rather than clamped by :func:`limvds`:
        a volt-scale clamp on ``Vds`` is meaningless for a device whose drain
        factor saturates within ten ``kT/q``, and its hard bounds can return the
        *same* clamp value on two successive iterates, which stalls Newton at a
        non-solution.
        """
        sg = self._sign
        vgs, vds = sg * (vg - vs), sg * (vd - vs)
        if self._vgs_old is not None:
            nvt = self.n * self.vt
            vcrit = self._vcrit
            vgd, vgd_old = vgs - vds, self._vgs_old - self._vds_old
            new_gs = self.vth + pnjlim(vgs - self.vth,
                                       self._vgs_old - self.vth, nvt, vcrit)
            new_gd = self.vth + pnjlim(vgd - self.vth, vgd_old - self.vth,
                                       nvt, vcrit)
            self.limited = (new_gs != vgs) or (new_gd != vgd)
            vgs, vds = new_gs, new_gs - new_gd
        self._vgs_old, self._vds_old = vgs, vds
        return vgs, vds


# ==========================================================================
# Switch
# ==========================================================================
class Switch(Device):
    """Voltage-controlled switch.

    Parameters
    ----------
    name
        Instance name.
    n1, n2
        Switched nodes.
    cp, cn
        Control nodes.
    ron, roff
        On and off resistance [ohm].
    von, voff
        Control voltages [V] at which the switch is fully on and fully off.
    model
        ``"smooth"`` (default) interpolates ``log(g)`` with the SPICE cubic, so
        the conductance and its derivative are continuous and the device has an
        exact Jacobian.  ``"hysteresis"`` is a hard two-state switch with
        memory, which is what a comparator-driven switch physically is but which
        makes the DC problem non-differentiable and possibly multi-valued.

    Notes
    -----
    In smooth mode the device is a genuine nonlinear four-terminal element:
    ``i = g(Vc)*(V(n1) - V(n2))``, so the Jacobian has *cross terms*
    ``di/dVc = g'(Vc)*(V(n1) - V(n2))`` into the control nodes.  Omitting those
    (a common shortcut) still converges when the control node is driven by a
    stiff source and fails silently when it is not.

    Assumptions: A3.
    """

    linear = False

    def __init__(self, name: str, n1: Any, n2: Any, cp: Any, cn: Any,
                 ron: float = 1.0, roff: float = 1e12,
                 von: float = 1.0, voff: float = 0.0,
                 model: str = "smooth"):
        super().__init__(name, (n1, n2, cp, cn))
        if ron <= 0.0 or roff <= 0.0:
            raise ValueError(f"switch {name!r}: ron and roff must be > 0 [ohm]")
        if von == voff:
            raise ValueError(f"switch {name!r}: von and voff must differ [V]")
        if model not in ("smooth", "hysteresis"):
            raise ValueError("model must be 'smooth' or 'hysteresis'")
        self.ron, self.roff = float(ron), float(roff)
        self.von, self.voff = float(von), float(voff)
        self.model = model
        self._closed = False

    def reset_state(self) -> None:
        self._closed = False

    def conductance(self, vc: float) -> tuple[float, float]:
        """Switch conductance [S] and ``dg/dVc`` [S/V] at control voltage ``vc``."""
        gon, goff = 1.0 / self.ron, 1.0 / self.roff
        if self.model == "hysteresis":
            if vc >= max(self.von, self.voff):
                self._closed = self.von > self.voff
            elif vc <= min(self.von, self.voff):
                self._closed = self.von < self.voff
            return (gon if self._closed else goff), 0.0
        lm = 0.5 * math.log(gon * goff)
        lr = math.log(gon / goff)
        span = self.von - self.voff
        u = (vc - 0.5 * (self.von + self.voff)) / span
        if u >= 0.5:
            return gon, 0.0
        if u <= -0.5:
            return goff, 0.0
        g = math.exp(lm + lr * (1.5 * u - 2.0 * u ** 3))
        return g, g * lr * (1.5 - 6.0 * u * u) / span

    def stamp_dc(self, G, I, x, nmap, t=None) -> None:
        a, b, cp, cn = (self._n(nmap, i) for i in range(4))
        v = _v(x, a) - _v(x, b)
        vc = _v(x, cp) - _v(x, cn)
        g, dg = self.conductance(vc)
        # i = g(vc)*v, so the companion source is -dg*v0*vc0: the conductance
        # part is already exact at the present iterate (i0 = g*v0), and only
        # the control-node linearisation needs removing from the RHS.
        _stamp_nl2(G, I, a, b, g, 0.0)
        if dg != 0.0:
            gc = dg * v
            stamp_g(G, a, cp, gc)
            stamp_g(G, a, cn, -gc)
            stamp_g(G, b, cp, -gc)
            stamp_g(G, b, cn, gc)
            stamp_i(I, a, gc * vc)
            stamp_i(I, b, -gc * vc)

    def stamp_ac(self, Y, J, x_op, omega, nmap) -> None:
        a, b, cp, cn = (self._n(nmap, i) for i in range(4))
        v = _v(x_op, a) - _v(x_op, b)
        g, dg = self.conductance(_v(x_op, cp) - _v(x_op, cn))
        for (i, j, val) in ((a, a, g), (a, b, -g), (b, a, -g), (b, b, g)):
            stamp_g(Y, i, j, val)
        if dg != 0.0:
            gc = dg * v
            stamp_g(Y, a, cp, gc)
            stamp_g(Y, a, cn, -gc)
            stamp_g(Y, b, cp, -gc)
            stamp_g(Y, b, cn, gc)

    def op(self, x, nmap):
        a, b, cp, cn = (self._n(nmap, i) for i in range(4))
        v = _v(x, a) - _v(x, b)
        g, dg = self.conductance(_v(x, cp) - _v(x, cn))
        return {"v": v, "i": g * v, "g": g, "r": 1.0 / g, "dg_dvc": dg}


SPICE_PREFIX: dict[str, type[Device]] = {
    "r": Resistor, "c": Capacitor, "l": Inductor, "v": VSource, "i": ISource,
    "e": VCVS, "g": VCCS, "d": Diode, "m": MOSFETL1, "s": Switch,
}
"""First-letter to class map for a SPICE-syntax netlist parser.

``mna.Netlist.from_spice`` owns the parsing; this table only fixes which class
a card letter means, so that the mapping is defined in one place.  ``EKV`` and
``SubthresholdTFT`` have no card letter of their own and are selected through a
model card, since both are ``M`` devices in any real netlist.
"""
