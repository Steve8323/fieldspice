"""Modified nodal analysis --- the lumped-circuit engine.

This is the SPICE half of fieldspice.  A :class:`Netlist` is a bag of two-
terminal and controlled elements plus (optionally) nonlinear compact-model
devices; :class:`MNASolver` turns it into a sparse linear system and solves it
for DC, transient and small-signal AC.

Why modified nodal analysis
---------------------------
Plain nodal analysis writes one KCL equation per node and cannot represent an
ideal voltage source, because such a source has no admittance --- its current
is whatever the rest of the circuit demands.  MNA fixes this by *adding an
unknown*: the branch current of every voltage source and of every inductor
becomes a solved variable, and the corresponding constraint (``v+ - v- = V``,
or ``v+ - v- = L di/dt``) becomes an extra row.  The result is the block system

.. code-block:: text

    [ Y    B ] [ v ]   [ i_src ]
    [ C    D ] [ i ] = [ v_src ]

with ``Y`` the node admittance matrix, ``B``/``C`` the +-1 incidence of the
branch unknowns and ``D`` carrying the reactive part of the branch constraint.
It is indefinite (not positive definite), so it wants an LU factorisation
rather than a Cholesky one, which is why this module uses ``splu`` and not the
conjugate-gradient path the field solvers use.

The connection to the field side of this project is exact rather than
analogical.  ``docs/CONTRACTS.md`` notes that ``G^T M_sigma G phi = i_inject``
*is* Kirchhoff's current law on a resistor mesh: the field discretisation is a
netlist whose nodes are grid nodes and whose resistors and capacitors are the
diagonal entries of the mass matrices.  ``circuit/coupling.py`` exploits that
by condensing a meshed region onto its terminals and stamping the resulting
dense admittance block into the very same matrix this module builds, which is
what :meth:`MNASolver.stamp` exists to expose.

Sign conventions (match SPICE exactly)
--------------------------------------
* A row of the system is KCL at that node written as *sum of currents leaving
  the node through elements = current injected from outside*.
* The branch current of ``V<name> n+ n- ...`` is positive when it flows **into
  the ``n+`` terminal and through the source**, i.e. when the source absorbs
  power.  ``i(V1)`` therefore has the same sign a SPICE ammeter would report,
  and the power *delivered* by the source is ``-v*i``.
* ``I<name> n+ n- value`` pushes ``value`` amperes out of ``n+``, through the
  source, and into ``n-`` --- so a positive value *sinks* current at ``n+``.
  This trips people up constantly and is the SPICE convention, not a choice.

The ground node
---------------
Node ``0`` (also ``gnd``, ``GND``, ``ground``) is the reference and is not
solved.  Its index is ``-1``.  That is not a sentinel to be tested for: the
matrix is assembled at size ``(n+1, n+1)`` with the last row and column
belonging to ground and discarded at the end, so in Python's negative indexing
``A[-1, j]`` *is* the ground row.  Element and device stamping code can
therefore be written branchlessly, with no ``if node is ground`` special case,
and the solution vector handed to a device is likewise augmented with a
trailing zero so that ``x[nmap[node]]`` returns 0 V for ground.

Assumptions
-----------
``A1`` (quasi-static, here taken to its lumped limit: no propagation delay
between elements, no radiation, and every element's terminal current is
instantaneous), ``A3`` (elements are linear and time-invariant unless an
explicit nonlinear device says otherwise), ``A14`` (one nominal instance per
device; mismatch is a compact-model parameter, not a solver feature).

Time integration
----------------
Reactive elements are replaced by *companion models*: a conductance plus a
history current source, derived from the chosen integration rule.  Three rules
are provided and the choice matters:

============  ======  ===========  =========================================
``method``    Order   Stability    Character
============  ======  ===========  =========================================
``"be"``      1       L-stable     Default.  Numerically damps; never rings.
``"trap"``    2       A-stable     Accurate, but rings on a step (see below).
``"gear2"``   2       L-stable     BDF2.  Second order without the ringing.
============  ======  ===========  =========================================

**The trapezoidal ringing artifact.**  Trapezoidal integration has an
amplification factor ``(1 + z/2)/(1 - z/2)`` which tends to ``-1``, not 0, as
``z = lambda*dt -> -inf``.  A stiff mode therefore alternates in sign instead
of dying, so a discontinuous input (a step, a switching event) excites a
sawtooth at the step frequency that decays only as fast as the mode's own
eigenvalue allows.  This is the classic SPICE "trapezoidal ringing", is a
property of the integrator and not of the circuit, and does not shrink under
``dt`` refinement in the first few steps after the discontinuity.  Backward
Euler is L-stable (amplification -> 0) and is the default for exactly this
reason; ``gear2`` buys second-order accuracy back without the artifact.
"""

from __future__ import annotations

import math
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ..solvers.base import ConvergenceError, Result, SolverConfig
from ..units import thermal_voltage

__all__ = [
    "Netlist", "MNASolver", "Element", "parse_value",
    "GROUND_NAMES", "SI_SUFFIXES",
]


# ==========================================================================
# SPICE value parsing
# ==========================================================================
GROUND_NAMES = frozenset({"0", "gnd", "gnd!", "ground"})
"""Node names that mean "the reference node".  Compared case-insensitively."""

SI_SUFFIXES: tuple[tuple[str, float], ...] = (
    # Longest-first: "meg" must be tried before "m", and "mil" before "m".
    ("meg", 1e6),
    ("mil", 25.4e-6),
    ("t", 1e12),
    ("g", 1e9),
    ("k", 1e3),
    ("m", 1e-3),
    ("u", 1e-6),
    ("µ", 1e-6),
    ("n", 1e-9),
    ("p", 1e-12),
    ("f", 1e-15),
)
"""SPICE engineering suffixes, in match order.

**The MEG/M trap.**  In SPICE ``M`` means *milli* and ``MEG`` means *mega*.
Suffixes are case-insensitive, so ``1M`` is 1e-3 and ``1MEG`` is 1e6 --- a
factor of **1e9** between two spellings that differ by two characters.
Writing ``R1 1 0 1M`` when you meant a megohm gives you a milliohm and a
silently wrong answer, which is why :func:`parse_value` is tested on exactly
this case.  Related: ``1MHz`` parses as 1 *milli*-hertz, because ``MHZ``
matches the ``M`` suffix and the leftover ``HZ`` is discarded as a unit
comment.  Both behaviours are what real SPICE does; fieldspice reproduces
them rather than quietly improving on them.

**The trailing-unit trap.**  Any alphabetic tail after the suffix is ignored,
so ``1kohm`` is 1000 and ``2.2uF`` is 2.2e-6.  But ``1F`` is *1 femto*, not
1 farad, and ``1.5e-6F`` is 1.5e-21, because ``F`` is itself a scale factor.
Never write the unit unless a scale factor precedes it.
"""

_NUM_RE = re.compile(
    r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([a-zA-Zµ]*)\s*$")


def parse_value(text: str | float | int) -> float:
    """Parse a SPICE numeric field into strict SI.

    Parameters
    ----------
    text
        A number, or a string such as ``"1k"``, ``"4.7uF"``, ``"1MEG"``,
        ``"-3.3e-9"``.  Case-insensitive.  See :data:`SI_SUFFIXES` for the
        multiplier table and for the two traps it encodes.

    Returns
    -------
    float
        The value in base SI units (ohm, farad, henry, volt, ampere, second,
        hertz --- whichever the caller's context implies; the suffix carries
        only the power of ten).

    Raises
    ------
    ValueError
        If the field is not a number with an optional suffix.

    Examples
    --------
    >>> parse_value("1k"), parse_value("1MEG"), parse_value("1M")
    (1000.0, 1000000.0, 0.001)
    >>> parse_value("4.7u")
    4.7e-06
    """
    if isinstance(text, (int, float, np.floating, np.integer)):
        return float(text)
    m = _NUM_RE.match(str(text))
    if m is None:
        raise ValueError(f"cannot parse numeric value {text!r}")
    mant = float(m.group(1))
    tail = m.group(2).lower()
    if not tail:
        return mant
    for suffix, mult in SI_SUFFIXES:
        if tail.startswith(suffix):
            return mant * mult
    # An unrecognised tail is a unit comment ("1ohm", "5volts"), not an error.
    return mant


def _canon_node(name: Any) -> str:
    """Canonical node name: stripped string, ground aliases folded to ``"0"``."""
    s = str(name).strip()
    if s.lower() in GROUND_NAMES:
        return "0"
    return s


def _is_ground(name: Any) -> bool:
    return str(name).strip().lower() in GROUND_NAMES


# ==========================================================================
# Netlist
# ==========================================================================
@dataclass
class Element:
    """One primitive circuit element.

    A single dataclass covers all seven linear primitives because they differ
    only in how they are stamped, not in what they store.  Nonlinear compact
    models are *not* Elements --- they are Devices (see
    :meth:`Netlist.add_device`).

    Attributes
    ----------
    kind
        One of ``"R"`` [ohm], ``"C"`` [F], ``"L"`` [H], ``"V"`` [V],
        ``"I"`` [A], ``"E"`` (VCVS, gain [V/V]), ``"G"`` (VCCS,
        transconductance [S]).
    name
        Unique instance name.
    nodes
        Canonical node names.  Two for R/C/L/V/I, four for E/G with the
        control pair last: ``(n_plus, n_minus, nc_plus, nc_minus)``.
    value
        Element value in strict SI, or a callable ``f(t) -> float`` [s -> V or
        A] for an independent source.
    ac_mag, ac_phase
        Small-signal drive for :meth:`MNASolver.ac`: magnitude [V or A] and
        phase [degrees].  ``None`` means this source is not an AC stimulus.
    ic
        Initial condition used when ``uic=True``: capacitor voltage [V] or
        inductor current [A].  ``None`` means "start from the operating point".
    """
    kind: str
    name: str
    nodes: tuple[str, ...]
    value: float | Callable[[float], float] = 0.0
    ac_mag: float | None = None
    ac_phase: float = 0.0
    ic: float | None = None

    def value_at(self, t: float) -> float:
        """Element value at time ``t`` [s], evaluating a callable source."""
        v = self.value
        return float(v(t)) if callable(v) else float(v)

    @property
    def has_branch(self) -> bool:
        """True if this element contributes an MNA branch-current unknown."""
        return self.kind in ("V", "L", "E")


class Netlist:
    """A lumped circuit: elements, devices, models and analysis directives.

    Nodes are named by strings (integers are accepted and stringified).  The
    reference node is ``"0"``; ``"gnd"``, ``"GND"`` and ``"ground"`` are
    aliases for it.  Node indices are assigned in order of first appearance,
    which makes the matrix layout reproducible and lets
    :mod:`fieldspice.circuit.coupling` predict where a terminal will land.

    Examples
    --------
    >>> nl = Netlist()
    >>> nl.add_vsource("V1", "in", "0", 10.0)
    >>> nl.add_resistor("R1", "in", "out", 1e3)
    >>> nl.add_resistor("R2", "out", "0", 3e3)
    >>> sol = MNASolver(nl, gmin=0.0).dc()
    >>> round(sol["out"], 12)
    7.5
    """

    def __init__(self) -> None:
        self.elements: list[Element] = []
        self.devices: list[Any] = []
        self.models: dict[str, dict[str, Any]] = {}
        self.analyses: list[dict[str, Any]] = []
        self.ic: dict[str, float] = {}
        self.title: str = ""
        self.meta: dict[str, Any] = {}
        self._nodes: list[str] = []
        self._node_index: dict[str, int] = {}
        self._used_names: set[str] = set()

    # -- node bookkeeping -------------------------------------------------
    @property
    def nodes(self) -> list[str]:
        """Non-ground node names, in matrix-row order."""
        return list(self._nodes)

    @property
    def n_nodes(self) -> int:
        """Number of solved (non-ground) nodes."""
        return len(self._nodes)

    def node_index(self, name: Any) -> int:
        """Matrix row index of a node.

        Parameters
        ----------
        name
            Node name.  Ground aliases return ``-1``.

        Returns
        -------
        int
            Row index in the MNA unknown vector, or ``-1`` for ground.

        Raises
        ------
        ValueError
            If the node is not present in the netlist.  Silently inventing a
            node here would turn a typo into a floating net.
        """
        n = _canon_node(name)
        if n == "0":
            return -1
        if n not in self._node_index:
            raise ValueError(
                f"unknown node {name!r}; netlist has {self._nodes}")
        return self._node_index[n]

    def _register(self, name: Any) -> str:
        n = _canon_node(name)
        if n != "0" and n not in self._node_index:
            self._node_index[n] = len(self._nodes)
            self._nodes.append(n)
        return n

    def _claim_name(self, name: str) -> str:
        key = str(name).strip()
        if not key:
            raise ValueError("element name must be a non-empty string")
        if key.lower() in self._used_names:
            raise ValueError(f"duplicate element name {name!r}")
        self._used_names.add(key.lower())
        return key

    # -- element constructors ---------------------------------------------
    def _add(self, kind: str, name: str, nodes: Sequence[Any],
             value: float | Callable[[float], float],
             **kw: Any) -> Element:
        el = Element(kind=kind, name=self._claim_name(name),
                     nodes=tuple(self._register(n) for n in nodes),
                     value=value, **kw)
        self.elements.append(el)
        return el

    def add_resistor(self, name: str, n1: Any, n2: Any, r: float) -> Element:
        """Add a resistor of ``r`` ohm between nodes ``n1`` and ``n2``.

        Raises
        ------
        ValueError
            If ``r`` is zero.  A zero-ohm resistor is not a short circuit in
            MNA, it is a division by zero; model an ideal short with a 0 V
            voltage source, which adds the branch-current unknown that makes
            the constraint representable.
        """
        r = float(parse_value(r))
        if r == 0.0:
            raise ValueError(
                f"resistor {name!r} has zero resistance; use a 0 V source "
                "for an ideal short so the branch current stays an unknown")
        if not math.isfinite(r):
            raise ValueError(f"resistor {name!r} value must be finite")
        return self._add("R", name, (n1, n2), r)

    def add_capacitor(self, name: str, n1: Any, n2: Any, c: float,
                      ic: float | None = None) -> Element:
        """Add a capacitor of ``c`` farad, optional initial voltage ``ic`` [V]."""
        c = float(parse_value(c))
        if c <= 0.0 or not math.isfinite(c):
            raise ValueError(f"capacitor {name!r} needs a positive finite value")
        return self._add("C", name, (n1, n2), c,
                         ic=None if ic is None else float(ic))

    def add_inductor(self, name: str, n1: Any, n2: Any, l: float,
                     ic: float | None = None) -> Element:
        """Add an inductor of ``l`` henry, optional initial current ``ic`` [A]."""
        l = float(parse_value(l))
        if l <= 0.0 or not math.isfinite(l):
            raise ValueError(f"inductor {name!r} needs a positive finite value")
        return self._add("L", name, (n1, n2), l,
                         ic=None if ic is None else float(ic))

    def add_vsource(self, name: str, n1: Any, n2: Any,
                    value: float | Callable[[float], float],
                    ac: float | None = None,
                    ac_phase: float = 0.0) -> Element:
        """Add an independent voltage source.

        Parameters
        ----------
        name
            Instance name.
        n1, n2
            Positive and negative terminal node names.
        value
            DC value [V], or a callable ``f(t) -> float`` with ``t`` in
            seconds.  The callable is evaluated at the *new* time point of
            each implicit step.
        ac
            Small-signal magnitude [V] for :meth:`MNASolver.ac`.
        ac_phase
            Small-signal phase [degrees].
        """
        if not callable(value):
            value = float(parse_value(value))
        return self._add("V", name, (n1, n2), value,
                         ac_mag=None if ac is None else float(ac),
                         ac_phase=float(ac_phase))

    def add_isource(self, name: str, n1: Any, n2: Any,
                    value: float | Callable[[float], float],
                    ac: float | None = None,
                    ac_phase: float = 0.0) -> Element:
        """Add an independent current source [A], flowing ``n1 -> n2`` inside
        the source (so a positive value *removes* current from ``n1``)."""
        if not callable(value):
            value = float(parse_value(value))
        return self._add("I", name, (n1, n2), value,
                         ac_mag=None if ac is None else float(ac),
                         ac_phase=float(ac_phase))

    def add_vcvs(self, name: str, n1: Any, n2: Any, nc1: Any, nc2: Any,
                 gain: float) -> Element:
        """Voltage-controlled voltage source: ``v(n1,n2) = gain*v(nc1,nc2)``."""
        return self._add("E", name, (n1, n2, nc1, nc2), float(parse_value(gain)))

    def add_vccs(self, name: str, n1: Any, n2: Any, nc1: Any, nc2: Any,
                 gm: float) -> Element:
        """Voltage-controlled current source: ``i(n1->n2) = gm*v(nc1,nc2)`` [S]."""
        return self._add("G", name, (n1, n2, nc1, nc2), float(parse_value(gm)))

    def add_device(self, dev: Any) -> Any:
        """Attach a nonlinear compact-model device.

        ``dev`` must satisfy the ``circuit/devices.py`` protocol, which this
        module consumes by duck typing so that the two files can be written
        independently:

        ``dev.nodes``
            Tuple of node names.
        ``dev.stamp_dc(G, I, x, nmap)``
            Stamp the Newton companion model linearised about ``x``:
            conductances into ``G``, equivalent current sources into ``I``,
            such that solving ``G x_new = I`` produces the next Newton
            iterate.  This is the SPICE convention (``geq``/``Ieq``), not a
            residual/Jacobian delta form.
        ``dev.stamp_tran(G, I, x, x_prev, dt, nmap)``
            The same, including charge-storage companion terms for a step of
            size ``dt`` [s] from state ``x_prev``.

        ``G`` is a ``scipy.sparse.lil_matrix`` of shape ``(n+1, n+1)`` and
        ``I`` is a ``float64`` array of length ``n+1``.  The trailing row and
        column are the **ground rail**: ``nmap`` maps ground to ``-1``, which
        in an ``(n+1)``-sized matrix *is* that row, so a device may either skip
        negative indices (what ``devices.stamp_g`` does) or stamp them blindly
        and have them discarded.  ``x`` and ``x_prev`` are likewise length
        ``n+1`` with a trailing zero, so ``x[nmap[node]]`` reads 0 V at ground.

        Optional hooks, all used if present and all part of the
        ``devices.Device`` interface: ``extra_unknowns()`` (names of private
        branch unknowns, appended to the unknown vector and registered in
        ``nmap`` **verbatim** so the device can find them again),
        ``stamp_ac(Y, J, x_op, omega, nmap)`` (complete complex small-signal
        stamp; a device that has one does *not* also get its DC conductance
        stamped, or the real part would be counted twice),
        ``set_time(t)``, ``reset_state()`` (before each Newton solve),
        ``accept_timestep(x, x_prev, dt, nmap)`` (after each converged step),
        and the ``linear`` flag (a netlist whose devices are all linear skips
        Newton entirely).
        """
        if not hasattr(dev, "nodes"):
            raise ValueError("device must expose a 'nodes' attribute")
        for missing in ("stamp_dc", "stamp_tran"):
            if not hasattr(dev, missing):
                raise ValueError(f"device is missing required {missing}()")
        dev.nodes = tuple(self._register(n) for n in dev.nodes)
        name = getattr(dev, "name", None) or f"dev{len(self.devices)}"
        dev.name = self._claim_name(name)
        self.devices.append(dev)
        return dev

    # -- introspection ----------------------------------------------------
    def by_name(self, name: str) -> Element:
        """Look up an element by (case-insensitive) name."""
        key = str(name).strip().lower()
        for el in self.elements:
            if el.name.lower() == key:
                return el
        raise ValueError(f"no element named {name!r}")

    def __len__(self) -> int:
        return len(self.elements) + len(self.devices)

    def __repr__(self) -> str:
        kinds = "".join(sorted(el.kind for el in self.elements))
        return (f"<Netlist {len(self.elements)} elements [{kinds}], "
                f"{len(self.devices)} devices, {self.n_nodes} nodes>")

    def summary(self) -> str:
        """Human-readable listing, one line per element."""
        lines = [repr(self)]
        for el in self.elements:
            v = el.value if not callable(el.value) else "f(t)"
            lines.append(f"  {el.kind} {el.name:<8} {' '.join(el.nodes):<20} {v}")
        for d in self.devices:
            lines.append(f"  D {getattr(d, 'name', '?'):<8} "
                         f"{' '.join(d.nodes)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # SPICE front end
    # ------------------------------------------------------------------
    @classmethod
    def from_spice(cls, text: str) -> "Netlist":
        """Build a netlist from a useful subset of SPICE deck syntax.

        Supported cards
        ---------------
        ============================================  ==========================
        ``R<n> n1 n2 value``                          resistor [ohm]
        ``C<n> n1 n2 value [IC=v]``                   capacitor [F]
        ``L<n> n1 n2 value [IC=i]``                   inductor [H]
        ``V<n> n+ n- [DC] val [AC m [ph]] [wave]``    voltage source
        ``I<n> n+ n- [DC] val [AC m [ph]] [wave]``    current source
        ``E<n> n+ n- nc+ nc- gain``                   VCVS
        ``G<n> n+ n- nc+ nc- gm``                     VCCS
        ``D<n> anode cathode model [area]``           diode
        ``M<n> d g s b model [W=..] [L=..]``          MOSFET
        ``.model name type (p=v ...)``                model card
        ``.tran tstep tstop [tstart [tmax]] [uic]``   transient directive
        ``.ac dec|oct|lin np fstart fstop``           AC directive
        ``.dc src start stop incr``                   DC sweep directive
        ``.ic v(node)=value ...``                     initial conditions
        ``.end``, ``.title``, ``.options``            accepted, mostly ignored
        ============================================  ==========================

        ``wave`` is one of ``PULSE(v1 v2 td tr tf pw per)``,
        ``SIN(vo va freq [td [theta]])`` or ``PWL(t1 v1 t2 v2 ...)``.
        Continuation lines start with ``+``.  ``*`` starts a full-line comment
        and ``;`` a trailing one.  Everything is case-insensitive; numeric
        fields go through :func:`parse_value`, so read the ``MEG``-versus-``M``
        warning there before typing a megohm.

        Deliberate deviations
        ---------------------
        A real SPICE deck treats its **first line as a title** and discards it
        unconditionally, which silently deletes the first element of any
        snippet that lacks one.  fieldspice does not do that: comment your
        title with ``*`` or declare it with ``.title``.  Every non-comment line
        is parsed.

        Element and node **names keep the case you typed** and are matched
        case-sensitively, so ``V1`` stays ``V1`` and ``result.scalars["i(V1)"]``
        is the key you expect.  Real SPICE folds everything to one case, which
        would silently rename your nodes.  Keywords (``DC``, ``AC``, ``PULSE``,
        ``IC=``), directives, model names and engineering suffixes remain
        case-insensitive.  The one thing this costs you: ``out`` and ``OUT``
        are two different nodes here, and one of them will end up floating,
        which the singular-matrix message will tell you about loudly.

        Not supported
        -------------
        ``K`` (mutual inductance), ``T``/``O``/``U`` (transmission lines),
        ``S``/``W`` (switches), ``X``/``.subckt`` (hierarchy),
        ``B`` (behavioural), ``.func``, ``.param``, ``.include``.  Each raises
        ``ValueError`` naming the card rather than being skipped, because a
        silently dropped element is a wrong answer that looks like a right one.

        Parameters
        ----------
        text
            The deck.

        Returns
        -------
        Netlist

        Raises
        ------
        ValueError
            On any unparseable or unsupported card, with the line quoted.
        """
        nl = cls()
        for lineno, line in _logical_lines(text):
            try:
                nl._parse_card(line)
            except ValueError as exc:
                raise ValueError(f"line {lineno}: {line!r}: {exc}") from None
        return nl

    # -- parser internals -------------------------------------------------
    def _parse_card(self, line: str) -> None:
        if line.startswith("."):
            self._parse_dot(line)
            return
        tok = line.split()
        letter, name = tok[0][0].lower(), tok[0]
        handler = {
            "r": self._card_r, "c": self._card_c, "l": self._card_l,
            "v": self._card_src, "i": self._card_src,
            "e": self._card_ctrl, "g": self._card_ctrl,
            "d": self._card_diode, "m": self._card_mos,
        }.get(letter)
        if handler is None:
            raise ValueError(
                f"unsupported element card {letter.upper()!r} "
                "(supported: R C L V I E G D M)")
        handler(name, tok)

    def _card_r(self, name: str, tok: list[str]) -> None:
        if len(tok) < 4:
            raise ValueError("expected 'Rname n1 n2 value'")
        self.add_resistor(name, tok[1], tok[2], parse_value(tok[3]))

    def _card_c(self, name: str, tok: list[str]) -> None:
        if len(tok) < 4:
            raise ValueError("expected 'Cname n1 n2 value [IC=v]'")
        self.add_capacitor(name, tok[1], tok[2], parse_value(tok[3]),
                           ic=_find_ic(tok[4:]))

    def _card_l(self, name: str, tok: list[str]) -> None:
        if len(tok) < 4:
            raise ValueError("expected 'Lname n1 n2 value [IC=i]'")
        self.add_inductor(name, tok[1], tok[2], parse_value(tok[3]),
                          ic=_find_ic(tok[4:]))

    def _card_ctrl(self, name: str, tok: list[str]) -> None:
        if len(tok) < 6:
            raise ValueError("expected '<name> n+ n- nc+ nc- gain'")
        add = self.add_vcvs if name[0].lower() == "e" else self.add_vccs
        add(name, tok[1], tok[2], tok[3], tok[4], parse_value(tok[5]))

    def _card_src(self, name: str, tok: list[str]) -> None:
        if len(tok) < 3:
            raise ValueError("expected '<name> n+ n- <value spec>'")
        # The value field holds only keywords and numbers, never identifiers,
        # so it is safe to case-fold; node and element names are not.
        dc, wave, ac_mag, ac_ph = _parse_source_spec(" ".join(tok[3:]).lower())
        value: float | Callable[[float], float] = wave if wave is not None else dc
        add = self.add_vsource if name[0].lower() == "v" else self.add_isource
        add(name, tok[1], tok[2], value, ac=ac_mag, ac_phase=ac_ph)

    def _card_diode(self, name: str, tok: list[str]) -> None:
        if len(tok) < 4:
            raise ValueError("expected 'Dname anode cathode model [area]'")
        params = dict(self.models.get(tok[3].lower(), {}))
        params.pop("TYPE", None)
        params["AREA"] = parse_value(tok[4]) if len(tok) > 4 else 1.0
        anode, cathode = tok[1], tok[2]
        if _external_class("Diode") is None and float(params.get("RS", 0.0)) > 0:
            # The fallback junction has no series resistance, so give it one
            # explicitly.  devices.Diode folds rs in exactly (a series
            # combination of conductances) and needs no extra node, so this
            # branch is taken only when devices.py is unavailable.
            rs = float(params.pop("RS")) / float(params["AREA"])
            inner = f"{name}#internal"
            self.add_resistor(f"{name}#rs", anode, inner, rs)
            anode = inner
        self.add_device(_make_device("Diode", _FallbackDiode, name,
                                     (anode, cathode), params))

    def _card_mos(self, name: str, tok: list[str]) -> None:
        if len(tok) < 6:
            raise ValueError("expected 'Mname d g s b model [W=..] [L=..]'")
        params = dict(self.models.get(tok[5].lower(), {}))
        mtype = str(params.pop("TYPE", "nmos")).lower()
        for kv in tok[6:]:
            if "=" in kv:
                k, v = kv.split("=", 1)
                params[k.strip().upper()] = parse_value(v)
        params.setdefault("POLARITY", "n" if mtype.startswith("n") else "p")
        self.add_device(_make_device("MOSFETL1", _FallbackMOSFET, name,
                                     tuple(tok[1:5]), params))

    def _parse_dot(self, line: str) -> None:
        tok = line.split()
        low = [t.lower() for t in tok]
        card = low[0]
        if card == ".model":
            if len(tok) < 3:
                raise ValueError("expected '.model name type (params)'")
            # Split on whitespace only twice: the parameter list may be glued
            # to the type token as ".model dx d(is=1e-14)", which a plain
            # token split would hide.
            body = line.split(None, 2)[2]
            mtype = re.match(r"\s*([A-Za-z0-9_]+)", body)
            params: dict[str, Any] = {
                "TYPE": mtype.group(1).lower() if mtype else "d"}
            for k, v in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
                                   r"([^\s,()]+)", body):
                params[k.upper()] = parse_value(v)
            self.models[tok[1].lower()] = params
        elif card == ".tran":
            if len(tok) < 3:
                raise ValueError("expected '.tran tstep tstop [tstart [tmax]]'")
            self.analyses.append({
                "type": "tran",
                "tstep": parse_value(tok[1]), "tstop": parse_value(tok[2]),
                "tstart": parse_value(tok[3]) if len(tok) > 3 else 0.0,
                "uic": "uic" in low})
        elif card == ".ac":
            if len(tok) < 5:
                raise ValueError("expected '.ac dec|oct|lin np fstart fstop'")
            self.analyses.append({
                "type": "ac", "spacing": low[1], "points": int(float(tok[2])),
                "fstart": parse_value(tok[3]), "fstop": parse_value(tok[4])})
        elif card == ".dc":
            if len(tok) < 5:
                raise ValueError("expected '.dc source start stop incr'")
            self.analyses.append({
                "type": "dc", "source": tok[1], "start": parse_value(tok[2]),
                "stop": parse_value(tok[3]), "incr": parse_value(tok[4])})
        elif card == ".ic":
            for k, v in re.findall(r"[vV]\s*\(\s*([^)\s]+)\s*\)\s*=\s*(\S+)",
                                   line):
                self.ic[_canon_node(k)] = parse_value(v)
        elif card == ".title":
            self.title = line[6:].strip()
        elif card in (".end", ".ends", ".options", ".option", ".op",
                      ".print", ".plot", ".probe", ".temp", ".width",
                      ".save", ".control", ".endc"):
            pass
        else:
            raise ValueError(f"unsupported directive {card!r}")


def _logical_lines(text: str) -> list[tuple[int, str]]:
    """Split a deck into (line number, logical line), joining ``+`` continuations."""
    out: list[tuple[int, str]] = []
    for i, raw in enumerate(str(text).splitlines(), start=1):
        line = raw.split(";", 1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith("*"):
            continue
        stripped = line.strip()
        if stripped.startswith("+"):
            if not out:
                raise ValueError(f"line {i}: continuation with nothing to continue")
            n, prev = out[-1]
            out[-1] = (n, prev + " " + stripped[1:].strip())
        else:
            out.append((i, stripped))
    return out


def _find_ic(tokens: Sequence[str]) -> float | None:
    for t in tokens:
        if t.lower().startswith("ic="):
            return parse_value(t.split("=", 1)[1])
    return None


_FUNC_RE = re.compile(r"\b(pulse|sine|sin|pwl|dc|exp)\s*\(([^)]*)\)")


def _parse_source_spec(rest: str
                       ) -> tuple[float, Callable[[float], float] | None,
                                  float | None, float]:
    """Parse the value field of a V/I card.

    Returns
    -------
    (dc_value [V or A], waveform callable or None, ac magnitude or None,
     ac phase [degrees])
    """
    rest = rest.strip()
    wave: Callable[[float], float] | None = None
    dc = 0.0
    ac_mag: float | None = None
    ac_ph = 0.0

    def _nums(body: str) -> list[float]:
        return [parse_value(s) for s in re.split(r"[,\s]+", body.strip()) if s]

    for m in list(_FUNC_RE.finditer(rest)):
        fn, body = m.group(1), m.group(2)
        vals = _nums(body)
        if fn == "dc":
            dc = vals[0] if vals else 0.0
        elif fn == "pulse":
            wave = _spice_pulse(vals)
        elif fn in ("sin", "sine"):
            wave = _spice_sin(vals)
        elif fn == "pwl":
            if len(vals) < 4 or len(vals) % 2:
                raise ValueError("PWL needs an even number of (t, v) values")
            wave = _spice_pwl(vals[0::2], vals[1::2])
        elif fn == "exp":
            raise ValueError("EXP source waveform is not supported")
        rest = rest[:m.start()] + " " * (m.end() - m.start()) + rest[m.end():]

    tok = rest.split()
    i = 0
    while i < len(tok):
        t = tok[i]
        if t == "dc":
            dc = parse_value(tok[i + 1]); i += 2
        elif t == "ac":
            ac_mag = 1.0
            if i + 1 < len(tok) and _looks_numeric(tok[i + 1]):
                ac_mag = parse_value(tok[i + 1]); i += 1
                if i + 1 < len(tok) and _looks_numeric(tok[i + 1]):
                    ac_ph = parse_value(tok[i + 1]); i += 1
            i += 1
        elif _looks_numeric(t):
            dc = parse_value(t); i += 1
        else:
            raise ValueError(f"unrecognised source field {t!r}")
    if wave is not None:
        # SPICE evaluates the transient waveform, not the DC value, during a
        # .tran run; the DC field only sets the operating point.
        dc = wave(0.0)
    return dc, wave, ac_mag, ac_ph


def _looks_numeric(tok: str) -> bool:
    return _NUM_RE.match(tok) is not None


def _spice_pulse(v: Sequence[float]) -> Callable[[float], float]:
    """``PULSE(v1 v2 td tr tf pw per)`` -> callable, via :mod:`fieldspice.sources`."""
    p = list(v) + [0.0] * (7 - len(v))
    v1, v2, td, tr, tf, pw, per = p[:7]
    from ..sources import pulse
    return pulse(t0=td, width=pw, v0=v1, v1=v2, trise=tr, tfall=tf,
                 period=(per if per > 0 else None))


def _spice_sin(v: Sequence[float]) -> Callable[[float], float]:
    """``SIN(vo va freq [td [theta]])`` -> callable.

    Written out here rather than reusing :func:`fieldspice.sources.sine`
    because the SPICE form carries a delay and an exponential damping factor
    ``theta`` [1/s] that the generic helper does not.
    """
    p = list(v) + [0.0] * (5 - len(v))
    vo, va, freq, td, theta = p[:5]

    def f(t: float | np.ndarray) -> Any:
        tt = np.asarray(t, dtype=float)
        tau = tt - td
        out = np.where(
            tau < 0.0, vo,
            vo + va * np.exp(-np.maximum(tau, 0.0) * theta)
            * np.sin(2.0 * np.pi * freq * np.maximum(tau, 0.0)))
        return float(out) if np.ndim(t) == 0 else out

    return f


def _spice_pwl(times: Sequence[float],
               values: Sequence[float]) -> Callable[[float], float]:
    from ..sources import pwl
    return pwl(times, values)


SPICE_PARAM_ALIAS: dict[str, str] = {
    # SPICE model-card keyword -> compact-model constructor keyword.
    "IS": "isat", "ISAT": "isat", "N": "n", "RS": "rs",
    "CJO": "cj0", "CJ0": "cj0", "VJ": "vj", "M": "m", "FC": "fc",
    "TT": "tt", "BV": "bv", "IBV": "ibv", "NBV": "nbv",
    "AREA": "area", "TEMP": "T", "T": "T",
    "VTO": "vth", "VT0": "vth", "VTH": "vth", "KP": "kp",
    "W": "w", "L": "l", "LAMBDA": "lam", "LAM": "lam",
    "GAMMA": "gamma", "PHI": "phi", "NSUB": "n_sub",
    "CGS": "cgs", "CGD": "cgd", "CGB": "cgb", "CDS": "cds",
    "POLARITY": "polarity", "GMIN": "gmin",
}
"""Translation from SPICE model-card spelling to compact-model keywords.

The two vocabularies genuinely differ (``IS`` versus ``isat``, ``VTO`` versus
``vth``) and ``devices.py`` owns the second one.  Anything not in this table
and not accepted by the constructor is **warned about**, never silently
dropped --- a discarded ``IS`` leaves a diode running on its default
saturation current, which is a plausible-looking wrong answer.
"""


def _external_class(name: str) -> type | None:
    """Fetch ``circuit.devices.<name>``, or ``None`` if unavailable.

    ``devices.py`` is the canonical home of compact models and is written
    independently of this file, so the SPICE front end reaches for it first and
    falls back to the minimal models defined here only when it is absent.  The
    fallbacks exist so that ``mna.py`` is self-contained and self-testable, not
    to compete with ``devices.py``.
    """
    try:
        from . import devices as _devmod
        cls = getattr(_devmod, name, None)
    except Exception:
        return None
    return cls if isinstance(cls, type) else None


def _make_device(preferred: str, fallback: type, name: str,
                 nodes: Sequence[str], params: Mapping[str, Any]) -> Any:
    """Build a compact model from SPICE-spelled parameters.

    Parameters are translated through :data:`SPICE_PARAM_ALIAS`, then filtered
    against the constructor's actual signature.  Anything left over raises a
    ``RuntimeWarning`` naming the parameter and the model.
    """
    import inspect

    cls = _external_class(preferred) or fallback
    kw: dict[str, Any] = {}
    unknown: list[str] = []
    try:
        accepted = set(inspect.signature(cls.__init__).parameters)
    except (TypeError, ValueError):
        accepted = set()
    for key, val in params.items():
        target = SPICE_PARAM_ALIAS.get(str(key).upper(), str(key).lower())
        if target in accepted or not accepted:
            kw[target] = val
        elif str(key).lower() in accepted:
            kw[str(key).lower()] = val
        else:
            unknown.append(str(key))
    if unknown:
        warnings.warn(
            f"{preferred} {name!r}: model parameters {sorted(unknown)} are not "
            f"understood by {cls.__module__}.{cls.__name__} and were ignored",
            RuntimeWarning, stacklevel=3)
    try:
        return cls(name, *nodes, **kw)
    except TypeError as exc:
        if cls is fallback:
            raise
        warnings.warn(f"{cls.__name__} rejected {kw} ({exc}); "
                      f"using the built-in fallback model",
                      RuntimeWarning, stacklevel=3)
        return _make_device(preferred, fallback, name, nodes,
                            {k: v for k, v in params.items()})


def _extra_unknowns(dev: Any) -> tuple[str, ...]:
    """Names of a device's private unknowns.

    ``devices.Device`` exposes this as a *method*; a simpler model may expose
    it as a plain attribute.  Both are accepted, and the names are registered
    verbatim so that ``devices._branch_index`` finds them.
    """
    ex = getattr(dev, "extra_unknowns", ())
    if callable(ex):
        ex = ex()
    return tuple(str(e) for e in ex)


def _dev_call(dev: Any, hook: str, *args: Any) -> bool:
    """Call an optional device hook.  Returns True if the hook existed."""
    fn = getattr(dev, hook, None)
    if fn is None or not callable(fn):
        return False
    fn(*args)
    return True


# ==========================================================================
# Minimal fallback compact models
# ==========================================================================
def _param(params: Mapping[str, Any], key: str, default: float) -> float:
    """Case-insensitive model-parameter lookup, since SPICE decks are unruly."""
    for k, v in params.items():
        if k.lower() == key.lower():
            return float(v)
    return float(default)


class _FallbackDiode:
    """Shockley junction, used only when ``circuit/devices.py`` is unavailable.

    ``I = area*IS*(exp(Vd/(N*Vt)) - 1)`` with the exponent clipped to +-400 as
    ``docs/CONTRACTS.md`` requires, plus SPICE junction limiting (``pnjlim``)
    on the iteration-to-iteration voltage step.  No charge storage, no
    breakdown, no series resistance (the parser inserts an explicit resistor
    for ``RS``).  Units: ``IS`` [A], ``N`` dimensionless, ``T`` [K].
    """

    def __init__(self, name: str, anode: str, cathode: str,
                 **params: Any) -> None:
        IS = _param(params, "IS", 1e-14)
        N = _param(params, "N", 1.0)
        T = _param(params, "T", 300.0)
        area = _param(params, "area", 1.0)
        self.name = name
        self.nodes = (anode, cathode)
        self.Is = float(IS) * float(area)
        self.n = float(N)
        self.vt = thermal_voltage(float(T)) * self.n
        self.vcrit = self.vt * math.log(self.vt / (math.sqrt(2.0) * self.Is))
        self._vd_last: float | None = None

    def reset(self) -> None:
        """Forget the junction-limiting state (called before each Newton solve)."""
        self._vd_last = None

    def _linearise(self, vd: float) -> tuple[float, float]:
        if self._vd_last is not None:
            vd = _pnjlim(vd, self._vd_last, self.vt, self.vcrit)
        self._vd_last = vd
        ex = math.exp(min(max(vd / self.vt, -400.0), 400.0))
        i = self.Is * (ex - 1.0)
        g = self.Is * ex / self.vt
        return g, i - g * vd          # (conductance [S], equivalent source [A])

    def stamp_dc(self, G: Any, I: np.ndarray, x: np.ndarray,
                 nmap: Mapping[str, int]) -> None:
        a, c = nmap[self.nodes[0]], nmap[self.nodes[1]]
        g, ieq = self._linearise(float(x[a] - x[c]))
        G[a, a] += g
        G[c, c] += g
        G[a, c] -= g
        G[c, a] -= g
        I[a] -= ieq
        I[c] += ieq

    def stamp_tran(self, G: Any, I: np.ndarray, x: np.ndarray,
                   x_prev: np.ndarray, dt: float,
                   nmap: Mapping[str, int]) -> None:
        self.stamp_dc(G, I, x, nmap)


def _pnjlim(vnew: float, vold: float, vt: float, vcrit: float) -> float:
    """SPICE ``pnjlim`` junction limiting.

    Newton on an exponential diverges wildly if an early iterate overshoots:
    one extra volt is 17 decades of current.  This damps the *voltage* step
    logarithmically once the junction is forward biased past ``vcrit``, which
    is the standard cure and converges far more reliably than a fixed step
    clamp.
    """
    if vnew > vcrit and abs(vnew - vold) > 2.0 * vt:
        if vold > 0.0:
            arg = 1.0 + (vnew - vold) / vt
            return vold + vt * math.log(arg) if arg > 0.0 else vcrit
        return vt * math.log(max(vnew, 1e-30) / vt)
    return vnew


class _FallbackMOSFET:
    """Level-1 (Shichman-Hodges) MOSFET, used only without ``devices.py``.

    Square-law strong inversion with channel-length modulation, hard cutoff
    below threshold, no subthreshold conduction, no capacitances, no body
    effect.  ``VTO`` [V], ``KP`` [A/V^2], ``W``/``L`` [m], ``LAMBDA`` [1/V].
    Bulk is accepted and ignored.  This is a placeholder good enough to let a
    parsed ``M`` card run; ``circuit/devices.py`` owns the real models.
    """

    def __init__(self, name: str, d: str, g: str, s: str, b: str,
                 **params: Any) -> None:
        self.name = name
        self.nodes = (d, g, s, b)
        self.vth = _param(params, "VTO", 0.7)
        self.beta = (_param(params, "KP", 2e-5) * _param(params, "W", 1e-5)
                     / _param(params, "L", 1e-6))
        self.lam = _param(params, "LAMBDA", 0.0)
        self.pol = _param(params, "POLARITY", 1.0)

    def reset(self) -> None:
        return None

    def stamp_dc(self, G: Any, I: np.ndarray, x: np.ndarray,
                 nmap: Mapping[str, int]) -> None:
        d, g, s = (nmap[self.nodes[i]] for i in range(3))
        p = self.pol
        vgs = p * float(x[g] - x[s])
        vds = p * float(x[d] - x[s])
        rev = vds < 0.0
        if rev:                      # symmetric device: swap source and drain
            d, s = s, d
            vgs, vds = vgs - vds, -vds
        vov = vgs - self.vth
        if vov <= 0.0:
            ids = gm = gds = 0.0
        elif vds < vov:
            ids = self.beta * (vov * vds - 0.5 * vds * vds)
            gm = self.beta * vds
            gds = self.beta * (vov - vds)
        else:
            ids = 0.5 * self.beta * vov * vov * (1.0 + self.lam * vds)
            gm = self.beta * vov * (1.0 + self.lam * vds)
            gds = 0.5 * self.beta * vov * vov * self.lam
        ieq = ids - gm * vgs - gds * vds
        G[d, d] += gds
        G[s, s] += gds + gm
        G[d, s] -= gds + gm
        G[s, d] -= gds
        G[d, g] += gm
        G[s, g] -= gm
        I[d] -= p * ieq
        I[s] += p * ieq

    def stamp_tran(self, G: Any, I: np.ndarray, x: np.ndarray,
                   x_prev: np.ndarray, dt: float,
                   nmap: Mapping[str, int]) -> None:
        self.stamp_dc(G, I, x, nmap)


# ==========================================================================
# Solver
# ==========================================================================
_METHOD_ALIAS = {
    "be": "be", "beuler": "be", "backward_euler": "be", "euler": "be",
    "trap": "trap", "tr": "trap", "trapezoidal": "trap",
    "gear2": "gear2", "gear": "gear2", "bdf2": "gear2",
}


class MNASolver:
    """Modified nodal analysis: DC, transient and AC on a :class:`Netlist`.

    Unlike the field solvers this class does **not** derive from
    :class:`fieldspice.solvers.base.SolverBase`, because that base is built
    around a :class:`~fieldspice.grid.RectilinearGrid` and an
    :class:`~fieldspice.operators.Operators` bundle and a lumped netlist has
    neither.  It keeps the same observable contract --- ``name``,
    ``assumptions``, and a :class:`~fieldspice.solvers.base.Result` carrying
    provenance in ``meta`` --- so downstream code (monitors, io, coupling)
    treats it identically.  Results are returned with ``grid=None``.

    Parameters
    ----------
    netlist
        The circuit.
    config
        Shared solver knobs; ``max_newton`` and ``verbose`` are honoured.
    gmin
        Conductance [S] added from every node to ground.  This is the
        ``gshunt`` form of SPICE's ``gmin``: it is what makes a node that
        touches nothing but capacitors solvable at DC (where capacitors are
        open circuits and the node would otherwise float, giving a singular
        matrix).  It perturbs every answer by roughly ``gmin*R`` in relative
        terms --- 1e-9 for a 1 kohm circuit at the 1e-12 default --- so set it
        to 0 when you want an exactly-computable linear answer and there are
        no floating nodes.
    reltol, vntol, abstol
        Newton convergence tolerances, SPICE-named: relative, absolute on node
        voltages [V], absolute on branch currents [A].
    temperature
        Ambient temperature [K] passed to fallback compact models.

    Attributes
    ----------
    node_map, branch_map
        ``{name: row index}``.  Ground maps to ``-1``.
    n
        Total number of unknowns (nodes + branch currents).
    """

    name: str = "mna"
    assumptions: tuple[str, ...] = ("A1", "A3", "A14")

    def __init__(self, netlist: Netlist, config: SolverConfig | None = None,
                 gmin: float = 1e-12, reltol: float = 1e-3,
                 vntol: float = 1e-6, abstol: float = 1e-12,
                 temperature: float = 300.0) -> None:
        if not isinstance(netlist, Netlist):
            raise ValueError("netlist must be a fieldspice Netlist")
        if netlist.n_nodes == 0:
            raise ValueError("netlist has no non-ground nodes")
        if gmin < 0.0:
            raise ValueError("gmin must be non-negative")
        self.netlist = netlist
        self.cfg = config or SolverConfig()
        self.gmin = float(gmin)
        self.reltol = float(reltol)
        self.vntol = float(vntol)
        self.abstol = float(abstol)
        self.T = float(temperature)

        self.node_names: list[str] = netlist.nodes
        self.n_nodes = len(self.node_names)
        self.node_map: dict[str, int] = {
            nm: i for i, nm in enumerate(self.node_names)}
        for gname in GROUND_NAMES:
            self.node_map[gname] = -1
        self.node_map["0"] = -1

        self.branch_names: list[str] = []
        self._branch_of: dict[str, int] = {}
        for el in netlist.elements:
            if el.has_branch:
                self._branch_of[el.name] = self.n_nodes + len(self.branch_names)
                self.branch_names.append(f"i({el.name})")
        for dev in netlist.devices:
            # Registered verbatim: devices.py looks its own branch row up by
            # the exact name extra_unknowns() returned.
            for key in _extra_unknowns(dev):
                if key in self._branch_of:
                    raise ValueError(f"duplicate branch unknown {key!r}")
                self._branch_of[key] = self.n_nodes + len(self.branch_names)
                self.branch_names.append(key)
        self.n_branch = len(self.branch_names)
        self.n = self.n_nodes + self.n_branch

        self.branch_map: dict[str, int] = dict(self._branch_of)
        self.nmap: dict[str, int] = dict(self.node_map)
        self.nmap.update(self._branch_of)

        # Row classification, used by the Newton convergence test (voltages and
        # currents need different absolute tolerances).
        self._is_node_row = np.zeros(self.n, dtype=bool)
        self._is_node_row[: self.n_nodes] = True

        self._caps = [el for el in netlist.elements if el.kind == "C"]
        self._cap_pos = {el.name: k for k, el in enumerate(self._caps)}
        self._cap_i = np.zeros(len(self._caps))
        self._x_lin = np.zeros(self.n)          # linearisation point
        self._hist: list[np.ndarray] = []
        self._matrix_cache: dict[Any, sp.csr_matrix] = {}
        self._lu_cache: dict[Any, Any] = {}
        self.dc_info: dict[str, Any] = {}
        self._newton_total = 0

    # ------------------------------------------------------------------
    # indexing helpers
    # ------------------------------------------------------------------
    @property
    def is_linear(self) -> bool:
        """True when no stamp depends on the solution, so one solve is exact.

        A netlist may hold devices and still be linear: ``devices.Resistor``
        and friends declare ``linear = True``.  Skipping Newton for those is
        not just faster, it makes the answer exact rather than converged.
        """
        return all(getattr(d, "linear", False) for d in self.netlist.devices)

    def index(self, name: str) -> int:
        """Row index of a node or branch unknown.  Ground is ``-1``."""
        if name in self.nmap:
            return self.nmap[name]
        key = _canon_node(name)
        if key in self.nmap:
            return self.nmap[key]
        raise ValueError(f"unknown node or branch {name!r}")

    def _row(self, name: str) -> int:
        """Assembly row: ground folded onto the discarded trailing row ``n``."""
        i = self.nmap[name]
        return self.n if i < 0 else i

    def _augment(self, x: np.ndarray | None) -> np.ndarray:
        """Append the ground rail (0 V) so ``x[-1]`` is a valid read."""
        out = np.zeros(self.n + 1)
        if x is not None:
            out[: self.n] = x
        return out

    # ------------------------------------------------------------------
    # assembly
    # ------------------------------------------------------------------
    def _static_matrix(self, dt: float | None, method: str,
                       gmin: float) -> sp.csr_matrix:
        """Assemble (and cache) every matrix entry that does not depend on x."""
        key = (dt, method, gmin)
        cached = self._matrix_cache.get(key)
        if cached is not None:
            return cached

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []

        def st(i: int, j: int, v: float) -> None:
            rows.append(i)
            cols.append(j)
            vals.append(v)

        def two_term(a: int, b: int, g: float) -> None:
            st(a, a, g)
            st(b, b, g)
            st(a, b, -g)
            st(b, a, -g)

        for el in self.netlist.elements:
            k = el.kind
            if k == "R":
                a, b = self._row(el.nodes[0]), self._row(el.nodes[1])
                two_term(a, b, 1.0 / float(el.value))
            elif k == "C":
                if dt is None:
                    continue                       # open circuit at DC
                a, b = self._row(el.nodes[0]), self._row(el.nodes[1])
                two_term(a, b, _cap_geq(float(el.value), dt, method))
            elif k == "G":
                a, b = self._row(el.nodes[0]), self._row(el.nodes[1])
                cp, cm = self._row(el.nodes[2]), self._row(el.nodes[3])
                gm = float(el.value)
                st(a, cp, gm)
                st(a, cm, -gm)
                st(b, cp, -gm)
                st(b, cm, gm)
            elif k in ("V", "L", "E"):
                a, b = self._row(el.nodes[0]), self._row(el.nodes[1])
                br = self._branch_of[el.name]
                st(a, br, 1.0)
                st(b, br, -1.0)
                st(br, a, 1.0)
                st(br, b, -1.0)
                if k == "L" and dt is not None:
                    st(br, br, -_ind_zeq(float(el.value), dt, method))
                elif k == "E":
                    cp, cm = self._row(el.nodes[2]), self._row(el.nodes[3])
                    st(br, cp, -float(el.value))
                    st(br, cm, float(el.value))

        if gmin > 0.0:
            for i in range(self.n_nodes):
                st(i, i, gmin)

        r = np.asarray(rows, dtype=np.intp)
        c = np.asarray(cols, dtype=np.intp)
        v = np.asarray(vals, dtype=float)
        keep = (r < self.n) & (c < self.n)          # drop the ground row/col
        A = sp.coo_matrix((v[keep], (r[keep], c[keep])),
                          shape=(self.n, self.n)).tocsr()
        A.sum_duplicates()
        self._matrix_cache[key] = A
        return A

    def _rhs(self, t: float, hist: Sequence[np.ndarray], dt: float | None,
             method: str, source_scale: float) -> np.ndarray:
        """Right-hand side (length n+1; the trailing ground entry is dropped)."""
        b = np.zeros(self.n + 1)
        xp = hist[0] if hist else None
        xp2 = hist[1] if len(hist) > 1 else None
        for el in self.netlist.elements:
            k = el.kind
            if k == "V":
                b[self._branch_of[el.name]] = source_scale * el.value_at(t)
            elif k == "I":
                a, m = self._row(el.nodes[0]), self._row(el.nodes[1])
                i_src = source_scale * el.value_at(t)
                b[a] -= i_src
                b[m] += i_src
            elif k == "C" and dt is not None:
                a, m = self._row(el.nodes[0]), self._row(el.nodes[1])
                idx = self._cap_pos[el.name]
                vprev = 0.0 if xp is None else float(xp[a] - xp[m])
                geq = _cap_geq(float(el.value), dt, method)
                if method == "be":
                    ieq = geq * vprev
                elif method == "trap":
                    ieq = geq * vprev + self._cap_i[idx]
                else:                              # gear2
                    vprev2 = 0.0 if xp2 is None else float(xp2[a] - xp2[m])
                    ieq = float(el.value) * (4.0 * vprev - vprev2) / (2.0 * dt)
                b[a] += ieq
                b[m] -= ieq
            elif k == "L" and dt is not None:
                br = self._branch_of[el.name]
                a, m = self._row(el.nodes[0]), self._row(el.nodes[1])
                iprev = 0.0 if xp is None else float(xp[br])
                L = float(el.value)
                if method == "be":
                    b[br] = -(L / dt) * iprev
                elif method == "trap":
                    vprev = 0.0 if xp is None else float(xp[a] - xp[m])
                    b[br] = -(2.0 * L / dt) * iprev - vprev
                else:                              # gear2
                    iprev2 = 0.0 if xp2 is None else float(xp2[br])
                    b[br] = -(L / (2.0 * dt)) * (4.0 * iprev - iprev2)
        return b

    def stamp(self, t: float, x_prev: np.ndarray | Sequence[np.ndarray] | None = None,
              dt: float | None = None, method: str = "be",
              x: np.ndarray | None = None, gmin: float | None = None,
              source_scale: float = 1.0) -> tuple[sp.csr_matrix, np.ndarray]:
        """Assemble the MNA system at time ``t``.

        This is the hook :mod:`fieldspice.circuit.coupling` uses: it condenses
        a meshed field region onto its terminals (a Schur complement giving a
        dense ``nt x nt`` admittance plus a history current) and adds that
        block to the matrix returned here, so that the field and the circuit
        are solved as **one** system rather than co-simulated with a handshake.
        The row index of a terminal is ``solver.index(node_name)``.

        Parameters
        ----------
        t
            Time [s] at which independent sources are evaluated.  For an
            implicit step this is the *new* time point.
        x_prev
            Previous solution vector(s), most recent first.  ``None`` requests
            the DC (steady-state) stamp.  ``gear2`` needs two.
        dt
            Step size [s], or ``None`` for DC (capacitors open, inductors
            short).
        method
            ``"be"``, ``"trap"`` or ``"gear2"``.
        x
            Linearisation point for nonlinear devices; defaults to the last
            Newton iterate or operating point.
        gmin
            Override the solver's node-to-ground conductance [S].
        source_scale
            Multiplies every independent source, for source-stepping homotopy.

        Returns
        -------
        A : scipy.sparse.csr_matrix
            ``(n, n)`` system matrix [mixed S and dimensionless rows].
        b : numpy.ndarray
            ``(n,)`` right-hand side [mixed A and V rows].

        Notes
        -----
        With ``method="trap"`` the capacitor companion needs the capacitor
        *current* at the previous step, which is not recoverable from ``x_prev``
        alone; the solver carries it internally and updates it in
        :meth:`transient`.  Calling ``stamp`` with ``"trap"`` outside that loop
        uses whatever history is currently stored (zeros after construction).
        """
        method = _check_method(method)
        gmin = self.gmin if gmin is None else float(gmin)
        if x_prev is None:
            hist: list[np.ndarray] = []
        elif isinstance(x_prev, np.ndarray):
            hist = [self._augment(x_prev)]
        else:
            hist = [self._augment(np.asarray(h, dtype=float)) for h in x_prev]

        A = self._static_matrix(dt, method, gmin)
        b = self._rhs(t, hist, dt, method, source_scale)

        if self.netlist.devices:
            xa = self._augment(self._x_lin if x is None else np.asarray(x, float))
            Dm = sp.lil_matrix((self.n + 1, self.n + 1))
            xp = hist[0] if hist else self._augment(None)
            for dev in self.netlist.devices:
                _dev_call(dev, "set_time", t)
                if dt is None:
                    dev.stamp_dc(Dm, b, xa, self.nmap)
                else:
                    dev.stamp_tran(Dm, b, xa, xp, dt, self.nmap)
            A = A + Dm.tocsr()[: self.n, : self.n]
        else:
            # Hand out a copy: the linear part is cached and a caller such as
            # coupling.py legitimately adds its own block to what it gets back.
            A = A.copy()
        return A.tocsr(), b[: self.n]

    # ------------------------------------------------------------------
    # linear algebra
    # ------------------------------------------------------------------
    def _solve(self, A: sp.spmatrix, b: np.ndarray,
               cache_key: Any = None) -> np.ndarray:
        """One sparse LU solve, reusing the factorisation when ``cache_key`` repeats.

        Caching is the difference between a usable and an unusable transient:
        for a linear circuit with constant ``dt`` the matrix never changes, so
        thousands of steps cost one factorisation plus thousands of triangular
        solves.
        """
        try:
            if cache_key is not None:
                lu = self._lu_cache.get(cache_key)
                if lu is None:
                    lu = spla.splu(sp.csc_matrix(A))
                    self._lu_cache[cache_key] = lu
                return np.asarray(lu.solve(b), dtype=float)
            with warnings.catch_warnings():
                warnings.simplefilter("error", spla.MatrixRankWarning)
                return np.asarray(spla.spsolve(sp.csc_matrix(A), b), dtype=float)
        except (RuntimeError, spla.MatrixRankWarning) as exc:
            raise ConvergenceError(
                f"MNA matrix is singular ({exc}). Usual causes: a node with no "
                "DC path to ground (raise gmin), a loop of ideal voltage "
                "sources, or a current source in series with an open circuit."
            ) from None

    # ------------------------------------------------------------------
    # Newton
    # ------------------------------------------------------------------
    def _converged(self, x_old: np.ndarray, x_new: np.ndarray) -> bool:
        d = np.abs(x_new - x_old)
        scale = np.maximum(np.abs(x_new), np.abs(x_old))
        tol = self.reltol * scale + np.where(self._is_node_row,
                                             self.vntol, self.abstol)
        return bool(np.all(d <= tol))

    def _newton(self, t: float, x0: np.ndarray, dt: float | None,
                method: str, hist: Sequence[np.ndarray], gmin: float,
                source_scale: float) -> tuple[np.ndarray, int, bool]:
        """Newton-Raphson on the companion-model system.

        Returns ``(x, iterations, converged)``.  The linear case exits after a
        single solve, which is exact.
        """
        linear = self.is_linear
        for dev in self.netlist.devices:
            _dev_call(dev, "reset_state") or _dev_call(dev, "reset")
        x = np.asarray(x0, dtype=float).copy()
        key = (dt, method, gmin) if linear else None
        for it in range(1, int(self.cfg.max_newton) + 1):
            A, b = self.stamp(t, hist or None, dt, method, x=x, gmin=gmin,
                              source_scale=source_scale)
            xn = self._solve(A, b, cache_key=key)
            self._newton_total += 1
            if not np.all(np.isfinite(xn)):
                return x, it, False
            if linear:
                return xn, it, True
            done = self._converged(x, xn)
            x = xn
            if done and it >= 2:
                return x, it, True
        return x, int(self.cfg.max_newton), False

    def _solve_dc_at(self, t: float, x0: np.ndarray | None = None,
                     homotopy: Sequence[str] = ("direct", "gmin", "source")
                     ) -> tuple[np.ndarray, dict[str, Any]]:
        """DC operating point, trying each homotopy in turn.

        1. **direct** --- Newton from ``x0`` (or zero).  With well-limited
           compact models this succeeds on essentially every physical circuit
           and the two fallbacks below never run.
        2. **gmin stepping** --- start with a large conductance from every node
           to ground, which swamps the nonlinearity and leaves a nearly linear
           problem, then reduce it by decades, each stage seeded with the
           previous solution.  This is the cure for a stamp that is an
           unlimited exponential: the shunt bounds the node voltage, so the
           exponent cannot run away.
        3. **source stepping** --- scale every independent source from 0 to
           full, backing off the increment on failure.  When the topology
           rather than the device is the problem (latches, bistable nodes), the
           zero-source state is a known-good starting point that gmin stepping
           does not provide.

        Which one succeeded is reported in :attr:`dc_info`.
        """
        x0 = np.zeros(self.n) if x0 is None else np.asarray(x0, float)
        want = [str(s).lower() for s in homotopy]
        bad = set(want) - {"direct", "gmin", "source"}
        if bad:
            raise ValueError(f"unknown homotopy stage(s) {sorted(bad)}")
        tried: list[str] = []

        if "direct" in want:
            tried.append("direct")
            x, it, ok = self._newton(t, x0, None, "be", (), self.gmin, 1.0)
            if ok:
                return x, {"homotopy": "direct", "iterations": it}

        if self.is_linear:
            raise ConvergenceError(
                "linear DC solve failed; the matrix is probably singular")

        if "gmin" in want:
            tried.append("gmin")
            # Start hard.  SPICE3 ramps from gmin*1e10 (about 1e-2 S), which is
            # too weak to hold a node below a junction's turn-on when the
            # source resistance is kilohms: the first iterate lands at half the
            # supply and the exponential explodes.  1 S clamps every node to
            # millivolts against any sane series resistance, and the twelve
            # decades back down cost twelve cheap solves.
            g = max(1.0, self.gmin * 1e12)
            xs, nstage, ok_all = x0.copy(), 0, True
            while g > self.gmin:
                xs, it, ok = self._newton(t, xs, None, "be", (), g, 1.0)
                nstage += 1
                if not ok:
                    ok_all = False
                    break
                g /= 10.0
            if ok_all:
                x, it, ok = self._newton(t, xs, None, "be", (), self.gmin, 1.0)
                if ok:
                    return x, {"homotopy": "gmin", "stages": nstage,
                               "iterations": it}

        if "source" in want:
            tried.append("source")
            alpha, step, xs, guard = 0.0, 0.25, np.zeros(self.n), 0
            while alpha < 1.0 and guard < 500:
                guard += 1
                trial = min(1.0, alpha + step)
                xt, it, ok = self._newton(t, xs, None, "be", (), self.gmin,
                                          trial)
                if ok:
                    alpha, xs = trial, xt
                    step = min(2.0 * step, 0.5)
                else:
                    step *= 0.25
                    if step < 1e-9:
                        break
            if alpha >= 1.0:
                return xs, {"homotopy": "source", "source_steps": guard}

        raise ConvergenceError(
            f"DC operating point failed after trying {tried}. Raise gmin, add "
            "series resistance to any ideal exponential device, or supply a "
            "better starting guess via operating_point()/x0.")

    # ------------------------------------------------------------------
    # public analyses
    # ------------------------------------------------------------------
    def dc(self, t: float = 0.0,
           homotopy: Sequence[str] = ("direct", "gmin", "source")
           ) -> dict[str, float]:
        """DC operating point.

        Capacitors are open circuits and inductors are short circuits, which is
        the ``d/dt -> 0`` limit of the companion models.

        Parameters
        ----------
        t
            Time [s] at which time-varying sources are sampled.  A transient
            run uses ``t = t_start`` so that the initial state is consistent
            with the waveform.
        homotopy
            Which continuation strategies to try, in order.  See
            :meth:`_solve_dc_at`; restricting the list is mainly useful for
            testing that a given strategy works on its own.

        Returns
        -------
        dict
            ``{node_name: voltage [V]}`` for every node (including ``"0"``,
            which is 0 by definition) plus ``{"i(NAME)": current [A]}`` for
            every branch unknown.  Which homotopy succeeded is left in
            :attr:`dc_info`.
        """
        x = self.operating_point(t, homotopy)
        out: dict[str, float] = {"0": 0.0}
        for nm, i in self.node_map.items():
            if i >= 0:
                out[nm] = float(x[i])
        for nm, i in self._branch_of.items():
            out[nm if nm.startswith("i(") else f"i({nm})"] = float(x[i])
        return out

    def operating_point(self, t: float = 0.0,
                        homotopy: Sequence[str] = ("direct", "gmin", "source")
                        ) -> np.ndarray:
        """DC solution as the raw unknown vector [V and A]."""
        x, info = self._solve_dc_at(t, homotopy=homotopy)
        self._x_lin = x
        self.dc_info = info
        return x

    def dc_sweep(self, source: str, values: Iterable[float]) -> Result:
        """Sweep one independent source and record everything.

        Each point is seeded with the previous solution, which is what makes a
        diode or transistor IV curve converge over many decades: the natural
        continuation parameter is the sweep itself.

        Parameters
        ----------
        source
            Name of a ``V`` or ``I`` element.
        values
            Source values [V] or [A].

        Returns
        -------
        Result
            ``scalars["sweep"]`` holds the swept values; ``scalars["v(node)"]``
            and ``scalars["i(elem)"]`` the responses.
        """
        el = self.netlist.by_name(source)
        if el.kind not in ("V", "I"):
            raise ValueError(f"{source!r} is not an independent source")
        vals = np.asarray(list(values), dtype=float)
        saved = el.value
        xs = np.zeros((vals.size, self.n))
        t0 = time.perf_counter()
        x = None
        try:
            for k, v in enumerate(vals):
                el.value = float(v)
                x, info = self._solve_dc_at(0.0, x)
                xs[k] = x
        finally:
            el.value = saved
        res = Result(grid=None)
        res.scalars["sweep"] = vals
        self._fill_scalars(res, xs)
        res.fields["x"] = xs
        res.meta.update(solver=self.name, analysis="dc_sweep",
                        assumptions=list(self.assumptions),
                        source=source, wall_time=time.perf_counter() - t0)
        return res

    def transient(self, t_end: float, dt: float, method: str = "be",
                  t_start: float = 0.0, uic: bool = False,
                  x0: np.ndarray | None = None,
                  monitors: Any = None,
                  store_every: int = 1) -> Result:
        """Fixed-step transient analysis.

        Parameters
        ----------
        t_end, t_start
            Time window [s].
        dt
            Step size [s].  Fixed: this solver does not do local truncation
            error control, so choose ``dt`` from the fastest edge you care
            about (20-100 points per rise time is the usual rule).
        method
            ``"be"`` (default, first order, L-stable), ``"trap"`` (second
            order, rings on a step) or ``"gear2"`` (second order, L-stable).
            See the module docstring for why the default is the crude one.
        uic
            Use initial conditions instead of the operating point.  Capacitor
            ``ic`` voltages and inductor ``ic`` currents are enforced by
            temporarily replacing each capacitor with a voltage source and each
            inductor with a current source and solving that circuit, which is
            the exact statement of "the state variables start here".
        x0
            Explicit initial unknown vector, overriding both of the above.
        monitors
            Optional object with ``record(state, t)`` / ``finalize()``.
        store_every
            Keep every ``store_every``-th step (the last step is always kept).

        Returns
        -------
        Result
            ``t`` [s], ``fields["x"]`` ``(nt, n)``, ``scalars["v(node)"]`` [V],
            ``scalars["i(branch)"]`` [A], ``scalars["i(C...)"]`` [A] for
            capacitor currents, and ``terminals[name]`` for every independent
            source.

        Notes
        -----
        **The first step is always backward Euler**, for both second-order
        methods, which is what SPICE does and is not a shortcut.  ``gear2``
        needs two past points and has only one.  ``trap`` needs the capacitor
        *current* at ``t_start``, which is not an input --- the initial state
        fixes charge, not current, and for two capacitors sharing a node the
        split is not even determined by KCL.  Seeding it with zero puts an
        ``O(1)`` error in the first step and measurably degrades trapezoidal
        integration to first order (it costs a factor of 2.7 in accuracy on
        the RC test in this module's docstring examples).  One backward-Euler
        step instead contributes a single ``O(dt^2)`` local error, which
        leaves the global second-order rate intact --- measured, not assumed.
        """
        method = _check_method(method)
        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if t_end <= t_start:
            raise ValueError("t_end must exceed t_start")
        if store_every < 1:
            raise ValueError("store_every must be >= 1")

        t0 = time.perf_counter()
        nsteps = int(math.ceil((t_end - t_start) / dt))
        self._newton_total = 0

        if x0 is not None:
            x = np.asarray(x0, dtype=float).copy()
            if x.shape != (self.n,):
                raise ValueError(f"x0 must have shape ({self.n},)")
            init = "given"
        elif uic:
            x = self._ic_state(t_start)
            init = "uic"
        else:
            x = self.operating_point(t_start)
            init = "op"

        self._cap_i = np.zeros(len(self._caps))
        hist = [x.copy()]
        keep = [0]
        xs = [x.copy()]
        ts = [t_start]
        nonconv = 0

        for k in range(1, nsteps + 1):
            t = t_start + k * dt
            m = "be" if k == 1 else method
            xn, it, ok = self._newton(t, hist[0], dt, m, hist, self.gmin, 1.0)
            if not ok:
                nonconv += 1
                if self.is_linear:
                    raise ConvergenceError(
                        f"linear transient solve failed at t={t:.6g} s")
            self._update_cap_currents(xn, hist[0], dt, m)
            for dev in self.netlist.devices:
                # Devices that carry their own integration history (charge on a
                # junction, a latched switch) commit it here, once per accepted
                # step -- never inside the Newton loop, where the step is still
                # provisional.
                _dev_call(dev, "accept_timestep", self._augment(xn),
                          self._augment(hist[0]), dt, self.nmap)
            hist = [xn.copy()] + hist[:1]
            x = xn
            if monitors is not None and hasattr(monitors, "record"):
                monitors.record({"x": x, "solver": self}, t)
            if k % store_every == 0 or k == nsteps:
                xs.append(x.copy())
                ts.append(t)
                keep.append(k)

        arr = np.asarray(xs)
        res = Result(grid=None, t=np.asarray(ts))
        res.fields["x"] = arr
        self._fill_scalars(res, arr)
        self._fill_terminals(res, arr, np.asarray(ts))
        res.meta.update(
            solver=self.name, analysis="transient",
            assumptions=list(self.assumptions), method=method, dt=dt,
            steps=nsteps, stored=len(ts), initial=init,
            newton_solves=self._newton_total,
            nonconverged_steps=nonconv,
            wall_time=time.perf_counter() - t0)
        if monitors is not None and hasattr(monitors, "finalize"):
            res.meta["monitors"] = monitors.finalize()
        return res

    def ac(self, freqs: Iterable[float] | float) -> Result:
        """Small-signal AC analysis: one complex solve per frequency.

        Nonlinear devices are linearised about the DC operating point.  A
        device contributes its DC conductance automatically (the companion
        stamp *is* the small-signal conductance at the operating point); it
        contributes small-signal *capacitance* only if it implements the
        optional ``stamp_ac`` hook, and ``meta["ac_device_reactance"]`` records
        whether any device did.

        Parameters
        ----------
        freqs
            Frequencies [Hz].  Zero is allowed and reproduces the DC solution.

        Returns
        -------
        Result
            ``scalars["freq"]`` [Hz], ``fields["x"]`` complex ``(nf, n)``, and
            complex ``scalars["v(node)"]`` [V] / ``scalars["i(branch)"]`` [A].

        Notes
        -----
        Drive is taken from each source's ``ac`` magnitude and phase.  If *no*
        source in the netlist declares one, every independent source is given
        unit magnitude and a warning is issued --- silently returning an
        all-zero spectrum is a worse outcome than a loud default.
        """
        seq = list(freqs) if hasattr(freqs, "__iter__") else [freqs]
        f = np.asarray(seq, dtype=float).ravel()
        if f.size == 0:
            raise ValueError("no frequencies given")
        if np.any(f < 0):
            raise ValueError("frequencies must be non-negative")
        t0 = time.perf_counter()

        x_op = self.operating_point(0.0) if self.netlist.devices else None
        srcs = [el for el in self.netlist.elements if el.kind in ("V", "I")]
        declared = any(el.ac_mag is not None for el in srcs)
        if not declared and srcs:
            warnings.warn(
                "no source declares an AC magnitude; driving every "
                "independent source with 1.0 for this analysis",
                RuntimeWarning, stacklevel=2)

        Gdev, ac_devs = self._device_small_signal(x_op)
        out = np.zeros((f.size, self.n), dtype=complex)
        for k, freq in enumerate(f):
            A, b = self._ac_system(freq, srcs, declared, Gdev, ac_devs, x_op)
            out[k] = spla.spsolve(sp.csc_matrix(A), b)

        res = Result(grid=None)
        res.scalars["freq"] = f
        res.fields["x"] = out
        self._fill_scalars(res, out)
        res.meta.update(solver=self.name, analysis="ac",
                        assumptions=list(self.assumptions),
                        n_freq=int(f.size),
                        ac_stamped_devices=[d.name for d in ac_devs],
                        ac_dc_only_devices=[
                            getattr(d, "name", "?")
                            for d in self.netlist.devices if d not in ac_devs],
                        ac_default_drive=not declared,
                        wall_time=time.perf_counter() - t0)
        return res

    # ------------------------------------------------------------------
    # internals for the analyses above
    # ------------------------------------------------------------------
    def _device_small_signal(self, x_op: np.ndarray | None
                             ) -> tuple[sp.csr_matrix | None, list[Any]]:
        """Split devices into "has an AC stamp" and "DC conductance only".

        A device's ``stamp_ac`` returns its *complete* small-signal admittance
        (``gd + j*omega*Cj`` for a diode), so using it *and* the DC companion
        conductance would double-count the real part.  Devices without one
        contribute only their operating-point conductance, which is exactly
        right for a memoryless model and misses the charge storage of a
        reactive one --- reported in ``meta["ac_dc_only_devices"]`` rather than
        left implicit.
        """
        if not self.netlist.devices:
            return None, []
        xa = self._augment(x_op)
        ac_devs: list[Any] = []
        dc_only: list[Any] = []
        for dev in self.netlist.devices:
            _dev_call(dev, "reset_state") or _dev_call(dev, "reset")
            hook = getattr(dev, "stamp_ac", None)
            if hook is None:
                dc_only.append(dev)
                continue
            try:                            # probe: the base class raises
                hook(sp.lil_matrix((self.n + 1, self.n + 1), dtype=complex),
                     np.zeros(self.n + 1, dtype=complex), xa, 0.0, self.nmap)
                ac_devs.append(dev)
            except NotImplementedError:
                dc_only.append(dev)
        Dm = sp.lil_matrix((self.n + 1, self.n + 1))
        junk = np.zeros(self.n + 1)
        for dev in dc_only:
            dev.stamp_dc(Dm, junk, xa, self.nmap)
        return Dm.tocsr()[: self.n, : self.n], ac_devs

    def _ac_system(self, freq: float, srcs: Sequence[Element], declared: bool,
                   Gdev: sp.csr_matrix | None, ac_devs: Sequence[Any],
                   x_op: np.ndarray | None
                   ) -> tuple[sp.csr_matrix, np.ndarray]:
        w = 2.0 * np.pi * float(freq)
        rows: list[int] = []
        cols: list[int] = []
        vals: list[complex] = []

        def st(i: int, j: int, v: complex) -> None:
            rows.append(i)
            cols.append(j)
            vals.append(v)

        def two_term(a: int, b: int, y: complex) -> None:
            st(a, a, y)
            st(b, b, y)
            st(a, b, -y)
            st(b, a, -y)

        b = np.zeros(self.n + 1, dtype=complex)
        for el in self.netlist.elements:
            k = el.kind
            if k == "R":
                two_term(self._row(el.nodes[0]), self._row(el.nodes[1]),
                         1.0 / float(el.value))
            elif k == "C":
                two_term(self._row(el.nodes[0]), self._row(el.nodes[1]),
                         1j * w * float(el.value))
            elif k == "G":
                a, m = self._row(el.nodes[0]), self._row(el.nodes[1])
                cp, cm = self._row(el.nodes[2]), self._row(el.nodes[3])
                st(a, cp, float(el.value))
                st(a, cm, -float(el.value))
                st(m, cp, -float(el.value))
                st(m, cm, float(el.value))
            elif k in ("V", "L", "E"):
                a, m = self._row(el.nodes[0]), self._row(el.nodes[1])
                br = self._branch_of[el.name]
                st(a, br, 1.0)
                st(m, br, -1.0)
                st(br, a, 1.0)
                st(br, m, -1.0)
                if k == "L":
                    st(br, br, -1j * w * float(el.value))
                elif k == "E":
                    cp, cm = self._row(el.nodes[2]), self._row(el.nodes[3])
                    st(br, cp, -float(el.value))
                    st(br, cm, float(el.value))
            if k in ("V", "I"):
                mag = 1.0 if not declared else (el.ac_mag or 0.0)
                phasor = mag * np.exp(1j * np.deg2rad(el.ac_phase))
                if k == "V":
                    b[self._branch_of[el.name]] = phasor
                else:
                    b[self._row(el.nodes[0])] -= phasor
                    b[self._row(el.nodes[1])] += phasor
        if self.gmin > 0.0:
            for i in range(self.n_nodes):
                st(i, i, self.gmin)

        r = np.asarray(rows, dtype=np.intp)
        c = np.asarray(cols, dtype=np.intp)
        v = np.asarray(vals, dtype=complex)
        keep = (r < self.n) & (c < self.n)
        A = sp.coo_matrix((v[keep], (r[keep], c[keep])),
                          shape=(self.n, self.n)).tocsr()
        if Gdev is not None:
            A = A + Gdev.astype(complex)
        if ac_devs:
            Am = sp.lil_matrix((self.n + 1, self.n + 1), dtype=complex)
            xa = self._augment(x_op)
            for dev in ac_devs:
                dev.stamp_ac(Am, b, xa, w, self.nmap)
            A = A + Am.tocsr()[: self.n, : self.n]
        return A.tocsr(), b[: self.n]

    def _ic_state(self, t: float) -> np.ndarray:
        """Initial state with capacitor voltages and inductor currents forced.

        Built by solving an auxiliary netlist in which every capacitor is a
        voltage source of value ``ic`` and every inductor a current source of
        value ``ic`` (missing ``ic`` means zero), then mapping the answer back
        by name.  That auxiliary circuit is the exact algebraic statement of
        "the state variables have these values and everything else follows",
        and it costs one extra DC solve.
        """
        aux = Netlist()
        for el in self.netlist.elements:
            ic = 0.0 if el.ic is None else float(el.ic)
            if el.kind == "C":
                aux.add_vsource(el.name, el.nodes[0], el.nodes[1], ic)
            elif el.kind == "L":
                aux.add_isource(el.name, el.nodes[0], el.nodes[1], ic)
            elif el.kind == "R":
                aux.add_resistor(el.name, el.nodes[0], el.nodes[1], el.value)
            elif el.kind == "V":
                aux.add_vsource(el.name, el.nodes[0], el.nodes[1], el.value)
            elif el.kind == "I":
                aux.add_isource(el.name, el.nodes[0], el.nodes[1], el.value)
            elif el.kind == "E":
                aux.add_vcvs(el.name, *el.nodes, el.value)
            elif el.kind == "G":
                aux.add_vccs(el.name, *el.nodes, el.value)
        for dev in self.netlist.devices:
            aux.devices.append(dev)
        for nd in self.netlist.nodes:          # keep every node alive
            aux._register(nd)

        sub = MNASolver(aux, self.cfg, gmin=max(self.gmin, 1e-12),
                        reltol=self.reltol, vntol=self.vntol,
                        abstol=self.abstol, temperature=self.T)
        xa = sub.operating_point(t)

        x = np.zeros(self.n)
        for nm, i in self.node_map.items():
            if i >= 0:
                x[i] = xa[sub.node_map[nm]]
        for nm, i in self._branch_of.items():
            if nm in sub._branch_of:
                x[i] = xa[sub._branch_of[nm]]
        for el in self.netlist.elements:        # inductor currents are given
            if el.kind == "L":
                x[self._branch_of[el.name]] = 0.0 if el.ic is None else float(el.ic)
        for nd, v in self.netlist.ic.items():   # .ic overrides
            if nd in self.node_map and self.node_map[nd] >= 0:
                x[self.node_map[nd]] = float(v)
        return x

    def _update_cap_currents(self, x: np.ndarray, xprev: np.ndarray,
                             dt: float, method: str) -> None:
        """Advance the stored capacitor currents [A] for the trapezoidal rule."""
        if not self._caps:
            return
        xa, xm = self._augment(x), self._augment(xprev)
        for idx, el in enumerate(self._caps):
            a, m = self.nmap[el.nodes[0]], self.nmap[el.nodes[1]]
            dv = float(xa[a] - xa[m]) - float(xm[a] - xm[m])
            geq = _cap_geq(float(el.value), dt, method)
            if method == "trap":
                self._cap_i[idx] = geq * dv - self._cap_i[idx]
            else:
                self._cap_i[idx] = geq * dv

    def _fill_scalars(self, res: Result, xs: np.ndarray) -> None:
        for nm, i in self.node_map.items():
            if i >= 0:
                res.scalars[f"v({nm})"] = xs[:, i]
        for nm, i in self._branch_of.items():
            # A device names its own branch unknown and usually already calls
            # it "i(name)"; do not wrap that a second time.
            res.scalars[nm if nm.startswith("i(") else f"i({nm})"] = xs[:, i]

    def _fill_terminals(self, res: Result, xs: np.ndarray,
                        ts: np.ndarray) -> None:
        aug = np.concatenate([xs, np.zeros((xs.shape[0], 1))], axis=1)
        for el in self.netlist.elements:
            if el.kind not in ("V", "I"):
                continue
            a, m = self.nmap[el.nodes[0]], self.nmap[el.nodes[1]]
            v = aug[:, a] - aug[:, m]
            if el.kind == "V":
                i = xs[:, self._branch_of[el.name]]
            else:
                i = np.array([el.value_at(float(t)) for t in ts])
            res.terminals[el.name] = {"v": v, "i": i}


def _check_method(method: str) -> str:
    key = str(method).strip().lower()
    if key not in _METHOD_ALIAS:
        raise ValueError(
            f"unknown integration method {method!r}; "
            f"choose from {sorted(set(_METHOD_ALIAS.values()))}")
    return _METHOD_ALIAS[key]


def _cap_geq(C: float, dt: float, method: str) -> float:
    """Companion conductance [S] of a capacitor ``C`` [F] over a step ``dt`` [s]."""
    if method == "be":
        return C / dt
    if method == "trap":
        return 2.0 * C / dt
    return 1.5 * C / dt                      # gear2 (BDF2)


def _ind_zeq(L: float, dt: float, method: str) -> float:
    """Companion "resistance" [ohm] on the branch row of an inductor ``L`` [H]."""
    if method == "be":
        return L / dt
    if method == "trap":
        return 2.0 * L / dt
    return 1.5 * L / dt                      # gear2 (BDF2)
