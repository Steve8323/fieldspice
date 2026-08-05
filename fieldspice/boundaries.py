"""Boundary conditions on the six domain walls.

A boundary condition in fieldspice is *data*, not code: a :class:`BC` object
says what the physics is, and the solver that owns the assembly decides how to
impose it.  That split exists because the same wall means different things to
different solvers --- ``Absorbing`` is a real absorber in ``fdtd`` and is
meaningless in ``eqs`` --- and because the one condition that beginners get
most wrong (Neumann) requires the assembler to do *nothing at all*.

The four things this module gets right
--------------------------------------

**1. Neumann is the natural boundary condition and needs no action.**
The nodal operator ``L = G^T M G`` is assembled only from edges that exist
inside the domain.  A boundary node therefore has no edge crossing the wall,
so its assembled row already states "the net flux through my dual box equals
what is injected into it", with the exposed face contributing exactly zero.
That *is* the homogeneous Neumann condition ``dphi/dn = 0``, exactly, with no
truncation error and with no ghost cells.  Attempting to "enforce" it --- by
mirroring a ghost node, or by overwriting the boundary row with a one-sided
difference ``phi[0] - phi[1] = 0`` --- imposes the condition twice, destroys
the symmetry of ``L`` (so CG and Cholesky are no longer available, costing
3-10x), and breaks discrete charge conservation because the overwritten row no
longer sums a physical flux.  :meth:`BoundarySpec.neumann_load` returns a load
vector that is identically zero for the default ``flux = 0``, which is the
honest way to say "nothing to do".

**2. Corner and edge nodes belong to several walls, so precedence is fixed.**
A node on ``xlo`` and ``ylo`` is claimed by two conditions.  The rule, applied
everywhere in this module and stated once here, is:

  a. **Dirichlet always beats Neumann.**  A prescribed potential is a hard
     constraint on the value; a prescribed flux is a statement about a face of
     the dual box.  When both are asserted at one node the value wins, and the
     flux load on that node is dropped (the solver replaces its row anyway).
  b. **Among Dirichlet-like walls, the later wall in the canonical order
     ``xlo, xhi, ylo, yhi, zlo, zhi`` wins.**  So a node shared by ``xlo`` at
     0 V and ``zhi`` at 1 V ends up at 1 V.

Rule (b) is arbitrary but *deterministic*, which is the property that matters:
two runs of the same problem, and two different solvers reading the same
:class:`BoundarySpec`, always fix the same node to the same number.  If the
two values genuinely differ, the geometry is telling you that two electrodes
touch, and you should model that with :class:`~fieldspice.solvers.base.Terminal`
node sets instead of with wall conditions.

**3. A collapsed direction is a symmetry plane, and its default is Neumann.**
A 2D problem is a grid with ``Nz == 1`` and a finite z-thickness (see
``grid.py``).  Translational invariance along z is precisely homogeneous
Neumann on ``zlo`` and ``zhi``, so the default :class:`BoundarySpec` --- which
is Neumann on all six walls --- already does the right thing and costs
nothing.  Blanket constructors accept an optional ``grid`` so that, for
example, ``BoundarySpec.all_dirichlet(0.0, grid=g)`` grounds the four resolved
walls of a 2D grid and leaves the collapsed pair natural.  Explicit conditions
on a collapsed direction are still honoured (a one-cell-thick parallel-plate
capacitor with Dirichlet on ``zlo``/``zhi`` is a legitimate and useful model).

**4. Periodicity is a constraint, not a wall value.**  :func:`periodic_pairs`
returns the node index mapping; building the constraint operator is the
solver's job.  See :class:`Periodic` for the exact algebra.

Open boundaries (A12)
---------------------
The quasi-static default, homogeneous Neumann, means "no normal current or
flux crosses the wall".  That is *exact* for a symmetry plane and for a
shielded enclosure.  For an **open** problem it is a modelling error: it
confines the field to the box and therefore **under-estimates fringing
capacitance** and over-estimates confinement.  The error does not shrink with
mesh refinement, only with domain padding --- pad the domain to at least 3x
the largest feature dimension before trusting a fringing number, and use
``validate.check_padding``.  The quasi-static solvers have no absorbing
boundary at all (an exact open quasi-static boundary needs a BEM coupling,
which is not implemented); :class:`Absorbing` is honoured only by ``fdtd``.
This is assumption **A12** in ``docs/ASSUMPTIONS.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Callable, ClassVar, Mapping

import numpy as np

from .grid import RectilinearGrid
from .units import eta0

__all__ = [
    "WALLS", "WALL_AXIS", "WALL_SIDE", "AXIS_NAMES",
    "BC", "Dirichlet", "Neumann", "Periodic", "Absorbing", "Symmetry",
    "BoundarySpec",
    "wall_nodes", "wall_dual_areas", "boundary_nodes", "periodic_pairs",
]


# ==========================================================================
# Wall naming
# ==========================================================================
WALLS: tuple[str, str, str, str, str, str] = (
    "xlo", "xhi", "ylo", "yhi", "zlo", "zhi")
"""Canonical wall order.  Precedence among Dirichlet-like walls is *later wins*
in exactly this order, so it is part of the public contract, not an
implementation detail."""

WALL_AXIS: Mapping[str, int] = MappingProxyType(
    {"xlo": 0, "xhi": 0, "ylo": 1, "yhi": 1, "zlo": 2, "zhi": 2})
"""Wall name -> axis index (x=0, y=1, z=2)."""

WALL_SIDE: Mapping[str, int] = MappingProxyType(
    {"xlo": 0, "xhi": 1, "ylo": 0, "yhi": 1, "zlo": 0, "zhi": 1})
"""Wall name -> 0 for the low-coordinate wall, 1 for the high-coordinate wall."""

AXIS_NAMES: tuple[str, str, str] = ("x", "y", "z")


def _check_wall(wall: str) -> str:
    if wall not in WALL_AXIS:
        raise ValueError(f"unknown wall {wall!r}; expected one of {WALLS}")
    return wall


def _axis_index(axis: int | str) -> int:
    """Accept 0/1/2 or 'x'/'y'/'z' and return the integer axis index."""
    if isinstance(axis, str):
        if axis not in AXIS_NAMES:
            raise ValueError(f"unknown axis {axis!r}; expected one of {AXIS_NAMES}")
        return AXIS_NAMES.index(axis)
    a = int(axis)
    if a not in (0, 1, 2):
        raise ValueError(f"axis index must be 0, 1 or 2, got {axis!r}")
    return a


def _node_strides(grid: RectilinearGrid) -> tuple[int, int, int]:
    """C-order flat-index strides of the ``(Nx+1, Ny+1, Nz+1)`` node array."""
    nx, ny, nz = grid.shape_nodes
    return (ny * nz, nz, 1)


def _is_collapsed(grid: RectilinearGrid, axis: int) -> bool:
    """True if ``axis`` holds a single cell, i.e. is a symmetry direction.

    ``grid._Axis.collapsed`` is set exactly when an axis has two node
    coordinates, which is the same statement as ``N == 1``; using the public
    ``ncell`` avoids reaching into the frozen grid internals.
    """
    return grid.ncell[axis] == 1


# ==========================================================================
# Wall index arithmetic
# ==========================================================================
def wall_nodes(grid: RectilinearGrid, wall: str) -> np.ndarray:
    """Flat node indices lying on one domain wall.

    The node array has shape ``(Nx+1, Ny+1, Nz+1)`` and is flattened in C
    order, so the flat index of node ``(i, j, k)`` is
    ``i*(Ny+1)*(Nz+1) + j*(Nz+1) + k``.  This function builds those indices by
    outer arithmetic on the two in-plane axes rather than by materialising the
    whole index array, and the result is *already sorted ascending* because the
    two free axes are iterated in increasing axis order with decreasing stride.

    Corner and edge nodes are returned by **every** wall that contains them;
    this function is pure geometry and applies no precedence.  Use
    :meth:`BoundarySpec.node_masks` with ``exclusive=True`` for disjoint sets.

    Parameters
    ----------
    grid
        The grid.  No units enter here; this is pure index arithmetic.
    wall
        One of ``"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"``.

    Returns
    -------
    np.ndarray
        ``intp`` array of flat node indices, ascending, length
        ``prod(shape_nodes) / shape_nodes[axis]``.

    Examples
    --------
    >>> from fieldspice.grid import RectilinearGrid
    >>> g = RectilinearGrid.uniform([(0, 1), (0, 1)], [2, 3])
    >>> idx = wall_nodes(g, "ylo")
    >>> np.all(np.unravel_index(idx, g.shape_nodes)[1] == 0)
    True
    """
    _check_wall(wall)
    shape = grid.shape_nodes
    strides = _node_strides(grid)
    axis = WALL_AXIS[wall]
    plane = 0 if WALL_SIDE[wall] == 0 else shape[axis] - 1
    d0, d1 = [d for d in (0, 1, 2) if d != axis]
    a = np.arange(shape[d0], dtype=np.intp) * strides[d0]
    b = np.arange(shape[d1], dtype=np.intp) * strides[d1]
    return (plane * strides[axis] + a[:, None] + b[None, :]).ravel()


def wall_dual_areas(grid: RectilinearGrid, wall: str) -> np.ndarray:
    """Exposed dual-box face area of every node on a wall [m^2].

    Each boundary node owns a dual (box-method) control volume whose face on
    the wall has area ``hd_a * hd_b`` over the two in-plane axes, where ``hd``
    are the dual widths from ``grid``.  These areas sum exactly to the wall
    area, which is what makes a Neumann flux load conservative.  The ordering
    matches :func:`wall_nodes` element for element.

    Returns
    -------
    np.ndarray
        Areas [m^2], same length and ordering as ``wall_nodes(grid, wall)``.
    """
    _check_wall(wall)
    axis = WALL_AXIS[wall]
    duals = (grid.hxd, grid.hyd, grid.hzd)
    d0, d1 = [d for d in (0, 1, 2) if d != axis]
    return (duals[d0][:, None] * duals[d1][None, :]).ravel()


def boundary_nodes(grid: RectilinearGrid) -> np.ndarray:
    """All distinct flat node indices on the domain surface, ascending."""
    return np.unique(np.concatenate([wall_nodes(grid, w) for w in WALLS]))


def periodic_pairs(grid: RectilinearGrid,
                   axis: int | str) -> tuple[np.ndarray, np.ndarray]:
    """Node index mapping that identifies the two walls normal to ``axis``.

    Parameters
    ----------
    grid
        The grid.
    axis
        ``0``/``1``/``2`` or ``"x"``/``"y"``/``"z"``.

    Returns
    -------
    (lo, hi) : tuple of np.ndarray
        Two ``intp`` arrays of equal length.  ``lo[m]`` and ``hi[m]`` are the
        *same* transverse position on the low and high walls, so the wrap-around
        constraint is the element-wise statement ``phi[hi] = phi[lo] + offset``.
        ``lo`` is ascending; ``hi`` is ``lo`` shifted by one constant stride.

    Notes
    -----
    The pairing is exact rather than interpolated because both walls are
    generated by :func:`wall_nodes` with the same in-plane ordering, and the
    two flat index sets therefore differ by the single constant
    ``(shape_nodes[axis] - 1) * stride[axis]``.

    Examples
    --------
    >>> from fieldspice.grid import RectilinearGrid
    >>> g = RectilinearGrid.uniform([(0, 1), (0, 1)], [3, 2])
    >>> lo, hi = periodic_pairs(g, "x")
    >>> int(np.unique(hi - lo).size), bool(np.all(np.unravel_index(
    ...     lo, g.shape_nodes)[0] == 0))
    (1, True)
    """
    a = _axis_index(axis)
    lo = wall_nodes(grid, WALLS[2 * a])
    hi = wall_nodes(grid, WALLS[2 * a + 1])
    return lo, hi


# ==========================================================================
# Boundary condition types
# ==========================================================================
class BC(ABC):
    """Base class for a condition on one domain wall.

    A :class:`BC` carries no grid and no state: it is a small immutable value
    object that a solver interrogates.  Subclasses set the class attribute
    :attr:`kind`, which is what solvers dispatch and serialise on.
    """

    kind: ClassVar[str] = "bc"

    @abstractmethod
    def describe(self) -> str:
        """One-line human-readable summary, used by :meth:`BoundarySpec.summary`."""

    # -- classification used by the precedence rule ------------------------
    @property
    def is_dirichlet_like(self) -> bool:
        """True if this wall prescribes the nodal value (and so wins ties)."""
        return False

    @property
    def is_natural(self) -> bool:
        """True if the assembled system already satisfies this wall condition."""
        return False

    @property
    def time_dependent(self) -> bool:
        """True if any prescribed quantity is a callable of time."""
        return False

    def prescribed_value(self, t: float = 0.0) -> np.ndarray | None:
        """Prescribed nodal value at time ``t`` [V], or ``None`` if not fixed.

        Returns a 0-d array for a uniform value, or a 1-d array matching the
        wall node count for a spatially varying one.
        """
        return None


@dataclass(frozen=True)
class Dirichlet(BC):
    """Prescribed potential on the wall: ``phi = value`` [V].

    Parameters
    ----------
    value
        Either a constant [V], or a callable ``f(t) -> float`` with ``t`` in
        seconds (the signature produced by :mod:`fieldspice.sources`).  The
        callable may also return an array whose length equals the number of
        nodes on that wall, giving a spatially varying electrode; the ordering
        is that of :func:`wall_nodes`.

    Notes
    -----
    This is the *essential* boundary condition: it constrains the unknown
    itself, so it must be imposed by eliminating rows and columns.  Hand the
    output of :meth:`BoundarySpec.dirichlet_nodes` straight to
    :func:`fieldspice.operators.apply_dirichlet`, which does the elimination
    symmetrically and therefore keeps the system amenable to CG / Cholesky.

    A Dirichlet wall is a perfect electric conductor held at a potential.  It
    removes the constant null vector of the all-Neumann nodal Laplacian, which
    is why a problem with at least one Dirichlet wall or one voltage-driven
    terminal is uniquely solvable and a problem with neither is not.
    """

    value: float | Callable[[float], float] = 0.0
    kind: ClassVar[str] = "dirichlet"

    def __post_init__(self) -> None:
        if not callable(self.value):
            try:
                float(self.value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Dirichlet value must be a float [V] or a callable "
                    f"f(t) -> float, got {type(self.value).__name__}") from exc

    @property
    def is_dirichlet_like(self) -> bool:
        return True

    @property
    def time_dependent(self) -> bool:
        return callable(self.value)

    def value_at(self, t: float = 0.0) -> np.ndarray:
        """Potential at time ``t`` [s], returned in volts.

        Returns
        -------
        np.ndarray
            0-d for a uniform wall, 1-d (length = wall node count) if the
            callable returned an array.
        """
        v = self.value(float(t)) if callable(self.value) else self.value
        arr = np.asarray(v, dtype=float)
        if arr.ndim > 1:
            raise ValueError(
                f"Dirichlet value must be scalar or 1-D, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Dirichlet value is not finite at t = {t} s")
        return arr

    def prescribed_value(self, t: float = 0.0) -> np.ndarray:
        return self.value_at(t)

    def describe(self) -> str:
        if callable(self.value):
            name = getattr(self.value, "__name__", type(self.value).__name__)
            return f"Dirichlet phi = {name}(t) V"
        return f"Dirichlet phi = {float(self.value):.6g} V"


@dataclass(frozen=True)
class Neumann(BC):
    """Prescribed outward normal flux density.  The *natural* condition.

    Parameters
    ----------
    flux
        Outward normal component of the flux density conjugate to the operator,
        or a callable ``f(t) -> float`` of time [s].  Units follow the mass
        matrix: ``J.n`` [A/m^2] when the edge mass carries ``sigma``, and
        ``D.n`` (a surface charge density) [C/m^2] when it carries ``eps``.
        The default ``0.0`` is homogeneous Neumann.

    Notes
    -----
    **Homogeneous Neumann requires no action whatsoever.**  ``G^T M G`` is
    assembled only from edges inside the domain, so the row of a boundary node
    already omits any flux through the wall --- the zero-flux condition holds
    exactly, to machine precision, with no ghost nodes and no modification.
    Implementers routinely get this wrong by trying to enforce it with a
    mirrored ghost node or a one-sided difference row; that imposes the
    condition a second time, breaks the symmetry of the matrix (losing CG and
    Cholesky), and destroys discrete charge conservation.  Do nothing.

    A non-zero ``flux`` is the only case that touches the assembly, and it
    enters through the right-hand side alone --- see
    :meth:`BoundarySpec.neumann_load`.  The matrix is untouched and stays
    symmetric.

    Physically, homogeneous Neumann is exact for a symmetry plane (no current
    crosses a mirror plane) and for a shielded enclosure.  On an **open**
    boundary it is a modelling error that confines the field and
    under-estimates fringing capacitance: assumption **A12**.  Pad the domain
    to at least 3x the largest feature dimension.
    """

    flux: float | Callable[[float], float] = 0.0
    kind: ClassVar[str] = "neumann"

    def __post_init__(self) -> None:
        if not callable(self.flux):
            try:
                float(self.flux)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Neumann flux must be a float or a callable f(t) -> float, "
                    f"got {type(self.flux).__name__}") from exc

    @property
    def is_natural(self) -> bool:
        # Only the homogeneous case is free; a driven flux needs an RHS load.
        return not callable(self.flux) and float(self.flux) == 0.0

    @property
    def time_dependent(self) -> bool:
        return callable(self.flux)

    def flux_at(self, t: float = 0.0) -> np.ndarray:
        """Outward normal flux density at time ``t`` [s]; see class docstring."""
        f = self.flux(float(t)) if callable(self.flux) else self.flux
        arr = np.asarray(f, dtype=float)
        if arr.ndim > 1:
            raise ValueError(
                f"Neumann flux must be scalar or 1-D, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Neumann flux is not finite at t = {t} s")
        return arr

    def describe(self) -> str:
        if callable(self.flux):
            name = getattr(self.flux, "__name__", type(self.flux).__name__)
            return f"Neumann flux = {name}(t) (driven)"
        f = float(self.flux)
        return ("Neumann flux = 0 (natural, no action)" if f == 0.0
                else f"Neumann flux = {f:.6g} outward")


@dataclass(frozen=True)
class Periodic(BC):
    """Wrap-around identification of the two walls normal to an axis.

    Must be set on **both** walls of the axis; a one-sided ``Periodic`` is a
    contradiction and :meth:`BoundarySpec.validate` raises on it.

    Parameters
    ----------
    offset
        Constant potential jump across one period [V]: the constraint is
        ``phi[hi] = phi[lo] + offset``.  Non-zero ``offset`` models a uniform
        applied field across a periodic unit cell (a drift bias on a repeating
        structure) without breaking periodicity of the *fields*.
    phase
        Floquet phase advance across one period [rad], meaningful only in the
        complex-valued ``ac`` solver where the constraint becomes
        ``phi[hi] = phi[lo] * exp(-1j*phase)``.  Ignored by the real-valued
        time-domain solvers, which raise if it is non-zero.

    Notes
    -----
    How a solver consumes this.  Let ``lo, hi = periodic_pairs(grid, axis)``.
    Build the prolongation ``P`` (shape ``n_nodes x n_free``) that maps the
    reduced unknown set --- all nodes except ``hi`` --- to all nodes, with the
    ``hi`` rows copying their partner in ``lo``, and let ``c`` be the vector
    that is ``offset`` on ``hi`` and zero elsewhere.  Then solve

    ``(P^T A P) u = P^T (b - A c)``  and recover ``phi = P u + c``.

    ``P^T A P`` is still symmetric, so nothing is lost.  Equivalently, and more
    cheaply in practice: add row/column ``hi`` into row/column ``lo`` and then
    delete ``hi``.

    The identification is geometrically consistent because the dual volumes of
    the two identified node planes add to ``h[0]/2 + h[-1]/2``, which is exactly
    the dual volume of that node on the resulting torus.  The user must ensure
    the *materials* also match across the wrap, which this module cannot check.
    """

    offset: float = 0.0
    phase: float = 0.0
    kind: ClassVar[str] = "periodic"

    def __post_init__(self) -> None:
        for nm in ("offset", "phase"):
            try:
                float(getattr(self, nm))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Periodic {nm} must be a float") from exc

    def pairs(self, grid: RectilinearGrid,
              axis: int | str) -> tuple[np.ndarray, np.ndarray]:
        """``(lo, hi)`` flat node indices to identify; see :func:`periodic_pairs`."""
        return periodic_pairs(grid, axis)

    def describe(self) -> str:
        extra = []
        if self.offset:
            extra.append(f"offset {self.offset:.6g} V")
        if self.phase:
            extra.append(f"phase {self.phase:.6g} rad")
        return "Periodic" + (" (" + ", ".join(extra) + ")" if extra else "")


@dataclass(frozen=True)
class Symmetry(BC):
    """A mirror plane: PEC (electric wall) or PMC (magnetic wall).

    Parameters
    ----------
    wall
        ``"pmc"`` (default) or ``"pec"``, case-insensitive.
    value
        Potential of a PEC mirror plane [V].  Ignored for PMC.  Zero is the
        usual choice: an antisymmetric (odd) potential vanishes on the plane.

    Notes
    -----
    Reduction to a scalar condition, which is what the quasi-static solvers
    need:

    * **PMC** (magnetic wall, ``H_t = 0``): no normal current and no normal
      displacement crosses the plane, so ``dphi/dn = 0``.  This is
      :class:`Neumann` --- natural, no action.  It is the correct plane for an
      *even* (symmetric) potential, and it is what a collapsed grid direction
      implies.
    * **PEC** (electric wall, ``E_t = 0``): the tangential field
      ``E_t = -grad_t phi`` vanishes, so ``phi`` is constant on the plane.
      This is :class:`Dirichlet` at ``value``, and it is the correct plane for
      an *odd* (antisymmetric) potential.

    Using a symmetry plane halves (or quarters, or eighths) the unknown count.
    The cost is that it also removes every antisymmetric mode from the
    spectrum, so a symmetry plane is only valid if the *excitation* shares the
    symmetry --- a differential drive across a plane you declared PMC will
    silently produce the common-mode answer.
    """

    wall: str = "pmc"
    value: float = 0.0
    kind: ClassVar[str] = "symmetry"

    def __post_init__(self) -> None:
        w = str(self.wall).lower()
        if w not in ("pec", "pmc"):
            raise ValueError(
                f"Symmetry wall must be 'pec' or 'pmc', got {self.wall!r}")
        object.__setattr__(self, "wall", w)
        try:
            float(self.value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Symmetry value must be a float [V]") from exc

    @property
    def is_electric(self) -> bool:
        """True for a PEC (electric) wall."""
        return self.wall == "pec"

    @property
    def is_dirichlet_like(self) -> bool:
        return self.is_electric

    @property
    def is_natural(self) -> bool:
        return not self.is_electric

    def value_at(self, t: float = 0.0) -> np.ndarray:
        """Prescribed potential [V] for a PEC plane; raises for PMC."""
        if not self.is_electric:
            raise ValueError("a PMC symmetry plane prescribes no potential")
        return np.asarray(float(self.value))

    def prescribed_value(self, t: float = 0.0) -> np.ndarray | None:
        return self.value_at(t) if self.is_electric else None

    def as_scalar_bc(self) -> BC:
        """The equivalent scalar-potential condition: Dirichlet or Neumann."""
        return Dirichlet(float(self.value)) if self.is_electric else Neumann(0.0)

    def describe(self) -> str:
        if self.is_electric:
            return f"Symmetry PEC (phi = {float(self.value):.6g} V, Dirichlet)"
        return "Symmetry PMC (dphi/dn = 0, natural)"


@dataclass(frozen=True)
class Absorbing(BC):
    """Open boundary for the full-wave solver: a CPML specification.

    This class is **data only**.  It carries the grading parameters of a
    convolutional perfectly matched layer and computes the profiles from them;
    the auxiliary-variable update that actually absorbs the wave lives in
    ``solvers/fdtd.py``.  The quasi-static solvers reject it: there is no
    quasi-static absorber in fieldspice (**A12**), because an exact open
    quasi-static boundary needs a boundary-element coupling that is not
    implemented.  Pad and use Neumann instead.

    Parameters
    ----------
    thickness
        Layer depth in **cells**.  8-12 is the usual range; 10 is the default.
        Fewer than ~6 cells reflects noticeably, more than ~16 rarely pays.
    order
        Polynomial grading exponent ``m`` for ``sigma`` and ``kappa``.  3-4 is
        optimal in practice; 3 is the default.
    sigma_max
        Peak CPML conductivity at the outer wall, in the ``eps0``-normalised
        convention below.  ``None`` (the default) means "use the optimal value"
        from :meth:`sigma_optimal`, which needs the local cell size.
    kappa_max
        Peak real coordinate stretch.  ``1.0`` (no stretch) is right for a
        compact domain; raise to 5-11 for an elongated domain or grazing
        incidence, where the evanescent and near-grazing spectrum otherwise
        reflects.
    alpha_max
        Peak complex-frequency-shift (CFS) parameter, same units as
        ``sigma_max``.  This is what makes a *convolutional* PML absorb
        evanescent and low-frequency content instead of amplifying it; 0.05 is
        the standard value and the reason to prefer CPML over a plain
        split-field PML for near-field circuit problems.
    alpha_order
        Grading exponent ``m_a`` for ``alpha``, which is graded the *opposite*
        way to ``sigma``: maximal at the inner interface, zero at the outer
        wall.  1.0 (linear) is standard.
    sigma_factor
        Coefficient in the optimal-conductivity formula, 0.8 by convention.

    Notes
    -----
    Convention.  ``sigma`` and ``alpha`` are normalised by ``eps0``, as in the
    Roden-Gedney stretched coordinate

    ``s = kappa + sigma / (alpha + 1j*omega*eps0)``

    so both have units of S/m but are *not* material conductivities.  Getting
    this wrong by a factor of ``eps_r`` is the classic CPML bug, which is why
    the material scaling appears explicitly in :meth:`sigma_optimal`.

    Grading, with ``d`` the normalised depth into the layer (0 at the inner
    interface, 1 at the outer wall):

    ``sigma(d) = sigma_max * d**m``,
    ``kappa(d) = 1 + (kappa_max - 1) * d**m``,
    ``alpha(d) = alpha_max * (1 - d)**m_a``.

    References: J. A. Roden and S. D. Gedney, "Convolutional PML (CPML)",
    Microwave Opt. Technol. Lett. 27(5):334-339, 2000; A. Taflove and
    S. C. Hagness, *Computational Electrodynamics*, 3rd ed., ch. 7 (eqs.
    7.60-7.66 for the polynomial grading and the optimal conductivity).
    """

    thickness: int = 10
    order: float = 3.0
    sigma_max: float | None = None
    kappa_max: float = 1.0
    alpha_max: float = 0.05
    alpha_order: float = 1.0
    sigma_factor: float = 0.8
    kind: ClassVar[str] = "absorbing"

    def __post_init__(self) -> None:
        if int(self.thickness) != self.thickness or self.thickness < 1:
            raise ValueError(
                f"Absorbing thickness must be a positive integer number of "
                f"cells, got {self.thickness!r}")
        object.__setattr__(self, "thickness", int(self.thickness))
        if self.order <= 0 or self.alpha_order < 0:
            raise ValueError("Absorbing grading exponents must be positive")
        if self.kappa_max < 1.0:
            raise ValueError("Absorbing kappa_max must be >= 1")
        if self.alpha_max < 0.0 or self.sigma_factor <= 0.0:
            raise ValueError("Absorbing alpha_max must be >= 0, "
                             "sigma_factor must be > 0")
        if self.sigma_max is not None and self.sigma_max < 0.0:
            raise ValueError("Absorbing sigma_max must be >= 0 or None")

    # -- conductivity choice ----------------------------------------------
    def sigma_optimal(self, dx: float, eps_r: float = 1.0,
                      mu_r: float = 1.0) -> float:
        """Optimal peak CPML conductivity for a cell size [S/m, eps0-normalised].

        ``sigma_max = sigma_factor * (m + 1) / (eta0 * dx * sqrt(eps_r*mu_r))``

        Parameters
        ----------
        dx
            Cell size along the wall normal, inside the layer [m].
        eps_r, mu_r
            Relative permittivity and permeability of the medium the layer
            truncates (dimensionless).  A PML matched to vacuum in front of a
            dielectric reflects, so pass the real material.
        """
        dx = float(dx)
        if dx <= 0.0:
            raise ValueError("dx must be positive [m]")
        if eps_r <= 0.0 or mu_r <= 0.0:
            raise ValueError("eps_r and mu_r must be positive")
        return (self.sigma_factor * (self.order + 1.0)
                / (eta0 * dx * np.sqrt(eps_r * mu_r)))

    def sigma_from_reflection(self, depth: float, r0: float = 1e-6,
                              eps_r: float = 1.0, mu_r: float = 1.0) -> float:
        """Peak conductivity giving a target normal-incidence reflection.

        ``sigma_max = -(m + 1) * ln(r0) / (2 * eta0 * depth * sqrt(eps_r*mu_r))``

        Parameters
        ----------
        depth
            Total physical thickness of the layer [m].
        r0
            Target amplitude reflection coefficient of the *continuous* PML,
            dimensionless, in ``(0, 1)``.  1e-6 is a typical target; the
            discretisation error usually dominates below ~1e-8.
        eps_r, mu_r
            Medium being truncated (dimensionless).
        """
        depth = float(depth)
        if depth <= 0.0:
            raise ValueError("depth must be positive [m]")
        if not 0.0 < r0 < 1.0:
            raise ValueError("target reflection r0 must lie in (0, 1)")
        return (-(self.order + 1.0) * np.log(r0)
                / (2.0 * eta0 * depth * np.sqrt(eps_r * mu_r)))

    def resolved_sigma_max(self, dx: float, eps_r: float = 1.0,
                           mu_r: float = 1.0) -> float:
        """``sigma_max`` if set, else :meth:`sigma_optimal` [S/m, normalised]."""
        if self.sigma_max is not None:
            return float(self.sigma_max)
        return self.sigma_optimal(dx, eps_r, mu_r)

    # -- profiles ----------------------------------------------------------
    def cell_depths(self, offset: float = 0.5) -> np.ndarray:
        """Normalised depths of the ``thickness`` layer cells, dimensionless.

        ``offset = 0.5`` samples cell centres (where the electric field update
        needs the profile); ``offset = 0.0`` samples the inner face of each
        cell (where the staggered magnetic update needs it).  Depth 0 is the
        inner interface with the physical domain, 1 is the outer wall.
        """
        return (np.arange(self.thickness, dtype=float) + float(offset)) \
            / float(self.thickness)

    def profile(self, depth: np.ndarray | float, dx: float,
                eps_r: float = 1.0, mu_r: float = 1.0
                ) -> dict[str, np.ndarray]:
        """CPML grading profiles at normalised depths.

        Parameters
        ----------
        depth
            Normalised depth(s) in ``[0, 1]``, dimensionless.
        dx
            Cell size along the wall normal [m], used only when ``sigma_max``
            was left as ``None``.
        eps_r, mu_r
            Medium being truncated (dimensionless).

        Returns
        -------
        dict
            ``"sigma"`` [S/m, eps0-normalised], ``"kappa"`` (dimensionless,
            >= 1) and ``"alpha"`` [S/m, eps0-normalised], each with the shape
            of ``depth``.
        """
        d = np.asarray(depth, dtype=float)
        if np.any(d < 0.0) or np.any(d > 1.0):
            raise ValueError("normalised depth must lie in [0, 1]")
        smax = self.resolved_sigma_max(dx, eps_r, mu_r)
        return {
            "sigma": smax * d ** self.order,
            "kappa": 1.0 + (self.kappa_max - 1.0) * d ** self.order,
            "alpha": self.alpha_max * (1.0 - d) ** self.alpha_order,
        }

    def describe(self) -> str:
        s = "auto" if self.sigma_max is None else f"{self.sigma_max:.4g}"
        return (f"Absorbing CPML {self.thickness} cells, m = {self.order:g}, "
                f"sigma_max {s}, kappa_max {self.kappa_max:g}, "
                f"alpha_max {self.alpha_max:g}")


# ==========================================================================
# The six-wall specification
# ==========================================================================
@dataclass
class BoundarySpec:
    """Boundary conditions on the six walls of the domain.

    The default is homogeneous :class:`Neumann` everywhere, which is the
    correct quasi-static default: no normal current or flux leaves the box.
    See the module docstring for the precedence rule at shared corner and edge
    nodes, and for the open-boundary caveat (**A12**).

    Parameters
    ----------
    xlo, xhi, ylo, yhi, zlo, zhi
        A :class:`BC` per wall.

    Examples
    --------
    >>> from fieldspice.grid import RectilinearGrid
    >>> g = RectilinearGrid.uniform([(0, 1), (0, 1)], [2, 2])
    >>> bc = BoundarySpec(xlo=Dirichlet(0.0), xhi=Dirichlet(1.0))
    >>> idx, val = bc.dirichlet_nodes(g)
    >>> idx.size, val.min(), val.max()
    (12, 0.0, 1.0)
    """

    xlo: BC = Neumann()
    xhi: BC = Neumann()
    ylo: BC = Neumann()
    yhi: BC = Neumann()
    zlo: BC = Neumann()
    zhi: BC = Neumann()

    def __post_init__(self) -> None:
        for wall in WALLS:
            bc = getattr(self, wall)
            if not isinstance(bc, BC):
                raise ValueError(
                    f"boundary {wall} must be a BC instance, got "
                    f"{type(bc).__name__}")

    # -- construction ------------------------------------------------------
    @classmethod
    def all_neumann(cls, grid: RectilinearGrid | None = None) -> "BoundarySpec":
        """Homogeneous Neumann on all six walls: the quasi-static default.

        ``grid`` is accepted and ignored, for signature symmetry with
        :meth:`all_dirichlet`; a collapsed direction already wants exactly this.
        """
        del grid
        return cls()

    @classmethod
    def all_dirichlet(cls, v: float | Callable[[float], float] = 0.0,
                      grid: RectilinearGrid | None = None) -> "BoundarySpec":
        """Prescribe ``phi = v`` [V] on all six walls (a grounded shield box).

        Parameters
        ----------
        v
            Potential [V] or callable ``f(t) -> float``.
        grid
            Optional.  When given, walls normal to a **collapsed** direction
            (``N == 1``) are left Neumann instead, because a collapsed
            direction is a translational-symmetry plane and clamping it would
            impose the wrong physics on a 2D cross-section.  Without ``grid``
            the request is taken literally on all six walls --- which is what a
            one-cell-thick parallel-plate capacitor actually wants.
        """
        bc = Dirichlet(v)
        spec = cls(bc, bc, bc, bc, bc, bc)
        return spec if grid is None else spec.relax_collapsed(grid)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, BC]) -> "BoundarySpec":
        """Build from a partial ``{wall: BC}`` dict; unlisted walls are Neumann."""
        unknown = set(mapping) - set(WALLS)
        if unknown:
            raise ValueError(f"unknown wall(s) {sorted(unknown)}; expected {WALLS}")
        return cls(**dict(mapping))

    def relax_collapsed(self, grid: RectilinearGrid) -> "BoundarySpec":
        """Copy with Neumann on the walls of every collapsed direction.

        A collapsed direction (one cell) is the 2D/1D reduction's symmetry
        direction; homogeneous Neumann there *is* translational invariance.
        This is a deliberate, explicit step rather than something applied
        silently inside :meth:`dirichlet_nodes`, so that an intentional
        condition on a thin direction is never quietly discarded.
        """
        repl = {}
        for wall in WALLS:
            if _is_collapsed(grid, WALL_AXIS[wall]):
                repl[wall] = Neumann()
        return replace(self, **repl) if repl else replace(self)

    # -- access ------------------------------------------------------------
    def get(self, wall: str) -> BC:
        """The :class:`BC` on one wall."""
        return getattr(self, _check_wall(wall))

    def __getitem__(self, wall: str) -> BC:
        return self.get(wall)

    def items(self) -> tuple[tuple[str, BC], ...]:
        """``((wall, bc), ...)`` in canonical :data:`WALLS` order."""
        return tuple((w, getattr(self, w)) for w in WALLS)

    @property
    def is_time_dependent(self) -> bool:
        """True if any wall prescribes a callable of time.

        Solvers use this to decide whether a cached matrix factorisation stays
        valid across steps: a time-dependent *value* only changes the
        right-hand side, so the factorisation survives, but the driven nodes
        must be re-evaluated every step.
        """
        return any(bc.time_dependent for _, bc in self.items())

    def kinds(self) -> tuple[str, ...]:
        """The six :attr:`BC.kind` strings in canonical order."""
        return tuple(bc.kind for _, bc in self.items())

    # -- node index sets ---------------------------------------------------
    def node_masks(self, grid: RectilinearGrid,
                   exclusive: bool = False) -> dict[str, np.ndarray]:
        """Flat node indices on each of the six walls.

        Parameters
        ----------
        grid
            The grid.
        exclusive
            ``False`` (default): every wall reports **all** of its nodes, so a
            corner node appears in three lists.  This is what you want for
            "which conditions touch this node".

            ``True``: each shared node is assigned to the *last* wall in
            :data:`WALLS` order that contains it, so the six sets are disjoint
            and their union is the full boundary.  This is what you want for
            surface integrals (no double counting) and for tiling PML corner
            regions.

        Returns
        -------
        dict
            ``{wall: intp array}``, keys in canonical order, indices ascending.
        """
        if not exclusive:
            return {w: wall_nodes(grid, w) for w in WALLS}
        # Later wall wins: write ordinals in order, last write survives.
        owner = np.full(grid.n_nodes, -1, dtype=np.int8)
        for n, wall in enumerate(WALLS):
            owner[wall_nodes(grid, wall)] = n
        return {wall: np.flatnonzero(owner == n).astype(np.intp)
                for n, wall in enumerate(WALLS)}

    def dirichlet_nodes(self, grid: RectilinearGrid, t: float = 0.0
                        ) -> tuple[np.ndarray, np.ndarray]:
        """Nodes with a prescribed potential, and their values at time ``t``.

        The two arrays are ready to hand straight to
        :func:`fieldspice.operators.apply_dirichlet`::

            idx, val = bc.dirichlet_nodes(grid, t)
            A_bc, b_bc = apply_dirichlet(A, b, idx, val)

        Both :class:`Dirichlet` walls and PEC :class:`Symmetry` walls
        contribute.  Time-dependent values are evaluated by calling the
        wall's callable with ``t``; a callable may return a scalar or an array
        of length equal to that wall's node count (ordering of
        :func:`wall_nodes`).

        Shared nodes follow the module precedence rule: **Dirichlet beats
        Neumann**, and among Dirichlet-like walls the **later wall in
        ``xlo, xhi, ylo, yhi, zlo, zhi`` wins**.  The result is deterministic
        and independent of dict iteration order.

        Parameters
        ----------
        grid
            The grid.
        t
            Time [s] at which to evaluate driven electrodes.

        Returns
        -------
        (indices, values) : tuple of np.ndarray
            ``intp`` flat node indices, ascending and unique, and the matching
            potentials [V].  Both empty when no wall is Dirichlet-like --- in
            which case the nodal Laplacian keeps its constant null vector and
            the solver must pin a node or rely on a voltage-driven terminal.
        """
        idx_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        for wall in WALLS:                      # canonical order = precedence
            bc = getattr(self, wall)
            if not bc.is_dirichlet_like:
                continue
            v = bc.prescribed_value(t)
            if v is None:                       # defensive: is_dirichlet_like lied
                raise ValueError(
                    f"wall {wall}: {type(bc).__name__} claims to be Dirichlet-like "
                    "but prescribes no value")
            idx = wall_nodes(grid, wall)
            try:
                vals = np.broadcast_to(v, idx.shape).astype(float, copy=True)
            except ValueError as exc:
                raise ValueError(
                    f"wall {wall}: Dirichlet value of shape {np.shape(v)} cannot "
                    f"be broadcast to the {idx.size} nodes on that wall") from exc
            idx_parts.append(idx)
            val_parts.append(vals)

        if not idx_parts:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=float)

        idx_all = np.concatenate(idx_parts)
        val_all = np.concatenate(val_parts)
        # np.unique returns the index of the FIRST occurrence of each value;
        # scanning the reversed array therefore keeps the LAST wall's value,
        # which is exactly the documented precedence rule.
        uniq, first_in_rev = np.unique(idx_all[::-1], return_index=True)
        return uniq.astype(np.intp, copy=False), val_all[::-1][first_in_rev]

    def neumann_load(self, grid: RectilinearGrid,
                     t: float = 0.0) -> np.ndarray:
        """Right-hand-side contribution of driven (non-zero) Neumann walls.

        For the default ``flux = 0`` this is **identically zero** and can be
        skipped entirely --- homogeneous Neumann is the natural condition of
        ``G^T M G`` and needs no action (see :class:`Neumann`).

        Sign convention.  With ``G^T M_sigma G phi = I_inject`` and ``flux``
        the *outward* normal current density, the current leaving a boundary
        node's dual box through its exposed face is ``flux * A``, so KCL at
        that node reads ``L phi = I_inject - flux * A``.  This function returns
        that ``-flux * A`` term, to be **added** to the right-hand side.

        Contributions from different walls **add** at a shared edge or corner
        node, because they are integrals over distinct faces of the same dual
        box.  Nodes that end up Dirichlet are zeroed, per the precedence rule
        (their row is replaced by the elimination, so the load is meaningless
        there and keeping it would corrupt any subsequent current balance).

        Parameters
        ----------
        grid
            The grid.
        t
            Time [s], for callable fluxes.

        Returns
        -------
        np.ndarray
            Length ``grid.n_nodes``.  Units [A] when the operator carries
            ``sigma``, [C] when it carries ``eps``.
        """
        load = np.zeros(grid.n_nodes, dtype=float)
        for wall in WALLS:
            bc = getattr(self, wall)
            if not isinstance(bc, Neumann) or bc.is_natural:
                continue
            f = bc.flux_at(t)
            idx = wall_nodes(grid, wall)
            area = wall_dual_areas(grid, wall)
            try:
                fv = np.broadcast_to(f, idx.shape)
            except ValueError as exc:
                raise ValueError(
                    f"wall {wall}: Neumann flux of shape {np.shape(f)} cannot be "
                    f"broadcast to the {idx.size} nodes on that wall") from exc
            # wall_nodes is duplicate-free within one wall, so plain fancy-index
            # accumulation is exact here and ~100x faster than np.add.at.  The
            # summation over shared edge/corner nodes happens across walls,
            # one wall per loop iteration.
            load[idx] += -fv * area
        fixed, _ = self.dirichlet_nodes(grid, t)
        load[fixed] = 0.0
        return load

    def periodic_pairs(self, grid: RectilinearGrid
                       ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Wrap-around node identifications, keyed by axis name.

        Returns
        -------
        dict
            ``{"x"|"y"|"z": (lo, hi)}`` for every axis whose two walls are
            :class:`Periodic`.  ``phi[hi] = phi[lo] + offset`` element-wise;
            see :class:`Periodic` for how to turn that into a constraint.

        Raises
        ------
        ValueError
            If exactly one wall of an axis is periodic.
        """
        self.validate(grid)
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for a, name in enumerate(AXIS_NAMES):
            if isinstance(getattr(self, WALLS[2 * a]), Periodic):
                out[name] = periodic_pairs(grid, a)
        return out

    # -- checking ----------------------------------------------------------
    def validate(self, grid: RectilinearGrid | None = None) -> None:
        """Raise :class:`ValueError` on an inconsistent specification.

        Checks that periodicity is declared on both walls of an axis with
        matching parameters, and that a periodic axis is not also driven with a
        prescribed potential on the same walls.  Everything else is legal, and
        legality is not the same as wisdom --- see **A12**.
        """
        for a, name in enumerate(AXIS_NAMES):
            lo, hi = getattr(self, WALLS[2 * a]), getattr(self, WALLS[2 * a + 1])
            plo, phi_ = isinstance(lo, Periodic), isinstance(hi, Periodic)
            if plo != phi_:
                raise ValueError(
                    f"axis {name}: Periodic must be set on both walls "
                    f"({WALLS[2 * a]} is {lo.kind}, {WALLS[2 * a + 1]} is "
                    f"{hi.kind}); a one-sided periodic boundary is meaningless")
            if plo and (lo.offset != hi.offset or lo.phase != hi.phase):
                raise ValueError(
                    f"axis {name}: the two Periodic walls must carry identical "
                    "offset and phase")
        if grid is not None:
            for wall in WALLS:
                bc = getattr(self, wall)
                if isinstance(bc, Absorbing):
                    n = grid.ncell[WALL_AXIS[wall]]
                    if bc.thickness >= n:
                        raise ValueError(
                            f"wall {wall}: CPML thickness {bc.thickness} cells "
                            f"does not fit in {n} cells along that axis")

    def require(self, allowed: set[str] | frozenset[str],
                context: str = "this solver") -> None:
        """Raise unless every wall's :attr:`BC.kind` is in ``allowed``.

        Solvers call this at the top of ``solve`` so that an unsupported
        boundary is a loud error rather than silently wrong physics --- the
        motivating case being :class:`Absorbing` handed to a quasi-static
        solver, which has no absorbing boundary at all (**A12**).
        """
        bad = [(w, bc.kind) for w, bc in self.items() if bc.kind not in allowed]
        if bad:
            listed = ", ".join(f"{w}={k}" for w, k in bad)
            raise ValueError(
                f"{context} does not support boundary condition(s): {listed}. "
                f"Supported kinds: {sorted(allowed)}")

    # -- reporting ---------------------------------------------------------
    def summary(self) -> str:
        """Multi-line human-readable description of all six walls."""
        lines = ["BoundarySpec"]
        for wall, bc in self.items():
            lines.append(f"  {wall}: {bc.describe()}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()
