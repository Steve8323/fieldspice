"""Constructive solid geometry and sub-cell voxelisation.

Geometry enters fieldspice in exactly one place: a per-cell array of material
properties.  This module builds that array by evaluating an implicit solid
(a :class:`Shape`, possibly a CSG tree) on the grid.

Why a *fill fraction* and not a boolean mask
--------------------------------------------
:func:`voxelize` returns a float array in ``[0, 1]``: the fraction of each cell
covered by the shape, estimated by midpoint sampling.  This is the single
cheapest mitigation available for assumption **A2** (staircased geometry).  A
boolean mask throws away everything the solver could have known about where
inside a cell the interface sits; the fill fraction keeps it, and
:class:`~fieldspice.materials.MaterialMap` turns it into an effective-medium
average.  For a smooth interface that recovers most of the accuracy lost to
staircasing, at the cost of a few extra point evaluations at setup time --- a
cost that is invisible next to a single linear solve.

The mixing rule is the consumer's business, not this module's, but the physics
worth stating here is that the *correct* rule depends on the field direction:
for E parallel to a planar interface the tangential component is continuous, so
the effective permittivity is the volume-weighted **arithmetic** mean; for E
perpendicular the normal D is continuous, so it is the volume-weighted
**harmonic** mean.  A cell straddling an interface sees both, which is why the
arithmetic (linear) rule is the usual default and the harmonic rule is the
usual escape hatch.

Boundary tie-breaks (chosen once, applied everywhere)
-----------------------------------------------------
* Every primitive is **closed**: a point exactly on the surface is *inside*.
* :class:`Prism` is closed as well, including points exactly on a polygon edge
  or vertex, which the bare crossing-number rule handles inconsistently.
* ``A - B`` is therefore half-open on the cut surface: points on the boundary
  of ``B`` are removed from ``A``.

None of this changes an answer at the resolutions anyone actually runs --- the
set of sample points landing exactly on a surface has measure zero unless the
geometry is grid-aligned, in which case the fill fraction is exact anyway --- but
an unstated tie-break is a bug waiting to be blamed on the solver.

Units
-----
Every coordinate, length, radius and thickness in this module is in **metres**,
per the strict-SI rule.  Use ``fieldspice.units`` multipliers
(``2 * um``) rather than bare exponents.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

from .grid import RectilinearGrid

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from .materials import Material, MaterialMap

__all__ = [
    "Shape", "Box", "Cylinder", "Sphere", "HalfSpace", "Prism", "Torus",
    "Union", "Intersection", "Difference", "Complement", "Transformed",
    "union", "intersection", "difference",
    "voxelize", "Layer", "LayerStack",
]

BBox = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
"""Axis-aligned outer bound ``((xlo, xhi), (ylo, yhi), (zlo, zhi))`` [m].

Components may be ``+-inf``.  A bounding box is only ever required to be a
*conservative superset* of the shape.
"""

_AXIS_NAMES = ("x", "y", "z")
_INF_BBOX: BBox = ((-np.inf, np.inf), (-np.inf, np.inf), (-np.inf, np.inf))


# ==========================================================================
# Small validation helpers
# ==========================================================================
def _axis_index(axis: str | int) -> int:
    """Map ``"x"|"y"|"z"|0|1|2`` to ``0|1|2``, raising on anything else."""
    if isinstance(axis, str):
        a = axis.strip().lower()
        if a not in _AXIS_NAMES:
            raise ValueError(
                f"axis must be one of 'x', 'y', 'z', 0, 1, 2; got {axis!r}")
        return _AXIS_NAMES.index(a)
    if isinstance(axis, (int, np.integer)) and not isinstance(axis, bool):
        a_i = int(axis)
        if a_i in (0, 1, 2):
            return a_i
    raise ValueError(f"axis must be one of 'x', 'y', 'z', 0, 1, 2; got {axis!r}")


def _plane_axes(axis: int) -> tuple[int, int]:
    """The two axes perpendicular to ``axis``, in right-handed cyclic order.

    ``z -> (x, y)``, ``x -> (y, z)``, ``y -> (z, x)``.  Cyclic (not sorted)
    order is what keeps a polygon's orientation meaningful whichever axis it is
    extruded along.
    """
    return (axis + 1) % 3, (axis + 2) % 3


def _as_vec3(v: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    """Validate and copy a 3-vector [m]."""
    a = np.asarray(v, dtype=float).ravel()
    if a.size != 3:
        raise ValueError(f"{name} must have 3 components, got {a.size}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite, got {a}")
    return a


def _as_coords(x, y, z) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Coerce three coordinate arguments [m] to float arrays and check broadcast."""
    arrs = tuple(np.asarray(v, dtype=float) for v in (x, y, z))
    try:
        np.broadcast_shapes(*(a.shape for a in arrs))
    except ValueError as exc:
        raise ValueError(
            "x, y and z must broadcast to a common shape, got "
            f"{arrs[0].shape}, {arrs[1].shape}, {arrs[2].shape}") from exc
    return arrs


def _bbox_arrays(bb: BBox) -> tuple[np.ndarray, np.ndarray]:
    """Validate a bounding box and return ``(lo, hi)`` as ``(3,)`` arrays [m]."""
    arr = np.asarray(bb, dtype=float)
    if arr.shape != (3, 2):
        raise ValueError(f"bbox must be 3 (lo, hi) pairs, got shape {arr.shape}")
    if np.any(np.isnan(arr)):
        raise ValueError(f"bbox contains NaN: {bb}")
    return arr[:, 0], arr[:, 1]


# ==========================================================================
# The Shape protocol
# ==========================================================================
class Shape(ABC):
    """An implicit solid: a membership test plus a conservative bounding box.

    Subclasses implement :meth:`_contains` (vectorised, may return an array of
    any shape that broadcasts against the inputs) and :meth:`bbox`.  The public
    :meth:`contains` validates the arguments and normalises the result to the
    full broadcast shape, so every shape in the library behaves identically.

    Notes
    -----
    :meth:`bbox` **must** be a conservative outer bound.  :func:`voxelize` uses
    it to skip grid chunks without evaluating them, so a bound that is too tight
    silently truncates the geometry.  Return an infinite bound when in doubt;
    the only cost is speed.
    """

    # -- interface ---------------------------------------------------------
    @abstractmethod
    def _contains(self, x: np.ndarray, y: np.ndarray,
                  z: np.ndarray) -> np.ndarray:
        """Vectorised membership test on validated float coordinate arrays."""

    @abstractmethod
    def bbox(self) -> BBox:
        """Conservative axis-aligned bound ``((xlo, xhi), ...)`` [m]."""

    def contains(self, x, y, z) -> np.ndarray:
        """Test whether points are inside the solid.

        Parameters
        ----------
        x, y, z : array_like
            Cartesian coordinates [m].  Any shapes that broadcast together;
            scalars are allowed.

        Returns
        -------
        np.ndarray of bool
            Shape is the common broadcast shape of ``x``, ``y`` and ``z``.
            Surfaces are included (see the module docstring on tie-breaks).

        Raises
        ------
        ValueError
            If the three coordinate arrays do not broadcast.
        """
        xa, ya, za = _as_coords(x, y, z)
        out = np.asarray(self._contains(xa, ya, za), dtype=bool)
        full = np.broadcast_shapes(xa.shape, ya.shape, za.shape)
        if out.shape != full:
            # A shape that ignores a coordinate (a slab, an infinite prism)
            # legitimately returns a lower-rank result; materialise it so the
            # caller always gets a writable array of the expected shape.
            out = np.array(np.broadcast_to(out, full))
        return out

    # -- CSG algebra -------------------------------------------------------
    def __or__(self, other: "Shape") -> "Shape":
        return Union(self, other)

    def __and__(self, other: "Shape") -> "Shape":
        return Intersection(self, other)

    def __sub__(self, other: "Shape") -> "Shape":
        return Difference(self, other)

    def __invert__(self) -> "Shape":
        return Complement(self)

    # -- rigid motions -----------------------------------------------------
    def translate(self, delta: Sequence[float]) -> "Shape":
        """Return this shape shifted by ``delta`` [m]."""
        return Transformed(self, np.eye(3), _as_vec3(delta, "delta"))

    def rotate(self, angle: float, axis: str | int = "z",
               center: Sequence[float] = (0.0, 0.0, 0.0)) -> "Shape":
        """Return this shape rotated by ``angle`` [rad] about ``axis``.

        The rotation is right-handed about ``axis`` and passes through
        ``center`` [m].  Rotated geometry is still voxelised on an axis-aligned
        grid, so it is staircased (A2); the sub-cell fill fraction from
        :func:`voxelize` is what keeps that affordable.
        """
        c = _as_vec3(center, "center")
        R = _rotation_matrix(float(angle), _axis_index(axis))
        return Transformed(self, R, c - R @ c)


def _rotation_matrix(angle: float, axis: int) -> np.ndarray:
    """Right-handed rotation matrix about a coordinate axis, angle in radians."""
    c, s = float(np.cos(angle)), float(np.sin(angle))
    u, v = _plane_axes(axis)
    R = np.eye(3)
    R[u, u] = c
    R[u, v] = -s
    R[v, u] = s
    R[v, v] = c
    return R


# ==========================================================================
# CSG combinators
# ==========================================================================
class Union(Shape):
    """Set union of two or more shapes (``a | b``)."""

    def __init__(self, *shapes: Shape):
        flat: list[Shape] = []
        for s in shapes:
            if not isinstance(s, Shape):
                raise ValueError(f"Union operands must be Shape, got {type(s).__name__}")
            # Flattening keeps deep chains of ``|`` from becoming deep recursion
            # during voxelisation, where every level costs a full-size temporary.
            flat.extend(s.shapes if isinstance(s, Union) else [s])
        if not flat:
            raise ValueError("Union needs at least one shape")
        self.shapes: tuple[Shape, ...] = tuple(flat)

    def _contains(self, x, y, z):
        out = self.shapes[0]._contains(x, y, z)
        for s in self.shapes[1:]:
            out = out | s._contains(x, y, z)
        return out

    def bbox(self) -> BBox:
        los, his = zip(*(_bbox_arrays(s.bbox()) for s in self.shapes))
        lo = np.min(np.stack(los), axis=0)
        hi = np.max(np.stack(his), axis=0)
        return tuple((float(a), float(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"Union({', '.join(repr(s) for s in self.shapes)})"


class Intersection(Shape):
    """Set intersection of two or more shapes (``a & b``)."""

    def __init__(self, *shapes: Shape):
        flat: list[Shape] = []
        for s in shapes:
            if not isinstance(s, Shape):
                raise ValueError(
                    f"Intersection operands must be Shape, got {type(s).__name__}")
            flat.extend(s.shapes if isinstance(s, Intersection) else [s])
        if not flat:
            raise ValueError("Intersection needs at least one shape")
        self.shapes: tuple[Shape, ...] = tuple(flat)

    def _contains(self, x, y, z):
        out = self.shapes[0]._contains(x, y, z)
        for s in self.shapes[1:]:
            out = out & s._contains(x, y, z)
        return out

    def bbox(self) -> BBox:
        los, his = zip(*(_bbox_arrays(s.bbox()) for s in self.shapes))
        lo = np.max(np.stack(los), axis=0)
        hi = np.min(np.stack(his), axis=0)
        # An empty intersection is reported as lo > hi; voxelize short-circuits.
        return tuple((float(a), float(b)) for a, b in zip(lo, hi))  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"Intersection({', '.join(repr(s) for s in self.shapes)})"


class Difference(Shape):
    """``a`` with ``b`` removed (``a - b``).  Half-open on the cut surface."""

    def __init__(self, a: Shape, b: Shape):
        if not isinstance(a, Shape) or not isinstance(b, Shape):
            raise ValueError("Difference operands must both be Shape")
        self.a = a
        self.b = b

    def _contains(self, x, y, z):
        return self.a._contains(x, y, z) & ~self.b._contains(x, y, z)

    def bbox(self) -> BBox:
        return self.a.bbox()

    def __repr__(self) -> str:
        return f"Difference({self.a!r}, {self.b!r})"


class Complement(Shape):
    """Everything outside a shape (``~a``).  Unbounded, so its bbox is infinite."""

    def __init__(self, a: Shape):
        if not isinstance(a, Shape):
            raise ValueError("Complement operand must be a Shape")
        self.a = a

    def _contains(self, x, y, z):
        return ~self.a._contains(x, y, z)

    def bbox(self) -> BBox:
        return _INF_BBOX

    def __repr__(self) -> str:
        return f"Complement({self.a!r})"


class Transformed(Shape):
    """A shape under an invertible affine map ``p_world = M p_local + offset``.

    Parameters
    ----------
    shape : Shape
        The shape in its own local frame.
    matrix : array_like, shape (3, 3)
        Linear part.  Must be invertible; rotations and anisotropic scalings
        are both allowed.
    offset : array_like, shape (3,)
        Translation [m].

    Notes
    -----
    The bounding box is the axis-aligned box around the eight transformed
    corners of the child's box --- conservative, and loose for a rotated slab.
    Any infinite child bound makes the whole result infinite, because the
    transformed corner arithmetic is meaningless there.
    """

    def __init__(self, shape: Shape, matrix: np.ndarray,
                 offset: Sequence[float] = (0.0, 0.0, 0.0)):
        if not isinstance(shape, Shape):
            raise ValueError("Transformed needs a Shape")
        M = np.asarray(matrix, dtype=float)
        if M.shape != (3, 3):
            raise ValueError(f"matrix must be (3, 3), got {M.shape}")
        if not np.all(np.isfinite(M)):
            raise ValueError("matrix must be finite")
        det = float(np.linalg.det(M))
        if abs(det) < 1e-300:
            raise ValueError("matrix must be invertible (det is zero)")
        self.shape = shape
        self.matrix = M
        self.offset = _as_vec3(offset, "offset")
        self._inv = np.linalg.inv(M)

    def _contains(self, x, y, z):
        A = self._inv
        dx = x - self.offset[0]
        dy = y - self.offset[1]
        dz = z - self.offset[2]
        xl = A[0, 0] * dx + A[0, 1] * dy + A[0, 2] * dz
        yl = A[1, 0] * dx + A[1, 1] * dy + A[1, 2] * dz
        zl = A[2, 0] * dx + A[2, 1] * dy + A[2, 2] * dz
        return self.shape._contains(xl, yl, zl)

    def bbox(self) -> BBox:
        lo, hi = _bbox_arrays(self.shape.bbox())
        if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
            return _INF_BBOX
        corners = np.array(np.meshgrid(*zip(lo, hi), indexing="ij")).reshape(3, -1)
        world = self.matrix @ corners + self.offset[:, None]
        return tuple((float(a), float(b))  # type: ignore[return-value]
                     for a, b in zip(world.min(axis=1), world.max(axis=1)))

    def __repr__(self) -> str:
        return f"Transformed({self.shape!r})"


def union(*shapes: Shape) -> Shape:
    """Union of any number of shapes."""
    return Union(*shapes)


def intersection(*shapes: Shape) -> Shape:
    """Intersection of any number of shapes."""
    return Intersection(*shapes)


def difference(a: Shape, *rest: Shape) -> Shape:
    """``a`` minus every shape in ``rest``."""
    return Difference(a, Union(*rest)) if rest else a


# ==========================================================================
# Primitives
# ==========================================================================
class Box(Shape):
    """Axis-aligned rectangular block.

    Give either ``center`` and ``size``, or ``lo`` and ``hi``.

    Parameters
    ----------
    center : array_like, shape (3,), optional
        Centre coordinates [m].
    size : array_like, shape (3,), optional
        Full edge lengths [m], all positive.  ``np.inf`` gives a slab that is
        unbounded in that direction, which is the idiomatic way to write a
        layer mask.
    lo, hi : array_like, shape (3,), optional
        Opposite corners [m], with ``lo < hi`` componentwise.  ``+-inf`` is
        allowed.

    Notes
    -----
    A box aligned to grid node planes is voxelised **exactly** at any
    ``subsample``, which is why planar processes cost nothing under assumption
    A2.  Use ``grid`` node coordinates as layer boundaries whenever you can.
    """

    def __init__(self, center: Sequence[float] | None = None,
                 size: Sequence[float] | None = None,
                 *, lo: Sequence[float] | None = None,
                 hi: Sequence[float] | None = None):
        by_corner = lo is not None or hi is not None
        by_center = center is not None or size is not None
        if by_corner and by_center:
            raise ValueError("give either center/size or lo/hi, not both")
        if by_corner:
            if lo is None or hi is None:
                raise ValueError("both lo and hi are required")
            lo_a = np.asarray(lo, dtype=float).ravel()
            hi_a = np.asarray(hi, dtype=float).ravel()
            if lo_a.size != 3 or hi_a.size != 3:
                raise ValueError("lo and hi must have 3 components")
        elif by_center:
            if center is None or size is None:
                raise ValueError("both center and size are required")
            c = _as_vec3(center, "center")
            s = np.asarray(size, dtype=float).ravel()
            if s.size != 3:
                raise ValueError("size must have 3 components")
            if np.any(np.isnan(s)) or np.any(s <= 0.0):
                raise ValueError(f"size components must be positive, got {size}")
            half = 0.5 * s
            lo_a, hi_a = c - half, c + half
        else:
            raise ValueError("Box needs center/size or lo/hi")
        if np.any(np.isnan(lo_a)) or np.any(np.isnan(hi_a)):
            raise ValueError("box bounds must not be NaN")
        if np.any(hi_a <= lo_a):
            raise ValueError(f"box must have positive extent, got lo={lo_a}, hi={hi_a}")
        self.lo = lo_a
        self.hi = hi_a

    @property
    def center(self) -> np.ndarray:
        """Centre [m] (infinite for an unbounded direction)."""
        return 0.5 * (self.lo + self.hi)

    @property
    def size(self) -> np.ndarray:
        """Full edge lengths [m]."""
        return self.hi - self.lo

    def _contains(self, x, y, z):
        lo, hi = self.lo, self.hi
        return ((x >= lo[0]) & (x <= hi[0])
                & (y >= lo[1]) & (y <= hi[1])
                & (z >= lo[2]) & (z <= hi[2]))

    def bbox(self) -> BBox:
        return tuple((float(a), float(b))  # type: ignore[return-value]
                     for a, b in zip(self.lo, self.hi))

    def __repr__(self) -> str:
        return f"Box(lo={self.lo.tolist()}, hi={self.hi.tolist()})"


class Sphere(Shape):
    """Solid sphere.

    Parameters
    ----------
    center : array_like, shape (3,)
        Centre [m].
    radius : float
        Radius [m], positive.
    """

    def __init__(self, center: Sequence[float], radius: float):
        self.center = _as_vec3(center, "center")
        r = float(radius)
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError(f"radius must be finite and positive, got {radius}")
        self.radius = r

    def _contains(self, x, y, z):
        c = self.center
        dx = x - c[0]
        dy = y - c[1]
        dz = z - c[2]
        return dx * dx + dy * dy + dz * dz <= self.radius * self.radius

    def bbox(self) -> BBox:
        c, r = self.center, self.radius
        return tuple((float(ci - r), float(ci + r)) for ci in c)  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"Sphere(center={self.center.tolist()}, radius={self.radius:g})"


class Cylinder(Shape):
    """Right circular cylinder aligned with a coordinate axis.

    Parameters
    ----------
    center : array_like, shape (3,)
        Centroid [m] --- the midpoint of the axis, not an end cap.
    radius : float
        Radius [m], positive.
    height : float, optional
        Full length along ``axis`` [m].  ``np.inf`` (the default) gives an
        infinite cylinder, useful as a via or wire mask inside a
        :class:`LayerStack`.
    axis : {'x', 'y', 'z'} or {0, 1, 2}, optional
        Cylinder axis.  Default ``'z'``.
    """

    def __init__(self, center: Sequence[float], radius: float,
                 height: float = np.inf, axis: str | int = "z"):
        self.center = _as_vec3(center, "center")
        r = float(radius)
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError(f"radius must be finite and positive, got {radius}")
        h = float(height)
        if np.isnan(h) or h <= 0.0:
            raise ValueError(f"height must be positive, got {height}")
        self.radius = r
        self.height = h
        self.axis = _axis_index(axis)

    def _contains(self, x, y, z):
        coords = (x, y, z)
        u, v = _plane_axes(self.axis)
        c = self.center
        du = coords[u] - c[u]
        dv = coords[v] - c[v]
        radial = du * du + dv * dv <= self.radius * self.radius
        if np.isinf(self.height):
            return radial
        half = 0.5 * self.height
        w = coords[self.axis] - c[self.axis]
        return radial & (w >= -half) & (w <= half)

    def bbox(self) -> BBox:
        c, r = self.center, self.radius
        half = 0.5 * self.height
        out = [(float(c[i] - r), float(c[i] + r)) for i in range(3)]
        out[self.axis] = (float(c[self.axis] - half), float(c[self.axis] + half))
        return tuple(out)  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (f"Cylinder(center={self.center.tolist()}, radius={self.radius:g}, "
                f"height={self.height:g}, axis={_AXIS_NAMES[self.axis]!r})")


class HalfSpace(Shape):
    """Everything on the back side of a plane.

    Parameters
    ----------
    point : array_like, shape (3,)
        Any point on the plane [m].
    normal : array_like, shape (3,)
        Plane normal, pointing **out of** the solid.  Need not be normalised.

    Notes
    -----
    The solid is ``{p : normal . (p - point) <= 0}``, so
    ``HalfSpace(p, -n)`` is the closure of ``~HalfSpace(p, n)``: the two share
    the plane itself.  Intersections of half-spaces are the cheapest way to
    build a chamfer or a wedge, which is otherwise a hole in a rectilinear
    CSG kit.
    """

    def __init__(self, point: Sequence[float], normal: Sequence[float]):
        self.point = _as_vec3(point, "point")
        n = _as_vec3(normal, "normal")
        norm = float(np.linalg.norm(n))
        if norm == 0.0:
            raise ValueError("normal must be non-zero")
        self.normal = n / norm

    def _contains(self, x, y, z):
        n, p = self.normal, self.point
        return (n[0] * (x - p[0]) + n[1] * (y - p[1])
                + n[2] * (z - p[2])) <= 0.0

    def bbox(self) -> BBox:
        # Bounded only when the plane is axis-aligned; that is the common case
        # (a substrate top surface) and worth detecting, because it lets
        # voxelize skip whole chunks.
        out: list[tuple[float, float]] = [(-np.inf, np.inf)] * 3
        nz = np.flatnonzero(np.abs(self.normal) > 0.0)
        if nz.size == 1:
            i = int(nz[0])
            if self.normal[i] > 0:
                out[i] = (-np.inf, float(self.point[i]))
            else:
                out[i] = (float(self.point[i]), np.inf)
        return tuple(out)  # type: ignore[return-value]

    def __repr__(self) -> str:
        return f"HalfSpace(point={self.point.tolist()}, normal={self.normal.tolist()})"


class Prism(Shape):
    """A polygon extruded along a coordinate axis.

    This is the workhorse for real layout: a wire, a pad, an L-bend, a guard
    ring cross-section.  The polygon may be non-convex and may be wound either
    way; it must not self-intersect.

    Parameters
    ----------
    vertices : array_like, shape (n, 2)
        Polygon vertices [m] **in the plane perpendicular to** ``axis``, in the
        right-handed cyclic in-plane frame: ``axis='z'`` takes ``(x, y)``,
        ``axis='x'`` takes ``(y, z)``, ``axis='y'`` takes ``(z, x)``.  A
        repeated closing vertex is accepted and dropped.
    height : float, optional
        Extrusion length [m].  Combined with ``center`` to give the extent
        along ``axis``.  Omit both ``height`` and ``lo``/``hi`` for an infinite
        prism (the idiomatic in-plane mask for :class:`LayerStack`).
    center : float, optional
        Midpoint of the extrusion along ``axis`` [m].  Default 0.
    axis : {'x', 'y', 'z'} or {0, 1, 2}, optional
        Extrusion direction.  Default ``'z'``.
    lo, hi : float, optional
        Explicit extent along ``axis`` [m], overriding ``height``/``center``.

    Notes
    -----
    Membership uses the crossing-number (ray casting) rule, evaluated with a
    half-open comparison in the in-plane ``v`` coordinate so that a ray passing
    exactly through a vertex is counted once, not twice.  That rule leaves
    points *on* an edge ambiguous, so an explicit segment-distance test is
    OR-ed in and the polygon is closed: a point on an edge or vertex is inside.
    The tolerance is ``1e-12`` times the polygon diameter, i.e. it is there to
    absorb round-off, not to fatten the shape.
    """

    def __init__(self, vertices: np.ndarray | Sequence[Sequence[float]],
                 height: float | None = None, center: float = 0.0,
                 axis: str | int = "z",
                 lo: float | None = None, hi: float | None = None):
        v = np.asarray(vertices, dtype=float)
        if v.ndim != 2 or v.shape[1] != 2:
            raise ValueError(f"vertices must have shape (n, 2), got {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("vertices must be finite")
        if v.shape[0] >= 2 and np.allclose(v[0], v[-1]):
            v = v[:-1]
        if v.shape[0] < 3:
            raise ValueError(f"a polygon needs at least 3 distinct vertices, got {v.shape[0]}")
        seg = np.roll(v, -1, axis=0) - v
        if np.any(np.all(seg == 0.0, axis=1)):
            raise ValueError("polygon has a repeated consecutive vertex")
        area2 = float(np.sum(v[:, 0] * np.roll(v[:, 1], -1)
                             - np.roll(v[:, 0], -1) * v[:, 1]))
        if abs(area2) == 0.0:
            raise ValueError("polygon is degenerate (zero area)")

        self.axis = _axis_index(axis)
        if lo is not None or hi is not None:
            if lo is None or hi is None:
                raise ValueError("give both lo and hi, or neither")
            lo_f, hi_f = float(lo), float(hi)
        elif height is not None:
            h = float(height)
            if np.isnan(h) or h <= 0.0:
                raise ValueError(f"height must be positive, got {height}")
            lo_f = float(center) - 0.5 * h
            hi_f = float(center) + 0.5 * h
        else:
            lo_f, hi_f = -np.inf, np.inf
        if not hi_f > lo_f:
            raise ValueError(f"prism needs hi > lo, got lo={lo_f}, hi={hi_f}")

        self.vertices = v
        self.lo = lo_f
        self.hi = hi_f
        self.area = 0.5 * abs(area2)
        diameter = float(np.max(v.max(axis=0) - v.min(axis=0)))
        self._tol = 1e-12 * diameter

    @property
    def height(self) -> float:
        """Extrusion length along ``axis`` [m] (``inf`` if unbounded)."""
        return self.hi - self.lo

    def _contains(self, x, y, z):
        coords = (x, y, z)
        iu, iv = _plane_axes(self.axis)
        u, v = coords[iu], coords[iv]
        shape = np.broadcast_shapes(np.shape(u), np.shape(v))
        inside = np.zeros(shape, dtype=bool)
        on_edge = np.zeros(shape, dtype=bool)
        px, py = self.vertices[:, 0], self.vertices[:, 1]
        tol = self._tol
        n = px.size
        for a in range(n):
            b = (a + 1) % n
            x1, y1, x2, y2 = px[a], py[a], px[b], py[b]
            ex, ey = x2 - x1, y2 - y1
            qx, qy = u - x1, v - y1
            if ey != 0.0:
                # Half-open in v: the endpoint with the larger v belongs to the
                # edge, so a ray through a vertex crosses exactly once.
                crossing = (y1 > v) != (y2 > v)
                x_int = x1 + (v - y1) * (ex / ey)
                inside ^= crossing & (u < x_int)
            seg_len = float(np.hypot(ex, ey))
            cross = ex * qy - ey * qx
            dot = qx * ex + qy * ey
            on_edge |= ((np.abs(cross) <= tol * seg_len)
                        & (dot >= -tol * seg_len)
                        & (dot <= seg_len * seg_len + tol * seg_len))
        inside |= on_edge
        if np.isinf(self.lo) and np.isinf(self.hi):
            return inside
        w = coords[self.axis]
        return inside & (w >= self.lo) & (w <= self.hi)

    def bbox(self) -> BBox:
        iu, iv = _plane_axes(self.axis)
        vmin = self.vertices.min(axis=0)
        vmax = self.vertices.max(axis=0)
        out: list[tuple[float, float]] = [(0.0, 0.0)] * 3
        out[iu] = (float(vmin[0]), float(vmax[0]))
        out[iv] = (float(vmin[1]), float(vmax[1]))
        out[self.axis] = (float(self.lo), float(self.hi))
        return tuple(out)  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (f"Prism({self.vertices.shape[0]} vertices, "
                f"axis={_AXIS_NAMES[self.axis]!r}, lo={self.lo:g}, hi={self.hi:g})")


class Torus(Shape):
    """Solid torus with its symmetry axis along a coordinate axis.

    Parameters
    ----------
    center : array_like, shape (3,)
        Centre of the hole [m].
    major_radius : float
        Distance from the centre to the tube axis [m], positive.
    minor_radius : float
        Tube radius [m], positive.  Values above ``major_radius`` give a
        self-intersecting (spindle) torus, which is still a well-defined solid
        and is allowed.
    axis : {'x', 'y', 'z'} or {0, 1, 2}, optional
        Symmetry axis.  Default ``'z'``.

    Notes
    -----
    Present because a single-turn inductor or a loop antenna feed is a torus,
    and because it is the sternest test of the fill-fraction machinery: the
    surface is doubly curved, so a boolean mask staircases it in two directions
    at once.
    """

    def __init__(self, center: Sequence[float], major_radius: float,
                 minor_radius: float, axis: str | int = "z"):
        self.center = _as_vec3(center, "center")
        R = float(major_radius)
        r = float(minor_radius)
        if not np.isfinite(R) or R <= 0.0:
            raise ValueError(f"major_radius must be finite and positive, got {major_radius}")
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError(f"minor_radius must be finite and positive, got {minor_radius}")
        self.major_radius = R
        self.minor_radius = r
        self.axis = _axis_index(axis)

    def _contains(self, x, y, z):
        coords = (x, y, z)
        u, v = _plane_axes(self.axis)
        c = self.center
        du = coords[u] - c[u]
        dv = coords[v] - c[v]
        dw = coords[self.axis] - c[self.axis]
        rho = np.sqrt(du * du + dv * dv) - self.major_radius
        return rho * rho + dw * dw <= self.minor_radius * self.minor_radius

    def bbox(self) -> BBox:
        c = self.center
        R, r = self.major_radius, self.minor_radius
        out = [(float(c[i] - R - r), float(c[i] + R + r)) for i in range(3)]
        out[self.axis] = (float(c[self.axis] - r), float(c[self.axis] + r))
        return tuple(out)  # type: ignore[return-value]

    def __repr__(self) -> str:
        return (f"Torus(center={self.center.tolist()}, R={self.major_radius:g}, "
                f"r={self.minor_radius:g}, axis={_AXIS_NAMES[self.axis]!r})")


# ==========================================================================
# Voxelisation
# ==========================================================================
def _axis_samples(nodes: np.ndarray, h: np.ndarray, frac: np.ndarray,
                  i0: int, i1: int) -> np.ndarray:
    """Sample coordinates [m] for cells ``i0:i1``, ``len(frac)`` per cell.

    Points sit at cell-interior positions ``x_lo + (m + 0.5) * h / s``, i.e. the
    midpoint rule on each sub-cell.  Corner sampling would double-count shared
    faces and bias every fill fraction upward; the half-step offset is what
    makes the estimator unbiased.  Grading is respected because ``h`` is
    per-cell.
    """
    return (nodes[i0:i1, None] + frac[None, :] * h[i0:i1, None]).ravel()


def _chunks(n: int, per: int) -> Iterable[tuple[int, int]]:
    """Split ``range(n)`` into contiguous blocks of at most ``per``."""
    for start in range(0, n, per):
        yield start, min(start + per, n)


def voxelize(grid: RectilinearGrid, shape: Shape, subsample: int = 2,
             *, chunk_points: int = 4_000_000,
             use_bbox: bool = True) -> np.ndarray:
    """Sample a shape onto the grid as a per-cell fill fraction.

    Parameters
    ----------
    grid : RectilinearGrid
        Target grid.  Cell coordinates are in metres.
    shape : Shape
        Solid to sample.  Any CSG tree.
    subsample : int, optional
        Number of sample points per cell **per direction**; ``subsample**3``
        points per cell in total.  ``1`` samples the cell centre only and
        therefore returns a hard 0/1 staircase mask.  Default 2.
    chunk_points : int, keyword-only, optional
        Upper bound on the number of sample points evaluated in one call to
        ``shape.contains``.  Controls peak memory (see Notes).  Default 4e6.
    use_bbox : bool, keyword-only, optional
        Skip grid chunks that lie outside ``shape.bbox()`` without evaluating
        them.  A large speed-up for small features on a padded domain.  Turn it
        off if a custom :class:`Shape` may report a bounding box that is not a
        conservative superset.

    Returns
    -------
    np.ndarray
        Shape ``grid.shape_cells`` ``(Nx, Ny, Nz)``, dtype float64, values in
        ``[0, 1]``: the fraction of each cell's volume occupied by ``shape``.

    Raises
    ------
    ValueError
        If ``shape`` is not a :class:`Shape`, if ``subsample < 1``, or if a
        custom shape returns a result of the wrong size.

    Notes
    -----
    **Why a fraction (assumption A2).**  The rectilinear grid staircases every
    non-axis-aligned surface, which costs roughly one order of convergence on
    that surface.  The fraction returned here is the input to effective-medium
    mixing in :class:`~fieldspice.materials.MaterialMap`, which recovers much
    of that loss for smooth interfaces.  Axis-aligned geometry --- most of
    microelectronics --- is captured exactly at any ``subsample``, so the extra
    sampling costs nothing there but also buys nothing.

    **Accuracy.**  The estimator is a midpoint quadrature of the indicator
    function.  Cells entirely inside or outside are exact; only cells cut by
    the surface carry error.  For a cut perpendicular to a grid axis the
    computed fraction is exactly ``round(s*f)/s``, so the per-cell error is
    bounded by ``1/(2*subsample)`` and that bound is attained --- first order,
    and it is the worst case.  For a surface tilted or curved with respect to
    the grid the sub-cell errors within one cell equidistribute and largely
    cancel, and the measured rate is closer to ``O(1/subsample**2)``.  Cost
    grows as ``subsample**3``, so ``subsample=2`` (8 points) is the sensible
    default, 3 or 4 for a curved surface that matters, above 6 almost never
    pays for itself compared with refining the grid.

    The *total* volume of a shape converges faster still and non-monotonically,
    because over- and under-counted cells cancel across the surface.  Do not
    use a volume check to tune ``subsample``; it flatters the result.

    **Memory.**  Coordinates are built once per chunk with
    ``np.meshgrid(..., indexing="ij", sparse=True)``, so the three coordinate
    arrays are 1-D and cost nothing; the full-size temporaries are those created
    inside ``contains``.  The domain is chunked along the slowest axis (x, and
    then y if a single x-slab is still over budget) so that no more than
    ``chunk_points`` sample points are live at once.  The transient working set
    is then roughly ``chunk_points`` times (8 bytes per live float64 temporary
    plus 1 byte per live bool), on top of the ``8 * n_cells`` byte result.
    Measured on a ``200**3`` grid at ``subsample=4`` --- 5.1e8 sample points ---
    with a three-node CSG tree: 158 MB of peak growth at the default
    ``chunk_points``, of which 61 MB is the result array itself, against 4.9 GB
    with chunking disabled.

    z is deliberately never chunked, because splitting the fastest axis would
    break the contiguity of the sub-cell reduction; a grid whose z extent alone
    exceeds the budget (``subsample**3 * Nz`` points) will overshoot it.

    Examples
    --------
    >>> from fieldspice.grid import RectilinearGrid
    >>> g = RectilinearGrid.uniform([(0, 1), (0, 1), (0, 1)], [4, 4, 4])
    >>> f = voxelize(g, Box(lo=(0, 0, 0), hi=(0.5, 1, 1)), subsample=1)
    >>> f.sum() * (1 / 4) ** 3          # volume [m^3]
    0.5
    """
    if not isinstance(grid, RectilinearGrid):
        raise ValueError(f"grid must be a RectilinearGrid, got {type(grid).__name__}")
    if not isinstance(shape, Shape):
        raise ValueError(f"shape must be a Shape, got {type(shape).__name__}")
    if isinstance(subsample, bool) or not isinstance(subsample, (int, np.integer)):
        raise ValueError(f"subsample must be an integer, got {subsample!r}")
    s = int(subsample)
    if s < 1:
        raise ValueError(f"subsample must be >= 1, got {s}")
    if not isinstance(chunk_points, (int, np.integer)) or chunk_points < 1:
        raise ValueError(f"chunk_points must be a positive integer, got {chunk_points!r}")

    nx, ny, nz = grid.shape_cells
    out = np.zeros((nx, ny, nz), dtype=np.float64)

    lo, hi = _bbox_arrays(shape.bbox())
    if use_bbox and np.any(hi < lo):
        return out  # provably empty (an intersection of disjoint shapes)

    xn, yn, zn = grid.xn, grid.yn, grid.zn
    if use_bbox and (zn[-1] < lo[2] or zn[0] > hi[2]):
        return out

    frac = (np.arange(s, dtype=float) + 0.5) / s
    zs = _axis_samples(zn, grid.hz, frac, 0, nz)

    # Chunk the slowest axis first; only if one x-slab already exceeds the
    # budget do we also chunk y.  z is never chunked, so the innermost stride
    # stays contiguous and the reduction stays fast.
    per_x_slab = s * (ny * s) * (nz * s)
    nx_per = min(nx, max(1, int(chunk_points // max(per_x_slab, 1))))
    if per_x_slab > chunk_points:
        per_xy_slab = s * s * (nz * s)
        ny_per = min(ny, max(1, int(chunk_points // max(per_xy_slab, 1))))
    else:
        ny_per = ny

    inv = 1.0 / float(s ** 3)
    for i0, i1 in _chunks(nx, nx_per):
        if use_bbox and (xn[i1] < lo[0] or xn[i0] > hi[0]):
            continue
        xs = _axis_samples(xn, grid.hx, frac, i0, i1)
        for j0, j1 in _chunks(ny, ny_per):
            if use_bbox and (yn[j1] < lo[1] or yn[j0] > hi[1]):
                continue
            ys = _axis_samples(yn, grid.hy, frac, j0, j1)
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij", sparse=True)
            inside = shape.contains(X, Y, Z)
            if inside.size != xs.size * ys.size * zs.size:
                raise ValueError(
                    f"{type(shape).__name__}.contains returned {inside.size} values, "
                    f"expected {xs.size * ys.size * zs.size}")
            blk = np.reshape(inside, (i1 - i0, s, j1 - j0, s, nz, s))
            out[i0:i1, j0:j1, :] = blk.sum(axis=(1, 3, 5), dtype=np.int64) * inv
    return out


# ==========================================================================
# Planar process stack
# ==========================================================================
@dataclass(frozen=True)
class Layer:
    """One deposited layer of a :class:`LayerStack`.

    Attributes
    ----------
    name : str
        Layer identifier.
    material : str or Material
        Passed straight through to :meth:`MaterialMap.assign`.
    lo, hi : float
        Extent along the stack axis [m].
    mask : Shape or None
        In-plane pattern.  ``None`` means a blanket (unpatterned) layer.
    axis : int
        Stack axis index (0, 1 or 2).
    """

    name: str
    material: "str | Material"
    lo: float
    hi: float
    mask: Shape | None
    axis: int

    @property
    def thickness(self) -> float:
        """Layer thickness [m]."""
        return self.hi - self.lo

    def solid(self) -> Shape:
        """The 3-D solid this layer occupies: its slab, intersected with the mask."""
        lo = [-np.inf, -np.inf, -np.inf]
        hi = [np.inf, np.inf, np.inf]
        lo[self.axis] = self.lo
        hi[self.axis] = self.hi
        slab = Box(lo=lo, hi=hi)
        return slab if self.mask is None else Intersection(slab, self.mask)


class LayerStack:
    """A planar fabrication process: layers stacked along one axis.

    This is the common case for anything fabricated --- CMOS, a thin-film
    transistor, a PCB cross-section --- and it is also the case where assumption
    A2 costs nothing, because every interface is a plane perpendicular to the
    stack axis and can be made to coincide exactly with a grid node plane.
    :meth:`check_alignment` tells you whether it does.

    Parameters
    ----------
    grid : RectilinearGrid
        Grid the stack will be applied to.
    axis : {'x', 'y', 'z'} or {0, 1, 2}, optional
        Growth direction.  Default ``'z'``.
    origin : float, optional
        Coordinate [m] of the bottom of the first layer.  Default 0.

    Examples
    --------
    >>> from fieldspice.grid import RectilinearGrid
    >>> g = RectilinearGrid.uniform([(0, 4e-6), (0, 4e-6), (0, 3e-7)],
    ...                             [8, 8, 6])
    >>> st = LayerStack(g, axis="z", origin=0.0)
    >>> _ = st.add("gate", 100e-9, "poly")
    >>> _ = st.add("oxide", 20e-9, "sio2")
    >>> _ = st.add("channel", 30e-9, "igzo",
    ...            mask=Box(lo=(1e-6, 1e-6, -np.inf), hi=(3e-6, 3e-6, np.inf)))
    >>> st.z_of("oxide")
    (1e-07, 1.2e-07)
    """

    def __init__(self, grid: RectilinearGrid, axis: str | int = "z",
                 origin: float = 0.0):
        if not isinstance(grid, RectilinearGrid):
            raise ValueError(f"grid must be a RectilinearGrid, got {type(grid).__name__}")
        o = float(origin)
        if not np.isfinite(o):
            raise ValueError(f"origin must be finite, got {origin}")
        self.grid = grid
        self.axis = _axis_index(axis)
        self.origin = o
        self._layers: list[Layer] = []

    # -- construction ------------------------------------------------------
    def add(self, name: str, thickness: float, material: "str | Material",
            mask: Shape | None = None) -> Layer:
        """Deposit a layer on top of the stack.

        Parameters
        ----------
        name : str
            Unique identifier.
        thickness : float
            Layer thickness [m], positive and finite.
        material : str or Material
            Handed to :meth:`MaterialMap.assign` unchanged, so either a library
            name or a ``Material`` instance works.
        mask : Shape or None, optional
            In-plane pattern, intersected with the layer's slab.  Use a shape
            that is unbounded along the stack axis (a :class:`Prism` with no
            height, or a :class:`Box` with ``+-inf`` bounds) so the mask only
            constrains the in-plane extent.  ``None`` deposits a blanket layer.

        Returns
        -------
        Layer
            The layer just added, so callers can keep a handle on its extent.

        Raises
        ------
        ValueError
            On a duplicate name, a non-positive thickness, or a mask that is
            not a :class:`Shape`.
        """
        if not isinstance(name, str) or not name:
            raise ValueError("layer name must be a non-empty string")
        if any(l.name == name for l in self._layers):
            raise ValueError(f"duplicate layer name {name!r}")
        t = float(thickness)
        if not np.isfinite(t) or t <= 0.0:
            raise ValueError(f"thickness must be finite and positive, got {thickness}")
        if mask is not None and not isinstance(mask, Shape):
            raise ValueError(f"mask must be a Shape or None, got {type(mask).__name__}")
        lo = self.top
        layer = Layer(name=name, material=material, lo=lo, hi=lo + t,
                      mask=mask, axis=self.axis)
        self._layers.append(layer)

        g_lo, g_hi = self.grid.bounds[self.axis]
        if layer.hi > g_hi + 1e-15 * max(abs(g_hi), 1.0) or layer.lo < g_lo:
            # Silently growing past the domain wall is the classic way to lose
            # half a stack; the layer is still recorded so the geometry stays
            # self-consistent, but the user gets told.
            warnings.warn(
                f"layer {name!r} spans [{layer.lo:g}, {layer.hi:g}] m along "
                f"{_AXIS_NAMES[self.axis]} but the grid covers "
                f"[{g_lo:g}, {g_hi:g}] m; it will be clipped by the domain",
                stacklevel=2)
        return layer

    # -- queries -----------------------------------------------------------
    @property
    def layers(self) -> tuple[Layer, ...]:
        """The layers, bottom first."""
        return tuple(self._layers)

    @property
    def names(self) -> tuple[str, ...]:
        """Layer names, bottom first."""
        return tuple(l.name for l in self._layers)

    @property
    def top(self) -> float:
        """Coordinate of the top of the stack along the stack axis [m]."""
        return self._layers[-1].hi if self._layers else self.origin

    @property
    def total_thickness(self) -> float:
        """Sum of all layer thicknesses [m]."""
        return self.top - self.origin

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        return iter(self._layers)

    def __getitem__(self, name: str) -> Layer:
        for l in self._layers:
            if l.name == name:
                return l
        raise ValueError(f"no layer named {name!r}; have {list(self.names)}")

    def z_of(self, name: str) -> tuple[float, float]:
        """``(lo, hi)`` extent of a layer along the stack axis [m].

        Named ``z_of`` because the stack axis is z in almost every process; it
        reports the extent along whichever axis the stack was built on.
        """
        l = self[name]
        return (l.lo, l.hi)

    def layer_at(self, coord: float) -> Layer | None:
        """The layer containing ``coord`` [m] on the stack axis, or ``None``.

        Boundaries belong to the lower layer, matching the closed-shape rule.
        """
        c = float(coord)
        for l in self._layers:
            if l.lo <= c <= l.hi:
                return l
        return None

    def solid(self, name: str) -> Shape:
        """The 3-D :class:`Shape` occupied by one layer."""
        return self[name].solid()

    def voxelize(self, name: str, subsample: int = 2, **kw) -> np.ndarray:
        """Fill fraction of one layer on the grid, shape ``grid.shape_cells``."""
        return voxelize(self.grid, self.solid(name), subsample, **kw)

    # -- diagnostics -------------------------------------------------------
    def check_alignment(self, rtol: float = 1e-9) -> list[str]:
        """Report layer boundaries that do not sit on a grid node plane.

        A planar process is represented *exactly* on a rectilinear grid only if
        every interface coincides with a node plane; otherwise the interface is
        smeared across a cell and assumption A2 starts costing accuracy even
        though the geometry itself is axis-aligned.  This is the cheapest
        correctness check available on a stack, so run it.

        Parameters
        ----------
        rtol : float, optional
            Relative tolerance, scaled by the grid extent along the stack axis.

        Returns
        -------
        list of str
            One human-readable message per misaligned boundary.  Empty means
            the stack is exactly representable.
        """
        nodes = (self.grid.xn, self.grid.yn, self.grid.zn)[self.axis]
        g_lo, g_hi = self.grid.bounds[self.axis]
        tol = rtol * max(abs(g_hi - g_lo), 1e-300)
        msgs: list[str] = []
        for l in self._layers:
            for what, coord in (("bottom", l.lo), ("top", l.hi)):
                if np.min(np.abs(nodes - coord)) > tol:
                    nearest = float(nodes[int(np.argmin(np.abs(nodes - coord)))])
                    msgs.append(
                        f"layer {l.name!r} {what} at {coord:.6g} m is not on a "
                        f"grid node plane (nearest node {nearest:.6g} m)")
        return msgs

    def summary(self) -> str:
        """Human-readable table of the stack, top layer first."""
        lines = [f"LayerStack along {_AXIS_NAMES[self.axis]}, "
                 f"origin {self.origin:g} m, {len(self._layers)} layers"]
        for l in reversed(self._layers):
            mat = getattr(l.material, "name", l.material)
            pat = "blanket" if l.mask is None else type(l.mask).__name__
            lines.append(f"  {l.name:<16s} {l.thickness:>10.4g} m  "
                         f"[{l.lo:.4g}, {l.hi:.4g}]  {mat}  {pat}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<LayerStack {len(self._layers)} layers, "
                f"{self.total_thickness:g} m along {_AXIS_NAMES[self.axis]}>")

    # -- application -------------------------------------------------------
    def apply(self, matmap: "MaterialMap", subsample: int = 2,
              binary: bool = False, threshold: float = 0.5) -> None:
        """Write every layer into a :class:`~fieldspice.materials.MaterialMap`.

        Layers are applied bottom first, so a later layer wins wherever two
        overlap.  Stacked layers never overlap along the axis, so this only
        matters when a mask is deliberately reused.

        Parameters
        ----------
        matmap : MaterialMap
            Target map.  Must be built on the same grid.
        subsample : int, optional
            Sampling density passed to :func:`voxelize`.  Default 2.
        binary : bool, optional
            If ``True``, hand ``MaterialMap.assign`` a boolean mask
            (``fraction >= threshold``) instead of the fill fraction, discarding
            the sub-cell information that partially cancels the staircase error
            of A2.  Default ``False``.
        threshold : float, optional
            Cutoff used when ``binary`` is ``True``.  Default 0.5.

        Raises
        ------
        ValueError
            If ``matmap`` exposes a ``grid`` that is not this stack's grid, or
            if it has no ``assign`` method.
        """
        assign = getattr(matmap, "assign", None)
        if not callable(assign):
            raise ValueError("matmap must expose an assign(mask, material) method")
        mm_grid = getattr(matmap, "grid", None)
        if mm_grid is not None and mm_grid is not self.grid:
            raise ValueError("matmap was built on a different grid than this LayerStack")
        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"threshold must lie in [0, 1], got {threshold}")

        for layer in self._layers:
            frac = voxelize(self.grid, layer.solid(), subsample)
            if binary:
                assign(frac >= float(threshold), layer.material)
                continue
            try:
                assign(frac, layer.material)
            except (TypeError, IndexError, ValueError) as exc:
                # MaterialMap is written independently; if this build of it only
                # accepts boolean masks, fall back rather than fail, but say so,
                # because the fallback throws away the sub-cell mixing that
                # makes fill fractions worth computing at all.
                warnings.warn(
                    f"MaterialMap.assign rejected a fill-fraction array "
                    f"({exc}); falling back to a boolean mask at threshold "
                    f"{threshold:g}, which forfeits sub-cell material mixing. "
                    f"Pass binary=True to silence this.",
                    stacklevel=2)
                assign(frac >= float(threshold), layer.material)
