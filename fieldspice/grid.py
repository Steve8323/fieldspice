"""Graded rectilinear Yee grid --- the geometric substrate for every solver.

Staggering convention
---------------------
fieldspice uses the standard Yee / discrete-exterior-calculus arrangement.
With ``(Nx, Ny, Nz)`` **cells**, the primal grid holds:

===========  =====================================  ==========================
Location     Shape                                  Quantities
===========  =====================================  ==========================
node (0D)    ``(Nx+1, Ny+1, Nz+1)``                 potential phi, n, p, doping
edge (1D)    x: ``(Nx,   Ny+1, Nz+1)``              E circulation, A, current
             y: ``(Nx+1, Ny,   Nz+1)``
             z: ``(Nx+1, Ny+1, Nz)``
face (2D)    x: ``(Nx+1, Ny,   Nz)``                B flux, H circulation
             y: ``(Nx,   Ny+1, Nz)``
             z: ``(Nx,   Ny,   Nz+1)``
cell (3D)    ``(Nx, Ny, Nz)``                       material id, permittivity
===========  =====================================  ==========================

An x-edge with index ``(i, j, k)`` connects node ``(i, j, k)`` to node
``(i+1, j, k)``; its length is ``hx[i]``.  An x-face with index ``(i, j, k)``
is the face normal to x sitting in the node-plane ``i``, spanning cell ``j``
in y and cell ``k`` in z; its area is ``hy[j] * hz[k]``.

Integrated variables
--------------------
Every field quantity is stored **integrated over its geometric element**, not
as a point value:

* node: potential ``phi`` [V]
* edge: voltage drop ``e = integral E.dl`` [V], vector potential
  ``a = integral A.dl`` [Wb], current ``i`` [A]
* face: magnetic flux ``b = integral B.dA`` [Wb], mmf ``h = integral H.dl`` [A]

This is what makes the discrete operators in :mod:`fieldspice.operators` pure
signed-incidence matrices with entries in ``{-1, 0, +1}`` and **no** geometry
in them at all.  All metric information lives in diagonal mass matrices whose
entries are ordinary circuit elements (conductance in S, capacitance in F,
reluctance in 1/H).  See ``docs/FORMULATION.md``.

Dimensionality
--------------
The grid is *always* three-dimensional internally.  A 2D problem is a grid with
``Nz == 1``; a 1D problem has ``Ny == Nz == 1``.  Collapsed directions carry a
finite user-specified thickness (default 1 m) so that extracted quantities are
naturally "per unit length" / "per unit area", and they default to homogeneous
Neumann boundaries, which is exactly the translational-symmetry assumption a
2D cross-section solve implies.  Keeping one code path for 1/2/3D removes a
large class of indexing bugs at negligible cost, because a collapsed direction
contributes no coupling to the sparse operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

__all__ = ["RectilinearGrid", "graded_1d", "auto_mesh_1d"]


# ==========================================================================
# Mesh generation helpers
# ==========================================================================
def graded_1d(a: float, b: float, n: int, ratio: float = 1.0,
              reverse: bool = False) -> np.ndarray:
    """``n+1`` node coordinates from ``a`` to ``b`` with geometric grading.

    ``ratio`` is the multiplicative growth of successive cell widths.
    ``ratio == 1`` gives a uniform mesh.  ``ratio > 1`` grows away from ``a``;
    set ``reverse=True`` to grow away from ``b`` instead.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if ratio <= 0:
        raise ValueError("ratio must be positive")
    if abs(ratio - 1.0) < 1e-12:
        w = np.ones(n)
    else:
        w = ratio ** np.arange(n)
    if reverse:
        w = w[::-1]
    w = w / w.sum() * (b - a)
    return np.concatenate([[a], a + np.cumsum(w)])


def auto_mesh_1d(extent: tuple[float, float],
                 features: Sequence[float] = (),
                 dx_min: float | None = None,
                 dx_max: float | None = None,
                 growth: float = 1.4,
                 feature_dx: Sequence[float] | None = None) -> np.ndarray:
    """Build a graded 1D mesh that is fine at ``features`` and coarse between.

    This is the workhorse for circuit geometry, where the ratio of the smallest
    to largest interesting length is routinely 1e4 (a 2 nm gate oxide inside a
    50 um die).  A uniform mesh at the fine scale is not merely slow, it is
    impossible; grading is mandatory, not an optimisation.

    Parameters
    ----------
    extent
        ``(lo, hi)`` domain bounds.
    features
        Coordinates that must be resolved (material interfaces, junctions,
        conductor surfaces).  Each becomes a node.
    dx_min
        Cell size adjacent to a feature.  Defaults to ``span / 2000``.
    dx_max
        Largest permitted cell.  Defaults to ``span / 20``.
    growth
        Geometric growth ratio moving away from a feature.  Values above ~1.5
        degrade the second-order accuracy of the operators noticeably; 1.2-1.4
        is the usual sweet spot.
    feature_dx
        Optional per-feature override of ``dx_min``.

    Returns
    -------
    np.ndarray
        Sorted, deduplicated node coordinates including both endpoints and
        every feature coordinate.
    """
    lo, hi = float(extent[0]), float(extent[1])
    if hi <= lo:
        raise ValueError("extent must be increasing")
    span = hi - lo
    dx_min = span / 2000.0 if dx_min is None else float(dx_min)
    dx_max = span / 20.0 if dx_max is None else float(dx_max)
    if dx_max < dx_min:
        dx_max = dx_min
    growth = max(1.0 + 1e-6, float(growth))

    feats = sorted({lo, hi, *(float(f) for f in features
                              if lo - 1e-15 <= f <= hi + 1e-15)})
    if feature_dx is None:
        fdx = {f: dx_min for f in feats}
    else:
        fdx = {f: dx_min for f in feats}
        for f, d in zip(features, feature_dx):
            fdx[float(f)] = float(d)

    nodes: list[float] = []
    for left, right in zip(feats[:-1], feats[1:]):
        seg = right - left
        dl, dr = fdx.get(left, dx_min), fdx.get(right, dx_min)
        pts = _grade_segment(left, right, dl, dr, dx_max, growth)
        nodes.extend(pts[:-1] if right != feats[-1] else pts)
        del seg
    nodes = np.array(sorted(set(np.round(nodes, 15))))
    return nodes


def _grade_segment(left: float, right: float, dl: float, dr: float,
                   dx_max: float, growth: float) -> np.ndarray:
    """Nodes on ``[left, right]``, fine at both ends, capped at ``dx_max``."""
    span = right - left
    if span <= 0:
        return np.array([left, right])
    # Grow from each end until cells meet or hit dx_max, then fill uniformly.
    def _ramp(d0: float) -> list[float]:
        out, d, tot = [], min(d0, span), 0.0
        while tot + d < span * 0.5 and d < dx_max:
            out.append(d)
            tot += d
            d = min(d * growth, dx_max)
        return out

    lr, rr = _ramp(dl), _ramp(dr)
    used = sum(lr) + sum(rr)
    mid = span - used
    if mid < 0:  # ramps overlap; fall back to a simple two-sided grading
        n = max(2, int(np.ceil(span / max(dl, dr, span / 200))))
        return np.linspace(left, right, n + 1)
    n_mid = max(1, int(np.ceil(mid / dx_max)))
    widths = lr + [mid / n_mid] * n_mid + rr[::-1]
    return np.concatenate([[left], left + np.cumsum(widths)])


# ==========================================================================
# The grid
# ==========================================================================
@dataclass(frozen=True)
class _Axis:
    """One coordinate direction: node positions plus derived spacings."""
    nodes: np.ndarray       # (N+1,) node coordinates
    h: np.ndarray           # (N,)   primal cell widths
    hd: np.ndarray          # (N+1,) dual widths, centred on nodes
    collapsed: bool         # True if this direction is a single symmetric cell


def _make_axis(nodes: Sequence[float] | np.ndarray | None,
               thickness: float) -> _Axis:
    if nodes is None:
        nodes_arr = np.array([0.0, float(thickness)])
        collapsed = True
    else:
        nodes_arr = np.asarray(nodes, dtype=float).ravel()
        collapsed = nodes_arr.size == 2
        if nodes_arr.size < 2:
            raise ValueError("an axis needs at least 2 node coordinates")
        if np.any(np.diff(nodes_arr) <= 0):
            raise ValueError("node coordinates must be strictly increasing")
    h = np.diff(nodes_arr)
    # Dual (node-centred) widths: half cells at the two boundaries.
    hd = np.empty(nodes_arr.size)
    hd[1:-1] = 0.5 * (h[:-1] + h[1:])
    hd[0] = 0.5 * h[0]
    hd[-1] = 0.5 * h[-1]
    return _Axis(nodes_arr, h, hd, collapsed)


class RectilinearGrid:
    """A graded, axis-aligned, tensor-product grid.

    Parameters
    ----------
    x, y, z
        Node coordinates along each axis (arrays of length ``N+1``).  Pass
        ``None`` for a collapsed (symmetry) direction.
    thickness
        ``(tx, ty, tz)`` extents used for collapsed directions.  Defaults to
        1 m, which makes results per-metre in that direction.

    Notes
    -----
    Only rectilinear tensor-product grids are supported.  Curved and slanted
    boundaries are therefore *staircased*.  This is the single largest accuracy
    compromise in fieldspice and is discussed in ``docs/ASSUMPTIONS.md`` (A2);
    it costs roughly first-order convergence on the staircased surface itself
    while retaining second order in the bulk.  In exchange, every operator is a
    banded matrix with a known stencil, assembly is O(N) with no mesh
    generator, and there is no element-quality failure mode.
    """

    def __init__(self,
                 x: Sequence[float] | np.ndarray,
                 y: Sequence[float] | np.ndarray | None = None,
                 z: Sequence[float] | np.ndarray | None = None,
                 thickness: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        self.ax = _make_axis(x, thickness[0])
        self.ay = _make_axis(y, thickness[1])
        self.az = _make_axis(z, thickness[2])
        self._axes = (self.ax, self.ay, self.az)

    # -- basic sizes -------------------------------------------------------
    @property
    def Nx(self) -> int: return self.ax.h.size

    @property
    def Ny(self) -> int: return self.ay.h.size

    @property
    def Nz(self) -> int: return self.az.h.size

    @property
    def ncell(self) -> tuple[int, int, int]:
        return (self.Nx, self.Ny, self.Nz)

    @property
    def ndim_effective(self) -> int:
        """Number of directions that are actually resolved (1, 2 or 3)."""
        return sum(0 if a.collapsed else 1 for a in self._axes) or 1

    # -- element counts ----------------------------------------------------
    @property
    def shape_nodes(self) -> tuple[int, int, int]:
        return (self.Nx + 1, self.Ny + 1, self.Nz + 1)

    @property
    def shape_cells(self) -> tuple[int, int, int]:
        return (self.Nx, self.Ny, self.Nz)

    @property
    def shape_edges(self) -> tuple[tuple[int, int, int], ...]:
        """Shapes of the x-, y- and z-directed edge arrays."""
        return ((self.Nx, self.Ny + 1, self.Nz + 1),
                (self.Nx + 1, self.Ny, self.Nz + 1),
                (self.Nx + 1, self.Ny + 1, self.Nz))

    @property
    def shape_faces(self) -> tuple[tuple[int, int, int], ...]:
        """Shapes of the x-, y- and z-normal face arrays."""
        return ((self.Nx + 1, self.Ny, self.Nz),
                (self.Nx, self.Ny + 1, self.Nz),
                (self.Nx, self.Ny, self.Nz + 1))

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.shape_nodes))

    @property
    def n_cells(self) -> int:
        return int(np.prod(self.shape_cells))

    @property
    def n_edges_each(self) -> tuple[int, int, int]:
        return tuple(int(np.prod(s)) for s in self.shape_edges)  # type: ignore

    @property
    def n_edges(self) -> int:
        return int(sum(self.n_edges_each))

    @property
    def n_faces_each(self) -> tuple[int, int, int]:
        return tuple(int(np.prod(s)) for s in self.shape_faces)  # type: ignore

    @property
    def n_faces(self) -> int:
        return int(sum(self.n_faces_each))

    # -- coordinates -------------------------------------------------------
    @property
    def xn(self) -> np.ndarray: return self.ax.nodes

    @property
    def yn(self) -> np.ndarray: return self.ay.nodes

    @property
    def zn(self) -> np.ndarray: return self.az.nodes

    @property
    def xc(self) -> np.ndarray: return 0.5 * (self.ax.nodes[:-1] + self.ax.nodes[1:])

    @property
    def yc(self) -> np.ndarray: return 0.5 * (self.ay.nodes[:-1] + self.ay.nodes[1:])

    @property
    def zc(self) -> np.ndarray: return 0.5 * (self.az.nodes[:-1] + self.az.nodes[1:])

    @property
    def hx(self) -> np.ndarray: return self.ax.h

    @property
    def hy(self) -> np.ndarray: return self.ay.h

    @property
    def hz(self) -> np.ndarray: return self.az.h

    @property
    def hxd(self) -> np.ndarray: return self.ax.hd

    @property
    def hyd(self) -> np.ndarray: return self.ay.hd

    @property
    def hzd(self) -> np.ndarray: return self.az.hd

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple((a.nodes[0], a.nodes[-1]) for a in self._axes)  # type: ignore

    # -- metric arrays -----------------------------------------------------
    def cell_volumes(self) -> np.ndarray:
        """``(Nx, Ny, Nz)`` primal cell volumes [m^3]."""
        return (self.hx[:, None, None] * self.hy[None, :, None]
                * self.hz[None, None, :])

    def node_volumes(self) -> np.ndarray:
        """``(Nx+1, Ny+1, Nz+1)`` dual (box/Voronoi) volumes [m^3].

        This is the control volume used by the box-method finite-volume
        discretisation of Poisson and the continuity equations.  It sums
        exactly to the domain volume.
        """
        return (self.hxd[:, None, None] * self.hyd[None, :, None]
                * self.hzd[None, None, :])

    def edge_lengths(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Primal edge lengths [m], one array per direction."""
        sx, sy, sz = self.shape_edges
        lx = np.broadcast_to(self.hx[:, None, None], sx)
        ly = np.broadcast_to(self.hy[None, :, None], sy)
        lz = np.broadcast_to(self.hz[None, None, :], sz)
        return np.ascontiguousarray(lx), np.ascontiguousarray(ly), \
            np.ascontiguousarray(lz)

    def edge_dual_areas(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dual face areas pierced by each primal edge [m^2].

        The product ``sigma * dual_area / length`` is the conductance of the
        resistor that the edge represents; ``eps * dual_area / length`` is its
        capacitance.  This is the entire metric content of the electroquasi-
        static solver.
        """
        sx, sy, sz = self.shape_edges
        ax = np.broadcast_to(self.hyd[None, :, None] * self.hzd[None, None, :], sx)
        ay = np.broadcast_to(self.hxd[:, None, None] * self.hzd[None, None, :], sy)
        az = np.broadcast_to(self.hxd[:, None, None] * self.hyd[None, :, None], sz)
        return (np.ascontiguousarray(ax), np.ascontiguousarray(ay),
                np.ascontiguousarray(az))

    def face_areas(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Primal face areas [m^2]."""
        sx, sy, sz = self.shape_faces
        ax = np.broadcast_to(self.hy[None, :, None] * self.hz[None, None, :], sx)
        ay = np.broadcast_to(self.hx[:, None, None] * self.hz[None, None, :], sy)
        az = np.broadcast_to(self.hx[:, None, None] * self.hy[None, :, None], sz)
        return (np.ascontiguousarray(ax), np.ascontiguousarray(ay),
                np.ascontiguousarray(az))

    def face_dual_lengths(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dual edge lengths threading each primal face [m]."""
        sx, sy, sz = self.shape_faces
        lx = np.broadcast_to(self.hxd[:, None, None], sx)
        ly = np.broadcast_to(self.hyd[None, :, None], sy)
        lz = np.broadcast_to(self.hzd[None, None, :], sz)
        return (np.ascontiguousarray(lx), np.ascontiguousarray(ly),
                np.ascontiguousarray(lz))

    # -- index helpers -----------------------------------------------------
    def node_index(self, i, j, k) -> np.ndarray:
        """Flat C-order index of node ``(i, j, k)``."""
        nx, ny, nz = self.shape_nodes
        return np.ravel_multi_index((i, j, k), (nx, ny, nz))

    def cell_index(self, i, j, k) -> np.ndarray:
        return np.ravel_multi_index((i, j, k), self.shape_cells)

    def edge_index(self, direction: int, i, j, k) -> np.ndarray:
        """Flat index of an edge in the concatenated ``[ex, ey, ez]`` vector."""
        off = (0, self.n_edges_each[0],
               self.n_edges_each[0] + self.n_edges_each[1])[direction]
        return off + np.ravel_multi_index((i, j, k), self.shape_edges[direction])

    def face_index(self, direction: int, i, j, k) -> np.ndarray:
        off = (0, self.n_faces_each[0],
               self.n_faces_each[0] + self.n_faces_each[1])[direction]
        return off + np.ravel_multi_index((i, j, k), self.shape_faces[direction])

    def node_coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Broadcast node coordinate arrays, each ``(Nx+1, Ny+1, Nz+1)``."""
        return np.meshgrid(self.xn, self.yn, self.zn, indexing="ij")

    def cell_coords(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Broadcast cell-centre coordinate arrays, each ``(Nx, Ny, Nz)``."""
        return np.meshgrid(self.xc, self.yc, self.zc, indexing="ij")

    def nearest_node(self, pt: Sequence[float]) -> tuple[int, int, int]:
        """Index of the node closest to a physical point."""
        return tuple(int(np.argmin(np.abs(a.nodes - p)))
                     for a, p in zip(self._axes, pt))  # type: ignore

    # -- quality / diagnostics --------------------------------------------
    def max_aspect_ratio(self) -> float:
        """Largest cell aspect ratio present, ignoring collapsed directions."""
        hs = [a.h for a in self._axes if not a.collapsed]
        if len(hs) < 2:
            return 1.0
        lo = min(float(h.min()) for h in hs)
        hi = max(float(h.max()) for h in hs)
        return hi / lo

    def max_growth_ratio(self) -> float:
        """Largest ratio between neighbouring cell widths.

        Values above ~1.5 measurably degrade second-order convergence; the
        solvers emit a warning above 2.0.
        """
        worst = 1.0
        for a in self._axes:
            if a.h.size < 2:
                continue
            r = a.h[1:] / a.h[:-1]
            worst = max(worst, float(np.max(np.maximum(r, 1.0 / r))))
        return worst

    def courant_dt(self, eps_r_min: float = 1.0, mu_r_min: float = 1.0) -> float:
        """Largest stable explicit-FDTD time step [s] for this grid.

        Only the full-wave solver is bound by this.  It is exposed on the grid
        so that quasi-static runs can *report* the speedup they buy: the ratio
        of the quasi-static step actually used to this number is the honest
        measure of what the approximation is worth.
        """
        v = 1.0 / np.sqrt(eps_r_min * mu_r_min)  # in units of c0
        from .units import c0
        inv2 = 0.0
        for a in self._axes:
            if not a.collapsed:
                inv2 += 1.0 / float(a.h.min()) ** 2
        if inv2 == 0.0:
            inv2 = 1.0 / float(self.ax.h.min()) ** 2
        return 1.0 / (c0 * v * np.sqrt(inv2))

    def summary(self) -> str:
        nx, ny, nz = self.ncell
        return (f"RectilinearGrid {nx}x{ny}x{nz} cells "
                f"({self.ndim_effective}D effective)\n"
                f"  nodes {self.n_nodes:,}  edges {self.n_edges:,}  "
                f"faces {self.n_faces:,}  cells {self.n_cells:,}\n"
                f"  extent  x [{self.xn[0]:.4g}, {self.xn[-1]:.4g}] m\n"
                f"          y [{self.yn[0]:.4g}, {self.yn[-1]:.4g}] m\n"
                f"          z [{self.zn[0]:.4g}, {self.zn[-1]:.4g}] m\n"
                f"  min cell {min(self.hx.min(), self.hy.min(), self.hz.min()):.4g} m"
                f"  max growth {self.max_growth_ratio():.3f}\n"
                f"  explicit-FDTD Courant dt {self.courant_dt():.4g} s")

    def __repr__(self) -> str:
        nx, ny, nz = self.ncell
        return f"<RectilinearGrid {nx}x{ny}x{nz}, {self.n_nodes:,} nodes>"

    # -- convenience constructors -----------------------------------------
    @classmethod
    def uniform(cls, extent: Sequence[tuple[float, float]],
                n: Sequence[int],
                thickness: tuple[float, float, float] = (1.0, 1.0, 1.0)
                ) -> "RectilinearGrid":
        """Uniform grid over ``extent`` with ``n`` cells per direction.

        ``extent`` and ``n`` may have length 1, 2 or 3; missing directions are
        collapsed.
        """
        axes: list[np.ndarray | None] = [None, None, None]
        for d, (ext, ncell) in enumerate(zip(extent, n)):
            axes[d] = np.linspace(ext[0], ext[1], int(ncell) + 1)
        return cls(axes[0], axes[1], axes[2], thickness=thickness)
