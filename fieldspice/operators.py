"""Discrete vector calculus on the Yee grid: incidence and mass matrices.

The whole numerical scheme rests on a split that is worth stating plainly,
because it is the reason fieldspice can be simultaneously a field solver and a
circuit solver:

**Topology and metric are separated.**

* The *topological* operators --- gradient, curl, divergence --- act on
  integrated quantities (node potentials, edge circulations, face fluxes) and
  are therefore exact signed-incidence matrices with entries in
  ``{-1, 0, +1}``.  They contain no grid spacing whatsoever, they are exact
  (not approximations), and they satisfy ``curl.grad = 0`` and
  ``div.curl = 0`` to machine precision on any grid, however distorted.

* All *metric* information --- lengths, areas, material properties --- lives in
  diagonal **mass matrices**.  On a rectilinear grid these are literally
  circuit elements::

      M_sigma[e] = sigma_e * A_dual_e / L_e     conductance  [S]
      M_eps[e]   = eps_e   * A_dual_e / L_e     capacitance  [F]
      M_nu[f]    = A_f / sum_c(mu_c * dL_c)     inverse inductance [1/H]

Consequences that matter in practice:

1. Charge is conserved to machine precision, because ``div`` of a discrete
   current is an exact telescoping sum, not a truncated Taylor series.
2. ``div B = 0`` holds identically for all time in the magnetic solvers, since
   ``B`` is only ever updated through a ``curl``.
3. The electroquasistatic system ``G^T M_sigma G phi = I`` *is* nodal analysis
   of a resistor mesh.  A field region and a SPICE netlist are the same kind of
   object, so coupling them is assembly, not interpolation.

Reference: Weiland's Finite Integration Technique; Bossavit's Whitney-form
finite elements; Tonti's cell method.  On a rectilinear grid all three agree
with each other and with Yee's original scheme.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .grid import RectilinearGrid

__all__ = [
    "grad_node_edge", "curl_edge_face", "div_face_cell",
    "edge_mass", "face_mass_nu", "node_volume_vector",
    "cell_to_edge", "cell_to_face", "cell_to_node",
    "nodal_laplacian", "curl_curl", "apply_dirichlet",
    "edge_vector_from_components", "split_edge_vector",
    "interpolate_edges_to_nodes", "Operators",
]


# ==========================================================================
# Incidence (topological) operators --- exact, metric-free
# ==========================================================================
def _ids(shape: tuple[int, int, int], offset: int = 0) -> np.ndarray:
    return (offset + np.arange(int(np.prod(shape)))).reshape(shape)


def _assemble(rows_cols_vals, shape) -> sp.csr_matrix:
    rows = np.concatenate([np.asarray(r).ravel() for r, _, _ in rows_cols_vals])
    cols = np.concatenate([np.asarray(c).ravel() for _, c, _ in rows_cols_vals])
    vals = np.concatenate([np.full(np.asarray(c).size, v, dtype=np.float64)
                           for _, c, v in rows_cols_vals])
    return sp.coo_matrix((vals, (rows, cols)), shape=shape).tocsr()


def grad_node_edge(grid: RectilinearGrid) -> sp.csr_matrix:
    """Discrete gradient ``G``: nodes -> edges, shape ``(n_edges, n_nodes)``.

    ``(G phi)_e = phi_head - phi_tail`` --- the potential *rise* along the edge
    direction.  The electric field circulation is ``e = -G phi``.

    Entries are exactly +-1: no grid spacing appears.  This is what makes
    ``curl_edge_face(g) @ grad_node_edge(g)`` vanish to machine precision.
    """
    nid = _ids(grid.shape_nodes)
    sx, sy, sz = grid.shape_edges
    nex, ney, _ = grid.n_edges_each
    ex = _ids(sx, 0)
    ey = _ids(sy, nex)
    ez = _ids(sz, nex + ney)
    terms = [
        (ex, nid[1:, :, :], +1.0), (ex, nid[:-1, :, :], -1.0),
        (ey, nid[:, 1:, :], +1.0), (ey, nid[:, :-1, :], -1.0),
        (ez, nid[:, :, 1:], +1.0), (ez, nid[:, :, :-1], -1.0),
    ]
    return _assemble(terms, (grid.n_edges, grid.n_nodes))


def curl_edge_face(grid: RectilinearGrid) -> sp.csr_matrix:
    """Discrete curl ``C``: edges -> faces, shape ``(n_faces, n_edges)``.

    ``(C e)_f`` is the closed line integral of E around the boundary of face
    ``f``, taken right-handed with respect to the face normal.  Stokes'
    theorem is therefore satisfied *exactly*, and Faraday's law
    ``db/dt = -C e`` conserves ``div b`` identically because ``D C = 0``.
    """
    sx, sy, sz = grid.shape_edges
    fx, fy, fz = grid.shape_faces
    nex, ney, _ = grid.n_edges_each
    nfx, nfy, _ = grid.n_faces_each
    ex, ey, ez = _ids(sx, 0), _ids(sy, nex), _ids(sz, nex + ney)
    Fx, Fy, Fz = _ids(fx, 0), _ids(fy, nfx), _ids(fz, nfx + nfy)
    terms = [
        # (curl e)_x = dez/dy - dey/dz
        (Fx, ez[:, 1:, :], +1.0), (Fx, ez[:, :-1, :], -1.0),
        (Fx, ey[:, :, :-1], +1.0), (Fx, ey[:, :, 1:], -1.0),
        # (curl e)_y = dex/dz - dez/dx
        (Fy, ex[:, :, 1:], +1.0), (Fy, ex[:, :, :-1], -1.0),
        (Fy, ez[:-1, :, :], +1.0), (Fy, ez[1:, :, :], -1.0),
        # (curl e)_z = dey/dx - dex/dy
        (Fz, ey[1:, :, :], +1.0), (Fz, ey[:-1, :, :], -1.0),
        (Fz, ex[:, :-1, :], +1.0), (Fz, ex[:, 1:, :], -1.0),
    ]
    return _assemble(terms, (grid.n_faces, grid.n_edges))


def div_face_cell(grid: RectilinearGrid) -> sp.csr_matrix:
    """Discrete divergence ``D``: faces -> cells, shape ``(n_cells, n_faces)``.

    ``(D b)_c`` is the net outward flux from cell ``c`` --- an exact
    telescoping sum, so Gauss' theorem holds to machine precision and
    ``D @ curl_edge_face(g)`` is exactly zero.
    """
    fx, fy, fz = grid.shape_faces
    nfx, nfy, _ = grid.n_faces_each
    Fx, Fy, Fz = _ids(fx, 0), _ids(fy, nfx), _ids(fz, nfx + nfy)
    cid = _ids(grid.shape_cells)
    terms = [
        (cid, Fx[1:, :, :], +1.0), (cid, Fx[:-1, :, :], -1.0),
        (cid, Fy[:, 1:, :], +1.0), (cid, Fy[:, :-1, :], -1.0),
        (cid, Fz[:, :, 1:], +1.0), (cid, Fz[:, :, :-1], -1.0),
    ]
    return _assemble(terms, (grid.n_cells, grid.n_faces))


# ==========================================================================
# Material averaging: cells -> edges / faces / nodes
# ==========================================================================
def cell_to_edge(grid: RectilinearGrid, prop: np.ndarray,
                 mode: str = "parallel") -> np.ndarray:
    """Average a per-cell property onto edges.

    Returns the concatenated ``[x, y, z]`` edge vector of length
    ``grid.n_edges``.

    An x-edge is threaded by up to four cell quadrants whose cross-sections
    tile its dual area.  For a property that governs transport *along* the edge
    (sigma, eps), those quadrants act as four elements **in parallel**, so the
    correct combination is the dual-area-weighted arithmetic mean --- this is
    ``mode="parallel"`` and is the default.

    ``mode="harmonic"`` gives the area-weighted harmonic mean, appropriate when
    the edge crosses a thin series barrier (a gate oxide sampled by one edge, a
    contact resistance layer).  ``mode="min"`` and ``mode="max"`` are provided
    for conservative bracketing of staircase error.

    Boundary cells are replicated outwards, which is equivalent to assuming the
    material continues past the domain wall --- consistent with the default
    homogeneous Neumann boundary.
    """
    prop = np.asarray(prop, dtype=float)
    if prop.shape != grid.shape_cells:
        raise ValueError(f"prop must have shape {grid.shape_cells}, got {prop.shape}")
    p = np.pad(prop, 1, mode="edge")  # (Nx+2, Ny+2, Nz+2)

    hy_, hz_ = grid.hy, grid.hz
    hx_ = grid.hx
    # Half-width contributions of the two cells flanking each node plane.
    def _halves(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(lo, hi) half widths at each of the N+1 node planes."""
        lo = np.concatenate([[0.0], 0.5 * h])          # cell below the plane
        hi = np.concatenate([0.5 * h, [0.0]])          # cell above the plane
        return lo, hi

    ylo, yhi = _halves(hy_)
    zlo, zhi = _halves(hz_)
    xlo, xhi = _halves(hx_)

    out = []
    # --- x edges: index (i, j, k), cells (i, j-1|j, k-1|k) ---------------
    sx = grid.shape_edges[0]
    q = np.empty((4,) + sx)
    w = np.empty((4,) + sx)
    c = p[1:-1, :, :]  # x-cell index aligned: c[i] is cell i
    combos = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for m, (dj, dk) in enumerate(combos):
        q[m] = c[:, dj:dj + sx[1], dk:dk + sx[2]]
        wy = (ylo if dj == 0 else yhi)[None, :, None]
        wz = (zlo if dk == 0 else zhi)[None, None, :]
        w[m] = np.broadcast_to(wy * wz, sx)
    out.append(_combine(q, w, mode))

    # --- y edges: index (i, j, k), cells (i-1|i, j, k-1|k) ---------------
    sy = grid.shape_edges[1]
    q = np.empty((4,) + sy)
    w = np.empty((4,) + sy)
    c = p[:, 1:-1, :]
    for m, (di, dk) in enumerate(combos):
        q[m] = c[di:di + sy[0], :, dk:dk + sy[2]]
        wx = (xlo if di == 0 else xhi)[:, None, None]
        wz = (zlo if dk == 0 else zhi)[None, None, :]
        w[m] = np.broadcast_to(wx * wz, sy)
    out.append(_combine(q, w, mode))

    # --- z edges: index (i, j, k), cells (i-1|i, j-1|j, k) ---------------
    sz = grid.shape_edges[2]
    q = np.empty((4,) + sz)
    w = np.empty((4,) + sz)
    c = p[:, :, 1:-1]
    for m, (di, dj) in enumerate(combos):
        q[m] = c[di:di + sz[0], dj:dj + sz[1], :]
        wx = (xlo if di == 0 else xhi)[:, None, None]
        wy = (ylo if dj == 0 else yhi)[None, :, None]
        w[m] = np.broadcast_to(wx * wy, sz)
    out.append(_combine(q, w, mode))

    return np.concatenate([o.ravel() for o in out])


def _combine(q: np.ndarray, w: np.ndarray, mode: str) -> np.ndarray:
    wsum = w.sum(axis=0)
    safe = np.where(wsum > 0, wsum, 1.0)
    if mode == "parallel" or mode == "arithmetic":
        return (q * w).sum(axis=0) / safe
    if mode == "harmonic" or mode == "series":
        qq = np.where(q > 0, q, np.finfo(float).tiny)
        return safe / (w / qq).sum(axis=0)
    if mode == "min":
        return q.min(axis=0)
    if mode == "max":
        return q.max(axis=0)
    raise ValueError(f"unknown averaging mode {mode!r}")


def cell_to_face(grid: RectilinearGrid, prop: np.ndarray,
                 mode: str = "series") -> np.ndarray:
    """Average a per-cell property onto faces (concatenated ``[x, y, z]``).

    The dual edge threading a face passes through exactly two cells, in series
    along the magnetic path, so the default is the length-weighted **harmonic**
    mean --- the rule that makes reluctances add.
    """
    prop = np.asarray(prop, dtype=float)
    if prop.shape != grid.shape_cells:
        raise ValueError(f"prop must have shape {grid.shape_cells}")
    p = np.pad(prop, 1, mode="edge")
    out = []
    for d, (shape, h) in enumerate(zip(grid.shape_faces,
                                       (grid.hx, grid.hy, grid.hz))):
        lo = np.concatenate([[0.0], 0.5 * h])
        hi = np.concatenate([0.5 * h, [0.0]])
        sl_lo = [slice(1, -1)] * 3
        sl_hi = [slice(1, -1)] * 3
        sl_lo[d] = slice(0, shape[d])
        sl_hi[d] = slice(1, shape[d] + 1)
        q = np.stack([p[tuple(sl_lo)], p[tuple(sl_hi)]])
        shp = [1, 1, 1]
        shp[d] = -1
        w = np.stack([np.broadcast_to(lo.reshape(shp), shape),
                      np.broadcast_to(hi.reshape(shp), shape)])
        out.append(_combine(q, w, mode))
    return np.concatenate([o.ravel() for o in out])


def cell_to_node(grid: RectilinearGrid, prop: np.ndarray) -> np.ndarray:
    """Volume-weighted average of a per-cell property onto nodes.

    Used for quantities that belong to the dual control volume: net doping,
    charge density, generation rate.  Returns shape ``grid.shape_nodes``.
    """
    prop = np.asarray(prop, dtype=float)
    if prop.shape != grid.shape_cells:
        raise ValueError(f"prop must have shape {grid.shape_cells}")
    p = np.pad(prop, 1, mode="edge")
    acc = np.zeros(grid.shape_nodes)
    wacc = np.zeros(grid.shape_nodes)
    halves = []
    for h in (grid.hx, grid.hy, grid.hz):
        halves.append((np.concatenate([[0.0], 0.5 * h]),
                       np.concatenate([0.5 * h, [0.0]])))
    nx, ny, nz = grid.shape_nodes
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                q = p[di:di + nx, dj:dj + ny, dk:dk + nz]
                w = (halves[0][di][:, None, None]
                     * halves[1][dj][None, :, None]
                     * halves[2][dk][None, None, :])
                acc += q * w
                wacc += w
    return acc / np.where(wacc > 0, wacc, 1.0)


# ==========================================================================
# Mass matrices (metric + material)
# ==========================================================================
def edge_mass(grid: RectilinearGrid, prop_edge: np.ndarray) -> sp.dia_matrix:
    """Diagonal edge mass matrix ``diag(prop * A_dual / L)``.

    With ``prop = sigma`` the entries are edge **conductances** [S]; with
    ``prop = eps`` they are edge **capacitances** [F].  The resulting operator
    ``G^T M G`` is exactly the nodal admittance matrix of the equivalent
    resistor (or capacitor) mesh.
    """
    prop_edge = np.asarray(prop_edge, dtype=float).ravel()
    if prop_edge.size != grid.n_edges:
        raise ValueError(f"expected {grid.n_edges} edge values, got {prop_edge.size}")
    L = np.concatenate([a.ravel() for a in grid.edge_lengths()])
    A = np.concatenate([a.ravel() for a in grid.edge_dual_areas()])
    return sp.diags(prop_edge * A / L, format="dia")


def face_mass_nu(grid: RectilinearGrid, mu_face: np.ndarray) -> sp.dia_matrix:
    """Diagonal face mass matrix ``diag(A_f / (mu_f * L_dual_f))`` [1/H].

    This is the inverse magnetic inductance of the flux tube through the face.
    ``C^T M_nu C`` is the discrete curl-curl (reluctance) operator.
    """
    mu_face = np.asarray(mu_face, dtype=float).ravel()
    if mu_face.size != grid.n_faces:
        raise ValueError(f"expected {grid.n_faces} face values")
    A = np.concatenate([a.ravel() for a in grid.face_areas()])
    L = np.concatenate([a.ravel() for a in grid.face_dual_lengths()])
    return sp.diags(A / (mu_face * L), format="dia")


def node_volume_vector(grid: RectilinearGrid) -> np.ndarray:
    """Flat vector of dual (box) volumes, one per node [m^3]."""
    return grid.node_volumes().ravel()


# ==========================================================================
# Composite operators
# ==========================================================================
def nodal_laplacian(grid: RectilinearGrid, prop_edge: np.ndarray,
                    G: sp.csr_matrix | None = None) -> sp.csr_matrix:
    """``G^T diag(prop*A/L) G`` --- symmetric positive-semidefinite [S] or [F].

    With ``prop = sigma`` this is the conductance matrix of the mesh and
    ``L phi = I_inject`` is Kirchhoff's current law.  With ``prop = eps`` it is
    the capacitance matrix and ``L phi = Q_node`` is Gauss' law.  The single
    null vector (the constant) is removed by any Dirichlet electrode.
    """
    G = grad_node_edge(grid) if G is None else G
    M = edge_mass(grid, prop_edge)
    return (G.T @ M @ G).tocsr()


def curl_curl(grid: RectilinearGrid, mu_face: np.ndarray,
              C: sp.csr_matrix | None = None) -> sp.csr_matrix:
    """``C^T diag(A/(mu*L)) C`` --- the discrete reluctance operator [1/H]."""
    C = curl_edge_face(grid) if C is None else C
    M = face_mass_nu(grid, mu_face)
    return (C.T @ M @ C).tocsr()


def apply_dirichlet(A: sp.spmatrix, b: np.ndarray,
                    fixed: np.ndarray, values: np.ndarray,
                    symmetric: bool = True) -> tuple[sp.csr_matrix, np.ndarray]:
    """Impose ``x[fixed] = values`` while keeping ``A`` symmetric.

    The fixed columns are multiplied out into the right-hand side before the
    corresponding rows and columns are replaced by identity rows.  Preserving
    symmetry matters: it lets the solvers use CG / Cholesky instead of GMRES /
    LU, which is a 3-10x difference on realistic grids.

    Parameters
    ----------
    A, b
        System to modify (not modified in place).
    fixed
        Integer indices, or a boolean mask of length ``A.shape[0]``.
    values
        Prescribed values, broadcast against ``fixed``.

    Returns
    -------
    (A_bc, b_bc)
    """
    A = sp.csr_matrix(A, copy=True)
    b = np.array(b, dtype=float, copy=True)
    n = A.shape[0]
    fixed = np.asarray(fixed)
    if fixed.dtype == bool:
        idx = np.flatnonzero(fixed)
    else:
        idx = fixed.astype(np.intp)
    vals = np.broadcast_to(np.asarray(values, dtype=float), idx.shape)

    x0 = np.zeros(n)
    x0[idx] = vals
    b -= A @ x0

    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    Dm = sp.diags(mask.astype(float))
    A = Dm @ A @ Dm
    if not symmetric:
        A = sp.csr_matrix(A)
    A = A + sp.diags((~mask).astype(float))
    b[idx] = vals
    return sp.csr_matrix(A), b


# ==========================================================================
# Vector packing helpers
# ==========================================================================
def edge_vector_from_components(grid: RectilinearGrid,
                                fx: np.ndarray, fy: np.ndarray,
                                fz: np.ndarray) -> np.ndarray:
    """Pack three shaped edge arrays into one flat edge vector."""
    shapes = grid.shape_edges
    for a, s, nm in zip((fx, fy, fz), shapes, "xyz"):
        if np.shape(a) != s:
            raise ValueError(f"{nm}-edge array must have shape {s}, got {np.shape(a)}")
    return np.concatenate([np.asarray(fx).ravel(), np.asarray(fy).ravel(),
                           np.asarray(fz).ravel()])


def split_edge_vector(grid: RectilinearGrid, v: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inverse of :func:`edge_vector_from_components`."""
    nx, ny, nz = grid.n_edges_each
    sx, sy, sz = grid.shape_edges
    return (v[:nx].reshape(sx), v[nx:nx + ny].reshape(sy),
            v[nx + ny:].reshape(sz))


def split_face_vector(grid: RectilinearGrid, v: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = grid.n_faces_each
    sx, sy, sz = grid.shape_faces
    return (v[:nx].reshape(sx), v[nx:nx + ny].reshape(sy),
            v[nx + ny:].reshape(sz))


def interpolate_edges_to_nodes(grid: RectilinearGrid, v: np.ndarray,
                               integrated: bool = True) -> np.ndarray:
    """Interpolate an edge vector to a nodal ``(Nn, 3)`` vector field.

    Set ``integrated=True`` (default) if ``v`` holds edge *circulations* (units
    of V or Wb); it is divided by the edge length to recover the field.  Use
    ``integrated=False`` if ``v`` already holds field values.

    For plotting and post-processing only --- never feed the result back into a
    solver, because averaging destroys the exactness of the incidence
    operators.
    """
    ex, ey, ez = split_edge_vector(grid, np.asarray(v, dtype=float))
    if integrated:
        lx, ly, lz = grid.edge_lengths()
        ex, ey, ez = ex / lx, ey / ly, ez / lz
    nx, ny, nz = grid.shape_nodes
    out = np.zeros((nx, ny, nz, 3))
    # x-component: average the two x-edges meeting at each node (one at ends).
    acc = np.zeros((nx, ny, nz)); cnt = np.zeros((nx, ny, nz))
    acc[:-1] += ex; cnt[:-1] += 1
    acc[1:] += ex; cnt[1:] += 1
    out[..., 0] = acc / np.maximum(cnt, 1)
    acc[:] = 0; cnt[:] = 0
    acc[:, :-1] += ey; cnt[:, :-1] += 1
    acc[:, 1:] += ey; cnt[:, 1:] += 1
    out[..., 1] = acc / np.maximum(cnt, 1)
    acc[:] = 0; cnt[:] = 0
    acc[:, :, :-1] += ez; cnt[:, :, :-1] += 1
    acc[:, :, 1:] += ez; cnt[:, :, 1:] += 1
    out[..., 2] = acc / np.maximum(cnt, 1)
    return out


# ==========================================================================
# Cached operator bundle
# ==========================================================================
class Operators:
    """Lazily-built, cached incidence operators for one grid.

    Building ``G``, ``C`` and ``D`` costs a few hundred milliseconds on a
    million-node grid, and every solver needs them, so they are shared.
    """

    def __init__(self, grid: RectilinearGrid):
        self.grid = grid
        self._G: sp.csr_matrix | None = None
        self._C: sp.csr_matrix | None = None
        self._D: sp.csr_matrix | None = None

    @property
    def G(self) -> sp.csr_matrix:
        if self._G is None:
            self._G = grad_node_edge(self.grid)
        return self._G

    @property
    def C(self) -> sp.csr_matrix:
        if self._C is None:
            self._C = curl_edge_face(self.grid)
        return self._C

    @property
    def D(self) -> sp.csr_matrix:
        if self._D is None:
            self._D = div_face_cell(self.grid)
        return self._D

    def check_identities(self, tol: float = 0.0) -> dict[str, float]:
        """Verify ``C G = 0`` and ``D C = 0``.  Both should be exactly zero."""
        cg = abs((self.C @ self.G)).max() if (self.C @ self.G).nnz else 0.0
        dc = abs((self.D @ self.C)).max() if (self.D @ self.C).nnz else 0.0
        return {"max|C@G|": float(cg), "max|D@C|": float(dc)}
