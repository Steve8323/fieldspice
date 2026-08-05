"""Plotting helpers --- matplotlib only, lazily imported, never opinionated.

Three rules govern every function here, and they exist because a plotting layer
that breaks them turns a field solver into a liar:

1. **matplotlib is imported inside the functions, never at module scope.**
   ``import fieldspice`` must succeed on a headless build machine with no
   plotting stack installed; only the moment someone actually asks for a figure
   is matplotlib required.
2. **Nothing global is touched.**  No ``plt.show()``, no ``plt.style.use``, no
   ``rcParams`` mutation, and no new figure is created when the caller supplies
   an ``ax``.  Every function takes ``ax=None`` and returns the ``Axes`` it drew
   on, so figures compose into subplot grids without surprises.
3. **The mesh is drawn as it is, not as it would be convenient.**  fieldspice
   grids are graded by construction --- a 2 nm gate oxide inside a 50 um die is
   the normal case, not an edge case --- so field maps use
   ``pcolormesh`` with the true node coordinates as the quadrilateral geometry.
   ``imshow`` is never used anywhere in this module: it would resample a 1e4
   dynamic-range graded mesh onto uniform pixels and silently misplace every
   interface in the picture.

Colour conventions
------------------
Signed quantities (potential differences, field components, charge) use
``RdBu_r`` with a zero-centred symmetric range, so the sign of a feature is
readable from its colour alone.  Non-negative magnitudes use ``viridis``.
``jet`` is never used: it is not perceptually uniform and invents banding
structure that is not in the data.

Units
-----
All coordinates are metres and all field values are in whatever SI unit the
solver produced (V for potential, V/m for interpolated E, A for current).
Nothing is rescaled; axis labels state the unit.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .grid import RectilinearGrid
from .operators import interpolate_edges_to_nodes

__all__ = [
    "plot_grid", "plot_materials", "plot_scalar", "plot_vector",
    "plot_terminals", "plot_iv", "animate", "plot_convergence",
    "eye_diagram",
]

_AXIS_NAMES = ("x", "y", "z")


# ==========================================================================
# Lazy backend access
# ==========================================================================
def _pyplot():
    """Import ``matplotlib.pyplot`` on demand.

    Deferring the import is what lets ``fieldspice`` be a dependency of a
    headless pipeline.  The error message names the fix rather than leaking a
    bare ``ModuleNotFoundError`` from three frames down.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "fieldspice.viz needs matplotlib; install it with "
            "'pip install matplotlib'. The rest of fieldspice works without it."
        ) from exc
    return plt


def _get_ax(ax, figsize: tuple[float, float] = (6.0, 4.5)):
    """Return ``(ax, created)``; only build a figure when none was supplied."""
    if ax is not None:
        return ax, False
    plt = _pyplot()
    _, ax = plt.subplots(figsize=figsize)
    return ax, True


# ==========================================================================
# Grid / slice bookkeeping
# ==========================================================================
def _axis_of(plane: str | int) -> int:
    """Normal-axis number (0, 1, 2) from ``"x"``/``"y"``/``"z"`` or an int."""
    if isinstance(plane, (int, np.integer)):
        a = int(plane)
        if a not in (0, 1, 2):
            raise ValueError(f"plane index must be 0, 1 or 2, got {a}")
        return a
    if not isinstance(plane, str):
        raise ValueError(f"plane must be 'x', 'y', 'z' or 0/1/2, got {plane!r}")
    key = plane.strip().lower()
    if key not in _AXIS_NAMES:
        raise ValueError(f"plane must be 'x', 'y' or 'z', got {plane!r}")
    return _AXIS_NAMES.index(key)


def _in_plane_axes(axis: int) -> tuple[int, int]:
    """The two in-plane axes, ordered (horizontal, vertical)."""
    return tuple(a for a in (0, 1, 2) if a != axis)  # type: ignore[return-value]


def _node_axis(grid: RectilinearGrid, axis: int) -> np.ndarray:
    return (grid.xn, grid.yn, grid.zn)[axis]


def _cell_axis(grid: RectilinearGrid, axis: int) -> np.ndarray:
    return (grid.xc, grid.yc, grid.zc)[axis]


def _resolve_index(index: int | None, n: int, what: str) -> int:
    """Validate and normalise a slice index into ``range(n)`` (negatives allowed)."""
    if n < 1:
        raise ValueError(f"cannot slice: the grid has no {what} along this axis")
    if index is None:
        return n // 2
    idx = int(index)
    if idx < 0:
        idx += n
    if not 0 <= idx < n:
        raise ValueError(
            f"{what} index {index} out of range for {n} {what}s along this axis")
    return idx


def _as_node_field(grid: RectilinearGrid, field: np.ndarray) -> np.ndarray:
    """Coerce a flat or shaped nodal field to ``grid.shape_nodes``."""
    arr = np.asarray(field)
    if arr.dtype == object:
        raise ValueError("field must be a numeric array")
    arr = arr.astype(float, copy=False)
    if arr.shape == grid.shape_nodes:
        return arr
    if arr.size == grid.n_nodes and arr.ndim == 1:
        return arr.reshape(grid.shape_nodes)
    raise ValueError(
        f"nodal field must have shape {grid.shape_nodes} or be flat with "
        f"{grid.n_nodes} entries, got shape {arr.shape}")


def _as_cell_field(grid: RectilinearGrid, field: np.ndarray) -> np.ndarray:
    """Coerce a flat or shaped cell field to ``grid.shape_cells``."""
    arr = np.asarray(field)
    if arr.shape == grid.shape_cells:
        return arr
    if arr.size == grid.n_cells and arr.ndim == 1:
        return arr.reshape(grid.shape_cells)
    raise ValueError(
        f"cell field must have shape {grid.shape_cells} or be flat with "
        f"{grid.n_cells} entries, got shape {arr.shape}")


def _label(axis: int) -> str:
    return f"{_AXIS_NAMES[axis]} [m]"


def _plane_title(grid: RectilinearGrid, axis: int, index: int,
                 nodal: bool) -> str:
    coord = (_node_axis(grid, axis) if nodal else _cell_axis(grid, axis))[index]
    kind = "node" if nodal else "cell"
    return f"{_AXIS_NAMES[axis]} = {coord:.4g} m ({kind} {index})"


def _finish_axes(ax, axis: int, aspect: str | float,
                 title: str | None) -> None:
    u, w = _in_plane_axes(axis)
    ax.set_xlabel(_label(u))
    ax.set_ylabel(_label(w))
    if aspect is not None:
        ax.set_aspect(aspect)
    if title:
        ax.set_title(title)


# ==========================================================================
# Colour mapping
# ==========================================================================
def _scalar_norm(data: np.ndarray, log: bool, cmap: str | None,
                 vmin: float | None, vmax: float | None,
                 symmetric: bool | None) -> tuple[np.ndarray, str, Any]:
    """Choose data masking, colormap and norm for a scalar field.

    Returns ``(plot_data, cmap_name, norm)``.  ``plot_data`` may be a masked
    array when ``log`` is requested on data containing non-positive values.

    Rules
    -----
    * Signed data (strictly straddling zero) gets ``RdBu_r`` with
      ``vmin = -vmax`` so that white is exactly zero.  Anything else gets
      ``viridis``.
    * ``log=True`` on all-positive data gives ``LogNorm``; zeros and negatives
      are masked out so the colorbar is not silently clipped.
    * ``log=True`` on signed data gives ``SymLogNorm``, whose linear window
      ``linthresh`` is the smallest non-zero magnitude present, clamped to
      ``[1e-12, 0.1] * max|data|`` so the norm cannot degenerate.
    """
    from matplotlib import colors as mcolors

    finite = data[np.isfinite(data)]
    if finite.size == 0:
        raise ValueError("field contains no finite values; nothing to plot")
    dmin, dmax = float(finite.min()), float(finite.max())
    if symmetric is None:
        symmetric = dmin < 0.0 < dmax
    if cmap is None:
        cmap = "RdBu_r" if symmetric else "viridis"

    if not log:
        if symmetric:
            amax = max(abs(dmin), abs(dmax)) or 1.0
            lo = -amax if vmin is None else float(vmin)
            hi = amax if vmax is None else float(vmax)
        else:
            lo = dmin if vmin is None else float(vmin)
            hi = dmax if vmax is None else float(vmax)
        lo, hi = _widen(lo, hi)
        return data, cmap, mcolors.Normalize(vmin=lo, vmax=hi)

    mag = np.abs(finite)
    nz = mag[mag > 0]
    if nz.size == 0:
        warnings.warn("log=True but the field is identically zero; "
                      "falling back to a linear scale", RuntimeWarning,
                      stacklevel=3)
        return _scalar_norm(data, False, cmap, vmin, vmax, symmetric)
    amax = float(nz.max())

    if dmin < 0.0:
        lin = float(nz.min())
        lin = min(max(lin, amax * 1e-12), amax * 0.1)
        lim = amax if vmax is None else abs(float(vmax))
        return (data, cmap,
                mcolors.SymLogNorm(linthresh=lin, vmin=-lim, vmax=lim, base=10))

    masked = np.ma.masked_less_equal(data, 0.0)
    lo = float(nz.min()) if vmin is None else float(vmin)
    hi = amax if vmax is None else float(vmax)
    if not hi > lo:
        hi = lo * 10.0 if lo > 0 else 1.0
    return masked, cmap, mcolors.LogNorm(vmin=lo, vmax=hi)


def _widen(lo: float, hi: float) -> tuple[float, float]:
    """Nudge a degenerate colour range apart so matplotlib has something to map."""
    if hi > lo:
        return lo, hi
    pad = abs(lo) * 1e-6 if lo != 0.0 else 0.5
    return lo - pad, hi + pad


# ==========================================================================
# Grid
# ==========================================================================
def plot_grid(grid: RectilinearGrid, ax=None, *, plane: str | int = "z",
              color: str = "0.5", lw: float = 0.5, max_lines: int = 400,
              aspect: str | float = "equal", title: str | None = None,
              **kw: Any):
    """Draw the mesh lines of one coordinate plane.

    Parameters
    ----------
    grid : RectilinearGrid
        Grid whose node lines are drawn.  Coordinates are metres.
    ax : matplotlib.axes.Axes or None
        Target axes; a new figure is created only when this is ``None``.
    plane : {'x', 'y', 'z'} or int
        Normal direction of the drawn plane.  Because the grid is a tensor
        product the line pattern is identical at every position along the
        normal, so no index is needed.
    color, lw
        Line colour and width passed to the ``LineCollection``.
    max_lines : int
        Cap on the number of drawn lines per direction.  A million-node mesh
        renders as solid black otherwise.  When the cap bites, every ``n``-th
        line is drawn (the last line always included) and the title says so, so
        the picture is never quietly wrong about mesh density.
    aspect : str or float
        Passed to ``ax.set_aspect``; ``"equal"`` shows true geometry.
    title : str or None
        Overrides the generated title.
    **kw
        Extra keyword arguments forwarded to ``LineCollection``.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.
    """
    from matplotlib.collections import LineCollection

    axis = _axis_of(plane)
    u_ax, w_ax = _in_plane_axes(axis)
    u = _node_axis(grid, u_ax)
    w = _node_axis(grid, w_ax)
    if max_lines is not None and max_lines < 2:
        raise ValueError("max_lines must be >= 2")

    su = 1 if max_lines is None else max(1, int(np.ceil(u.size / max_lines)))
    sw = 1 if max_lines is None else max(1, int(np.ceil(w.size / max_lines)))
    ui = np.unique(np.concatenate([u[::su], u[-1:]]))
    wi = np.unique(np.concatenate([w[::sw], w[-1:]]))

    segs = [[(x, w[0]), (x, w[-1])] for x in ui]
    segs += [[(u[0], y), (u[-1], y)] for y in wi]

    ax, _ = _get_ax(ax)
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, **kw))
    ax.set_xlim(u[0], u[-1])
    ax.set_ylim(w[0], w[-1])

    if title is None:
        nu, nw = u.size - 1, w.size - 1
        title = (f"{_AXIS_NAMES[u_ax]}-{_AXIS_NAMES[w_ax]} mesh, "
                 f"{nu} x {nw} cells")
        if su > 1 or sw > 1:
            title += f" (every {su}th / {sw}th line drawn)"
    _finish_axes(ax, axis, aspect, title)
    return ax


# ==========================================================================
# Materials
# ==========================================================================
def plot_materials(matmap: Any, plane: str | int = "z",
                   index: int | None = None, ax=None, *,
                   grid: RectilinearGrid | None = None,
                   legend: bool = True, edgecolor: str | None = None,
                   aspect: str | float = "equal", title: str | None = None,
                   **kw: Any):
    """Draw the material assignment of one cell plane as a filled map.

    Parameters
    ----------
    matmap : MaterialMap
        Anything exposing ``ids() -> (Nx, Ny, Nz)`` integer array and
        ``material_at(i, j, k) -> Material``.  Colours come from
        ``Material.color`` and labels from ``Material.name``; both are optional
        and fall back to a categorical palette / ``"id N"``.
    plane : {'x', 'y', 'z'} or int
        Normal direction of the slice.
    index : int or None
        **Cell** index along the normal (materials are per cell, unlike the
        nodal quantities plotted elsewhere).  ``None`` takes the middle cell.
        Negative indices count from the end.
    ax : matplotlib.axes.Axes or None
    grid : RectilinearGrid or None
        Grid supplying the node coordinates [m].  Taken from ``matmap.grid``
        when not given.
    legend : bool
        Add a patch legend naming the materials present in this slice.
    edgecolor : str or None
        Cell outline colour; ``None`` leaves cells un-outlined.
    aspect, title
        As in :func:`plot_grid`.
    **kw
        Forwarded to ``pcolormesh``.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    Cells are drawn with their true node coordinates as quadrilateral edges, so
    a graded mesh and its staircased interfaces (assumption A2) are visible
    exactly as the solver sees them.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    if grid is None:
        grid = getattr(matmap, "grid", None)
    if grid is None:
        raise ValueError(
            "cannot find the grid: pass grid=... explicitly, or use a "
            "MaterialMap that stores it as .grid")
    if not hasattr(matmap, "ids"):
        raise ValueError("matmap must expose ids() -> (Nx, Ny, Nz) int array")

    ids = _as_cell_field(grid, np.asarray(matmap.ids()))
    axis = _axis_of(plane)
    idx = _resolve_index(index, grid.shape_cells[axis], "cell")
    sl = np.take(ids, idx, axis=axis)

    uniq = np.unique(sl)
    colors: list[Any] = []
    labels: list[str] = []
    fallback = _categorical_colors(uniq.size)
    for n, uid in enumerate(uniq):
        mat = _material_for(matmap, sl, uniq_id=uid, axis=axis, index=idx)
        colors.append(getattr(mat, "color", None) or fallback[n])
        labels.append(str(getattr(mat, "name", f"id {int(uid)}")))

    # Remap sparse material ids onto 0..n-1 so a ListedColormap indexes them.
    compact = np.searchsorted(uniq, sl).astype(float)

    u_ax, w_ax = _in_plane_axes(axis)
    U, W = np.meshgrid(_node_axis(grid, u_ax), _node_axis(grid, w_ax),
                       indexing="ij")

    ax, _ = _get_ax(ax)
    kw.setdefault("shading", "flat")
    if edgecolor is not None:
        kw.setdefault("edgecolors", edgecolor)
        kw.setdefault("linewidth", 0.2)
    ax.pcolormesh(U, W, compact,
                  cmap=ListedColormap(colors),
                  norm=BoundaryNorm(np.arange(uniq.size + 1) - 0.5, uniq.size),
                  **kw)
    if legend:
        ax.legend(handles=[Patch(facecolor=c, edgecolor="none", label=l)
                           for c, l in zip(colors, labels)],
                  loc="best", fontsize="small", framealpha=0.85)
    _finish_axes(ax, axis, aspect,
                 title if title is not None
                 else "materials, " + _plane_title(grid, axis, idx, nodal=False))
    return ax


def _material_for(matmap: Any, sl: np.ndarray, uniq_id: Any,
                  axis: int, index: int) -> Any:
    """Look up a representative Material for one id present in a slice."""
    if not hasattr(matmap, "material_at"):
        return None
    hit = np.argwhere(sl == uniq_id)
    if hit.size == 0:
        return None
    ijk = list(hit[0])
    ijk.insert(axis, index)
    try:
        return matmap.material_at(int(ijk[0]), int(ijk[1]), int(ijk[2]))
    except Exception:  # a map without the full API still gets a usable plot
        return None


def _categorical_colors(n: int) -> list[Any]:
    """``n`` distinct fallback colours (tab20, cycled) for unnamed materials."""
    from matplotlib import colormaps
    base = colormaps["tab20"].colors
    return [base[i % len(base)] for i in range(max(n, 1))]


# ==========================================================================
# Scalar node fields
# ==========================================================================
def _pcolor_nodal(grid: RectilinearGrid, data2d: np.ndarray, axis: int, ax,
                  cmap: str | None, norm: Any, shading: str, **kw: Any):
    """Draw one nodal slice with true node coordinates.  Returns the QuadMesh.

    ``shading="gouraud"`` treats node values as the point samples they are and
    interpolates bilinearly between the true node positions.  ``"flat"``
    averages the four corner nodes of each cell and fills the cell.  Either way
    the quadrilateral geometry is the real (graded) mesh --- this is why
    ``imshow`` is never an option here.
    """
    u_ax, w_ax = _in_plane_axes(axis)
    U, W = np.meshgrid(_node_axis(grid, u_ax), _node_axis(grid, w_ax),
                       indexing="ij")
    if shading == "gouraud":
        c = data2d
    elif shading == "nearest":
        # Node coordinates as cell CENTRES: matplotlib puts the quad edges at
        # the midpoints, which is exactly the dual control volume of the box
        # method (assumption A10).
        c = data2d
    elif shading == "flat":
        c = 0.25 * (data2d[:-1, :-1] + data2d[1:, :-1]
                    + data2d[:-1, 1:] + data2d[1:, 1:])
    else:
        raise ValueError(
            f"shading must be 'gouraud', 'flat' or 'nearest', got {shading!r}")
    return ax.pcolormesh(U, W, c, cmap=cmap, norm=norm, shading=shading, **kw)


def plot_scalar(grid: RectilinearGrid, field: np.ndarray,
                plane: str | int = "z", index: int | None = None, ax=None,
                log: bool = False, *, cmap: str | None = None,
                vmin: float | None = None, vmax: float | None = None,
                symmetric: bool | None = None, shading: str = "gouraud",
                colorbar: bool = True, label: str | None = None,
                aspect: str | float = "equal", title: str | None = None,
                **kw: Any):
    """Colour map of a nodal scalar field on one slice plane.

    Parameters
    ----------
    grid : RectilinearGrid
        Grid the field lives on; supplies node coordinates [m].
    field : np.ndarray
        Nodal scalar, either shaped ``grid.shape_nodes`` or flat with
        ``grid.n_nodes`` entries.  Units are the solver's: V for potential,
        C for nodal charge, m^-3 for carrier density.
    plane : {'x', 'y', 'z'} or int
        Normal direction of the slice.
    index : int or None
        **Node** index along the normal; ``None`` takes the middle node.
        Negative indices count from the end.
    ax : matplotlib.axes.Axes or None
    log : bool
        Logarithmic colour scale.  All-positive data uses ``LogNorm`` with
        non-positive entries masked; data that straddles zero uses
        ``SymLogNorm`` so negative values keep their sign instead of vanishing.
    cmap : str or None
        ``None`` selects ``RdBu_r`` for signed data and ``viridis`` otherwise.
    vmin, vmax : float or None
        Colour limits in field units.  For signed data the default range is
        symmetric (``vmin = -vmax``) so that white is exactly zero.
    symmetric : bool or None
        Force the symmetric/one-sided choice instead of inferring it.
    shading : {'gouraud', 'flat', 'nearest'}
        ``'gouraud'`` (default) interpolates between node samples,
        ``'nearest'`` fills each node's dual box, ``'flat'`` averages nodes onto
        cells.
    colorbar : bool
        Attach a colorbar to the parent figure.
    label : str or None
        Colorbar label; state the unit here.
    aspect, title
        As in :func:`plot_grid`.
    **kw
        Forwarded to ``pcolormesh``.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    The slice is drawn with ``pcolormesh`` on the true node coordinates, never
    ``imshow``: fieldspice meshes are graded (assumption A2/A10) and an image
    would place every interface at the wrong position while looking perfectly
    plausible.
    """
    arr = _as_node_field(grid, field)
    axis = _axis_of(plane)
    idx = _resolve_index(index, grid.shape_nodes[axis], "node")
    sl = np.take(arr, idx, axis=axis)

    data, cmap_name, norm = _scalar_norm(sl, log, cmap, vmin, vmax, symmetric)
    ax, _ = _get_ax(ax)
    mesh = _pcolor_nodal(grid, data, axis, ax, cmap_name, norm, shading, **kw)
    if colorbar:
        cb = ax.get_figure().colorbar(mesh, ax=ax)
        if label:
            cb.set_label(label)
    _finish_axes(ax, axis, aspect,
                 title if title is not None
                 else _plane_title(grid, axis, idx, nodal=True))
    return ax


# ==========================================================================
# Vector edge fields
# ==========================================================================
def plot_vector(grid: RectilinearGrid, edge_vec: np.ndarray,
                plane: str | int = "z", index: int | None = None, ax=None,
                stride: int = 2, *, integrated: bool = True,
                normalize: bool = False, cmap: str = "viridis",
                colorbar: bool = True, label: str | None = None,
                scale: float | None = None, aspect: str | float = "equal",
                title: str | None = None, **kw: Any):
    """Quiver plot of an edge vector field, interpolated to nodes.

    Parameters
    ----------
    grid : RectilinearGrid
    edge_vec : np.ndarray
        Flat edge vector of length ``grid.n_edges`` in the concatenated
        ``[x, y, z]`` layout.  By default it is interpreted as **integrated**
        edge circulations (V for ``e``, Wb for ``a``, per the grid convention),
        so it is divided by the edge length to recover a field (V/m, Wb/m)
        before plotting.  Set ``integrated=False`` if the array already holds
        field values.
    plane : {'x', 'y', 'z'} or int
        Normal direction of the slice; the two in-plane components are drawn
        and the out-of-plane component is discarded (not projected).
    index : int or None
        **Node** index along the normal; ``None`` takes the middle node.
    ax : matplotlib.axes.Axes or None
    stride : int
        Draw every ``stride``-th node in each direction.  A quiver with one
        arrow per node is unreadable above roughly 50x50.
    integrated : bool
        See ``edge_vec``.
    normalize : bool
        Draw unit-length arrows coloured by magnitude.  Use this when the
        magnitude spans decades and true-length arrows would collapse to dots.
    cmap : str
        Colormap for the magnitude; ``viridis`` because a magnitude is
        non-negative.
    colorbar : bool
    label : str or None
        Colorbar label; state the unit (e.g. ``"|E| [V/m]"``).
    scale : float or None
        Passed to ``quiver``; ``None`` lets matplotlib autoscale.
    aspect, title
        As in :func:`plot_grid`.
    **kw
        Forwarded to ``quiver``.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    Interpolating edge circulations to nodes averages away the exactness of the
    incidence operators, which is why
    :func:`fieldspice.operators.interpolate_edges_to_nodes` is documented as
    post-processing only.  Never feed the interpolated field back into a solver.
    """
    vec = np.asarray(edge_vec, dtype=float).ravel()
    if vec.size != grid.n_edges:
        raise ValueError(
            f"edge_vec must have {grid.n_edges} entries, got {vec.size}")
    if int(stride) < 1:
        raise ValueError("stride must be >= 1")
    stride = int(stride)

    nodal = interpolate_edges_to_nodes(grid, vec, integrated=integrated)
    axis = _axis_of(plane)
    idx = _resolve_index(index, grid.shape_nodes[axis], "node")
    sl = np.take(nodal, idx, axis=axis)          # (nu, nw, 3)

    u_ax, w_ax = _in_plane_axes(axis)
    u = _node_axis(grid, u_ax)[::stride]
    w = _node_axis(grid, w_ax)[::stride]
    U, W = np.meshgrid(u, w, indexing="ij")
    fu = sl[::stride, ::stride, u_ax]
    fw = sl[::stride, ::stride, w_ax]
    mag = np.hypot(fu, fw)

    du, dw = fu, fw
    if normalize:
        safe = np.where(mag > 0, mag, 1.0)
        du, dw = fu / safe, fw / safe

    ax, _ = _get_ax(ax)
    kw.setdefault("pivot", "mid")
    q = ax.quiver(U, W, du, dw, mag, cmap=cmap, scale=scale, **kw)
    if colorbar:
        cb = ax.get_figure().colorbar(q, ax=ax)
        cb.set_label(label if label else "magnitude")
    _finish_axes(ax, axis, aspect,
                 title if title is not None
                 else _plane_title(grid, axis, idx, nodal=True))
    return ax


# ==========================================================================
# Terminal time series
# ==========================================================================
def _series_x(result: Any, n: int) -> tuple[np.ndarray, str]:
    """Time axis for a terminal trace, or a sample index when there is none."""
    t = np.asarray(getattr(result, "t", np.zeros(0))).ravel()
    if t.size == n and n > 0:
        return t, "t [s]"
    return np.arange(n, dtype=float), "sample"


def plot_terminals(result: Any, ax=None, *, which: str = "both",
                   names: Sequence[str] | None = None,
                   title: str | None = None, **kw: Any):
    """Plot terminal voltages and currents against time.

    Parameters
    ----------
    result : Result
        Any object with a ``terminals`` mapping ``{name: {"v": (nt,),
        "i": (nt,)}}`` and a time array ``t`` [s].  Voltages are volts,
        currents amperes.
    ax : matplotlib.axes.Axes or None
    which : {'both', 'v', 'i'}
        ``'both'`` puts currents on a twin y-axis (dashed) sharing the time
        axis, and returns the **voltage** axes.
    names : sequence of str or None
        Subset of terminals to draw; ``None`` draws all of them.
    title : str or None
    **kw
        Forwarded to ``plot``.

    Returns
    -------
    matplotlib.axes.Axes
        The primary axes (voltage when ``which`` includes voltage).
    """
    terms = getattr(result, "terminals", None)
    if not terms:
        raise ValueError("result has no terminal data to plot")
    if which not in ("both", "v", "i"):
        raise ValueError(f"which must be 'both', 'v' or 'i', got {which!r}")
    keys = list(terms) if names is None else list(names)
    missing = [k for k in keys if k not in terms]
    if missing:
        raise ValueError(f"unknown terminals {missing}; have {list(terms)}")

    ax, _ = _get_ax(ax, figsize=(6.5, 4.0))
    xlabel = "t [s]"
    ax_i = None
    want_v = which in ("both", "v")
    want_i = which in ("both", "i")

    for n, key in enumerate(keys):
        rec = terms[key]
        color = f"C{n}"
        if want_v and "v" in rec:
            v = np.asarray(rec["v"], dtype=float).ravel()
            x, xlabel = _series_x(result, v.size)
            ax.plot(x, v, color=color, label=f"{key} V", **kw)
        if want_i and "i" in rec:
            i = np.asarray(rec["i"], dtype=float).ravel()
            x, xlabel = _series_x(result, i.size)
            target = ax
            if which == "both":
                if ax_i is None:
                    ax_i = ax.twinx()
                    ax_i.set_ylabel("i [A]")
                target = ax_i
            target.plot(x, i, color=color, ls="--", label=f"{key} I", **kw)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("v [V]" if want_v else "i [A]")
    handles, labels = ax.get_legend_handles_labels()
    if ax_i is not None:
        h2, l2 = ax_i.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    if handles:
        ax.legend(handles, labels, loc="best", fontsize="small")
    if title:
        ax.set_title(title)
    return ax


# ==========================================================================
# I-V curves
# ==========================================================================
def plot_iv(result: Any, ax=None, log: bool = False, *,
            terminal: str | None = None, v_terminal: str | None = None,
            marker: str | None = None, label: str | None = None,
            title: str | None = None, **kw: Any):
    """Plot terminal current against terminal voltage.

    Parameters
    ----------
    result : Result
        Object with ``terminals`` mapping to ``{"v": [V], "i": [A]}``.
    ax : matplotlib.axes.Axes or None
    log : bool
        Logarithmic current axis showing ``|I|``.  This is the standard
        subthreshold / diode view; the absolute value is stated in the y label
        because a log axis cannot show a sign.  Zero-current points are dropped
        rather than clipped.
    terminal : str or None
        Terminal supplying the current (and, by default, the voltage).
        ``None`` picks the first terminal that has both a current and a
        non-constant voltage, falling back to the first terminal.
    v_terminal : str or None
        Terminal supplying the sweep voltage when it differs from ``terminal``
        (the usual case for a transfer curve: gate voltage against drain
        current).
    marker, label, title
    **kw
        Forwarded to ``plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    terms = getattr(result, "terminals", None)
    if not terms:
        raise ValueError("result has no terminal data to plot")
    if terminal is None:
        terminal = _pick_iv_terminal(terms)
    if terminal not in terms:
        raise ValueError(f"unknown terminal {terminal!r}; have {list(terms)}")
    src = terms[terminal]
    if "i" not in src:
        raise ValueError(f"terminal {terminal!r} carries no current record")
    i = np.asarray(src["i"], dtype=float).ravel()

    vkey = terminal if v_terminal is None else v_terminal
    if vkey not in terms or "v" not in terms[vkey]:
        raise ValueError(f"no voltage record on terminal {vkey!r}")
    v = np.asarray(terms[vkey]["v"], dtype=float).ravel()
    if v.size != i.size:
        raise ValueError(
            f"voltage ({v.size}) and current ({i.size}) records have "
            "different lengths")

    ax, _ = _get_ax(ax, figsize=(5.5, 4.0))
    lbl = label if label is not None else (
        f"I({terminal})" if vkey == terminal else f"I({terminal}) vs V({vkey})")
    if log:
        keep = i != 0.0
        if not np.any(keep):
            raise ValueError("log=True but every current sample is zero")
        ax.semilogy(v[keep], np.abs(i[keep]), marker=marker, label=lbl, **kw)
        ax.set_ylabel("|i| [A]")
    else:
        ax.plot(v, i, marker=marker, label=lbl, **kw)
        ax.set_ylabel("i [A]")
    ax.set_xlabel(f"v({vkey}) [V]")
    ax.legend(loc="best", fontsize="small")
    if title:
        ax.set_title(title)
    return ax


def _pick_iv_terminal(terms: dict) -> str:
    """First terminal with a current and a swept voltage; else the first one."""
    for name, rec in terms.items():
        if "i" not in rec or "v" not in rec:
            continue
        v = np.asarray(rec["v"], dtype=float).ravel()
        if v.size > 1 and np.ptp(v) > 0.0:
            return name
    return next(iter(terms))


# ==========================================================================
# Animation
# ==========================================================================
def animate(result: Any, field: str | np.ndarray, plane: str | int = "z",
            fps: int = 20, path: str | Path | None = None, *,
            index: int | None = None, log: bool = False,
            cmap: str | None = None, vmin: float | None = None,
            vmax: float | None = None, symmetric: bool | None = None,
            shading: str = "gouraud", every: int = 1, ax=None,
            colorbar: bool = True, label: str | None = None,
            aspect: str | float = "equal", dpi: int = 120,
            **kw: Any):
    """Animate a stored nodal field history over one slice plane.

    Parameters
    ----------
    result : Result
        Must carry ``grid`` and, when ``field`` is a name, ``fields[field]``
        with shape ``(nt,) + grid.shape_nodes`` or ``(nt, grid.n_nodes)``.
        ``result.t`` [s] labels the frames when its length matches.
    field : str or np.ndarray
        Name of a stored field, or the history array itself.
    plane : {'x', 'y', 'z'} or int
    fps : int
        Frames per second for playback and for the saved file.
    path : str, pathlib.Path or None
        When given, the animation is written here.  ``.mp4``/``.mov``/``.webm``
        use ffmpeg; ``.gif`` uses pillow.  If ffmpeg is missing the animation
        falls back to an animated GIF beside the requested path (with a
        warning); if neither writer exists a ``RuntimeError`` names both fixes.
    index : int or None
        Node index along the normal; ``None`` takes the middle node.
    log, cmap, vmin, vmax, symmetric, shading, colorbar, label, aspect
        As in :func:`plot_scalar`.  Colour limits are computed **once** over
        every displayed frame, because a per-frame rescale turns a decaying
        transient into a constant-looking movie.
    every : int
        Keep every ``every``-th time sample.
    ax : matplotlib.axes.Axes or None
    dpi : int
        Resolution used when saving.
    **kw
        Forwarded to ``pcolormesh``.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Keep a reference to the returned object: matplotlib animations stop
        when garbage collected.
    """
    from matplotlib.animation import FuncAnimation

    grid = getattr(result, "grid", None)
    if grid is None:
        raise ValueError("result has no .grid; cannot place the field")
    if isinstance(field, str):
        fields = getattr(result, "fields", {}) or {}
        if field not in fields:
            raise ValueError(
                f"result has no stored field {field!r}; have {list(fields)}")
        hist = np.asarray(fields[field], dtype=float)
        fname = field
    else:
        hist = np.asarray(field, dtype=float)
        fname = "field"
    if hist.ndim < 2:
        raise ValueError("field history must have a leading time axis")
    if int(every) < 1:
        raise ValueError("every must be >= 1")

    nt = hist.shape[0]
    frames = np.arange(0, nt, int(every))
    if frames.size == 0:
        raise ValueError("no frames to animate")
    stack = np.stack([_as_node_field(grid, hist[k]) for k in frames])

    axis = _axis_of(plane)
    idx = _resolve_index(index, grid.shape_nodes[axis], "node")
    slices = np.take(stack, idx, axis=axis + 1)      # (nf, nu, nw)

    # One global norm for the whole movie, so brightness means the same thing
    # in every frame.
    all_data, cmap_name, norm = _scalar_norm(slices, log, cmap, vmin, vmax,
                                             symmetric)
    masked = np.ma.getmaskarray(all_data) if np.ma.isMaskedArray(all_data) \
        else None

    t = np.asarray(getattr(result, "t", np.zeros(0))).ravel()
    times = t[frames] if t.size == nt else frames.astype(float)
    tlabel = "t = {:.4g} s" if t.size == nt else "frame {:.0f}"

    ax, created = _get_ax(ax)
    first = all_data[0]
    mesh = _pcolor_nodal(grid, first, axis, ax, cmap_name, norm, shading, **kw)
    if colorbar:
        cb = ax.get_figure().colorbar(mesh, ax=ax)
        cb.set_label(label if label else fname)
    _finish_axes(ax, axis, aspect,
                 f"{fname}, {_plane_title(grid, axis, idx, nodal=True)}")
    text = ax.text(0.02, 0.98, tlabel.format(times[0]), transform=ax.transAxes,
                   va="top", ha="left", fontsize="small")

    def _frame_data(k: int) -> np.ndarray:
        d = slices[k]
        if masked is not None:
            d = np.ma.array(d, mask=masked[k])
        if shading == "flat":
            d = 0.25 * (d[:-1, :-1] + d[1:, :-1] + d[:-1, 1:] + d[1:, 1:])
        return d

    def _update(k: int):
        mesh.set_array(_frame_data(k))
        text.set_text(tlabel.format(times[k]))
        return mesh, text

    anim = FuncAnimation(ax.get_figure(), _update, frames=slices.shape[0],
                         interval=1000.0 / max(float(fps), 1e-6), blit=False)
    if path is not None:
        _save_animation(anim, Path(path), fps=fps, dpi=dpi)
    return anim


def _save_animation(anim: Any, path: Path, fps: int, dpi: int) -> Path:
    """Write an animation, preferring ffmpeg and degrading to an animated GIF."""
    from matplotlib import animation as manim

    want_gif = path.suffix.lower() == ".gif"
    order = ("pillow", "ffmpeg") if want_gif else ("ffmpeg", "pillow")
    chosen = next((w for w in order if manim.writers.is_available(w)), None)
    if chosen is None:
        raise RuntimeError(
            f"cannot save {path}: no animation writer is available. Install "
            "ffmpeg (brew install ffmpeg / apt-get install ffmpeg) for video, "
            "or pillow (pip install pillow) for an animated GIF.")
    out = path
    if chosen == "pillow" and not want_gif:
        out = path.with_suffix(".gif")
        warnings.warn(
            f"ffmpeg is not available, so {path.name} could not be written; "
            f"saved an animated GIF to {out} instead", RuntimeWarning,
            stacklevel=3)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = (manim.PillowWriter(fps=fps) if chosen == "pillow"
              else manim.FFMpegWriter(fps=fps))
    anim.save(str(out), writer=writer, dpi=dpi)
    return out


# ==========================================================================
# Convergence history
# ==========================================================================
_CONVERGENCE_KEYS = ("residuals", "residual", "residual_history",
                     "newton_residuals", "newton_history", "convergence",
                     "history")


def plot_convergence(result: Any, ax=None, *, key: str | None = None,
                     marker: str = "o", title: str | None = None, **kw: Any):
    """Semilog plot of a solver's residual history.

    Parameters
    ----------
    result : Result
        Searched for a residual trace in ``meta`` then ``scalars``.  A flat
        sequence is treated as one iteration history; a sequence of sequences
        is treated as one history per time step and each is drawn separately.
        Residuals are dimensionless (relative) or in the solver's own units;
        the label reports the key that was found.
    ax : matplotlib.axes.Axes or None
    key : str or None
        Force a specific ``meta``/``scalars`` key instead of searching.
    marker : str
    title : str or None
    **kw
        Forwarded to ``semilogy``.

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    A Newton failure is always diagnosed from this curve: flat means a bad
    Jacobian, sawtoothed means the line search is fighting the step, and a
    clean quadratic drop means the model is fine and the tolerance is too
    tight.
    """
    name, histories = _find_convergence(result, key)
    ax, _ = _get_ax(ax, figsize=(5.5, 4.0))
    multi = len(histories) > 1
    for n, h in enumerate(histories):
        y = np.abs(np.asarray(h, dtype=float).ravel())
        if y.size == 0:
            continue
        y = np.where(y > 0, y, np.nan)  # a zero residual is exact, not -inf
        ax.semilogy(np.arange(1, y.size + 1), y, marker=marker,
                    alpha=0.35 if multi else 1.0,
                    label=None if multi else name, **kw)
    ax.set_xlabel("iteration")
    ax.set_ylabel(f"|{name}|")
    ax.grid(True, which="both", alpha=0.3)
    if not multi:
        ax.legend(loc="best", fontsize="small")
    ax.set_title(title if title is not None
                 else (f"{name}: {len(histories)} solves" if multi else name))
    return ax


def _find_convergence(result: Any, key: str | None
                      ) -> tuple[str, list[np.ndarray]]:
    """Locate a residual trace; raise with the available keys if there is none."""
    meta = getattr(result, "meta", {}) or {}
    scalars = getattr(result, "scalars", {}) or {}

    def _pull(k: str):
        if k in meta:
            return meta[k]
        if k in scalars:
            return scalars[k]
        return None

    if key is not None:
        raw = _pull(key)
        if raw is None:
            raise ValueError(
                f"no key {key!r} in result.meta ({list(meta)}) or "
                f"result.scalars ({list(scalars)})")
        found = key
    else:
        found, raw = None, None
        for cand in _CONVERGENCE_KEYS:
            raw = _pull(cand)
            if raw is not None:
                found = cand
                break
        if found is None:
            for cand in list(scalars) + list(meta):
                if any(s in cand.lower() for s in ("resid", "error", "converg")):
                    found, raw = cand, _pull(cand)
                    break
        if found is None:
            raise ValueError(
                "no convergence history found; looked for "
                f"{list(_CONVERGENCE_KEYS)} in result.meta ({list(meta)}) and "
                f"result.scalars ({list(scalars)}). Pass key=... to select one.")

    if isinstance(raw, dict):
        for cand in _CONVERGENCE_KEYS:
            if cand in raw:
                found, raw = f"{found}.{cand}", raw[cand]
                break
        else:
            raise ValueError(f"{found!r} is a dict with no residual entry")

    histories = _as_histories(raw)
    if not histories:
        raise ValueError(f"convergence entry {found!r} is empty")
    return found, histories


def _as_histories(raw: Any) -> list[np.ndarray]:
    """Normalise a scalar / 1D / ragged-2D residual record to a list of 1D arrays."""
    if np.isscalar(raw):
        return [np.array([float(raw)])]
    arr = np.asarray(raw, dtype=object) if _is_ragged(raw) else \
        np.asarray(raw, dtype=float)
    if arr.dtype == object:
        return [np.asarray(h, dtype=float).ravel() for h in raw
                if np.size(h) > 0]
    if arr.ndim <= 1:
        return [np.atleast_1d(arr)]
    return [np.asarray(row, dtype=float).ravel() for row in arr]


def _is_ragged(raw: Any) -> bool:
    if not isinstance(raw, (list, tuple)):
        return False
    sizes = {np.size(x) for x in raw}
    return len(sizes) > 1 and all(np.ndim(x) >= 1 for x in raw)


# ==========================================================================
# Eye diagram
# ==========================================================================
def eye_diagram(t: np.ndarray, v: np.ndarray, period: float, ax=None, *,
                n_phase: int = 256, t_start: float | None = None,
                offset: float = 0.0, max_traces: int | None = None,
                color: str = "C0", alpha: float = 0.25, lw: float = 0.8,
                title: str | None = None, **kw: Any):
    """Fold a transient waveform on a period to form an eye diagram.

    Parameters
    ----------
    t : np.ndarray
        Sample times [s], strictly increasing.  Non-uniform spacing is fine and
        is the normal case for an adaptive transient.
    v : np.ndarray
        Waveform samples [V] (or any unit), same length as ``t``.
    period : float
        Unit interval [s] to fold on.  One UI for NRZ, two bit periods if you
        want to see both transitions.
    ax : matplotlib.axes.Axes or None
    n_phase : int
        Samples per folded trace.  Because the record almost never contains an
        integer number of samples per period, each trace is resampled by linear
        interpolation onto a common phase grid rather than by index slicing --
        index slicing would smear the eye by up to one sample period per UI and
        close it artificially.
    t_start : float or None
        Time of the first fold boundary [s]; defaults to ``t[0]``.  Use it to
        discard a startup transient.
    offset : float
        Extra phase shift [s] applied when sampling, for centring the eye.
    max_traces : int or None
        Cap on the number of drawn traces (the most recent ones are kept), for
        very long records.
    color, alpha, lw
        Trace styling.  Low alpha is what makes the density structure of the
        eye visible.
    title : str or None
    **kw
        Forwarded to ``LineCollection``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    from matplotlib.collections import LineCollection

    t = np.asarray(t, dtype=float).ravel()
    v = np.asarray(v, dtype=float).ravel()
    if t.size != v.size:
        raise ValueError(f"t and v must be the same length, got {t.size} and {v.size}")
    if t.size < 2:
        raise ValueError("need at least 2 samples")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing")
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be a positive, finite time in seconds")
    if int(n_phase) < 2:
        raise ValueError("n_phase must be >= 2")

    t0 = float(t[0]) if t_start is None else float(t_start)
    if not t[0] - 1e-15 <= t0 <= t[-1]:
        raise ValueError("t_start lies outside the sampled record")
    span = t[-1] - t0 - offset
    n_per = int(np.floor(span / period))
    if n_per < 1:
        raise ValueError(
            f"record spans {span:.4g} s after t_start, less than one period "
            f"({period:.4g} s); nothing to fold")
    if max_traces is not None and max_traces < 1:
        raise ValueError("max_traces must be >= 1")

    phase = np.linspace(0.0, period, int(n_phase))
    starts = t0 + offset + period * np.arange(n_per)
    if max_traces is not None and n_per > max_traces:
        starts = starts[-int(max_traces):]

    # Resample every window onto the shared phase grid; np.interp handles the
    # non-integer samples-per-period case exactly.
    sample_t = starts[:, None] + phase[None, :]
    folded = np.interp(sample_t.ravel(), t, v).reshape(sample_t.shape)

    ax, _ = _get_ax(ax, figsize=(6.0, 4.0))
    segs = [np.column_stack([phase, row]) for row in folded]
    ax.add_collection(LineCollection(segs, colors=color, alpha=alpha,
                                     linewidths=lw, **kw))
    ax.set_xlim(phase[0], phase[-1])
    lo, hi = float(np.min(folded)), float(np.max(folded))
    pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.05
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("time within period [s]")
    ax.set_ylabel("v [V]")
    ax.set_title(title if title is not None
                 else f"eye, {len(segs)} traces folded on {period:.4g} s")
    return ax
