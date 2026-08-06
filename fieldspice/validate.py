"""Pre-flight physics checks: the assumptions in ``docs/ASSUMPTIONS.md``, executable.

A quasi-static field solver is fast because it discards physics.  Every one of
those discards has a validity condition, and the dangerous ones are dangerous
precisely because violating them produces a *plausible* answer rather than a
crash or a divergence.  A1 says it plainly: a modelling error does not shrink
under mesh refinement, so the usual "refine until it stops changing" reflex
cannot detect it.  This module is the detector.

Each function returns a :class:`Report` carrying

* a level --- ``"ok"``, ``"warn"`` or ``"error"``,
* the assumption tag it enforces (``A1``, ``A4a``, ...),
* the *numbers*: the computed ratio, the threshold it was compared against,
  and where in the grid the worst offender sits,
* a message that says what to change.

Nothing here solves anything, so all of it is cheap enough to run
unconditionally before a solve.  Typical use::

    rep = validate.check_all(grid, eps, sigma, mu, t_rise=20 * ps,
                             dt=1 * ps, conductor_mask=metal)
    print(rep)
    rep.raise_if_error()

Units are strict SI throughout, matching :mod:`fieldspice.units`: permittivity
in F/m (**absolute**, not relative), permeability in H/m (absolute),
conductivity in S/m, length in m, time in s, frequency in Hz, doping in m^-3,
temperature in K.  Passing relative permittivity or permeability is the most
common unit mistake in a field solver, so it is detected and rejected rather
than quietly producing an answer that is wrong by eleven orders of magnitude.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .grid import RectilinearGrid
from .units import T_ROOM, c0, eps0, kB, mu0, q

__all__ = [
    "LEVELS",
    "Report",
    "check_quasistatic",
    "check_skin_depth",
    "check_mesh_quality",
    "check_padding",
    "check_dielectric_relaxation",
    "check_debye_length",
    "check_all",
]


LEVELS: tuple[str, str, str] = ("ok", "warn", "error")
"""Severity levels in increasing order."""

_RANK: dict[str, int] = {lvl: i for i, lvl in enumerate(LEVELS)}

# Band edges of the A1 validity table, kept as module constants so a test can
# reference the same numbers the check uses.
QS_EXCELLENT = 0.01
QS_GOOD = 0.1
QS_MARGINAL = 0.3

# A4a: cells per skin depth.  Below 3 the exponential current profile is
# resolved by fewer than three samples and the AC resistance is wrong; below 1
# the profile is not represented at all.
SKIN_CELLS_PER_DELTA = 3.0

# A10: neighbouring-cell width ratio.
GROWTH_WARN = 1.5
GROWTH_ERROR = 2.0

# Cell aspect ratio above which the CG condition number starts to hurt.  There
# is no accuracy threshold here --- a tensor-product grid has no element-quality
# failure mode (A2) --- so this is a solver-cost warning only.
ASPECT_WARN = 1.0e3

# A12: domain padding, in multiples of the largest conductor dimension.
PAD_WARN = 3.0
PAD_ERROR = 1.0


# ==========================================================================
# Report
# ==========================================================================
@dataclass
class Report:
    """Outcome of one check (or of a combined set of checks).

    Attributes
    ----------
    check
        Name of the check that produced this report.
    level
        One of ``"ok"``, ``"warn"``, ``"error"``.  ``"warn"`` means the result
        will be quantitatively degraded; ``"error"`` means the model is the
        wrong model and refining the mesh will not help.
    message
        Human-readable, actionable summary including the offending numbers.
    details
        Machine-readable numbers behind the verdict.  Keys are named with their
        units where a unit exists (``"delta_min_m"``, ``"freq_Hz"``).
    assumption
        Tag(s) from ``docs/ASSUMPTIONS.md`` that this check enforces.
    children
        Sub-reports, populated by :meth:`combine`.

    Notes
    -----
    ``ok`` is ``True`` only for level ``"ok"``; a warning is not ok.  Use
    ``report.level != "error"`` if you only want to gate on fatal problems.
    """

    check: str
    level: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    assumption: str = ""
    children: tuple["Report", ...] = ()

    def __post_init__(self) -> None:
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {LEVELS}, got {self.level!r}")

    # -- status ------------------------------------------------------------
    @property
    def ok(self) -> bool:
        """``True`` only when the level is exactly ``"ok"``."""
        return self.level == "ok"

    def __bool__(self) -> bool:
        # Truthiness follows .ok so that `if not rep:` is not a silent no-op.
        return self.ok

    @property
    def rank(self) -> int:
        """Integer severity, 0 = ok, 2 = error.  Used to combine reports."""
        return _RANK[self.level]

    def raise_if_error(self) -> "Report":
        """Raise :class:`ValueError` if the level is ``"error"``; else return self."""
        if self.level == "error":
            raise ValueError(str(self))
        return self

    def walk(self) -> list["Report"]:
        """This report followed by every descendant, depth first."""
        out = [self]
        for c in self.children:
            out.extend(c.walk())
        return out

    # -- combination -------------------------------------------------------
    @classmethod
    def combine(cls, reports: Sequence["Report"], check: str = "check_all",
                message: str | None = None,
                details: dict[str, Any] | None = None) -> "Report":
        """Merge several reports into one whose level is the worst level present.

        Parameters
        ----------
        reports
            Sub-reports, kept in order as :attr:`children`.
        check
            Name for the combined report.
        message
            Override the auto-generated one-line tally.
        details
            Extra keys merged into the combined ``details`` alongside one entry
            per child (keyed by the child's ``check`` name).

        Returns
        -------
        Report
        """
        reports = tuple(reports)
        for r in reports:
            if not isinstance(r, Report):
                raise ValueError(f"combine expects Report objects, got {type(r)!r}")
        level = LEVELS[max((r.rank for r in reports), default=0)]
        counts = {lvl: sum(1 for r in reports if r.level == lvl) for lvl in LEVELS}
        if message is None:
            bad = [r.check for r in reports if r.level != "ok"]
            message = (f"{len(reports)} check(s): {counts['ok']} ok, "
                       f"{counts['warn']} warn, {counts['error']} error.")
            if bad:
                message += " Flagged: " + ", ".join(bad) + "."
            else:
                message += " No assumption violations detected."
        tags: list[str] = []
        for r in reports:
            for t in r.assumption.split(","):
                t = t.strip()
                if t and t not in tags:
                    tags.append(t)
        merged: dict[str, Any] = {"levels": counts}
        merged.update(details or {})
        for r in reports:
            merged[r.check] = r.details
        return cls(check=check, level=level, message=message, details=merged,
                   assumption=", ".join(tags), children=reports)

    # -- display -----------------------------------------------------------
    def __str__(self) -> str:
        tag = f" [{self.assumption}]" if self.assumption else ""
        head = f"[{self.level.upper():5s}] {self.check}{tag}"
        lines = [head]
        for para in self.message.split("\n"):
            lines.extend("  " + ln for ln in _wrap(para, 76))
        for key, val in self.details.items():
            if key in {r.check for r in self.children} or key == "levels":
                continue
            lines.append(f"    {key:<26s} = {_fmt(val)}")
        for child in self.children:
            lines.append("")
            lines.extend("  " + ln for ln in str(child).split("\n"))
        return "\n".join(lines)


def _fmt(val: Any) -> str:
    """Compact display of a details value."""
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (int, np.integer)):
        return str(int(val))
    if isinstance(val, (float, np.floating)):
        v = float(val)
        if not np.isfinite(v):
            return str(v)
        return f"{v:.4g}"
    if isinstance(val, tuple):
        return "(" + ", ".join(_fmt(v) for v in val) + ")"
    if isinstance(val, list):
        return "[" + ", ".join(_fmt(v) for v in val) + "]"
    if isinstance(val, dict):
        return "{" + ", ".join(f"{k}: {_fmt(v)}" for k, v in val.items()) + "}"
    return str(val)


def _wrap(text: str, width: int) -> list[str]:
    """Minimal greedy word wrap (avoids a textwrap import for one call site)."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    lines.append(cur)
    return lines or [""]


# ==========================================================================
# Input validation helpers
# ==========================================================================
def _as_cell_array(arr: np.ndarray, grid: RectilinearGrid, name: str) -> np.ndarray:
    """Coerce to a float ``(Nx, Ny, Nz)`` array, raising on any mismatch."""
    a = np.asarray(arr, dtype=float)
    if a.shape != grid.shape_cells:
        raise ValueError(
            f"{name} must have shape {grid.shape_cells} (Nx, Ny, Nz), got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values")
    return a


def _require_absolute_eps(a: np.ndarray, name: str) -> np.ndarray:
    """Reject a relative permittivity passed where an absolute one is required.

    Any real material has ``eps < 1e-4 F/m`` (that is ``eps_r < 1.1e7``), while
    a relative permittivity is by definition ``>= 1``.  The two ranges are nine
    decades apart, so the confusion is detectable with certainty.
    """
    if np.any(a <= 0.0):
        raise ValueError(f"{name} must be strictly positive [F/m]")
    if float(a.max()) > 1e-4:
        raise ValueError(
            f"{name} has a maximum of {float(a.max()):.4g}, which is not a "
            f"permittivity in F/m. fieldspice is strict SI: pass the ABSOLUTE "
            f"permittivity, i.e. eps_r * units.eps0 (eps0 = {eps0:.4g} F/m).")
    return a


def _require_absolute_mu(a: np.ndarray, name: str) -> np.ndarray:
    """Reject a relative permeability passed where an absolute one is required.

    Unlike permittivity the two ranges overlap, so the test needs two separate
    arguments.  ``mu_r >= 1`` by definition, while an absolute permeability runs
    upwards from ``mu0 = 1.257e-6 H/m``; a lone value of 5 could therefore mean
    ``mu_r = 5`` or ``mu = 5 H/m``.  What breaks the tie:

    * ``mu > 10 H/m`` is ``mu_r > 8e6``, above any real material, so it is a
      unit error whatever else is in the array;
    * ``mu >= 1 H/m`` in **every** cell means ``mu_r >= 8e5`` everywhere, i.e. a
      domain made entirely of mu-metal with no air, no dielectric and no
      conductor anywhere in it.  No meshed problem looks like that, whereas an
      array of relative permeabilities in ``[1, 10]`` (mu_r = 1 for everything
      non-magnetic, 2 for a weakly magnetic layer) is an everyday mistake.

    Without the second test the interval ``1 < mu_r <= 10`` passed through
    silently and every wavelength and skin depth downstream came out wrong by
    ``1 / sqrt(mu0) = 892``.  The residual blind spot is a domain that is
    uniformly supermalloy; pass ``mu_r * units.mu0`` as documented and it is
    unreachable.
    """
    if np.any(a <= 0.0):
        raise ValueError(f"{name} must be strictly positive [H/m]")
    lo, hi = float(a.min()), float(a.max())
    if hi > 10.0 or lo >= 1.0 or np.allclose(a, 1.0):
        raise ValueError(
            f"{name} spans {lo:.4g} to {hi:.4g}, which is a RELATIVE "
            f"permeability, not a permeability in H/m (that would be "
            f"mu_r from {lo / mu0:.4g} to {hi / mu0:.4g}). fieldspice is strict "
            f"SI: pass the ABSOLUTE permeability, i.e. mu_r * units.mu0 "
            f"(mu0 = {mu0:.4g} H/m).")
    return a


def _require_positive(value: float, name: str, unit: str) -> float:
    """Coerce to a strictly positive finite float, raising ``ValueError`` if not."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive {unit}, got {value!r}") from exc
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be a positive {unit}, got {value}")
    return v


def _resolved(grid: RectilinearGrid) -> list[int]:
    """Indices of the axes that are actually meshed (a collapsed axis is not).

    A collapsed direction carries a nominal 1 m thickness so that results are
    per-unit-length; including it in any size comparison would swamp every
    real length in the problem.
    """
    axes = [d for d, a in enumerate((grid.ax, grid.ay, grid.az)) if not a.collapsed]
    return axes or [0]


def _widths(grid: RectilinearGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (grid.hx, grid.hy, grid.hz)


def _widths_by_axis(grid: RectilinearGrid, kind: str = "cell"
                    ) -> dict[str, np.ndarray]:
    """Cell width along each *resolved* axis, broadcast over cells or nodes [m].

    Returns ``{"x": arr, ...}`` with each array shaped ``grid.shape_cells``
    (``kind="cell"``) or ``grid.shape_nodes`` (``kind="node"``); a node takes the
    larger of the two cells touching it, since the profile has to be resolved on
    both sides of it.

    Keeping the axes separate rather than collapsing them immediately to a
    single "cell size" is what lets the skin-depth and Debye checks distinguish
    a mesh that is coarse in a direction where nothing happens (along a
    transmission line) from one that is coarse across a surface, which is the
    only direction those two lengths are measured in.  Collapsed directions are
    omitted: their nominal 1 m thickness is a normalisation, not a size.
    """
    if kind not in ("cell", "node"):
        raise ValueError(f"kind must be 'cell' or 'node', got {kind!r}")
    shape = grid.shape_cells if kind == "cell" else grid.shape_nodes
    out: dict[str, np.ndarray] = {}
    for d in _resolved(grid):
        h = _widths(grid)[d]
        if kind == "node":
            h = np.maximum(np.concatenate([h[:1], h]), np.concatenate([h, h[-1:]]))
        shp = [1, 1, 1]
        shp[d] = -1
        # A broadcast view, not a copy: these arrays are only ever read from
        # (compared, masked, reduced), and on a million-cell grid the three
        # copies would be tens of megabytes for nothing.
        out["xyz"[d]] = np.broadcast_to(h.reshape(shp), shape)
    return out


def _surface_normal_widths(grid: RectilinearGrid, cond: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray,
                                      dict[str, np.ndarray], bool]:
    """Cell widths in the directions in which a conductor actually has a surface.

    The skin effect is an exponential decay measured *normal to a conductor
    surface*, so the mesh only has to resolve ``delta`` along axes in which the
    conductor is bounded.  A line of cells that conducts from one domain wall to
    the other has no surface with that normal *inside* the domain --- with the
    default homogeneous Neumann wall the conductor continues through it --- so
    the cell size along that line cannot affect the skin profile.  A line that
    is not conducting end to end contains only bounded runs of conductor, so
    every conducting cell on it does face a surface with that normal.  Those two
    statements are exhaustive, which makes the classification exact rather than
    heuristic.

    This replaces an earlier rule that took the largest cell dimension outright.
    That rule called a transmission line meshed coarsely along its length and
    finely across it a hard error, which is a false statement about a mesh
    already far finer than required in every direction the current profile
    varies in.

    The exception is a conductor that fills the domain in *every* resolved
    direction, which is genuinely ambiguous: it is either an infinite slab (the
    walls are symmetry planes, no skin profile exists) or a wire meshed right up
    to its own surface (the walls *are* the surface and the profile must be
    resolved).  Nothing in the mask distinguishes the two, so the function falls
    back to the conservative largest-dimension rule and reports that it did, so
    the caller can refuse to call the result an error.

    Parameters
    ----------
    grid
        The grid.
    cond
        Boolean ``(Nx, Ny, Nz)`` mask of conducting cells.

    Returns
    -------
    width
        Largest cell dimension among the surface-normal axes, per cell [m];
        zero in cells that are not evaluated.
    evaluated
        ``True`` in conducting cells the criterion applies to.
    per_axis
        ``{axis_name: width}`` with ``inf`` where that axis is not a surface
        normal, so ``delta / width`` is directly a cells-per-delta count.
    ambiguous
        ``True`` when the fallback above was taken.
    """
    shape = grid.shape_cells
    by_axis = _widths_by_axis(grid, "cell")
    width = np.zeros(shape)
    evaluated = np.zeros(shape, dtype=bool)
    per_axis: dict[str, np.ndarray] = {}
    for name, hd in by_axis.items():
        d = "xyz".index(name)
        through = np.all(cond, axis=d, keepdims=True)
        rel = cond & ~through
        per_axis[name] = np.where(rel, hd, np.inf)
        width = np.where(rel, np.maximum(width, hd), width)
        evaluated |= rel
    if evaluated.any() or not cond.any():
        return width, evaluated, per_axis, False
    for name, hd in by_axis.items():
        per_axis[name] = np.where(cond, hd, np.inf)
        width = np.where(cond, np.maximum(width, hd), width)
    return width, cond.copy(), per_axis, True


def _max_cell_aspect_ratio(grid: RectilinearGrid) -> tuple[float, str, str]:
    """Largest ratio between two dimensions of a *single* cell, and the two axes.

    On a tensor-product grid the axes are independent, so the worst cell is the
    one that takes the widest cell of one axis and the narrowest of another; the
    maximum is therefore ``max(h[d]) / min(h[e])`` over ordered pairs of
    distinct resolved axes.  ``grid.max_aspect_ratio()`` reports
    ``max(h) / min(h)`` over *all* resolved axes at once, which also admits
    ``d == e`` and so measures the grading range of the mesh rather than the
    shape of any cell in it; on a graded axis it can exceed every real cell
    aspect ratio.  The frozen grid method is reported separately.
    """
    dims = _resolved(grid)
    if len(dims) < 2:
        return 1.0, "", ""
    h = _widths(grid)
    best, wide, thin = 1.0, "", ""
    for d in dims:
        for e in dims:
            if d == e:
                continue
            r = float(h[d].max()) / float(h[e].min())
            if r > best:
                best, wide, thin = r, "xyz"[d], "xyz"[e]
    return best, wide, thin


def _component_features(mask: np.ndarray, grid: RectilinearGrid,
                        dims: Sequence[int]) -> tuple[list[float], int]:
    """Largest bounded dimension of each connected conductor [m], and their count.

    Connectivity is face-wise (the 6-neighbour structure), which is the same
    notion of "one conductor" the solvers use: two cells that share only an edge
    or a corner are not electrically joined by any edge of the Yee grid.

    A component that touches both walls along an axis is infinite in that
    direction, so that extent is excluded from *its* feature size --- but only
    from its own, which is the whole point of doing this per component.  A
    component that spans every resolved axis has no bounded dimension at all and
    falls back to its largest extent.

    ``scipy.ndimage`` does the labelling; SciPy is a core dependency, but if the
    import fails the function degrades to reporting nothing, which restores the
    union-only behaviour rather than crashing a pre-flight check.

    The caller skips this entirely when no axis is spanned, because then every
    component extent is bounded above by the union extent along the same axis
    and the answer cannot change --- worth doing, since labelling a
    million-cell mask costs about as much as all the other checks together.
    """
    try:
        from scipy import ndimage
    except ImportError:  # pragma: no cover - SciPy is a hard dependency
        return [], 1
    labels, n_comp = ndimage.label(mask)
    if n_comp <= 1:
        # One conductor: the per-component answer is the union answer already.
        return [], int(n_comp)
    nodes_all = (grid.xn, grid.yn, grid.zn)
    out: list[float] = []
    for box in ndimage.find_objects(labels):
        if box is None:  # pragma: no cover - find_objects only skips empty labels
            continue
        extents, bounded = [], []
        for d in dims:
            nd = nodes_all[d]
            lo, hi = box[d].start, box[d].stop
            size = float(nd[hi] - nd[lo])
            extents.append(size)
            if lo > 0 or hi < mask.shape[d]:
                bounded.append(size)
        out.append(max(bounded) if bounded else max(extents))
    return out, int(n_comp)


def _locate(flat: int, shape: tuple[int, ...], grid: RectilinearGrid | None,
            kind: str = "cell") -> dict[str, Any]:
    """Index (and physical position, when a grid is available) of a flat index."""
    idx = tuple(int(v) for v in np.unravel_index(int(flat), shape))
    out: dict[str, Any] = {"worst_index": idx}
    if grid is not None:
        if kind == "cell":
            coords = (grid.xc[idx[0]], grid.yc[idx[1]], grid.zc[idx[2]])
        else:
            coords = (grid.xn[idx[0]], grid.yn[idx[1]], grid.zn[idx[2]])
        out["worst_position_m"] = tuple(float(c) for c in coords)
    return out


def _loc_str(det: dict[str, Any]) -> str:
    idx = det.get("worst_index")
    pos = det.get("worst_position_m")
    if pos is None:
        return f"cell {idx}"
    return f"cell {idx} at ({pos[0]:.4g}, {pos[1]:.4g}, {pos[2]:.4g}) m"


# ==========================================================================
# A1 --- quasi-static validity
# ==========================================================================
def check_quasistatic(grid: RectilinearGrid,
                      eps_cell: np.ndarray,
                      mu_cell: np.ndarray | None = None,
                      t_rise: float | None = None,
                      freq: float | None = None) -> Report:
    """Is the structure electrically small enough for eqs / mqs / darwin? (A1)

    Computes ``L / lambda`` with ``L`` the largest *resolved* domain dimension
    and ``lambda = c0 / (f * sqrt(eps_r * mu_r))`` evaluated in the slowest
    material present (the cell with the largest ``eps_r * mu_r``), then applies
    the A1 band table.

    This is the most important check in the module.  Violating A1 is a
    *modelling* error, not a discretisation error: it does not shrink when the
    mesh is refined, so no convergence study can reveal it.

    Parameters
    ----------
    grid
        The grid the problem lives on.  Collapsed (symmetry) directions are
        excluded from ``L``, since their nominal 1 m thickness is a
        normalisation, not a physical size.
    eps_cell
        Absolute permittivity per cell [F/m], shape ``(Nx, Ny, Nz)``.
    mu_cell
        Absolute permeability per cell [H/m], same shape.  ``None`` means
        ``mu0`` everywhere.
    t_rise
        Signal 10-90% rise time [s].  Converted to the knee frequency
        ``f = 0.35 / t_rise``, which is where a trapezoidal edge's spectrum
        rolls off.
    freq
        Excitation frequency [Hz].  If both ``t_rise`` and ``freq`` are given
        the higher of the two frequencies is used, because the check must be
        governed by the fastest thing in the problem.

    Returns
    -------
    Report
        ``details`` carries ``L_m``, ``lambda_m``, ``L_over_lambda``,
        ``freq_Hz``, ``eps_r``, ``mu_r``, ``band``, plus the largest frequency
        and smallest rise time that would put the problem in the "excellent"
        band, and the largest domain that would do so at this frequency.

    Raises
    ------
    ValueError
        If neither ``t_rise`` nor ``freq`` is given, if either is non-positive,
        or if the material arrays have the wrong shape or look like relative
        (rather than absolute) permittivity or permeability.
    """
    eps = _require_absolute_eps(_as_cell_array(eps_cell, grid, "eps_cell"), "eps_cell")
    if mu_cell is None:
        mu = np.full(grid.shape_cells, mu0)
    else:
        mu = _require_absolute_mu(_as_cell_array(mu_cell, grid, "mu_cell"), "mu_cell")

    if t_rise is None and freq is None:
        raise ValueError("check_quasistatic needs t_rise [s] or freq [Hz]")
    f_from_rise = (None if t_rise is None
                   else 0.35 / _require_positive(t_rise, "t_rise", "time in seconds"))
    f_direct = (None if freq is None
                else _require_positive(freq, "freq", "frequency in Hz"))
    f = max(v for v in (f_from_rise, f_direct) if v is not None)

    eps_r = eps / eps0
    mu_r = mu / mu0
    n2 = eps_r * mu_r
    worst = int(np.argmax(n2))
    n = float(np.sqrt(n2.flat[worst]))

    lo_hi = grid.bounds
    L = max(float(lo_hi[d][1] - lo_hi[d][0]) for d in _resolved(grid))
    lam = c0 / (f * n)
    ratio = L / lam

    det: dict[str, Any] = {
        "L_m": L,
        "freq_Hz": f,
        "wavelength_m": lam,
        "L_over_lambda": ratio,
        "eps_r_worst": float(eps_r.flat[worst]),
        "mu_r_worst": float(mu_r.flat[worst]),
        "refractive_index": n,
        "freq_source": ("t_rise knee" if f_from_rise is not None and f == f_from_rise
                        else "freq"),
        # What it would take to be firmly quasi-static.
        "freq_for_excellent_Hz": QS_EXCELLENT * c0 / (L * n),
        "t_rise_for_excellent_s": 0.35 * L * n / (QS_EXCELLENT * c0),
        "L_for_excellent_m": QS_EXCELLENT * lam,
    }
    det.update(_locate(worst, grid.shape_cells, grid))

    fix = (f"To reach L/lambda < {QS_EXCELLENT:g} either keep the domain below "
           f"{det['L_for_excellent_m']:.4g} m, or keep the excitation below "
           f"{det['freq_for_excellent_Hz']:.4g} Hz "
           f"(rise time above {det['t_rise_for_excellent_s']:.4g} s).")
    # Naming a "worst cell" is only informative when the material varies.
    homogeneous = bool(np.all(n2 == n2.flat[worst]))
    det["material_homogeneous"] = homogeneous
    mat = (f"eps_r = {det['eps_r_worst']:.4g}, mu_r = {det['mu_r_worst']:.4g}"
           + ("" if homogeneous else f", at {_loc_str(det)}"))
    common = (f"L = {L:.4g} m, f = {f:.4g} Hz, lambda = {lam:.4g} m in the "
              f"{'material' if homogeneous else 'slowest material'} ({mat}), "
              f"so L/lambda = {ratio:.4g}.")

    if ratio < QS_EXCELLENT:
        det["band"] = "< 0.01 excellent"
        level = "ok"
        msg = (f"Quasi-static is excellent here (error below ~1%). {common} "
               f"Use eqs, mqs or darwin; fdtd would cost "
               f"{grid.courant_dt():.4g} s per step for no benefit.")
    elif ratio < QS_GOOD:
        det["band"] = "0.01 - 0.1 good"
        level = "ok"
        msg = (f"Quasi-static is good, but inductive and capacitive coupling are "
               f"both significant at this size. {common} Prefer DarwinSolver over "
               f"EQSSolver or MQSSolver alone, since each of those drops one of "
               f"the two. {fix}")
    elif ratio < QS_MARGINAL:
        det["band"] = "0.1 - 0.3 marginal"
        level = "warn"
        msg = (f"Marginal for a quasi-static model. {common} Cross-check the "
               f"answer against FDTDSolver on the same grid before trusting it; "
               f"expect several percent error in the terminal quantities. {fix}")
    else:
        det["band"] = "> 0.3 full wave"
        level = "error"
        msg = (f"The structure is electrically LARGE, so the quasi-static "
               f"solvers are the wrong model. {common} Wave propagation, "
               f"retardation and radiation all matter above L/lambda = "
               f"{QS_MARGINAL:g}, and this is a modelling error, not a "
               f"discretisation error: refining the mesh will NOT reduce it. "
               f"Use FDTDSolver. {fix}")

    return Report("check_quasistatic", level, msg, det, "A1")


# ==========================================================================
# A4a --- skin depth resolution
# ==========================================================================
def check_skin_depth(grid: RectilinearGrid,
                     sigma_cell: np.ndarray,
                     mu_cell: np.ndarray | None,
                     freq: float) -> Report:
    """Does the mesh resolve the skin depth inside every conductor? (A4a)

    ``delta = sqrt(2 / (omega * mu * sigma))``.  The volumetric-conductivity
    treatment used by eqs / mqs / darwin reproduces skin and proximity effects
    exactly, but only if the current profile is sampled: fewer than
    ``SKIN_CELLS_PER_DELTA`` (3) cells per skin depth and the AC resistance
    comes out low, because a coarse cell averages the exponential profile into
    a uniform one.

    Parameters
    ----------
    grid
        The grid.
    sigma_cell
        Conductivity per cell [S/m], shape ``(Nx, Ny, Nz)``.  Cells with
        ``sigma <= 0`` are ignored.
    mu_cell
        Absolute permeability per cell [H/m], or ``None`` for ``mu0``.
    freq
        Frequency at which to evaluate the skin depth [Hz].  For a transient
        run use the knee frequency ``0.35 / t_rise``.

    Returns
    -------
    Report
        ``details`` carries ``delta_min_m``, ``delta_max_m``, the number of
        conducting and under-resolved cells, the worst
        ``min_cells_per_skin_depth``, ``cells_per_skin_depth_by_axis``, the
        worst cell's index and position, and the cell size that would fix it.
        The resolution keys are absent when no cell conducts at all, because
        there is then nothing to resolve and any number would be an invention.

    Notes
    -----
    Two escapes keep the criterion from firing where it is vacuous.

    A conductor thinner than its own skin depth carries an essentially uniform
    current, so the check compares ``delta`` against the bounding-box size of
    the conducting region first and reports ``ok`` when the skin depth exceeds
    it.  With several disjoint conductors the bounding box is larger than any
    one of them, which makes the escape harder to trigger --- conservative in
    the safe direction.

    The cell size compared against ``delta`` is the largest cell dimension among
    the axes in which the conductor is *bounded*, not the largest dimension
    outright: see :func:`_surface_normal_widths`.  A direction in which the
    conductor runs from wall to wall has no surface inside the domain and no
    skin profile to resolve, which is the usual situation along a transmission
    line.  A conductor that fills the domain in every direction is ambiguous
    (infinite slab, or a wire meshed to its own surface), so there the largest
    dimension is used after all and the verdict is capped at ``warn``.

    In full-wave runs the alternative to meshing through ``delta`` is the
    surface-impedance boundary condition (A4b), which is valid in exactly the
    regime this check complains about.
    """
    sigma = _as_cell_array(sigma_cell, grid, "sigma_cell")
    if np.any(sigma < 0.0):
        raise ValueError("sigma_cell must be non-negative [S/m]")
    if mu_cell is None:
        mu = np.full(grid.shape_cells, mu0)
    else:
        mu = _require_absolute_mu(_as_cell_array(mu_cell, grid, "mu_cell"), "mu_cell")
    freq = _require_positive(freq, "freq", "frequency in Hz")

    omega = 2.0 * np.pi * float(freq)
    cond = sigma > 0.0
    n_cond = int(cond.sum())
    det: dict[str, Any] = {"freq_Hz": float(freq), "n_conducting_cells": n_cond,
                           "target_cells_per_skin_depth": SKIN_CELLS_PER_DELTA}
    if n_cond == 0:
        det["n_under_resolved"] = 0
        return Report("check_skin_depth", "ok",
                      "No cell has sigma > 0, so there is no skin effect to "
                      "resolve. If that is a surprise, the conductivity array "
                      "was probably never assigned.", det, "A4a")

    delta = np.full(grid.shape_cells, np.inf)
    delta[cond] = np.sqrt(2.0 / (omega * mu[cond] * sigma[cond]))
    det["delta_min_m"] = float(delta[cond].min())
    det["delta_max_m"] = float(delta[cond].max())
    det["sigma_max_S_per_m"] = float(sigma.max())

    # Extent of the conducting region, used for the thin-conductor escape.
    span = 0.0
    for d in _resolved(grid):
        proj = cond.any(axis=tuple(a for a in range(3) if a != d))
        idx = np.flatnonzero(proj)
        nodes = (grid.xn, grid.yn, grid.zn)[d]
        span = max(span, float(nodes[idx[-1] + 1] - nodes[idx[0]]))
    det["conductor_span_m"] = span

    h, checked, per_axis, ambiguous = _surface_normal_widths(grid, cond)
    by_axis: dict[str, float] = {}
    for name, w in per_axis.items():
        sel = checked & np.isfinite(w)
        if sel.any():
            by_axis[name] = float(np.min(delta[sel] / w[sel]))
    det["cells_per_skin_depth_by_axis"] = by_axis
    det["surface_normal_axes"] = sorted(by_axis)
    det["n_cells_with_a_surface"] = int(checked.sum())
    det["conductor_fills_the_domain"] = ambiguous

    res = np.full(grid.shape_cells, np.inf)
    res[checked] = delta[checked] / h[checked]
    worst = int(np.argmin(np.where(checked, res, np.inf)))
    worst_res = float(res.flat[worst])
    worst_axis = next((name for name, w in per_axis.items()
                       if w.flat[worst] == h.flat[worst]), "")
    det["min_cells_per_skin_depth"] = worst_res
    det["worst_cell_size_m"] = float(h.flat[worst])
    det["worst_normal_axis"] = worst_axis
    det["worst_delta_m"] = float(delta.flat[worst])
    det["required_cell_size_m"] = float(delta.flat[worst] / SKIN_CELLS_PER_DELTA)
    det["n_under_resolved"] = int(np.count_nonzero(res < SKIN_CELLS_PER_DELTA))
    det.update(_locate(worst, grid.shape_cells, grid))

    if det["delta_min_m"] >= span:
        return Report(
            "check_skin_depth", "ok",
            f"The skin depth ({det['delta_min_m']:.4g} m at {freq:.4g} Hz) is at "
            f"least as large as the whole conducting region "
            f"({span:.4g} m), so the current density is essentially uniform and "
            f"no skin-depth meshing is required. The mesh currently gives "
            f"{worst_res:.4g} cells per skin depth, which does not matter here.",
            det, "A4a")

    n_checked = det["n_cells_with_a_surface"]
    frac = 100.0 * det["n_under_resolved"] / n_checked
    where = _loc_str(det)
    axes = ", ".join(f"{k} {v:.4g}" for k, v in sorted(by_axis.items()))
    skipped_axes = [name for name in ("xyz"[d] for d in _resolved(grid))
                    if name not in by_axis]
    axis_note = (f" Cells per delta by direction: {axes}."
                 + (f" Direction(s) {', '.join(skipped_axes)} carry no conductor "
                    f"surface (the conductor spans the domain there), so the cell "
                    f"size along them cannot affect the current profile and is "
                    f"not counted." if skipped_axes else "")
                 + (" The conductor fills the domain in every resolved "
                    "direction, so whether these cells face a surface at all "
                    "depends on what the walls mean: if they are symmetry planes "
                    "the conductor is infinite and none of this matters, and if "
                    "they are the conductor's own surface it all does. That "
                    "ambiguity is why this cannot be reported as an error."
                    if ambiguous else ""))
    if worst_res >= SKIN_CELLS_PER_DELTA:
        level = "ok"
        msg = (f"Every conducting cell resolves the skin depth: the worst is "
               f"{worst_res:.4g} cells per delta (target "
               f"{SKIN_CELLS_PER_DELTA:g}), with delta between "
               f"{det['delta_min_m']:.4g} and {det['delta_max_m']:.4g} m at "
               f"{freq:.4g} Hz.{axis_note}")
    else:
        level = "warn" if (worst_res >= 1.0 or ambiguous) else "error"
        sev = ("the current profile is not represented at all"
               if worst_res < 1.0 else "the AC resistance will come out low")
        msg = (f"{det['n_under_resolved']} of {n_checked} conducting cells with "
               f"a surface ({frac:.1f}%) are coarser than "
               f"delta/{SKIN_CELLS_PER_DELTA:g}, so {sev}. Worst: {where} has "
               f"h = {det['worst_cell_size_m']:.4g} m along "
               f"{det['worst_normal_axis'] or '?'} against delta = "
               f"{det['worst_delta_m']:.4g} m, i.e. {worst_res:.4g} cells per "
               f"skin depth. Refine that region to "
               f"{det['required_cell_size_m']:.4g} m or finer (use "
               f"grid.auto_mesh_1d with the conductor surfaces as features), "
               f"lower the analysis frequency, or --- in fdtd --- switch the "
               f"conductor to the surface-impedance treatment of A4b instead of "
               f"meshing through it.{axis_note}")
    return Report("check_skin_depth", level, msg, det, "A4a")


# ==========================================================================
# A10 --- mesh quality
# ==========================================================================
def check_mesh_quality(grid: RectilinearGrid) -> Report:
    """Grading and aspect ratio of the mesh (A10, A2).

    The box method is second-order accurate on a smooth mesh and degrades
    toward first order where the cell size jumps.  Two numbers describe that:

    * **growth ratio** --- the largest ratio between neighbouring cell widths
      along an axis.  Above ``GROWTH_WARN`` (1.5) the convergence order is
      measurably degraded, above ``GROWTH_ERROR`` (2.0) the discretisation is
      no longer trustworthy in that region.
    * **aspect ratio** --- the largest ratio between two dimensions of a single
      cell, ``max(h[d]) / min(h[e])`` over distinct resolved axes.  A
      tensor-product grid has no element-quality failure mode (A2), so a large
      aspect ratio costs conditioning rather than accuracy; it is reported and
      warned about only at ``ASPECT_WARN`` (1e3), where an iterative solver
      starts to struggle.

    ``grid.max_aspect_ratio()`` is *not* used for that verdict and is reported
    separately as ``size_range``: it takes ``max(h) / min(h)`` over all resolved
    axes at once, which also admits both ends coming from the same graded axis,
    so on a graded mesh it exceeds the aspect ratio of every cell actually
    present and would raise the report to ``warn`` on a quantity no cell attains.

    Parameters
    ----------
    grid
        The grid to inspect.  Collapsed directions are excluded from the aspect
        ratio, since their nominal thickness is a normalisation.

    Returns
    -------
    Report
        ``details`` carries ``max_growth_ratio``, the axis and index where it
        occurs and the coordinate there, ``max_aspect_ratio`` with the pair of
        axes that produces it, ``size_range``, the smallest and largest cell,
        and the cell counts.
    """
    growth = grid.max_growth_ratio()
    aspect, wide_axis, thin_axis = _max_cell_aspect_ratio(grid)

    worst_axis, worst_i, worst_val = "x", 0, 1.0
    for name, h in zip("xyz", _widths(grid)):
        if h.size < 2:
            continue
        r = h[1:] / h[:-1]
        r = np.maximum(r, 1.0 / r)
        i = int(np.argmax(r))
        if float(r[i]) > worst_val:
            worst_axis, worst_i, worst_val = name, i, float(r[i])

    nodes = {"x": grid.xn, "y": grid.yn, "z": grid.zn}[worst_axis]
    hs = [h for d, h in enumerate(_widths(grid)) if d in _resolved(grid)]
    det: dict[str, Any] = {
        "max_growth_ratio": float(growth),
        "growth_axis": worst_axis,
        "growth_between_cells": (worst_i, worst_i + 1),
        "growth_position_m": float(nodes[worst_i + 1]),
        "max_aspect_ratio": float(aspect),
        "aspect_axes": (wide_axis, thin_axis),
        # Reported for provenance only: max(h)/min(h) over all resolved axes,
        # which is a grading range rather than any cell's shape.
        "size_range": float(grid.max_aspect_ratio()),
        "min_cell_m": float(min(float(h.min()) for h in hs)),
        "max_cell_m": float(max(float(h.max()) for h in hs)),
        "n_cells": int(grid.n_cells),
        "n_nodes": int(grid.n_nodes),
        "ndim_effective": int(grid.ndim_effective),
        "growth_warn": GROWTH_WARN,
        "growth_error": GROWTH_ERROR,
    }

    where = (f"between {worst_axis}-cells {worst_i} and {worst_i + 1}, at "
             f"{worst_axis} = {det['growth_position_m']:.4g} m")
    parts: list[str] = []
    if growth > GROWTH_ERROR:
        level = "error"
        parts.append(
            f"Mesh growth ratio {growth:.4g} exceeds the hard limit "
            f"{GROWTH_ERROR:g} ({where}). The box method loses its second-order "
            f"truncation error across a jump that large, and the local answer is "
            f"not trustworthy. Rebuild the axis with auto_mesh_1d(..., "
            f"growth=1.4) or insert intermediate node coordinates there.")
    elif growth > GROWTH_WARN:
        level = "warn"
        parts.append(
            f"Mesh growth ratio {growth:.4g} exceeds {GROWTH_WARN:g} ({where}). "
            f"Convergence is between first and second order in that region. "
            f"Keep grading below 1.4 for production runs: auto_mesh_1d(..., "
            f"growth=1.4).")
    else:
        level = "ok"
        # Below 1.001 the "worst" location is round-off in the node coordinates
        # rather than real grading, so quoting it would be misleading.
        parts.append(
            f"Mesh grading is smooth: worst neighbouring-cell ratio {growth:.4g} "
            + (f"(limit {GROWTH_WARN:g})." if growth <= 1.001
               else f"(limit {GROWTH_WARN:g}), {where}."))

    if aspect > ASPECT_WARN:
        level = LEVELS[max(_RANK[level], _RANK["warn"])]
        parts.append(
            f"Cell aspect ratio reaches {aspect:.4g} (widest {wide_axis}-cell "
            f"{_widths(grid)['xyz'.index(wide_axis)].max():.4g} m against "
            f"narrowest {thin_axis}-cell "
            f"{_widths(grid)['xyz'.index(thin_axis)].min():.4g} m). That is not "
            f"an accuracy problem on a tensor-product grid (A2), but it drives "
            f"the condition number, so prefer config.linear_solver='direct' or "
            f"'amg' over plain 'cg' at this anisotropy.")
    else:
        parts.append(f"Cell aspect ratio {aspect:.4g} is benign.")

    parts.append(f"{grid.n_cells:,} cells, {grid.n_nodes:,} nodes, "
                 f"{grid.ndim_effective}D effective.")
    return Report("check_mesh_quality", level, " ".join(parts), det, "A10")


# ==========================================================================
# A12 --- open-boundary padding
# ==========================================================================
def check_padding(grid: RectilinearGrid,
                  conductor_mask: np.ndarray) -> Report:
    """Is the domain padded enough for the Neumann wall not to distort fringing? (A12)

    The default quasi-static boundary is homogeneous Neumann, which forbids
    flux from leaving the box.  For a shielded or symmetric problem that is
    exact; for an open one it squeezes the fringing field back into the domain
    and **under-estimates capacitance**.  A12 asks for at least
    ``PAD_WARN`` (3) times the largest conductor dimension of clearance on
    every open wall.

    Parameters
    ----------
    grid
        The grid.
    conductor_mask
        Boolean ``(Nx, Ny, Nz)`` array, ``True`` in cells belonging to any
        conductor.

    Returns
    -------
    Report
        ``details`` carries ``feature_size_m`` (the fringing scale, see Notes),
        a ``gap_m`` and ``pad_ratio`` per open wall, the list of flush walls, the
        spanning axes and the worst wall.  ``n_conductor_components`` appears
        only when some axis is spanned, since that is the only case in which the
        conductors have to be labelled separately.

    Notes
    -----
    A wall the conductor *touches* is treated as intentional --- a ground plane
    that runs off the edge of the domain, or a symmetry cut --- and is reported
    but not warned about, because warning there would fire on the majority of
    correctly-posed problems.  For the same reason, an axis along which a
    conductor reaches *both* walls does not contribute to that conductor's
    feature size: in that direction the structure is effectively infinite and
    its in-domain length is an artifact of where the box was cut.  Collapsed
    directions are skipped entirely; they are symmetry directions by
    construction.

    That exclusion is applied **per connected conductor**, not to the union of
    them, which matters more than it sounds.  Applied to the union, a single
    ground plane running wall to wall in x deletes the x-extent of every signal
    conductor above it, and a truncated microstrip cross-section --- the exact
    case A12 exists for --- comes back ``ok`` because the only surviving length
    is the trace *thickness*.  The feature size is therefore the largest of: the
    union extent along each axis no conductor spans, and each individual
    conductor's largest bounded dimension.  Keeping the union term as well is
    what makes two neighbouring traces judged against the width of the pair
    rather than of one trace.
    """
    mask = np.asarray(conductor_mask)
    if mask.shape != grid.shape_cells:
        raise ValueError(
            f"conductor_mask must have shape {grid.shape_cells}, got {mask.shape}")
    mask = mask.astype(bool)

    det: dict[str, Any] = {"pad_warn": PAD_WARN, "pad_error": PAD_ERROR}
    if not mask.any():
        det["n_conductor_cells"] = 0
        return Report("check_padding", "warn",
                      "conductor_mask selects no cells, so padding cannot be "
                      "assessed. Pass the mask of the metal cells (for example "
                      "sigma_cell > 1e3) if you want this check to run.",
                      det, "A12")
    det["n_conductor_cells"] = int(mask.sum())

    dims = _resolved(grid)
    nodes_all = (grid.xn, grid.yn, grid.zn)
    extents: dict[str, float] = {}
    gaps: dict[str, float] = {}
    spanning: list[str] = []
    for d in dims:
        proj = mask.any(axis=tuple(a for a in range(3) if a != d))
        idx = np.flatnonzero(proj)
        nd = nodes_all[d]
        lo_face, hi_face = float(nd[idx[0]]), float(nd[idx[-1] + 1])
        name = "xyz"[d]
        extents[name] = hi_face - lo_face
        gaps[name + "lo"] = lo_face - float(nd[0])
        gaps[name + "hi"] = float(nd[-1]) - hi_face
        if gaps[name + "lo"] <= 0.0 and gaps[name + "hi"] <= 0.0:
            spanning.append(name)

    # A conductor that runs off both ends of an axis is effectively infinite in
    # that direction (a ground plane, a long trace in a 2D cross-section), so
    # its in-domain length is an artifact of where the box was cut, not a
    # feature size that sets the fringing scale.  Judging padding against it
    # would flag every correctly-posed transmission-line cross-section.  The
    # exclusion is per conductor: see the Notes.
    if spanning:
        comp_features, n_comp = _component_features(mask, grid, dims)
        det["n_conductor_components"] = n_comp
        if comp_features:
            det["largest_component_feature_m"] = max(comp_features)
    else:
        comp_features, n_comp = [], 1
    union_usable = [v for k, v in extents.items() if k not in spanning]
    feature = (max(union_usable + comp_features)
               if (union_usable or comp_features) else max(extents.values()))
    det["feature_size_m"] = feature
    det["spanning_axes"] = spanning
    det["conductor_extent_m"] = {k: float(v) for k, v in extents.items()}
    det["gap_m"] = {k: float(v) for k, v in gaps.items()}
    det["pad_ratio"] = {k: (float(v / feature) if feature > 0 else np.inf)
                        for k, v in gaps.items()}

    flush = [k for k, v in gaps.items() if v <= 0.0]
    open_walls = {k: v for k, v in det["pad_ratio"].items() if k not in flush}
    det["flush_walls"] = flush

    if not open_walls:
        return Report("check_padding", "ok",
                      f"The conductor reaches all {len(flush)} resolved walls "
                      f"({', '.join(flush)}), so every boundary is being used as "
                      f"a shield or a symmetry plane and there is no fringing "
                      f"field to truncate. Nothing to pad.", det, "A12")

    worst_wall = min(open_walls, key=lambda k: open_walls[k])
    worst = open_walls[worst_wall]
    det["worst_wall"] = worst_wall
    det["worst_pad_ratio"] = float(worst)
    det["required_gap_m"] = float(PAD_WARN * feature)
    det["extra_needed_m"] = float(max(0.0, PAD_WARN * feature - gaps[worst_wall]))

    flush_note = (f" Walls {', '.join(flush)} are flush with the conductor and "
                  f"are assumed to be shields or symmetry planes."
                  if flush else "")
    if spanning:
        flush_note += (f" Some conductor runs the full width of "
                       f"{', '.join(spanning)}, so it is treated as infinite "
                       f"there and that length is excluded from its own feature "
                       f"size (but not from that of the other "
                       f"{max(0, n_comp - 1)} conductor(s)).")
    if worst >= PAD_WARN:
        level = "ok"
        msg = (f"Padding is adequate: the tightest open wall, {worst_wall}, is "
               f"{gaps[worst_wall]:.4g} m from the conductor, which is "
               f"{worst:.4g}x the largest conductor dimension "
               f"({feature:.4g} m); A12 asks for {PAD_WARN:g}x.{flush_note}")
    else:
        level = "warn" if worst >= PAD_ERROR else "error"
        sev = ("severely truncated" if level == "error" else "truncated")
        msg = (f"Fringing field is {sev} at wall {worst_wall}: the gap is "
               f"{gaps[worst_wall]:.4g} m, only {worst:.4g}x the largest "
               f"conductor dimension ({feature:.4g} m) against the "
               f"{PAD_WARN:g}x that A12 asks for. The homogeneous Neumann "
               f"boundary reflects the field back in, so extracted capacitance "
               f"will be UNDER-estimated. Extend that wall by at least "
               f"{det['extra_needed_m']:.4g} m (total gap "
               f"{det['required_gap_m']:.4g} m) --- graded cells make this "
               f"nearly free --- or state that the wall is a real shield or a "
               f"symmetry plane, in which case the answer is already "
               f"correct.{flush_note}")
    return Report("check_padding", level, msg, det, "A12")


# ==========================================================================
# Dielectric relaxation vs the time step
# ==========================================================================
def check_dielectric_relaxation(eps_cell: np.ndarray,
                                sigma_cell: np.ndarray,
                                dt: float,
                                grid: RectilinearGrid | None = None) -> Report:
    """Is the charge-relaxation time resolved, ignored, or awkwardly in between?

    Every conducting cell has a dielectric relaxation time ``tau = eps / sigma``
    over which injected space charge decays.  Three regimes exist, and only one
    of them is a problem:

    * ``tau >> dt`` --- resolved.  The cell behaves capacitively and the
      transient is captured.
    * ``tau << dt`` --- fully relaxed.  Backward Euler is L-stable, so it
      returns the correct asymptotic (resistive) limit.  Copper sits here by a
      factor of 1e7 and that is exactly what makes implicit quasi-static
      stepping possible: an explicit scheme would need ``dt <= 2 tau``, i.e.
      ~3e-19 s, which is why A1's opening argument holds.
    * ``tau ~ dt`` --- neither.  The relaxation is half-integrated, the answer
      in those cells has O(1) error, and nothing about the output looks wrong.
      This check exists to find that band.

    Parameters
    ----------
    eps_cell
        Absolute permittivity per cell [F/m].
    sigma_cell
        Conductivity per cell [S/m], same shape.  Cells with ``sigma <= 0``
        have infinite ``tau`` and are skipped.
    dt
        Time step [s] the transient solver will use.
    grid
        Optional.  When given, both arrays must have shape ``grid.shape_cells``
        and the worst cell index is turned into a physical position.  Without it
        any common shape is accepted and no position is reported.

    Returns
    -------
    Report
        ``details`` carries ``tau_min_s``, ``tau_max_s``, the stiffness ratio,
        the count of marginal cells, the worst ``tau_over_dt``, and both ways
        out (a ``dt`` small enough to resolve, or large enough to relax).

    Raises
    ------
    ValueError
        Shape mismatch --- between the two arrays, or against ``grid`` when one
        is given --- non-positive ``dt``, negative ``sigma``, or a permittivity
        that looks relative rather than absolute.
    """
    if grid is not None:
        # Without this a wrong-shaped array either crashes with IndexError deep
        # inside _locate or, worse, reports a position unravelled against the
        # wrong shape.  Ground rule 7: raise eagerly, never coerce.
        eps = _as_cell_array(eps_cell, grid, "eps_cell")
        sigma = _as_cell_array(sigma_cell, grid, "sigma_cell")
    else:
        eps = np.asarray(eps_cell, dtype=float)
        sigma = np.asarray(sigma_cell, dtype=float)
        if not np.all(np.isfinite(eps)) or not np.all(np.isfinite(sigma)):
            raise ValueError("eps_cell / sigma_cell contain non-finite values")
    if eps.shape != sigma.shape:
        raise ValueError(
            f"eps_cell {eps.shape} and sigma_cell {sigma.shape} must have the "
            f"same shape")
    if np.any(sigma < 0.0):
        raise ValueError("sigma_cell must be non-negative [S/m]")
    _require_absolute_eps(eps, "eps_cell")
    dt = _require_positive(dt, "dt", "time in seconds")

    cond = sigma > 0.0
    det: dict[str, Any] = {"dt_s": dt, "n_conducting_cells": int(cond.sum())}
    if not cond.any():
        return Report("check_dielectric_relaxation", "ok",
                      "No cell has sigma > 0, so there is no charge relaxation: "
                      "the problem is purely capacitive and any dt is stable.",
                      det, "A1")

    tau = np.full(eps.shape, np.inf)
    tau[cond] = eps[cond] / sigma[cond]
    tmin, tmax = float(tau[cond].min()), float(tau[cond].max())
    det["tau_min_s"] = tmin
    det["tau_max_s"] = tmax
    det["stiffness_ratio"] = tmax / tmin if tmin > 0 else np.inf
    det["explicit_dt_limit_s"] = 2.0 * tmin
    det["implicit_speedup_vs_relaxation"] = dt / (2.0 * tmin)

    # "Marginal" = the step neither resolves the relaxation nor completes it.
    ratio = tau / dt
    marginal = cond & (ratio > 0.05) & (ratio < 10.0)
    det["n_marginal_cells"] = int(marginal.sum())

    if marginal.any():
        # The worst offender is the one closest to tau == dt on a log scale.
        score = np.where(marginal, np.abs(np.log(np.where(marginal, ratio, 1.0))),
                         np.inf)
        worst = int(np.argmin(score))
        det["worst_tau_s"] = float(tau.flat[worst])
        det["worst_tau_over_dt"] = float(ratio.flat[worst])
        det["dt_to_resolve_s"] = float(tau.flat[worst] / 10.0)
        det["dt_to_relax_s"] = float(tau.flat[worst] / 0.05)
        det.update(_locate(worst, eps.shape, grid))
        msg = (f"{det['n_marginal_cells']} of {det['n_conducting_cells']} "
               f"conducting cells have a relaxation time within a decade of the "
               f"time step. Worst: {_loc_str(det)} has tau = "
               f"{det['worst_tau_s']:.4g} s against dt = {dt:.4g} s "
               f"(tau/dt = {det['worst_tau_over_dt']:.4g}). Backward Euler "
               f"neither resolves that transient nor completes it, so the space "
               f"charge in those cells carries O(1) error. Either step down to "
               f"{det['dt_to_resolve_s']:.4g} s to resolve it, or up to "
               f"{det['dt_to_relax_s']:.4g} s to accept the relaxed resistive "
               f"limit deliberately. Relaxation times in this problem span "
               f"{tmin:.4g} to {tmax:.4g} s.")
        level = "warn"
    else:
        resolved = int(np.count_nonzero(cond & (ratio >= 10.0)))
        relaxed = int(np.count_nonzero(cond & (ratio <= 0.05)))
        det["n_resolved_cells"] = resolved
        det["n_relaxed_cells"] = relaxed
        msg = (f"Charge relaxation is cleanly separated from the time step: "
               f"{resolved} cells resolve it (tau >= 10 dt) and {relaxed} are "
               f"fully relaxed (tau <= dt/20), with none in between. tau spans "
               f"{tmin:.4g} to {tmax:.4g} s against dt = {dt:.4g} s. An explicit "
               f"scheme would need dt <= {det['explicit_dt_limit_s']:.4g} s "
               f"here, so implicit stepping is buying a factor of "
               f"{det['implicit_speedup_vs_relaxation']:.4g}.")
        level = "ok"
    return Report("check_dielectric_relaxation", level, msg, det, "A1")


# ==========================================================================
# A5 / A7 --- Debye length resolution
# ==========================================================================
def check_debye_length(doping: np.ndarray,
                       T: float = T_ROOM,
                       grid: RectilinearGrid | None = None,
                       eps: np.ndarray | float | None = None,
                       eps_r: float = 11.7) -> Report:
    """Does the mesh resolve the Debye screening length in the semiconductor? (A5, A7)

    ``LD = sqrt(eps kT / (q^2 N))`` with ``N`` the majority-carrier density,
    taken equal to the net doping under complete ionisation (A11).  It is the
    distance over which the semiconductor screens a potential step, so it sets
    the width of the space-charge transition at every junction and at every
    accumulation or inversion layer.

    Scharfetter-Gummel (A7) keeps the discretisation *stable* at any cell size
    --- that is what it is for --- but it does not make it *accurate*: a cell
    wider than ``LD`` smears the depletion edge, which shows up directly as an
    error in junction capacitance and in the subthreshold slope.  Aim for at
    least one cell per Debye length, and two or three across a junction.

    Parameters
    ----------
    doping
        Net doping magnitude ``|Nd - Na|`` [m^-3].  Use
        ``1e17 * units.per_cm3`` to write a cm^-3 number.  Node-shaped
        ``(Nx+1, Ny+1, Nz+1)`` or cell-shaped ``(Nx, Ny, Nz)`` are both
        accepted when a ``grid`` is given; any shape is accepted without one.
        Cells with zero doping screen over the intrinsic length and are skipped.
    T
        Lattice temperature [K] (A6: uniform and constant).
    grid
        Optional.  Without it the check reports the Debye length statistics and
        the cell size they demand; with it, the mesh is actually compared.
    eps
        Absolute permittivity [F/m], scalar or an array shaped like ``doping``.
        Overrides ``eps_r``.
    eps_r
        Relative permittivity used when ``eps`` is not given.  Default 11.7
        (silicon).

    Returns
    -------
    Report
        ``details`` carries ``LD_min_m``, ``LD_max_m``, ``N_max_per_m3``, the
        required cell size, and --- with a grid --- the worst
        ``min_cells_per_debye_length``,
        ``cells_per_debye_length_by_axis``, the best-resolved direction, and the
        worst location.

    Raises
    ------
    ValueError
        Non-positive ``T``, non-finite doping, ``eps_r < 1``, an ``eps`` that
        does not broadcast against ``doping``, or a ``doping`` array that
        matches neither the node nor the cell shape of the supplied grid.  A
        signed doping array is *not* an error: only the magnitude
        ``|Nd - Na|`` enters the Debye length, so the sign is taken off.

    Notes
    -----
    Unlike the skin depth, the direction the screening happens in cannot be
    recovered from the inputs: it is set by the junctions and the gate stack,
    not by the doping array alone (a uniformly doped MOS capacitor screens
    normal to an oxide interface this function never sees).  The headline number
    is therefore the *worst* resolved direction, which is the conservative
    reading, but a mesh that resolves LD in at least one direction everywhere
    and fails in another is capped at ``warn``: on a planar device that is the
    normal, correct mesh (fine vertically, coarse laterally), and calling it an
    ``error`` --- "the model is the wrong model" --- would abort a valid run
    through :meth:`Report.raise_if_error`.  ``cells_per_debye_length_by_axis``
    reports every direction so the caller can apply its own knowledge of the
    geometry.
    """
    N = np.abs(np.asarray(doping, dtype=float))
    if not np.all(np.isfinite(N)):
        raise ValueError("doping contains non-finite values")
    T = _require_positive(T, "T", "temperature in kelvin")
    kind = "cell"
    if grid is not None:
        if N.shape == grid.shape_nodes:
            kind = "node"
        elif N.shape == grid.shape_cells:
            kind = "cell"
        else:
            raise ValueError(
                f"doping must have shape {grid.shape_nodes} (nodes) or "
                f"{grid.shape_cells} (cells), got {N.shape}")

    if eps is None:
        if eps_r < 1.0:
            raise ValueError(f"eps_r must be >= 1, got {eps_r}")
        eps_arr = np.full(N.shape, float(eps_r) * eps0)
    else:
        eps_arr = np.asarray(eps, dtype=float)
        _require_absolute_eps(np.atleast_1d(eps_arr), "eps")
        try:
            eps_arr = np.broadcast_to(eps_arr, N.shape).astype(float)
        except ValueError as exc:
            raise ValueError(
                f"eps with shape {np.shape(eps)} does not broadcast against "
                f"doping with shape {N.shape}") from exc

    doped = N > 0.0
    det: dict[str, Any] = {"T_K": float(T), "n_doped_cells": int(doped.sum()),
                           "thermal_voltage_V": float(kB * T / q)}
    if not doped.any():
        return Report("check_debye_length", "ok",
                      "Net doping is zero everywhere, so screening happens over "
                      "the intrinsic Debye length and there is no doping-set "
                      "mesh requirement. If this array was meant to hold a "
                      "profile, it was never filled in.", det, "A5, A7")

    LD = np.full(N.shape, np.inf)
    LD[doped] = np.sqrt(eps_arr[doped] * kB * T / (q * q * N[doped]))
    det["LD_min_m"] = float(LD[doped].min())
    det["LD_max_m"] = float(LD[doped].max())
    det["N_max_per_m3"] = float(N.max())
    det["N_max_per_cm3"] = float(N.max()) / 1e6
    det["required_cell_size_m"] = det["LD_min_m"]

    if grid is None:
        return Report(
            "check_debye_length", "ok",
            f"Debye length runs from {det['LD_min_m']:.4g} m (at N = "
            f"{det['N_max_per_cm3']:.4g} cm^-3) to {det['LD_max_m']:.4g} m at "
            f"T = {T:.4g} K. The mesh must put at least one cell inside "
            f"{det['LD_min_m']:.4g} m wherever that doping appears, and two or "
            f"three across a junction. Pass grid= to have that checked rather "
            f"than merely stated.", det, "A5, A7")

    by_axis_h = _widths_by_axis(grid, kind)
    # Pairwise rather than ufunc.reduce over the list, which would first stack
    # the broadcast views into one real (naxes, Nx, Ny, Nz) array.
    h = functools.reduce(np.maximum, by_axis_h.values())       # worst direction
    h_best = functools.reduce(np.minimum, by_axis_h.values())  # best direction
    res = np.full(N.shape, np.inf)
    res[doped] = LD[doped] / h[doped]
    res_best = np.full(N.shape, np.inf)
    res_best[doped] = LD[doped] / h_best[doped]
    worst = int(np.argmin(np.where(doped, res, np.inf)))
    worst_res = float(res.flat[worst])
    best_dir_res = float(np.min(np.where(doped, res_best, np.inf)))
    det["min_cells_per_debye_length"] = worst_res
    det["best_direction_cells_per_debye_length"] = best_dir_res
    det["cells_per_debye_length_by_axis"] = {
        name: float(np.min(LD[doped] / wa[doped])) for name, wa in by_axis_h.items()}
    det["worst_LD_m"] = float(LD.flat[worst])
    det["worst_cell_size_m"] = float(h.flat[worst])
    det["worst_doping_per_cm3"] = float(N.flat[worst]) / 1e6
    det["n_under_resolved"] = int(np.count_nonzero(res < 1.0))
    det.update(_locate(worst, N.shape, grid, kind))
    axes = ", ".join(f"{k} {v:.4g}"
                     for k, v in sorted(det["cells_per_debye_length_by_axis"].items()))

    if worst_res >= 1.0:
        level = "ok"
        msg = (f"The mesh resolves the Debye length everywhere: worst case "
               f"{worst_res:.4g} cells per LD at {_loc_str(det)}, where "
               f"LD = {det['worst_LD_m']:.4g} m for N = "
               f"{det['worst_doping_per_cm3']:.4g} cm^-3 and the cell is "
               f"{det['worst_cell_size_m']:.4g} m. LD spans "
               f"{det['LD_min_m']:.4g} to {det['LD_max_m']:.4g} m.")
    elif best_dir_res >= 1.0:
        # Anisotropic mesh: every doped location resolves LD along at least one
        # axis and fails along another.  This function cannot tell which axis
        # the screening runs along, and on a planar device (fine normal to the
        # gate, coarse along the channel) the mesh is correct as it stands, so
        # this can never be more than a warning.
        level = "warn"
        per_axis = det["cells_per_debye_length_by_axis"]
        # The branch is decided cell by cell, so it is possible (if different
        # locations are resolved by different axes) for no single axis to clear
        # the threshold everywhere; name the directions generically then.
        thin = [k for k, v in per_axis.items() if v >= 1.0]
        fat = [k for k, v in per_axis.items() if v < 1.0]
        good = ", ".join(thin) if thin else "its finest direction at each location"
        bad = ", ".join(fat) if fat else "its coarsest direction at each location"
        msg = (f"The mesh is strongly anisotropic about the Debye length: it "
               f"resolves LD along {good} ({best_dir_res:.4g} cells "
               f"per LD in the best direction at the worst location) but not "
               f"along {bad} ({worst_res:.4g} cells per LD). Cells "
               f"per LD by direction: {axes}. That is exactly the right mesh for "
               f"a planar device whose junctions and inversion layer are normal "
               f"to {good}, and the wrong one if any space-charge "
               f"region varies along {bad} --- a lateral source/drain "
               f"junction, for instance. Worst: {_loc_str(det)} has "
               f"h = {det['worst_cell_size_m']:.4g} m against "
               f"LD = {det['worst_LD_m']:.4g} m at N = "
               f"{det['worst_doping_per_cm3']:.4g} cm^-3.")
    else:
        level = "warn" if best_dir_res >= 0.2 else "error"
        sev = ("badly under-resolved" if level == "error"
               else "marginally under-resolved")
        msg = (f"The space-charge region is {sev} in every direction: "
               f"{det['n_under_resolved']} of {det['n_doped_cells']} doped "
               f"locations have cells wider than a Debye length. Worst: "
               f"{_loc_str(det)} has h = {det['worst_cell_size_m']:.4g} m "
               f"against LD = {det['worst_LD_m']:.4g} m at N = "
               f"{det['worst_doping_per_cm3']:.4g} cm^-3, i.e. "
               f"{worst_res:.4g} cells per LD ({best_dir_res:.4g} even in the "
               f"best-resolved direction). Cells per LD by direction: {axes}. "
               f"Scharfetter-Gummel (A7) keeps the "
               f"solve stable at this spacing, so it will converge and return a "
               f"smooth, wrong depletion edge: junction capacitance and "
               f"subthreshold slope are the quantities that suffer. Refine to "
               f"{det['worst_LD_m']:.4g} m or finer there --- "
               f"auto_mesh_1d(extent, features=[junction_positions], "
               f"dx_min={det['worst_LD_m'] / 2:.3g}) is the intended way.")
    return Report("check_debye_length", level, msg, det, "A5, A7")


# ==========================================================================
# Everything at once
# ==========================================================================
def check_all(grid: RectilinearGrid,
              eps_cell: np.ndarray | None = None,
              sigma_cell: np.ndarray | None = None,
              mu_cell: np.ndarray | None = None,
              *,
              t_rise: float | None = None,
              freq: float | None = None,
              dt: float | None = None,
              conductor_mask: np.ndarray | None = None,
              conductor_sigma_min: float = 1.0e3,
              doping: np.ndarray | None = None,
              T: float = T_ROOM,
              eps_semi: np.ndarray | float | None = None,
              eps_r_semi: float = 11.7) -> Report:
    """Run every applicable check and return one combined :class:`Report`.

    Checks whose inputs are missing are skipped rather than guessed at, and the
    combined ``details["skipped"]`` says which and why.  Only
    :func:`check_mesh_quality` always runs, since it needs nothing but the grid.

    Parameters
    ----------
    grid
        The grid.
    eps_cell, sigma_cell, mu_cell
        Absolute per-cell permittivity [F/m], conductivity [S/m] and
        permeability [H/m], each shaped ``(Nx, Ny, Nz)``.  ``mu_cell=None``
        means ``mu0``.
    t_rise
        Signal rise time [s], converted to the knee frequency ``0.35 / t_rise``.
    freq
        Excitation frequency [Hz].  Both the A1 band and the skin depth are
        evaluated at the **higher** of ``freq`` and the knee frequency, because
        both must be governed by the fastest thing in the problem.  Evaluating
        the skin depth at the lower of the two is how a 0.5-cell-per-delta mesh
        gets reported as ``ok``: supplying more information must never make the
        verdict weaker.
    dt
        Planned transient time step [s], for the relaxation check.
    conductor_mask
        Boolean ``(Nx, Ny, Nz)`` conductor mask for the padding check.  If it
        is not given but ``sigma_cell`` is, ``sigma_cell > conductor_sigma_min``
        is used instead --- and if *that* selects no cell, the padding check is
        skipped rather than warned about, since an all-insulating ``sigma_cell``
        is a statement about the problem, not a missing input.  An explicitly
        passed empty mask still warns.
    conductor_sigma_min
        Conductivity above which a cell counts as a conductor for the padding
        check [S/m].  Default 1e3, which sits between poly-Si (1e4) and any
        semiconductor or lossy dielectric.
    doping
        Net doping magnitude [m^-3], node- or cell-shaped.
    T
        Lattice temperature [K].
    eps_semi, eps_r_semi
        Permittivity used for the Debye length: absolute [F/m], or relative
        (default 11.7, silicon).

    Returns
    -------
    Report
        Level is the worst level of any sub-check; ``children`` holds them in
        order.  ``str(report)`` prints the whole thing.
    """
    reports: list[Report] = []
    skipped: dict[str, str] = {}
    details: dict[str, Any] = {}

    # Validate the scalars here rather than letting the first sub-check that
    # happens to use one raise: which checks run depends on which arrays were
    # supplied, so otherwise a bad t_rise could pass silently.
    f_knee = (None if t_rise is None
              else 0.35 / _require_positive(t_rise, "t_rise", "time in seconds"))
    f_ac = (None if freq is None
            else _require_positive(freq, "freq", "frequency in Hz"))

    reports.append(check_mesh_quality(grid))

    if eps_cell is not None and (f_knee is not None or f_ac is not None):
        reports.append(check_quasistatic(grid, eps_cell, mu_cell,
                                         t_rise=t_rise, freq=freq))
    else:
        skipped["check_quasistatic"] = "needs eps_cell and one of t_rise / freq"

    # The fastest thing in the problem governs, exactly as in check_quasistatic.
    candidates = [f for f in (f_knee, f_ac) if f is not None]
    f_skin = max(candidates) if candidates else None
    if sigma_cell is not None and f_skin is not None:
        details["skin_freq_Hz"] = f_skin
        details["skin_freq_source"] = ("t_rise knee" if f_skin == f_knee else "freq")
        reports.append(check_skin_depth(grid, sigma_cell, mu_cell, f_skin))
    else:
        skipped["check_skin_depth"] = "needs sigma_cell and one of freq / t_rise"

    if conductor_mask is not None:
        reports.append(check_padding(grid, conductor_mask))
    elif sigma_cell is not None:
        derived = _as_cell_array(sigma_cell, grid, "sigma_cell") > conductor_sigma_min
        if derived.any():
            reports.append(check_padding(grid, derived))
        else:
            # A purely capacitive problem is a legitimate one, not a missing
            # input: warning here would fail the parallel-plate reference case
            # and would punish the caller for passing sigma_cell = 0 explicitly.
            skipped["check_padding"] = (
                f"no cell has sigma above conductor_sigma_min "
                f"({conductor_sigma_min:g} S/m), so there is no conductor to "
                f"pad around")
    else:
        skipped["check_padding"] = "needs conductor_mask or sigma_cell"

    if eps_cell is not None and sigma_cell is not None and dt is not None:
        reports.append(check_dielectric_relaxation(eps_cell, sigma_cell, dt,
                                                   grid=grid))
    else:
        skipped["check_dielectric_relaxation"] = ("needs eps_cell, sigma_cell "
                                                  "and dt")

    if doping is not None:
        reports.append(check_debye_length(doping, T, grid=grid, eps=eps_semi,
                                          eps_r=eps_r_semi))
    else:
        skipped["check_debye_length"] = "needs doping"

    if skipped:
        details["skipped"] = skipped
    rep = Report.combine(reports, check="check_all", details=details or None)
    if skipped:
        rep.message += (" Skipped " + ", ".join(sorted(skipped))
                        + " for want of inputs.")
    return rep
