"""Electrostatics: Poisson, nonlinear Poisson, and the shared linear-algebra backend.

This module holds three things that the rest of fieldspice builds on.

**1. The electrostatic solve.**  With the potential on nodes and permittivity on
edges, Gauss' law is exactly the nodal analysis of a capacitor mesh::

    G^T M_eps G phi = q_node          [F][V] = [C]

``G`` is the metric-free +-1 incidence matrix from :mod:`fieldspice.operators`
and ``M_eps = edge_mass(grid, eps_edge)`` is a diagonal matrix of *capacitances*
in farads.  There is no separate "finite-difference stencil": the discretisation
*is* a circuit, which is why a meshed region and a SPICE netlist can later be
stamped into one matrix.

Sign convention, derived rather than asserted.  ``(G phi)_e = phi_head -
phi_tail`` is the potential *rise* along the edge, so the electric field
circulation is ``e = -G phi`` and the displacement flux through the dual face
pierced by edge ``e`` is ``psi_e = -(M_eps G phi)_e``.  The outward flux from
the dual box of node ``n`` picks up ``-G[e, n] psi_e`` per incident edge, whose
sum is ``+(G^T M_eps G phi)_n``.  Gauss' theorem then reads
``G^T M_eps G phi = Q_node`` with ``Q_node`` the *free* charge enclosed by the
dual box, in coulombs.  This is the same sign structure as
``G^T M_sigma G phi = I_inject``, as it must be.

**2. The nonlinear (Newton) electrostatic solve.**  ``rho(phi)`` closes the
system, which is what makes it a semiconductor equilibrium solve: with Boltzmann
carriers ``rho = q (p - n + Nd - Na)`` and the solution is the built-in
potential profile of whatever doping you hand it.  The Newton loop uses the
residual-monotone line search specified in ``docs/CONTRACTS.md``; a bare step
clamp is *not* sufficient and is documented there as a measured failure.

**3. ``solve_linear`` / :class:`LinearSystem`.**  Every other solver in the
project routes its sparse solves through here, so the choice of direct
factorisation vs Krylov vs algebraic multigrid is made in exactly one place and
can be audited in exactly one place.  It also owns the singular all-Neumann
case, which is where a "why is my capacitance zero" bug report usually starts.

Assumptions invoked: **A1a** (electroquasistatic, here in its static limit),
**A2** (staircased rectilinear grid), **A3** (linear isotropic materials),
**A10** (box method on the dual grid), **A12** (truncated open boundary), and
for the nonlinear solver **A5** and **A11** when the charge model is a
semiconductor one.
"""

from __future__ import annotations

import inspect
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

from ..boundaries import BoundarySpec
from ..grid import RectilinearGrid
from ..operators import (
    Operators,
    apply_dirichlet,
    cell_to_edge,
    edge_mass,
    node_volume_vector,
)
from ..units import eps0, q as q_elem, thermal_voltage
from .base import ConvergenceError, Result, SolverBase, SolverConfig, Terminal

__all__ = [
    "LinearSystem",
    "solve_linear",
    "PoissonSolver",
    "NonlinearPoissonSolver",
    "EquilibriumCharge",
    "boltzmann_equilibrium_charge",
    "verify_parallel_plate",
    "verify_series_stack",
    "sphere_capacitance_convergence",
    "verify_built_in_potential",
    "verify_built_in_potential_sweep",
    "self_test",
]


# ==========================================================================
# Tunable thresholds.  Module-level *constants* only --- no mutable state.
# ==========================================================================
DIRECT_MAX_UNKNOWNS = 200_000
"""Crossover above which ``linear_solver="auto"`` stops using a direct solve.

A sparse Cholesky of a 3D 7-point Laplacian costs O(n^2) flops and O(n^(4/3))
memory because of fill-in, so on a typical workstation the factorisation of a
~2e5-unknown 3D problem is still a second or two and a few hundred megabytes,
while 1e6 unknowns is minutes and tens of gigabytes.  In 1D and 2D the direct
solve stays cheap far beyond this, but the threshold is deliberately
conservative because guessing wrong towards "direct" is fatal (swap death) and
guessing wrong towards "iterative" merely costs time.
"""

SINGULAR_ROWSUM_TOL = 1e-9
"""Relative tolerance for declaring ``A @ ones == 0``, i.e. a floating system."""

SYMMETRY_TOL = 1e-10
"""Relative tolerance on ``max|A - A^T|`` before CG/Cholesky are refused."""

EXP_CLIP = 400.0
"""Clip on the argument of ``exp`` in carrier statistics.

``exp(400) = 5.2e173`` is finite in float64 while ``exp(710)`` overflows.  The
clip exists so that a wild intermediate Newton iterate produces a large but
*finite* residual that the line search can then reject, instead of an inf that
poisons the whole vector.  Measured value from ``docs/CONTRACTS.md``.
"""

_CG_RTOL_KW = ("rtol" if "rtol" in inspect.signature(spla.cg).parameters
               else "tol")
# SciPy renamed the relative-tolerance keyword of the Krylov solvers in 1.12
# and deprecated the old spelling; probe once at import rather than per solve.


def _pyamg():
    """Import pyamg lazily.  Returns the module or ``None``."""
    try:
        import pyamg  # noqa: PLC0415
    except Exception:                       # pragma: no cover - env dependent
        return None
    return pyamg


def _cholmod():
    """Import scikit-sparse's CHOLMOD binding lazily.  Returns it or ``None``."""
    try:
        from sksparse.cholmod import cholesky  # noqa: PLC0415
    except Exception:                       # pragma: no cover - env dependent
        return None
    return cholesky


# ==========================================================================
# Shared sparse linear-algebra backend
# ==========================================================================
def _abs_max(A: sp.spmatrix) -> float:
    """``max|A_ij|``, zero for an all-zero matrix."""
    return float(np.abs(A.data).max()) if A.nnz else 0.0


def _row_abs_sum(A: sp.csr_matrix) -> np.ndarray:
    """``sum_j |A_ij|`` without materialising ``abs(A)`` as a second matrix."""
    n = A.shape[0]
    counts = np.diff(A.indptr)
    if A.nnz == 0:
        return np.zeros(n)
    rows = np.repeat(np.arange(n), counts)
    return np.bincount(rows, weights=np.abs(A.data), minlength=n)


class LinearSystem:
    """A sparse system prepared once and solved for many right-hand sides.

    This is the single place fieldspice decides *how* to solve ``A x = b``.
    Separating preparation from solution is not a micro-optimisation: a Maxwell
    capacitance extraction with ``nt`` terminals, a transient with a fixed time
    step, and a Newton iteration with a frozen Jacobian all reuse one
    factorisation, and that reuse is the difference between a usable tool and an
    unusable one.

    Strategy, in the order it is decided:

    1. **Structure.**  Rows that are entirely zero (unknowns coupled to nothing,
       e.g. nodes inside a ``sigma = 0`` island of a conductance matrix) are
       pinned to zero, because they carry no information and would otherwise
       make the matrix exactly singular.
    2. **Null space.**  If ``A @ ones`` vanishes there is no Dirichlet
       condition anywhere and the operator has the constant null vector: the
       classic all-Neumann floating system.  One node is pinned to 0 V, a
       :class:`RuntimeWarning` is issued, and the caller is told that only
       potential *differences* are physical.  ``info["pinned"]`` records it.
    3. **Symmetry.**  ``max|A - A^T|`` is compared with ``max|A|``.  CG and
       Cholesky are used only if the matrix passes, because CG applied to a
       non-symmetric matrix does not merely converge slowly, it converges to the
       wrong answer.
    4. **Method.**  ``"direct"`` below :data:`DIRECT_MAX_UNKNOWNS` unknowns
       (CHOLMOD if ``scikit-sparse`` is installed and the matrix is symmetric,
       otherwise SuperLU), AMG-preconditioned CG above if ``pyamg`` is
       installed, plain CG with an incomplete-LU or Jacobi preconditioner
       otherwise.  Non-symmetric systems fall back to GMRES.

    Parameters
    ----------
    A : scipy.sparse matrix
        Square system matrix.  Units are whatever the caller's physics says;
        this class is unit-agnostic.
    config : SolverConfig, optional
        ``linear_solver`` selects the strategy (``"auto"``, ``"direct"``,
        ``"cg"``, ``"amg"``); ``tol`` is the relative residual demanded of the
        iterative methods; ``verbose`` controls chatter.
    pin : bool, default True
        Whether to repair a detected singularity by pinning.  ``False`` makes a
        singular system a hard :class:`ValueError`, which is what a solver that
        believes it has a Dirichlet condition should ask for.
    maxiter : int, optional
        Krylov iteration cap.  Defaults to ``max(500, 10*sqrt(n))``.
    equilibrate : bool, default True
        Apply the symmetric Jacobi scaling ``D^-1/2 A D^-1/2``.  Leave it on:
        without it the identity rows written by
        :func:`fieldspice.operators.apply_dirichlet` sit at 1.0 while the
        physical entries of a capacitance matrix sit at 1e-17, and the resulting
        1e17 condition number destroys every Krylov method.  The scaling is a
        congruence transform, so symmetry and definiteness are preserved
        exactly, and reported residuals are converted back to the original
        scaling before you see them.

    Attributes
    ----------
    info : dict
        Diagnostics: ``method``, ``n``, ``nnz``, ``symmetric``, ``pinned``,
        ``singular``, ``setup_time`` [s], and after each :meth:`solve` also
        ``iterations``, ``residual`` (true relative 2-norm) and ``solve_time``
        [s].

    Examples
    --------
    >>> import numpy as np, scipy.sparse as sp
    >>> A = sp.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(5, 5)).tocsr()
    >>> ls = LinearSystem(A)
    >>> x = ls.solve(np.ones(5))
    >>> bool(np.allclose(A @ x, np.ones(5)))
    True
    """

    def __init__(self, A: sp.spmatrix,
                 config: SolverConfig | None = None,
                 *,
                 pin: bool = True,
                 maxiter: int | None = None,
                 equilibrate: bool = True) -> None:
        t0 = time.perf_counter()
        self.cfg = config or SolverConfig()
        A = sp.csr_matrix(A)
        if A.shape[0] != A.shape[1]:
            raise ValueError(f"matrix must be square, got {A.shape}")
        n = A.shape[0]
        if n == 0:
            raise ValueError("cannot solve an empty system")
        self.n = n
        self._complex = np.iscomplexobj(A.data)

        scale = _abs_max(A)
        if scale == 0.0:
            raise ValueError("system matrix is identically zero")

        # -- structural singularities -------------------------------------
        # Tested against exact zero, not against a fraction of max|A|.  A
        # Dirichlet-eliminated capacitance matrix carries identity rows of 1.0
        # alongside physical rows of ~1e-17 F, so any threshold relative to the
        # global maximum declares the entire physical block empty.  That was a
        # real bug here; the row's own magnitude is the only sane yardstick.
        row_abs = _row_abs_sum(A)
        empty = np.flatnonzero(row_abs == 0.0)
        if empty.size:
            warnings.warn(
                f"{empty.size} unknown(s) are coupled to nothing (empty rows); "
                "they are pinned to zero.  In a conductance matrix this means a "
                "region of zero conductivity that no terminal reaches.",
                RuntimeWarning, stacklevel=2)

        # -- constant null space (all-Neumann / floating system) ----------
        ones = np.ones(n, dtype=A.dtype)
        rowsum = np.abs(A @ ones)
        keep = row_abs > 0.0
        self.singular = bool(keep.any() and np.all(
            rowsum[keep] <= SINGULAR_ROWSUM_TOL * row_abs[keep]))
        pin_idx = list(empty)
        self._gauge: int | None = None
        if self.singular:
            if not pin:
                raise ValueError(
                    "system is singular: every row sums to zero, so there is no "
                    "Dirichlet node anywhere and the potential is only defined "
                    "up to a constant.  Drive a terminal, ground a wall, or "
                    "pass pin=True.")
            diag = np.abs(A.diagonal())
            diag[~keep] = -1.0
            self._gauge = int(np.argmax(diag))
            pin_idx.append(self._gauge)
            warnings.warn(
                "singular all-Neumann system: no Dirichlet node was found, so "
                f"node {self._gauge} is pinned to 0 to fix the gauge.  Only "
                "potential DIFFERENCES in the result are meaningful; the "
                "absolute level is arbitrary.",
                RuntimeWarning, stacklevel=2)
        self.pinned = np.array(sorted(set(pin_idx)), dtype=np.intp)

        if self.pinned.size:
            A = _pin_matrix(A, self.pinned)

        # -- symmetry ------------------------------------------------------
        AT = A.getH() if self._complex else A.T
        diff = (A - AT).tocoo()
        self.symmetric = bool(diff.nnz == 0
                              or np.abs(diff.data).max() <= SYMMETRY_TOL * scale)
        dreal = A.diagonal().real
        self.positive_diagonal = bool(np.all(dreal > 0))
        self._spd_candidate = self.symmetric and self.positive_diagonal

        # -- symmetric Jacobi equilibration --------------------------------
        # A' = D^-1/2 A D^-1/2 with D = diag(A).  Exact (it is a congruence, so
        # symmetry and definiteness survive) and essential rather than cosmetic:
        # apply_dirichlet writes identity rows of 1.0 into a matrix whose
        # physical entries are ~1e-17 F, which inflates the condition number by
        # 1e17 and makes CG useless while leaving a direct solve untouched.
        # Equilibrating puts every diagonal at 1 and the damage disappears.
        self._equilibrated = bool(equilibrate and self.positive_diagonal)
        if self._equilibrated:
            self._dhalf = np.sqrt(dreal)
            self._dinv = 1.0 / self._dhalf
            S = sp.diags(self._dinv)
            A = sp.csr_matrix(S @ A @ S)
        else:
            self._dhalf = np.ones(n)
            self._dinv = np.ones(n)
        self.A = A

        # -- method selection ---------------------------------------------
        want = self.cfg.linear_solver
        if want not in ("auto", "direct", "cg", "amg"):
            raise ValueError(
                f"unknown linear_solver {want!r}; expected one of "
                "'auto', 'direct', 'cg', 'amg'")
        notes: list[str] = []
        if want == "auto":
            if n <= DIRECT_MAX_UNKNOWNS:
                want = "direct"
            elif _pyamg() is not None:
                want = "amg"
            else:
                want = "cg"
                notes.append("pyamg not installed; using preconditioned CG")
        if want in ("cg", "amg") and not self._spd_candidate:
            notes.append(
                "matrix is not symmetric with a positive diagonal, so CG is "
                "unsafe; falling back to "
                + ("a direct factorisation" if n <= 4 * DIRECT_MAX_UNKNOWNS
                   else "GMRES"))
            want = "direct" if n <= 4 * DIRECT_MAX_UNKNOWNS else "gmres"
        if want == "amg" and _pyamg() is None:
            notes.append("pyamg not installed; using preconditioned CG")
            want = "cg"

        self._maxiter = int(maxiter) if maxiter is not None else None
        self._lu = None
        self._chol = None
        self._M = None
        self.method = want
        self._setup(notes)

        self.info: dict[str, Any] = {
            "method": self.method,
            "n": n,
            "nnz": int(A.nnz),
            "symmetric": self.symmetric,
            "singular": self.singular,
            "pinned": (int(self._gauge) if self._gauge is not None else None),
            "n_pinned": int(self.pinned.size),
            "setup_time": time.perf_counter() - t0,
            "notes": tuple(notes),
        }
        if self.cfg.verbose >= 1 and notes:
            for msg in notes:
                print(f"[linear] {msg}", flush=True)

    # -- setup -------------------------------------------------------------
    def _setup(self, notes: list[str]) -> None:
        A = self.A
        if self.method == "direct":
            chol = _cholmod() if self._spd_candidate and not self._complex else None
            if chol is not None:
                try:
                    self._chol = chol(A.tocsc())
                    self.method = "cholmod"
                    return
                except Exception as exc:    # pragma: no cover - env dependent
                    notes.append(f"CHOLMOD failed ({exc}); using SuperLU")
            # MMD on A+A^T is the right ordering for a structurally symmetric
            # matrix and roughly halves the fill of COLAMD here.
            perm = "MMD_AT_PLUS_A" if self.symmetric else "COLAMD"
            self._lu = spla.splu(A.tocsc(), permc_spec=perm)
            self.method = "splu"
            return

        if self.method == "amg":
            pyamg = _pyamg()
            ml = pyamg.smoothed_aggregation_solver(A.tocsr(), max_coarse=500)
            self._M = ml.aspreconditioner(cycle="V")
            self._amg_levels = len(ml.levels)
            return

        # Krylov without AMG.  An incomplete LU is an incomplete Cholesky in all
        # but name for a symmetric matrix and is worth an order of magnitude in
        # iteration count; it can run out of memory on large 3D problems, hence
        # the guarded fallback to Jacobi.
        if self.method in ("cg", "gmres"):
            try:
                ilu = spla.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=10.0)
                self._M = spla.LinearOperator(A.shape, ilu.solve, dtype=A.dtype)
                self._prec = "ilu"
            except Exception as exc:
                d = A.diagonal()
                d = np.where(np.abs(d) > 0, d, 1.0)
                self._M = spla.LinearOperator(A.shape, lambda v: v / d,
                                              dtype=A.dtype)
                self._prec = "jacobi"
                notes.append(f"incomplete-LU preconditioner unavailable ({exc}); "
                             "using Jacobi")

    # -- solving -----------------------------------------------------------
    def solve(self, b: np.ndarray, x0: np.ndarray | None = None) -> np.ndarray:
        """Solve ``A x = b`` for one right-hand side.

        Parameters
        ----------
        b : np.ndarray
            Right-hand side, length ``n``.
        x0 : np.ndarray, optional
            Initial guess for the iterative methods; ignored by the direct ones.

        Returns
        -------
        np.ndarray
            The solution.  Pinned unknowns are exactly zero.
        """
        t0 = time.perf_counter()
        b = np.asarray(b)
        if b.ndim != 1 or b.size != self.n:
            raise ValueError(f"right-hand side must have shape ({self.n},), "
                             f"got {b.shape}")
        b = b.astype(self.A.dtype, copy=True)

        if self.singular:
            # A consistent singular system needs 1^T b == 0.  If it does not
            # hold, the pinned node silently absorbs the imbalance, so say so.
            imbalance = abs(complex(b.sum()))
            gross = float(np.abs(b).sum())
            if gross > 0 and imbalance > 1e-8 * gross:
                warnings.warn(
                    f"right-hand side of a floating (all-Neumann) system sums to "
                    f"{imbalance:.6g} rather than 0, so it is not in the range "
                    "of the operator.  The imbalance is absorbed at the pinned "
                    f"node {self._gauge}; physically this means the structure is "
                    "not charge/current neutral and needs a return electrode.",
                    RuntimeWarning, stacklevel=2)
        b[self.pinned] = 0.0
        nb = float(np.linalg.norm(b))
        bs = b * self._dinv if self._equilibrated else b

        iters = 0
        if self.method == "cholmod":                # pragma: no cover - env dep
            xs = np.asarray(self._chol(bs)).ravel()
        elif self.method == "splu":
            xs = self._lu.solve(bs)
        else:
            n = self.n
            maxiter = self._maxiter
            if maxiter is None:
                maxiter = max(500, int(10 * np.sqrt(n)))
            counter = {"n": 0}

            def _cb(_):
                counter["n"] += 1

            kw = {_CG_RTOL_KW: self.cfg.tol, "atol": 0.0, "maxiter": maxiter,
                  "M": self._M, "callback": _cb}
            if x0 is not None:
                kw["x0"] = np.asarray(x0, dtype=self.A.dtype) * self._dhalf
            if self.method == "gmres":
                xs, flag = spla.gmres(self.A, bs, restart=min(200, n), **kw)
            else:
                xs, flag = spla.cg(self.A, bs, **kw)
            iters = counter["n"]
            if flag != 0:
                warnings.warn(
                    f"{self.method} returned flag {flag} after {iters} "
                    "iterations", RuntimeWarning, stacklevel=2)

        x = xs * self._dinv if self._equilibrated else xs
        # Always report the TRUE residual, never the Krylov solver's estimate:
        # a preconditioned residual can be orders of magnitude optimistic.  The
        # equilibrated residual is mapped back with D^1/2 so the number quoted
        # refers to the system the caller actually handed in.
        res_vec = (bs - self.A @ xs) * self._dhalf
        res = float(np.linalg.norm(res_vec) / nb) if nb > 0 else 0.0
        self.info.update(iterations=iters, residual=res,
                         solve_time=time.perf_counter() - t0)
        if self.method not in ("splu", "cholmod"):
            hard = max(1e-6, 100.0 * self.cfg.tol)
            if res > hard:
                raise ConvergenceError(
                    f"{self.method} failed: relative residual {res:.3e} after "
                    f"{iters} iterations (requested {self.cfg.tol:.1e})",
                    history=[res])
            if res > max(self.cfg.tol * 10.0, 1e-12):
                warnings.warn(
                    f"{self.method} stopped at relative residual {res:.3e}, "
                    f"looser than the requested {self.cfg.tol:.1e}",
                    RuntimeWarning, stacklevel=2)
        self._log(x)
        return x

    def _log(self, x: np.ndarray) -> None:
        if self.cfg.verbose >= 2:
            i = self.info
            print(f"[linear] {i['method']} n={i['n']} nnz={i['nnz']} "
                  f"iters={i.get('iterations', 0)} "
                  f"res={i.get('residual', 0.0):.2e} "
                  f"t={i.get('solve_time', 0.0):.3f}s", flush=True)


def _pin_matrix(A: sp.csr_matrix, idx: np.ndarray) -> sp.csr_matrix:
    """Replace rows and columns ``idx`` of ``A`` by identity rows.

    Symmetric elimination, so an SPD matrix stays SPD.  Because the pinned value
    is zero there is no corresponding right-hand-side correction.
    """
    n = A.shape[0]
    mask = np.ones(n, dtype=A.dtype)
    mask[idx] = 0
    Dm = sp.diags(mask)
    keep = sp.diags((mask == 0).astype(A.dtype))
    return sp.csr_matrix(Dm @ A @ Dm + keep)


def solve_linear(A: sp.spmatrix, b: np.ndarray,
                 cfg: SolverConfig | None = None,
                 *,
                 x0: np.ndarray | None = None,
                 info: dict[str, Any] | None = None,
                 pin: bool = True) -> np.ndarray:
    """Solve one sparse system, choosing the strategy automatically.

    Thin wrapper around :class:`LinearSystem` for the one-shot case.  When you
    have several right-hand sides for the same matrix, build a
    :class:`LinearSystem` instead and call :meth:`LinearSystem.solve` repeatedly;
    that is the difference between one factorisation and ``nt`` of them.

    Parameters
    ----------
    A : scipy.sparse matrix
        Square system matrix.
    b : np.ndarray
        Right-hand side.
    cfg : SolverConfig, optional
        Strategy and tolerances.  ``None`` uses the defaults.
    x0 : np.ndarray, optional
        Initial guess for iterative methods.
    info : dict, optional
        If given, updated in place with the diagnostics described in
        :attr:`LinearSystem.info`.
    pin : bool, default True
        Repair a singular all-Neumann system by pinning one unknown (with a
        warning) instead of raising.

    Returns
    -------
    np.ndarray
        Solution vector.

    Notes
    -----
    A singular system is *detected*, not guessed: ``G^T M G`` has exactly zero
    row sums, so ``max|A @ 1|`` compared against ``max|A|`` is an exact test for
    "no Dirichlet condition anywhere".
    """
    ls = LinearSystem(A, cfg, pin=pin)
    x = ls.solve(b, x0=x0)
    if info is not None:
        info.update(ls.info)
    return x


# ==========================================================================
# Equipotential / periodic constraints
# ==========================================================================
@dataclass(frozen=True)
class _Reduction:
    """Node-merging map ``phi = P x``, used for floating and periodic nodes.

    ``P`` has one 1 per row, so ``P^T A P`` sums the rows and columns of merged
    nodes.  Summing rows is exactly the physical statement "the total charge
    flowing into the electrode is the sum of the charges into its nodes", and
    summing columns is "all of its nodes share one potential", so the reduced
    operator stays symmetric positive semidefinite and no penalty parameter or
    Lagrange multiplier is needed.
    """
    P: sp.csr_matrix | None
    group: np.ndarray            # (n_nodes,) reduced index of each node
    n_reduced: int

    @classmethod
    def identity(cls, n: int) -> "_Reduction":
        return cls(None, np.arange(n, dtype=np.intp), n)

    def restrict(self, A: sp.spmatrix) -> sp.csr_matrix:
        return sp.csr_matrix(A) if self.P is None else sp.csr_matrix(
            self.P.T @ A @ self.P)

    def rhs(self, b: np.ndarray) -> np.ndarray:
        return b if self.P is None else self.P.T @ b

    def expand(self, x: np.ndarray) -> np.ndarray:
        return x if self.P is None else self.P @ x

    def index(self, nodes: np.ndarray) -> np.ndarray:
        return self.group[nodes]


def _build_reduction(n: int, groups: Sequence[np.ndarray]) -> _Reduction:
    """Merge each node set in ``groups`` into a single unknown."""
    pairs_a: list[np.ndarray] = []
    pairs_b: list[np.ndarray] = []
    for g in groups:
        g = np.asarray(g, dtype=np.intp).ravel()
        if g.size < 2:
            continue
        pairs_a.append(np.full(g.size - 1, g[0]))
        pairs_b.append(g[1:])
    if not pairs_a:
        return _Reduction.identity(n)
    a = np.concatenate(pairs_a)
    b = np.concatenate(pairs_b)
    adj = sp.coo_matrix((np.ones(a.size), (a, b)), shape=(n, n))
    ncomp, label = connected_components(adj, directed=False)
    if ncomp == n:
        return _Reduction.identity(n)
    label = label.astype(np.intp)
    P = sp.coo_matrix((np.ones(n), (np.arange(n), label)),
                      shape=(n, ncomp)).tocsr()
    return _Reduction(P, label, int(ncomp))


def _unique_last(idx: np.ndarray, val: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate ``idx`` keeping the value of the LAST occurrence."""
    if idx.size == 0:
        return idx.astype(np.intp), val.astype(float)
    uniq, first_in_rev = np.unique(idx[::-1], return_index=True)
    return uniq.astype(np.intp), val[::-1][first_in_rev]


# ==========================================================================
# Input coercion
# ==========================================================================
def _as_cell_array(a: np.ndarray, grid: RectilinearGrid, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.shape == ():
        arr = np.full(grid.shape_cells, float(arr))
    if arr.shape != grid.shape_cells:
        raise ValueError(f"{name} must have shape {grid.shape_cells}, "
                         f"got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_node_vector(a: np.ndarray, grid: RectilinearGrid, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=float)
    if arr.shape == ():
        return np.full(grid.n_nodes, float(arr))
    if arr.shape == grid.shape_nodes:
        return arr.ravel()
    if arr.shape == (grid.n_nodes,):
        return arr
    raise ValueError(
        f"{name} must have shape {grid.shape_nodes} or ({grid.n_nodes},), got "
        f"{arr.shape}.  Remember that a collapsed direction still has two node "
        "planes, so a '1D' grid with N cells has 4*(N+1) nodes, not N+1.")


def _check_terminals(terminals: Sequence[Terminal] | None,
                     grid: RectilinearGrid) -> tuple[Terminal, ...]:
    if terminals is None:
        return ()
    if isinstance(terminals, Terminal):
        raise ValueError("pass a sequence of Terminal, not a bare Terminal")
    out = tuple(terminals)
    seen: dict[int, str] = {}
    names: set[str] = set()
    for t in out:
        if not isinstance(t, Terminal):
            raise ValueError(f"expected Terminal instances, got {type(t).__name__}")
        if t.name in names:
            raise ValueError(f"duplicate terminal name {t.name!r}")
        names.add(t.name)
        if t.nodes.size == 0:
            raise ValueError(f"terminal {t.name!r} has no nodes")
        if t.nodes.min() < 0 or t.nodes.max() >= grid.n_nodes:
            raise ValueError(
                f"terminal {t.name!r} references node index outside "
                f"[0, {grid.n_nodes})")
        for nd in t.nodes:
            other = seen.get(int(nd))
            if other is not None:
                raise ValueError(
                    f"terminals {other!r} and {t.name!r} share node {int(nd)}; "
                    "electrodes must be disjoint")
            seen[int(nd)] = t.name
    return out


# ==========================================================================
# Linear electrostatics
# ==========================================================================
class PoissonSolver(SolverBase):
    r"""Electrostatic potential from ``div(eps grad phi) = -rho``.

    Discretely this is nodal analysis of a capacitor mesh::

        G^T M_eps G phi = q_node

    with ``G`` the exact +-1 gradient incidence matrix, ``M_eps`` the diagonal
    matrix of edge capacitances ``eps*A_dual/L`` [F], ``phi`` the node potential
    [V] and ``q_node`` the *free* charge in each dual box [C].  The matrix is
    symmetric positive semidefinite; it becomes definite as soon as one node is
    fixed by a voltage-driven terminal or a Dirichlet wall.

    Parameters
    ----------
    grid : RectilinearGrid
        The mesh.
    eps_cell : np.ndarray
        **Absolute** permittivity per cell [F/m], shape ``grid.shape_cells``.
        Use ``MaterialMap.eps()``.  Passing a *relative* permittivity is
        detected and raises, because the resulting capacitance would be wrong by
        eleven orders of magnitude and otherwise looks plausible.
    config : SolverConfig, optional
        Linear-solver strategy and tolerances.
    operators : Operators, optional
        Shared, cached incidence matrices.  Pass the same instance to several
        solvers on one grid to build ``G`` once.
    eps_mode : str, default "parallel"
        Cell-to-edge averaging rule (see
        :func:`fieldspice.operators.cell_to_edge`).  ``"parallel"`` (dual-area
        weighted arithmetic mean) is correct for the four cell quadrants that
        tile an edge's dual area, because they act as capacitors in parallel.
        ``"harmonic"`` is right only when the edge crosses a thin series barrier.

    Attributes
    ----------
    L : scipy.sparse.csr_matrix
        The nodal capacitance matrix ``G^T M_eps G`` [F], built on first use.

    Notes
    -----
    Assumptions **A1a** (no induction: ``curl E = 0``), **A2** (staircased
    rectilinear geometry), **A3** (linear isotropic non-dispersive materials),
    **A10** (box method on the dual grid), **A12** (the domain wall truncates an
    open problem; homogeneous Neumann confines the field and *under*-estimates
    fringing unless the box is padded to >= 3x the largest feature).

    Examples
    --------
    A parallel-plate capacitor, plates spanning the whole cross-section:

    >>> import numpy as np
    >>> from fieldspice.grid import RectilinearGrid
    >>> from fieldspice.solvers.base import Terminal
    >>> from fieldspice.units import eps0
    >>> g = RectilinearGrid.uniform([(0, 2e-6), (0, 5e-6), (0, 4e-6)], [8, 4, 4])
    >>> ps = PoissonSolver(g, np.full(g.shape_cells, 3.9 * eps0))
    >>> nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    >>> lo = Terminal("lo", nid[0].ravel(), voltage=0.0)
    >>> hi = Terminal("hi", nid[-1].ravel(), voltage=1.0)
    >>> C = ps.capacitance_matrix([lo, hi])
    >>> float(abs(C[1, 1] / (3.9 * eps0 * 20e-12 / 2e-6) - 1.0)) < 1e-12
    True
    """

    name = "poisson"
    assumptions: tuple[str, ...] = ("A1a", "A2", "A3", "A10", "A12")

    #: Boundary kinds this solver understands.
    SUPPORTED_BC = frozenset({"dirichlet", "neumann", "symmetry", "periodic"})

    def __init__(self, grid: RectilinearGrid, eps_cell: np.ndarray,
                 config: SolverConfig | None = None,
                 operators: Operators | None = None,
                 *, eps_mode: str = "parallel") -> None:
        super().__init__(grid, config, operators)
        eps_cell = _as_cell_array(eps_cell, grid, "eps_cell")
        if np.any(eps_cell <= 0.0):
            raise ValueError("eps_cell must be strictly positive [F/m]")
        if float(eps_cell.min()) >= 1e-6:
            raise ValueError(
                f"eps_cell looks like a RELATIVE permittivity (min "
                f"{eps_cell.min():.4g}); fieldspice is strict SI and wants the "
                "absolute value in F/m.  Multiply by fieldspice.units.eps0, or "
                "use MaterialMap.eps().")
        self.eps_cell = eps_cell
        self.eps_mode = eps_mode
        self._eps_edge = cell_to_edge(grid, eps_cell, mode=eps_mode)
        self._L: sp.csr_matrix | None = None
        self._vnode = node_volume_vector(grid)
        self.last_capacitance_report: dict[str, Any] = {}

    # -- assembled operators ----------------------------------------------
    @property
    def L(self) -> sp.csr_matrix:
        """Nodal capacitance matrix ``G^T M_eps G`` [F] (built lazily, cached)."""
        if self._L is None:
            G = self.ops.G
            self._L = sp.csr_matrix(G.T @ edge_mass(self.grid, self._eps_edge) @ G)
        return self._L

    @property
    def eps_edge(self) -> np.ndarray:
        """Permittivity averaged onto edges [F/m], length ``grid.n_edges``."""
        return self._eps_edge

    @property
    def node_volumes(self) -> np.ndarray:
        """Dual (box) volume of each node [m^3], length ``grid.n_nodes``."""
        return self._vnode

    # -- physics helpers ---------------------------------------------------
    def energy(self, phi: np.ndarray) -> float:
        """Stored electrostatic energy [J]: ``0.5 phi^T L phi``.

        Identical to ``0.5 * integral(eps |E|^2 dV)``, because
        ``phi^T G^T M_eps G phi = sum_e C_e (dphi_e)^2`` and
        ``C_e (dphi_e)^2 = eps A L E^2``.
        """
        phi = _as_node_vector(phi, self.grid, "phi")
        return 0.5 * float(phi @ (self.L @ phi))

    def node_charge(self, phi: np.ndarray) -> np.ndarray:
        """Total charge in each node's dual box [C]: ``L phi``.

        For an interior node with no free charge this is zero to roundoff; on an
        electrode it is the charge the driving source had to supply.
        """
        phi = _as_node_vector(phi, self.grid, "phi")
        return self.L @ phi

    def field_edges(self, phi: np.ndarray) -> np.ndarray:
        """Edge voltage ``e = -G phi`` [V], length ``grid.n_edges``.

        Divide by ``grid.edge_lengths()`` to get the electric field [V/m]; the
        stored quantity is the *integrated* circulation, per the formulation.
        """
        phi = _as_node_vector(phi, self.grid, "phi")
        return -(self.ops.G @ phi)

    # -- system assembly ---------------------------------------------------
    def _prepare_bc(self, bc: BoundarySpec | None) -> BoundarySpec:
        bc = BoundarySpec.all_neumann() if bc is None else bc
        if not isinstance(bc, BoundarySpec):
            raise ValueError(f"bc must be a BoundarySpec, got {type(bc).__name__}")
        bc.validate(self.grid)
        bc.require(self.SUPPORTED_BC, context="PoissonSolver")
        return bc

    def _periodic_groups(self, bc: BoundarySpec) -> list[np.ndarray]:
        groups: list[np.ndarray] = []
        for axis, (lo, hi) in bc.periodic_pairs(self.grid).items():
            wall = {"x": "xlo", "y": "ylo", "z": "zlo"}[axis]
            if float(getattr(bc.get(wall), "offset", 0.0)) != 0.0:
                raise ValueError(
                    f"axis {axis}: PoissonSolver supports periodic boundaries "
                    "only with offset = 0.  A non-zero offset is an affine "
                    "constraint whose bookkeeping at corner nodes (shared by two "
                    "periodic axes) is ambiguous; impose the drop with a driven "
                    "terminal instead.")
            groups.extend(np.array([a, b], dtype=np.intp)
                          for a, b in zip(lo, hi))
        return groups

    def _external_charge(self, bc: BoundarySpec, rho_node: np.ndarray | None,
                         t: float) -> np.ndarray:
        """Free charge in each dual box [C] plus the driven-Neumann load."""
        b = np.zeros(self.grid.n_nodes)
        if rho_node is not None:
            b += _as_node_vector(rho_node, self.grid, "rho_node") * self._vnode
        b += bc.neumann_load(self.grid, t)
        return b

    # -- the solve ---------------------------------------------------------
    def solve(self, terminals: Sequence[Terminal] | None = None,
              bc: BoundarySpec | None = None,
              rho_node: np.ndarray | None = None,
              *, t: float = 0.0) -> Result:
        """Solve for the potential.

        Parameters
        ----------
        terminals : sequence of Terminal, optional
            Electrodes.  Their node sets must be disjoint.  How each kind is
            treated:

            * ``voltage`` set --- Dirichlet: every node of the electrode is held
              at that potential [V], and the reported charge is what the source
              had to supply.
            * ``current`` set --- read as a prescribed **total charge** [C] on a
              floating equipotential electrode.  There is no current in an
              electrostatic problem, so charge is the conjugate drive; the field
              is reused rather than inventing a parallel attribute.
            * neither --- a floating equipotential island carrying zero net
              charge, which is the correct model for an unconnected metal plate.
        bc : BoundarySpec, optional
            Wall conditions.  Default is homogeneous Neumann on all six walls:
            no normal flux leaves the box.  **A12**: on an open problem that
            artificially confines the field and under-estimates fringing
            capacitance unless the domain is padded to >= 3x the largest feature.
            ``Periodic`` is supported with zero offset; ``Absorbing`` is not (a
            quasi-static absorbing boundary needs a BEM coupling, which is not
            implemented).
        rho_node : np.ndarray, optional
            Free charge **density** [C/m^3] at nodes, shape ``grid.shape_nodes``
            or ``(grid.n_nodes,)``.  Multiplied internally by the dual box
            volume, so the box method conserves charge exactly.
        t : float, default 0.0
            Time [s] at which to evaluate time-dependent terminal and boundary
            values.  An electrostatic solve is instantaneous; this only selects
            the drive level.

        Returns
        -------
        Result
            ``fields["phi"]`` has shape ``(1,) + grid.shape_nodes`` [V];
            ``fields["e"]`` has shape ``(1, n_edges)`` and holds the edge
            voltage ``-G phi`` [V].  ``terminals[name]`` carries ``"v"`` [V] and
            ``"q"`` [C].  There is deliberately no ``"i"``: an electrostatic
            solve carries no current, and reporting a zero would invite it to be
            mistaken for a computed result.  ``scalars`` holds ``"energy"`` [J]
            and ``"total_charge"`` [C].

        Raises
        ------
        ValueError
            On mismatched array shapes, overlapping electrodes, unsupported
            boundary kinds, or a relative permittivity passed as absolute.

        Notes
        -----
        With no Dirichlet condition anywhere (all-Neumann walls and no
        voltage-driven terminal) the operator keeps its constant null vector.
        That is detected in :class:`LinearSystem`, one node is pinned, a warning
        is issued, and only potential *differences* in the result are physical.
        """
        self._start()
        grid = self.grid
        terms = _check_terminals(terminals, grid)
        bc = self._prepare_bc(bc)

        L = self.L
        b_free = self._external_charge(bc, rho_node, t)

        # Floating and charge-driven electrodes become single merged unknowns;
        # so do periodic wall pairs.  Both are the same algebraic operation.
        merge: list[np.ndarray] = [tt.nodes for tt in terms
                                   if tt.driven != "voltage"]
        merge.extend(self._periodic_groups(bc))
        red = _build_reduction(grid.n_nodes, merge)

        b_ext = b_free.copy()
        for tt in terms:
            if tt.driven == "current":
                # Whole prescribed charge on one node of the group; P^T sums it.
                b_ext[tt.nodes[0]] += float(tt.value_at(t))

        fixed, vals = self._dirichlet_sets(bc, terms, t)
        fixed_r = red.index(fixed)
        A_r = red.restrict(L)
        b_r = red.rhs(b_ext)
        A_bc, b_bc = apply_dirichlet(A_r, b_r, fixed_r, vals)

        info: dict[str, Any] = {}
        x_r = solve_linear(A_bc, b_bc, self.cfg, info=info)
        phi = red.expand(x_r)

        res = self._make_result(phi, terms, b_free, info)
        res.meta["n_unknowns"] = int(red.n_reduced)
        res.meta["n_dirichlet"] = int(fixed.size)
        return self._finish(res)

    def _dirichlet_sets(self, bc: BoundarySpec, terms: Sequence[Terminal],
                        t: float) -> tuple[np.ndarray, np.ndarray]:
        """Fixed node indices and values [V]; a terminal beats a wall."""
        idx_parts: list[np.ndarray] = []
        val_parts: list[np.ndarray] = []
        w_idx, w_val = bc.dirichlet_nodes(self.grid, t)
        if w_idx.size:
            idx_parts.append(w_idx)
            val_parts.append(w_val)
        for tt in terms:
            if tt.driven == "voltage":
                idx_parts.append(tt.nodes)
                val_parts.append(np.full(tt.nodes.size, float(tt.value_at(t))))
        if not idx_parts:
            return np.empty(0, dtype=np.intp), np.empty(0, dtype=float)
        return _unique_last(np.concatenate(idx_parts), np.concatenate(val_parts))

    def _make_result(self, phi: np.ndarray, terms: Sequence[Terminal],
                     b_free: np.ndarray, info: dict[str, Any]) -> Result:
        grid = self.grid
        qnode = self.L @ phi - b_free
        res = Result(grid=grid)
        res.fields["phi"] = phi.reshape((1,) + grid.shape_nodes)
        res.fields["e"] = (-(self.ops.G @ phi)).reshape(1, -1)
        for tt in terms:
            res.terminals[tt.name] = {
                "v": np.array([float(phi[tt.nodes[0]])]),
                "q": np.array([float(qnode[tt.nodes].sum())]),
            }
        res.scalars["energy"] = np.array([self.energy(phi)])
        res.scalars["total_charge"] = np.array([float(qnode.sum())])
        res.meta["linear"] = info
        return res

    # -- capacitance extraction -------------------------------------------
    def capacitance_matrix(self, terminals: Sequence[Terminal],
                           bc: BoundarySpec | None = None,
                           *, report: dict[str, Any] | None = None,
                           symmetrise: bool = True) -> np.ndarray:
        r"""Maxwell capacitance matrix [F] by unit excitation.

        Terminal ``j`` is driven to 1 V, every other terminal is grounded, and
        the charge induced on terminal ``i`` is integrated::

            C[i, j] = Q_i  when  V_j = 1 V,  V_k = 0 for k != j

        Any ``Dirichlet`` wall in ``bc`` is treated as an extra grounded
        conductor at 0 V regardless of the value it carries, because a
        capacitance matrix is defined by short-circuit excitations.  Its charge
        is *not* part of the matrix, which is where the rows-sum-to-zero
        property goes if you use a grounded shield --- see the report below.

        Maxwell versus SPICE
        --------------------
        These are two different matrices and confusing them is the most common
        error in capacitance extraction.

        * The **Maxwell** (short-circuit) matrix returned here relates electrode
          charges to electrode potentials, ``Q = C V``.  Its diagonal is
          positive, its off-diagonals are **negative** (raising one electrode
          pushes charge *off* its neighbours), and for a closed system --- one
          where every field line starts and ends on a listed electrode --- each
          row sums to zero.
        * The **SPICE / mutual** matrix is what you put in a netlist: a
          two-terminal capacitor ``C_ij = -C_maxwell[i, j] > 0`` between each
          pair, plus ``C_ii = sum_j C_maxwell[i, j]`` from electrode ``i`` to
          ground.  Get it from :meth:`spice_capacitance_matrix`.

        Parameters
        ----------
        terminals : sequence of Terminal
            Electrodes.  Their ``voltage`` / ``current`` settings are ignored:
            extraction defines its own excitations.  Node sets must be disjoint.
        bc : BoundarySpec, optional
            Default all-Neumann.  See **A12**: with Neumann walls the extracted
            capacitance is a *lower* bound on the open-domain value until the
            box is padded.
        report : dict, optional
            Filled in place with quality metrics (see below).  The same dict is
            also stored on ``self.last_capacitance_report``.
        symmetrise : bool, default True
            Replace ``C`` by ``(C + C^T)/2``.  The true matrix is symmetric by
            reciprocity, so the measured asymmetry is a free *a posteriori*
            error estimate for the whole discretisation-plus-linear-solve chain;
            it is reported rather than hidden.

        Returns
        -------
        np.ndarray
            ``(nt, nt)`` Maxwell capacitance matrix [F].

        Other Parameters
        ----------------
        The report dictionary contains:

        ``asymmetry_abs`` [F]
            ``max|C - C^T|`` before symmetrisation.
        ``asymmetry_rel``
            The same divided by ``max|diag(C)|``.  A good solve gives 1e-12 or
            below; 1e-3 means the linear solves are under-converged or the mesh
            is badly graded.
        ``row_sum`` [F], ``row_sum_rel``
            ``C @ ones``.  Near zero for a closed system; equal to minus the
            charge coupled to a grounded shield or lost through the wall
            otherwise.
        ``max_offdiag`` [F]
            Largest off-diagonal entry.  Must be <= 0 on physical grounds; a
            positive value indicates an unconverged solve.
        ``shield_charge`` [C]
            Charge on the ``bc`` Dirichlet nodes for each excitation, if any.
        ``linear``
            Diagnostics from :class:`LinearSystem`, including the factorisation
            method and the residual of the last back-solve.

        Notes
        -----
        The matrix is factorised **once**: every excitation shares the same set
        of Dirichlet nodes and differs only in the right-hand side, so the cost
        is one factorisation plus ``nt`` back-solves, not ``nt`` factorisations.
        """
        self._start()
        grid = self.grid
        terms = _check_terminals(terminals, grid)
        if not terms:
            raise ValueError("capacitance extraction needs at least one terminal")
        bc = self._prepare_bc(bc)

        L = self.L
        red = _build_reduction(grid.n_nodes, self._periodic_groups(bc))
        A_r = red.restrict(L)

        wall_idx, _ = bc.dirichlet_nodes(grid, 0.0)
        term_nodes = np.concatenate([tt.nodes for tt in terms])
        # Grounded shield first, electrodes second: an electrode node that also
        # sits on a Dirichlet wall is driven, not shorted to the shield.
        fixed = np.concatenate([wall_idx, term_nodes]).astype(np.intp)
        fixed, _ = _unique_last(fixed, np.zeros(fixed.size))
        fixed_r = red.index(fixed)
        shield = np.setdiff1d(wall_idx, term_nodes, assume_unique=False)

        A_bc, _ = apply_dirichlet(A_r, np.zeros(A_r.shape[0]), fixed_r,
                                  np.zeros(fixed_r.size))
        ls = LinearSystem(A_bc, self.cfg)

        nt = len(terms)
        C = np.zeros((nt, nt))
        q_shield = np.zeros(nt)
        for j, tj in enumerate(terms):
            x0_r = np.zeros(red.n_reduced)
            x0_r[red.index(tj.nodes)] = 1.0
            # Mirror apply_dirichlet's elimination exactly, but only on the RHS.
            b_r = -(A_r @ x0_r)
            b_r[fixed_r] = x0_r[fixed_r]
            phi = red.expand(ls.solve(b_r))
            qnode = L @ phi
            for i, ti in enumerate(terms):
                C[i, j] = float(qnode[ti.nodes].sum())
            if shield.size:
                q_shield[j] = float(qnode[shield].sum())

        rep: dict[str, Any] = {}
        scale = float(np.abs(np.diag(C)).max()) or 1.0
        asym = float(np.abs(C - C.T).max())
        rep["asymmetry_abs"] = asym
        rep["asymmetry_rel"] = asym / scale
        if symmetrise:
            C = 0.5 * (C + C.T)
        rows = C.sum(axis=1)
        rep["row_sum"] = rows
        rep["row_sum_rel"] = float(np.abs(rows).max() / scale)
        off = C - np.diag(np.diag(C))
        rep["max_offdiag"] = float(off.max()) if nt > 1 else 0.0
        rep["shield_charge"] = q_shield
        rep["linear"] = dict(ls.info)
        rep["terminals"] = tuple(tt.name for tt in terms)
        if rep["max_offdiag"] > 1e-9 * scale:
            warnings.warn(
                f"Maxwell capacitance matrix has a positive off-diagonal entry "
                f"({rep['max_offdiag']:.4g} F): the solve is not converged, or "
                "two electrodes are not electrically distinct.",
                RuntimeWarning, stacklevel=2)
        self.last_capacitance_report = rep
        if report is not None:
            report.update(rep)
        return C

    def spice_capacitance_matrix(self, terminals: Sequence[Terminal],
                                 bc: BoundarySpec | None = None,
                                 *, report: dict[str, Any] | None = None
                                 ) -> np.ndarray:
        """Mutual (SPICE) capacitance matrix [F].

        Built from the Maxwell matrix ``Cm`` by::

            C_spice[i, j] = -Cm[i, j]          i != j   (coupling capacitor)
            C_spice[i, i] =  sum_j Cm[i, j]             (capacitor to ground)

        Every entry of a physical result is non-negative, and the netlist you
        would write is one two-terminal capacitor per off-diagonal plus one
        capacitor to ground per diagonal.  If the diagonal comes out slightly
        negative, the domain is not closed: field lines are leaving through the
        wall or landing on a grounded shield that is not in the terminal list.

        Parameters
        ----------
        terminals, bc, report
            As for :meth:`capacitance_matrix`.

        Returns
        -------
        np.ndarray
            ``(nt, nt)`` mutual capacitance matrix [F].
        """
        Cm = self.capacitance_matrix(terminals, bc, report=report)
        Cs = -Cm.copy()
        np.fill_diagonal(Cs, Cm.sum(axis=1))
        return Cs


# ==========================================================================
# Nonlinear electrostatics (semiconductor equilibrium)
# ==========================================================================
class NonlinearPoissonSolver(PoissonSolver):
    r"""Poisson with a field-dependent charge ``rho(phi)``, solved by Newton.

    Solves::

        F(phi) = L phi - [rho(phi) + rho_ext] * V_node - load = 0
        J(phi) = L - diag(drho/dphi * V_node)

    With Boltzmann carriers, ``drho/dphi = -q (n + p)/Vt`` is strictly negative,
    so ``J`` is ``L`` plus a positive diagonal: symmetric positive definite for
    every iterate, and CG or Cholesky apply throughout.  That is the whole
    reason the equilibrium semiconductor problem is easy while the full
    drift-diffusion problem is not.

    The damping is the two-stage scheme measured in ``docs/CONTRACTS.md``::

        lam = min(1, step_limit * v_scale / max|dphi|)     # keep exp() in range
        while ||F(phi + lam dphi)|| >= (1 - armijo*lam) ||F(phi)||:
            lam *= 0.5                                     # Armijo backtracking

    The second stage is not optional.  A bare clamp with no line search has
    nothing forcing the residual down, and on a pn junction at
    ``ni = 9.65e15 m^-3`` it enters a stable limit cycle (residual alternating
    forever) instead of converging.  The residual norm is measured over the
    **free** nodes only, since Dirichlet rows are identically zero and would
    otherwise dilute it.

    Parameters
    ----------
    grid : RectilinearGrid
        The mesh.  It must resolve the Debye length of the most heavily doped
        region, or the answer is meaningless however well Newton converges.
    eps_cell : np.ndarray
        Absolute permittivity per cell [F/m].
    rho : callable
        ``rho(phi) -> np.ndarray`` giving the charge **density** [C/m^3] at each
        node, for a flat node-potential vector [V] of length ``grid.n_nodes``.
    drho_dphi : callable
        ``drho_dphi(phi) -> np.ndarray`` [C/(V m^3)], the analytic derivative.
        A finite-difference substitute is not accepted: it destroys the
        quadratic convergence that makes the 1e-9 built-in-potential tolerance
        reachable, and it is the derivative that keeps the Jacobian SPD.
    config : SolverConfig, optional
        ``newton_tol`` is the update tolerance in units of ``v_scale``;
        ``max_newton`` caps the iteration count; ``damping`` scales the first
        trial step.
    operators : Operators, optional
        Shared incidence matrices.
    v_scale : float, optional
        Potential scale over which ``rho`` varies by roughly one e-fold [V].
        For a semiconductor this is the thermal voltage ``kT/q``; it defaults to
        ``thermal_voltage(300)`` = 25.852 mV.
    step_limit : float, default 5.0
        First-stage clamp in units of ``v_scale``.  The measured value.
    armijo : float, default 1e-4
        Sufficient-decrease constant of the line search.
    eps_mode : str, default "parallel"
        Cell-to-edge averaging, as for :class:`PoissonSolver`.

    Notes
    -----
    Assumptions **A1a**, **A2**, **A3**, **A10**, **A12** as for the linear
    solver, plus **A5** (drift-diffusion physics, here in its zero-current
    equilibrium limit) and **A11** (complete dopant ionisation) whenever the
    charge model handed in is a semiconductor one.
    """

    name = "nonlinear_poisson"
    assumptions: tuple[str, ...] = ("A1a", "A2", "A3", "A5", "A10", "A11", "A12")

    def __init__(self, grid: RectilinearGrid, eps_cell: np.ndarray,
                 rho: Callable[[np.ndarray], np.ndarray],
                 drho_dphi: Callable[[np.ndarray], np.ndarray],
                 config: SolverConfig | None = None,
                 operators: Operators | None = None,
                 *, v_scale: float | None = None,
                 step_limit: float = 5.0,
                 armijo: float = 1e-4,
                 eps_mode: str = "parallel") -> None:
        super().__init__(grid, eps_cell, config, operators, eps_mode=eps_mode)
        if not callable(rho) or not callable(drho_dphi):
            raise ValueError("rho and drho_dphi must be callables of phi")
        if step_limit <= 0:
            raise ValueError("step_limit must be positive")
        if not 0.0 < armijo < 0.5:
            raise ValueError("armijo must lie in (0, 0.5)")
        self.rho = rho
        self.drho_dphi = drho_dphi
        self.v_scale = float(thermal_voltage() if v_scale is None else v_scale)
        if self.v_scale <= 0:
            raise ValueError("v_scale must be positive [V]")
        self.step_limit = float(step_limit)
        self.armijo = float(armijo)

    def solve(self, terminals: Sequence[Terminal] | None = None,
              bc: BoundarySpec | None = None,
              rho_node: np.ndarray | None = None,
              phi0: np.ndarray | None = None,
              *, t: float = 0.0) -> Result:
        """Newton solve for the self-consistent potential.

        Parameters
        ----------
        terminals : sequence of Terminal, optional
            As for :meth:`PoissonSolver.solve`.  For an equilibrium
            semiconductor solve the usual setup is one ohmic contact held at its
            charge-neutral bulk potential and homogeneous Neumann everywhere
            else.
        bc : BoundarySpec, optional
            Default all-Neumann.  With no Dirichlet node anywhere the gauge is
            pinned and warned about; the built-in potential, being a
            *difference*, is unaffected.
        rho_node : np.ndarray, optional
            Additional fixed charge density [C/m^3] at nodes, added to
            ``rho(phi)``.  Use it for trapped or interface charge that does not
            respond to the potential.
        phi0 : np.ndarray, optional
            Initial guess [V].  Default is zero, which is a poor guess for a
            semiconductor; pass the local charge-neutral solution
            (:func:`boltzmann_equilibrium_charge` returns one) and Newton
            typically needs about ten iterations instead of dozens.

        Returns
        -------
        Result
            As :meth:`PoissonSolver.solve`, with ``scalars["residual"]`` holding
            the residual history [C] and ``meta["newton"]`` the iteration count,
            damping factors used and final update size.

        Raises
        ------
        ConvergenceError
            If ``max_newton`` iterations are exhausted, or the line search has
            to shrink the step below 1e-12 (which means the Newton direction is
            not a descent direction --- almost always a wrong ``drho_dphi``).
        """
        self._start()
        grid = self.grid
        terms = _check_terminals(terminals, grid)
        bc = self._prepare_bc(bc)

        L = self.L
        vnode = self._vnode
        b_free = self._external_charge(bc, rho_node, t)

        merge: list[np.ndarray] = [tt.nodes for tt in terms
                                   if tt.driven != "voltage"]
        merge.extend(self._periodic_groups(bc))
        red = _build_reduction(grid.n_nodes, merge)

        b_ext = b_free.copy()
        for tt in terms:
            if tt.driven == "current":
                b_ext[tt.nodes[0]] += float(tt.value_at(t))

        fixed, vals = self._dirichlet_sets(bc, terms, t)
        fixed_r = red.index(fixed)

        phi = (np.zeros(grid.n_nodes) if phi0 is None
               else _as_node_vector(phi0, grid, "phi0").copy())
        phi[fixed] = vals

        b_ext_r = red.rhs(b_ext)

        def residual(p: np.ndarray) -> np.ndarray:
            """Reduced residual [C] with Dirichlet rows zeroed."""
            f = red.rhs(L @ p - self.rho(p) * vnode) - b_ext_r
            f[fixed_r] = 0.0
            return f

        F = residual(phi)
        hist = [float(np.linalg.norm(F))]
        lams: list[float] = []
        cfg = self.cfg
        tol_update = cfg.newton_tol * self.v_scale
        info: dict[str, Any] = {}
        converged = False
        last_update = np.inf

        for it in range(int(cfg.max_newton)):
            if hist[-1] <= 1e-14 * max(hist[0], 1e-300):
                converged = True
                break
            J = L - sp.diags(self.drho_dphi(phi) * vnode)
            J_r = red.restrict(J)
            A_bc, rhs = apply_dirichlet(J_r, -F, fixed_r,
                                        np.zeros(fixed_r.size))
            dx = solve_linear(A_bc, rhs, cfg, info=info)
            dphi = red.expand(dx)

            big = float(np.abs(dphi).max())
            lam = float(cfg.damping)
            if big > 0.0:
                lam = min(lam, self.step_limit * self.v_scale / big)
            f0 = hist[-1]
            while True:
                Fn = residual(phi + lam * dphi)
                fn = float(np.linalg.norm(Fn))
                if np.isfinite(fn) and fn < (1.0 - self.armijo * lam) * f0:
                    break
                lam *= 0.5
                if lam < 1e-12:
                    raise ConvergenceError(
                        f"Newton line search collapsed at iteration {it}: no "
                        "step size reduces the residual, which means the Newton "
                        "direction is not a descent direction.  Check that "
                        "drho_dphi is the exact derivative of rho.",
                        history=hist, last_state=phi)
            phi = phi + lam * dphi
            F, _ = Fn, None
            hist.append(fn)
            lams.append(lam)
            last_update = lam * big
            self._log(2, f"newton {it + 1}: |F| = {fn:.6e}, lam = {lam:.4f}, "
                         f"max|dphi| = {last_update:.4e} V")
            if last_update <= tol_update:
                converged = True
                break

        if not converged:
            raise ConvergenceError(
                f"Newton did not converge in {cfg.max_newton} iterations; "
                f"last |F| = {hist[-1]:.4e} C, last update {last_update:.4e} V "
                f"(tolerance {tol_update:.4e} V)",
                history=hist, last_state=phi)

        res = self._make_result(phi, terms, b_free + self.rho(phi) * vnode, info)
        res.scalars["residual"] = np.asarray(hist)
        res.meta["newton"] = {
            "iterations": len(lams),
            "residual_history": tuple(hist),
            "damping": tuple(lams),
            "final_update": float(last_update),
            "v_scale": self.v_scale,
        }
        res.meta["n_unknowns"] = int(red.n_reduced)
        res.meta["n_dirichlet"] = int(fixed.size)
        return self._finish(res)


# ==========================================================================
# Boltzmann equilibrium charge model
# ==========================================================================
class EquilibriumCharge(NamedTuple):
    """Charge model for an equilibrium (zero-current) semiconductor solve.

    Attributes
    ----------
    rho : callable
        ``rho(phi) -> [C/m^3]`` at nodes.
    drho_dphi : callable
        ``drho_dphi(phi) -> [C/(V m^3)]`` at nodes; strictly negative.
    phi_guess : np.ndarray
        Local charge-neutral potential [V], the standard Newton initial guess.
    v_scale : float
        Thermal voltage kT/q [V].
    """
    rho: Callable[[np.ndarray], np.ndarray]
    drho_dphi: Callable[[np.ndarray], np.ndarray]
    phi_guess: np.ndarray
    v_scale: float


def boltzmann_equilibrium_charge(doping_node: np.ndarray,
                                 ni: float,
                                 T: float = 300.0) -> EquilibriumCharge:
    """Boltzmann carrier charge and its derivative, for equilibrium solves.

    With the potential referenced to the intrinsic level::

        n = ni exp(+phi/Vt)          p = ni exp(-phi/Vt)
        rho = q (p - n + Nd - Na)
        drho/dphi = -q (n + p)/Vt

    The zero-current (equilibrium) condition is what lets the carrier densities
    be written as explicit functions of the potential; out of equilibrium they
    become independent unknowns and you need the drift-diffusion solver.

    Parameters
    ----------
    doping_node : np.ndarray
        Net doping ``Nd - Na`` [m^-3] at each node, flat or node-shaped.  Use
        ``fieldspice.units.per_cm3`` to convert from cm^-3.
    ni : float
        Intrinsic carrier density [m^-3].  For silicon at 300 K use
        ``fieldspice.reference.NI_SI_300 = 9.65e15``.
    T : float, default 300.0
        Lattice temperature [K].  Isothermal (**A6**).

    Returns
    -------
    EquilibriumCharge
        Callables plus the charge-neutral initial guess.

    Notes
    -----
    Assumes Boltzmann statistics and complete ionisation (**A11**), so it is
    quantitatively wrong above roughly 1e19 cm^-3 where degeneracy sets in.  The
    exponent is clipped at :data:`EXP_CLIP` so that a wild trial step during the
    line search produces a large but finite residual, which backtracking can
    then reject.

    Examples
    --------
    >>> import numpy as np
    >>> doping = np.array([-1e23, 1e23])
    >>> ec = boltzmann_equilibrium_charge(doping, 9.65e15)
    >>> bool(np.all(np.abs(ec.rho(ec.phi_guess)) < 1e-9))
    True
    """
    dop = np.asarray(doping_node, dtype=float).ravel()
    if not np.all(np.isfinite(dop)):
        raise ValueError("doping_node contains non-finite values")
    if ni <= 0:
        raise ValueError("ni must be positive [m^-3]")
    Vt = thermal_voltage(T)

    def _np(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = np.clip(np.asarray(phi, dtype=float) / Vt, -EXP_CLIP, EXP_CLIP)
        return ni * np.exp(z), ni * np.exp(-z)

    def rho(phi: np.ndarray) -> np.ndarray:
        n, p = _np(phi)
        return q_elem * (p - n + dop)

    def drho(phi: np.ndarray) -> np.ndarray:
        n, p = _np(phi)
        return -q_elem * (n + p) / Vt

    # Exact root of rho = 0: -2 ni sinh(phi/Vt) + Nd - Na = 0.
    phi_guess = Vt * np.arcsinh(dop / (2.0 * ni))
    return EquilibriumCharge(rho, drho, phi_guess, Vt)


# ==========================================================================
# Verification against the analytic oracle
# ==========================================================================
def _plate_terminals(grid: RectilinearGrid, axis: int = 0
                     ) -> tuple[Terminal, Terminal]:
    """Two full-cross-section electrodes on the low and high faces of ``axis``."""
    nid = np.arange(grid.n_nodes).reshape(grid.shape_nodes)
    lo = np.take(nid, 0, axis=axis).ravel()
    hi = np.take(nid, -1, axis=axis).ravel()
    return (Terminal("lo", lo, voltage=0.0), Terminal("hi", hi, voltage=1.0))


def verify_parallel_plate(gap: float = 2e-6, wy: float = 5e-6, wz: float = 4e-6,
                          eps_r: float = 3.9,
                          n: tuple[int, int, int] = (24, 10, 8)) -> dict[str, Any]:
    """Parallel-plate capacitance against ``eps A / d``.

    The plates span the entire cross-section and the side walls are Neumann, so
    there is no fringing to approximate and the discrete answer is the analytic
    one to roundoff.  That makes this a machine-precision test of the assembly,
    the mass matrix and the charge integration, not a 1% physics check.

    Parameters
    ----------
    gap : float
        Plate separation [m].
    wy, wz : float
        Plate dimensions [m].
    eps_r : float
        Relative permittivity of the fill.
    n : tuple of int
        Cells per direction.

    Returns
    -------
    dict
        ``C`` [F], ``C_ref`` [F], ``rel_err``, ``report`` (the capacitance
        quality metrics), ``uniform_dev`` (largest deviation of the potential
        within a plane of constant x [V], which must vanish).
    """
    from ..reference import parallel_plate_capacitance

    grid = RectilinearGrid.uniform([(0, gap), (0, wy), (0, wz)], n)
    eps = np.full(grid.shape_cells, eps_r * eps0)
    ps = PoissonSolver(grid, eps)
    lo, hi = _plate_terminals(grid)
    rep: dict[str, Any] = {}
    C = ps.capacitance_matrix([lo, hi], report=rep)
    C_ref = parallel_plate_capacitance(wy * wz, gap, eps_r)

    res = ps.solve([lo, hi])
    phi = res.fields["phi"][0]
    uniform_dev = float(np.abs(phi - phi[:, :1, :1]).max())
    return {
        "C": float(C[1, 1]), "C_ref": float(C_ref),
        "rel_err": abs(float(C[1, 1]) / C_ref - 1.0),
        "report": rep, "uniform_dev": uniform_dev,
        "n_nodes": grid.n_nodes,
    }


def verify_series_stack(d1: float = 1.2e-6, d2: float = 0.8e-6,
                        eps_r1: float = 3.9, eps_r2: float = 7.5,
                        wy: float = 5e-6, wz: float = 4e-6,
                        n1: int = 12, n2: int = 8) -> dict[str, Any]:
    """Two dielectrics in series between plates, against the series formula.

    ``1/C = d1/(eps1 A) + d2/(eps2 A)``.  Exact on this mesh because the
    interface lies on a node plane, so no cell straddles the discontinuity;
    the interface is placed there deliberately, and moving it half a cell is
    what turns a 1e-15 error into a 1e-2 one.

    Returns
    -------
    dict
        ``C`` [F], ``C_ref`` [F], ``rel_err``, ``v_interface`` (potential at the
        dielectric interface [V]) and ``v_interface_ref`` from the capacitive
        divider ``(d1/eps1)/(d1/eps1 + d2/eps2)``.
    """
    x = np.concatenate([np.linspace(0.0, d1, n1 + 1),
                        np.linspace(d1, d1 + d2, n2 + 1)[1:]])
    grid = RectilinearGrid(x, np.linspace(0, wy, 5), np.linspace(0, wz, 4))
    eps = np.empty(grid.shape_cells)
    eps[:n1] = eps_r1 * eps0
    eps[n1:] = eps_r2 * eps0
    ps = PoissonSolver(grid, eps)
    lo, hi = _plate_terminals(grid)
    C = ps.capacitance_matrix([lo, hi])
    A = wy * wz
    C_ref = 1.0 / (d1 / (eps_r1 * eps0 * A) + d2 / (eps_r2 * eps0 * A))

    res = ps.solve([lo, hi])
    phi = res.fields["phi"][0]
    v_if = float(phi[n1, 0, 0])
    v_ref = (d1 / eps_r1) / (d1 / eps_r1 + d2 / eps_r2)
    return {
        "C": float(C[1, 1]), "C_ref": float(C_ref),
        "rel_err": abs(float(C[1, 1]) / C_ref - 1.0),
        "v_interface": v_if, "v_interface_ref": v_ref,
        "v_rel_err": abs(v_if / v_ref - 1.0),
    }


def sphere_capacitance_convergence(radius: float = 1e-6,
                                   pads: Sequence[float] = (2.0, 3.0, 5.0, 9.0),
                                   cells_per_radius: int = 14,
                                   eps_r: float = 1.0,
                                   growth: float = 1.25) -> dict[str, Any]:
    """Isolated-sphere capacitance versus box padding: the **A12** measurement.

    An isolated conductor has no return electrode, so the domain wall *is* the
    second electrode and the extracted capacitance depends on where you put it.
    This function measures that dependence, which is the honest way to quantify
    the open-boundary truncation error of a quasi-static solver that has no
    absorbing boundary (**A12**).

    The model uses octant symmetry: the sphere sits at the origin of a
    ``[0, R]^3`` box, the three inner walls are Neumann (symmetry planes) and
    the three outer walls are grounded Dirichlet.  The full-sphere capacitance
    is eight times the octant value.

    **A Neumann outer wall gives exactly zero, not a small answer.**  With no
    Dirichlet wall and a single 1 V electrode, the unique solution of
    ``div(eps grad phi) = 0`` is ``phi = 1`` everywhere, so no charge is induced
    at all.  An isolated-object capacitance therefore *requires* a grounded far
    boundary, and the result converges to ``4 pi eps a`` from **above** as the
    shield recedes --- a concentric spherical shield of radius ``R`` gives
    ``C = 4 pi eps a / (1 - a/R)`` exactly.  The reported ``deshielded`` column
    removes that leading term; what is left is staircase error (**A2**) plus the
    difference between a cubic box and a spherical shield.

    Parameters
    ----------
    radius : float
        Sphere radius [m].
    pads : sequence of float
        Box half-widths in units of the radius.
    cells_per_radius : int
        Mesh resolution across the radius.  Held fixed across ``pads`` so the
        staircase error stays roughly constant and the trend isolates the
        boundary error.
    eps_r : float
        Relative permittivity of the surrounding medium.
    growth : float
        Mesh grading ratio away from the sphere surface.

    Returns
    -------
    dict
        ``pads``, ``C`` [F] per pad, ``C_ref`` [F], ``rel_err`` per pad,
        ``shell_pred`` (``4 pi eps a/(1 - a/R)``), ``deshielded``
        (``C * (1 - a/R)``), ``n_nodes`` per pad.
    """
    from ..grid import auto_mesh_1d
    from ..boundaries import Dirichlet, Neumann
    from ..reference import sphere_capacitance

    C_ref = sphere_capacitance(radius, eps_r)
    out: dict[str, Any] = {"pads": tuple(float(p) for p in pads),
                           "C_ref": float(C_ref)}
    Cs, errs, shells, desh, nn = [], [], [], [], []
    for pad in pads:
        R = float(pad) * radius
        ax = auto_mesh_1d((0.0, R), features=(radius,),
                          dx_min=radius / cells_per_radius,
                          dx_max=max(radius, R / 6.0), growth=growth)
        grid = RectilinearGrid(ax, ax, ax)
        eps = np.full(grid.shape_cells, eps_r * eps0)
        ps = PoissonSolver(grid, eps)
        X, Y, Z = grid.node_coords()
        inside = (X ** 2 + Y ** 2 + Z ** 2) <= radius ** 2 * (1.0 + 1e-12)
        nodes = np.flatnonzero(inside.ravel())
        term = Terminal("sphere", nodes, voltage=1.0)
        bc = BoundarySpec(xhi=Dirichlet(0.0), yhi=Dirichlet(0.0),
                          zhi=Dirichlet(0.0),
                          xlo=Neumann(), ylo=Neumann(), zlo=Neumann())
        C_oct = float(ps.capacitance_matrix([term], bc)[0, 0])
        C_full = 8.0 * C_oct
        Cs.append(C_full)
        errs.append(C_full / C_ref - 1.0)
        shells.append(4.0 * np.pi * eps_r * eps0 * radius / (1.0 - 1.0 / pad))
        desh.append(C_full * (1.0 - 1.0 / pad))
        nn.append(grid.n_nodes)
    out.update(C=tuple(Cs), rel_err=tuple(errs), shell_pred=tuple(shells),
               deshielded=tuple(desh), n_nodes=tuple(nn))
    return out


def verify_built_in_potential(Na: float = 1e23, Nd: float = 1e23,
                              ni: float | None = None, T: float = 300.0,
                              eps_r: float = 11.7,
                              debye_multiple: float = 45.0,
                              cells_per_debye: float = 10.0,
                              verbose: int = 0) -> dict[str, Any]:
    """Equilibrium pn junction, checked against ``Vt ln(Na Nd / ni^2)``.

    Setup: a 1D silicon bar with an abrupt junction at ``x = 0``.  The **p**
    contact at ``x = -Lp`` is an ohmic Dirichlet node held at the exact
    charge-neutral bulk potential ``-Vt asinh(Na/(2 ni))``; every other wall is
    homogeneous Neumann.  The built-in potential is then read off as
    ``phi(x = +Ln) - phi(x = -Lp)``, so only *one* end is prescribed and the
    other is a genuine output of the solve.  Fixing both ends would make the
    test circular.

    Parameters
    ----------
    Na, Nd : float
        Acceptor and donor concentrations [m^-3].
    ni : float, optional
        Intrinsic density [m^-3].  Default: the measured silicon value from
        :mod:`fieldspice.reference`.
    T : float
        Temperature [K].
    eps_r : float
        Relative permittivity (11.7 for silicon).
    debye_multiple : float
        Length of each quasi-neutral region in Debye lengths.  The bulk
        potential approaches its asymptote like ``exp(-x/LD)``, so 45 leaves
        headroom of many orders of magnitude below the 1e-9 tolerance.
    cells_per_debye : float
        Mesh resolution at the junction.
    verbose : int
        Passed to :class:`SolverConfig`.

    Returns
    -------
    dict
        ``Vbi``, ``Vbi_ref`` [V], ``rel_err``, ``Vbi_exact`` [V],
        ``rel_err_exact``, ``iterations``, ``residual_history`` [C],
        ``net_charge`` [C] (the integral of the space charge, which must vanish
        for a neutral structure), ``n_nodes``, ``uniform_dev`` [V].

    Notes
    -----
    Two references are reported, and the difference between them matters.
    ``Vbi_ref`` is :func:`fieldspice.reference.built_in_potential`, the textbook
    ``Vt ln(Na Nd / ni^2)``.  ``Vbi_exact`` is the exact charge-neutral bulk
    potential ``Vt [asinh(Na/2ni) + asinh(Nd/2ni)]``, which is what the discrete
    equations actually solve.  They differ by ``Vt (ni^2/Na^2 + ni^2/Nd^2)``,
    which is 1e-15 relative at 1e17 cm^-3 but grows to 7e-12 at 1e15 cm^-3.  A
    solve that matches ``Vbi_exact`` to 1e-15 while sitting 7e-12 from
    ``Vbi_ref`` is not 7e-12 wrong; the *formula* is, and chasing that gap would
    mean fitting the discretisation to a truncated series.
    """
    from ..boundaries import Dirichlet
    from ..grid import auto_mesh_1d
    from ..reference import NI_SI_300, built_in_potential, debye_length

    ni = float(NI_SI_300 if ni is None else ni)
    Vt = thermal_voltage(T)
    LD_p = debye_length(Na, T, eps_r)
    LD_n = debye_length(Nd, T, eps_r)
    Lp, Ln = debye_multiple * LD_p, debye_multiple * LD_n
    dx_min = min(LD_p, LD_n) / cells_per_debye
    x = auto_mesh_1d((-Lp, Ln), features=(0.0,), dx_min=dx_min,
                     dx_max=max(Lp, Ln) / 15.0, growth=1.25)
    grid = RectilinearGrid(x)

    X, _, _ = grid.node_coords()
    doping = np.where(X > 0.0, Nd, -Na).ravel()
    ec = boltzmann_equilibrium_charge(doping, ni, T)
    eps = np.full(grid.shape_cells, eps_r * eps0)

    cfg = SolverConfig(verbose=verbose, newton_tol=1e-10)
    ps = NonlinearPoissonSolver(grid, eps, ec.rho, ec.drho_dphi, cfg,
                                v_scale=ec.v_scale)
    phi_p = -Vt * float(np.arcsinh(Na / (2.0 * ni)))
    bc = BoundarySpec(xlo=Dirichlet(phi_p))
    res = ps.solve(bc=bc, phi0=ec.phi_guess)

    phi = res.fields["phi"][0]
    Vbi = float(phi[-1, 0, 0] - phi[0, 0, 0])
    Vbi_ref = built_in_potential(Na, Nd, ni, T)
    Vbi_exact = Vt * float(np.arcsinh(Na / (2.0 * ni))
                           + np.arcsinh(Nd / (2.0 * ni)))
    rho_final = ec.rho(phi.ravel())
    net_charge = float(np.sum(rho_final * ps.node_volumes))
    uniform_dev = float(np.abs(phi - phi[:, :1, :1]).max())
    return {
        "Vbi": Vbi, "Vbi_ref": float(Vbi_ref),
        "rel_err": abs(Vbi / Vbi_ref - 1.0),
        "Vbi_exact": Vbi_exact,
        "rel_err_exact": abs(Vbi / Vbi_exact - 1.0),
        "iterations": res.meta["newton"]["iterations"],
        "residual_history": res.meta["newton"]["residual_history"],
        "damping": res.meta["newton"]["damping"],
        "net_charge": net_charge,
        "n_nodes": grid.n_nodes,
        "uniform_dev": uniform_dev,
    }


def verify_built_in_potential_sweep(verbose: bool = False) -> dict[str, Any]:
    """The doping / ``ni`` sweep prescribed in ``docs/CONTRACTS.md``.

    ``ni`` in ``{9.65e15, 1e16, 1.45e16}`` m^-3 crossed with ``(Na, Nd)`` in
    ``{(1e17,1e17), (1e15,1e18), (1e19,1e16), (5e16,3e17)}`` cm^-3.  The
    contract records 9-11 Newton iterations and 1e-12 relative or better in
    every case *with the line search*, and a permanent limit cycle without it.

    Returns
    -------
    dict
        ``cases`` (one dict per case), ``max_rel_err``, ``max_iterations``.
    """
    nis = (9.65e15, 1.0e16, 1.45e16)
    dopings = ((1e17, 1e17), (1e15, 1e18), (1e19, 1e16), (5e16, 3e17))
    cases = []
    for ni in nis:
        for Na_cm3, Nd_cm3 in dopings:
            r = verify_built_in_potential(Na_cm3 * 1e6, Nd_cm3 * 1e6, ni=ni)
            row = {"ni": ni, "Na_cm3": Na_cm3, "Nd_cm3": Nd_cm3,
                   "Vbi": r["Vbi"], "Vbi_ref": r["Vbi_ref"],
                   "rel_err": r["rel_err"], "rel_err_exact": r["rel_err_exact"],
                   "iterations": r["iterations"], "n_nodes": r["n_nodes"]}
            cases.append(row)
            if verbose:
                print(f"  ni={ni:.3g} Na={Na_cm3:.3g} Nd={Nd_cm3:.3g} cm^-3 : "
                      f"Vbi={row['Vbi']:.9f} V  rel_err={row['rel_err']:.2e}  "
                      f"(vs exact {row['rel_err_exact']:.2e})  "
                      f"iters={row['iterations']}", flush=True)
    return {
        "cases": cases,
        "max_rel_err": max(c["rel_err"] for c in cases),
        "max_rel_err_exact": max(c["rel_err_exact"] for c in cases),
        "max_iterations": max(c["iterations"] for c in cases),
    }


def self_test(verbose: bool = True) -> dict[str, Any]:
    """Run every analytic check in this module and return the measured errors.

    Nothing here is asserted; the numbers are returned and printed so that a
    discrepancy is visible rather than swallowed.  The pass/fail thresholds live
    in the test suite, where they belong.
    """
    out: dict[str, Any] = {}
    pp = verify_parallel_plate()
    out["parallel_plate"] = pp
    ss = verify_series_stack()
    out["series_stack"] = ss
    pn = verify_built_in_potential()
    out["pn_junction"] = pn
    sph = sphere_capacitance_convergence()
    out["sphere"] = sph
    if verbose:
        print(f"parallel plate : C = {pp['C']:.6e} F, ref {pp['C_ref']:.6e} F, "
              f"rel err {pp['rel_err']:.2e}")
        print(f"                 capacitance asymmetry {pp['report']['asymmetry_rel']:.2e}, "
              f"row sum {pp['report']['row_sum_rel']:.2e}")
        print(f"series stack   : C = {ss['C']:.6e} F, ref {ss['C_ref']:.6e} F, "
              f"rel err {ss['rel_err']:.2e}")
        print(f"pn junction    : Vbi = {pn['Vbi']:.9f} V, ref "
              f"{pn['Vbi_ref']:.9f} V, rel err {pn['rel_err']:.2e}, "
              f"{pn['iterations']} Newton iterations")
        print(f"sphere (A12)   : ref {sph['C_ref']:.6e} F")
        for pad, C, e, d in zip(sph["pads"], sph["C"], sph["rel_err"],
                                sph["deshielded"]):
            print(f"   pad {pad:5.1f} a : C = {C:.6e} F  rel err {e:+.4f}  "
                  f"de-shielded {d:.6e} F")
    return out


if __name__ == "__main__":       # pragma: no cover - manual entry point
    self_test()
