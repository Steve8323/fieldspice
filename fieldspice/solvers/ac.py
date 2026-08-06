"""Frequency-domain quasi-static solver.

Replace ``d/dt`` with ``j*omega`` and the transient becomes **one complex
linear solve per frequency**.  For extracting an impedance, an admittance or a
set of S-parameters that is 100-1000x cheaper than running a transient and
Fourier-transforming it, and it carries no time-discretisation error at all.

    mode="eqs"    G^T (M_sigma + j w M_eps) G phi = i

A second, less obvious advantage: in the frequency domain the electric and
magnetic subproblems can be solved **monolithically**, so the Darwin coupling
has no splitting error here, unlike the staggered time-domain scheme (A8).

The systems are **complex symmetric**, not Hermitian, so conjugate gradients is
invalid; this module uses a sparse LU throughout.  Using CG on a complex
symmetric system is a classic error that appears to work (it converges) while
returning the wrong answer.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from ..grid import RectilinearGrid
from ..operators import apply_dirichlet, cell_to_edge, nodal_laplacian
from .base import Result, SolverBase, SolverConfig, Terminal

__all__ = ["ACSolver"]


class ACSolver(SolverBase):
    """Harmonic (phasor) quasi-static solver.

    Parameters
    ----------
    grid
        The mesh.
    eps_cell
        Absolute permittivity [F/m] per cell.  May be complex to represent
        dielectric loss (``eps' - j eps''``), or a callable ``f(freq)``
        returning such an array --- the A3 escape hatch for dispersive media.
    sigma_cell
        Conductivity [S/m] per cell.
    """

    name = "ac"
    assumptions = ("A1a", "A2", "A10", "A12")

    def __init__(self, grid: RectilinearGrid,
                 eps_cell: np.ndarray | Callable[[float], np.ndarray],
                 sigma_cell: np.ndarray, config: SolverConfig | None = None,
                 operators=None):
        super().__init__(grid, config, operators)
        self._eps_fn = eps_cell if callable(eps_cell) else None
        if not callable(eps_cell):
            eps = np.asarray(eps_cell)
            if eps.shape != grid.shape_cells:
                raise ValueError(f"eps_cell must have shape {grid.shape_cells}")
            self._eps = eps
        sigma_cell = np.asarray(sigma_cell, dtype=float)
        if sigma_cell.shape != grid.shape_cells:
            raise ValueError(f"sigma_cell must have shape {grid.shape_cells}")
        if np.any(sigma_cell < 0):
            raise ValueError("sigma_cell must be non-negative")
        self.L_sigma = nodal_laplacian(grid, cell_to_edge(grid, sigma_cell))
        self._Leps_cache: dict[int, sp.csr_matrix] = {}

    def _L_eps(self, freq: float) -> sp.csr_matrix:
        eps = self._eps_fn(freq) if self._eps_fn is not None else self._eps
        key = id(eps) if self._eps_fn is None else hash(float(freq))
        if key in self._Leps_cache:
            return self._Leps_cache[key]
        real = cell_to_edge(self.grid, np.real(eps).astype(float))
        L = nodal_laplacian(self.grid, real).astype(complex)
        if np.iscomplexobj(eps) and np.any(np.imag(eps) != 0.0):
            imag = cell_to_edge(self.grid, np.imag(eps).astype(float))
            L = L + 1j * nodal_laplacian(self.grid, imag)
        self._Leps_cache[key] = L
        return L

    # ------------------------------------------------------------------
    def _system(self, freq: float) -> sp.csr_matrix:
        w = 2.0 * np.pi * float(freq)
        return (self.L_sigma.astype(complex) + 1j * w * self._L_eps(freq)).tocsr()

    def solve(self, terminals: Sequence[Terminal],
              freqs: Sequence[float] | np.ndarray, bc=None) -> Result:
        """Solve at each frequency; returns terminal voltages and currents."""
        self._start()
        freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        terminals = list(terminals)
        n = self.grid.n_nodes
        out_v = {t.name: np.zeros(freqs.size, dtype=complex) for t in terminals}
        out_i = {t.name: np.zeros(freqs.size, dtype=complex) for t in terminals}

        for k, f in enumerate(freqs):
            K = self._system(f)
            fixed = np.concatenate([t.nodes for t in terminals])
            vals = np.concatenate([
                np.full(t.nodes.size, complex(t.value_at(0.0) or 0.0))
                for t in terminals])
            A, b = _apply_dirichlet_complex(K, np.zeros(n, dtype=complex),
                                            fixed, vals)
            phi = spl.spsolve(A.tocsc(), b)
            for t in terminals:
                out_v[t.name][k] = np.mean(phi[t.nodes])
                out_i[t.name][k] = (K @ phi)[t.nodes].sum()

        res = Result(grid=self.grid, t=np.zeros(0))
        res.scalars["freq"] = freqs
        for t in terminals:
            res.terminals[t.name] = {"v": out_v[t.name], "i": out_i[t.name]}
        return self._finish(res, mode="ac", n_freqs=freqs.size)

    def admittance_matrix(self, terminals: Sequence[Terminal],
                          freqs: Sequence[float] | np.ndarray,
                          bc=None) -> np.ndarray:
        """``(nf, nt, nt)`` admittance [S] by unit excitation.

        ``Y[k, i, j]`` is the current into terminal ``i`` when terminal ``j`` is
        driven to 1 V and all others are grounded.  Must be symmetric for a
        reciprocal structure --- check it, it is nearly free and catches most
        setup errors.
        """
        terminals = list(terminals)
        freqs = np.atleast_1d(np.asarray(freqs, dtype=float))
        nt = len(terminals)
        n = self.grid.n_nodes
        Y = np.zeros((freqs.size, nt, nt), dtype=complex)
        fixed = np.concatenate([t.nodes for t in terminals])
        for k, f in enumerate(freqs):
            K = self._system(f)
            for j in range(nt):
                vals = np.concatenate([
                    np.full(t.nodes.size, 1.0 + 0j if m == j else 0.0 + 0j)
                    for m, t in enumerate(terminals)])
                A, b = _apply_dirichlet_complex(K, np.zeros(n, dtype=complex),
                                                fixed, vals)
                phi = spl.spsolve(A.tocsc(), b)
                cur = K @ phi
                for i, t in enumerate(terminals):
                    Y[k, i, j] = cur[t.nodes].sum()
        return Y

    def impedance_matrix(self, terminals, freqs, bc=None) -> np.ndarray:
        """Pseudo-inverse of the admittance matrix [ohm].

        The admittance matrix of a floating structure is singular (a uniform
        potential shift drives no current), so a plain inverse is wrong; the
        pseudo-inverse is the physically meaningful object.
        """
        Y = self.admittance_matrix(terminals, freqs, bc)
        return np.stack([np.linalg.pinv(y) for y in Y])

    def s_parameters(self, terminals, freqs, z0: float = 50.0,
                     bc=None) -> np.ndarray:
        from ..extraction import s_parameters as _s
        return _s(self.admittance_matrix(terminals, freqs, bc), z0)


def _apply_dirichlet_complex(A: sp.spmatrix, b: np.ndarray,
                             fixed: np.ndarray, values: np.ndarray):
    """Complex-capable version of :func:`operators.apply_dirichlet`."""
    A = sp.csr_matrix(A, copy=True)
    b = np.array(b, dtype=complex, copy=True)
    n = A.shape[0]
    idx = np.asarray(fixed, dtype=np.intp)
    vals = np.asarray(values, dtype=complex)
    x0 = np.zeros(n, dtype=complex)
    x0[idx] = vals
    b -= A @ x0
    mask = np.ones(n, dtype=bool)
    mask[idx] = False
    Dm = sp.diags(mask.astype(float))
    A = Dm @ A @ Dm + sp.diags((~mask).astype(float))
    b[idx] = vals
    return sp.csr_matrix(A), b
