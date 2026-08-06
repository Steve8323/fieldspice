"""Exact field-circuit coupling --- the point of the project.

A meshed 3D field region and a lumped SPICE netlist are solved as **one**
system, not co-simulated with a handshake.

This is possible because of the formulation in ``docs/FORMULATION.md``: the
electroquasistatic system ``G^T M_sigma G phi = i`` *is* nodal analysis of a
resistor mesh, so a field region is already the same kind of object as a
netlist element.  Making them one matrix is assembly, not interpolation.

The method
----------
Per time step the field system is ``K phi = f``, with
``K = L_eps/dt + L_sigma`` and ``f = (L_eps/dt) phi_prev``.

Split the field nodes into *electrode* nodes (tied to circuit nodes) and
*interior* nodes.  Let ``P`` map electrode potentials to field nodes (a 0/1
matrix, one row per electrode) and ``S`` select the interior.  Eliminating the
interior exactly gives

    Y_eff = P K P^T - (P K S^T) K_ii^-1 (S K P^T)        [S]
    f_eff = P f     - (P K S^T) K_ii^-1 (S f)            [A]

and the current the region draws from circuit node ``e`` is
``i_e = sum_b Y_eff[e,b] v_b - f_eff[e]``.  That is a dense ``nE x nE``
admittance plus a history current source --- exactly the shape of an ordinary
multi-terminal MNA element, so it stamps like one.

**This is exact for a linear field region.**  No relaxation, no convergence
loop, no handshake, no splitting error.  The cost is one sparse factorisation
of ``K_ii`` plus ``nE`` back-solves per distinct ``dt``, both cached; after
that each time step is a small dense stamp and one back-solve to recover the
interior.

Because ``K`` is symmetric, ``Y_eff`` is symmetric too.  It is measured rather
than assumed --- :attr:`FieldRegion.asymmetry` reports the residual, and a
value far above solver tolerance means something is wrong upstream.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

from ..grid import RectilinearGrid
from ..operators import cell_to_edge, nodal_laplacian
from ..solvers.base import SolverConfig, Terminal
from ..solvers.poisson import LinearSystem
from .devices import Device, stamp_g, stamp_i

__all__ = ["FieldRegion", "FieldCircuitSystem"]


class FieldRegion(Device):
    """A meshed quasi-static field region stamped into a netlist.

    Parameters
    ----------
    name
        Instance name.
    grid
        The field mesh.
    eps_cell, sigma_cell
        Absolute permittivity [F/m] and conductivity [S/m] per cell.
    terminals
        Electrodes.  Each becomes one circuit node.
    circuit_nodes
        Netlist node label for each terminal, in the same order.  If omitted,
        each terminal's ``circuit_node`` attribute is used, falling back to its
        name.

    Notes
    -----
    Assumptions: **A1a**, **A2**, **A3**, **A10**, **A12** --- the region is
    electroquasistatic, so it contributes R and C to the circuit but no
    inductance.  For a region whose inductance matters, extract an L matrix
    with :mod:`fieldspice.extraction` and add it to the netlist explicitly, or
    use the Darwin solver.
    """

    linear = True
    dynamic = True
    assumptions = ("A1a", "A2", "A3", "A10", "A12")

    def __init__(self, name: str, grid: RectilinearGrid,
                 eps_cell: np.ndarray, sigma_cell: np.ndarray,
                 terminals: Sequence[Terminal],
                 circuit_nodes: Sequence[str] | None = None,
                 config: SolverConfig | None = None):
        self.name = str(name)
        self.grid = grid
        self.cfg = config or SolverConfig()
        self.terminals = list(terminals)
        if len(self.terminals) < 1:
            raise ValueError("a field region needs at least one terminal")

        if circuit_nodes is None:
            circuit_nodes = [t.circuit_node or t.name for t in self.terminals]
        circuit_nodes = [str(c) for c in circuit_nodes]
        if len(circuit_nodes) != len(self.terminals):
            raise ValueError("circuit_nodes must match terminals in length")
        self.nodes = tuple(circuit_nodes)

        eps_cell = np.asarray(eps_cell, dtype=float)
        sigma_cell = np.asarray(sigma_cell, dtype=float)
        for arr, nm in ((eps_cell, "eps_cell"), (sigma_cell, "sigma_cell")):
            if arr.shape != grid.shape_cells:
                raise ValueError(f"{nm} must have shape {grid.shape_cells}")
        if np.any(eps_cell <= 0):
            raise ValueError("eps_cell must be positive and ABSOLUTE [F/m]")

        self.L_eps = nodal_laplacian(grid, cell_to_edge(grid, eps_cell))
        self.L_sigma = nodal_laplacian(grid, cell_to_edge(grid, sigma_cell))

        n = grid.n_nodes
        elec_nodes = [t.nodes for t in self.terminals]
        flat = np.concatenate(elec_nodes)
        if np.unique(flat).size != flat.size:
            raise ValueError("terminals overlap: a node belongs to two electrodes")
        interior = np.setdiff1d(np.arange(n), flat)
        self._interior = interior
        nE = len(self.terminals)

        rows = np.concatenate([np.full(e.size, k) for k, e in enumerate(elec_nodes)])
        self.P = sp.csr_matrix((np.ones(flat.size), (rows, flat)), shape=(nE, n))
        self.S = sp.csr_matrix(
            (np.ones(interior.size), (np.arange(interior.size), interior)),
            shape=(interior.size, n))

        self._cache: dict[float, dict[str, Any]] = {}
        self.phi = np.zeros(n)
        self._phi_pending: np.ndarray | None = None
        self.asymmetry: float = 0.0

    # ------------------------------------------------------------------
    def extra_unknowns(self) -> tuple[str, ...]:
        return ()

    def _reduce(self, dt: float | None) -> dict[str, Any]:
        """Schur-complement reduction for a given step size (cached)."""
        key = float(dt) if dt is not None else 0.0
        if key in self._cache:
            return self._cache[key]

        K = (self.L_sigma if dt is None
             else (self.L_eps / dt + self.L_sigma)).tocsr()
        P, S = self.P, self.S
        K_tt = (P @ K @ P.T).toarray()
        K_ti = (P @ K @ S.T).tocsr()
        K_ii = (S @ K @ S.T).tocsr()

        # A purely capacitive region (sigma == 0) has NO dc conductance at all,
        # so its dc stamp is an open circuit and K is identically zero. That is
        # correct physics, not a degenerate case to be worked around -- but it
        # must be caught before handing a zero matrix to a factorisation.
        if K.nnz == 0 or float(abs(K).max()) == 0.0:
            entry = {"K": K, "Y": np.zeros_like(K_tt), "W": None,
                     "solver": None, "K_ti": K_ti, "dt": dt}
            self.asymmetry = 0.0
            self._cache[key] = entry
            return entry

        if K_ii.shape[0] == 0:
            Y = K_tt
            solver = None
            W = None
        else:
            solver = LinearSystem(K_ii, self.cfg)
            # W = K_ii^-1 K_it, one back-solve per electrode. nE is small.
            K_it = K_ti.T.tocsc()
            W = np.column_stack([solver.solve(np.asarray(K_it[:, j].todense()).ravel())
                                 for j in range(K_it.shape[1])])
            Y = K_tt - K_ti @ W

        self.asymmetry = float(np.abs(Y - Y.T).max()
                               / max(np.abs(Y).max(), 1e-300))
        Y = 0.5 * (Y + Y.T)
        entry = {"K": K, "Y": Y, "W": W, "solver": solver,
                 "K_ti": K_ti, "dt": dt}
        self._cache[key] = entry
        return entry

    def _history(self, entry: dict[str, Any], dt: float | None) -> np.ndarray:
        """``f_eff`` [A]: the history current source from the previous step."""
        if dt is None:
            return np.zeros(len(self.terminals))
        f = (self.L_eps / dt) @ self.phi
        f_t = self.P @ f
        if entry["solver"] is None:
            return f_t
        f_i = self.S @ f
        return f_t - entry["K_ti"] @ entry["solver"].solve(f_i)

    # ------------------------------------------------------------------
    def stamp_dc(self, G, I, x, nmap: Mapping[str, int], t=None) -> None:
        entry = self._reduce(None)
        Y = entry["Y"]
        idx = [nmap.get(n, -1) for n in self.nodes]
        for a in range(len(idx)):
            for b in range(len(idx)):
                stamp_g(G, idx[a], idx[b], Y[a, b])

    def stamp_tran(self, G, I, x, x_prev, dt: float,
                   nmap: Mapping[str, int], t=None) -> None:
        entry = self._reduce(dt)
        Y = entry["Y"]
        f = self._history(entry, dt)
        idx = [nmap.get(n, -1) for n in self.nodes]
        for a in range(len(idx)):
            for b in range(len(idx)):
                stamp_g(G, idx[a], idx[b], Y[a, b])
            stamp_i(I, idx[a], f[a])

    def stamp_ac(self, Y_mat, J, x_op, omega: float,
                 nmap: Mapping[str, int]) -> None:
        """Frequency domain: ``K = L_sigma + j w L_eps`` needs no history."""
        P, S = self.P, self.S
        K = (self.L_sigma.astype(complex) + 1j * omega * self.L_eps).tocsr()
        K_tt = (P @ K @ P.T).toarray()
        K_ti = (P @ K @ S.T).tocsr()
        K_ii = (S @ K @ S.T).tocsc()
        if K_ii.shape[0]:
            import scipy.sparse.linalg as spl
            lu = spl.splu(K_ii)
            K_it = K_ti.T.tocsc()
            W = np.column_stack([lu.solve(np.asarray(K_it[:, j].todense()).ravel())
                                 for j in range(K_it.shape[1])])
            Yeff = K_tt - K_ti @ W
        else:
            Yeff = K_tt
        idx = [nmap.get(n, -1) for n in self.nodes]
        for a in range(len(idx)):
            for b in range(len(idx)):
                stamp_g(Y_mat, idx[a], idx[b], Yeff[a, b])

    # ------------------------------------------------------------------
    def accept_timestep(self, x: np.ndarray, x_prev, dt: float,
                        nmap: Mapping[str, int]) -> None:
        """Back-substitute for the interior field once the step is accepted."""
        entry = self._reduce(dt)
        v = np.array([0.0 if nmap.get(n, -1) < 0 else float(x[nmap[n]])
                      for n in self.nodes])
        phi = np.zeros(self.grid.n_nodes)
        phi[np.concatenate([t.nodes for t in self.terminals])] = np.concatenate(
            [np.full(t.nodes.size, v[k]) for k, t in enumerate(self.terminals)])
        if entry["solver"] is not None and dt is not None:
            f_i = self.S @ ((self.L_eps / dt) @ self.phi)
            u = entry["solver"].solve(f_i - (entry["K_ti"].T @ v))
            phi[self._interior] = u
        self.phi = phi

    def reset_state(self) -> None:
        """Clear per-Newton-iteration state. Deliberately a no-op.

        ``MNASolver`` calls this **before every Newton solve**, i.e. once per
        time step, to discard iteration-local memory such as a limiter's last
        voltage. It must NOT touch ``self.phi``, which is *integration
        history* and has to survive from one accepted step to the next.

        Clearing the history here is a real bug we hit: it makes ``f_eff``
        identically zero, so the region degenerates to a bare conductance
        ``C/dt`` and an RC charging curve collapses to a resistive divider
        ``(dt/C)/(R + dt/C)``. Use :meth:`reset_history` to start a new
        transient.
        """
        return None

    def reset_history(self) -> None:
        """Discard the stored field state, starting a fresh transient."""
        self.phi = np.zeros(self.grid.n_nodes)

    def terminal_admittance(self, dt: float | None = None) -> np.ndarray:
        """The reduced ``nE x nE`` terminal admittance [S] for a given ``dt``."""
        return self._reduce(dt)["Y"]

    def op(self, x, nmap: Mapping[str, int]) -> dict[str, float]:
        return {f"{self.name}.v[{n}]": (0.0 if nmap.get(n, -1) < 0
                                        else float(x[nmap[n]]))
                for n in self.nodes}


class FieldCircuitSystem:
    """Convenience wrapper: a field region plus a netlist, solved together."""

    def __init__(self, region: FieldRegion, netlist):
        self.region = region
        self.netlist = netlist
        netlist.add_device(region)

    def transient(self, t_end: float, dt: float, **kw):
        from .mna import MNASolver
        return MNASolver(self.netlist).transient(t_end=t_end, dt=dt, **kw)

    def dc(self):
        from .mna import MNASolver
        return MNASolver(self.netlist).dc()
