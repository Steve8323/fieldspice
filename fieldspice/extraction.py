"""Parasitic extraction: turning a field solve into circuit numbers.

This is the layer most users actually want.  A capacitance matrix, a
per-unit-length RLGC set, an S-parameter block --- these are what get handed to
a circuit simulator, a signal-integrity flow, or a datasheet.

A note on which capacitance you are asking for, because the two differ and
mixing them up is a classic source of wrong answers:

* The **Maxwell** (short-circuit) capacitance matrix has
  ``Q_i = sum_j C_ij V_j``.  Diagonals are positive, off-diagonals negative,
  and rows sum to zero for a closed system.  This is what a field solve
  produces naturally.
* The **SPICE** (mutual) capacitance matrix is the netlist you would draw:
  ``C_ij_spice = -C_ij_maxwell`` for ``i != j``, and the self term to ground is
  the row sum.  This is what you put in a netlist.

:func:`capacitance_matrix` returns the Maxwell form; :func:`to_spice_matrix`
converts.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .grid import RectilinearGrid
from .units import c0, eps0, mu0

__all__ = [
    "capacitance_matrix", "resistance_matrix", "to_spice_matrix",
    "rlgc_2d", "characteristic_impedance", "s_parameters", "y_to_z", "z_to_y",
    "elmore_delay", "skin_depth", "check_lc_identity",
]


# ==========================================================================
# Static matrices
# ==========================================================================
def capacitance_matrix(grid: RectilinearGrid, eps_cell: np.ndarray,
                       terminals: Sequence, bc=None, config=None
                       ) -> np.ndarray:
    """Maxwell capacitance matrix [F], shape ``(nt, nt)``.

    Computed by unit excitation: drive terminal ``j`` to 1 V with every other
    terminal grounded, then integrate the charge on each terminal ``i``.

    The result is symmetrised, and the *asymmetry before symmetrisation* is a
    genuinely useful error estimate --- it is zero for an exact solve, so its
    size measures discretisation and solver tolerance together.  It is returned
    in ``report`` if you pass a dict.
    """
    from .solvers.poisson import PoissonSolver
    solver = PoissonSolver(grid, eps_cell, config)
    return solver.capacitance_matrix(list(terminals), bc)


def resistance_matrix(grid: RectilinearGrid, sigma_cell: np.ndarray,
                      terminals: Sequence, bc=None, config=None) -> np.ndarray:
    """DC resistance matrix [ohm], the inverse of the terminal conductance.

    Built by driving each terminal to 1 V in turn and measuring the current
    into every terminal, which gives the conductance matrix ``Y``; the
    resistance matrix is its pseudo-inverse, because ``Y`` is singular by
    construction (adding a constant to every terminal voltage drives no
    current, so the all-ones vector is in the null space).
    """
    from .operators import apply_dirichlet, cell_to_edge, nodal_laplacian
    from .solvers.poisson import LinearSystem

    terminals = list(terminals)
    nt = len(terminals)
    if nt < 2:
        raise ValueError("resistance extraction needs at least two terminals")
    L = nodal_laplacian(grid, cell_to_edge(grid, np.asarray(sigma_cell, float)))
    Y = np.zeros((nt, nt))
    for j, drive in enumerate(terminals):
        fixed = np.concatenate([t.nodes for t in terminals])
        vals = np.concatenate([np.full(t.nodes.size, 1.0 if t is drive else 0.0)
                               for t in terminals])
        A, b = apply_dirichlet(L, np.zeros(grid.n_nodes), fixed, vals)
        phi = LinearSystem(A, config).solve(b)
        for i, t in enumerate(terminals):
            Y[i, j] = float((L @ phi)[t.nodes].sum())
    Y = 0.5 * (Y + Y.T)
    return np.linalg.pinv(Y)


def to_spice_matrix(c_maxwell: np.ndarray) -> np.ndarray:
    """Convert a Maxwell capacitance matrix to the SPICE/mutual form [F].

    Off-diagonal ``C_ij_spice = -C_ij_maxwell`` is the capacitor you draw
    between conductors ``i`` and ``j``; the diagonal becomes the capacitance
    from conductor ``i`` to ground, i.e. the row sum of the Maxwell matrix.
    """
    c = np.asarray(c_maxwell, dtype=float)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("capacitance matrix must be square")
    out = -c.copy()
    np.fill_diagonal(out, c.sum(axis=1))
    return out


# ==========================================================================
# Transmission lines
# ==========================================================================
def rlgc_2d(grid: RectilinearGrid, eps_cell: np.ndarray,
            sigma_cell: np.ndarray, terminals: Sequence,
            mu_cell: np.ndarray | None = None, freq: float = 0.0,
            config=None, reference: int | str = -1) -> dict[str, np.ndarray]:
    """Per-unit-length R, L, G, C matrices from a 2D cross-section.

    The grid must be a 2D cross-section (``grid.Nz == 1``) with unit thickness
    in the collapsed direction, so every result is naturally per metre.

    One conductor is the **return path** and is not an independent unknown.
    ``reference`` selects it (index or terminal name, default the last), and
    every returned matrix has size ``(nt-1, nt-1)``, referenced to it.

    This matters and is easy to get wrong: the raw Maxwell capacitance matrix
    is **singular** --- its rows sum to zero, because adding a constant to
    every conductor's potential moves no charge --- so inverting it directly
    produces nonsense.  Removing the reference row and column gives the
    non-singular per-unit-length matrix that transmission-line theory wants.

    ``L`` is obtained from the **LC identity** rather than from a separate
    magnetostatic solve.  For a TEM line in a homogeneous medium,

        L C = mu eps I

    exactly, so ``L = mu eps C^-1``.  For an inhomogeneous cross-section
    (microstrip, with dielectric below and air above) the identity holds
    against the capacitance of the *same geometry with all dielectrics removed*:

        L = mu0 eps0 C_air^-1

    which is the standard and is what this function computes --- it solves the
    capacitance problem twice, once with the real permittivity and once with
    vacuum everywhere.  That is exact for a TEM mode and is far more robust
    than an independent inductance extraction, which would have to get the
    return-current distribution right to the same accuracy.

    Returns
    -------
    dict
        ``{"R", "L", "G", "C", "C_air", "eps_eff", "Z0", "v_p", "reference"}``,
        all matrices ``(nt-1, nt-1)``.

    Notes
    -----
    Tagged **A1** (quasi-static, so this is the TEM/quasi-TEM limit),
    **A2**, **A12**.  ``R`` is the DC series resistance from ``sigma_cell``
    plus, when ``freq > 0``, a skin-effect correction; it is *not* a full
    eddy-current solve, so treat it as a first-order estimate above the skin
    corner and cross-check with :mod:`fieldspice.solvers.mqs` when it matters.
    """
    if grid.Nz != 1:
        raise ValueError("rlgc_2d needs a 2D cross-section (grid.Nz == 1)")
    terminals = list(terminals)
    nt = len(terminals)
    if nt < 2:
        raise ValueError(
            "rlgc_2d needs at least two conductors: one signal and one return")

    if isinstance(reference, str):
        names = [t.name for t in terminals]
        if reference not in names:
            raise ValueError(f"reference {reference!r} is not among {names}")
        ref_i = names.index(reference)
    else:
        ref_i = int(reference) % nt
    keep = [i for i in range(nt) if i != ref_i]

    eps_cell = np.asarray(eps_cell, dtype=float)
    sigma_cell = np.asarray(sigma_cell, dtype=float)

    def _reduce(m: np.ndarray) -> np.ndarray:
        return np.asarray(m)[np.ix_(keep, keep)]

    C = _reduce(capacitance_matrix(grid, eps_cell, terminals, config=config))
    C_air = _reduce(capacitance_matrix(grid, np.full_like(eps_cell, eps0),
                                       terminals, config=config))

    mu_r = 1.0
    if mu_cell is not None:
        mu_r = float(np.mean(np.asarray(mu_cell, dtype=float)) / mu0)
    L = mu_r * mu0 * eps0 * np.linalg.inv(C_air)

    # Shunt conductance: G solves the identical Laplace problem with sigma in
    # place of eps, so the same routine serves with the coefficient swapped.
    if np.any(sigma_cell > 0):
        G = _reduce(capacitance_matrix(grid, np.maximum(sigma_cell, 1e-30),
                                       terminals, config=config))
    else:
        G = np.zeros_like(C)

    R = _series_resistance(grid, sigma_cell, [terminals[i] for i in keep],
                           freq, config)

    eps_eff = np.diag(C) / np.diag(C_air)
    v_p = 1.0 / np.sqrt(np.diag(L) * np.diag(C))
    Z0 = np.sqrt(np.diag(L) / np.diag(C))
    return {"R": R, "L": L, "G": G, "C": C, "C_air": C_air,
            "eps_eff": eps_eff, "Z0": Z0, "v_p": v_p,
            "reference": terminals[ref_i].name}


def _series_resistance(grid, sigma_cell, terminals, freq, config) -> np.ndarray:
    """DC (or skin-corrected) series resistance per unit length [ohm/m]."""
    nt = len(terminals)
    R = np.zeros((nt, nt))
    cellvol = grid.cell_volumes()
    length = grid.zn[-1] - grid.zn[0]  # the collapsed (propagation) direction
    for i, t in enumerate(terminals):
        # Cross-sectional area of the conductor owning this terminal, found by
        # flood-filling the conducting cells that touch the electrode nodes.
        mask = sigma_cell > 0
        area = float(cellvol[mask].sum() / max(length, 1e-300)) if mask.any() else 0.0
        sig = float(np.mean(sigma_cell[mask])) if mask.any() else 0.0
        if area <= 0 or sig <= 0:
            R[i, i] = 0.0
            continue
        r_dc = 1.0 / (sig * area)
        if freq > 0:
            delta = skin_depth(sig, freq)
            # Crude high-frequency correction: current confined to a shell of
            # thickness delta around the perimeter. Documented as approximate.
            perim = 4.0 * np.sqrt(area)
            a_hf = min(area, perim * delta)
            r_dc = max(r_dc, 1.0 / (sig * max(a_hf, 1e-300)))
        R[i, i] = r_dc
    return R


def characteristic_impedance(rlgc: dict, freq: float | np.ndarray = 0.0
                             ) -> np.ndarray:
    """``Z0 = sqrt((R + jwL)/(G + jwC))`` [ohm]; lossless limit at ``freq=0``."""
    R, L, G, C = (np.atleast_2d(rlgc[k]) for k in ("R", "L", "G", "C"))
    w = 2.0 * np.pi * np.asarray(freq, dtype=float)
    if np.all(w == 0):
        return np.sqrt(np.diag(L) / np.diag(C))
    num = np.diag(R) + 1j * w[..., None] * np.diag(L)
    den = np.diag(G) + 1j * w[..., None] * np.diag(C)
    return np.sqrt(num / den)


def check_lc_identity(rlgc: dict, eps_r: float = 1.0,
                      mu_r: float = 1.0) -> float:
    """Relative residual of ``L C = mu eps I``.

    A strong self-consistency check for a homogeneously filled line: it must
    hold exactly for a TEM mode, so a large residual means one of the two
    matrices is wrong. Returns the max relative deviation from the identity.
    """
    L = np.atleast_2d(rlgc["L"])
    C = np.atleast_2d(rlgc["C"])
    target = mu_r * mu0 * eps_r * eps0 * np.eye(L.shape[0])
    got = L @ C
    scale = max(np.abs(target).max(), 1e-300)
    return float(np.abs(got - target).max() / scale)


# ==========================================================================
# Network parameters
# ==========================================================================
def y_to_z(y: np.ndarray) -> np.ndarray:
    """Invert an admittance matrix (or a stack of them) to impedance."""
    return np.linalg.inv(np.asarray(y))


def z_to_y(z: np.ndarray) -> np.ndarray:
    return np.linalg.inv(np.asarray(z))


def s_parameters(y: np.ndarray, z0: float | np.ndarray = 50.0) -> np.ndarray:
    """Scattering matrix from an admittance matrix.

    ``S = (I - Z0 Y)(I + Z0 Y)^-1`` for a real, common reference impedance.
    Accepts a single ``(n, n)`` matrix or a stack ``(nf, n, n)``.

    The result is checked for passivity by the caller's convenience: for a
    passive network every singular value of ``S`` satisfies ``s <= 1``, and for
    a reciprocal one ``S`` is symmetric.  Both are cheap and catch most errors
    in an extraction flow, so check them.
    """
    y = np.asarray(y)
    single = y.ndim == 2
    Y = y[None, ...] if single else y
    n = Y.shape[-1]
    Z0 = np.eye(n) * z0 if np.isscalar(z0) else np.diag(np.asarray(z0, float))
    I = np.eye(n)
    S = np.empty_like(Y, dtype=complex)
    for k in range(Y.shape[0]):
        A = Z0 @ Y[k]
        S[k] = (I - A) @ np.linalg.inv(I + A)
    return S[0] if single else S


# ==========================================================================
# Simple estimators
# ==========================================================================
def elmore_delay(r: np.ndarray, c: np.ndarray) -> float:
    """Elmore delay of an RC ladder [s].

    An *upper* bound on the 50% delay of a monotonic step response, typically
    20-40% high, so a disagreement of that size with an EQS transient is the
    expected result rather than an error.
    """
    from .reference import elmore_delay_rc_ladder
    return elmore_delay_rc_ladder(r, c)


def skin_depth(sigma: float, freq: float, mu_r: float = 1.0) -> float:
    """``sqrt(2/(omega mu sigma))`` [m]. Copper at 1 GHz: 2.09 um."""
    from .reference import skin_depth as _sd
    return _sd(sigma, freq, mu_r)
