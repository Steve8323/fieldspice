"""Tests for the frozen core: grid, operators, and the reference oracle.

These are the tests that everything else stands on. If any of them fails, no
result produced by any solver can be trusted, so they are deliberately strict:
most assert agreement at or near machine precision rather than at an
engineering tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from fieldspice import operators as ops
from fieldspice import reference as ref
from fieldspice.grid import RectilinearGrid, auto_mesh_1d, graded_1d
from fieldspice.units import eps0, mu0, q, thermal_voltage


# --------------------------------------------------------------------------
# Grid
# --------------------------------------------------------------------------
def _grids():
    """A deliberately awkward zoo: graded, shrinking, collapsed, single-cell."""
    x = graded_1d(0, 1e-3, 7, 1.3)
    y = graded_1d(0, 2e-3, 5, 0.8)      # cells shrink, not grow
    z = graded_1d(0, 3e-4, 4, 1.5)
    return {
        "3d_graded": RectilinearGrid(x, y, z),
        "2d": RectilinearGrid(x, y, None),
        "1d": RectilinearGrid(x, None, None),
        "uniform": RectilinearGrid.uniform([(0, 1), (0, 1), (0, 1)], [4, 3, 2]),
        "single": RectilinearGrid(np.array([0.0, 1.0])),
    }


@pytest.mark.parametrize("name", list(_grids()))
def test_grid_volumes_are_exact(name):
    g = _grids()[name]
    total = np.prod([b[1] - b[0] for b in g.bounds])
    assert g.cell_volumes().sum() == pytest.approx(total, rel=1e-14)
    # The dual boxes tile the domain exactly -- this is what makes the box
    # method conservative.
    assert g.node_volumes().sum() == pytest.approx(total, rel=1e-14)


@pytest.mark.parametrize("name", list(_grids()))
def test_element_counts_consistent(name):
    g = _grids()[name]
    nx, ny, nz = g.ncell
    assert g.n_nodes == (nx + 1) * (ny + 1) * (nz + 1)
    assert g.n_cells == nx * ny * nz
    assert g.n_edges == sum(int(np.prod(s)) for s in g.shape_edges)
    assert g.n_faces == sum(int(np.prod(s)) for s in g.shape_faces)


def test_collapsed_direction_has_two_node_planes():
    """The gotcha that catches everyone: a '1D' grid is not a 1D array."""
    g = RectilinearGrid(np.linspace(0, 1, 137))
    assert g.shape_nodes == (137, 2, 2)
    assert g.n_nodes == 137 * 4
    assert g.ndim_effective == 1


def test_edge_dual_areas_tile_the_cross_section():
    g = _grids()["3d_graded"]
    ax, ay, az = g.edge_dual_areas()
    Lx, Ly, Lz = g.bounds
    # Summing dual areas over a transverse node plane must recover the full
    # cross-sectional area.
    assert ax[0].sum() == pytest.approx((Ly[1] - Ly[0]) * (Lz[1] - Lz[0]), rel=1e-13)
    assert ay[:, 0, :].sum() == pytest.approx((Lx[1] - Lx[0]) * (Lz[1] - Lz[0]), rel=1e-13)
    assert az[:, :, 0].sum() == pytest.approx((Lx[1] - Lx[0]) * (Ly[1] - Ly[0]), rel=1e-13)


def test_graded_and_auto_mesh():
    n = graded_1d(0.0, 1.0, 10, 1.2)
    assert n.size == 11
    assert n[0] == 0.0 and n[-1] == pytest.approx(1.0)
    assert np.all(np.diff(n) > 0)
    h = np.diff(n)
    assert np.allclose(h[1:] / h[:-1], 1.2)

    m = auto_mesh_1d((0, 800e-9), [400e-9], dx_min=0.5e-9, dx_max=8e-9, growth=1.25)
    assert np.all(np.diff(m) > 0)
    assert m[0] == 0.0 and m[-1] == pytest.approx(800e-9)
    assert np.any(np.isclose(m, 400e-9))          # the feature became a node
    assert np.diff(m).min() <= 1.0e-9             # actually refined there
    assert np.diff(m).max() <= 8e-9 * 1.001       # respected the cap


def test_auto_mesh_honours_the_requested_growth_ratio():
    """The mesh generator must deliver the grading it was asked for.

    A ramp joined to a uniform fill can hand over with a 5x jump if the fill
    width is chosen only from dx_max, which silently degrades the box method to
    first order exactly where the geometry is interesting (A10).
    """
    cases = [
        ((0, 2.7e-6), (4e-7,), 4e-8, 2.5e-7),
        ((0, 5.5e-6), (2e-6, 2.5e-6, 3e-6, 3.5e-6), 4e-8, 2.5e-7),
        ((0, 8e-7), (4e-7,), 1e-9, 8e-9),
        ((0, 1e-3), (5e-4,), 1e-6, 5e-5),
        ((0, 1.0), (0.1, 0.9), 1e-3, 0.05),
    ]
    for growth in (1.2, 1.35, 1.5, 1.8):
        for ext, feats, dmin, dmax in cases:
            m = auto_mesh_1d(ext, feats, dx_min=dmin, dx_max=dmax, growth=growth)
            h = np.diff(m)
            worst = float(np.max(np.maximum(h[1:] / h[:-1], h[:-1] / h[1:])))
            assert worst <= growth * 1.10 + 1e-9, (growth, ext, worst)
            assert m[0] == pytest.approx(ext[0])
            assert m[-1] == pytest.approx(ext[1])
            for f in feats:
                assert np.any(np.isclose(m, f)), f


def test_grid_rejects_bad_input():
    with pytest.raises(ValueError):
        RectilinearGrid([1.0, 0.0])               # decreasing
    with pytest.raises(ValueError):
        RectilinearGrid([0.0])                    # too short
    with pytest.raises(ValueError):
        graded_1d(0, 1, 0)


# --------------------------------------------------------------------------
# Discrete calculus identities -- the heart of the scheme
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(_grids()))
def test_curl_of_grad_is_exactly_zero(name):
    g = _grids()[name]
    CG = ops.curl_edge_face(g) @ ops.grad_node_edge(g)
    # Not "close to zero" -- exactly zero. The entries are +-1 integers, so the
    # cancellation is exact in floating point on any grid whatsoever.
    assert CG.nnz == 0 or np.abs(CG.data).max() == 0.0


@pytest.mark.parametrize("name", list(_grids()))
def test_div_of_curl_is_exactly_zero(name):
    g = _grids()[name]
    DC = ops.div_face_cell(g) @ ops.curl_edge_face(g)
    assert DC.nnz == 0 or np.abs(DC.data).max() == 0.0


@pytest.mark.parametrize("name", list(_grids()))
def test_incidence_entries_are_signed_units(name):
    g = _grids()[name]
    for M in (ops.grad_node_edge(g), ops.curl_edge_face(g), ops.div_face_cell(g)):
        assert set(np.unique(M.data)).issubset({-1.0, 1.0})


def test_gradient_of_constant_vanishes():
    g = _grids()["3d_graded"]
    G = ops.grad_node_edge(g)
    assert np.abs(G @ np.ones(g.n_nodes)).max() == 0.0


def test_gradient_reproduces_a_linear_field_exactly():
    """G phi holds potential DIFFERENCES, so a linear ramp gives exactly h."""
    g = _grids()["3d_graded"]
    X, _, _ = g.node_coords()
    G = ops.grad_node_edge(g)
    gx, gy, gz = ops.split_edge_vector(g, G @ X.ravel())
    lx, _, _ = g.edge_lengths()
    assert np.abs(gx - lx).max() < 1e-18
    assert np.abs(gy).max() == 0.0
    assert np.abs(gz).max() == 0.0


def test_operators_cache_and_identity_report():
    g = _grids()["3d_graded"]
    O = ops.Operators(g)
    assert O.G is O.G and O.C is O.C and O.D is O.D
    ident = O.check_identities()
    assert ident["max|C@G|"] == 0.0
    assert ident["max|D@C|"] == 0.0


# --------------------------------------------------------------------------
# Mass matrices and material averaging
# --------------------------------------------------------------------------
def test_edge_mass_is_a_conductance():
    g = RectilinearGrid.uniform([(0, 2.0), (0, 3.0), (0, 5.0)], [1, 1, 1])
    sig = np.full(g.shape_cells, 7.0)
    M = ops.edge_mass(g, ops.cell_to_edge(g, sig))
    ex, _, _ = ops.split_edge_vector(g, M.diagonal())
    # One cell: an x-edge spans length 2 with dual area (3/2)*(5/2).
    assert ex.ravel()[0] == pytest.approx(7.0 * (1.5 * 2.5) / 2.0)


def test_cell_to_edge_parallel_average_is_area_weighted():
    g = RectilinearGrid.uniform([(0, 1.0), (0, 1.0), (0, 1.0)], [1, 2, 1])
    prop = np.zeros(g.shape_cells)
    prop[0, 0, 0] = 10.0
    prop[0, 1, 0] = 20.0
    e = ops.cell_to_edge(g, prop)
    ex, _, _ = ops.split_edge_vector(g, e)
    # The interior x-edge at j=1 is flanked equally by both cells.
    assert ex[0, 1, 0] == pytest.approx(15.0)
    # Edges on the outer walls see only their own cell (edge-padded).
    assert ex[0, 0, 0] == pytest.approx(10.0)
    assert ex[0, 2, 0] == pytest.approx(20.0)


def test_cell_to_edge_harmonic_mode():
    g = RectilinearGrid.uniform([(0, 1.0), (0, 1.0), (0, 1.0)], [1, 2, 1])
    prop = np.zeros(g.shape_cells)
    prop[0, 0, 0] = 10.0
    prop[0, 1, 0] = 20.0
    ex, _, _ = ops.split_edge_vector(g, ops.cell_to_edge(g, prop, mode="harmonic"))
    assert ex[0, 1, 0] == pytest.approx(2.0 / (1 / 10.0 + 1 / 20.0))


def test_cell_to_node_conserves_a_uniform_field():
    g = _grids()["3d_graded"]
    prop = np.full(g.shape_cells, 3.7)
    assert np.allclose(ops.cell_to_node(g, prop), 3.7)


def test_material_shape_validation():
    g = _grids()["3d_graded"]
    with pytest.raises(ValueError):
        ops.cell_to_edge(g, np.zeros((2, 2, 2)))
    with pytest.raises(ValueError):
        ops.edge_mass(g, np.zeros(3))
    with pytest.raises(ValueError):
        ops.cell_to_edge(g, np.ones(g.shape_cells), mode="nonsense")


# --------------------------------------------------------------------------
# Physics: the scheme reproduces analytic C and R
# --------------------------------------------------------------------------
def _plate_problem(prop_cell, n=(24, 10, 8), d=2e-6, W=5e-6, H=4e-6):
    g = RectilinearGrid(np.linspace(0, d, n[0] + 1),
                        np.linspace(0, W, n[1] + 1),
                        np.linspace(0, H, n[2] + 1))
    L = ops.nodal_laplacian(g, ops.cell_to_edge(g, prop_cell(g)))
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    lo, hi = nid[0].ravel(), nid[-1].ravel()
    fixed = np.concatenate([lo, hi])
    vals = np.concatenate([np.ones(lo.size), np.zeros(hi.size)])
    A, b = ops.apply_dirichlet(L, np.zeros(g.n_nodes), fixed, vals)
    phi = spl.spsolve(A.tocsc(), b)
    return g, L, phi, lo


def test_parallel_plate_capacitance_is_machine_exact():
    er, d, W, H = 3.9, 2e-6, 5e-6, 4e-6
    g, L, phi, lo = _plate_problem(lambda g: np.full(g.shape_cells, er * eps0))
    C = (L @ phi)[lo].sum()
    assert C == pytest.approx(ref.parallel_plate_capacitance(W * H, d, er), rel=1e-12)


def test_slab_resistance_is_machine_exact():
    sigma, d, W, H = 1.0e4, 2e-6, 5e-6, 4e-6
    g, L, phi, lo = _plate_problem(lambda g: np.full(g.shape_cells, sigma))
    R = 1.0 / (L @ phi)[lo].sum()
    assert R == pytest.approx(ref.slab_resistance(d, W * H, sigma), rel=1e-12)


def test_potential_between_plates_is_linear():
    g, L, phi, lo = _plate_problem(lambda g: np.full(g.shape_cells, eps0))
    prof = phi.reshape(g.shape_nodes)[:, 4, 4]
    assert np.abs(prof - np.linspace(1, 0, prof.size)).max() < 1e-12


def test_capacitance_is_geometry_independent_on_a_graded_mesh():
    """The same physical capacitor on a nastily graded mesh gives the same C."""
    er, d, W, H = 3.9, 2e-6, 5e-6, 4e-6
    x = graded_1d(0, d, 30, 1.25)
    g = RectilinearGrid(x, graded_1d(0, W, 9, 0.85), np.linspace(0, H, 7))
    L = ops.nodal_laplacian(g, ops.cell_to_edge(g, np.full(g.shape_cells, er * eps0)))
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    lo, hi = nid[0].ravel(), nid[-1].ravel()
    A, b = ops.apply_dirichlet(L, np.zeros(g.n_nodes),
                               np.concatenate([lo, hi]),
                               np.concatenate([np.ones(lo.size), np.zeros(hi.size)]))
    C = (L @ spl.spsolve(A.tocsc(), b))[lo].sum()
    assert C == pytest.approx(ref.parallel_plate_capacitance(W * H, d, er), rel=1e-10)


def test_series_dielectric_stack_matches_series_capacitors():
    er1, er2, d, W, H = 4.0, 2.0, 4e-6, 3e-6, 2e-6
    def prop(g):
        e = np.empty(g.shape_cells)
        left = g.xc < d / 2
        e[left] = er1 * eps0
        e[~left] = er2 * eps0
        return e
    g, L, phi, lo = _plate_problem(prop, n=(40, 4, 3), d=d, W=W, H=H)
    C = (L @ phi)[lo].sum()
    C1 = ref.parallel_plate_capacitance(W * H, d / 2, er1)
    C2 = ref.parallel_plate_capacitance(W * H, d / 2, er2)
    assert C == pytest.approx(C1 * C2 / (C1 + C2), rel=1e-10)


def test_dirichlet_preserves_symmetry():
    g = _grids()["3d_graded"]
    L = ops.nodal_laplacian(g, ops.cell_to_edge(g, np.full(g.shape_cells, eps0)))
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    A, b = ops.apply_dirichlet(L, np.zeros(g.n_nodes), nid[0].ravel(), 1.0)
    assert (abs(A - A.T) > 1e-18).nnz == 0


def test_laplacian_null_space_is_the_constant():
    g = _grids()["uniform"]
    L = ops.nodal_laplacian(g, ops.cell_to_edge(g, np.full(g.shape_cells, eps0)))
    assert np.abs(L @ np.ones(g.n_nodes)).max() < 1e-24


# --------------------------------------------------------------------------
# Transient: EQS scheme against the analytic RC response
# --------------------------------------------------------------------------
def _rc_transient(steps_per_tau, ntau=8, N=60):
    d, W, H = 4e-6, 3e-6, 2e-6
    sigma1, er1, er2 = 1.0e3, 4.0, 2.0
    g = RectilinearGrid(np.linspace(0, d, N + 1), np.linspace(0, W, 4),
                        np.linspace(0, H, 3))
    left = g.xc < d / 2
    sig = np.zeros(g.shape_cells)
    sig[left] = sigma1
    eps = np.empty(g.shape_cells)
    eps[left] = er1 * eps0
    eps[~left] = er2 * eps0
    Ls = ops.nodal_laplacian(g, ops.cell_to_edge(g, sig))
    Le = ops.nodal_laplacian(g, ops.cell_to_edge(g, eps))
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    lo, hi = nid[0].ravel(), nid[-1].ravel()
    fixed = np.concatenate([lo, hi])
    vals = np.concatenate([np.ones(lo.size), np.zeros(hi.size)])

    A = W * H
    R1 = (d / 2) / (sigma1 * A)
    C1 = ref.parallel_plate_capacitance(A, d / 2, er1)
    C2 = ref.parallel_plate_capacitance(A, d / 2, er2)
    tau = R1 * (C1 + C2)
    v0 = C1 / (C1 + C2)

    dt = tau / steps_per_tau
    K = (Le / dt + Ls).tocsc()
    Kbc, _ = ops.apply_dirichlet(K, np.zeros(g.n_nodes), fixed, vals)
    lu = spl.splu(Kbc.tocsc())
    # Consistent initial condition: the ELECTROSTATIC solution, not zero.
    Ebc, b0 = ops.apply_dirichlet(Le, np.zeros(g.n_nodes), fixed, vals)
    phi = spl.spsolve(Ebc.tocsc(), b0)

    mid = nid[N // 2, 1, 1]
    ts, vm = [0.0], [phi[mid]]
    for k in range(int(ntau * steps_per_tau)):
        _, b = ops.apply_dirichlet(K, Le @ phi / dt, fixed, vals)
        phi = lu.solve(b)
        ts.append((k + 1) * dt)
        vm.append(phi[mid])
    ts, vm = np.array(ts), np.array(vm)
    return ts, vm, ref.rc_step(ts, R1, C1 + C2, 1.0, v0), tau, v0


def test_eqs_initial_condition_is_the_capacitive_divider():
    _, vm, _, _, v0 = _rc_transient(50)
    assert vm[0] == pytest.approx(v0, rel=1e-9)
    assert v0 == pytest.approx(2.0 / 3.0, rel=1e-9)


def test_eqs_transient_matches_analytic_rc():
    ts, vm, ana, tau, _ = _rc_transient(400)
    assert np.abs(vm - ana).max() < 2e-4


def test_eqs_backward_euler_is_first_order():
    errs = [np.abs(v - a).max() for _, v, a, _, _ in
            (_rc_transient(s) for s in (50, 100, 200, 400))]
    for lo, hi in zip(errs[:-1], errs[1:]):
        assert 1.7 < lo / hi < 2.3, f"expected O(dt), got ratio {lo / hi}"


# --------------------------------------------------------------------------
# Nonlinear Poisson: the pn junction
# --------------------------------------------------------------------------
def test_pn_junction_built_in_potential_is_machine_exact():
    Vt = thermal_voltage(300.0)
    ni, er = ref.NI_SI_300, 11.7
    Na = Nd = 1e23
    L, xj = 800e-9, 400e-9

    g = RectilinearGrid(auto_mesh_1d((0, L), [xj], dx_min=0.5e-9,
                                     dx_max=8e-9, growth=1.25))
    X, _, _ = g.node_coords()
    Le = ops.nodal_laplacian(g, ops.cell_to_edge(g, np.full(g.shape_cells, er * eps0)))
    Vn = ops.node_volume_vector(g)
    dop = np.where(X.ravel() < xj, -Na, Nd)
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    fixed = np.concatenate([nid[0].ravel(), nid[-1].ravel()])
    psi_p = Vt * np.arcsinh(-Na / (2 * ni))
    psi_n = Vt * np.arcsinh(Nd / (2 * ni))
    vals = np.concatenate([np.full(nid[0].size, psi_p), np.full(nid[-1].size, psi_n)])

    psi = np.interp(X.ravel(), [0, xj, L], [psi_p, 0.5 * (psi_p + psi_n), psi_n])
    psi[fixed] = vals
    free = np.setdiff1d(np.arange(g.n_nodes), fixed)

    def residual(ps):
        n = ni * np.exp(np.clip(ps / Vt, -400, 400))
        p = ni * np.exp(np.clip(-ps / Vt, -400, 400))
        return Le @ ps - q * (p - n + dop) * Vn, n, p

    # Damped Newton with a residual-monotone line search. A bare step clamp
    # (lam = k*Vt/max|dpsi|) is NOT enough: it produces a limit cycle for some
    # doping/ni combinations because nothing forces ||F|| to decrease.
    for _ in range(200):
        F, n, p = residual(psi)
        Fnorm = np.linalg.norm(F[free])
        J = (Le + sp.diags(q * (n + p) / Vt * Vn)).tocsc()
        Jbc, Fbc = ops.apply_dirichlet(J, -F, fixed, np.zeros(fixed.size))
        dpsi = spl.spsolve(Jbc.tocsc(), Fbc)
        m = float(np.abs(dpsi).max())
        lam = min(1.0, 5 * Vt / m) if m > 0 else 1.0
        for _ in range(60):
            if np.linalg.norm(residual(psi + lam * dpsi)[0][free]) < \
                    (1 - 1e-4 * lam) * Fnorm:
                break
            lam *= 0.5
        psi = psi + lam * dpsi
        if m * lam < 1e-12:
            break
    else:
        pytest.fail("Newton did not converge")

    prof = psi.reshape(g.shape_nodes)[:, 0, 0]
    Vbi = ref.built_in_potential(Na, Nd, ni)
    assert prof[-1] - prof[0] == pytest.approx(Vbi, rel=1e-9)

    # Global charge neutrality: the two space-charge lobes must cancel.
    x1 = g.xn
    n = ni * np.exp(prof / Vt)
    p = ni * np.exp(-prof / Vt)
    rho = q * (p - n + np.where(x1 < xj, -Na, Nd))
    vol = ops.node_volume_vector(g).reshape(g.shape_nodes)[:, 0, 0] * 4
    assert abs(float((rho * vol).sum())) < 1e-14

    # The solution must be uniform across the collapsed directions.
    assert np.allclose(psi.reshape(g.shape_nodes),
                       psi.reshape(g.shape_nodes)[:, :1, :1])


# --------------------------------------------------------------------------
# Reference oracle self-consistency
# --------------------------------------------------------------------------
def test_reference_values_against_textbook_numbers():
    assert ref.subthreshold_slope() == pytest.approx(59.5e-3, rel=2e-3)
    assert ref.skin_depth(5.8e7, 1e9) == pytest.approx(2.09e-6, rel=2e-2)
    assert ref.skin_depth(5.8e7, 1e6) == pytest.approx(66.1e-6, rel=2e-2)
    assert ref.internal_inductance_round_wire() == pytest.approx(mu0 / (8 * np.pi))
    assert ref.debye_length(1e23) == pytest.approx(12.9e-9, rel=2e-2)
    assert ref.intrinsic_carrier_density_si(300.0) == pytest.approx(9.65e15, rel=1e-6)
    # A 50-ohm FR4 microstrip is famously near w/h = 2.
    z0, ee = ref.microstrip_z0(2e-3, 1e-3, 4.4)
    assert z0 == pytest.approx(50.0, abs=3.0)
    assert 3.0 < ee < 3.6


def test_coaxial_z0_round_trip():
    a, b, er = 1.0, 3.5, 2.1
    Z0 = np.sqrt(ref.coaxial_inductance(a, b) / ref.coaxial_capacitance(a, b, er))
    assert Z0 == pytest.approx(51.8, rel=0.02)
    # LC must reproduce the medium's phase velocity exactly.
    v = 1.0 / np.sqrt(ref.coaxial_inductance(a, b) * ref.coaxial_capacitance(a, b, er))
    assert v == pytest.approx(299792458.0 / np.sqrt(er), rel=1e-9)


def test_electrical_length_bands_pick_the_right_solver():
    # On-chip: deeply quasi-static.
    assert ref.electrical_length(100e-6, t_rise=20e-12, eps_r=4.0) < 0.02
    # A 10 cm board trace at the same edge rate: firmly full-wave.
    assert ref.electrical_length(0.1, t_rise=20e-12, eps_r=4.0) > 1.0


def test_rlc_step_covers_all_damping_regimes():
    t = np.linspace(0, 5e-9, 500)
    under = ref.rlc_step(t, 1.0, 1e-9, 1e-12)
    crit = ref.rlc_step(t, 2 * np.sqrt(1e-9 / 1e-12), 1e-9, 1e-12)
    over = ref.rlc_step(t, 1e4, 1e-9, 1e-12)
    assert under.max() > 1.5           # rings and overshoots
    assert crit.max() <= 1.0 + 1e-9    # no overshoot
    assert over.max() <= 1.0 + 1e-9
    for y in (under, crit, over):
        assert y[0] == pytest.approx(0.0, abs=1e-12)
    f, zeta, Q = ref.rlc_ringdown(1.0, 1e-9, 1e-12)
    assert Q == pytest.approx(31.62, rel=1e-3)


def test_elmore_delay_matches_hand_calculation():
    r = np.array([1.0, 2.0, 3.0])
    c = np.array([4.0, 5.0, 6.0])
    assert ref.elmore_delay_rc_ladder(r, c) == pytest.approx(
        4 * 1 + 5 * 3 + 6 * 6)
    with pytest.raises(ValueError):
        ref.elmore_delay_rc_ladder(r, c[:2])


def test_fdtd_dispersion_approaches_c_as_grid_refines():
    lam = 1.0
    prev = 0.0
    for cells in (10, 20, 40, 80):
        dx = lam / cells
        dt = ref.courant_limit(dx, dx) * 0.99
        v = ref.fdtd_numerical_dispersion(dx, dt, 299792458.0 / lam)
        rel = v / 299792458.0
        assert rel < 1.0 + 1e-9          # Yee is always slow on the diagonal axis
        assert rel > prev                # monotone improvement
        prev = rel
    assert prev > 0.999
