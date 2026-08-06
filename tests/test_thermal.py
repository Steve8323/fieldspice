"""Heat conduction and coupled electro-thermal validation.

The thermal solver is checked three ways: against closed-form conduction
solutions, against a *different code path* in this package (the MNA circuit
solver driving the same network as literal resistors and capacitors), and
against the analytic electro-thermal fixed point including its runaway
threshold.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fieldspice.boundaries import BoundarySpec, Dirichlet
from fieldspice.grid import RectilinearGrid
from fieldspice.operators import node_volume_vector
from fieldspice.physics import PhysicsOptions
from fieldspice.solvers.base import Terminal
from fieldspice.solvers.electrothermal import (ElectroThermalSolver,
                                               ThermalRunaway,
                                               runaway_threshold)
from fieldspice.solvers.thermal import ThermalSolver, joule_heating_nodes


# ==========================================================================
# Pure conduction
# ==========================================================================
def _bar(n=40, L=1e-3, k=150.0):
    g = RectilinearGrid(np.linspace(0, L, n + 1), np.linspace(0, 1e-3, 2),
                        np.linspace(0, 1e-3, 2))
    return g, ThermalSolver(g, np.full(g.shape_cells, k))


def test_uniform_heating_gives_the_parabolic_profile():
    """q constant, both ends fixed: T = T0 + q x (L-x) / 2k, exact for the
    3-point Laplacian because the solution is quadratic."""
    L, k, T0, qv = 1e-3, 150.0, 300.0, 1e9
    g, s = _bar(40, L, k)
    q = qv * node_volume_vector(g)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = s.steady(q, bc=BoundarySpec(xlo=Dirichlet(T0), xhi=Dirichlet(T0)))
    T = r.fields["T"][0].reshape(g.shape_nodes)[:, 0, 0]
    ana = T0 + qv * g.xn * (L - g.xn) / (2 * k)
    assert np.abs(T - ana).max() < 1e-9
    assert T.max() - T0 == pytest.approx(qv * L ** 2 / (8 * k), rel=1e-9)


def test_heat_equation_is_second_order():
    """Sinusoidal source, so the exact solution is not a polynomial and the
    truncation error is actually visible."""
    L, k, T0, q0 = 1e-3, 150.0, 300.0, 1e9
    errs = []
    for n in (10, 20, 40, 80):
        g, s = _bar(n, L, k)
        X, _, _ = g.node_coords()
        q = q0 * np.sin(np.pi * X.ravel() / L) * node_volume_vector(g)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = s.steady(q, bc=BoundarySpec(xlo=Dirichlet(T0), xhi=Dirichlet(T0)))
        T = r.fields["T"][0].reshape(g.shape_nodes)[:, 0, 0]
        ana = T0 + q0 * L ** 2 / (np.pi ** 2 * k) * np.sin(np.pi * g.xn / L)
        errs.append(np.abs(T - ana).max())
    for lo, hi in zip(errs[:-1], errs[1:]):
        assert 3.6 < lo / hi < 4.4, f"expected O(h^2), got ratio {lo / hi}"


def test_slab_thermal_resistance():
    Lz, k, A = 2e-4, 150.0, 1e-4 * 1e-4
    g = RectilinearGrid(np.linspace(0, 1e-4, 9), np.linspace(0, 1e-4, 9),
                        np.linspace(0, Lz, 25))
    s = ThermalSolver(g, np.full(g.shape_cells, k))
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    r = s.steady(np.zeros(g.n_nodes),
                 bc=BoundarySpec(zlo=Dirichlet(300.0), zhi=Dirichlet(301.0)))
    Q = float((s.K @ r.fields["T"][0])[nid[:, :, -1].ravel()].sum())
    assert 1.0 / Q == pytest.approx(Lz / (k * A), rel=1e-9)


def test_convective_wall_gives_the_expected_thermal_resistance():
    """R_th = 1/(h A_surface) for an isothermal body."""
    Lc, Wc, Hc, h = 100e-6, 20e-6, 20e-6, 2e4
    g = RectilinearGrid(np.linspace(0, Lc, 21), np.linspace(0, Wc, 5),
                        np.linspace(0, Hc, 5))
    s = ThermalSolver(g, np.full(g.shape_cells, 5e6))   # effectively isothermal
    conv = {w: (h, 300.0) for w in
            ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")}
    P = 1e-3
    q = np.full(g.n_nodes, P / g.n_nodes)
    r = s.steady(q, convection=conv)
    A = 2 * (Lc * Wc + Lc * Hc + Wc * Hc)
    R_sim = (float(r.fields["T"][0].mean()) - 300.0) / P
    assert R_sim == pytest.approx(1.0 / (h * A), rel=1e-5)


def test_fully_adiabatic_problem_warns():
    g, s = _bar(10)
    with pytest.warns(RuntimeWarning, match="adiabatic"):
        s.steady(np.full(g.n_nodes, 1e-6))


def test_thermal_rejects_bad_input():
    g = RectilinearGrid.uniform([(0, 1e-3), (0, 1e-3)], [4, 4])
    with pytest.raises(ValueError):
        ThermalSolver(g, np.zeros(g.shape_cells))          # kappa == 0
    with pytest.raises(ValueError):
        ThermalSolver(g, np.ones((2, 2, 2)))               # wrong shape
    s = ThermalSolver(g, np.ones(g.shape_cells))
    with pytest.raises(ValueError):
        s.transient(0.0, 1.0, 0.1)                         # no rho_cp given


# ==========================================================================
# Joule heating
# ==========================================================================
def test_joule_heating_is_exactly_conservative():
    """sum(q_node) must equal phi^T L_sigma phi to machine precision."""
    import scipy.sparse.linalg as spl

    from fieldspice.operators import (apply_dirichlet, cell_to_edge,
                                      nodal_laplacian)
    g = RectilinearGrid(np.linspace(0, 2e-6, 13), np.linspace(0, 5e-6, 6),
                        np.linspace(0, 4e-6, 5))
    se = cell_to_edge(g, np.full(g.shape_cells, 1e4))
    L = nodal_laplacian(g, se)
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    lo, hi = nid[0].ravel(), nid[-1].ravel()
    A, b = apply_dirichlet(L, np.zeros(g.n_nodes), np.concatenate([lo, hi]),
                           np.concatenate([np.ones(lo.size), np.zeros(hi.size)]))
    phi = spl.spsolve(A.tocsc(), b)
    q = joule_heating_nodes(g, se, phi)
    assert q.sum() == pytest.approx(float(phi @ (L @ phi)), rel=1e-12)
    assert (q >= -1e-30).all(), "Joule heating cannot be negative anywhere"


# ==========================================================================
# Cross-check against a different code path
# ==========================================================================
@pytest.mark.slow
def test_thermal_transient_matches_the_mna_rc_network():
    """The same physical network solved by the circuit engine instead.

    Thermal/electrical analogy: T <-> V, heat flow <-> current, R_th <-> R,
    C_th <-> C. Two entirely separate solvers must agree.
    """
    from fieldspice import sources as S
    from fieldspice.circuit.mna import MNASolver, Netlist

    L, k, rho, cp = 2e-4, 150.0, 2329.0, 700.0
    W = H = 1e-5
    A = W * H
    N, T0, P = 40, 300.0, 0.05
    g = RectilinearGrid(np.linspace(0, L, N + 1), np.linspace(0, W, 2),
                        np.linspace(0, H, 2))
    s = ThermalSolver(g, np.full(g.shape_cells, k),
                      np.full(g.shape_cells, rho * cp))
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    src = nid[0].ravel()
    tau = (L / (k * A)) * (rho * cp * A * L)
    t0, dt = 0.5 * tau, tau / 400

    q0 = np.zeros(g.n_nodes)
    q0[src] = P / src.size

    dx = L / N
    net = Netlist()
    net.add_isource("I1", "0", "n0", S.step(t0, 0.0, P))
    for i in range(N):
        net.add_resistor(f"R{i}", f"n{i}", f"n{i + 1}", dx / (k * A))
    for i in range(N + 1):
        net.add_capacitor(f"C{i}", f"n{i}", "0",
                          rho * cp * A * dx * (0.5 if i in (0, N) else 1.0))
    net.add_vsource("Vsink", f"n{N}", "0", 0.0)
    ms = MNASolver(net)
    rc = ms.transient(t_end=t0 + 6 * tau, dt=dt, method="be")
    v = rc.fields["x"][:, ms.index("n0")]

    rt = s.transient(lambda t: (q0 if t >= t0 else q0 * 0.0),
                     t_end=t0 + 6 * tau, dt=dt, T0=T0,
                     bc=BoundarySpec(xhi=Dirichlet(T0)))
    Tf = rt.scalars["T_max"] - T0
    n = min(len(Tf), len(v))
    assert np.abs(Tf[:n] - v[:n]).max() < 1e-3, "field and MNA disagree"


# ==========================================================================
# Physics toggles
# ==========================================================================
def test_options_default_to_everything_off():
    o = PhysicsOptions()
    assert o.enabled() == ()
    assert o.relaxed_assumptions() == frozenset()
    assert o.remaining_assumptions(("A5", "A6")) == ("A5", "A6")


def test_enabling_self_heating_relaxes_a6():
    o = PhysicsOptions(self_heating=True)
    assert "A6" in o.relaxed_assumptions()
    assert "A6" not in o.remaining_assumptions(("A1a", "A6", "A11"))
    assert "A1a" in o.remaining_assumptions(("A1a", "A6", "A11"))


def test_sigma_temperature_dependence_follows_self_heating():
    assert PhysicsOptions(self_heating=True).sigma_varies_with_T is True
    assert PhysicsOptions().sigma_varies_with_T is False
    assert PhysicsOptions(self_heating=True,
                          temperature_dependent_sigma=False
                          ).sigma_varies_with_T is False


def test_options_validate():
    with pytest.raises(ValueError):
        PhysicsOptions(ambient_temperature=0.0)
    with pytest.raises(ValueError):
        PhysicsOptions(coupling_tolerance=0.0)
    with pytest.raises(ValueError):
        PhysicsOptions(max_coupling_iterations=0)


def test_material_temperature_dependence():
    from fieldspice.materials import get
    si, cu = get("si"), get("cu")
    assert si.kappa_at(300.0) == pytest.approx(si.kappa)
    assert si.kappa_at(500.0) < 0.6 * si.kappa      # falls as T^-1.3
    assert cu.sigma_at(300.0) == pytest.approx(cu.sigma)
    assert cu.sigma_at(400.0) == pytest.approx(cu.sigma / (1 + 3.93e-3 * 100),
                                               rel=1e-9)
    assert cu.tcr > 0 and si.tcr == 0.0


# ==========================================================================
# Coupled electro-thermal
# ==========================================================================
def _lump():
    """A current/voltage-driven bar that is isothermal by construction, so the
    lumped analytic fixed point applies exactly."""
    Lc, Wc, Hc = 100e-6, 20e-6, 20e-6
    sigma0, alpha, h = 1e6, 3.93e-3, 2e4
    g = RectilinearGrid(np.linspace(0, Lc, 21), np.linspace(0, Wc, 5),
                        np.linspace(0, Hc, 5))
    sig = np.full(g.shape_cells, sigma0)
    kap = np.full(g.shape_cells, 5e5)
    A_wall = 2 * (Lc * Wc + Lc * Hc + Wc * Hc)
    R_th = 1.0 / (h * A_wall)
    R0 = Lc / (sigma0 * Wc * Hc)
    conv = {w: (h, 300.0) for w in
            ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")}
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    return g, sig, kap, alpha, R_th, R0, conv, nid


def _solve(term_a, g, sig, kap, alpha, conv, nid, T_limit=1e9, damping=0.6):
    tb = Terminal("b", nid[-1].ravel(), voltage=0.0)
    o = PhysicsOptions(self_heating=True, max_coupling_iterations=3000,
                       coupling_tolerance=1e-10)
    s = ElectroThermalSolver(g, sig, kap, alpha, o)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.solve([term_a, tb], convection=conv, damping=damping,
                       T_limit=T_limit)


@pytest.mark.physics
@pytest.mark.parametrize("V", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_voltage_driven_matches_the_analytic_fixed_point(V):
    """P = V^2/R with R rising: negative feedback, always stable.

    dT solves alpha dT^2 + dT - B = 0 with B = R_th V^2 / R0.
    """
    g, sig, kap, alpha, R_th, R0, conv, nid = _lump()
    r = _solve(Terminal("a", nid[0].ravel(), voltage=V),
               g, sig, kap, alpha, conv, nid)
    B = R_th * V * V / R0
    ana = (-1 + np.sqrt(1 + 4 * alpha * B)) / (2 * alpha)
    assert float(r.scalars["T_rise"][0]) == pytest.approx(ana, rel=1e-5)


@pytest.mark.physics
@pytest.mark.parametrize("frac", [0.2, 0.5, 0.9])
def test_current_driven_matches_the_analytic_fixed_point(frac):
    """P = I^2 R with R rising: positive feedback. dT = A/(1 - A alpha)."""
    g, sig, kap, alpha, R_th, R0, conv, nid = _lump()
    I = frac * runaway_threshold(R_th, R0, alpha)
    r = _solve(Terminal("a", nid[0].ravel(), current=I),
               g, sig, kap, alpha, conv, nid)
    A = R_th * I * I * R0
    assert float(r.scalars["T_rise"][0]) == pytest.approx(A / (1 - A * alpha),
                                                          rel=1e-4)


@pytest.mark.physics
@pytest.mark.parametrize("frac", [1.01, 1.1, 1.5, 3.0])
def test_current_drive_above_threshold_runs_away(frac):
    g, sig, kap, alpha, R_th, R0, conv, nid = _lump()
    I = frac * runaway_threshold(R_th, R0, alpha)
    with pytest.raises(ThermalRunaway):
        _solve(Terminal("a", nid[0].ravel(), current=I),
               g, sig, kap, alpha, conv, nid, T_limit=5000.0)


def test_voltage_drive_is_never_declared_a_runaway():
    """Negative feedback cannot run away. The first iterate overshoots badly
    (13,936 K at V=1 before settling at 2,281 K), and an implementation that
    trips on a single hot iterate gets both the verdict and the mechanism
    wrong."""
    g, sig, kap, alpha, R_th, R0, conv, nid = _lump()
    r = _solve(Terminal("a", nid[0].ravel(), voltage=1.0),
               g, sig, kap, alpha, conv, nid, T_limit=5000.0)
    assert float(r.scalars["T_rise"][0]) == pytest.approx(2280.93, rel=1e-3)
    hist = r.scalars["history"]
    assert hist.max() > 5000.0, "expected a transient overshoot above T_limit"


def test_runaway_threshold_formula():
    assert runaway_threshold(1e3, 1.0, 1e-3) == pytest.approx(1.0)
    assert runaway_threshold(1e3, 1.0, 0.0) == float("inf")   # no feedback
    assert runaway_threshold(1e3, 1.0, -1e-3) == float("inf")  # negative tcr


def test_zero_tcr_makes_the_coupling_one_pass():
    """Without feedback the solve is just electrical-then-thermal."""
    g, sig, kap, alpha, R_th, R0, conv, nid = _lump()
    # Full damping: with no feedback the fixed point is reached in one pass,
    # and the second pass only confirms it. Under-relaxation would still take
    # ~33 iterations to drive a 0.4^n error decay below the 1e-10 K tolerance,
    # which would measure the damping rather than the coupling.
    r = _solve(Terminal("a", nid[0].ravel(), voltage=0.5),
               g, sig, kap, 0.0, conv, nid, damping=1.0)
    P = 0.5 ** 2 / R0
    assert float(r.scalars["T_rise"][0]) == pytest.approx(R_th * P, rel=1e-4)
    assert r.meta["iterations"] <= 3


def test_electrothermal_requires_self_heating_enabled():
    g, sig, kap, alpha, _, _, _, _ = _lump()
    with pytest.raises(ValueError, match="self_heating"):
        ElectroThermalSolver(g, sig, kap, alpha, PhysicsOptions())


def test_result_records_that_a6_no_longer_applies():
    g, sig, kap, alpha, R_th, R0, conv, nid = _lump()
    r = _solve(Terminal("a", nid[0].ravel(), voltage=0.1),
               g, sig, kap, alpha, conv, nid)
    assert "A6" not in r.meta["assumptions"]
    assert "A1a" in r.meta["assumptions"]
    assert "self_heating" in r.meta["physics"]
