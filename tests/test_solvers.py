"""Quantitative validation of the solvers against the closed-form oracle.

Every test here compares against :mod:`fieldspice.reference` with a tolerance
chosen to match the accuracy the reference formula actually has. Where a
reference is itself approximate (Hammerstad-Jensen, the depletion
approximation) the tolerance says so in a comment, because a test demanding
more accuracy than its own oracle possesses is a broken test.
"""

from __future__ import annotations

import numpy as np
import pytest

from fieldspice import reference as ref
from fieldspice import sources as S
from fieldspice.grid import RectilinearGrid, auto_mesh_1d
from fieldspice.units import c0, eps0, thermal_voltage


# ==========================================================================
# Poisson
# ==========================================================================
def test_poisson_parallel_plate():
    from fieldspice.solvers import poisson as P
    r = P.verify_parallel_plate()
    assert r["rel_err"] < 1e-10
    assert r["C"] == pytest.approx(r["C_ref"], rel=1e-10)


def test_poisson_series_dielectric_stack():
    from fieldspice.solvers import poisson as P
    r = P.verify_series_stack()
    assert r["rel_err"] < 1e-10
    # The interface potential is fixed by the ratio of the two capacitances.
    assert r["v_interface"] == pytest.approx(r["v_interface_ref"], rel=1e-9)


def test_capacitance_matrix_is_symmetric_and_conservative():
    from fieldspice.solvers import poisson as P
    r = P.verify_parallel_plate()["report"]
    assert r["asymmetry_rel"] < 1e-9
    # Rows of a Maxwell capacitance matrix sum to zero for a closed system.
    assert r["row_sum_rel"] < 1e-9
    assert r["linear"]["symmetric"] is True


def test_nonlinear_poisson_built_in_potential_sweep():
    """12 doping/ni combinations; the damped Newton must converge in all."""
    from fieldspice.solvers import poisson as P
    r = P.verify_built_in_potential_sweep(verbose=False)
    assert r["max_rel_err"] < 1e-9, r["max_rel_err"]
    assert r["max_iterations"] <= 25
    for case in r["cases"]:
        assert case["rel_err"] < 1e-9


# ==========================================================================
# Drift-diffusion
# ==========================================================================
def test_bernoulli_is_accurate_across_the_whole_real_line():
    """B(x)=x/(exp(x)-1). Naive evaluation is 0/0 at 0 and overflows at 800."""
    from fieldspice.solvers.dd import bernoulli
    x = np.array([-800.0, -40.0, -1.0, -1e-9, 0.0, 1e-9, 1.0, 40.0, 700.0])
    got = np.asarray(bernoulli(x), dtype=float)
    assert np.all(np.isfinite(got))
    assert got[4] == pytest.approx(1.0, rel=1e-15)          # B(0) = 1 exactly
    assert got[0] == pytest.approx(800.0, rel=1e-12)        # B(x) -> -x, x << 0
    assert got[2] == pytest.approx(1.0 / (np.e - 1.0) * 1.0 * -1.0 / -1.0
                                   if False else 1.5819767068693265, rel=1e-12)
    assert got[7] == pytest.approx(40.0 / (np.exp(40.0) - 1.0), rel=1e-12)
    assert got[8] > 0.0                                     # no underflow to zero
    # Monotone decreasing everywhere.
    xs = np.linspace(-50, 50, 4001)
    b = np.asarray(bernoulli(xs), dtype=float)
    assert np.all(np.diff(b) <= 1e-12)


def test_bernoulli_series_branch_matches_expansion():
    from fieldspice.solvers.dd import bernoulli
    for x in (1e-9, -1e-9, 1e-12, -1e-12):
        expected = 1.0 - x / 2.0 + x * x / 12.0
        assert float(np.atleast_1d(bernoulli(np.array([x])))[0]) == \
            pytest.approx(expected, rel=1e-13)


@pytest.mark.slow
@pytest.mark.physics
def test_pn_diode_ideality_at_low_injection():
    """Forward I-V must be exponential with n ~= 1 over several decades."""
    from fieldspice.materials import MaterialMap
    from fieldspice.solvers.base import SolverConfig, Terminal
    from fieldspice.solvers.dd import DriftDiffusionSolver

    L, xj, Na, Nd = 1000e-9, 500e-9, 1e23, 1e23
    g = RectilinearGrid(auto_mesh_1d((0, L), [xj], dx_min=1e-9,
                                     dx_max=15e-9, growth=1.3))
    X, _, _ = g.node_coords()
    dop = np.where(X.ravel() < xj, -Na, Nd)
    sol = DriftDiffusionSolver(g, MaterialMap(g, background="si"), dop,
                               T=300.0, config=SolverConfig(verbose=0))
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    anode = Terminal("anode", nid[0].ravel(), voltage=0.0)
    cath = Terminal("cathode", nid[-1].ravel(), voltage=0.0)

    V = np.arange(0.0, 0.45, 0.05)
    res = sol.iv_curve(anode, list(V), others=[cath])
    I = np.asarray(res.i("anode"), dtype=float)

    m = (V >= 0.10) & (V <= 0.35)
    slope = np.polyfit(V[m], np.log(np.abs(I[m])), 1)[0]
    n_ideality = 1.0 / (slope * thermal_voltage(300.0))
    decades = np.log10(np.abs(I[m]).max() / np.abs(I[m]).min())
    assert decades > 3.0, f"only {decades:.2f} decades of current"
    assert n_ideality == pytest.approx(1.0, abs=0.05), n_ideality


@pytest.mark.slow
@pytest.mark.physics
@pytest.mark.xfail(reason="known open defect: spurious current at zero bias, "
                          "see docs/KNOWN_ISSUES.md", strict=False)
def test_pn_diode_current_vanishes_at_zero_bias():
    """Detailed balance: at V=0 the junction is in equilibrium and J is zero."""
    from fieldspice.materials import MaterialMap
    from fieldspice.solvers.base import SolverConfig, Terminal
    from fieldspice.solvers.dd import DriftDiffusionSolver

    L, xj, Na, Nd = 1000e-9, 500e-9, 1e23, 1e23
    g = RectilinearGrid(auto_mesh_1d((0, L), [xj], dx_min=1e-9,
                                     dx_max=15e-9, growth=1.3))
    X, _, _ = g.node_coords()
    dop = np.where(X.ravel() < xj, -Na, Nd)
    sol = DriftDiffusionSolver(g, MaterialMap(g, background="si"), dop,
                               T=300.0, config=SolverConfig(verbose=0))
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    anode = Terminal("anode", nid[0].ravel(), voltage=0.0)
    cath = Terminal("cathode", nid[-1].ravel(), voltage=0.0)
    res = sol.iv_curve(anode, [0.0, 0.10], others=[cath])
    I = np.asarray(res.i("anode"), dtype=float)
    assert abs(I[0]) < 1e-3 * abs(I[1])


# ==========================================================================
# MNA
# ==========================================================================
def _rc_netlist(method, dt_frac, t0_frac=2.0):
    from fieldspice.circuit.mna import MNASolver, Netlist
    R, C = 1e3, 1e-12
    tau = R * C
    t0 = t0_frac * tau
    n = Netlist()
    n.add_vsource("V1", "in", "0", S.step(t0, 0.0, 1.0))
    n.add_resistor("R1", "in", "out", R)
    n.add_capacitor("C1", "out", "0", C)
    s = MNASolver(n)
    res = s.transient(t_end=t0 + 8 * tau, dt=tau * dt_frac, method=method)
    v = res.fields["x"][:, s.index("out")]
    m = res.t >= t0
    return np.abs(v[m] - ref.rc_step(res.t[m] - t0, R, C)).max()


@pytest.mark.parametrize("method", ["be", "trap", "gear2"])
def test_mna_rc_step_matches_analytic(method):
    assert _rc_netlist(method, 1 / 200) < 1e-2


def test_mna_transient_starts_from_the_dc_operating_point():
    """SPICE semantics: .tran begins at the DC solution, not at zero."""
    from fieldspice.circuit.mna import MNASolver, Netlist
    R, C = 1e3, 1e-12
    n = Netlist()
    n.add_vsource("V1", "in", "0", 1.0)
    n.add_resistor("R1", "in", "out", R)
    n.add_capacitor("C1", "out", "0", C)
    s = MNASolver(n)
    res = s.transient(t_end=10 * R * C, dt=R * C / 100, method="be")
    v = res.fields["x"][:, s.index("out")]
    assert v[0] == pytest.approx(1.0, abs=1e-6)   # already charged
    assert np.ptp(v) < 1e-6                        # and stays there


def _order_with_smooth_source(method):
    """Measure the integrator's true order using a C-infinity excitation.

    A step edge quantised to the time grid injects its own O(dt) error, which
    masks the method order: with a step, backward Euler, trapezoidal and Gear-2
    all measure as first order. A sine has no discontinuity, so the scheme's own
    order is what shows up.
    """
    from fieldspice.circuit.mna import MNASolver, Netlist
    R, C = 1e3, 1e-12
    tau = R * C
    f = 1.0 / (20 * tau)
    w = 2 * np.pi * f
    errs = []
    for frac in (1 / 10, 1 / 20, 1 / 40, 1 / 80):
        n = Netlist()
        n.add_vsource("V1", "in", "0", S.sine(f, 1.0))
        n.add_resistor("R1", "in", "out", R)
        n.add_capacitor("C1", "out", "0", C)
        s = MNASolver(n)
        res = s.transient(t_end=3 / f, dt=tau * frac, method=method)
        v = res.fields["x"][:, s.index("out")]
        t = res.t
        A = 1.0 / np.sqrt(1 + (w * R * C) ** 2)
        ph = np.arctan(w * R * C)
        ana = A * np.sin(w * t - ph) + (-A * np.sin(-ph)) * np.exp(-t / tau)
        m = t > 2 * tau
        errs.append(np.abs(v[m] - ana[m]).max())
    return [a / b for a, b in zip(errs[:-1], errs[1:])]


def test_backward_euler_is_first_order():
    assert all(1.8 < r < 2.2 for r in _order_with_smooth_source("be"))


def test_trapezoidal_is_second_order():
    assert all(3.5 < r < 4.5 for r in _order_with_smooth_source("trap"))


def test_gear2_is_second_order():
    assert all(3.5 < r < 4.5 for r in _order_with_smooth_source("gear2"))


@pytest.mark.parametrize("R,tag", [(1.0, "under"), (63.2455532, "crit"), (1e4, "over")])
def test_mna_rlc_all_damping_regimes(R, tag):
    """Requires R, L and C to all be right simultaneously."""
    from fieldspice.circuit.mna import MNASolver, Netlist
    L, C = 1e-9, 1e-12
    T = 2e-8 if tag != "over" else 6e-7
    t0 = 0.05 * T
    n = Netlist()
    n.add_vsource("V1", "in", "0", S.step(t0, 0.0, 1.0))
    n.add_resistor("R1", "in", "a", R)
    n.add_inductor("L1", "a", "out", L)
    n.add_capacitor("C1", "out", "0", C)
    s = MNASolver(n)
    res = s.transient(t_end=T, dt=T / 40000, method="trap")
    v = res.fields["x"][:, s.index("out")]
    m = res.t >= t0
    ana = ref.rlc_step(res.t[m] - t0, R, L, C)
    assert v[m].max() == pytest.approx(ana.max(), rel=2e-3)
    assert np.abs(v[m] - ana).max() < 1e-2


def test_spice_meg_versus_m_suffix():
    """SPICE's classic trap: M is milli, MEG is mega. 1e9 apart."""
    from fieldspice.circuit.mna import parse_value
    assert parse_value("1MEG") == pytest.approx(1e6)
    assert parse_value("1M") == pytest.approx(1e-3)
    assert parse_value("1k") == pytest.approx(1e3)
    assert parse_value("1u") == pytest.approx(1e-6)
    assert parse_value("2.5n") == pytest.approx(2.5e-9)


def test_resistive_divider_dc():
    from fieldspice.circuit.mna import MNASolver, Netlist
    n = Netlist()
    n.add_vsource("V1", "in", "0", 3.0)
    n.add_resistor("R1", "in", "mid", 2e3)
    n.add_resistor("R2", "mid", "0", 1e3)
    op = MNASolver(n).dc()
    assert op["mid"] == pytest.approx(1.0, rel=1e-9)


# ==========================================================================
# Devices
# ==========================================================================
def test_limexp_never_overflows():
    from fieldspice.circuit.devices import limexp
    for x in (0.0, 10.0, 40.0, 100.0, 1e4):
        y = float(np.atleast_1d(limexp(x))[0])
        assert np.isfinite(y) and y > 0


def test_diode_matches_shockley():
    from fieldspice.circuit.devices import Diode
    d = Diode("D1", "a", "0", isat=1e-15, n=1.0, gmin=0.0)
    V = np.linspace(0.1, 0.6, 20)
    got = np.array([d.current(float(v)) for v in V])
    exp = ref.shockley_diode(V, 1e-15, 1.0)
    assert np.allclose(got, exp, rtol=0.05)


def test_subthreshold_tft_spans_many_decades():
    """The analog-exp cell: must hold its slope over 8+ decades."""
    from fieldspice.circuit.devices import SubthresholdTFT
    t = SubthresholdTFT("M1", "d", "g", "s", i0=1e-12, vth=0.5, n=1.0,
                        w=1e-6, l=1e-6, gmin=0.0)
    Vg = np.linspace(0.0, 0.5, 60)
    I = np.array([t.drain_current(float(v), 1.0) for v in Vg])
    I = I[I > 0]
    assert np.log10(I.max() / I.min()) > 8.0
    slope = np.polyfit(Vg[-len(I):], np.log10(I), 1)[0]
    assert 1.0 / slope == pytest.approx(ref.subthreshold_slope(), rel=0.05)


def test_mosfet_level1_matches_square_law():
    from fieldspice.circuit.devices import MOSFETL1
    m = MOSFETL1("M1", "d", "g", "s", "b", vth=0.7, kp=100e-6, w=10e-6,
                 l=1e-6, subthreshold=False, gmin=0.0)
    k = 100e-6 * 10.0
    for vgs, vds in ((1.2, 0.1), (1.2, 2.0), (0.5, 1.0), (2.0, 1.0)):
        got = m.drain_current(vgs, vds)
        exp = float(ref.mosfet_square_law(np.array(vgs), np.array(vds), 0.7, k))
        assert got == pytest.approx(exp, rel=0.05, abs=1e-9)


# ==========================================================================
# FDTD
# ==========================================================================
def test_fdtd_refuses_an_unstable_timestep():
    from fieldspice.solvers.fdtd import FDTDSolver
    g = RectilinearGrid(np.linspace(0, 1e-2, 51))
    s = FDTDSolver(g, np.full(g.shape_cells, eps0))
    dt = s.stable_dt()
    assert dt > 0
    with pytest.raises(ValueError):
        s.solve(sources=None, t_end=10 * dt, dt=dt * 5.0)


@pytest.mark.slow
@pytest.mark.physics
def test_fdtd_pulse_propagates_at_the_speed_of_light():
    """First-arrival timing is the clean measurement; peak timing is
    quantisation-limited to +-1 step."""
    from fieldspice.solvers.fdtd import CurrentSource, FDTDSolver
    N, dx = 600, 1e-4
    g = RectilinearGrid(np.linspace(0, N * dx, N + 1))
    s = FDTDSolver(g, np.full(g.shape_cells, eps0))
    dt = s.stable_dt(safety=0.99)
    t0, tau = 60 * dt, 12 * dt
    src = CurrentSource(edges=np.atleast_1d(g.edge_index(0, 100, 0, 0)),
                        waveform=lambda t: np.exp(-((t - t0) / tau) ** 2))
    res = s.solve(sources=[src], t_end=700 * dt, dt=dt,
                  store=("e",), store_every=1)
    E = res.fields["e"]

    def arrival(cell):
        tr = np.abs(E[:, g.edge_index(0, cell, 0, 0)])
        return int(np.argmax(tr > 1e-3 * tr.max()))

    a2, a3, a4 = arrival(200), arrival(300), arrival(400)
    steps_per_100 = np.array([a3 - a2, a4 - a3], dtype=float)
    courant = dt * c0 / dx
    expected = 100.0 / courant
    assert np.allclose(steps_per_100, expected, rtol=0.03), steps_per_100


@pytest.mark.slow
@pytest.mark.xfail(reason="known open defect: CPML reflection is -0.6 dB, "
                          "see docs/KNOWN_ISSUES.md", strict=False)
def test_fdtd_pml_absorbs():
    from fieldspice.solvers.fdtd import FDTDSolver
    r = FDTDSolver.measure_pml_reflection(n_cells=200, dx=2e-5, thickness=10)
    assert r["reflection_dB"] < -40.0, r
