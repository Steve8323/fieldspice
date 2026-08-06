"""Validation of the quasi-static layer: EQS, AC, extraction, field-circuit coupling.

These are the modules that make fieldspice a circuit tool rather than a field
viewer, so they are tested against closed-form circuit answers throughout.
"""

from __future__ import annotations

import numpy as np
import pytest

from fieldspice import extraction as EX
from fieldspice import reference as ref
from fieldspice import sources as S
from fieldspice.grid import RectilinearGrid
from fieldspice.solvers.base import SolverConfig, Terminal
from fieldspice.units import c0, eps0, mu0


# ==========================================================================
# helpers
# ==========================================================================
def _rc_stack(n=60):
    """Lossy dielectric stack: conductive left half, insulating right half."""
    d, W, H = 4e-6, 3e-6, 2e-6
    sigma1, er1, er2 = 1e3, 4.0, 2.0
    g = RectilinearGrid(np.linspace(0, d, n + 1), np.linspace(0, W, 4),
                        np.linspace(0, H, 3))
    left = g.xc < d / 2
    sig = np.zeros(g.shape_cells)
    sig[left] = sigma1
    eps = np.empty(g.shape_cells)
    eps[left] = er1 * eps0
    eps[~left] = er2 * eps0
    A = W * H
    R1 = (d / 2) / (sigma1 * A)
    C1 = ref.parallel_plate_capacitance(A, d / 2, er1)
    C2 = ref.parallel_plate_capacitance(A, d / 2, er2)
    return g, eps, sig, R1, C1, C2, n


# ==========================================================================
# EQS
# ==========================================================================
def _eqs_run(steps_per_tau):
    from fieldspice.solvers.eqs import EQSSolver
    g, eps, sig, R1, C1, C2, n = _rc_stack()
    tau = R1 * (C1 + C2)
    v0 = C1 / (C1 + C2)
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    s = EQSSolver(g, eps, sig, SolverConfig(verbose=0))
    hi = Terminal("hi", nid[0].ravel(), voltage=1.0)
    lo = Terminal("lo", nid[-1].ravel(), voltage=0.0)
    r = s.solve([hi, lo], t_end=8 * tau, dt=tau / steps_per_tau, store=True)
    vm = r.fields["phi"][:, nid[n // 2, 1, 1]]
    ts = r.scalars["t_stored"]
    ana = ref.rc_step(ts, R1, C1 + C2, 1.0, v0)
    return r, vm, ana, tau, v0


def test_eqs_initial_condition_is_the_capacitive_divider():
    """At t=0+ no charge has moved: the stack is a capacitive divider."""
    _, vm, _, _, v0 = _eqs_run(50)
    assert v0 == pytest.approx(2.0 / 3.0, rel=1e-12)
    assert vm[0] == pytest.approx(v0, rel=1e-9)


def test_eqs_matches_analytic_rc_transient():
    _, vm, ana, _, _ = _eqs_run(400)
    assert np.abs(vm - ana).max() < 2e-4


def test_eqs_backward_euler_is_first_order():
    errs = [np.abs(v - a).max() for _, v, a, _, _ in
            (_eqs_run(s) for s in (50, 100, 200, 400))]
    # Measured in docs/CONTRACTS.md: 1.22e-3 -> 6.11e-4 -> 3.06e-4 -> 1.53e-4.
    assert errs[0] == pytest.approx(1.216e-3, rel=0.05)
    for lo, hi in zip(errs[:-1], errs[1:]):
        assert 1.8 < lo / hi < 2.2


def test_eqs_factorises_once_for_a_constant_timestep():
    """Refactorising per step is the difference between usable and not."""
    r, *_ = _eqs_run(100)
    assert r.meta["n_factorisations"] == 1


def test_eqs_terminal_currents_conserve():
    r, *_ = _eqs_run(100)
    total = r.i("hi") + r.i("lo")
    scale = max(np.abs(r.i("hi")).max(), 1e-30)
    assert np.abs(total).max() / scale < 1e-8


def test_eqs_records_its_assumptions():
    r, *_ = _eqs_run(50)
    assert "A1a" in r.meta["assumptions"]


def test_eqs_steady_state_is_the_resistive_solution():
    from fieldspice.solvers.eqs import EQSSolver
    d, W, H, sigma = 2e-6, 5e-6, 4e-6, 1e4
    g = RectilinearGrid(np.linspace(0, d, 16), np.linspace(0, W, 6),
                        np.linspace(0, H, 5))
    s = EQSSolver(g, np.full(g.shape_cells, 3.9 * eps0),
                  np.full(g.shape_cells, sigma))
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    hi = Terminal("hi", nid[0].ravel(), voltage=1.0)
    lo = Terminal("lo", nid[-1].ravel(), voltage=0.0)
    r = s.steady_state([hi, lo])
    R = 1.0 / abs(float(r.i("hi")[0]))
    assert R == pytest.approx(ref.slab_resistance(d, W * H, sigma), rel=1e-9)


def test_eqs_rejects_relative_permittivity():
    from fieldspice.solvers.eqs import EQSSolver
    g = RectilinearGrid.uniform([(0, 1e-6), (0, 1e-6)], [4, 4])
    with pytest.raises(ValueError):
        EQSSolver(g, np.full(g.shape_cells, 3.9), np.zeros(g.shape_cells))
    with pytest.raises(ValueError):
        EQSSolver(g, np.full(g.shape_cells, eps0), np.full(g.shape_cells, -1.0))


# ==========================================================================
# AC
# ==========================================================================
def _lossy_slab():
    d, W, H, er, sigma = 2e-6, 5e-6, 4e-6, 3.9, 1e-2
    g = RectilinearGrid(np.linspace(0, d, 16), np.linspace(0, W, 6),
                        np.linspace(0, H, 5))
    eps = np.full(g.shape_cells, er * eps0)
    sig = np.full(g.shape_cells, sigma)
    C = ref.parallel_plate_capacitance(W * H, d, er)
    G = sigma * W * H / d
    return g, eps, sig, C, G


@pytest.mark.parametrize("freq", [1e3, 1e6, 1e9, 1e12])
def test_ac_admittance_is_g_plus_jwc(freq):
    from fieldspice.solvers.ac import ACSolver
    g, eps, sig, C, G = _lossy_slab()
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    t1 = Terminal("hi", nid[0].ravel())
    t2 = Terminal("lo", nid[-1].ravel())
    Y = ACSolver(g, eps, sig).admittance_matrix([t1, t2], [freq])
    assert Y[0, 0, 0] == pytest.approx(G + 2j * np.pi * freq * C, rel=1e-9)


def test_ac_admittance_is_reciprocal():
    from fieldspice.solvers.ac import ACSolver
    g, eps, sig, _, _ = _lossy_slab()
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    t1 = Terminal("hi", nid[0].ravel())
    t2 = Terminal("lo", nid[-1].ravel())
    Y = ACSolver(g, eps, sig).admittance_matrix([t1, t2], [1e9])
    assert abs(Y[0, 0, 1] - Y[0, 1, 0]) < 1e-12 * abs(Y[0, 0, 0])


def test_ac_dc_limit_matches_the_resistive_solve():
    from fieldspice.solvers.ac import ACSolver
    g, eps, sig, _, G = _lossy_slab()
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    t1 = Terminal("hi", nid[0].ravel())
    t2 = Terminal("lo", nid[-1].ravel())
    Y = ACSolver(g, eps, sig).admittance_matrix([t1, t2], [0.0])
    assert Y[0, 0, 0].real == pytest.approx(G, rel=1e-9)
    assert abs(Y[0, 0, 0].imag) < 1e-20


# ==========================================================================
# Extraction
# ==========================================================================
def _plate_line():
    er, W, Hgap = 4.0, 200e-6, 50e-6
    g = RectilinearGrid(np.linspace(0, Hgap, 40), np.linspace(0, W, 60), None)
    eps = np.full(g.shape_cells, er * eps0)
    sig = np.zeros(g.shape_cells)
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    t1 = Terminal("sig", nid[0].ravel())
    t2 = Terminal("gnd", nid[-1].ravel())
    return g, eps, sig, [t1, t2], er, W, Hgap


def test_rlgc_matches_the_parallel_plate_line():
    g, eps, sig, terms, er, W, Hgap = _plate_line()
    r = EX.rlgc_2d(g, eps, sig, terms, reference="gnd")
    assert r["C"][0, 0] == pytest.approx(er * eps0 * W / Hgap, rel=1e-9)
    assert r["L"][0, 0] == pytest.approx(mu0 * Hgap / W, rel=1e-9)
    assert r["Z0"][0] == pytest.approx(
        np.sqrt((mu0 * Hgap / W) / (er * eps0 * W / Hgap)), rel=1e-9)
    assert r["v_p"][0] == pytest.approx(c0 / np.sqrt(er), rel=1e-9)
    assert r["eps_eff"][0] == pytest.approx(er, rel=1e-9)


def test_lc_identity_holds_for_a_homogeneous_line():
    """L C = mu eps I exactly for a TEM mode: the strongest self-check."""
    g, eps, sig, terms, er, _, _ = _plate_line()
    r = EX.rlgc_2d(g, eps, sig, terms, reference="gnd")
    assert EX.check_lc_identity(r, eps_r=er) < 1e-9


def test_rlgc_needs_a_return_conductor():
    g, eps, sig, terms, *_ = _plate_line()
    with pytest.raises(ValueError):
        EX.rlgc_2d(g, eps, sig, terms[:1])
    with pytest.raises(ValueError):
        EX.rlgc_2d(g, eps, sig, terms, reference="nope")


def test_rlgc_rejects_a_3d_grid():
    g = RectilinearGrid.uniform([(0, 1e-6), (0, 1e-6), (0, 1e-6)], [4, 4, 4])
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    with pytest.raises(ValueError):
        EX.rlgc_2d(g, np.full(g.shape_cells, eps0), np.zeros(g.shape_cells),
                   [Terminal("a", nid[0].ravel()), Terminal("b", nid[-1].ravel())])


def test_maxwell_to_spice_capacitance_conversion():
    c = np.array([[3.0, -1.0, -2.0], [-1.0, 4.0, -3.0], [-2.0, -3.0, 5.0]])
    s = EX.to_spice_matrix(c)
    assert s[0, 1] == pytest.approx(1.0)
    assert s[0, 0] == pytest.approx(0.0)      # row sum: fully coupled, none to gnd
    with pytest.raises(ValueError):
        EX.to_spice_matrix(np.zeros((2, 3)))


def test_s_parameters_of_a_matched_load_are_zero():
    """A 50-ohm shunt at each port on a 50-ohm reference reflects nothing."""
    Y = np.diag([1 / 50.0, 1 / 50.0])
    S_ = EX.s_parameters(Y, 50.0)
    assert np.abs(S_).max() < 1e-12


def test_s_parameters_are_passive():
    Y = np.array([[1 / 50 + 1e-6, -1e-6], [-1e-6, 1 / 50 + 1e-6]])
    S_ = EX.s_parameters(Y, 50.0)
    assert np.linalg.svd(S_, compute_uv=False).max() <= 1.0 + 1e-12


# ==========================================================================
# Field-circuit coupling
# ==========================================================================
def _field_resistor(sigma=1e4):
    d, W, H = 2e-6, 5e-6, 4e-6
    g = RectilinearGrid(np.linspace(0, d, 16), np.linspace(0, W, 6),
                        np.linspace(0, H, 5))
    sig = np.full(g.shape_cells, sigma)
    eps = np.full(g.shape_cells, 3.9 * eps0)
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    terms = [Terminal("a", nid[0].ravel()), Terminal("b", nid[-1].ravel())]
    return g, eps, sig, terms, d / (sigma * W * H)


def test_coupled_resistive_divider_is_exact():
    from fieldspice.circuit.coupling import FieldRegion
    from fieldspice.circuit.mna import MNASolver, Netlist
    g, eps, sig, terms, Rf = _field_resistor()
    fr = FieldRegion("FR", g, eps, sig, terms, circuit_nodes=["mid", "0"])
    n = Netlist()
    n.add_vsource("V1", "in", "0", 1.0)
    n.add_resistor("R1", "in", "mid", Rf)
    n.add_device(fr)
    op = MNASolver(n).dc()
    assert op["mid"] == pytest.approx(0.5, abs=1e-9)


def test_reduced_admittance_reproduces_the_field_resistance():
    from fieldspice.circuit.coupling import FieldRegion
    g, eps, sig, terms, Rf = _field_resistor()
    fr = FieldRegion("FR", g, eps, sig, terms, circuit_nodes=["a", "b"])
    Y = fr.terminal_admittance(None)
    assert 1.0 / Y[0, 0] == pytest.approx(Rf, rel=1e-9)
    assert fr.asymmetry < 1e-12


def test_reduced_admittance_reproduces_the_field_capacitance():
    """For an insulating region, Y*dt must equal C exactly."""
    from fieldspice.circuit.coupling import FieldRegion
    d, W, H, er = 2e-6, 5e-6, 4e-6, 3.9
    g = RectilinearGrid(np.linspace(0, d, 16), np.linspace(0, W, 6),
                        np.linspace(0, H, 5))
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    terms = [Terminal("a", nid[0].ravel()), Terminal("b", nid[-1].ravel())]
    fr = FieldRegion("FR", g, np.full(g.shape_cells, er * eps0),
                     np.zeros(g.shape_cells), terms, circuit_nodes=["a", "b"])
    dt = 1e-12
    Y = fr.terminal_admittance(dt)
    assert Y[0, 0] * dt == pytest.approx(
        ref.parallel_plate_capacitance(W * H, d, er), rel=1e-9)


def test_capacitive_field_region_is_an_open_circuit_at_dc():
    """sigma == 0 means no dc conductance. Correct physics, not a degenerate case."""
    from fieldspice.circuit.coupling import FieldRegion
    g = RectilinearGrid.uniform([(0, 2e-6), (0, 5e-6), (0, 4e-6)], [8, 4, 3])
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    terms = [Terminal("a", nid[0].ravel()), Terminal("b", nid[-1].ravel())]
    fr = FieldRegion("FR", g, np.full(g.shape_cells, 3.9 * eps0),
                     np.zeros(g.shape_cells), terms, circuit_nodes=["a", "b"])
    assert np.abs(fr.terminal_admittance(None)).max() == 0.0


def test_coupled_rc_transient_matches_analytic():
    from fieldspice.circuit.coupling import FieldRegion
    from fieldspice.circuit.mna import MNASolver, Netlist
    d, W, H, er = 2e-6, 5e-6, 4e-6, 3.9
    g = RectilinearGrid(np.linspace(0, d, 16), np.linspace(0, W, 6),
                        np.linspace(0, H, 5))
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    terms = [Terminal("a", nid[0].ravel()), Terminal("b", nid[-1].ravel())]
    Cf = ref.parallel_plate_capacitance(W * H, d, er)
    R, = (1e6,)
    tau = R * Cf
    t0 = 2 * tau

    errs = []
    for spt in (50, 200, 800):
        fr = FieldRegion("FR", g, np.full(g.shape_cells, er * eps0),
                         np.zeros(g.shape_cells), terms,
                         circuit_nodes=["out", "0"])
        n = Netlist()
        n.add_vsource("V1", "in", "0", S.step(t0, 0.0, 1.0))
        n.add_resistor("R1", "in", "out", R)
        n.add_device(fr)
        s = MNASolver(n)
        res = s.transient(t_end=t0 + 8 * tau, dt=tau / spt, method="be")
        v = res.fields["x"][:, s.index("out")]
        m = res.t >= t0
        errs.append(np.abs(v[m] - ref.rc_step(res.t[m] - t0, R, Cf)).max())
    assert errs[-1] < 3e-3
    for lo, hi in zip(errs[:-1], errs[1:]):
        assert 3.0 < lo / hi < 5.0        # first order in dt, 4x steps -> 4x better


def test_reset_state_must_not_destroy_integration_history():
    """MNA calls reset_state() every Newton solve; clearing phi there breaks
    the transient into a bare resistive divider."""
    from fieldspice.circuit.coupling import FieldRegion
    g, eps, sig, terms, _ = _field_resistor()
    fr = FieldRegion("FR", g, eps, sig, terms, circuit_nodes=["a", "b"])
    fr.phi = np.ones(g.n_nodes) * 0.25
    fr.reset_state()
    assert np.allclose(fr.phi, 0.25)
    fr.reset_history()
    assert np.allclose(fr.phi, 0.0)


def test_overlapping_terminals_are_rejected():
    from fieldspice.circuit.coupling import FieldRegion
    g, eps, sig, terms, _ = _field_resistor()
    bad = [terms[0], Terminal("dup", terms[0].nodes)]
    with pytest.raises(ValueError):
        FieldRegion("FR", g, eps, sig, bad, circuit_nodes=["a", "b"])
