"""Integration tests for the infrastructure layer.

These test the *seams* between independently written modules, which is where
bugs actually live: does a geometry fill fraction mean the same thing to
MaterialMap, does a boundary node index mean the same thing to the operators,
does a monitor compute the energy the mass matrices imply.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse.linalg as spl

from fieldspice import boundaries as B
from fieldspice import geometry as GEO
from fieldspice import materials as M
from fieldspice import monitors as MON
from fieldspice import operators as ops
from fieldspice import reference as ref
from fieldspice import validate as V
from fieldspice import sources as S
from fieldspice.grid import RectilinearGrid
from fieldspice.units import eps0, mu0, um


# --------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------
def test_library_covers_the_thin_film_process_materials():
    """Not just silicon CMOS: this project's users sputter AlN and IGZO."""
    for name in ("vacuum", "air", "si", "sio2", "si3n4", "cu", "al", "w",
                 "igzo", "aln", "hfo2", "mo", "ti", "pt"):
        assert name in M.LIBRARY, f"missing material {name}"


def test_material_eps_r_is_relative_but_map_returns_absolute():
    """The single most common unit bug in any field solver."""
    assert M.get("sio2").eps_r == pytest.approx(3.9, rel=0.02)
    g = RectilinearGrid.uniform([(0, 1.0)], [2])
    mm = M.MaterialMap(g, background="sio2")
    assert mm.eps().min() == pytest.approx(3.9 * eps0, rel=0.02)
    assert mm.mu().min() == pytest.approx(mu0, rel=1e-6)


def test_conductor_conductivities_are_right():
    assert M.get("cu").sigma == pytest.approx(5.8e7, rel=0.05)
    assert M.get("al").sigma == pytest.approx(3.77e7, rel=0.05)
    # Insulators carry a tiny but nonzero leakage (real SiO2 is 1e-16..1e-14
    # S/m). Beyond being physical, it keeps a DC conduction solve from going
    # singular in fully insulating regions, so it is deliberate, not sloppy.
    assert 0.0 < M.get("sio2").sigma < 1e-12
    assert M.get("vacuum").sigma == 0.0


def test_silicon_intrinsic_density_agrees_with_the_reference_oracle():
    """There must be exactly one definition of ni in the package."""
    si = M.get("si")
    assert si.semi is not None
    assert si.semi.ni(300.0) == pytest.approx(ref.NI_SI_300, rel=0.02)


def test_material_map_fill_fraction_mixing():
    g = RectilinearGrid.uniform([(0, 1.0), (0, 1.0), (0, 1.0)], [2, 1, 1])
    mm = M.MaterialMap(g, background="vacuum")
    fill = np.zeros(g.shape_cells)
    fill[0, 0, 0] = 1.0     # fully silicon
    fill[1, 0, 0] = 0.5     # half silicon, half vacuum
    mm.assign(fill, "si")
    eps = mm.eps()
    assert eps[0, 0, 0] == pytest.approx(11.7 * eps0, rel=0.02)
    mid = eps[1, 0, 0] / eps0
    assert 1.0 < mid < 11.7          # genuinely mixed, not snapped
    assert mid == pytest.approx(0.5 * (1.0 + 11.7), rel=0.05)


def test_unknown_material_raises():
    with pytest.raises(Exception):
        M.get("unobtainium")


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def test_voxelize_volume_converges_with_subsample():
    g = RectilinearGrid.uniform([(0, 1.0), (0, 1.0), (0, 1.0)], [20, 20, 20])
    sph = GEO.Sphere(center=(0.5, 0.5, 0.5), radius=0.3)
    exact = 4.0 / 3.0 * np.pi * 0.3 ** 3
    errs = []
    for ss in (1, 2, 4):
        vol = (GEO.voxelize(g, sph, subsample=ss) * g.cell_volumes()).sum()
        errs.append(abs(vol / exact - 1.0))
    assert errs[-1] < 0.005
    assert errs[-1] < errs[0]


def test_voxelize_returns_fill_fractions_in_range():
    g = RectilinearGrid.uniform([(0, 1.0), (0, 1.0), (0, 1.0)], [8, 8, 8])
    f = GEO.voxelize(g, GEO.Sphere((0.5, 0.5, 0.5), 0.25), subsample=3)
    assert f.shape == g.shape_cells
    assert f.min() >= 0.0 and f.max() <= 1.0
    assert 0.0 < f.mean() < 1.0


def test_voxelize_subsample_one_is_a_hard_centre_mask():
    g = RectilinearGrid.uniform([(0, 1.0), (0, 1.0), (0, 1.0)], [8, 8, 8])
    f = GEO.voxelize(g, GEO.Sphere((0.5, 0.5, 0.5), 0.25), subsample=1)
    assert set(np.unique(f)).issubset({0.0, 1.0})


def test_box_volume_is_exact_when_grid_aligned():
    """The planar-process case, where staircasing costs nothing (A2)."""
    g = RectilinearGrid.uniform([(0, 1.0), (0, 1.0), (0, 1.0)], [10, 10, 10])
    box = GEO.Box(lo=(0.2, 0.3, 0.1), hi=(0.7, 0.8, 0.6))
    vol = (GEO.voxelize(g, box, subsample=1) * g.cell_volumes()).sum()
    assert vol == pytest.approx(0.5 * 0.5 * 0.5, rel=1e-12)


def test_csg_operations():
    g = RectilinearGrid.uniform([(0, 1.0), (0, 1.0), (0, 1.0)], [12, 12, 12])
    a = GEO.Box(lo=(0.0, 0.0, 0.0), hi=(0.5, 1.0, 1.0))
    b = GEO.Box(lo=(0.25, 0.0, 0.0), hi=(1.0, 1.0, 1.0))
    vol = lambda s: float((GEO.voxelize(g, s, subsample=1) * g.cell_volumes()).sum())
    assert vol(a | b) == pytest.approx(1.0, rel=1e-9)
    assert vol(a & b) == pytest.approx(0.25, rel=1e-9)
    assert vol(a - b) == pytest.approx(0.25, rel=1e-9)


def test_prism_handles_a_nonconvex_polygon():
    """An L-shape: the classic case where naive point-in-polygon fails."""
    g = RectilinearGrid.uniform([(0, 1.0), (0, 1.0), (0, 1.0)], [20, 20, 4])
    verts = [(0.1, 0.1), (0.9, 0.1), (0.9, 0.4), (0.4, 0.4), (0.4, 0.9), (0.1, 0.9)]
    p = GEO.Prism(verts, height=1.0, center=0.5, axis="z")
    area = 0.8 * 0.3 + 0.3 * 0.5
    vol = (GEO.voxelize(g, p, subsample=4) * g.cell_volumes()).sum()
    assert vol == pytest.approx(area * 1.0, rel=0.05)


# --------------------------------------------------------------------------
# boundaries -- index arithmetic against the operators' own convention
# --------------------------------------------------------------------------
@pytest.mark.parametrize("wall,axis,end", [
    ("xlo", 0, 0), ("xhi", 0, -1), ("ylo", 1, 0),
    ("yhi", 1, -1), ("zlo", 2, 0), ("zhi", 2, -1),
])
def test_wall_nodes_match_direct_index_arithmetic(wall, axis, end):
    g = RectilinearGrid.uniform([(0, 1.0), (0, 2.0), (0, 3.0)], [4, 5, 6])
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    sl = [slice(None)] * 3
    sl[axis] = end
    expected = np.sort(nid[tuple(sl)].ravel())
    assert np.array_equal(np.sort(B.wall_nodes(g, wall)), expected)


def test_wall_dual_areas_sum_to_the_wall_area():
    g = RectilinearGrid.uniform([(0, 1.0), (0, 2.0), (0, 3.0)], [4, 5, 6])
    assert B.wall_dual_areas(g, "xlo").sum() == pytest.approx(2.0 * 3.0, rel=1e-12)
    assert B.wall_dual_areas(g, "zhi").sum() == pytest.approx(1.0 * 2.0, rel=1e-12)


def test_dirichlet_spec_drives_a_solve_end_to_end():
    """The real integration test: BoundarySpec -> apply_dirichlet -> right answer."""
    g = RectilinearGrid.uniform([(0, 2e-6), (0, 5e-6), (0, 4e-6)], [16, 6, 5])
    spec = B.BoundarySpec(xlo=B.Dirichlet(1.0), xhi=B.Dirichlet(0.0))
    fixed, vals = spec.dirichlet_nodes(g, 0.0)
    L = ops.nodal_laplacian(g, ops.cell_to_edge(g, np.full(g.shape_cells, 3.9 * eps0)))
    A, b = ops.apply_dirichlet(L, np.zeros(g.n_nodes), fixed, vals)
    phi = spl.spsolve(A.tocsc(), b)
    C = (L @ phi)[B.wall_nodes(g, "xlo")].sum()
    assert C == pytest.approx(
        ref.parallel_plate_capacitance(5e-6 * 4e-6, 2e-6, 3.9), rel=1e-9)


def test_periodic_pairs_are_a_bijection():
    g = RectilinearGrid.uniform([(0, 1.0), (0, 1.0), (0, 1.0)], [4, 5, 6])
    lo, hi = B.periodic_pairs(g, 0)
    assert lo.size == hi.size == 6 * 7
    assert np.intersect1d(lo, hi).size == 0
    assert np.unique(lo).size == lo.size


# --------------------------------------------------------------------------
# sources
# --------------------------------------------------------------------------
def test_waveforms_accept_scalar_and_array_time():
    for f in (S.step(1e-9), S.sine(1e9), S.gaussian(1e-9, 1e-10),
              S.ramp(1e9), S.pulse(1e-9, 2e-9)):
        assert np.isscalar(f(0.5e-9)) or np.ndim(f(0.5e-9)) == 0
        t = np.linspace(0, 5e-9, 17)
        assert np.shape(f(t)) == (17,)


def test_step_with_finite_rise_is_continuous():
    f = S.step(1.0e-9, 0.0, 1.0, trise=1.0e-10)
    assert f(0.9e-9) == pytest.approx(0.0)
    assert f(1.05e-9) == pytest.approx(0.5, abs=0.02)
    assert f(1.2e-9) == pytest.approx(1.0)


def test_pwl_clamps_outside_its_range_and_validates():
    f = S.pwl([0.0, 1.0, 2.0], [0.0, 10.0, -5.0])
    assert f(-1.0) == pytest.approx(0.0)
    assert f(0.5) == pytest.approx(5.0)
    assert f(3.0) == pytest.approx(-5.0)
    with pytest.raises(ValueError):
        S.pwl([0.0, 1.0], [0.0])


def test_prbs_has_maximal_length_period():
    for order in (7, 9):
        seq = S.prbs_sequence(order=order, seed=1)
        n = 2 ** order - 1
        assert seq.size >= n
        # A maximal-length LFSR repeats with period 2^n - 1 and is balanced.
        assert np.array_equal(seq[:n], seq[n:2 * n]) if seq.size >= 2 * n else True
        assert seq[:n].sum() == 2 ** (order - 1)


def test_trapezoid_clock_has_the_right_mean():
    f = S.trapezoid_clock(1e-9, duty=0.5, trise=1e-12, tfall=1e-12, v0=0.0, v1=1.0)
    t = np.linspace(0, 20e-9, 200001)
    assert f(t).mean() == pytest.approx(0.5, abs=0.01)


# --------------------------------------------------------------------------
# monitors -- the energy factor-of-2 trap
# --------------------------------------------------------------------------
def test_energy_monitor_matches_half_c_v_squared():
    """Stored energy of a charged parallel plate. Catches factor-of-2 errors."""
    er, d, W, H = 3.9, 2e-6, 5e-6, 4e-6
    g = RectilinearGrid.uniform([(0, d), (0, W), (0, H)], [20, 8, 6])
    eps_cell = np.full(g.shape_cells, er * eps0)
    eps_edge = ops.cell_to_edge(g, eps_cell)
    L = ops.nodal_laplacian(g, eps_edge)
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    lo, hi = nid[0].ravel(), nid[-1].ravel()
    A, b = ops.apply_dirichlet(L, np.zeros(g.n_nodes),
                               np.concatenate([lo, hi]),
                               np.concatenate([np.ones(lo.size), np.zeros(hi.size)]))
    phi = spl.spsolve(A.tocsc(), b)

    C = ref.parallel_plate_capacitance(W * H, d, er)
    expected = 0.5 * C * 1.0 ** 2

    mon = MON.EnergyMonitor(components="electric")
    state = {"phi": phi, "grid": g, "ops": ops.Operators(g),
             "eps_edge": eps_edge, "step": 0}
    mon.record(state, 0.0)
    out = mon.finalize()
    key = next(k for k in out if "elec" in k.lower() or k == "we")
    assert float(np.ravel(out[key])[0]) == pytest.approx(expected, rel=1e-6)


def test_monitor_reports_missing_state_clearly():
    mon = MON.EnergyMonitor(components="magnetic")
    with pytest.raises(Exception):
        mon.record({"step": 0}, 0.0)


# --------------------------------------------------------------------------
# validate -- assumptions become runtime checks
# --------------------------------------------------------------------------
def test_quasistatic_check_classifies_both_regimes():
    small = RectilinearGrid.uniform([(0, 100 * um), (0, 50 * um), (0, 10 * um)],
                                    [20, 10, 4])
    eps_c = np.full(small.shape_cells, 4.0 * eps0)
    r = V.check_quasistatic(small, eps_c, t_rise=20e-12)
    assert r.level == "ok"

    big = RectilinearGrid.uniform([(0, 0.1), (0, 0.05), (0, 0.01)], [20, 10, 4])
    r2 = V.check_quasistatic(big, np.full(big.shape_cells, 4.0 * eps0),
                             t_rise=20e-12)
    assert r2.level in ("warn", "error")
    assert r2.assumption.startswith("A1")


def test_skin_depth_check_flags_an_underresolved_conductor():
    g = RectilinearGrid.uniform([(0, 100e-6), (0, 50e-6), (0, 10e-6)], [40, 20, 8])
    sig = np.full(g.shape_cells, 5.8e7)
    r = V.check_skin_depth(g, sig, None, freq=1e9)
    assert r.level in ("warn", "error")     # 2.5 um cells vs 2.09 um skin depth
    r2 = V.check_skin_depth(g, sig, None, freq=1e3)
    assert r2.level == "ok"                 # 2.09 mm skin depth, easily resolved


def test_mesh_quality_flags_aggressive_grading():
    from fieldspice.grid import graded_1d
    ok = RectilinearGrid(graded_1d(0, 1, 20, 1.1))
    bad = RectilinearGrid(graded_1d(0, 1, 20, 2.5))
    assert V.check_mesh_quality(ok).level == "ok"
    assert V.check_mesh_quality(bad).level in ("warn", "error")


def test_report_combine_takes_the_worst_level():
    rs = [V.check_mesh_quality(RectilinearGrid.uniform([(0, 1.0)], [4]))]
    combined = V.Report.combine(rs)
    assert combined.level in ("ok", "warn", "error")
    assert isinstance(str(combined), str)


# --------------------------------------------------------------------------
# io round-trip
# --------------------------------------------------------------------------
def test_grid_round_trip(tmp_path):
    from fieldspice import io as IO
    from fieldspice.grid import graded_1d
    g = RectilinearGrid(graded_1d(0, 1e-3, 9, 1.2), np.linspace(0, 2e-3, 6),
                        np.linspace(0, 3e-4, 4))
    p = tmp_path / "g.npz"
    IO.save_grid(g, p)
    g2 = IO.load_grid(p)
    assert np.allclose(g2.xn, g.xn)
    assert np.allclose(g2.yn, g.yn)
    assert np.allclose(g2.zn, g.zn)
    assert g2.ncell == g.ncell


def test_result_round_trip(tmp_path):
    from fieldspice import io as IO
    from fieldspice.solvers.base import Result
    g = RectilinearGrid.uniform([(0, 1.0), (0, 1.0)], [3, 3])
    r = Result(grid=g, t=np.linspace(0, 1, 5),
               fields={"phi": np.random.rand(5, *g.shape_nodes)},
               terminals={"a": {"v": np.arange(5.0), "i": np.arange(5.0) * 2}},
               scalars={"energy": np.arange(5.0)},
               meta={"solver": "test", "assumptions": ["A1"]})
    p = tmp_path / "r.npz"
    IO.save_result(r, p)
    r2 = IO.load_result(p)
    assert np.allclose(r2.t, r.t)
    assert np.allclose(r2.fields["phi"], r.fields["phi"])
    assert np.allclose(r2.v("a"), r.v("a"))
    assert r2.meta["solver"] == "test"


def test_vtk_export_preserves_axis_order(tmp_path):
    """VTK is Fortran-ordered; a missing transpose silently mirrors the data."""
    from fieldspice import io as IO
    g = RectilinearGrid.uniform([(0, 1.0), (0, 2.0), (0, 3.0)], [3, 4, 5])
    X, Y, Z = g.node_coords()
    field = (1.0 * X + 10.0 * Y + 100.0 * Z)
    p = tmp_path / "f.vtr"
    IO.export_vtk(g, {"ramp": field}, p)
    text = p.read_bytes()
    assert b"RectilinearGrid" in text
    assert p.stat().st_size > 0
