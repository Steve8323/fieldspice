"""Standalone verification of fieldspice/validate.py against hand computations."""
from __future__ import annotations

import math
import numpy as np

from fieldspice import validate as V
from fieldspice.grid import RectilinearGrid, graded_1d
from fieldspice.units import c0, eps0, mu0, q, kB, per_cm3, um, nm, ps, ns, GHz, MHz

PASS, FAIL = [], []


def chk(name, got, want, rtol=1e-12):
    ok = abs(got - want) <= rtol * max(abs(want), 1e-300)
    (PASS if ok else FAIL).append(name)
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: got {got!r}  want {want!r}")


def chks(name, got, want):
    ok = got == want
    (PASS if ok else FAIL).append(name)
    print(f"  {'OK  ' if ok else 'FAIL'} {name}: got {got!r}  want {want!r}")


def raises(name, fn, *a, **k):
    try:
        fn(*a, **k)
    except ValueError as e:
        PASS.append(name)
        print(f"  OK   {name}: ValueError({str(e)[:70]}...)")
        return
    except Exception as e:  # pragma: no cover
        FAIL.append(name)
        print(f"  FAIL {name}: raised {type(e).__name__} not ValueError")
        return
    FAIL.append(name)
    print(f"  FAIL {name}: no exception")


# =====================================================================
print("\n=== 1. check_quasistatic: on-chip net, 100 um, 20 ps edge, vacuum ===")
g = RectilinearGrid.uniform([(0.0, 100 * um), (0.0, 10 * um)], [20, 4])
eps = np.full(g.shape_cells, eps0)
r = V.check_quasistatic(g, eps, None, t_rise=20 * ps)
f_hand = 0.35 / 20e-12
lam_hand = c0 / f_hand
ratio_hand = 100e-6 / lam_hand
print(r)
chk("f = 0.35/t_rise", r.details["freq_Hz"], f_hand)
chk("lambda = c0/f", r.details["wavelength_m"], lam_hand)
chk("L = largest resolved extent", r.details["L_m"], 100e-6)
chk("L/lambda", r.details["L_over_lambda"], ratio_hand)
print(f"       (hand: f={f_hand:.6g} Hz  lambda={lam_hand:.6g} m  "
      f"L/lam={ratio_hand:.6g})")
chks("band 1 -> ok", (r.level, r.details["band"]), ("ok", "< 0.01 excellent"))
chks("homogeneous -> no cell called out", r.details["material_homogeneous"], True)
chks("no location in message", "at cell" in r.message, False)
chks(".ok true", r.ok, True)
chks("truthiness", bool(r), True)
# L/lambda == 0.01 exactly when L = 0.01*lambda
chk("L_for_excellent = 0.01*lambda", r.details["L_for_excellent_m"],
    0.01 * lam_hand)
chk("freq_for_excellent", r.details["freq_for_excellent_Hz"],
    0.01 * c0 / 100e-6)
chk("t_rise_for_excellent", r.details["t_rise_for_excellent_s"],
    0.35 * 100e-6 / (0.01 * c0))

print("\n--- same net in SiO2 (eps_r 3.9) -> n=1.9748, ratio scales by n ---")
r2 = V.check_quasistatic(g, np.full(g.shape_cells, 3.9 * eps0), None,
                         t_rise=20 * ps)
chk("n = sqrt(3.9)", r2.details["refractive_index"], math.sqrt(3.9))
chk("ratio scales with n", r2.details["L_over_lambda"], ratio_hand * math.sqrt(3.9))
# eps_r 3.9 slows the wave by 1.975x, pushing 0.00584 -> 0.01153: band 2.
chks("eps_r pushes into band 2", r2.details["band"], "0.01 - 0.1 good")

print("\n--- 10 cm PCB trace, same edge, FR4 eps_r 4.4 -> full wave ---")
gp = RectilinearGrid.uniform([(0.0, 0.10), (0.0, 1e-3)], [20, 4])
rp = V.check_quasistatic(gp, np.full(gp.shape_cells, 4.4 * eps0), None,
                         t_rise=20 * ps)
print(rp)
chk("L/lambda", rp.details["L_over_lambda"],
    0.10 * f_hand * math.sqrt(4.4) / c0)
chks("band 4 -> error", (rp.level, rp.details["band"]),
     ("error", "> 0.3 full wave"))
chks(".ok false", rp.ok, False)
raises("raise_if_error", rp.raise_if_error)

print("\n--- band boundaries hit exactly: L chosen so L/lambda = target ---")
for target, want_band, want_level in [(0.005, "< 0.01 excellent", "ok"),
                                      (0.05, "0.01 - 0.1 good", "ok"),
                                      (0.2, "0.1 - 0.3 marginal", "warn"),
                                      (1.0, "> 0.3 full wave", "error")]:
    L = target * c0 / 1e9          # lambda at 1 GHz in vacuum = 0.29979 m
    gb = RectilinearGrid.uniform([(0.0, L), (0.0, L / 10)], [4, 2])
    rb = V.check_quasistatic(gb, np.full(gb.shape_cells, eps0), None, freq=1 * GHz)
    chk(f"ratio at target {target}", rb.details["L_over_lambda"], target, 1e-12)
    chks(f"band at {target}", (rb.details["band"], rb.level),
         (want_band, want_level))

print("\n--- both freq and t_rise given: the faster one wins ---")
rb = V.check_quasistatic(g, eps, None, t_rise=1 * ns, freq=100 * GHz)
chk("uses max(f)", rb.details["freq_Hz"], 100e9)
rb = V.check_quasistatic(g, eps, None, t_rise=1 * ps, freq=1 * MHz)
chk("uses knee", rb.details["freq_Hz"], 0.35 / 1e-12)

print("\n--- input validation ---")
raises("no excitation", V.check_quasistatic, g, eps, None)
raises("negative t_rise", V.check_quasistatic, g, eps, None, t_rise=-1.0)
raises("zero freq", V.check_quasistatic, g, eps, None, freq=0.0)
raises("wrong eps shape", V.check_quasistatic, g, np.ones((3, 3, 3)), None,
       freq=1e9)
raises("relative eps", V.check_quasistatic, g, np.full(g.shape_cells, 3.9),
       None, freq=1e9)
raises("relative mu", V.check_quasistatic, g, eps, np.ones(g.shape_cells),
       freq=1e9)
raises("nan eps", V.check_quasistatic, g,
       np.full(g.shape_cells, np.nan), None, freq=1e9)

# =====================================================================
print("\n=== 2. check_skin_depth: copper at 1 GHz, delta = 2.09 um ===")
delta_hand = math.sqrt(2.0 / (2 * math.pi * 1e9 * mu0 * 5.8e7))
print(f"  hand delta(Cu, 1 GHz) = {delta_hand:.6g} m   "
      f"(ASSUMPTIONS.md says 2.1 um)")
chk("delta formula vs doc 1 GHz", round(delta_hand * 1e6, 1), 2.1)
d_1mhz = math.sqrt(2.0 / (2 * math.pi * 1e6 * mu0 * 5.8e7))
chk("delta formula vs doc 1 MHz", round(d_1mhz * 1e6, 0), 66.0)

# 40 um wide bar of copper, 5 um cells -> badly under-resolved
gs = RectilinearGrid.uniform([(0.0, 100 * um), (0.0, 40 * um)], [20, 8])
sig = np.zeros(gs.shape_cells)
sig[:, :, :] = 0.0
sig[4:16, :, :] = 5.8e7          # bar spanning the full y extent, 60 um in x
rs = V.check_skin_depth(gs, sig, None, 1 * GHz)
print(rs)
chk("delta_min", rs.details["delta_min_m"], delta_hand, 1e-12)
chk("cells per delta", rs.details["min_cells_per_skin_depth"],
    delta_hand / 5e-6, 1e-12)
chk("required cell size", rs.details["required_cell_size_m"], delta_hand / 3.0)
chks("under-resolved -> error", rs.level, "error")
chks("all conductor cells flagged", rs.details["n_under_resolved"],
     rs.details["n_conducting_cells"])
chk("conductor span (60 um in x)", rs.details["conductor_span_m"], 60e-6)

print("\n--- same copper, 0.5 um cells -> 4.18 cells per delta -> ok ---")
gf = RectilinearGrid.uniform([(0.0, 100 * um), (0.0, 40 * um)], [200, 80])
sf = np.zeros(gf.shape_cells)
sf[40:160, :, :] = 5.8e7
rf = V.check_skin_depth(gf, sf, None, 1 * GHz)
chk("cells per delta", rf.details["min_cells_per_skin_depth"],
    delta_hand / 0.5e-6, 1e-12)
chks("resolved -> ok", rf.level, "ok")
chks("none under-resolved", rf.details["n_under_resolved"], 0)

print("\n--- warn band: 1 <= cells/delta < 3 (cells of 1 um) ---")
gw = RectilinearGrid.uniform([(0.0, 100 * um), (0.0, 40 * um)], [100, 40])
sw = np.zeros(gw.shape_cells)
sw[20:80, :, :] = 5.8e7
rw = V.check_skin_depth(gw, sw, None, 1 * GHz)
chk("cells per delta", rw.details["min_cells_per_skin_depth"], delta_hand / 1e-6)
chks("marginal -> warn", rw.level, "warn")

print("\n--- thin film escape: 1 um thick Cu, delta 2.09 um > span ---")
gt = RectilinearGrid.uniform([(0.0, 1 * um), (0.0, 0.5 * um)], [4, 2])
st = np.full(gt.shape_cells, 5.8e7)
rt = V.check_skin_depth(gt, st, None, 1 * GHz)
print(rt)
chks("thin film -> ok", rt.level, "ok")
chk("span = 1 um", rt.details["conductor_span_m"], 1e-6)

print("\n--- no conductors at all ---")
rn = V.check_skin_depth(gs, np.zeros(gs.shape_cells), None, 1 * GHz)
chks("no sigma -> ok", (rn.level, rn.details["n_conducting_cells"]), ("ok", 0))
raises("negative sigma", V.check_skin_depth, gs, -sig, None, 1e9)
raises("zero freq", V.check_skin_depth, gs, sig, None, 0.0)
raises("bad sigma shape", V.check_skin_depth, gs, np.ones((2, 2, 2)), None, 1e9)

# =====================================================================
print("\n=== 3. check_mesh_quality ===")
ru = V.check_mesh_quality(RectilinearGrid.uniform([(0, 1e-3), (0, 1e-3)], [10, 10]))
print(ru)
chks("uniform mesh quotes no bogus location", "between" in ru.message, False)
chk("uniform growth", ru.details["max_growth_ratio"], 1.0)
chk("uniform aspect", ru.details["max_aspect_ratio"], 1.0)
chks("uniform -> ok", ru.level, "ok")

for ratio, want in [(1.3, "ok"), (1.6, "warn"), (2.5, "error")]:
    gg = RectilinearGrid(graded_1d(0.0, 1e-3, 6, ratio=ratio),
                         np.linspace(0, 1e-3, 3))
    rg = V.check_mesh_quality(gg)
    chk(f"growth ratio {ratio}", rg.details["max_growth_ratio"], ratio, 1e-12)
    chks(f"level at growth {ratio}", rg.level, want)
    chks(f"growth axis {ratio}", rg.details["growth_axis"], "x")
print(V.check_mesh_quality(RectilinearGrid(graded_1d(0.0, 1e-3, 6, ratio=2.5),
                                           np.linspace(0, 1e-3, 3))))

print("\n--- high aspect ratio warns on conditioning only ---")
ga = RectilinearGrid(np.linspace(0, 1.0, 3), np.linspace(0, 1e-4, 3))
ra = V.check_mesh_quality(ga)
chk("aspect = 0.5 / 5e-5", ra.details["max_aspect_ratio"], 0.5 / 5e-5)
chks("aspect warn", ra.level, "warn")
chk("growth still 1", ra.details["max_growth_ratio"], 1.0)

print("\n--- 1D grid (both y,z collapsed): aspect must ignore the 1 m thickness ---")
g1 = RectilinearGrid(np.linspace(0, 1e-3, 11))
r1 = V.check_mesh_quality(g1)
chk("1D aspect", r1.details["max_aspect_ratio"], 1.0)
chks("1D ndim", r1.details["ndim_effective"], 1)

# =====================================================================
print("\n=== 4. check_padding ===")
gp2 = RectilinearGrid.uniform([(0.0, 100 * um), (0.0, 100 * um)], [100, 100])
m = np.zeros(gp2.shape_cells, dtype=bool)
m[45:55, 45:55, :] = True                 # 10 um x 10 um block, centred
rp2 = V.check_padding(gp2, m)
print(rp2)
chk("feature size", rp2.details["feature_size_m"], 10e-6)
chk("gap xlo", rp2.details["gap_m"]["xlo"], 45e-6)
chk("pad ratio", rp2.details["worst_pad_ratio"], 4.5)
chks("padded -> ok", rp2.level, "ok")

m2 = np.zeros(gp2.shape_cells, dtype=bool)
m2[30:70, 30:70, :] = True                # 40 um block -> gaps 30 um -> 0.75x
rp3 = V.check_padding(gp2, m2)
print(rp3)
chk("feature", rp3.details["feature_size_m"], 40e-6)
chk("pad ratio 30/40", rp3.details["worst_pad_ratio"], 0.75)
chk("extra needed = 3*40-30 um", rp3.details["extra_needed_m"], 90e-6)
chks("under 1x -> error", rp3.level, "error")

m3 = np.zeros(gp2.shape_cells, dtype=bool)
m3[40:60, 40:60, :] = True                # 20 um block -> gaps 40 um -> 2.0x
rp4 = V.check_padding(gp2, m3)
chk("pad ratio 40/20", rp4.details["worst_pad_ratio"], 2.0)
chks("1x..3x -> warn", rp4.level, "warn")

m4 = np.zeros(gp2.shape_cells, dtype=bool)
m4[:, 0:5, :] = True                       # ground plane spanning x, on ylo
rp5 = V.check_padding(gp2, m4)
print(rp5)
chks("flush walls detected", sorted(rp5.details["flush_walls"]),
     ["xhi", "xlo", "ylo"])
chk("yhi gap", rp5.details["gap_m"]["yhi"], 95e-6)
# x is spanned wall-to-wall so the plane is infinite there: the feature size is
# its 5 um thickness, not its 100 um truncated length.
chks("spanning axis excluded", rp5.details["spanning_axes"], ["x"])
chk("feature = thickness", rp5.details["feature_size_m"], 5e-6, 1e-9)
chk("pad ratio 95/5", rp5.details["worst_pad_ratio"], 19.0, 1e-9)
chks("ground plane with 19x clearance -> ok", rp5.level, "ok")
# ... but shrink the clearance to 10 um (2x the 5 um thickness) and it warns.
gp8 = RectilinearGrid.uniform([(0.0, 100 * um), (0.0, 15 * um)], [100, 15])
m8 = np.zeros(gp8.shape_cells, dtype=bool)
m8[:, 0:5, :] = True
rp8 = V.check_padding(gp8, m8)
chk("pad ratio 10/5", rp8.details["worst_pad_ratio"], 2.0, 1e-9)
chks("thin clearance -> warn", rp8.level, "warn")

m5 = np.ones(gp2.shape_cells, dtype=bool)
rp6 = V.check_padding(gp2, m5)
chks("conductor fills domain -> ok (all shields)", rp6.level, "ok")
rp7 = V.check_padding(gp2, np.zeros(gp2.shape_cells, dtype=bool))
chks("empty mask -> warn", rp7.level, "warn")
raises("bad mask shape", V.check_padding, gp2, np.zeros((3, 3, 3), bool))

# =====================================================================
print("\n=== 5. check_dielectric_relaxation ===")
tau_cu = eps0 / 5.8e7
print(f"  hand tau(Cu) = {tau_cu:.6g} s  (ASSUMPTIONS.md says 1.5e-19 s)")
chk("tau_cu vs doc", round(tau_cu, 20), 1.5e-19, 0.05)

shape = (4, 4, 1)
e = np.full(shape, eps0)
s = np.full(shape, 5.8e7)
rd = V.check_dielectric_relaxation(e, s, 1 * ps)
print(rd)
chk("tau_min", rd.details["tau_min_s"], tau_cu)
chk("explicit dt limit = 2 tau", rd.details["explicit_dt_limit_s"], 2 * tau_cu)
chk("implicit speedup", rd.details["implicit_speedup_vs_relaxation"],
    1e-12 / (2 * tau_cu))
chks("fully relaxed -> ok", rd.level, "ok")
chks("no marginal cells", rd.details["n_marginal_cells"], 0)

print("\n--- silicon substrate sigma=10 S/m: tau = 1.0359e-11 s ---")
e2 = np.full(shape, 11.7 * eps0)
s2 = np.full(shape, 10.0)
tau_si = 11.7 * eps0 / 10.0
print(f"  hand tau(Si, 10 S/m) = {tau_si:.6g} s")
rm = V.check_dielectric_relaxation(e2, s2, 1e-11)
print(rm)
chk("tau", rm.details["tau_min_s"], tau_si)
chk("tau/dt", rm.details["worst_tau_over_dt"], tau_si / 1e-11)
chks("tau ~ dt -> warn", rm.level, "warn")
chk("dt to resolve", rm.details["dt_to_resolve_s"], tau_si / 10.0)
chk("dt to relax", rm.details["dt_to_relax_s"], tau_si / 0.05)
chks("resolved with small dt", V.check_dielectric_relaxation(e2, s2, 1e-13).level,
     "ok")
chks("relaxed with big dt", V.check_dielectric_relaxation(e2, s2, 1e-8).level,
     "ok")
rz = V.check_dielectric_relaxation(e2, np.zeros(shape), 1e-12)
chks("no conductors -> ok", rz.level, "ok")
raises("shape mismatch", V.check_dielectric_relaxation, e2, np.zeros((2, 2, 1)), 1e-12)
raises("dt <= 0", V.check_dielectric_relaxation, e2, s2, 0.0)
raises("relative eps", V.check_dielectric_relaxation, np.full(shape, 11.7), s2, 1e-12)

# =====================================================================
print("\n=== 6. check_debye_length ===")
N = 1e16 * per_cm3
LD_hand = math.sqrt(11.7 * eps0 * kB * 300.0 / (q * q * N))
print(f"  hand LD(Si, 1e16 cm^-3, 300 K) = {LD_hand:.6g} m = {LD_hand*1e9:.3g} nm")
rdb = V.check_debye_length(np.full((4, 4, 1), N), 300.0)
print(rdb)
chk("LD_min", rdb.details["LD_min_m"], LD_hand)
chk("N in cm^-3", rdb.details["N_max_per_cm3"], 1e16)
chk("kT/q", rdb.details["thermal_voltage_V"], kB * 300.0 / q)
chks("no grid -> ok", rdb.level, "ok")
# LD scales as 1/sqrt(N)
rdb2 = V.check_debye_length(np.full((4, 4, 1), 100 * N), 300.0)
chk("LD ~ 1/sqrt(N)", rdb2.details["LD_min_m"], LD_hand / 10.0)

print("\n--- with a grid: 40.9 nm LD against 200 nm cells -> error ---")
gd = RectilinearGrid.uniform([(0.0, 2 * um), (0.0, 1 * um)], [10, 5])
dop = np.full(gd.shape_nodes, N)
rd2 = V.check_debye_length(dop, 300.0, grid=gd)
print(rd2)
chk("cell size 200 nm", rd2.details["worst_cell_size_m"], 200e-9)
chk("cells per LD", rd2.details["min_cells_per_debye_length"], LD_hand / 200e-9)
chks("0.204 cells per LD -> warn", rd2.level, "warn")
# 1 um cells: 0.0409 cells per LD, below the 0.2 error threshold.
gd0 = RectilinearGrid.uniform([(0.0, 10 * um), (0.0, 5 * um)], [10, 5])
rd0 = V.check_debye_length(np.full(gd0.shape_nodes, N), 300.0, grid=gd0)
chk("cells per LD (1 um cells)", rd0.details["min_cells_per_debye_length"],
    LD_hand / 1e-6)
chks("0.041 cells per LD -> error", rd0.level, "error")

gd2 = RectilinearGrid.uniform([(0.0, 2 * um), (0.0, 1 * um)], [100, 50])
rd3 = V.check_debye_length(np.full(gd2.shape_nodes, N), 300.0, grid=gd2)
chk("cells per LD (20 nm cells)", rd3.details["min_cells_per_debye_length"],
    LD_hand / 20e-9)
chks("2 cells per LD -> ok", rd3.level, "ok")

gd3 = RectilinearGrid.uniform([(0.0, 2 * um), (0.0, 1 * um)], [25, 12])
rd4 = V.check_debye_length(np.full(gd3.shape_cells, N), 300.0, grid=gd3)
chk("cell-shaped doping accepted", rd4.details["worst_cell_size_m"], 1e-6 / 12)
chks("0.49 cells per LD -> warn", rd4.level, "warn")

chks("zero doping -> ok", V.check_debye_length(np.zeros((3, 3, 1))).level, "ok")
chk("eps override", V.check_debye_length(np.full((2, 2, 1), N), 300.0,
                                         eps=3.9 * eps0).details["LD_min_m"],
    math.sqrt(3.9 * eps0 * kB * 300.0 / (q * q * N)))
raises("bad doping shape vs grid", V.check_debye_length,
       np.ones((3, 3, 3)) * N, 300.0, gd)
raises("T <= 0", V.check_debye_length, np.full((2, 2, 1), N), 0.0)
raises("relative eps arg", V.check_debye_length, np.full((2, 2, 1), N), 300.0,
       None, 11.7)

# =====================================================================
print("\n=== 7. Report.combine / __str__ / check_all ===")
c = V.Report.combine([ru, rp3, rd])
chks("combine takes worst level", c.level, "error")
chks("children kept", len(c.children), 3)
chks("tags merged", c.assumption, "A10, A12, A1")
chks("levels tally", c.details["levels"], {"ok": 2, "warn": 0, "error": 1})
chk("child details reachable", c.details["check_padding"]["feature_size_m"],
    40e-6, 1e-9)
chks("walk", len(c.walk()), 4)
raises("combine rejects non-Report", V.Report.combine, [ru, "nope"])
raises("bad level", V.Report, "x", "fatal", "m")
chks("combine of all-ok is ok", V.Report.combine([ru, rd]).level, "ok")

print("\n--- check_all on a realistic on-chip cross-section ---")
gA = RectilinearGrid.uniform([(0.0, 100 * um), (0.0, 20 * um)], [200, 40])
epsA = np.full(gA.shape_cells, 3.9 * eps0)
sigA = np.zeros(gA.shape_cells)
sigA[80:120, 10:14, :] = 5.8e7            # 20 um x 2 um copper wire
epsA[80:120, 10:14, :] = eps0
rall = V.check_all(gA, epsA, sigA, None, t_rise=20 * ps, dt=1 * ps)
print(rall)
# The wire sits 5 um from the ylo wall (0.25x its own 20 um length) and the
# 0.5 um cells give 1 cell per 0.5 um skin depth at 17.5 GHz: both are errors.
chks("check_all level", rall.level, "error")
chks("worst child is an error", sorted({c.level for c in rall.children}),
     ["error", "ok"])
chks("5 checks ran", len(rall.children), 5)
chks("debye skipped", "check_debye_length" in rall.details["skipped"], True)
names = [ch.check for ch in rall.children]
chks("child order", names, ["check_mesh_quality", "check_quasistatic",
                            "check_skin_depth", "check_padding",
                            "check_dielectric_relaxation"])

print("\n--- check_all with only a grid ---")
rmin = V.check_all(gA)
chks("only mesh check runs", len(rmin.children), 1)
chks("four skipped", len(rmin.details["skipped"]), 5)

print("\n--- check_all with doping ---")
rdd = V.check_all(gA, epsA, sigA, None, freq=1 * GHz, dt=1 * ps,
                  doping=np.full(gA.shape_nodes, 1e18 * per_cm3))
chks("6 checks", len(rdd.children), 6)
chks("no skips", "skipped" not in rdd.details, True)

print("\n" + "=" * 60)
print(f"{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    raise SystemExit(1)
print("ALL CHECKS PASSED")
