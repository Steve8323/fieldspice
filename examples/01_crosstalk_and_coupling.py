"""Coupled interconnect: extract the parasitics, then drive them from a netlist.

Two parallel on-chip wires over a ground plane.  The script

1. extracts the full capacitance matrix from a 3D field solve,
2. reports the coupling ratio that sets crosstalk,
3. drops the *same field region* into a SPICE netlist as a single element and
   runs a transient, so the aggressor edge and the victim's response are
   computed from the fields rather than from a hand-built RC model.

Step 3 is the part that is unusual.  The field region is not co-simulated with
the netlist through a handshake --- it is reduced by a Schur complement to an
exact terminal admittance and stamped into the MNA matrix like any other
element, so the answer is the one you would get by meshing the whole problem.

Run:  python examples/01_crosstalk_and_coupling.py
"""

from __future__ import annotations

import numpy as np

import fieldspice as fs
from fieldspice import extraction as EX
from fieldspice import reference as ref
from fieldspice import sources as S
from fieldspice.circuit.coupling import FieldRegion
from fieldspice.circuit.mna import MNASolver, Netlist
from fieldspice.grid import RectilinearGrid
from fieldspice.solvers.base import Terminal
from fieldspice.units import eps0, fF, nm, ns, ps, um


def main() -> None:
    # ------------------------------------------------------------------
    # Geometry: two 0.5 um wires, 0.5 um apart, 0.4 um above a ground plane,
    # embedded in SiO2. A representative intermediate-metal on-chip pitch.
    # ------------------------------------------------------------------
    w, space, h_ox, t_metal = 0.5 * um, 0.5 * um, 0.4 * um, 0.3 * um
    pad = 2.0 * um                      # A12: pad the box or fringing is lost
    Lx = 2 * pad + 2 * w + space
    Lz = pad + h_ox + t_metal
    Ly = 1.0 * um                       # a slice of an otherwise long line

    # Grade the mesh around every material interface rather than unioning a
    # uniform grid with the feature coordinates: that union leaves a 30 nm cell
    # next to a 90 nm one, a growth ratio of 5.5, and fieldspice.validate warns
    # above 1.5 for good reason (A10 --- the box method drops toward first order
    # where cells change size abruptly).
    from fieldspice.grid import auto_mesh_1d
    x = auto_mesh_1d((0, Lx), [pad, pad + w, pad + w + space,
                               pad + 2 * w + space],
                     dx_min=40 * nm, dx_max=250 * nm, growth=1.35)
    z = auto_mesh_1d((0, Lz), [h_ox, h_ox + t_metal],
                     dx_min=40 * nm, dx_max=250 * nm, growth=1.35)
    g = RectilinearGrid(x, np.linspace(0, Ly, 4), z)

    eps = np.full(g.shape_cells, 3.9 * eps0)          # SiO2 everywhere
    X, _, Z = g.cell_coords()

    def wire(x0):
        return ((X >= x0) & (X <= x0 + w)
                & (Z >= h_ox) & (Z <= h_ox + t_metal))

    a_cells = wire(pad)
    v_cells = wire(pad + w + space)

    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
    Xn, _, Zn = g.node_coords()

    def wire_nodes(x0):
        m = ((Xn >= x0 - 1e-12) & (Xn <= x0 + w + 1e-12)
             & (Zn >= h_ox - 1e-12) & (Zn <= h_ox + t_metal + 1e-12))
        return nid[m].ravel()

    aggressor = Terminal("agg", wire_nodes(pad))
    victim = Terminal("vic", wire_nodes(pad + w + space))
    ground = Terminal("gnd", nid[:, :, 0].ravel())
    terms = [aggressor, victim, ground]

    print(g.summary())
    from fieldspice import validate as V
    print(" ", V.check_mesh_quality(g))
    print()

    # ------------------------------------------------------------------
    # 1. Capacitance extraction
    # ------------------------------------------------------------------
    C = EX.capacitance_matrix(g, eps, terms)
    Cs = EX.to_spice_matrix(C)
    names = [t.name for t in terms]
    print("Maxwell capacitance matrix [fF] (per %.1f um of line):" % (Ly / um))
    print("        " + "".join(f"{n:>12}" for n in names))
    for i, n in enumerate(names):
        print(f"  {n:>5} " + "".join(f"{C[i, j] / fF:>12.4f}" for j in range(3)))

    c_couple = Cs[0, 1]
    c_agg_gnd = Cs[0, 2]
    print(f"\n  coupling  C_agg-vic = {c_couple / fF:.4f} fF")
    print(f"  to ground C_agg-gnd = {c_agg_gnd / fF:.4f} fF")
    print(f"  coupling ratio      = {c_couple / (c_couple + c_agg_gnd):.4f}")
    print("  (that ratio is the capacitive-divider bound on far-end crosstalk")
    print("   when the victim floats: the classic first estimate)")

    asym = float(np.abs(C - C.T).max() / np.abs(C).max())
    rows = float(np.abs(C.sum(axis=1)).max() / np.abs(C).max())
    print(f"\n  quality: asymmetry {asym:.2e}, row-sum {rows:.2e} "
          "(both should be ~0)")

    # ------------------------------------------------------------------
    # 2. Regime check --- is quasi-static even legitimate here?
    # ------------------------------------------------------------------
    t_rise = 20 * ps
    ratio = ref.electrical_length(100 * um, t_rise=t_rise, eps_r=3.9)
    print(f"\n  L/lambda for a 100 um net at a {t_rise / ps:.0f} ps edge: "
          f"{ratio:.2e}  ->  quasi-static is valid (A1)")

    # ------------------------------------------------------------------
    # 3. The same field region, driven from a netlist
    # ------------------------------------------------------------------
    sigma = np.zeros(g.shape_cells)     # pure dielectric; metals are electrodes
    region = FieldRegion("INT", g, eps, sigma, terms,
                         circuit_nodes=["agg", "vic", "0"])

    r_drv, r_term = 200.0, 5e3          # driver output R, victim termination
    t0 = 50 * ps
    net = Netlist()
    net.add_vsource("VIN", "in", "0", S.step(t0, 0.0, 1.0, trise=t_rise))
    net.add_resistor("RD", "in", "agg", r_drv)
    net.add_resistor("RV", "vic", "0", r_term)
    net.add_device(region)

    dt = 1 * ps
    solver = MNASolver(net)
    res = solver.transient(t_end=t0 + 400 * ps, dt=dt, method="be")
    t = res.t
    v_agg = res.fields["x"][:, solver.index("agg")]
    v_vic = res.fields["x"][:, solver.index("vic")]

    peak = float(np.max(np.abs(v_vic)))
    print(f"\n  transient ({len(t)} steps of {dt / ps:.0f} ps):")
    print(f"    aggressor settles to {v_agg[-1]:.4f} V")
    print(f"    victim peak crosstalk {peak * 1e3:.3f} mV "
          f"({peak * 100:.3f} % of the 1 V edge)")
    print(f"    victim settles back to {v_vic[-1] * 1e3:.4f} mV")
    print(f"    reduced admittance asymmetry {region.asymmetry:.2e}")

    # The victim is resistively terminated, so the peak is well below the
    # floating capacitive-divider bound; that the transient stays under it is
    # a consistency check on the whole chain.
    bound = c_couple / (c_couple + c_agg_gnd)
    print(f"    floating-victim bound would be {bound * 100:.2f} % "
          f"-> terminated peak is {peak / bound:.3f} of it, as expected < 1")

    print("\n  The field region contributed one dense 3x3 admittance block to")
    print("  the MNA matrix. No relaxation, no handshake: for a linear region")
    print("  the Schur complement is exact.")


if __name__ == "__main__":
    main()
