"""Self-heating: the same device, isothermal and coupled, and why it matters.

A metal micro-bridge carrying current, cooled by convection. Run three ways:

1. isothermal (assumption A6 in force) --- the default everywhere else,
2. coupled electro-thermal, voltage driven --- negative feedback, stable,
3. coupled electro-thermal, current driven --- positive feedback, and above a
   threshold **no steady state exists at all**.

The point of case 3 is that the isothermal model reports a perfectly ordinary
operating point for a device that physically destroys itself. No amount of mesh
refinement finds that, because it is a missing mechanism rather than a
truncation error.

Run:  python examples/02_self_heating.py
"""

from __future__ import annotations

import warnings

import numpy as np

from fieldspice.grid import RectilinearGrid
from fieldspice.physics import PhysicsOptions
from fieldspice.solvers.base import Terminal
from fieldspice.solvers.electrothermal import (ElectroThermalSolver,
                                               ThermalRunaway,
                                               runaway_threshold)
from fieldspice.units import um


def main() -> None:
    # A 100 x 20 x 20 um metal bridge, copper-like tcr, convectively cooled.
    Lc, Wc, Hc = 100 * um, 20 * um, 20 * um
    sigma0, tcr, kappa, h = 1.0e6, 3.93e-3, 5.0e5, 2.0e4

    g = RectilinearGrid(np.linspace(0, Lc, 21), np.linspace(0, Wc, 5),
                        np.linspace(0, Hc, 5))
    sig = np.full(g.shape_cells, sigma0)
    kap = np.full(g.shape_cells, kappa)
    conv = {w: (h, 300.0) for w in
            ("xlo", "xhi", "ylo", "yhi", "zlo", "zhi")}
    nid = np.arange(g.n_nodes).reshape(g.shape_nodes)

    A_surface = 2 * (Lc * Wc + Lc * Hc + Wc * Hc)
    R_th = 1.0 / (h * A_surface)
    R0 = Lc / (sigma0 * Wc * Hc)
    I_crit = runaway_threshold(R_th, R0, tcr)

    print(f"bridge: R0 = {R0:.4f} ohm, R_th = {R_th:.1f} K/W, tcr = {tcr:.2e}/K")
    print(f"runaway threshold I_crit = 1/sqrt(R_th R0 tcr) = {I_crit:.4f} A\n")

    def solve(term, damping=0.6, T_limit=1e9):
        opts = PhysicsOptions(self_heating=True, max_coupling_iterations=3000,
                              coupling_tolerance=1e-10)
        s = ElectroThermalSolver(g, sig, kap, tcr, opts)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return s.solve([term, Terminal("b", nid[-1].ravel(), voltage=0.0)],
                           convection=conv, damping=damping, T_limit=T_limit)

    # ------------------------------------------------------------------
    print("1. VOLTAGE DRIVEN --- P = V^2/R, heating raises R, power falls.")
    print("   Negative feedback: stable at any drive.\n")
    print("     V (V)   dT isothermal(K)   dT coupled(K)   analytic(K)   error")
    for V in (0.1, 0.5, 1.0, 2.0):
        r = solve(Terminal("a", nid[0].ravel(), voltage=V))
        dT = float(r.scalars["T_rise"][0])
        iso = R_th * V * V / R0                     # A6: R never changes
        B = R_th * V * V / R0
        ana = (-1 + np.sqrt(1 + 4 * tcr * B)) / (2 * tcr)
        print(f"     {V:5.1f}   {iso:14.2f}   {dT:13.2f}   {ana:11.2f}"
              f"   {abs(dT / ana - 1):.1e}")
    print("\n   The isothermal column OVER-predicts the temperature here: it")
    print("   never lets the resistance rise, so it never throttles the power.")

    # ------------------------------------------------------------------
    print("\n2. CURRENT DRIVEN --- P = I^2 R, heating raises R, power rises.")
    print("   Positive feedback: stable only below I_crit.\n")
    print("     I/I_crit   dT isothermal(K)   dT coupled(K)   analytic(K)   error")
    for frac in (0.2, 0.5, 0.8, 0.95):
        I = frac * I_crit
        r = solve(Terminal("a", nid[0].ravel(), current=I))
        dT = float(r.scalars["T_rise"][0])
        iso = R_th * I * I * R0
        A = R_th * I * I * R0
        ana = A / (1 - A * tcr)
        print(f"     {frac:8.2f}   {iso:14.2f}   {dT:13.2f}   {ana:11.2f}"
              f"   {abs(dT / ana - 1):.1e}")
    print("\n   Here the isothermal column UNDER-predicts, and the gap widens")
    print("   without bound as I approaches I_crit.")

    # ------------------------------------------------------------------
    print("\n3. ABOVE THRESHOLD --- no fixed point exists.\n")
    for frac in (1.05, 1.5):
        I = frac * I_crit
        iso = R_th * I * I * R0
        try:
            r = solve(Terminal("a", nid[0].ravel(), current=I), T_limit=5000.0)
            print(f"     {frac:.2f} I_crit: converged to {r.scalars['T_rise'][0]:.0f} K")
        except ThermalRunaway as exc:
            print(f"     {frac:.2f} I_crit ({I:.4f} A):")
            print(f"       isothermal model says: a calm {iso:.0f} K rise, "
                  f"device fine")
            print(f"       coupled model says:    ThermalRunaway after "
                  f"{len(exc.history)} iterations")
            print(f"       temperature history:   "
                  f"{' -> '.join(f'{t:.0f}' for t in exc.history[:5])} ... K")
    print("\n   Copper melts at 1358 K. The isothermal answer is not merely")
    print("   inaccurate, it is the wrong verdict about whether the part")
    print("   survives --- and refining the mesh converges to it precisely.")


if __name__ == "__main__":
    main()
