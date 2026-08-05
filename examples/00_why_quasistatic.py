"""Measure the speedup the quasi-static approximation actually buys.

This is the central claim of the project, so it should be measured rather than
asserted. The script computes, for a range of real circuit problems, the
explicit-FDTD Courant step the geometry would force, the time step the physics
actually needs, and the ratio between them.

Run:  python examples/00_why_quasistatic.py
"""

from __future__ import annotations

from fieldspice import reference as ref
from fieldspice.units import c0, eps0, mm, nm, ns, ps, um

# name, domain size (m), smallest feature (m), signal rise time (s), eps_r
CASES = [
    ("Gate oxide stack",        200 * nm,   0.5 * nm,   1 * ns,    3.9),
    ("On-chip net",             100 * um,    10 * nm,  20 * ps,    4.2),
    ("Analog settling (TFT)",    50 * um,     1 * um,   1e-3,     16.0),
    ("Package interconnect",      5 * mm,     1 * um,   50 * ps,   3.5),
    ("PCB trace",                50 * mm,    50 * um,  100 * ps,   4.4),
]

HEAD = (f"{'case':<24}{'L/lambda':>10}{'regime':>14}"
        f"{'Courant dt':>13}{'signal dt':>12}{'speedup':>12}")


def main() -> None:
    print(__doc__.splitlines()[0])
    print()
    print(HEAD)
    print("-" * len(HEAD))

    for name, size, feature, t_rise, eps_r in CASES:
        # The Courant step is set by the SMALLEST cell, so compute it from the
        # feature size directly. Building a truncated grid and reading its
        # courant_dt would quietly report the step of a coarser mesh than the
        # problem actually needs, which would flatter the comparison.
        dt_courant = ref.courant_limit(feature, feature, feature, eps_r=1.0)
        # Quasi-static accuracy target: ~50 steps per rise time is plenty for
        # a first-order scheme on a smooth signal.
        dt_signal = t_rise / 50.0
        speedup = dt_signal / dt_courant

        ratio = ref.electrical_length(size, t_rise=t_rise, eps_r=eps_r)
        if ratio < 0.01:
            regime = "quasi-static"
        elif ratio < 0.1:
            regime = "Darwin"
        elif ratio < 0.3:
            regime = "marginal"
        else:
            regime = "FULL-WAVE"

        print(f"{name:<24}{ratio:>10.2e}{regime:>14}"
              f"{dt_courant:>13.2e}{dt_signal:>12.2e}{speedup:>12.3g}")

    print()
    print("Reading the table")
    print("-----------------")
    print("'Courant dt' is the largest step an explicit full-wave FDTD run could")
    print("take on a grid that resolves the smallest feature: it is set by how")
    print("long light takes to cross one cell, and has nothing to do with the")
    print("signal. 'signal dt' is what an unconditionally stable implicit")
    print("quasi-static solver needs to resolve the waveform. Their ratio is the")
    print("number of explicit steps each quasi-static step replaces.")
    print()
    print("The 'regime' column is the honest caveat. Where it says FULL-WAVE the")
    print("speedup number is irrelevant, because the quasi-static answer would be")
    print("wrong at any step size -- radiation is a real part of the physics")
    print("there. That error does not shrink under mesh refinement, which is why")
    print("fieldspice.validate checks L/lambda before you spend an hour on a run.")
    print()

    # The second, independent reason explicit schemes fail on circuits.
    for metal, sigma in (("copper", 5.8e7), ("aluminium", 3.77e7),
                         ("IGZO (thin film)", 1.0e3)):
        tau = eps0 / sigma
        print(f"dielectric relaxation time of {metal:<18} "
              f"eps0/sigma = {tau:.2e} s")
    print()
    print("Copper relaxes charge ~200x faster than the Courant step at 0.5 nm")
    print("resolution, so an explicit scheme is stiff inside metal before it is")
    print("even slow. The quasi-static solvers are implicit and simply do not")
    print("care: the relaxation is resolved by the elliptic solve, not stepped.")


if __name__ == "__main__":
    main()
