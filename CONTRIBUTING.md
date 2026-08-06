# Contributing to fieldspice

Thanks for looking. This project has an unusual bar for contributions, and it
is worth stating up front so nobody wastes an afternoon.

## The one rule

**Every physical claim must be measured against a closed-form reference.**

`fieldspice/reference.py` is the oracle: a module of analytic solutions that
ships with the package. If you add a solver, a material, or a mechanism, add
the reference solution it should reproduce and a test that reproduces it *with
a number in the assertion*. "It looks right" is not a result.

If your change disagrees with the reference, say by how much and why. An
honest "this is 12% off because the reference formula is itself a curve fit"
is worth far more than a passing test with a loosened tolerance.

## Things that will get a PR rejected

- Loosening an existing tolerance to make a test pass.
- A benchmark or claim with no executed evidence behind it.
- A feature flag that raises `NotImplementedError`. An honest absence is
  better than an implied capability — see `docs/KNOWN_ISSUES.md` for how
  missing things are recorded.
- Adding a physical mechanism without adding its assumption tag to
  `docs/ASSUMPTIONS.md` and wiring it through `PhysicsOptions`.
- Editing `grid.py`, `operators.py`, `units.py` or `solvers/base.py` casually.
  These are the frozen core; `curl grad = 0` and `div curl = 0` hold *exactly*
  and a great deal depends on that staying true. Changes there need a very
  good reason and a very thorough test.

## Setup

```bash
git clone https://github.com/Steve8323/fieldspice
cd fieldspice
pip install -e ".[all]"
pytest -q                 # 173 passing, 2 xfail (known open defects)
```

Optional dependencies (`matplotlib`, `h5py`, `pyamg`, `numba`) must stay
optional: the package has to import and run without any of them. There is a
test that checks this.

## Style

- Strict SI everywhere. No unit scaling, ever.
- Comments explain **why**, not what.
- Docstrings state the **units** of every physical argument and return value.
- NumPy-style docstrings, full type hints, `from __future__ import annotations`.
- Vectorised NumPy; loop only over time steps or Newton iterations.
- No emoji in code or docs.

## Good first contributions

The most valuable open items, roughly in order:

1. **A working absorbing boundary for `fdtd`.** The CPML currently reflects
   −0.58 dB (see `docs/KNOWN_ISSUES.md` §1). A first-order Mur ABC that
   measurably hits −30 dB would be a strict improvement on what is there.
2. **The drift-diffusion zero-bias current** (§2). Ideality is right; there is
   an additive offset that should be identically zero by detailed balance.
3. **`mqs` / `darwin`** — magnetoquasistatics and full R+L+C. Specified in
   `docs/CONTRACTS.md`, never built. This is the largest missing capability:
   there is currently no inductance anywhere in the package.
4. **A better 3D linear solver path.** Direct factorisation is O(N²) in 3D and
   scipy ships no nested-dissection ordering; a PyAMG-based path exists but is
   under-tuned.

## Reporting a bug

Include the measurement, not just the symptom. `fieldspice.reference` is the
oracle — if a solver disagrees with it, say by how much, on what grid, and at
what refinement.
