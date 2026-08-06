## What this changes

<!-- One or two sentences. -->

## Evidence

<!--
Required for anything touching physics or numerics. Paste the actual numbers.
"Looks right" is not a result; a passing test with a loosened tolerance is
worse than a failing one.
-->

- [ ] Compared against a closed-form reference in `fieldspice/reference.py`
- [ ] Measured error: `________`
- [ ] Convergence order measured (if applicable): `________`
- [ ] `pytest -q` passes locally

## Assumptions

- [ ] Any new approximation is documented in `docs/ASSUMPTIONS.md` with a tag
- [ ] Any new optional mechanism is wired through `PhysicsOptions` and removes
      its assumption tag from `Result.meta["assumptions"]`
- [ ] No frozen-core file (`grid.py`, `operators.py`, `units.py`,
      `solvers/base.py`) was modified — or, if it was, the reason is explained
      below and `curl grad = 0` / `div curl = 0` still hold exactly

## Notes

<!-- Anything you are unsure about, or a limitation you are knowingly shipping. -->
