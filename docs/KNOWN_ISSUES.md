# Known issues

Open defects found by independent validation, with the measurements that
demonstrate them. Each has an `xfail` test in `tests/` so the suite stays green
without hiding the problem, and so the day a fix lands the test flips to
`xpass` and tells us.

Nothing here is a *modelling* limitation — those live in
[`ASSUMPTIONS.md`](ASSUMPTIONS.md) and are deliberate. These are bugs.

---

## 1. `fdtd.py` — CPML does not absorb (major)

**Test:** `tests/test_solvers.py::test_fdtd_pml_absorbs`

```python
FDTDSolver.measure_pml_reflection(n_cells=200, dx=2e-5, thickness=10)
# reflection      = 0.9356      (93.6% of the wave returns)
# reflection_dB   = -0.578 dB
# reflection_rms  = -3.93 dB
```

A working CPML reaches −60 to −80 dB. At −0.6 dB the layer is doing essentially
nothing, so **any open-boundary full-wave result is dominated by boundary
reflection** and should not be trusted.

The solver's own instrumentation reports this correctly, which is worth noting:
the measurement is right, the absorber is not.

**Scope of impact.** Closed-domain full-wave problems (PEC cavities, the
propagation and dispersion tests) are unaffected — the wave never reaches the
boundary, and those tests pass. Only open-region radiation problems are
affected. Since `fieldspice` is a quasi-static tool whose full-wave solver
exists mainly as a *reference*, and quasi-static solvers use Neumann walls
rather than PML, this does not touch the main code path.

**Recommended fix.** Ship a first-order Mur ABC (~−30 dB, a few lines, hard to
get wrong) as the default and keep CPML behind a flag until it is measured
good. An honest Mur beats a broken CPML.

---

## 2. `dd.py` — spurious current at zero bias (major)

**Test:** `tests/test_solvers.py::test_pn_diode_current_vanishes_at_zero_bias`

Symmetric Si pn junction, `Na = Nd = 1e17 cm^-3`, 1 µm long, 1 nm graded mesh:

```
I(V=0)    = -5.0033e-06 A     <-- must be EXACTLY zero
I(V=0.05) = +1.8572e-05 A
|I(0)| / |I(0.1 V)| = 6.7e-02
```

At zero bias the junction is in thermodynamic equilibrium, so detailed balance
requires the terminal current to vanish identically. It does not: the offset is
~7% of the current at 0.1 V and contaminates the low-bias region.

**What is *not* wrong.** The transport physics is right. In the clean window
(0.10–0.35 V) the measured ideality is **n = 1.0029 over 3.36 decades**, and the
per-step current ratio is 6.866 against the ideal `exp(0.05/Vt) = 6.897`. This
is an additive offset, not a slope error.

**Ruled out.** SRH generation cannot explain it: at equilibrium `np = ni²`
exactly, so generation and recombination cancel identically.

**Remaining suspects.** Terminal-current extraction dropping a term (most
likely); ohmic contacts not held at exact equilibrium by the bias ramp;
a recombination-current imbalance at the contact nodes.

---

## 3. `dd.py` — `iv_curve` does not record the swept voltage (minor)

`Result.v("anode")` returns `0.0` at every sweep point. It records the
`Terminal` object's static `.voltage` attribute rather than the value actually
applied at each step, so the returned `Result` cannot plot an I–V curve without
the caller re-supplying the sweep values it already passed in.

Currents (`Result.i(...)`) are recorded correctly.

---

## 4. `mqs` and `darwin` are not implemented (scope gap, not a bug)

Specified in [`CONTRACTS.md`](CONTRACTS.md), never built --- the agent fleet
that was writing them hit a session rate limit. Consequence: **the quasi-static
path models R and C but not L.** There is no eddy-current, skin-effect or
proximity-effect capability, and no inductive transient.

`extraction.rlgc_2d` does return an inductance matrix, but from the LC identity
`L = mu0 eps0 C_air^-1`, which is exact for a TEM line and says nothing about
current redistribution inside a conductor. Do not read it as an eddy-current
result.

---

## Reporting

Found something else? Please include the measurement, not just the symptom —
the numbers above are what make these actionable. `fieldspice.reference` is the
oracle; if a solver disagrees with it, say by how much.
