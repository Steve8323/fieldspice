# fieldspice

**Electromagnetic field simulation for analog and digital circuits, from first principles.**

A field solver built the way [MEEP](https://meep.readthedocs.io) is built — a
geometry description, a staggered Yee grid, sources, monitors, a Python API —
but aimed at the regime MEEP is not: structures far smaller than a wavelength,
full of conductors and semiconductors, driven for microseconds rather than
picoseconds.

```python
import numpy as np
import fieldspice as fs

# A 2 um oxide gap between two plates.
g = fs.RectilinearGrid.uniform([(0, 2*fs.um), (0, 5*fs.um), (0, 4*fs.um)], [24, 10, 8])
mats = fs.MaterialMap(g, background="sio2")

nid = np.arange(g.n_nodes).reshape(g.shape_nodes)
top    = fs.Terminal("top",    nid[0].ravel(),  voltage=1.0)
bottom = fs.Terminal("bottom", nid[-1].ravel(), voltage=0.0)

C = fs.extraction.capacitance_matrix(g, mats.eps(), [top, bottom])
print(C[0, 0])        # 3.4531e-16 F, matching eps*A/d to 1e-12
```

---

## Why not just run FDTD?

Because you cannot. Explicit full-wave FDTD is bound by the
Courant–Friedrichs–Lewy condition, so its time step is set by how long light
takes to cross the *smallest cell*. Circuits are the worst possible case for
this — tiny features, long time scales:

| Problem | Smallest cell | Courant Δt | Time of interest | Steps needed |
|---|---|---|---|---|
| Gate oxide, 2 nm | 0.5 nm | 9.6e-19 s | 1 ns switching | **1.0e9** |
| On-chip wire | 10 nm | 1.9e-17 s | 100 ps edge | 5.2e6 |
| PCB trace | 50 µm | 9.6e-14 s | 10 ns | 1.0e5 |
| Analog settling | 1 µm | 1.9e-15 s | 1 ms | **5.2e11** |

A billion time steps on a multi-million-cell grid is not a slow simulation, it
is an impossible one. Conductors make it worse in a second, independent way: the
dielectric relaxation time of copper is ε₀/σ ≈ 1.5e-19 s, some 200× *below* the
Courant step even at 0.5 nm resolution, so any explicit scheme that resolves
charge relaxation inside metal is unstable before it is slow.

**But circuits do not need the wave.** They are electrically small: the
structure is much shorter than a wavelength, so the radiative coupling between E
and B contributes nothing to the answer. Drop it and the system becomes elliptic
in space and parabolic in time — no CFL condition, unconditionally stable
implicit stepping, and a time step set by *the signal* instead of the speed of
light.

`examples/00_why_quasistatic.py` measures the ratio for real problems
(number of explicit steps each implicit quasi-static step replaces):

```
case                      L/lambda        regime   Courant dt   signal dt     speedup
Gate oxide stack          4.61e-07  quasi-static     9.63e-19    2.00e-11    2.08e+07
On-chip net               1.20e-02        Darwin     1.93e-17    4.00e-13    2.08e+04
Analog settling (TFT)     2.33e-10  quasi-static     1.93e-15    2.00e-05    1.04e+10
Package interconnect      2.18e-01      marginal     1.93e-15    1.00e-12         519
PCB trace                 1.22e+00     FULL-WAVE     9.63e-14    2.00e-12        20.8
```

Four to ten orders of magnitude, obtained entirely by discarding a physical
effect that was not doing any work — and note the last row, where the speedup
number is meaningless because quasi-static is the *wrong model* there. Every
solver reports `speedup_vs_courant()` so the claim is measured per run rather
than asserted.

The catch, stated plainly: a quasi-static solver **cannot** produce a radiated
far field, model an antenna, or show a standing-wave resonance. That is a
modelling error, not a discretisation error, so it does **not** shrink under
mesh refinement — the most dangerous failure mode of any tool in this class.
`fieldspice.validate` computes `L/λ` for your actual geometry and tells you when
you have left the regime.

---

## The formulation

Everything rests on one idea: **separate topology from metric.**

The discrete gradient, curl and divergence act on quantities *integrated over
their geometric element* — node potentials [V], edge voltages [V], face fluxes
[Wb]. Written that way they are exact signed-incidence matrices with entries in
`{-1, 0, +1}` and **no grid spacing in them at all**:

```python
G = grad_node_edge(grid)     # nodes -> edges
C = curl_edge_face(grid)     # edges -> faces
D = div_face_cell(grid)      # faces -> cells

assert (C @ G).nnz == 0      # curl grad = 0, EXACTLY, on any grid
assert (D @ C).nnz == 0      # div curl  = 0, EXACTLY
```

Those identities hold to machine precision on arbitrarily distorted grids
because ±1 integers cancel exactly in floating point. Consequences: charge is
conserved to machine precision, and `div B = 0` holds for all time in the
magnetic solvers because `B` is only ever updated through a curl.

All metric and material information lives in diagonal **mass matrices**, and on
a rectilinear grid those are literally circuit elements:

```
M_sigma[e] = sigma * A_dual / L     conductance        [S]
M_eps[e]   = eps   * A_dual / L     capacitance        [F]
M_nu[f]    = A / (mu * L_dual)      inverse inductance [1/H]
```

So the electroquasistatic system

```
G.T @ M_sigma @ G @ phi = i_inject
```

**is** nodal analysis of a resistor mesh. A field region and a SPICE netlist are
the same kind of object, which is why coupling them (`circuit/coupling.py`) is
exact assembly via a Schur complement rather than an iterative handshake.

This is the Finite Integration Technique (Weiland); it coincides with Whitney-form
finite elements (Bossavit) and Tonti's cell method, and on a rectilinear grid all
three agree with Yee's original scheme.

---

## Solvers

| Solver | Equation | Captures | Status |
|---|---|---|---|
| `poisson` | `∇·(ε∇φ) = -ρ` | electrostatics, capacitance; nonlinear Poisson for semiconductor equilibrium | shipped |
| `eqs` | `∇·[(σ + ε∂ₜ)∇φ] = 0` | R, C, RC delay, crosstalk, IR drop, substrate coupling | shipped |
| `ac` | `∂ₜ → jω` | harmonic response, Y/Z/S parameters | shipped |
| `dd` | van Roosbroeck + Scharfetter–Gummel | semiconductor device physics | shipped (one open defect) |
| `fdtd` | explicit Yee leapfrog | full-wave reference, radiation | shipped (PML broken) |
| `circuit.mna` | modified nodal analysis | lumped netlists, SPICE subset | shipped |
| `circuit.coupling` | Schur complement | **field + netlist as one system** | shipped |
| `extraction` | — | C, R, RLGC, Z₀, S-parameters | shipped |
| `thermal` | `ρc ∂ₜT = ∇·(κ∇T) + q` | heat conduction, thermal R and C | shipped |
| `electrothermal` | the two above, coupled | self-heating, **thermal runaway** | shipped |
| `mqs` | `∇×(ν∇×A) + σ(∂ₜA + ∇φ) = J` | inductance, eddy currents, skin effect | **not implemented** |
| `darwin` | both, minus transverse displacement current | full R+L+C | **not implemented** |

> **Inductance is not yet modelled.** `mqs` and `darwin` are specified in
> [`docs/CONTRACTS.md`](docs/CONTRACTS.md) but not built. The quasi-static path
> today is resistive–capacitive only. `extraction.rlgc_2d` *does* return an
> inductance matrix, but it gets it from the LC identity (`L = μ₀ε₀ C_air⁻¹`),
> which is exact for a TEM line and says nothing about eddy currents or skin
> effect. If your problem is inductive, this release will not serve it.

Pick with `fieldspice.reference.electrical_length(size, t_rise=...)`:
below 0.01 quasi-static is excellent, above 0.3 you need `fdtd`.

---

## Assumptions

Every approximation is written down in **[`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md)**
with a tag (`A1`…`A14`), a validity condition, what it buys, and what it costs.
Solvers record the tags that were active in `Result.meta["assumptions"]`, so any
number this code produces can be traced back to the physics that produced it.

The headline ones:

- **A1 Quasi-static** — radiation dropped. Valid for `L/λ ≪ 1`. This is the
  approximation that makes the whole project possible.
- **A2 Rectilinear grid** — geometry is staircased. Exact for planar processes,
  which is most of microelectronics; sub-cell fill fractions recover much of the
  loss elsewhere.
- **A5/A7 Drift-diffusion + Scharfetter–Gummel** — good at and above ~100 nm.
  No ballistic transport, no quantum confinement.
- **A12 Neumann open boundaries** — underestimates fringing capacitance unless
  you pad the domain. `validate.check_padding` warns you.

`fieldspice.validate` turns each of these from documentation into a runtime
check against your actual geometry.

---

## Validation

Correctness is defined by [`fieldspice/reference.py`](fieldspice/reference.py),
a module of closed-form solutions that ships with the package so the acceptance
criteria are auditable and users can check their own setups.

Measured on the frozen core:

| Quantity | Result |
|---|---|
| `curl grad`, `div curl` | **exactly 0** (±1 integer entries cancel exactly) |
| Parallel-plate capacitance | 8.5e-14 relative |
| Slab resistance | 3.4e-14 relative |
| Series dielectric stack | matches series capacitors to 1.2e-14 |
| pn built-in potential | ≤7e-12 across **all 12** doping/nᵢ cases, 7–12 Newton iterations |
| Bernoulli (Scharfetter–Gummel core) | machine precision over x ∈ [−800, 700] |
| EQS step response | capacitive-divider initial condition exact; O(Δt) confirmed (1.216e-3 → 6.106e-4 → 3.059e-4 → 1.531e-4), **1 factorization per run** |
| MNA integration order | measured: backward Euler 2.00, trapezoidal 3.96, Gear-2 4.02 |
| Series RLC | all three damping regimes; under-damped peak 1.95150 vs 1.95153 |
| Diode ideality (drift-diffusion) | n = 1.0029 over 3.36 decades |
| FDTD propagation | at c; first-arrival matches 1/Courant exactly |
| AC admittance | Y = G + jωC to 2e-14 over 9 decades; reciprocity 1e-23 |
| RLGC extraction | C, L, Z₀ to 8e-14; **LC identity residual exactly 0** |
| Field↔circuit Schur complement | divider exact to 1e-12; Y·Δt = C to 2e-14; asymmetry 1e-16 |

143 tests pass; 2 are `xfail` for the open defects in
[`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md).

Some deliberate honesty about tolerances, because a test demanding more accuracy
than its reference formula possesses is a broken test:

- **Depletion width disagrees with the analytic value by ~20%, and that is
  correct.** The closed form is the depletion approximation with abrupt
  space-charge edges; the true solution has exponential tails a few Debye lengths
  long. The discrepancy does not shrink under refinement. We test the built-in
  potential (exact) and the reverse-bias *scaling* (asymptotically exact) instead.
- **Hammerstad–Jensen microstrip Z₀ is itself a ~1% curve fit**, so agreement to
  5% is agreeing as well as the reference deserves.
- **Silicon nᵢ is anchored to measurement (9.65e9 cm⁻³, Sproul & Green 1991),
  not computed from Nc/Nv/Eg.** The textbook constants reproduce 6.7e9 cm⁻³, not
  the ~1e10 they are quoted alongside; a simulator that uses the formula
  inherits a ~35% error in nᵢ and a ~2 kT/q error in every built-in potential.

---

## Prior art, and what this is not

This is not the first tool in this space, and it would be dishonest to imply
otherwise. Commercial TCAD (Sentaurus, Silvaco), commercial field solvers
(COMSOL, ANSYS Q3D/HFSS), and excellent open-source work
([DEVSIM](https://devsim.org), [Elmer](https://www.elmerfem.org),
[FastCap/FastHenry](https://www.rle.mit.edu/cpg/research_codes.htm),
[Ngspice](https://ngspice.sourceforge.net), [OpenEMS](https://www.openems.de))
all cover parts of it, and several are more mature than this in their own
domain.

What fieldspice does that is unusual is put **all** of it behind one grid, one
set of operators and one Python API: electrostatics, eddy currents, full Darwin
R+L+C, drift-diffusion device physics, a SPICE engine, and exact field↔circuit
coupling — with the full-wave solver included specifically so the quasi-static
approximation can be *checked* rather than trusted. The unifying formulation
above is what makes that practical: every solver is the same incidence matrices
with a different mass matrix.

It is **not** an optics tool. No nonlinear susceptibility, no gain media, no
mode solving. Use MEEP, which is excellent and which this is explicitly not
trying to replace. The domains are disjoint: MEEP lives where `L/λ ≫ 1`,
fieldspice where `L/λ ≪ 1`.

---

## Install

```bash
git clone https://github.com/Steve8323/fieldspice
cd fieldspice
pip install -e ".[all]"
pytest -q
```

Core requires only NumPy and SciPy. `matplotlib`, `h5py`, `pyamg` and `numba`
are all genuinely optional — the package imports and runs without any of them.

## Documentation

- [`docs/ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) — every approximation, tagged, with validity conditions
- [`docs/CONTRACTS.md`](docs/CONTRACTS.md) — interface spec and verified reference numbers
- [`examples/`](examples/) — runnable end-to-end cases

## License

MIT.
