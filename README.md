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

For a representative on-chip problem that is a **~50,000× reduction in cost**,
obtained entirely by discarding a physical effect that was not doing any work.
Every solver reports `speedup_vs_courant()` so the claim is measured per run
rather than asserted.

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

| Solver | Equation | Captures | Misses |
|---|---|---|---|
| `poisson` | `∇·(ε∇φ) = -ρ` | electrostatics, capacitance | everything dynamic |
| `eqs` | `∇·[(σ + ε∂ₜ)∇φ] = 0` | R, C, RC delay, crosstalk, IR drop, substrate coupling | inductance, radiation |
| `mqs` | `∇×(ν∇×A) + σ(∂ₜA + ∇φ) = J` | inductance, eddy currents, skin & proximity effect | displacement current |
| `darwin` | both, minus the transverse displacement current | full R+L+C | radiation |
| `ac` | `∂ₜ → jω` | all of the above, per frequency | — |
| `dd` | van Roosbroeck + Scharfetter–Gummel | semiconductor device physics | ballistic, quantum |
| `fdtd` | explicit Yee leapfrog | everything, including radiation | nothing (but Courant-limited) |
| `circuit.mna` | modified nodal analysis | lumped netlists | fields |
| `circuit.coupling` | Schur complement | **field + netlist as one system** | — |

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
| `curl grad`, `div curl` | exactly 0 |
| Parallel-plate capacitance | 1.5e-14 relative error |
| Slab resistance | 3.4e-14 relative error |
| Series dielectric stack | matches series capacitors to 1e-10 |
| EQS step response | correct capacitive-divider initial condition; O(Δt) convergence confirmed (1.22e-3 → 6.11e-4 → 3.06e-4) |
| pn junction built-in potential | 6.7e-16 relative, 9–11 Newton iterations |

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
