# Assumptions, and what each one buys

This is the most important document in the repository. A field solver is only
as trustworthy as its stated approximations, and every assumption below is
there because it converts an intractable computation into a tractable one.
Each has a tag (`A1`, `A2`, ...) that appears in solver docstrings and in
`Result.meta["assumptions"]`, so any number this code produces can be traced
back to the physics that produced it.

Where an assumption has a validity condition, that condition is checkable, and
`fieldspice.validate` will check it for you and warn.

---

## The central problem: why you cannot just run MEEP on a circuit

MEEP and other full-wave FDTD codes solve Maxwell's equations explicitly. That
carries the Courant–Friedrichs–Lewy stability limit:

```
dt <= 1 / (c * sqrt(1/dx^2 + 1/dy^2 + 1/dz^2))
```

The time step is tied to the time light takes to cross the *smallest cell*.
Circuits are the worst possible case for this, because they combine tiny
features with long time scales:

| Problem | Smallest cell | Courant dt | Time of interest | Steps required |
|---|---|---|---|---|
| Gate oxide, 2 nm | 0.5 nm | 9.6e-19 s | 1 ns switching | **1.0e9** |
| On-chip wire, 10 nm | 10 nm | 1.9e-17 s | 100 ps edge | 5.2e6 |
| Package, 1 um | 1 um | 1.9e-15 s | 1 ns | 5.2e5 |
| PCB trace, 50 um | 50 um | 9.6e-14 s | 10 ns | 1.0e5 |
| Analog settling, 1 um | 1 um | 1.9e-15 s | 1 ms | **5.2e11** |

A billion time steps on a multi-million-cell grid is not a slow simulation, it
is an impossible one. Worse, conductors make explicit FDTD outright unstable
in a second way: the dielectric relaxation time of copper is

```
tau = eps0 / sigma = 8.85e-12 / 5.8e7 = 1.5e-19 s
```

which is ~200x *smaller* than the Courant step even at 0.5 nm resolution. Any
explicit scheme that resolves charge relaxation inside a metal is dead on
arrival.

**The resolution is that circuits do not need the wave.** Below, the single
assumption that makes this project possible.

---

## A1 — Quasi-static approximation (the big one)

**Statement.** In the `eqs`, `mqs`, `darwin` and `dd` solvers, the radiative
coupling between electric and magnetic fields is dropped. Concretely, one or
both of the induction terms is removed from Maxwell's equations:

- **A1a — electroquasistatic (EQS).** Drop `dB/dt` from Faraday's law, so
  `curl E = 0` and `E = -grad phi`. Charge conservation
  `div J + drho/dt = 0` with `J = sigma E + dD/dt` gives the single scalar
  equation

  ```
  div[ (sigma + eps d/dt) grad phi ] = 0
  ```

  Captures: resistance, capacitance, RC delay, capacitive crosstalk, IR drop,
  substrate coupling, dielectric relaxation, charge redistribution.
  Misses: inductance, magnetic energy, radiation.

- **A1b — magnetoquasistatic (MQS).** Drop the displacement current `dD/dt`
  from Ampere's law, so `curl H = J`. Solved in the `A`-`phi` form

  ```
  Cᵀ M_nu C a + M_sigma (da/dt + G phi) = i_src
  ```

  Captures: inductance, eddy currents, skin effect, proximity effect, AC
  resistance, magnetic energy. Misses: capacitive displacement current.

- **A1c — Darwin.** Keep both induction mechanisms but drop only the
  *transverse* (solenoidal) part of the displacement current, which is the
  term responsible for radiation. This retains full R, L and C coupling while
  removing the wave, and is the right model for essentially all chip, package
  and board interconnect below a few GHz.

**Validity.** The structure must be electrically small: `L << lambda`, where
`lambda = c / (f * sqrt(eps_r))` and `f ~ 0.35 / t_rise`. In practice:

| `L / lambda` | Verdict |
|---|---|
| < 0.01 | Quasi-static is excellent (<1% error). Use it. |
| 0.01 – 0.1 | Quasi-static good; Darwin recommended over EQS/MQS alone. |
| 0.1 – 0.3 | Marginal. Cross-check against `fdtd`. |
| > 0.3 | Wave effects and radiation matter. Use `fdtd`. |

A 100 um on-chip net with a 20 ps edge sits at `L/lambda ~ 0.008`: firmly
quasi-static. A 10 cm PCB trace with the same edge sits at `L/lambda ~ 8`:
firmly full-wave. `fieldspice.validate.check_quasistatic(grid, materials,
t_rise)` computes this ratio and warns.

**What it buys.** The system becomes elliptic in space and parabolic in time.
There is no wave to resolve, so there is no CFL condition, and implicit time
stepping is unconditionally stable. The step size is then set by *accuracy on
the signal*, typically 20–100 steps per rise time, instead of by the speed of
light. From the table above that is a **1e4 to 1e8 reduction in step count**,
and it is why this project exists.

**Cost.** Each step requires solving a sparse linear system rather than doing
an explicit update. With a cached factorisation (constant `dt`) or algebraic
multigrid, that is roughly 3–30x the cost of one explicit step — utterly
dominated by the 1e4–1e8 saving in step count.

**Honesty note.** The quasi-static solvers cannot produce a radiated far
field, cannot model an antenna, and will not show a resonance whose mechanism
is a standing wave. If your answer depends on any of those, the model is wrong
and no amount of mesh refinement will fix it. This is a modelling error, not a
discretisation error, so it does *not* shrink under refinement — the single
most dangerous failure mode of any quasi-static tool.

---

## A2 — Rectilinear tensor-product grid; staircased geometry

Only axis-aligned graded rectilinear grids are supported. Curved and slanted
boundaries are staircase-approximated.

**Buys.** Every operator is a banded matrix with an exact known stencil;
assembly is O(N) with no mesh generator; there is no element-quality failure
mode; and the discrete calculus identities (`curl grad = 0`, `div curl = 0`)
hold *exactly* rather than to within mesh quality.

**Costs.** Accuracy on a staircased surface degrades from second to
approximately first order. Fields at a staircased conductor corner are
over-estimated (the classic sharp-corner artifact).

**Mitigations implemented.** (i) Sub-cell fill fractions from
`geometry.voxelize` feed effective-medium material mixing, which recovers much
of the loss for smooth interfaces; (ii) `auto_mesh_1d` grades the mesh so
interfaces can be resolved cheaply; (iii) for planar processes — which is most
of microelectronics — geometry is *exactly* axis-aligned and this assumption
costs nothing at all.

---

## A3 — Linear, isotropic, non-dispersive, time-invariant materials (default)

`eps`, `mu` and `sigma` are real scalars per cell.

**Buys.** Diagonal mass matrices, symmetric positive-definite systems, and
therefore conjugate-gradient / Cholesky instead of GMRES / LU — a 3–10x
speed difference, and a guarantee of a unique solution.

**Escape hatches.** The `ac` solver accepts complex permittivity
`eps' - j eps''` per frequency, which covers dielectric loss and Debye/
Cole–Cole relaxation exactly. The `fdtd` solver supports Drude and Lorentz
poles via auxiliary differential equations. Anisotropic (tensor) materials and
ferroelectrics are **not** supported; a diagonal-tensor extension is
straightforward within this formulation, off-diagonal is not.

---

## A4 — Conductor treatment

**A4a (quasi-static).** Conductors are volumetric `sigma` regions. This is
exact for EQS/MQS/Darwin and correctly reproduces skin and proximity effects
*provided the mesh resolves the skin depth* `delta = sqrt(2/(omega mu sigma))`.
At 1 GHz in copper `delta = 2.1 um`; at 1 MHz it is 66 um. `fieldspice.validate`
warns when cells in a conductor are coarser than `delta/3`.

**A4b (full-wave).** Gridding through the skin depth in FDTD is usually
prohibitive, so the default for a good conductor is a surface-impedance
boundary condition (SIBC), valid when `delta << feature size`. Perfect
conductors (`sigma = inf`) are handled by simply removing their interior
unknowns, which is both exact and cheap.

**Not modelled.** Anomalous skin effect, surface roughness (a real and often
dominant loss term above ~5 GHz on PCB copper), grain-boundary scattering in
sub-100 nm wires, and superconductivity.

---

## A5 — Drift-diffusion transport for semiconductors

Carrier transport uses the van Roosbroeck system: Poisson plus electron and
hole continuity, with `J_n = q mu_n n E + q D_n grad n` and its hole
counterpart.

**Assumes.** Local, instantaneous relation between current and field/gradient;
carriers in thermal equilibrium with the lattice; parabolic bands; Boltzmann
statistics by default (Fermi–Dirac optional, and required above ~1e19 cm^-3);
the Einstein relation `D = mu kT/q`.

**Not modelled.** Velocity overshoot and ballistic transport (needs
hydrodynamic or Monte Carlo — matters below ~50 nm channel length), quantum
confinement and tunnelling (matters for oxides below ~3 nm and for
FinFET/nanosheet inversion layers), band-to-band tunnelling, hot-carrier
effects, impact ionisation (available as an optional generation term only).

**Buys.** Three coupled scalar PDEs instead of a 6-D Boltzmann transport
equation. This is the standard TCAD compromise and is quantitatively good for
devices at or above roughly 100 nm — which includes every technology this
project's users are likely to fabricate (IGZO TFTs, thin-film transistors,
180–500 nm CMOS, discrete power devices).

---

## A6 — Isothermal operation

Lattice temperature is uniform and constant (default 300 K). No self-heating.

**Buys.** Removes a fourth coupled PDE (heat) and its strong nonlinear
feedback through mobility and `ni`, roughly halving cost and greatly improving
Newton convergence.

**When it breaks.** Power devices, high-current analog, and anything where
`I*V / thermal_conductance` exceeds a few kelvin. An optional lattice heat
equation is provided but is *not* on by default and is not validated to the
same standard as the electrical solvers.

---

## A7 — Scharfetter–Gummel exponential fitting

Carrier flux on an edge uses the Scharfetter–Gummel form with the Bernoulli
function rather than central differencing.

This is not an optimisation, it is a correctness requirement. Central
differencing of the drift-diffusion current produces negative carrier
concentrations and oscillatory garbage as soon as the potential drop across a
cell exceeds about `2 kT/q = 52 mV`. Since a 1 V junction drop falls across
~50 nm, resolving it by brute force would need cells of ~1 nm; SG makes the
scheme stable at any cell size, at the cost of assuming the potential varies
*linearly* and the carrier density *exponentially* along each edge.

---

## A8 — Staggered (weak) Darwin coupling by default

In `darwin`, the electric and magnetic subproblems are solved sequentially
within a time step, exchanging the inductive EMF, rather than as one monolithic
system.

**Buys.** Two well-conditioned symmetric solves instead of one large indefinite
coupled system, and reuse of the EQS and MQS machinery unchanged.

**Costs.** Introduces an O(dt) splitting error in the R–L–C coupling. Set
`coupling="picard"` with `max_inner > 1` to iterate it away when precision
matters; the splitting error is then reduced to the inner tolerance.

---

## A9 — No optical or photonic modelling

`fieldspice` deliberately does not target optics. No nonlinear susceptibility,
no gain media, no mode solving, no dispersion engineering. For that, use MEEP —
which is excellent, and which this project is explicitly *not* trying to
replace. The domains are disjoint: MEEP lives where `L/lambda >> 1`,
`fieldspice` lives where `L/lambda << 1`.

---

## A10 — Box-method (finite-volume) discretisation on the dual grid

Scalar unknowns live on nodes with control volumes given by the dual boxes.
Second-order accurate on smooth uniform meshes; formally reduces toward first
order where the mesh grades sharply. `grid.max_growth_ratio()` reports the
worst neighbouring-cell ratio, and solvers warn above 2.0. Keep grading below
~1.4 for production runs.

---

## A11 — Complete ionisation of dopants

Donors and acceptors are fully ionised (`Nd+ = Nd`, `Na- = Na`). Wrong below
about 100 K, and increasingly wrong for deep dopants and wide-bandgap
materials (notably Al acceptors in SiC, and the deep traps that dominate
amorphous-oxide semiconductors such as IGZO). An incomplete-ionisation model
is available but off by default.

---

## A12 — Open-boundary truncation

The default quasi-static boundary is homogeneous Neumann: no normal current or
flux crosses the domain wall. This is exact for a symmetry plane and for a
shielded enclosure, but for an *open* problem it artificially confines the
field and **under-estimates fringing capacitance**. Pad the domain to at least
3x the largest feature dimension; `validate.check_padding` warns otherwise.
The `fdtd` solver has a proper CPML absorbing boundary; the quasi-static
solvers do not (an infinite-domain quasi-static boundary needs a BEM coupling,
which is not implemented).

---

## A13 — Frozen geometry and mesh

No moving boundaries, no adaptive mesh refinement during a solve, no MEMS
displacement. The mesh is built once. Adaptivity is a plausible extension; it
is not implemented, and no result depends on it.

---

## A14 — Deterministic, single-instance device physics

Solvers compute one nominal device. Process variation, random dopant
fluctuation and mismatch are **not** propagated through the field solve;
`circuit/devices.py` exposes mismatch parameters at the compact-model level
instead (`vth_sigma`, `beta_sigma`), so statistical analysis happens in the
circuit domain where a Monte Carlo loop is affordable. Running 1000 field
solves to get a mismatch distribution is possible but is the user's decision,
not a default.

---

## Summary: the speedup ledger

For a representative on-chip problem (100 um net, 10 nm minimum feature,
1 ns of simulated time):

| | Explicit full-wave | fieldspice quasi-static |
|---|---|---|
| Time step | 1.9e-17 s (Courant) | 1e-11 s (accuracy) |
| Steps | 5.2e7 | 100 |
| Cost per step | 1 explicit sweep | ~10 explicit sweeps (cached factorisation) |
| **Relative total cost** | **1** | **~2e-5** |

Roughly a **50,000x** reduction, obtained entirely by discarding a physical
effect (radiation) that contributes nothing to the answer in this regime. Every
solver reports `speedup_vs_courant()` so this claim is measured per run, not
asserted.
