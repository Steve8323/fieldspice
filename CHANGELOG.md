# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning is [semantic](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-08-05

First public release. Alpha: the API is not yet stable.

### Core

- Graded rectilinear Yee grid with 1D/2D/3D support via collapsed dimensions.
- Discrete exterior calculus on integrated variables: gradient, curl and
  divergence are exact signed-incidence matrices. `curl grad = 0` and
  `div curl = 0` hold **exactly** (verified `nnz == 0`), on any grid.
- Diagonal mass matrices that are literally circuit elements — conductance [S],
  capacitance [F], inverse inductance [1/H].
- `reference.py`: closed-form oracle shipped with the package, used as the
  acceptance criterion for everything else.

### Solvers

- `poisson` — electrostatics, Maxwell and SPICE capacitance matrices, and a
  nonlinear Poisson solver for semiconductor equilibrium.
- `eqs` — electroquasistatic transient (R, C, RC delay, crosstalk, IR drop,
  substrate coupling).
- `ac` — frequency domain, Y/Z/S parameters.
- `dd` — drift-diffusion with Scharfetter–Gummel exponential fitting.
- `fdtd` — explicit full-wave Yee, present as a reference for checking the
  quasi-static approximation rather than as the fast path.
- `thermal` — Fourier heat conduction, steady and transient, with Dirichlet,
  adiabatic and convective (Robin) boundaries.
- `electrothermal` — coupled self-heating with thermal-runaway detection.
- `circuit.mna` — modified nodal analysis with a SPICE-syntax subset.
- `circuit.devices` — R, L, C, sources, diode, MOSFET level-1, EKV,
  subthreshold TFT, switches.
- `circuit.coupling` — exact field↔netlist coupling by Schur complement.
- `extraction` — C, R, per-unit-length RLGC, Z₀, S-parameters.

### Physics options

- `PhysicsOptions` gates optional mechanisms, each tied to the assumption it
  relaxes, so `Result.meta["assumptions"]` stays truthful automatically.
- Self-heating, temperature-dependent σ and κ, field-dependent mobility,
  Fermi–Dirac statistics, incomplete ionisation, impact ionisation.

### Validation

173 tests. Selected measurements:

| Quantity | Result |
|---|---|
| `curl grad`, `div curl` | exactly 0 |
| Parallel-plate capacitance | 8.5e-14 relative |
| Slab resistance | 3.4e-14 relative |
| pn built-in potential | ≤7e-12 across 12 doping/nᵢ cases |
| EQS backward Euler | O(Δt) confirmed, 1 factorisation per run |
| MNA integration order | BE 2.00, trapezoidal 3.96, Gear-2 4.02 |
| Diode ideality | n = 1.0029 over 3.36 decades |
| Heat equation | second order confirmed (4.01, 4.00, 4.00) |
| Thermal vs independent MNA network | 1.2e-4 K in a 666.7 K rise |
| Electro-thermal fixed point | ~1e-6, both drive modes |
| RLGC | C, L, Z₀ to 8e-14; LC identity residual exactly 0 |
| Field↔circuit Schur complement | divider exact to 1e-12 |

### Known limitations

See `docs/KNOWN_ISSUES.md`. In brief: the FDTD CPML reflects −0.58 dB and does
not absorb; drift-diffusion has a small spurious current at zero bias;
`mqs`/`darwin` are unimplemented so **there is no inductance anywhere**; no
noise, no dispersion, no anisotropy, no ferroelectrics, no ballistic transport.
Not an optics tool — use MEEP.
