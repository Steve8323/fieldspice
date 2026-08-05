# Implementation contracts

**Every implementer must read this file before writing a line of code.**
It fixes the interfaces so that modules written independently compose. Do not
change a signature listed here; if a signature is genuinely wrong, say so in
your report rather than silently diverging.

## Ground rules

1. **Strict SI everywhere.** metres, seconds, volts, amps, kelvin. No scaling.
   Use `fieldspice.units` multipliers (`um`, `ns`, `pF`, `per_cm3`).
2. **NumPy + SciPy only** for core numerics. Optional accelerators
   (`numba`, `pyamg`, `mlx`) must be imported lazily inside a `try` and the
   module must work without them.
3. **No global state.** No module-level mutable singletons.
4. Python 3.10+, `from __future__ import annotations`, full type hints.
5. Docstrings are NumPy style and must state **units** of every physical
   argument and return value.
6. Every physical approximation gets a tag from `docs/ASSUMPTIONS.md`
   (`A1`..`A14`) named in the docstring, and appears in
   `Solver.assumptions`.
7. Raise `ValueError` on bad input eagerly. Never silently coerce a wrong
   shape.
8. **Do not** modify `grid.py`, `operators.py`, `units.py`, or
   `solvers/base.py` — they are frozen and already validated.

## Frozen core (already built and tested)

```python
from fieldspice.grid import RectilinearGrid, graded_1d, auto_mesh_1d
from fieldspice.operators import (
    grad_node_edge, curl_edge_face, div_face_cell,   # +-1 incidence matrices
    edge_mass, face_mass_nu, node_volume_vector,
    cell_to_edge, cell_to_face, cell_to_node,
    nodal_laplacian, curl_curl, apply_dirichlet,
    split_edge_vector, split_face_vector, Operators,
)
from fieldspice.solvers.base import (
    SolverBase, TimeSteppingSolver, Result, SolverConfig,
    Terminal, ConvergenceError,
)
```

### Discrete calculus you must build on

| Symbol | Call | Maps | Entries |
|---|---|---|---|
| `G` | `grad_node_edge(grid)` | nodes → edges | ±1 |
| `C` | `curl_edge_face(grid)` | edges → faces | ±1 |
| `D` | `div_face_cell(grid)` | faces → cells | ±1 |

`C @ G == 0` and `D @ C == 0` **exactly** (verified). All geometry lives in
diagonal mass matrices:

- `edge_mass(grid, sigma_edge)` → conductance [S] per edge
- `edge_mass(grid, eps_edge)` → capacitance [F] per edge
- `face_mass_nu(grid, mu_face)` → inverse inductance [1/H] per face

Composite operators:

- `nodal_laplacian(grid, prop_edge)` = `Gᵀ M G`, symmetric PSD.
  With σ it is the **nodal conductance matrix** (`L φ = I_inject` is KCL);
  with ε it is the **capacitance matrix** (`L φ = Q_node` is Gauss' law).
- `curl_curl(grid, mu_face)` = `Cᵀ M_ν C`, the reluctance operator.

### Sign conventions (get these right)

- `G φ` is the potential **rise** along the edge. Field circulation
  `e = -G φ`.
- Edge current along the edge direction: `i = -M_σ (G φ)`.
- `Gᵀ i` is net current **into** each node.
- Therefore `Gᵀ M_σ G φ = I_inject` with `I_inject` the current pushed into
  the node from outside. Same sign structure for `Gᵀ M_ε G φ = Q_node`.

### Array shapes

```
nodes  (Nx+1, Ny+1, Nz+1)      flat length grid.n_nodes
cells  (Nx,   Ny,   Nz)        flat length grid.n_cells
edges  concatenated [x,y,z], flat length grid.n_edges
       x:(Nx, Ny+1, Nz+1)  y:(Nx+1, Ny, Nz+1)  z:(Nx+1, Ny+1, Nz)
faces  concatenated [x,y,z], flat length grid.n_faces
       x:(Nx+1, Ny, Nz)    y:(Nx, Ny+1, Nz)    z:(Nx, Ny, Nz+1)
```

Flattening is always C-order `.ravel()`. Use `split_edge_vector` /
`split_face_vector` to unpack.

**2D and 1D** are grids with `Nz == 1` (and `Ny == 1`). Never special-case
dimensionality; the operators already handle it. Collapsed directions have a
finite thickness (default 1 m) so results are per-unit-length.

> **Gotcha, read this twice.** A collapsed direction still has `N+1 = 2` node
> planes. A "1D" grid with 136 cells has node shape `(137, 2, 2)` and
> `n_nodes = 548`, **not** 137. Never index a flat node vector as if it were
> 1D. Build nodal quantities with `grid.node_coords()` (which returns properly
> broadcast `(Nx+1, Ny+1, Nz+1)` arrays) and `.ravel()` them, and select
> electrodes with `np.arange(n_nodes).reshape(grid.shape_nodes)[0].ravel()`.
> The solution is uniform across collapsed directions — assert that in your
> tests, it is a good correctness check.

## Module assignments and required public API

Each module below is owned by exactly one implementer. Create only your file.

### `fieldspice/materials.py`
```python
@dataclass(frozen=True)
class Material:
    name: str
    eps_r: float = 1.0          # relative permittivity
    mu_r: float = 1.0
    sigma: float = 0.0          # [S/m]
    kind: str = "dielectric"    # dielectric|conductor|semiconductor|vacuum
    semi: SemiconductorParams | None = None
    color: str = "#888888"

@dataclass(frozen=True)
class SemiconductorParams:
    Eg: float; ni_300: float; chi: float
    mu_n: float; mu_p: float          # [m^2/(V s)] low field
    Nc_300: float; Nv_300: float
    tau_n: float; tau_p: float        # SRH lifetimes [s]
    vsat_n: float; vsat_p: float
    C_auger_n: float; C_auger_p: float
    def ni(self, T: float) -> float: ...
    def Nc(self, T: float) -> float: ...
    def Nv(self, T: float) -> float: ...

LIBRARY: dict[str, Material]   # vacuum, air, si, sio2, si3n4, cu, al, w,
                               # poly, fr4, alumina, aln, scaln, igzo,
                               # a_si, ito, mo, ti, pt, gaas, sic, hfo2
def get(name: str) -> Material
def register(mat: Material) -> None
class MaterialMap:              # per-cell assembly
    def __init__(self, grid, background="vacuum")
    def assign(self, mask: np.ndarray, material: str | Material) -> None
    def eps(self) -> np.ndarray      # (Nx,Ny,Nz) absolute [F/m]
    def mu(self) -> np.ndarray       # absolute [H/m]
    def sigma(self) -> np.ndarray    # [S/m]
    def ids(self) -> np.ndarray      # int cell array
    def material_at(self, i, j, k) -> Material
```
Conductivities: Cu 5.8e7, Al 3.77e7, W 1.79e7, Mo 1.87e7, Ti 2.38e6,
Pt 9.4e6, ITO 1e6, poly-Si 1e4. SiO2 eps_r 3.9, Si3N4 7.5, Si 11.7,
FR4 4.4, AlN 8.9, HfO2 25, alumina 9.8, IGZO 16 (mu_n ~ 10 cm^2/Vs).

### `fieldspice/geometry.py`
```python
class Shape(ABC):
    def contains(self, x, y, z) -> np.ndarray   # broadcast bool
    def bbox(self) -> tuple[tuple[float,float], ...]
    def __or__/__and__/__sub__/__invert__       # CSG
class Box(Shape):        # center, size  (or lo=, hi=)
class Cylinder(Shape):   # center, radius, height, axis="z"
class Sphere(Shape)
class HalfSpace(Shape)   # point, normal
class Prism(Shape)       # polygon vertices + extrusion along axis
class Torus(Shape)
def voxelize(grid, shape, subsample: int = 2) -> np.ndarray
    # (Nx,Ny,Nz) float FILL FRACTION in [0,1] via subsample^3 sampling.
    # subsample=1 gives a hard 0/1 staircase mask.
class LayerStack:        # planar-process helper, the common case for chips
    def __init__(self, grid, axis="z", origin=0.0)
    def add(self, name, thickness, material, mask: Shape | None = None)
    def apply(self, matmap: MaterialMap) -> None
    def z_of(self, name) -> tuple[float, float]
```
`voxelize` returning a **fill fraction** (not a bool) is deliberate: it lets
`MaterialMap` do sub-cell effective-medium mixing, which recovers a large part
of the staircase error at ~0 cost. Document that mixing eps linearly is
correct for fields parallel to the interface and harmonic mixing is correct
for perpendicular; default to linear and expose the choice.

### `fieldspice/boundaries.py`
```python
class BC(ABC): ...
class Dirichlet(BC):   value: float | Callable[[float], float]
class Neumann(BC):     flux: float = 0.0     # natural / symmetry
class Periodic(BC)
class Absorbing(BC):   # full-wave only (CPML); order, sigma_max, kappa_max
class Symmetry(BC):    # PEC/PMC mirror
@dataclass
class BoundarySpec:
    xlo: BC; xhi: BC; ylo: BC; yhi: BC; zlo: BC; zhi: BC
    @classmethod
    def all_neumann(cls); def all_dirichlet(cls, v=0.0)
    def node_masks(self, grid) -> dict[str, np.ndarray]   # flat node indices
    def dirichlet_nodes(self, grid, t=0.0) -> tuple[np.ndarray, np.ndarray]
```
Default for quasi-static: **Neumann** on all walls (zero normal current /
flux). Document that Neumann on an open boundary means "no field leaves",
which for capacitance extraction *underestimates* fringing unless the box is
padded — recommend >=3x the feature size of padding, and say so in the
docstring.

### `fieldspice/sources.py`
```python
def step(t0, v0=0.0, v1=1.0, trise=0.0) -> Callable[[float], float]
def pulse(t0, width, v0=0, v1=1, trise=0, tfall=0, period=None)
def sine(freq, amp=1.0, phase=0.0, offset=0.0)
def pwl(times, values)                       # linear interpolation
def gaussian(t0, tau, amp=1.0)
def ramp(slope, t0=0.0)
def prbs(bit_period, order=7, v0=0.0, v1=1.0, seed=0, trise=None)
def trapezoid_clock(period, duty=0.5, trise=None, tfall=None, v0=0, v1=1)
class Waveform:      # wraps a callable, supports + * and .delay(), .sampled()
```
All return plain callables `f(t) -> float` (vectorised over array `t` too).

### `fieldspice/monitors.py`
```python
class Monitor(ABC):
    name: str
    def record(self, state: dict, t: float) -> None
    def finalize(self) -> dict[str, np.ndarray]
class NodeProbe(Monitor)        # potential at a point or node index
class TerminalProbe(Monitor)    # V and I at a Terminal
class FieldSnapshot(Monitor)    # store named field every `every` steps
class EnergyMonitor(Monitor)    # electric/magnetic stored energy, dissipation
class FluxMonitor(Monitor)      # current through an oriented surface
class ChargeMonitor(Monitor)
class MonitorSet:               # container, dispatches record/finalize
```

### `fieldspice/solvers/poisson.py`
```python
class PoissonSolver(SolverBase):
    name = "poisson"
    def __init__(self, grid, eps_cell, config=None, operators=None)
    def solve(self, terminals, bc=None, rho_node=None) -> Result
    def capacitance_matrix(self, terminals, bc=None) -> np.ndarray
        # Maxwell capacitance matrix [F], (nt, nt), by unit-excitation
class NonlinearPoissonSolver(PoissonSolver):
    # rho(phi) callback + analytic dRho/dphi; Newton with line search.
    # This is the equilibrium (zero-current) semiconductor solve and must
    # reproduce the depletion approximation for an abrupt junction.
def solve_linear(A, b, cfg) -> np.ndarray    # shared backend dispatcher
```
`solve_linear` is the one place that chooses direct vs CG vs AMG. Everything
else calls it. Must handle the singular all-Neumann case by pinning one node
and warning.

### `fieldspice/solvers/eqs.py`
```python
class EQSSolver(TimeSteppingSolver):
    name = "eqs"; assumptions = ("A1", "A3", "A5")
    # Solves  Gᵀ(M_sigma + M_eps d/dt) G phi = i_inject
    # i.e. div[(sigma + eps d/dt) grad phi] = 0
    def __init__(self, grid, eps_cell, sigma_cell, config=None, ...)
    def solve(self, terminals, t_end, dt, bc=None, theta=1.0,
              monitors=None, rho_node=None) -> Result
    def steady_state(self, terminals, bc=None) -> Result   # DC / resistive
```
`theta` selects backward Euler (1.0) or trapezoidal (0.5). Backward Euler is
L-stable and the default because circuit inputs have discontinuous
derivatives; trapezoidal ringing on a step edge is a classic SPICE artifact
and must be documented as such. Factorise once when `dt` is constant and
reuse — this is the difference between usable and unusable.

### `fieldspice/solvers/mqs.py`
```python
class MQSSolver(TimeSteppingSolver):
    name = "mqs"; assumptions = ("A1b", "A3", "A4")
    # Cᵀ M_nu C a + M_sigma (da/dt + G phi) = i_src, with A-phi formulation.
    # Handles eddy currents, skin and proximity effect, internal inductance.
    def solve(self, terminals, t_end, dt, bc=None, monitors=None) -> Result
    def inductance_matrix(self, loops, freq=0.0) -> np.ndarray
```
The curl-curl operator is singular in non-conducting regions (gradient null
space). Handle it: either add a gauge/regularisation term `epsilon*Gᵀ...G`,
use a tree-cotree gauge, or rely on a Krylov solver with a consistent RHS.
State which you chose and why in the docstring, and prove the choice with a
skin-depth test.

### `fieldspice/solvers/darwin.py`
```python
class DarwinSolver(TimeSteppingSolver):
    name = "darwin"; assumptions = ("A1c", "A3", "A8")
    # Full R + L + C, no radiation: the electroquasistatic and magneto-
    # quasistatic systems solved together with inductive back-EMF.
    def solve(self, terminals, t_end, dt, bc=None, coupling="staggered",
              max_inner=1, monitors=None) -> Result
```
`coupling="staggered"` = one EQS solve then one MQS solve per step, feeding
`-da/dt` back as an EMF (cheap, documented as A8, error O(dt) in the coupling).
`coupling="picard"` = iterate to convergence within the step (`max_inner`).
Validate against a known RLC ring-down and against the transmission-line
solution below the wave regime.

### `fieldspice/solvers/ac.py`
```python
class ACSolver(SolverBase):
    name = "ac"
    # Complex frequency-domain: replace d/dt with j*omega. One complex solve
    # per frequency instead of thousands of time steps.
    def __init__(self, grid, eps_cell, sigma_cell, mu_cell=None, mode="eqs")
    def solve(self, terminals, freqs, bc=None) -> Result   # mode eqs|mqs|darwin
    def admittance_matrix(self, terminals, freqs) -> np.ndarray  # (nf,nt,nt)
    def impedance_matrix(...); def s_parameters(..., z0=50.0)
```

### `fieldspice/solvers/fdtd.py`
```python
class FDTDSolver(TimeSteppingSolver):
    name = "fdtd"; assumptions = ("A3", "A4b", "A9")
    # Explicit leapfrog Yee on the SAME grid/operators:
    #   b -= dt * C e ;  e += dt * M_eps^-1 (Cᵀ M_nu b - M_sigma e - i_src)
    # Exists to VALIDATE the quasi-static approximation, and to cover the
    # regime where it fails. Reference implementation, not the fast path.
    def __init__(self, grid, eps_cell, sigma_cell, mu_cell=None, config=None)
    def solve(self, sources, t_end, dt=None, bc=None, monitors=None) -> Result
    def stable_dt(self, safety=0.99) -> float
class CPML: ...        # convolutional PML, needed for open problems
```
Lossy media: use exponential time differencing so a conductive cell cannot go
unstable. Good conductors must NOT be gridded through the skin depth by
default — document the surface-impedance alternative (A4b).

### `fieldspice/solvers/dd.py`
```python
class DriftDiffusionSolver(SolverBase):
    name = "drift_diffusion"; assumptions = ("A5", "A6", "A7", "A10", "A11")
    # van Roosbroeck system on nodes:
    #   Gᵀ M_eps G psi = q(p - n + Nd - Na) * Vnode
    #   dn/dt = +(1/q) div Jn - R ;  dp/dt = -(1/q) div Jp - R
    # Edge currents use SCHARFETTER-GUMMEL exponential fitting -- mandatory,
    # central differencing is unconditionally wrong once the potential drop
    # across an edge exceeds ~2 kT/q.
    def __init__(self, grid, matmap, doping_node, T=300.0, config=None)
    def equilibrium(self) -> np.ndarray                 # psi at zero bias
    def solve_dc(self, terminals, bc=None, ramp=True) -> Result
    def solve_transient(self, terminals, t_end, dt, ...) -> Result
    def iv_curve(self, sweep_terminal, values, others) -> Result
def bernoulli(x) -> np.ndarray     # B(x)=x/(exp(x)-1), numerically safe
                                   # for |x| < 1e-10 and |x| > 700
```
Newton on the coupled `(psi, n, p)` block system, with a Gummel fallback.
Scale the variables (psi by kT/q, n and p by a reference density) or the
Jacobian condition number will be ~1e30. Must reproduce: 60 mV/decade
subthreshold slope at 300 K, Shockley diode ideality, depletion-width
scaling. Use `q*Vnode` weighting so the box method conserves charge.

### `fieldspice/circuit/mna.py`
```python
class Netlist:
    def add_resistor(self, name, n1, n2, r); add_capacitor; add_inductor
    def add_vsource(self, name, n1, n2, value)   # value: float|callable
    def add_isource(...); def add_device(self, dev: Device)
    def node_index(self, name) -> int
    @classmethod
    def from_spice(cls, text: str) -> "Netlist"   # subset of SPICE syntax
class MNASolver:
    def dc(self) -> dict[str, float]
    def transient(self, t_end, dt, method="be") -> Result   # be|trap|gear2
    def ac(self, freqs) -> Result
    def stamp(self, t, x_prev, dt) -> tuple[sp.csr_matrix, np.ndarray]
```
Standard modified nodal analysis: node voltages plus branch currents for
voltage sources and inductors. Companion models for reactive elements.

### `fieldspice/circuit/devices.py`
```python
class Device(ABC):
    nodes: tuple[str, ...]
    def stamp_dc(self, G, I, x, nmap) -> None
    def stamp_tran(self, G, I, x, x_prev, dt, nmap) -> None
class Resistor/Capacitor/Inductor/VSource/ISource/VCVS/VCCS(Device)
class Diode(Device)          # Shockley + series R + junction cap
class MOSFETL1(Device)       # Shockley level-1 with subthreshold + CLM
class EKV(Device)            # continuous weak-to-strong inversion
class SubthresholdTFT(Device)  # exponential I-V; the analog-NN workhorse
class Switch(Device)
```
`SubthresholdTFT` matters: it is the device behind analog exponential/softmax
cells, so it must be accurate over many decades of current and expose
mismatch parameters (`vth_sigma`, `beta_sigma`).

### `fieldspice/circuit/coupling.py`
```python
class FieldCircuitSystem:
    # THE point of this project: a meshed field region and a lumped netlist
    # solved as ONE system, not co-simulated with a handshake.
    def __init__(self, field_solver, netlist, terminal_map: dict[str, str])
    def transient(self, t_end, dt, monitors=None) -> Result
    def dc(self) -> Result
    def terminal_admittance(self, t, dt) -> np.ndarray   # dI/dV, (nt, nt)
```
Method: eliminate interior field unknowns onto the terminals (a Schur
complement giving a dense `nt x nt` admittance plus a history term), stamp
that block into the MNA matrix, solve the small joint system, then back-
substitute the interior. Exact, not iterative, when the field region is
linear. Document the cost: one sparse factorisation plus `nt` back-solves per
distinct `dt`.

### `fieldspice/extraction.py`
```python
def capacitance_matrix(grid, eps_cell, terminals, bc=None) -> np.ndarray
def resistance_matrix(grid, sigma_cell, terminals, bc=None) -> np.ndarray
def inductance_matrix(grid, mu_cell, sigma_cell, terminals, freq) -> np.ndarray
def rlgc_2d(grid, eps_cell, sigma_cell, mu_cell, conductors, freq=0.0) -> dict
    # per-unit-length R, L, G, C matrices from a 2D cross-section
def characteristic_impedance(rlgc, freq) -> np.ndarray
def s_parameters(y, z0=50.0) -> np.ndarray
def elmore_delay(R, C) -> float
def skin_depth(sigma, freq, mu=mu0) -> float
```

### `fieldspice/viz.py`
```python
def plot_grid(grid, ax=None, **kw)
def plot_materials(matmap, plane="z", index=None, ax=None)
def plot_scalar(grid, field, plane="z", index=None, ax=None, log=False)
def plot_vector(grid, edge_vec, plane="z", index=None, ax=None, stride=2)
def plot_terminals(result, ax=None)
def plot_iv(result, ax=None, log=False)
def animate(result, field, plane="z", fps=20, path=None)
def plot_convergence(result, ax=None)
def eye_diagram(t, v, period, ax=None)
```
matplotlib only, no seaborn. Every function accepts and returns an `Axes`.
Never call `plt.show()` inside library code.

### `fieldspice/io.py`
```python
def save_result(result, path) -> None      # .h5 via h5py if present, else .npz
def load_result(path) -> Result
def save_grid(grid, path); def load_grid(path)
def export_vtk(grid, fields: dict, path) -> None   # legacy .vtr, ParaView
def to_touchstone(freqs, s, path, z0=50.0) -> None # .sNp
```
`h5py` is optional: fall back to `.npz` and say so. Never hard-require it.

## Testing contract

Put tests in `tests/test_<module>.py`, pytest style. Every module needs:

- a **shape/contract test** (wrong input raises `ValueError`),
- at least one **analytic ground-truth test** with a quantitative tolerance,
- for solvers, a **convergence test** showing the expected order.

Analytic references that must be reproduced (these are the acceptance
criteria, not suggestions):

| Test | Reference | Tolerance |
|---|---|---|
| parallel-plate C | `eps*A/d` | 1e-6 rel |
| coaxial C, L | `2*pi*eps/ln(b/a)`, `mu*ln(b/a)/(2*pi)` | 1% |
| slab resistance | `L/(sigma*A)` | 1e-6 rel |
| RC step response | see **verified** note below | 1e-3 |
| RLC ring-down | damped sinusoid | 1% |
| skin depth | `sqrt(2/(omega*mu*sigma))` | 3% |
| microstrip Z0 | Hammerstad–Jensen | 5% |
| FDTD plane wave | phase velocity `c0`, numerical dispersion | 1% |
| pn junction built-in potential | `Vt*ln(Na*Nd/ni^2)` | **1e-9 rel** |
| pn depletion width | `sqrt(2 eps Vbi (Na+Nd)/(q Na Nd))` | **25%, see note** |
| pn depletion scaling | `W(Vr) ∝ sqrt(Vbi+Vr)` for `Vr >> Vt` | 2% |
| MOSFET subthreshold | 60 mV/decade at 300 K | 3% |
| diode | Shockley ideality n≈1 | 5% |
| EQS vs FDTD | agree when `L/lambda < 0.01` | 2% |

The last row is the most important test in the project: it is the only
direct evidence that the quasi-static approximation is doing what we claim.

### Verified reference numbers (measured on the frozen core, trust these)

These were computed with the frozen `grid.py` / `operators.py` and are known
good. If your module disagrees with one, your module is wrong.

**Electrostatics / EQS.** Parallel plate (d=2 um, 5x4 um, eps_r=3.9) gives
C to 1.5e-14 relative error; a uniform slab gives R to 3.4e-14. A series
R–C stack (left half sigma=1e3, eps_r=4; right half insulating, eps_r=2)
steps from the **capacitive-divider** value `eps1/(eps1+eps2) = 2/3` at
`t=0+` and relaxes to 1.0 with `tau = R1*(C1+C2)`. Backward Euler error is
first order: 1.22e-3 → 6.11e-4 → 3.06e-4 as dt halves. Note the correct
initial condition is the **electrostatic** solve (`L_eps psi = Q`), not
zero; starting from zero is a physically wrong initial condition and is a
common bug.

**pn junction (Si, Na=Nd=1e17 cm^-3, 300 K).** `Vbi = 0.833370 V` and the
nonlinear Poisson Newton reproduces it to **6.7e-16 relative** in 10
iterations on a 0.5 nm-graded mesh. Net charge integrates to 1.2e-16 C.

**Do not chase a tight tolerance on depletion width.** The analytic `W`
above is the *depletion approximation*, which assumes abrupt space-charge
edges. The true solution has exponential tails of order the Debye length
`LD = sqrt(eps*Vt/(q*N))` (12.9 nm in the case above), so a
10%-of-peak-charge measurement gives 179 nm against an analytic 147 nm — a
22% discrepancy that is **correct physics, not error**, and does not shrink
under mesh refinement. Test `Vbi` (exact) and the reverse-bias *scaling*
(asymptotically exact); treat absolute `W` as a sanity check only.

**Newton damping — measured, do not improvise.** Clip the exponent
argument to about ±400 before `np.exp`, then damp in two stages:

```python
lam = min(1.0, 5*Vt/max(abs(dpsi)))        # 1. keep exp() in range
while ||F(psi + lam*dpsi)|| >= (1 - 1e-4*lam) * ||F(psi)||:
    lam *= 0.5                              # 2. Armijo line search
```

Measure `||F||` over the **free** nodes only (Dirichlet rows are
identically zero and dilute the norm).

A bare step clamp with no line search is **not sufficient** and is a trap
we actually fell into: with `lam = min(1, 3*Vt/max|dpsi|)` the equilibrium
pn solve converges in 10 iterations at `ni = 1.0e16` but enters a stable
**limit cycle** (residual alternating 2.93e-2 / 4.82e-2 forever) at
`ni = 9.65e15`. Nothing in a pure clamp forces the residual to decrease,
so convergence is luck. With the line search added, the same solver takes
**9-11 iterations** across `ni` in {9.65e15, 1e16, 1.45e16} crossed with
`(Na, Nd)` in {(1e17,1e17), (1e15,1e18), (1e19,1e16), (5e16,3e17)} cm^-3,
reaching `Vbi` to 1e-12 relative or better in every case. Reproduce that
sweep in your tests.

## Style

- Comments explain **why**, never what.
- Prefer vectorised NumPy; loop only over time steps or Newton iterations.
- Sparse matrices: build with `coo_matrix` then convert once.
- Never `assert` for user-facing validation.
- No emoji or decorative unicode in code or docstrings.
