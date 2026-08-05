# The discrete formulation

Why fieldspice can be a field solver and a circuit solver at the same time.

---

## 1. The idea in one paragraph

Discretise Maxwell's equations by integrating each field over the geometric
element it naturally lives on — potentials on nodes, electric field along edges,
magnetic flux through faces, charge in cells. Written in those *integrated*
variables, the gradient, curl and divergence become exact signed-incidence
matrices with entries in {−1, 0, +1} containing no geometry at all. Every metric
quantity (length, area, material property) then collects into diagonal **mass
matrices**, and on a rectilinear grid those diagonal entries are ordinary
circuit elements: conductances, capacitances and reluctances. A field region is
therefore literally a large RLC network, which is why coupling it to a SPICE
netlist is exact assembly rather than an interpolation scheme.

This is Weiland's **Finite Integration Technique**. It coincides with
Whitney-form (edge-element) finite elements and with Tonti's cell method, and on
a rectilinear grid all three reduce to Yee's 1966 scheme.

---

## 2. Placement

With `(Nx, Ny, Nz)` cells:

| Element | Count | Holds | Shape |
|---|---|---|---|
| node | `(Nx+1)(Ny+1)(Nz+1)` | φ [V], n, p, doping | `(Nx+1, Ny+1, Nz+1)` |
| edge | 3 families | e = ∮E·dl [V], a = ∮A·dl [Wb], i [A] | x: `(Nx, Ny+1, Nz+1)` etc. |
| face | 3 families | b = ∫B·dA [Wb], h = ∮H·dl [A] | x: `(Nx+1, Ny, Nz)` etc. |
| cell | `Nx·Ny·Nz` | material id, ε, µ, σ | `(Nx, Ny, Nz)` |

An x-edge `(i,j,k)` joins node `(i,j,k)` to `(i+1,j,k)` and has length `hx[i]`.
An x-face `(i,j,k)` sits in node-plane `i`, spans cell `j` in y and `k` in z, and
has area `hy[j]·hz[k]`.

**Everything is integrated over its element.** `e` is a voltage in volts, not a
field in V/m. `b` is a flux in webers, not a flux density in tesla. This is the
whole trick, and it is the one thing to keep straight when reading the code.

---

## 3. The topological operators

### Gradient, nodes → edges

```
(G φ)_e = φ_head − φ_tail
```

Exact: the potential difference along an edge is exactly the difference of the
two node values. No approximation, no grid spacing.

### Curl, edges → faces

For an x-face, traversing its boundary right-handed about +x:

```
(C e)_x[i,j,k] = e_y[i,j,k] − e_y[i,j,k+1] + e_z[i,j+1,k] − e_z[i,j,k]
```

This is Stokes' theorem stated exactly: the sum of the four edge circulations
*is* the closed line integral around the face. It is not a finite-difference
approximation to a curl; it is the curl, evaluated exactly, on integrated
quantities.

### Divergence, faces → cells

```
(D b)[i,j,k] = (b_x[i+1] − b_x[i]) + (b_y[j+1] − b_y[j]) + (b_z[k+1] − b_z[k])
```

Gauss' theorem, exactly: net outward flux as a telescoping sum.

### The two identities

```
C G = 0        curl grad = 0
D C = 0        div curl  = 0
```

Both hold **exactly**, on any grid, however graded or distorted — the entries
are ±1 integers, so the cancellation is exact in floating point rather than
merely small. `tests/test_core.py` asserts `nnz == 0`, not `< tol`.

Two consequences worth stating, because they are the practical payoff:

1. **Charge is conserved to machine precision.** The divergence of a discrete
   current is an exact telescoping sum, so nothing leaks between cells no matter
   how badly the mesh is graded.
2. **`div B = 0` holds for all time in the magnetic solvers**, automatically,
   because `b` is only ever updated through a curl and `D C = 0`. No divergence
   cleaning, no projection step, no drift.

---

## 4. The mass matrices

All geometry and material data lives here, and only here.

### Edge mass — conductance and capacitance

An x-edge is threaded by up to four cell quadrants whose cross-sections tile its
dual area `A_dual = hyd[j]·hzd[k]`. For transport *along* the edge those
quadrants are four elements **in parallel**, so the correct combination is the
dual-area-weighted arithmetic mean:

```
M_sigma[e] = σ̄_e · A_dual_e / L_e        [S]   conductance
M_eps[e]   = ε̄_e · A_dual_e / L_e        [F]   capacitance
```

### Face mass — reluctance

The dual edge threading a face passes through two cells **in series** along the
magnetic path, so reluctances add and the correct rule is the length-weighted
harmonic mean:

```
M_nu[f] = A_f / Σ_c (µ_c · ΔL_c)          [1/H]
```

Unit check: `M_nu · b` = (1/H)(Wb) = A. ✔

---

## 5. The systems

### Electroquasistatic (A1a)

Charge conservation `∇·J + ∂ρ/∂t = 0` with `J = σE + ∂D/∂t` and `E = −∇φ`:

```
Gᵀ (M_sigma + M_eps d/dt) G φ = i_inject
```

`Gᵀ M_sigma G` is the **nodal conductance matrix** and `L φ = i` is Kirchhoff's
current law for a resistor mesh. `Gᵀ M_eps G` is the **capacitance matrix** and
`L φ = q` is Gauss' law. Both are symmetric positive semi-definite with a single
null vector (the constant), removed by any Dirichlet electrode.

Backward Euler:

```
(L_eps/Δt + L_sigma) φⁿ⁺¹ = (L_eps/Δt) φⁿ + i_inject
```

Unconditionally stable. One sparse SPD solve per step; with constant Δt the
factorisation is computed once and reused.

**Initial condition.** For a step-driven problem the consistent start is the
*electrostatic* solution `L_eps φ = q`, not zero — at `t=0⁺` no charge has moved,
so a lossy dielectric stack sits at its capacitive-divider value. Starting from
zero is a physically wrong initial condition; it makes a correct solver look
broken, and it is the single most common way to misdiagnose this scheme.

### Magnetoquasistatic (A1b)

```
Cᵀ M_nu C a + M_sigma (da/dt + G φ) = i_src
```

`Cᵀ M_nu C` is singular: its null space is the range of `G` (all gradient
fields), which is unregularised wherever σ = 0. Gauging is mandatory, and
`solvers/mqs.py` documents which strategy it uses and demonstrates it with a
skin-depth measurement rather than asserting it.

### Darwin (A1c)

Both of the above, dropping only the transverse (solenoidal) part of the
displacement current — the term responsible for radiation. Full R + L + C, no
wave. The right model for essentially all chip, package and board interconnect
below a few GHz.

In the **frequency domain** the Darwin coupling can be solved monolithically
with no splitting error at all; in the time domain the default is a staggered
exchange with an O(Δt) splitting error (A8), removable by Picard iteration.

### Full-wave (reference)

```
b  −= Δt · C e
e  += Δt · M_eps⁻¹ (Cᵀ M_nu b − M_sigma e − i_src)
```

The same operators, explicit and Courant-limited. It exists so the quasi-static
approximation can be *checked* rather than trusted.

### Drift-diffusion (A5, A7)

```
Gᵀ M_eps G ψ = q(p − n + N_D − N_A) · V_node
∂n/∂t = +(1/q) div J_n − R
∂p/∂t = −(1/q) div J_p − R
```

Edge currents use **Scharfetter–Gummel** exponential fitting with the Bernoulli
function `B(x) = x/(eˣ−1)`. This is a correctness requirement, not an
optimisation: central differencing produces negative carrier concentrations as
soon as the potential drop across a cell exceeds ~2 kT/q ≈ 52 mV, and a 1 V
junction drop falls across ~50 nm.

---

## 6. Why this makes field–circuit coupling exact

Because `Gᵀ M_sigma G φ = i` *is* nodal analysis, a field region and a netlist
are the same kind of object. Partition the field unknowns into terminal (t) and
interior (i) sets:

```
[K_tt  K_ti] [φ_t]   [f_t]
[K_it  K_ii] [φ_i] = [f_i]
```

Eliminate the interior exactly:

```
Y_eff = K_tt − K_ti K_ii⁻¹ K_it
f_eff = f_t   − K_ti K_ii⁻¹ f_i
```

`Y_eff` is a dense `nt × nt` admittance that stamps into the MNA matrix exactly
like any other multi-terminal element. For a linear field region this is
**exact** — no relaxation, no convergence loop, no handshake. The cost is one
sparse factorisation of `K_ii` plus `nt` back-solves per distinct Δt, both
cached.

---

## 7. What this formulation does not give you

- **Conformal geometry.** Rectilinear tensor-product grids only; curved
  boundaries are staircased (A2). Sub-cell fill fractions recover much of it,
  and planar processes are exactly axis-aligned so lose nothing.
- **Anisotropic materials.** The mass matrices are diagonal, which assumes
  isotropy. A diagonal-tensor extension is straightforward; off-diagonal
  coupling is not, because it destroys the diagonality that makes the scheme
  cheap.
- **Unconditional accuracy on graded meshes.** The box method is second order on
  smooth meshes and degrades toward first order where cells change size
  abruptly. Keep `grid.max_growth_ratio()` below ~1.4.

---

## References

- T. Weiland, *A discretization method for the solution of Maxwell's equations
  for six-component fields*, AEÜ 31 (1977) 116.
- A. Bossavit, *Whitney forms: a class of finite elements for three-dimensional
  computations in electromagnetism*, IEE Proc. A 135 (1988) 493.
- E. Tonti, *A direct discrete formulation of field laws: the cell method*,
  CMES 2 (2001) 237.
- K. Yee, IEEE Trans. Antennas Propag. 14 (1966) 302.
- D. Scharfetter and H. Gummel, IEEE Trans. Electron Devices 16 (1969) 64.
- C. W. Ho, A. E. Ruehli, P. A. Brennan, *The modified nodal approach to network
  analysis*, IEEE Trans. Circuits Syst. 22 (1975) 504.
