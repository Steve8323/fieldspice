"""Closed-form reference solutions --- the oracle the whole project is tested against.

Every quantitative claim fieldspice makes is checked against a formula in this
module. Keeping them in one place, in the shipped package rather than buried in
the test suite, has three benefits: the acceptance criteria are auditable, users
can sanity-check their own setups against a known case, and there is exactly one
definition of "correct" rather than one per test file.

Each function documents its own validity range. Several of these formulas are
themselves approximations (Hammerstad-Jensen microstrip is a curve fit; the
depletion approximation assumes abrupt space-charge edges), and where that is
true the docstring says so and gives the expected size of the discrepancy. A
test that demands agreement tighter than the reference formula's own accuracy is
a broken test, not a passing grade.
"""

from __future__ import annotations

import numpy as np

from .units import c0, eps0, mu0, q, kB, thermal_voltage

__all__ = [
    "parallel_plate_capacitance", "coaxial_capacitance", "coaxial_inductance",
    "sphere_capacitance", "two_wire_capacitance", "two_wire_inductance",
    "slab_resistance", "microstrip_z0", "stripline_z0",
    "skin_depth", "internal_inductance_round_wire", "ac_resistance_round_wire",
    "rc_step", "rl_step", "rlc_step", "rlc_ringdown", "elmore_delay_rc_ladder",
    "lossless_line_z0", "lossy_line_gamma", "telegrapher_step",
    "debye_length", "built_in_potential", "depletion_width",
    "depletion_capacitance", "shockley_diode", "subthreshold_slope",
    "intrinsic_carrier_density_si", "bandgap_varshni",
    "mosfet_square_law", "fdtd_numerical_dispersion", "courant_limit",
    "plane_wave_impedance", "electrical_length",
]


# ==========================================================================
# Electrostatics
# ==========================================================================
def parallel_plate_capacitance(area: float, gap: float,
                               eps_r: float = 1.0) -> float:
    """Ideal parallel-plate capacitance [F]: ``eps_r*eps0*A/d``.

    Neglects fringing entirely, so it is a *lower* bound on the true
    capacitance of a finite plate. Exact only in the limit of infinite plates
    (or with perfectly Neumann side walls, which is precisely how a fieldspice
    box with the plates spanning the full cross-section behaves --- which is
    why this is a machine-precision test rather than a 1% one).
    """
    return eps_r * eps0 * area / gap


def coaxial_capacitance(r_inner: float, r_outer: float,
                        eps_r: float = 1.0) -> float:
    """Coaxial capacitance per unit length [F/m]: ``2*pi*eps/ln(b/a)``."""
    if r_outer <= r_inner:
        raise ValueError("r_outer must exceed r_inner")
    return 2.0 * np.pi * eps_r * eps0 / np.log(r_outer / r_inner)


def coaxial_inductance(r_inner: float, r_outer: float,
                       mu_r: float = 1.0) -> float:
    """Coaxial external inductance per unit length [H/m]: ``mu*ln(b/a)/(2*pi)``.

    External only --- excludes the internal inductance of the centre conductor,
    which is frequency dependent (see :func:`internal_inductance_round_wire`)
    and vanishes at high frequency as current crowds to the surface.
    """
    if r_outer <= r_inner:
        raise ValueError("r_outer must exceed r_inner")
    return mu_r * mu0 * np.log(r_outer / r_inner) / (2.0 * np.pi)


def sphere_capacitance(radius: float, eps_r: float = 1.0) -> float:
    """Isolated sphere capacitance [F]: ``4*pi*eps*a``.

    Useful as an open-boundary test: a fieldspice box with Neumann walls will
    *undershoot* this until the padding is large (assumption A12), so the
    convergence of the computed value toward this number as the box grows is a
    direct measurement of the open-boundary error.
    """
    return 4.0 * np.pi * eps_r * eps0 * radius


def two_wire_capacitance(radius: float, separation: float,
                         eps_r: float = 1.0) -> float:
    """Two parallel round wires, capacitance per unit length [F/m].

    ``pi*eps / arccosh(D/(2a))``. Exact for any ratio (the arccosh form already
    accounts for proximity-driven charge redistribution); the often-quoted
    ``ln(D/a)`` version is the thin-wire limit only.
    """
    return np.pi * eps_r * eps0 / np.arccosh(separation / (2.0 * radius))


def two_wire_inductance(radius: float, separation: float,
                        mu_r: float = 1.0) -> float:
    """Two parallel round wires, external inductance per unit length [H/m]."""
    return mu_r * mu0 * np.arccosh(separation / (2.0 * radius)) / np.pi


def slab_resistance(length: float, area: float, sigma: float) -> float:
    """Uniform slab resistance [ohm]: ``L/(sigma*A)``."""
    return length / (sigma * area)


# ==========================================================================
# Transmission lines
# ==========================================================================
def microstrip_z0(w: float, h: float, eps_r: float,
                  t: float = 0.0) -> tuple[float, float]:
    """Microstrip characteristic impedance and effective permittivity.

    Hammerstad-Jensen synthesis formulas. Returns ``(Z0 [ohm], eps_eff)``.

    **This is a curve fit, not an exact solution.** Its own quoted accuracy is
    about 1% for ``0.05 <= w/h <= 20`` and ``eps_r <= 16``, so a field solver
    agreeing to 5% is agreeing as well as the reference deserves. Zero-thickness
    conductor unless ``t`` is given, in which case a first-order width
    correction is applied.
    """
    if h <= 0 or w <= 0:
        raise ValueError("w and h must be positive")
    u = w / h
    if t > 0:
        # Effective width correction for finite metal thickness.
        dw = (t / np.pi) * np.log(1.0 + 4.0 * np.e / (t / h)
                                  / ((1.0 / np.tanh(np.sqrt(6.517 * u))) ** 2))
        u = u + dw / h

    # Effective relative permittivity (Hammerstad-Jensen).
    a = (1.0 + (1.0 / 49.0)
         * np.log((u ** 4 + (u / 52.0) ** 2) / (u ** 4 + 0.432))
         + (1.0 / 18.7) * np.log(1.0 + (u / 18.1) ** 3))
    b = 0.564 * ((eps_r - 0.9) / (eps_r + 3.0)) ** 0.053
    eps_eff = (eps_r + 1.0) / 2.0 + (eps_r - 1.0) / 2.0 * (1.0 + 10.0 / u) ** (-a * b)

    # Characteristic impedance of the equivalent air-filled line.
    fu = 6.0 + (2.0 * np.pi - 6.0) * np.exp(-(30.666 / u) ** 0.7528)
    z01 = (376.730313668 / (2.0 * np.pi)) * np.log(
        fu / u + np.sqrt(1.0 + (2.0 / u) ** 2))
    return float(z01 / np.sqrt(eps_eff)), float(eps_eff)


def stripline_z0(w: float, b: float, eps_r: float) -> float:
    """Symmetric stripline characteristic impedance [ohm].

    Cohn's formula. Unlike microstrip this line is homogeneously filled, so the
    mode is exactly TEM and ``eps_eff == eps_r`` with no approximation.
    Accuracy ~1% for ``w/b < 0.35``.
    """
    we = w / b
    return float(30.0 * np.pi / np.sqrt(eps_r)
                 / (we + 0.441) if we > 0.35 else
                 60.0 / np.sqrt(eps_r) * np.log(4.0 * b / (np.pi * w)))


def lossless_line_z0(L: float, C: float) -> float:
    """``sqrt(L/C)`` [ohm] from per-unit-length L and C."""
    return float(np.sqrt(L / C))


def lossy_line_gamma(R: float, L: float, G: float, C: float,
                     freq: float | np.ndarray) -> np.ndarray:
    """Complex propagation constant ``gamma = sqrt((R+jwL)(G+jwC))`` [1/m]."""
    w = 2.0 * np.pi * np.asarray(freq, dtype=float)
    return np.sqrt((R + 1j * w * L) * (G + 1j * w * C))


def telegrapher_step(z: float, t: np.ndarray, R: float, L: float,
                     G: float, C: float, v0: float = 1.0,
                     n_freq: int = 4096) -> np.ndarray:
    """Step response of a lossy line at distance ``z``, by numerical inversion.

    Provided so the Darwin solver can be checked against a genuine
    distributed-line solution in the regime where both are valid (low enough
    frequency that the line is electrically short, but with real R and L).
    Uses a real-FFT inversion; accuracy is set by ``n_freq``.
    """
    t = np.asarray(t, dtype=float)
    T = float(t[-1] - t[0]) * 2.0
    f = np.fft.rfftfreq(n_freq, d=T / n_freq)
    w = 2.0 * np.pi * f
    gam = np.sqrt((R + 1j * w * L) * (G + 1j * w * C))
    step_f = np.zeros_like(f, dtype=complex)
    step_f[0] = v0 * T / 2.0
    nz = f > 0
    step_f[nz] = v0 / (1j * w[nz])
    resp = np.fft.irfft(step_f * np.exp(-gam * z), n=n_freq)
    tt = np.arange(n_freq) * T / n_freq
    return np.interp(t, tt, resp)


# ==========================================================================
# Magnetoquasistatics
# ==========================================================================
def skin_depth(sigma: float, freq: float, mu_r: float = 1.0) -> float:
    """Classical skin depth [m]: ``sqrt(2/(omega*mu*sigma))``.

    Copper at 1 GHz: 2.09 um. At 1 MHz: 66.1 um. Valid when the skin depth is
    much larger than the electron mean free path (i.e. not in the anomalous
    regime) and much smaller than the conductor dimension.
    """
    if freq <= 0:
        return float("inf")
    return float(np.sqrt(2.0 / (2.0 * np.pi * freq * mu_r * mu0 * sigma)))


def internal_inductance_round_wire(mu_r: float = 1.0) -> float:
    """DC internal inductance of a round wire [H/m]: ``mu/(8*pi)``.

    Independent of radius --- a pleasing and frequently doubted result. Falls
    toward zero above the skin-effect corner.
    """
    return mu_r * mu0 / (8.0 * np.pi)


def ac_resistance_round_wire(radius: float, sigma: float, freq: float,
                             mu_r: float = 1.0) -> float:
    """AC resistance per unit length of a round wire [ohm/m].

    Uses the exact Bessel-function (Kelvin) solution, falling back to the DC
    value at zero frequency. This is the reference for validating that the MQS
    solver reproduces skin and proximity effects rather than merely being
    stable.
    """
    R_dc = 1.0 / (sigma * np.pi * radius ** 2)
    if freq <= 0:
        return float(R_dc)
    from scipy.special import ber, bei, berp, beip
    delta = skin_depth(sigma, freq, mu_r)
    Qq = np.sqrt(2.0) * radius / delta
    num = ber(Qq) * beip(Qq) - bei(Qq) * berp(Qq)
    den = berp(Qq) ** 2 + beip(Qq) ** 2
    return float(R_dc * (Qq / 2.0) * num / den)


# ==========================================================================
# Lumped transients
# ==========================================================================
def rc_step(t: np.ndarray, R: float, C: float, v_final: float = 1.0,
            v_initial: float = 0.0) -> np.ndarray:
    """RC step response ``Vf - (Vf - V0)*exp(-t/RC)``.

    The ``v_initial`` argument matters more than it looks. A field region that
    is both conductive and dielectric does **not** start at zero when stepped:
    at ``t=0+`` charge has not yet moved, so the potential distribution is the
    *electrostatic* one (a capacitive divider). Starting a comparison from zero
    is the most common way to conclude, wrongly, that a working EQS solver is
    broken.
    """
    t = np.asarray(t, dtype=float)
    return v_final - (v_final - v_initial) * np.exp(-t / (R * C))


def rl_step(t: np.ndarray, R: float, L: float, v: float = 1.0) -> np.ndarray:
    """Current in a series RL driven by a voltage step [A]."""
    t = np.asarray(t, dtype=float)
    return (v / R) * (1.0 - np.exp(-t * R / L))


def rlc_step(t: np.ndarray, R: float, L: float, C: float,
             v: float = 1.0) -> np.ndarray:
    """Capacitor voltage of a series RLC driven by a voltage step [V].

    Covers all three damping regimes exactly. The under-damped case is the
    sharpest available test of a Darwin solver, because getting it right
    requires R, L and C to all be correct *simultaneously* --- an error in any
    one shifts either the ring frequency or the envelope.
    """
    t = np.asarray(t, dtype=float)
    w0 = 1.0 / np.sqrt(L * C)
    zeta = (R / 2.0) * np.sqrt(C / L)
    if zeta < 1.0 - 1e-12:
        wd = w0 * np.sqrt(1.0 - zeta ** 2)
        env = np.exp(-zeta * w0 * t)
        return v * (1.0 - env * (np.cos(wd * t)
                                 + (zeta * w0 / wd) * np.sin(wd * t)))
    if zeta > 1.0 + 1e-12:
        s = w0 * np.sqrt(zeta ** 2 - 1.0)
        a, b = -zeta * w0 + s, -zeta * w0 - s
        return v * (1.0 + (b * np.exp(a * t) - a * np.exp(b * t)) / (a - b))
    return v * (1.0 - np.exp(-w0 * t) * (1.0 + w0 * t))


def rlc_ringdown(R: float, L: float, C: float) -> tuple[float, float, float]:
    """``(f_damped [Hz], zeta, Q)`` of a series RLC."""
    w0 = 1.0 / np.sqrt(L * C)
    zeta = (R / 2.0) * np.sqrt(C / L)
    wd = w0 * np.sqrt(max(0.0, 1.0 - zeta ** 2))
    Q = float("inf") if zeta == 0 else 1.0 / (2.0 * zeta)
    return float(wd / (2.0 * np.pi)), float(zeta), float(Q)


def elmore_delay_rc_ladder(r: np.ndarray, c: np.ndarray) -> float:
    """Elmore delay of an RC ladder [s]: ``sum_i c_i * sum_{j<=i} r_j``.

    The standard first-moment interconnect delay estimate. Always an
    *upper* bound on the 50% delay of a monotonic step response, and typically
    20-40% high --- which is the expected discrepancy against an EQS transient,
    not an error.
    """
    r = np.asarray(r, dtype=float)
    c = np.asarray(c, dtype=float)
    if r.shape != c.shape:
        raise ValueError("r and c must have the same shape")
    return float(np.sum(c * np.cumsum(r)))


# ==========================================================================
# Semiconductor
# ==========================================================================
def bandgap_varshni(T: float = 300.0, Eg0: float = 1.1696,
                    alpha: float = 4.73e-4, beta: float = 636.0) -> float:
    """Varshni temperature-dependent bandgap [eV]. Silicon defaults."""
    return float(Eg0 - alpha * T ** 2 / (T + beta))


NI_SI_300 = 9.65e15
"""Silicon intrinsic carrier density at 300 K [m^-3] = 9.65e9 cm^-3.

Measured value from Sproul & Green, J. Appl. Phys. 70, 846 (1991), which is the
modern accepted number and what current TCAD tools use.
"""


def intrinsic_carrier_density_si(T: float = 300.0) -> float:
    """Silicon intrinsic carrier density [m^-3], anchored to measurement.

    Returns ``NI_SI_300 = 9.65e15 m^-3`` at 300 K, with the temperature
    dependence taken from ``sqrt(Nc*Nv)*exp(-Eg(T)/2kT)`` (``Nc,Nv ~ T^3/2``,
    Varshni gap) *normalised* to hit the measured value at 300 K.

    **Why the normalisation is not a fudge.** Evaluating the physical formula
    directly from the usual textbook constants (``Nc = 2.8e19 cm^-3``,
    ``Nv = 1.04e19 cm^-3``, ``Eg = 1.12 eV``) gives ``ni(300) = 6.7e9 cm^-3``,
    which disagrees with every measured value in the literature (published
    numbers run 9.65e9 to 1.45e10 cm^-3). The inconsistency is real and
    well known: the simple expression omits the temperature dependence of the
    density-of-states effective masses, band non-parabolicity, and
    Fermi-Dirac corrections. Textbooks quote the measured ``ni`` alongside
    constants that do not reproduce it, and a simulator that silently uses the
    formula inherits a ~35% error in ``ni``, which becomes a ~2 kT/q error in
    every built-in potential.

    We therefore anchor the absolute value to experiment and use the formula
    only for its temperature *shape*, which is the standard TCAD practice.
    Pass an explicit ``ni`` to the junction functions to override.
    """
    def _shape(TT: float) -> float:
        Nc = 2.8e25 * (TT / 300.0) ** 1.5
        Nv = 1.04e25 * (TT / 300.0) ** 1.5
        return float(np.sqrt(Nc * Nv) * np.exp(-bandgap_varshni(TT) * q
                                               / (2.0 * kB * TT)))
    return float(NI_SI_300 * _shape(T) / _shape(300.0))


def debye_length(N: float, T: float = 300.0, eps_r: float = 11.7) -> float:
    """Extrinsic Debye length [m]: ``sqrt(eps*kT/(q^2*N))``.

    The mesh must resolve this or the drift-diffusion solve is meaningless.
    Silicon at 1e17 cm^-3, 300 K: 12.9 nm.
    """
    return float(np.sqrt(eps_r * eps0 * kB * T / (q * q * N)))


def built_in_potential(Na: float, Nd: float, ni: float | None = None,
                       T: float = 300.0) -> float:
    """pn junction built-in potential [V]: ``Vt*ln(Na*Nd/ni^2)``.

    Exact within Boltzmann statistics and complete ionisation, and reproduced
    by the nonlinear Poisson solver to machine precision --- so this is a
    1e-9-tolerance test, not a 1% one.
    """
    ni = intrinsic_carrier_density_si(T) if ni is None else ni
    return float(thermal_voltage(T) * np.log(Na * Nd / (ni * ni)))


def depletion_width(Na: float, Nd: float, V_reverse: float = 0.0,
                    eps_r: float = 11.7, ni: float | None = None,
                    T: float = 300.0) -> float:
    """Depletion width [m] under the depletion approximation.

    ``sqrt(2*eps*(Vbi+Vr)*(Na+Nd)/(q*Na*Nd))``.

    **Expect ~20% disagreement with a real solve at zero bias.** The depletion
    approximation assumes the space charge stops abruptly; the true solution
    decays exponentially over a few Debye lengths at each edge, making the
    measured width larger. The discrepancy does *not* vanish under mesh
    refinement because it is a property of the reference formula, not of the
    discretisation. It shrinks as reverse bias grows and the depletion region
    becomes large compared with the Debye length, which is why the *scaling*
    with ``sqrt(Vbi+Vr)`` is the meaningful test.
    """
    Vbi = built_in_potential(Na, Nd, ni, T)
    return float(np.sqrt(2.0 * eps_r * eps0 * (Vbi + V_reverse)
                         * (Na + Nd) / (q * Na * Nd)))


def depletion_capacitance(Na: float, Nd: float, area: float = 1.0,
                          V_reverse: float = 0.0, eps_r: float = 11.7,
                          ni: float | None = None, T: float = 300.0) -> float:
    """Junction depletion capacitance [F]: ``eps*A/W``."""
    W = depletion_width(Na, Nd, V_reverse, eps_r, ni, T)
    return float(eps_r * eps0 * area / W)


def shockley_diode(V: np.ndarray, I_s: float = 1e-15, n: float = 1.0,
                   T: float = 300.0) -> np.ndarray:
    """Ideal diode current [A]: ``Is*(exp(V/(n*Vt)) - 1)``."""
    Vt = thermal_voltage(T)
    return I_s * (np.exp(np.clip(np.asarray(V, dtype=float) / (n * Vt),
                                 -400.0, 400.0)) - 1.0)


def subthreshold_slope(T: float = 300.0, n: float = 1.0) -> float:
    """Subthreshold swing [V/decade]: ``n*ln(10)*kT/q``.

    59.5 mV/decade at 300 K for an ideal body factor ``n = 1``. This is the
    single most diagnostic number a drift-diffusion MOSFET simulation can
    produce: it depends only on Boltzmann statistics and the electrostatics of
    the channel, so getting it wrong means something fundamental is wrong. Real
    devices have ``n = 1 + Cd/Cox`` in the range 1.1-1.5.
    """
    return float(n * np.log(10.0) * thermal_voltage(T))


def mosfet_square_law(Vgs: np.ndarray, Vds: np.ndarray, Vth: float,
                      k: float, lam: float = 0.0) -> np.ndarray:
    """Level-1 MOSFET drain current [A], triode and saturation.

    ``k = mu*Cox*W/L``. Channel-length modulation via ``lam``. Cutoff below
    threshold returns exactly zero, which is why this model alone cannot
    reproduce the subthreshold slope above --- a compact-model limitation, not
    a device one.
    """
    Vgs = np.asarray(Vgs, dtype=float)
    Vds = np.asarray(Vds, dtype=float)
    Vov = Vgs - Vth
    tri = k * (Vov * Vds - 0.5 * Vds ** 2)
    sat = 0.5 * k * Vov ** 2 * (1.0 + lam * Vds)
    return np.where(Vov <= 0.0, 0.0, np.where(Vds < Vov, tri, sat))


# ==========================================================================
# Full-wave / regime checks
# ==========================================================================
def courant_limit(dx: float, dy: float | None = None, dz: float | None = None,
                  eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """Maximum stable explicit-FDTD time step [s]."""
    inv2 = 1.0 / dx ** 2
    if dy is not None:
        inv2 += 1.0 / dy ** 2
    if dz is not None:
        inv2 += 1.0 / dz ** 2
    return float(np.sqrt(eps_r * mu_r) / (c0 * np.sqrt(inv2)))


def fdtd_numerical_dispersion(dx: float, dt: float, freq: float,
                              theta: float = 0.0) -> float:
    """Numerical phase velocity of 2D Yee FDTD [m/s], relative to ``c0``.

    Solves the discrete dispersion relation for the numerical wavenumber along
    a propagation angle ``theta``. Grid dispersion is the dominant error in any
    long full-wave run, and quantifying it is how the FDTD solver earns the
    right to be called a reference: an implementation whose measured phase
    velocity does not match this formula has a bug.
    """
    w = 2.0 * np.pi * freq
    s = np.sin(w * dt / 2.0) / (c0 * dt)
    ct, st = np.cos(theta), np.sin(theta)

    def resid(k: float) -> float:
        return ((np.sin(k * ct * dx / 2.0) / dx) ** 2
                + (np.sin(k * st * dx / 2.0) / dx) ** 2 - s ** 2)

    k_lo, k_hi = 1e-6, np.pi / dx * 0.999999
    if resid(k_lo) * resid(k_hi) > 0:
        return float("nan")
    for _ in range(200):
        k_mid = 0.5 * (k_lo + k_hi)
        if resid(k_lo) * resid(k_mid) <= 0:
            k_hi = k_mid
        else:
            k_lo = k_mid
    return float(w / (0.5 * (k_lo + k_hi)))


def plane_wave_impedance(eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """Wave impedance [ohm] of a plane wave in a lossless medium."""
    return float(np.sqrt(mu_r * mu0 / (eps_r * eps0)))


def electrical_length(size: float, freq: float | None = None,
                      t_rise: float | None = None,
                      eps_r: float = 1.0, mu_r: float = 1.0) -> float:
    """``L/lambda`` --- the number that decides which solver you should use.

    Give either ``freq`` or ``t_rise``; a rise time is converted with the usual
    knee-frequency rule ``f = 0.35/t_rise``. Compare the result against the
    bands in ``docs/ASSUMPTIONS.md`` A1: below 0.01 quasi-static is excellent,
    above 0.3 you need the full-wave solver.
    """
    if freq is None and t_rise is None:
        raise ValueError("give freq or t_rise")
    if freq is None:
        freq = 0.35 / float(t_rise)  # type: ignore[arg-type]
    v = c0 / np.sqrt(eps_r * mu_r)
    return float(size * freq / v)
