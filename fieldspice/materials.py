"""Material definitions and per-cell material assembly.

This module is the single place where physical constants of *matter* enter
fieldspice.  Everything downstream --- the mass matrices in
:mod:`fieldspice.operators`, every solver, every extracted capacitance --- reads
its permittivity, permeability and conductivity from a :class:`MaterialMap`
built here.

Two conventions that are easy to get wrong, stated once and enforced everywhere
below:

* :attr:`Material.eps_r` and :attr:`Material.mu_r` are **relative**
  (dimensionless).  :meth:`MaterialMap.eps` and :meth:`MaterialMap.mu` return
  **absolute** SI values, ``eps0 * eps_r`` [F/m] and ``mu0 * mu_r`` [H/m].
  The solvers want absolute values, because ``edge_mass(grid, eps_edge)`` must
  produce farads.
* :attr:`SemiconductorParams.Eg` and :attr:`SemiconductorParams.chi` are in
  **electronvolts**, the universal TCAD convention.  An electronvolt is not an
  SI unit, so this is the one documented departure from strict SI in the
  package; ``Eg`` in eV is numerically identical to the gap expressed as a
  *voltage* ``Eg/q`` [V], and volts are SI.  Use :meth:`SemiconductorParams.Eg_J`
  when joules are actually wanted.

Assumptions invoked (see ``docs/ASSUMPTIONS.md``)
-------------------------------------------------
* **A3** --- materials are linear, isotropic, non-dispersive and
  time-invariant.  ``eps``, ``mu`` and ``sigma`` are real scalars per cell.
  Frequency-dependent loss is representable only through the ``ac`` solver's
  complex permittivity; :attr:`Material.loss_tangent` is carried here purely as
  data for that solver and is ignored by the quasi-static path.
* **A2** --- sub-cell fill fractions from :func:`fieldspice.geometry.voxelize`
  are blended by an effective-medium rule in :meth:`MaterialMap.assign`, which
  recovers a large part of the staircase error at essentially zero cost.
* **A5**, **A11** --- semiconductor parameters here feed the drift-diffusion
  solver, which assumes Boltzmann statistics and complete dopant ionisation.
  Both break for the amorphous-oxide entries (``igzo``, ``a_si``), whose
  transport is trap-limited; the parameter blocks for those materials are
  order-of-magnitude placeholders and are flagged as such.

On the module-level ``LIBRARY``
-------------------------------
``docs/CONTRACTS.md`` forbids global mutable state but also mandates a module
level ``LIBRARY`` dict with a ``register`` function.  The tension is resolved by
making every value a frozen dataclass (so an entry cannot be mutated in place)
and by making :func:`register` refuse to silently replace an existing name.  The
dangerous failure mode --- one script quietly changing the silicon another
script is using --- is therefore blocked rather than merely discouraged.

Provenance of the numbers
-------------------------
Bulk metal conductivities are 20 C handbook values (CRC Handbook of Chemistry
and Physics, 97th ed.).  Silicon and GaAs parameters follow Sze & Ng, *Physics
of Semiconductor Devices*, 3rd ed. (2007), Appendix G, with Green (1990) for the
band-edge densities of state.  Values that are strongly process-dependent or
that disagree across the literature carry a ``ref`` string saying so; they are
flagged rather than given false precision.  Thin-film conductivities in
particular (Mo, Ti, Pt, ITO, IGZO) are routinely 2-5x worse than the bulk
figures quoted here.
"""

from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, replace

import numpy as np

from .grid import RectilinearGrid
from .units import T_ROOM, cm2_per_Vs, eps0, kB, mu0, per_cm3, q

__all__ = [
    "SemiconductorParams",
    "Material",
    "UnknownMaterial",
    "LIBRARY",
    "get",
    "register",
    "available",
    "mix_property",
    "MaterialMap",
]


# ==========================================================================
# Errors
# ==========================================================================
class UnknownMaterial(KeyError, ValueError):
    """Raised when a material name is not in :data:`LIBRARY`.

    It inherits from both ``KeyError`` and ``ValueError`` so that either of the
    two natural ``except`` clauses catches it: the lookup *is* a mapping miss
    (``KeyError``), and it is also exactly the "bad input, raise eagerly" case
    that ``docs/CONTRACTS.md`` asks to report as ``ValueError``.
    """


# ==========================================================================
# Semiconductor parameters
# ==========================================================================
@dataclass(frozen=True)
class SemiconductorParams:
    """Band-structure and transport parameters for one semiconductor.

    All energies are in electronvolts [eV] (see the module docstring for why),
    all densities in m^-3, all mobilities in m^2/(V s), all times in seconds,
    all velocities in m/s.

    Parameters
    ----------
    Eg : float
        Bandgap at ``T = 300 K`` [eV].
    ni_300 : float
        Intrinsic carrier concentration at 300 K [m^-3].  This is treated as a
        *measured anchor*, not as a derived quantity -- see :meth:`ni`.
    chi : float
        Electron affinity [eV], the energy from the vacuum level to the
        conduction-band edge.  Used for heterojunction band offsets and for
        metal/semiconductor barrier heights.
    mu_n, mu_p : float
        Low-field electron and hole mobilities [m^2/(V s)].  Use
        ``units.cm2_per_Vs`` to convert from the usual cm^2/(V s).  These are
        lightly-doped 300 K values; a doping- and field-dependent model belongs
        in the drift-diffusion solver, not here.
    Nc_300, Nv_300 : float
        Effective density of states at the conduction and valence band edges at
        300 K [m^-3].
    tau_n, tau_p : float
        Shockley-Read-Hall minority-carrier lifetimes [s].  Process-dependent
        over many orders of magnitude; the library values are nominal.
    vsat_n, vsat_p : float
        Saturation drift velocities [m/s].
    C_auger_n, C_auger_p : float
        Auger recombination coefficients [m^6/s].  Multiply a cm^6/s figure by
        1e-12 to convert.
    varshni_alpha : float
        Varshni coefficient ``alpha`` [eV/K].  Set to 0 to make the gap
        temperature-independent, which is the honest choice for an amorphous
        material where no Varshni fit exists.
    varshni_beta : float
        Varshni coefficient ``beta`` [K].
    ref : str
        Provenance / caveat string.  Read it before trusting a number.

    Notes
    -----
    Tagged **A5** (drift-diffusion transport) and **A11** (complete ionisation):
    these parameters only mean what they say inside a Boltzmann-statistics,
    fully-ionised drift-diffusion model.
    """

    Eg: float
    ni_300: float
    chi: float
    mu_n: float
    mu_p: float
    Nc_300: float
    Nv_300: float
    tau_n: float = 1e-6
    tau_p: float = 1e-6
    vsat_n: float = 1.0e5
    vsat_p: float = 1.0e5
    C_auger_n: float = 0.0
    C_auger_p: float = 0.0
    varshni_alpha: float = 0.0
    varshni_beta: float = 1.0
    ref: str = ""

    def __post_init__(self) -> None:
        for nm in ("Eg", "ni_300", "chi", "mu_n", "mu_p", "Nc_300", "Nv_300",
                   "tau_n", "tau_p", "vsat_n", "vsat_p"):
            v = getattr(self, nm)
            if not np.isfinite(v) or v <= 0.0:
                raise ValueError(
                    f"SemiconductorParams.{nm} must be finite and positive, got {v!r}")
        for nm in ("C_auger_n", "C_auger_p", "varshni_alpha"):
            v = getattr(self, nm)
            if not np.isfinite(v) or v < 0.0:
                raise ValueError(
                    f"SemiconductorParams.{nm} must be finite and >= 0, got {v!r}")
        if self.varshni_beta <= 0.0:
            raise ValueError("varshni_beta must be positive [K]")

    # -- energies ----------------------------------------------------------
    @property
    def Eg_J(self) -> float:
        """Bandgap at 300 K in joules [J]."""
        return self.Eg * q

    @property
    def Eg_V(self) -> float:
        """Bandgap at 300 K expressed as a voltage ``Eg/q`` [V].

        Numerically identical to :attr:`Eg` in eV; provided so that code doing
        band bookkeeping in volts can say so.
        """
        return self.Eg

    def Eg_at(self, T: float = T_ROOM) -> float:
        """Bandgap at temperature ``T`` [K], in eV, via the Varshni relation.

        ``Eg(T) = Eg(0) - alpha T^2 / (T + beta)``, with ``Eg(0)`` back-computed
        from the stored 300 K value so that ``Eg_at(300) == Eg`` exactly.  For
        silicon (``alpha = 4.73e-4 eV/K``, ``beta = 636 K``) this reproduces
        ``Eg(0) = 1.170 eV``.

        Parameters
        ----------
        T : float
            Lattice temperature [K], must be positive.

        Returns
        -------
        float
            Bandgap [eV].
        """
        T = float(T)
        if T <= 0.0:
            raise ValueError(f"temperature must be positive [K], got {T}")
        if self.varshni_alpha == 0.0:
            return self.Eg
        a, b = self.varshni_alpha, self.varshni_beta
        Eg0 = self.Eg + a * T_ROOM * T_ROOM / (T_ROOM + b)
        return Eg0 - a * T * T / (T + b)

    # -- band-edge densities of state --------------------------------------
    def Nc(self, T: float = T_ROOM) -> float:
        """Conduction-band effective density of states at ``T`` [m^-3].

        Scales as ``T^(3/2)``, which is the parabolic-band result
        ``Nc = 2 (2 pi m_e* kT / h^2)^(3/2)`` with a temperature-independent
        effective mass.  The residual temperature dependence of ``m*`` is a few
        percent over 200-500 K and is not modelled.
        """
        T = float(T)
        if T <= 0.0:
            raise ValueError(f"temperature must be positive [K], got {T}")
        return self.Nc_300 * (T / T_ROOM) ** 1.5

    def Nv(self, T: float = T_ROOM) -> float:
        """Valence-band effective density of states at ``T`` [m^-3]. ``T^(3/2)``."""
        T = float(T)
        if T <= 0.0:
            raise ValueError(f"temperature must be positive [K], got {T}")
        return self.Nv_300 * (T / T_ROOM) ** 1.5

    # -- intrinsic concentration -------------------------------------------
    def ni(self, T: float = T_ROOM) -> float:
        """Intrinsic carrier concentration at ``T`` [m^-3].

        Uses the mass-action form

        ``ni(T) = sqrt(Nc(T) Nv(T)) exp(-Eg(T) / (2 kT))``

        but *anchored* to the measured :attr:`ni_300`, i.e. evaluated as the
        ratio to its own 300 K value::

            ni(T) = ni_300 * (T/300)^(3/2)
                            * exp( Eg(300)/(2 Vt(300)) - Eg(T)/(2 Vt(T)) )

        which is algebraically the same expression with ``sqrt(Nc0 Nv0)``
        replaced by whatever prefactor reproduces ``ni_300``.

        **Why anchor.**  The textbook silicon triple (``Nc = 2.8e19 cm^-3``,
        ``Nv = 1.04e19 cm^-3``, ``Eg = 1.12 eV``) does *not* reproduce the
        measured ``ni(300) = 1.0e10 cm^-3``; it gives ``6.2e9 cm^-3``, low by
        about 38 percent.  This is a well-known inconsistency in the standard
        tables: the quoted ``Nv`` uses a single heavy-hole mass and ignores the
        light-hole and split-off bands and the temperature dependence of the
        density-of-states mass, while ``ni`` is the directly measured quantity
        (Sproul & Green, *J. Appl. Phys.* **70**, 846 (1991)).  Self-consistent
        parameter sets used by commercial TCAD carry ``Nv ~ 3.1e19 cm^-3``
        instead.  Since ``ni`` is what actually enters the drift-diffusion
        equations through ``n p = ni^2`` and ``n = ni exp((psi - phi_n)/Vt)``,
        the measured value wins and the density-of-states product is kept only
        for band bookkeeping.  :meth:`ni_dos` exposes the unanchored value and
        :meth:`dos_consistency` reports the discrepancy, so nothing is hidden.

        Parameters
        ----------
        T : float
            Lattice temperature [K].

        Returns
        -------
        float
            Intrinsic concentration [m^-3].

        Notes
        -----
        Tagged **A5**.  Boltzmann statistics assumed; degenerate doping above
        ~1e19 cm^-3 needs Fermi-Dirac and bandgap narrowing, neither of which
        is applied here.
        """
        T = float(T)
        if T <= 0.0:
            raise ValueError(f"temperature must be positive [K], got {T}")
        Vt = kB * T / q
        Vt0 = kB * T_ROOM / q
        expo = self.Eg / (2.0 * Vt0) - self.Eg_at(T) / (2.0 * Vt)
        return self.ni_300 * (T / T_ROOM) ** 1.5 * math.exp(expo)

    def ni_dos(self, T: float = T_ROOM) -> float:
        """Intrinsic concentration from the density-of-states product [m^-3].

        ``sqrt(Nc(T) Nv(T)) exp(-Eg(T)/(2 kT))``, with no anchoring.  Provided
        so the inconsistency described in :meth:`ni` is measurable rather than
        asserted.
        """
        Vt = kB * float(T) / q
        return math.sqrt(self.Nc(T) * self.Nv(T)) * math.exp(-self.Eg_at(T) / (2.0 * Vt))

    def dos_consistency(self, T: float = T_ROOM) -> float:
        """``ni_dos(T) / ni(T)``: 1.0 means the parameter set is self-consistent.

        Silicon with the standard textbook triple returns about 0.62.
        """
        return self.ni_dos(T) / self.ni(T)

    def Eg_eff_for_ni(self, T: float = T_ROOM) -> float:
        """Gap [eV] that would reconcile ``Nc``, ``Nv`` and ``ni_300``.

        ``2 kT ln( sqrt(Nc Nv) / ni )``.  For silicon this is 1.099 eV against a
        true gap of 1.124 eV; the 25 meV difference is the size of the tabulated
        inconsistency, not a physical bandgap narrowing.
        """
        Vt = kB * float(T) / q
        return 2.0 * Vt * math.log(math.sqrt(self.Nc(T) * self.Nv(T)) / self.ni(T))

    # -- derived transport quantities --------------------------------------
    def D_n(self, T: float = T_ROOM) -> float:
        """Electron diffusivity [m^2/s] from the Einstein relation ``mu kT/q``."""
        return self.mu_n * kB * float(T) / q

    def D_p(self, T: float = T_ROOM) -> float:
        """Hole diffusivity [m^2/s] from the Einstein relation ``mu kT/q``."""
        return self.mu_p * kB * float(T) / q

    def sigma_intrinsic(self, T: float = T_ROOM) -> float:
        """Conductivity of the undoped material [S/m], ``q ni (mu_n + mu_p)``."""
        return q * self.ni(T) * (self.mu_n + self.mu_p)


# ==========================================================================
# Material
# ==========================================================================
_KINDS = ("dielectric", "conductor", "semiconductor", "vacuum")


@dataclass(frozen=True)
class Material:
    """One isotropic, linear, non-dispersive material (assumption **A3**).

    Parameters
    ----------
    name : str
        Canonical lowercase identifier, also the :data:`LIBRARY` key.
    eps_r : float
        **Relative** permittivity [dimensionless].  Multiply by
        ``units.eps0`` for [F/m]; :attr:`eps` does it for you.
    mu_r : float
        **Relative** permeability [dimensionless].
    sigma : float
        Electrical conductivity [S/m].  For a semiconductor this is the bulk
        value used when the material is treated as a lossy dielectric by the
        quasi-static solvers; the drift-diffusion solver ignores it and
        computes transport from :attr:`semi` instead.
    kind : str
        One of ``"dielectric"``, ``"conductor"``, ``"semiconductor"``,
        ``"vacuum"``.  Advisory: it selects defaults and plotting, and lets
        ``validate`` decide whether to check the skin depth in this region.
    semi : SemiconductorParams or None
        Required if and only if ``kind == "semiconductor"``.
    color : str
        Hex colour used by :mod:`fieldspice.viz`.
    loss_tangent : float
        ``tan(delta) = eps'' / eps'`` at roughly 1 GHz [dimensionless], carried
        as data for the ``ac`` solver's complex permittivity.  **Ignored by the
        quasi-static and FDTD paths**, which are real-valued by construction
        (A3); a nonzero value here does not make a time-domain run lossy.
    ref : str
        Provenance and caveats for the numbers above.

    Examples
    --------
    >>> from fieldspice.materials import get
    >>> get("sio2").eps          # doctest: +ELLIPSIS
    3.45...e-11
    """

    name: str
    eps_r: float = 1.0
    mu_r: float = 1.0
    sigma: float = 0.0
    kind: str = "dielectric"
    semi: SemiconductorParams | None = None
    color: str = "#888888"
    loss_tangent: float = 0.0
    ref: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Material.name must be a non-empty string")
        if not np.isfinite(self.eps_r) or self.eps_r <= 0.0:
            raise ValueError(
                f"{self.name}: eps_r must be finite and > 0, got {self.eps_r!r}")
        if not np.isfinite(self.mu_r) or self.mu_r <= 0.0:
            raise ValueError(
                f"{self.name}: mu_r must be finite and > 0, got {self.mu_r!r}")
        if not np.isfinite(self.sigma) or self.sigma < 0.0:
            raise ValueError(
                f"{self.name}: sigma must be finite and >= 0 [S/m], got "
                f"{self.sigma!r}.  Perfect conductors are handled by removing "
                "their unknowns, not by sigma = inf.")
        if not np.isfinite(self.loss_tangent) or self.loss_tangent < 0.0:
            raise ValueError(f"{self.name}: loss_tangent must be finite and >= 0")
        if self.kind not in _KINDS:
            raise ValueError(
                f"{self.name}: kind must be one of {_KINDS}, got {self.kind!r}")
        if self.kind == "semiconductor" and self.semi is None:
            raise ValueError(
                f"{self.name}: kind='semiconductor' requires SemiconductorParams; "
                "without them the drift-diffusion solver has nothing to solve")
        if self.semi is not None and self.kind != "semiconductor":
            raise ValueError(
                f"{self.name}: SemiconductorParams supplied but kind is "
                f"{self.kind!r}; set kind='semiconductor'")
        if self.semi is not None and not isinstance(self.semi, SemiconductorParams):
            raise ValueError(f"{self.name}: semi must be a SemiconductorParams")

    # -- absolute properties ----------------------------------------------
    @property
    def eps(self) -> float:
        """Absolute permittivity [F/m], ``eps0 * eps_r``."""
        return eps0 * self.eps_r

    @property
    def mu(self) -> float:
        """Absolute permeability [H/m], ``mu0 * mu_r``."""
        return mu0 * self.mu_r

    @property
    def is_conductor(self) -> bool:
        """True for materials whose conduction dominates their displacement.

        The test is the dielectric relaxation frequency: a material is called a
        conductor here when ``sigma / (eps0 eps_r) > 1e12 s^-1``, i.e. charge
        relaxes in under a picosecond, which is faster than any signal edge this
        package targets.
        """
        return self.kind == "conductor" or self.sigma / self.eps > 1e12

    def relaxation_time(self) -> float:
        """Dielectric relaxation time ``eps / sigma`` [s]; ``inf`` if lossless.

        This is the time constant an explicit scheme would have to resolve
        inside the material.  For copper it is 1.5e-19 s, which is why explicit
        full-wave FDTD cannot step through a metal (see ``docs/ASSUMPTIONS.md``).
        """
        return math.inf if self.sigma == 0.0 else self.eps / self.sigma

    def scaled(self, **kw: float) -> "Material":
        """Return a copy with fields replaced, e.g. ``si.scaled(sigma=1e3)``.

        Materials are frozen, so this is the supported way to make a variant.
        Remember to give the copy a new :attr:`name` before registering it.
        """
        return replace(self, **kw)

    def __repr__(self) -> str:
        bits = [f"eps_r={self.eps_r:g}"]
        if self.mu_r != 1.0:
            bits.append(f"mu_r={self.mu_r:g}")
        if self.sigma:
            bits.append(f"sigma={self.sigma:g} S/m")
        return f"<Material {self.name} {self.kind} " + " ".join(bits) + ">"


# ==========================================================================
# The library
# ==========================================================================
_CM3 = per_cm3          # cm^-3 -> m^-3
_MOB = cm2_per_Vs       # cm^2/(V s) -> m^2/(V s)
_CM6 = 1e-12            # cm^6/s -> m^6/s


_SI_SEMI = SemiconductorParams(
    Eg=1.124,                       # Varshni fit at 300 K; 1.12 is the rounded value
    ni_300=1.0e10 * _CM3,           # measured; Sproul & Green 1991 give 9.65e9 cm^-3
    chi=4.05,
    mu_n=1350.0 * _MOB,
    mu_p=480.0 * _MOB,
    Nc_300=2.8e19 * _CM3,
    Nv_300=1.04e19 * _CM3,          # textbook value; inconsistent with ni, see ni()
    tau_n=1.0e-5,
    tau_p=1.0e-5,
    vsat_n=1.035e5,                 # Canali 1975, 1.035e7 cm/s at 300 K
    vsat_p=8.37e4,
    C_auger_n=2.8e-31 * _CM6,       # Dziewior & Schmid 1977
    C_auger_p=9.9e-32 * _CM6,
    varshni_alpha=4.73e-4,
    varshni_beta=636.0,
    ref=("Sze & Ng 3rd ed. App. G; Green JAP 67, 2944 (1990). "
         "SRH lifetime is float-zone-grade and spans 1e-9..1e-2 s in practice. "
         "Nv=1.04e19 cm^-3 does not reproduce ni=1e10 cm^-3: see ni()."),
)

_GAAS_SEMI = SemiconductorParams(
    Eg=1.424,
    ni_300=2.1e6 * _CM3,
    chi=4.07,
    mu_n=8500.0 * _MOB,
    mu_p=400.0 * _MOB,
    Nc_300=4.7e17 * _CM3,
    Nv_300=9.0e18 * _CM3,
    tau_n=1.0e-8,
    tau_p=1.0e-8,
    vsat_n=7.7e4,
    vsat_p=8.0e4,
    C_auger_n=1.0e-30 * _CM6,
    C_auger_p=1.0e-30 * _CM6,
    varshni_alpha=5.405e-4,
    varshni_beta=204.0,
    ref=("Sze & Ng 3rd ed. UNCERTAIN: GaAs has negative differential mobility "
         "above ~3.5 kV/cm, so a single vsat is a poor model and the Gunn "
         "regime is outside drift-diffusion. Recombination is radiative-"
         "dominated (B ~ 7.2e-10 cm^3/s), which this parameter set cannot "
         "express -- tau_n/tau_p are effective values only."),
)

_SIC_SEMI = SemiconductorParams(
    Eg=3.23,                        # 4H polytype
    ni_300=1.5e-8 * _CM3,           # UNCERTAIN: sources span 1e-9..1e-8 cm^-3
    chi=3.7,                        # UNCERTAIN: reported 3.6..4.0 eV
    mu_n=900.0 * _MOB,              # perpendicular to c; ~1200 along c (anisotropic)
    mu_p=120.0 * _MOB,
    Nc_300=1.7e19 * _CM3,
    Nv_300=2.5e19 * _CM3,
    tau_n=5.0e-7,
    tau_p=5.0e-7,
    vsat_n=2.0e5,
    vsat_p=2.0e5,
    C_auger_n=5.0e-31 * _CM6,
    C_auger_p=5.0e-31 * _CM6,
    varshni_alpha=6.5e-4,
    varshni_beta=1300.0,            # gives Eg(0) = 3.267 eV, matching 4H-SiC
    ref=("4H-SiC, Ioffe database + Levinshtein et al. UNCERTAIN throughout: "
         "SiC is anisotropic (mobility and permittivity differ ~15 percent "
         "between axes) and this isotropic entry averages that away (A3). "
         "Aluminium acceptors are deep (~200 meV), so complete ionisation "
         "(A11) is wrong for p-type SiC at 300 K by roughly an order of "
         "magnitude."),
)

_IGZO_SEMI = SemiconductorParams(
    Eg=3.05,
    ni_300=1.0e-1,                  # m^-3; physically meaningless, see ref
    chi=4.16,
    mu_n=10.0 * _MOB,
    mu_p=1.0e-2 * _MOB,             # a-IGZO has essentially no hole transport
    Nc_300=5.0e18 * _CM3,
    Nv_300=5.0e18 * _CM3,
    tau_n=1.0e-6,
    tau_p=1.0e-6,
    vsat_n=1.0e5,
    vsat_p=1.0e5,
    C_auger_n=0.0,
    C_auger_p=0.0,
    varshni_alpha=0.0,              # amorphous: no Varshni fit exists
    varshni_beta=1.0,
    ref=("a-IGZO (In:Ga:Zn:O ~ 1:1:1). LOW CONFIDENCE. Eg 3.0-3.2 eV and "
         "chi ~4.16 eV are well established; everything else is a placeholder. "
         "ni is meaningless -- free electrons come from oxygen vacancies at "
         "1e15..1e17 cm^-3, not from thermal generation -- and conduction is "
         "controlled by the sub-gap trap DOS, which violates both Boltzmann "
         "statistics and complete ionisation (A5, A11). Hole transport does "
         "not exist in any useful sense; mu_p is a numerical floor, not a "
         "measurement. Field-effect mobility 5-20 cm^2/Vs is the reliable "
         "number and it is what mu_n is set from."),
)

_ASI_SEMI = SemiconductorParams(
    Eg=1.75,                        # Tauc optical gap; mobility gap ~1.8 eV
    ni_300=5.0e11,                  # m^-3, from the DOS product; see ref
    chi=3.9,
    mu_n=1.0 * _MOB,                # band mobility; TFT field-effect 0.5-1
    mu_p=5.0e-3 * _MOB,
    Nc_300=2.5e20 * _CM3,
    Nv_300=2.5e20 * _CM3,
    tau_n=1.0e-6,
    tau_p=1.0e-6,
    vsat_n=1.0e5,
    vsat_p=1.0e5,
    C_auger_n=0.0,
    C_auger_p=0.0,
    varshni_alpha=0.0,              # amorphous
    varshni_beta=1.0,
    ref=("a-Si:H. LOW CONFIDENCE. Transport is dominated by band-tail and "
         "dangling-bond states; a two-band drift-diffusion model with these "
         "numbers reproduces the shape of a TFT curve but not its magnitude. "
         "Nc/Nv are conventional band-edge values for an amorphous DOS and "
         "are not measurements. Also subject to Staebler-Wronski degradation, "
         "which is not modelled anywhere in fieldspice."),
)


def _M(*a, **kw) -> Material:
    return Material(*a, **kw)


LIBRARY: dict[str, Material] = {
    # -- reference media ---------------------------------------------------
    "vacuum": _M("vacuum", eps_r=1.0, mu_r=1.0, sigma=0.0, kind="vacuum",
                 color="#ffffff", ref="Exact by definition."),
    "air": _M("air", eps_r=1.000589, mu_r=1.00000037, sigma=0.0,
              kind="dielectric", color="#eef6ff",
              ref="Dry air, 1 atm, 20 C, 1 MHz (CRC Handbook). Humidity "
                  "raises eps_r by ~1e-4 at 50 percent RH; irrelevant here."),

    # -- semiconductors ----------------------------------------------------
    "si": _M("si", eps_r=11.7, sigma=2.9e-4, kind="semiconductor",
             semi=_SI_SEMI, color="#6a6a80",
             ref="eps_r 11.7 (11.9 is also widely quoted; the spread is real "
                 "and ~2 percent). sigma is the intrinsic value "
                 "q*ni*(mu_n+mu_p) = 2.9e-4 S/m, i.e. rho = 3.4e5 ohm cm; the "
                 "frequently quoted 2.3e5 ohm cm dates from the older "
                 "ni = 1.45e10 cm^-3 and is not consistent with the ni used "
                 "here. It applies ONLY when silicon is treated as a lossy "
                 "dielectric by EQS. Doped silicon must either get its own "
                 "Material via .scaled(sigma=...) or be handed to the "
                 "drift-diffusion solver."),
    "gaas": _M("gaas", eps_r=12.9, sigma=1.0e-9, kind="semiconductor",
               semi=_GAAS_SEMI, color="#8b1a1a",
               ref="eps_r 12.9 static. Semi-insulating GaAs reaches 1e-9..1e-7 "
                   "S/m; the figure here is nominal."),
    "sic": _M("sic", eps_r=9.76, sigma=1.0e-12, kind="semiconductor",
              semi=_SIC_SEMI, color="#2f2f33",
              ref="4H-SiC, eps_r 9.76 perpendicular to c and 10.32 parallel; "
                  "the isotropic entry uses the perpendicular value (A3)."),
    "igzo": _M("igzo", eps_r=16.0, sigma=1.0e-2, kind="semiconductor",
               semi=_IGZO_SEMI, color="#ff8c42",
               ref="a-IGZO. eps_r 16 is the value fixed by docs/CONTRACTS.md; "
                   "the literature spans 10-16 depending on composition and "
                   "measurement frequency, so treat it as +-40 percent. sigma "
                   "1e-2 S/m corresponds to n ~ 1e17 cm^-3 at 10 cm^2/Vs and "
                   "is STRONGLY process dependent -- annealed channel IGZO "
                   "spans 1e-6..1e2 S/m across the literature."),
    "a_si": _M("a_si", eps_r=11.9, sigma=1.0e-7, kind="semiconductor",
               semi=_ASI_SEMI, color="#8b7d6b",
               ref="a-Si:H. sigma is the dark conductivity of device-grade "
                   "intrinsic material, ~1e-9 S/cm; it is photoconductive over "
                   "4-5 orders of magnitude under illumination, which is not "
                   "modelled."),
    "poly": _M("poly", eps_r=11.7, sigma=1.0e4, kind="conductor",
               color="#b04ea6",
               ref="Doped polysilicon gate/interconnect. sigma 1e4 S/m (fixed "
                   "by docs/CONTRACTS.md) is rho = 1e-2 ohm cm, a moderately "
                   "doped film; n+ poly with a silicide strap reaches 1e5..1e6 "
                   "S/m. Classed as a conductor because that is how the field "
                   "solvers should treat it, even though it is silicon."),

    # -- dielectrics -------------------------------------------------------
    "sio2": _M("sio2", eps_r=3.9, sigma=1.0e-14, kind="dielectric",
               color="#7fd4f0", loss_tangent=1.0e-4,
               ref="Thermal oxide. sigma is nominal: bulk resistivity spans "
                   "1e14..1e16 ohm cm, and in a thin gate oxide the actual "
                   "leakage is direct tunnelling, which is not a conductivity "
                   "at all and is not modelled (A5 exclusions). The nonzero "
                   "value exists to keep the DC conductance matrix "
                   "nonsingular; note it puts 21 orders of magnitude between "
                   "oxide and copper, so use a direct linear solver."),
    "si3n4": _M("si3n4", eps_r=7.5, sigma=1.0e-14, kind="dielectric",
                color="#2e8b57", loss_tangent=5.0e-4,
                ref="LPCVD stoichiometric nitride. PECVD films are "
                    "silicon-rich and run eps_r 6-9 with much higher leakage."),
    "hfo2": _M("hfo2", eps_r=25.0, sigma=1.0e-12, kind="dielectric",
               color="#8a2be2", loss_tangent=2.0e-3,
               ref="ALD HfO2. eps_r 25 is the fixed contract value; measured "
                   "films run 16-25 (monoclinic ~16-18, tetragonal ~30+), so "
                   "this is phase dependent to +-40 percent. sigma is a "
                   "placeholder -- real leakage is trap-assisted tunnelling."),
    "alumina": _M("alumina", eps_r=9.8, sigma=1.0e-12, kind="dielectric",
                  color="#e0d5c8", loss_tangent=1.0e-4,
                  ref="99.5 percent sintered Al2O3 substrate at 1 MHz. ALD "
                      "amorphous Al2O3 is lower, ~7-9."),
    "aln": _M("aln", eps_r=8.9, sigma=1.0e-11, kind="dielectric",
              color="#cfe6ff", loss_tangent=3.0e-4,
              ref="AlN is uniaxial: eps_r 8.5 parallel to c and 9.14 "
                  "perpendicular (Collins et al.); 8.9 is the isotropic "
                  "average this contract specifies (A3 discards the "
                  "anisotropy). Sputtered c-axis films on Mo/Pt are the "
                  "relevant case and match the bulk value within ~5 percent. "
                  "Piezoelectricity (e33 ~ 1.55 C/m^2) is NOT modelled."),
    "scaln": _M("scaln", eps_r=16.0, sigma=1.0e-11, kind="dielectric",
                color="#9ec5ff", loss_tangent=2.0e-3,
                ref="Sc0.3Al0.7N. LOW CONFIDENCE: eps_r rises steeply with Sc "
                    "content and reported values for x~0.3 span 12-22 "
                    "(Fichtner et al. JAP 125, 114103 (2019) and later work), "
                    "with a further large increase once the film is "
                    "ferroelectric and being switched. 16 is a mid-range "
                    "figure for the linear, unswitched state. Ferroelectric "
                    "hysteresis is explicitly NOT supported (A3)."),
    "fr4": _M("fr4", eps_r=4.4, sigma=1.0e-11, kind="dielectric",
              color="#2e7d32", loss_tangent=0.020,
              ref="Woven-glass epoxy laminate. eps_r spans 4.2-4.7 by weave "
                  "and resin content and is anisotropic and frequency "
                  "dependent -- +-8 percent is the honest error bar. The DC "
                  "sigma given here is nearly irrelevant: at 1 GHz the "
                  "effective loss conductivity omega*eps*tan_d is 4.9e-3 S/m, "
                  "eight orders larger, and only the ac solver's complex "
                  "permittivity can represent it (A3)."),

    # -- metals ------------------------------------------------------------
    "cu": _M("cu", sigma=5.8e7, kind="conductor", color="#b87333",
             ref="Annealed copper at 20 C (IACS 100 percent = 5.80e7 S/m). "
                 "Damascene copper below ~100 nm linewidth is 30-100 percent "
                 "more resistive from grain-boundary and surface scattering, "
                 "which is not modelled (A4)."),
    "al": _M("al", sigma=3.77e7, kind="conductor", color="#c8c8d0",
             ref="Pure aluminium at 20 C. Al-Cu-Si metallisation alloys run "
                 "2.8e7..3.3e7 S/m."),
    "w": _M("w", sigma=1.79e7, kind="conductor", color="#4f4f5a",
            ref="Bulk tungsten at 20 C. CVD W plug/via films are typically "
                "1.0e7..1.4e7 S/m."),
    "mo": _M("mo", sigma=1.87e7, kind="conductor", color="#708090",
             ref="Bulk molybdenum at 20 C. Sputtered Mo electrodes -- the "
                 "usual bottom metal under AlN/ScAlN -- measure 3e6..1e7 S/m, "
                 "so use a scaled copy for thin-film work."),
    "ti": _M("ti", sigma=2.38e6, kind="conductor", color="#9aa0a6",
             ref="Bulk titanium at 20 C. Used as an adhesion/barrier layer "
                 "where it is thin enough that its sheet resistance, not its "
                 "bulk sigma, is what matters."),
    "pt": _M("pt", sigma=9.4e6, kind="conductor", color="#dcdcdc",
             ref="Bulk platinum at 20 C. Sputtered Pt films run 4e6..8e6 S/m."),
    "ito": _M("ito", eps_r=9.0, sigma=1.0e6, kind="conductor", color="#4fd1c5",
              ref="Sn-doped In2O3, degenerate n-type. sigma 1e6 S/m is the "
                  "contract value and corresponds to a good annealed film "
                  "(~10 ohm/sq at 100 nm); as-deposited room-temperature films "
                  "are 3-10x worse. eps_r 9 is the static In2O3 value and is "
                  "UNCERTAIN -- the optical-frequency value is ~4, and ITO is "
                  "a Drude metal above its ~1.5 um plasma wavelength, which "
                  "the real-scalar model (A3) cannot represent."),
}

# Human spellings and common variants, all mapped onto the canonical keys.
_ALIASES: dict[str, str] = {
    "silicon": "si", "c_si": "si", "c-si": "si",
    "oxide": "sio2", "silicon_dioxide": "sio2", "glass": "sio2",
    "quartz": "sio2", "thermox": "sio2",
    "nitride": "si3n4", "silicon_nitride": "si3n4",
    "copper": "cu", "aluminum": "al", "aluminium": "al",
    "tungsten": "w", "molybdenum": "mo", "titanium": "ti", "platinum": "pt",
    "polysilicon": "poly", "poly_si": "poly", "polysi": "poly",
    "al2o3": "alumina", "sapphire": "alumina",
    "amorphous_si": "a_si", "asi": "a_si",
    "gallium_arsenide": "gaas", "4h_sic": "sic", "sic_4h": "sic",
    "hafnia": "hfo2", "hf02": "hfo2",
    "scalnn": "scaln", "sc_aln": "scaln", "alscn": "scaln",
    "indium_tin_oxide": "ito",
    "vac": "vacuum", "free_space": "vacuum",
}


def _norm(name: str) -> str:
    """Canonicalise a material name: lowercase, ``-`` and spaces to ``_``."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def available() -> tuple[str, ...]:
    """Sorted tuple of every registered canonical material name."""
    return tuple(sorted(LIBRARY))


def get(name: str | Material) -> Material:
    """Look up a material by name.

    Parameters
    ----------
    name : str or Material
        Case-insensitive name; ``-`` and spaces are treated as ``_`` so
        ``"a-Si"``, ``"A_SI"`` and ``"a_si"`` all resolve.  A
        :class:`Material` instance is passed straight through, which lets every
        API in fieldspice accept ``str | Material`` without duplicating logic.

    Returns
    -------
    Material

    Raises
    ------
    UnknownMaterial
        If the name is not registered.  The message lists close matches.
        The exception derives from both ``KeyError`` and ``ValueError``.
    """
    if isinstance(name, Material):
        return name
    if not isinstance(name, str):
        raise ValueError(
            f"material must be a str or Material, got {type(name).__name__}")
    key = _norm(name)
    key = _ALIASES.get(key, key)
    try:
        return LIBRARY[key]
    except KeyError:
        near = difflib.get_close_matches(key, list(LIBRARY) + list(_ALIASES), n=4)
        hint = f"  Did you mean: {', '.join(near)}?" if near else ""
        raise UnknownMaterial(
            f"unknown material {name!r}. Known: {', '.join(available())}.{hint}"
        ) from None


def register(mat: Material, overwrite: bool = False) -> None:
    """Add a material to :data:`LIBRARY`.

    Parameters
    ----------
    mat : Material
        The material.  Its :attr:`Material.name` is canonicalised to form the
        key.
    overwrite : bool, optional
        Permit replacing an existing, *different* material of the same name.

    Raises
    ------
    ValueError
        If ``mat`` is not a :class:`Material`, if the name collides with an
        alias, or if the name is already taken by a different material and
        ``overwrite`` is False.

    Notes
    -----
    Re-registering an *identical* material is a no-op rather than an error, so
    that re-executing a cell in a notebook does not blow up, while a genuine
    redefinition still has to be made explicit.  This is the compromise that
    keeps the mandated module-level registry from behaving as unguarded global
    state.
    """
    if not isinstance(mat, Material):
        raise ValueError(f"register expects a Material, got {type(mat).__name__}")
    key = _norm(mat.name)
    if key in _ALIASES:
        raise ValueError(
            f"{mat.name!r} collides with the alias {key!r} -> {_ALIASES[key]!r}")
    existing = LIBRARY.get(key)
    if existing is not None and existing != mat and not overwrite:
        raise ValueError(
            f"material {key!r} is already registered as {existing!r}; "
            "pass overwrite=True to replace it")
    LIBRARY[key] = mat


# ==========================================================================
# Effective-medium mixing
# ==========================================================================
def mix_property(old: np.ndarray | float, new: float,
                 fill: np.ndarray | float, rule: str = "linear") -> np.ndarray:
    """Blend a background property with a filling material's value.

    Parameters
    ----------
    old : array_like or float
        Existing per-cell value (any units).
    new : float
        Value of the material being introduced, same units.
    fill : array_like or float
        Volume fraction of ``new`` in each cell, in ``[0, 1]``.
    rule : {"linear", "harmonic"}
        ``"linear"`` (a.k.a. parallel, arithmetic):
        ``p = (1-f) p_old + f p_new``.  Exact when the field component is
        **parallel** to the material interface, because the two phases then see
        the same field and their responses add.

        ``"harmonic"`` (a.k.a. series):
        ``1/p = (1-f)/p_old + f/p_new``.  Exact when the field is
        **perpendicular** to the interface, because the two phases then carry
        the same flux and their reciprocals add.

    Returns
    -------
    np.ndarray
        Blended value, same shape as the broadcast of ``old`` and ``fill``.

    Notes
    -----
    Tagged **A2**.  The two rules bracket the true effective property
    (Wiener bounds): linear is the upper bound, harmonic the lower.  A cell
    straddling an interface at a general angle lies between them, so the gap
    between the two answers is a usable *error bar* on the staircase
    discretisation -- run both and compare.

    ``rule="harmonic"`` requires both values to be strictly positive; a zero
    conductivity phase makes the harmonic mean zero, which would disconnect the
    mesh.
    """
    f = np.asarray(fill, dtype=float)
    o = np.asarray(old, dtype=float)
    if rule in ("linear", "parallel", "arithmetic"):
        return (1.0 - f) * o + f * float(new)
    if rule in ("harmonic", "series"):
        if new <= 0.0 or np.any(o <= 0.0):
            raise ValueError(
                "harmonic mixing needs strictly positive values on both sides "
                "(a zero-valued phase drives the harmonic mean to zero); use "
                "rule='linear' for conductivity")
        return 1.0 / ((1.0 - f) / o + f / float(new))
    raise ValueError(f"unknown mixing rule {rule!r}, expected 'linear' or 'harmonic'")


# ==========================================================================
# MaterialMap
# ==========================================================================
class MaterialMap:
    """Per-cell material property assembly on a grid.

    Holds one relative-permittivity, one relative-permeability and one
    conductivity array of shape ``grid.shape_cells``, plus an integer material
    id per cell.  :meth:`eps`, :meth:`mu` and :meth:`sigma` return the
    **absolute** SI arrays the solvers consume.

    Parameters
    ----------
    grid : RectilinearGrid
        The grid to build on.
    background : str or Material, optional
        Material filling every cell before any :meth:`assign` call.  Defaults
        to vacuum.  For a chip cross-section ``"sio2"`` is usually the right
        background; for a PCB, ``"fr4"``.

    Examples
    --------
    >>> from fieldspice.grid import RectilinearGrid
    >>> from fieldspice.materials import MaterialMap
    >>> g = RectilinearGrid.uniform([(0, 1e-6), (0, 1e-6), (0, 1e-6)], (4, 4, 4))
    >>> mm = MaterialMap(g, "sio2")
    >>> import numpy as np
    >>> mask = np.zeros(g.shape_cells, dtype=bool); mask[:, :, :2] = True
    >>> mm.assign(mask, "si")
    >>> float(mm.eps()[0, 0, 0] / mm.eps()[0, 0, 3])   # 11.7 / 3.9
    3.0

    Notes
    -----
    Tagged **A3** (real scalar properties per cell) and **A2** (sub-cell
    effective-medium mixing of fill fractions).
    """

    def __init__(self, grid: RectilinearGrid,
                 background: str | Material = "vacuum") -> None:
        if not isinstance(grid, RectilinearGrid):
            raise ValueError(
                f"grid must be a RectilinearGrid, got {type(grid).__name__}")
        self.grid = grid
        bg = get(background)
        shape = grid.shape_cells
        self._eps_r = np.full(shape, bg.eps_r, dtype=float)
        self._mu_r = np.full(shape, bg.mu_r, dtype=float)
        self._sigma = np.full(shape, bg.sigma, dtype=float)
        self._ids = np.zeros(shape, dtype=np.int32)
        self._materials: list[Material] = [bg]
        self._index: dict[str, int] = {bg.name: 0}

    # -- introspection -----------------------------------------------------
    @property
    def background(self) -> Material:
        """The material every cell started as."""
        return self._materials[0]

    @property
    def materials(self) -> tuple[Material, ...]:
        """Materials in id order; ``materials[ids()[i, j, k]]`` is the cell's."""
        return tuple(self._materials)

    @property
    def names(self) -> tuple[str, ...]:
        """Material names in id order."""
        return tuple(m.name for m in self._materials)

    def index_of(self, material: str | Material) -> int:
        """Integer id of a material already present in this map.

        Raises
        ------
        ValueError
            If the material has never been assigned to this map.
        """
        mat = get(material)
        if mat.name not in self._index:
            raise ValueError(
                f"{mat.name!r} is not present in this map; present: "
                f"{', '.join(self.names)}")
        return self._index[mat.name]

    # -- assignment --------------------------------------------------------
    def assign(self, mask: np.ndarray, material: str | Material,
               mix: str = "linear", mix_sigma: str = "linear") -> None:
        """Place a material into the cells selected by ``mask``.

        Parameters
        ----------
        mask : np.ndarray
            Either a **boolean** array (hard 0/1 selection) or a **float**
            array of sub-cell fill fractions in ``[0, 1]``, as produced by
            :func:`fieldspice.geometry.voxelize`.  Shape must be
            ``grid.shape_cells``, or any shape that broadcasts to it
            (``(Nx, 1, 1)`` for a slab, for instance).
        material : str or Material
            What to put there.
        mix : {"linear", "harmonic"}
            Effective-medium rule for **permittivity** in partially filled
            cells.  ``"linear"`` (default) is exact for field components
            parallel to the interface; ``"harmonic"`` is exact for the
            perpendicular component.  See :func:`mix_property`.
        mix_sigma : {"linear", "harmonic"}
            Rule for **conductivity**.  Defaults to ``"linear"`` and should
            normally stay there: harmonic mixing of a conductor with a perfect
            insulator gives exactly zero, which would sever the current path
            through every boundary cell of every wire.  The escape hatch exists
            because in the electroquasistatic operator sigma and eps enter
            identically, so a cell straddling a *perpendicular* interface
            strictly wants both mixed harmonically to get its RC time constant
            right.
        Returns
        -------
        None
            The map is modified in place; later calls overwrite earlier ones
            where ``fill == 1`` and blend where ``0 < fill < 1``.

        Raises
        ------
        ValueError
            On a shape that does not broadcast to the cell grid, on fill values
            outside ``[0, 1]``, on NaN, or on a name already used in this map by
            a different material.

        Notes
        -----
        Tagged **A2**.  The integer id of a mixed cell is assigned by majority:
        a cell takes the new material's id when ``fill >= 0.5``.  That matters
        for :meth:`material_at` and hence for the drift-diffusion solver, which
        needs an unambiguous :class:`SemiconductorParams` per cell -- run
        semiconductor problems with hard masks (``voxelize(..., subsample=1)``)
        so that no cell is ambiguous.
        """
        mat = get(material)
        f = self._as_fill(mask)

        prev = self._index.get(mat.name)
        if prev is not None and self._materials[prev] != mat:
            raise ValueError(
                f"a different material is already registered in this map under "
                f"the name {mat.name!r}; rename one of them")
        if prev is None:
            prev = len(self._materials)
            self._materials.append(mat)
            self._index[mat.name] = prev

        touched = f > 0.0
        if not np.any(touched):
            return  # material still recorded, so ids stay stable across runs

        self._eps_r = mix_property(self._eps_r, mat.eps_r, f, rule=mix)
        self._sigma = mix_property(self._sigma, mat.sigma, f, rule=mix_sigma)
        # mu is mixed harmonically because the dual edge threading a face
        # crosses its two cells in series along the magnetic path, which is the
        # same convention operators.cell_to_face uses by default.
        self._mu_r = mix_property(self._mu_r, mat.mu_r, f, rule="harmonic")
        self._ids = np.where(f >= 0.5, np.int32(prev), self._ids)

    def _as_fill(self, mask: np.ndarray) -> np.ndarray:
        """Validate and normalise a mask/fill array to float in [0, 1]."""
        a = np.asarray(mask)
        if a.dtype == bool:
            a = a.astype(float)
        else:
            a = a.astype(float, copy=False)
            if not np.all(np.isfinite(a)):
                raise ValueError("fill fractions must be finite (no NaN or inf)")
            if a.size and (a.min() < 0.0 or a.max() > 1.0):
                raise ValueError(
                    f"fill fractions must lie in [0, 1], got range "
                    f"[{a.min():g}, {a.max():g}]")
        shape = self.grid.shape_cells
        if a.shape == shape:
            return a
        try:
            return np.broadcast_to(a, shape).astype(float)
        except ValueError:
            raise ValueError(
                f"mask shape {a.shape} does not match or broadcast to the cell "
                f"grid {shape}") from None

    # -- property arrays ---------------------------------------------------
    def eps(self) -> np.ndarray:
        """Absolute permittivity per cell [F/m], shape ``grid.shape_cells``.

        This is ``eps0 * eps_r``.  Feed it to
        ``operators.cell_to_edge(grid, matmap.eps())`` and then to
        ``operators.edge_mass`` to get edge capacitances in farads.
        """
        return eps0 * self._eps_r

    def mu(self) -> np.ndarray:
        """Absolute permeability per cell [H/m], ``mu0 * mu_r``."""
        return mu0 * self._mu_r

    def sigma(self) -> np.ndarray:
        """Conductivity per cell [S/m]."""
        return self._sigma.copy()

    def eps_r(self) -> np.ndarray:
        """Relative permittivity per cell [dimensionless]."""
        return self._eps_r.copy()

    def mu_r(self) -> np.ndarray:
        """Relative permeability per cell [dimensionless]."""
        return self._mu_r.copy()

    def ids(self) -> np.ndarray:
        """Dominant material id per cell, ``int32``, shape ``grid.shape_cells``.

        Index into :attr:`materials`.  Mixed cells are attributed by majority
        fill, so this array is a *classification*, not the quantity the solvers
        use -- those read :meth:`eps`, :meth:`mu` and :meth:`sigma`, which do
        carry the blend.
        """
        return self._ids.copy()

    def mask(self, material: str | Material) -> np.ndarray:
        """Boolean array of cells whose dominant material is ``material``."""
        return self._ids == self.index_of(material)

    def volume_fraction(self, material: str | Material) -> float:
        """Fraction of the domain **volume** whose dominant material is this one.

        Volume weighted, so a graded mesh does not skew the answer.
        """
        vol = self.grid.cell_volumes()
        return float(vol[self.mask(material)].sum() / vol.sum())

    # -- point queries -----------------------------------------------------
    def material_at(self, i: int, j: int, k: int) -> Material:
        """Dominant :class:`Material` of cell ``(i, j, k)``.

        Negative indices count from the end, as in NumPy.

        Raises
        ------
        ValueError
            If an index is out of range for the cell grid.
        """
        shape = self.grid.shape_cells
        idx = []
        for n, (v, s) in enumerate(zip((i, j, k), shape)):
            v = int(v)
            if v < 0:
                v += s
            if not 0 <= v < s:
                raise ValueError(
                    f"cell index {n} out of range: got {(i, j, k)[n]} for a "
                    f"grid of {shape} cells")
            idx.append(v)
        return self._materials[int(self._ids[idx[0], idx[1], idx[2]])]

    def semiconductor_mask(self) -> np.ndarray:
        """Boolean cell array: True where the dominant material is a semiconductor."""
        semi_ids = [n for n, m in enumerate(self._materials)
                    if m.kind == "semiconductor"]
        if not semi_ids:
            return np.zeros(self.grid.shape_cells, dtype=bool)
        return np.isin(self._ids, semi_ids)

    # -- diagnostics -------------------------------------------------------
    def contrast(self) -> dict[str, float]:
        """Property ratios across the map --- a conditioning smell test.

        Returns
        -------
        dict
            ``eps_ratio``, ``sigma_ratio`` (largest over smallest *nonzero*),
            and ``min_relaxation_time`` [s].  A ``sigma_ratio`` above ~1e12
            means an iterative linear solver will struggle and the direct path
            should be used; a metal against a good insulator routinely gives
            1e18 or more, which is physics, not a bug.
        """
        s = self._sigma[self._sigma > 0.0]
        eps_abs = eps0 * self._eps_r
        return {
            "eps_ratio": float(self._eps_r.max() / self._eps_r.min()),
            "sigma_ratio": float(s.max() / s.min()) if s.size else 1.0,
            "min_relaxation_time": (float((eps_abs / self._sigma)[self._sigma > 0].min())
                                    if s.size else math.inf),
        }

    def summary(self) -> str:
        """Multi-line human-readable description of the map."""
        lines = [f"MaterialMap on {self.grid!r}",
                 f"  background {self.background.name}"]
        for n, m in enumerate(self._materials):
            frac = self.volume_fraction(m)
            lines.append(f"  [{n}] {m.name:<8s} eps_r={m.eps_r:<7g} "
                         f"sigma={m.sigma:<10.4g} S/m  {100 * frac:6.2f} % of volume")
        c = self.contrast()
        lines.append(f"  eps ratio {c['eps_ratio']:.3g}, "
                     f"sigma ratio {c['sigma_ratio']:.3g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"<MaterialMap {self.grid.ncell} cells, "
                f"{len(self._materials)} materials: {', '.join(self.names)}>")
