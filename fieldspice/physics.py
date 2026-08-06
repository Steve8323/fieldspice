"""Switchable physics: which mechanisms are active, and which assumptions remain.

Every optional mechanism in fieldspice is off by default and turned on here.
The reason for a single object rather than scattered keyword arguments is that
each switch **relaxes a documented assumption**, and the two must not drift
apart: if you enable self-heating, assumption A6 (isothermal) no longer
applies and should stop appearing in ``Result.meta["assumptions"]``.

    >>> from fieldspice.physics import PhysicsOptions
    >>> opts = PhysicsOptions(self_heating=True)
    >>> "A6" in opts.remaining_assumptions(("A5", "A6", "A11"))
    False

Defaults are chosen so that turning nothing on reproduces the isothermal,
deterministic, linear-material behaviour every existing result in this
repository was validated against. Enabling a mechanism can only *add* physics.

What is deliberately not here
-----------------------------
Switches are only offered for mechanisms that are actually implemented. There
is no ``ballistic_transport``, ``noise`` or ``ferroelectric`` flag, because a
flag that raises ``NotImplementedError`` is worse than an honest absence --- it
implies the capability exists and is merely disabled. See
``docs/KNOWN_ISSUES.md`` for what is genuinely missing.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

__all__ = ["PhysicsOptions", "SWITCH_ASSUMPTIONS"]


#: Which assumption tag each switch relaxes when enabled.
SWITCH_ASSUMPTIONS: dict[str, str] = {
    "self_heating": "A6",
    "temperature_dependent_sigma": "A6",
    "temperature_dependent_kappa": "A6",
    "field_dependent_mobility": "A5",
    "fermi_dirac": "A5",
    "incomplete_ionisation": "A11",
    "impact_ionisation": "A5",
}


@dataclass(frozen=True)
class PhysicsOptions:
    """Which optional mechanisms a solver should include.

    Attributes
    ----------
    self_heating
        Solve the lattice heat equation coupled to the electrical problem.
        Joule power becomes a heat source, temperature feeds back into
        conductivity (and, in drift-diffusion, into mobility and ``ni``).
        Relaxes **A6**. This is the mechanism most likely to change an answer
        qualitatively rather than quantitatively, because the feedback can be
        unstable: see :class:`~fieldspice.solvers.electrothermal.ThermalRunaway`.
    temperature_dependent_sigma
        Use ``Material.sigma_at(T)`` instead of the nominal conductivity.
        Implied by ``self_heating`` unless explicitly disabled; a metal with a
        positive temperature coefficient is what closes the runaway loop, so
        turning self-heating on without this models heating that cannot bite
        back.
    temperature_dependent_kappa
        Use ``Material.kappa_at(T)``. Matters for silicon (kappa nearly halves
        between 300 K and 500 K) and much less for metals.
    field_dependent_mobility
        Caughey-Thomas velocity saturation in drift-diffusion.
    fermi_dirac
        Fermi-Dirac rather than Boltzmann statistics. Needed above roughly
        1e19 cm^-3, where Boltzmann overestimates carrier density.
    incomplete_ionisation
        Dopants not fully ionised. Relaxes **A11**; matters below ~100 K and
        for deep dopants in wide-bandgap material.
    impact_ionisation
        Avalanche generation term.
    ambient_temperature
        Heat-sink / ambient temperature [K] used by the thermal boundary
        conditions.
    max_coupling_iterations
        Cap on the electro-thermal Gummel loop.
    coupling_tolerance
        Convergence threshold on the temperature update [K].
    """

    self_heating: bool = False
    temperature_dependent_sigma: bool | None = None
    temperature_dependent_kappa: bool = False
    field_dependent_mobility: bool = False
    fermi_dirac: bool = False
    incomplete_ionisation: bool = False
    impact_ionisation: bool = False

    ambient_temperature: float = 300.0
    max_coupling_iterations: int = 50
    coupling_tolerance: float = 1e-4

    def __post_init__(self):
        if self.ambient_temperature <= 0.0:
            raise ValueError("ambient_temperature must be positive [K]")
        if self.max_coupling_iterations < 1:
            raise ValueError("max_coupling_iterations must be >= 1")
        if self.coupling_tolerance <= 0.0:
            raise ValueError("coupling_tolerance must be positive [K]")

    @property
    def sigma_varies_with_T(self) -> bool:
        """Resolved value of ``temperature_dependent_sigma``.

        ``None`` (the default) means "follow ``self_heating``", so the common
        case of switching heating on gives a physically complete loop without
        a second flag.
        """
        if self.temperature_dependent_sigma is None:
            return self.self_heating
        return bool(self.temperature_dependent_sigma)

    def enabled(self) -> tuple[str, ...]:
        """Names of the switches that are on."""
        out = []
        for f in fields(self):
            if f.name in SWITCH_ASSUMPTIONS:
                val = (self.sigma_varies_with_T
                       if f.name == "temperature_dependent_sigma"
                       else getattr(self, f.name))
                if val:
                    out.append(f.name)
        return tuple(out)

    def relaxed_assumptions(self) -> frozenset[str]:
        """Assumption tags that no longer apply, given the enabled switches."""
        return frozenset(SWITCH_ASSUMPTIONS[n] for n in self.enabled())

    def remaining_assumptions(self, base: tuple[str, ...]) -> tuple[str, ...]:
        """Filter a solver's assumption tuple by what is still assumed."""
        relaxed = self.relaxed_assumptions()
        return tuple(a for a in base if a not in relaxed)

    def summary(self) -> str:
        on = self.enabled()
        if not on:
            return "PhysicsOptions: all optional mechanisms off (isothermal, " \
                   "Boltzmann, fully ionised, temperature-independent)"
        relaxed = sorted(self.relaxed_assumptions())
        return (f"PhysicsOptions: {', '.join(on)} enabled; "
                f"relaxes {', '.join(relaxed)}")
