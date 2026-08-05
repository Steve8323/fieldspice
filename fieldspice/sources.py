"""Time-domain excitation waveforms for terminals, netlists and field sources.

Every generator in this module returns a *callable* ``f(t)`` that evaluates the
waveform.  That is all a source is in fieldspice: :class:`~fieldspice.solvers.
base.Terminal` stores ``voltage=f`` or ``current=f`` and calls it once per time
step, the MNA netlist does the same for its independent sources, and the
full-wave solver uses one to modulate an injected current density.  Nothing
here knows about the grid, so the same waveform object drives a field solve, a
circuit solve and a coupled solve identically.

Units
-----
Strict SI, as everywhere in fieldspice.  Every time argument, and the argument
of the returned callable, is in **seconds**.  Amplitudes are *dimensionless as
far as the generator is concerned*: the same object is in volts when it is
attached to ``Terminal.voltage`` and in amperes when it is attached to
``Terminal.current``.  Consequently ``ramp(slope=...)`` is in V/s or A/s,
``sine(freq=...)`` is in Hz, and ``prbs(bit_period=...)`` is in seconds per
bit.  Phase is in radians.

Vectorisation
-------------
Every returned callable accepts a scalar ``t`` **or** a NumPy array of any
shape, and returns the matching thing: a Python ``float`` for scalar input, an
array of the same shape for array input.  Time-stepping solvers call these
scalar-at-a-time; post-processing and plotting call them on a whole time axis.
Neither path should have to think about it.

Rise time, and why an ideal step is a trap
------------------------------------------
An instantaneous step has a spectrum falling only as ``1/f`` with no corner:
its energy never stops.  Fed to a transient solve with time step ``dt``,
everything above the Nyquist frequency ``1/(2*dt)`` is unresolved, and how it
is mishandled depends on the integrator.  Backward Euler smears the edge over
roughly one step (artificial damping, an under-estimated peak); trapezoidal
integration instead rings at ``1/(2*dt)`` for the rest of the simulation --- the
classic "trapezoidal ringing on a step edge" SPICE artifact --- and no mesh
refinement removes it, because it is a property of the excitation, not of the
mesh.

A real driver has a finite edge, so use one.  The rules of thumb that work:

* ``trise >= 5*dt`` to have the edge resolved at all, ``>= 20*dt`` for a clean
  answer.  Equivalently, pick ``dt`` from the edge, not from the period.
* the useful bandwidth of an edge is ``f_knee ~ 0.35 / trise``; this is also the
  number that decides whether the quasi-static approximation is legitimate
  (see ``docs/ASSUMPTIONS.md``, A1).
* ``shape="smooth"`` (a C1-continuous smoothstep) rolls the spectrum off as
  ``1/f^3`` instead of the ``1/f^2`` of a linear ramp, for free.  Use it when
  the answer is sensitive to high-frequency content.

:func:`step` keeps ``trise=0.0`` as its default only because a mathematical
step is what "step response" means and because the DC-to-DC transition is
sometimes genuinely what is wanted.  It is almost never the right choice for a
transient run.

Composition
-----------
:class:`Waveform` wraps any callable and gives it arithmetic (``+``, ``-``,
``*``, ``/``, unary minus), a time shift (:meth:`Waveform.delay`), function
composition (:meth:`Waveform.compose`) and bulk evaluation
(:meth:`Waveform.sampled`).  Generators return plain closures; wrap them when
you want the algebra::

    v = Waveform(sine(1 * GHz, amp=0.2)) + Waveform(step(1 * ns, trise=50 * ps))
    noisy_clock = Waveform(trapezoid_clock(1 * ns)).delay(120 * ps) * 1.8
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

__all__ = [
    "step", "pulse", "sine", "pwl", "gaussian", "ramp", "prbs",
    "trapezoid_clock", "Waveform", "prbs_sequence", "PRBS_TAPS",
    "EDGE_SHAPES",
]


# The contract writes this as ``Callable[[float], float]``.  It is widened here
# to say what is actually true: array in, array out.  A scalar-only caller is
# unaffected.
TimeFunc = Callable[[float | np.ndarray], float | np.ndarray]


EDGE_SHAPES: tuple[str, ...] = ("linear", "smooth", "smoother")
"""Permitted ``shape`` values for finite edges.

``"linear"`` is the SPICE convention (C0 continuous, spectrum ~1/f^2),
``"smooth"`` is the cubic smoothstep ``3u^2 - 2u^3`` (C1, ~1/f^3),
``"smoother"`` is the quintic ``6u^5 - 15u^4 + 10u^3`` (C2, ~1/f^4).
All three have the same 0-to-100% transition time and the same mean value over
the edge, so swapping one for another does not move a DC operating point.
"""


PRBS_TAPS: dict[int, tuple[int, int]] = {
    7: (7, 6),
    9: (9, 5),
    11: (11, 9),
    15: (15, 14),
    23: (23, 18),
    31: (31, 28),
}
"""Maximal-length LFSR feedback polynomials ``x^n + x^k + 1``, keyed by order.

These are the ITU-T O.150 / O.151 test-pattern polynomials, which is why these
particular orders are the ones everybody's BERT generates:

======  ==================  ============  =========================
order   polynomial          period        common name
======  ==================  ============  =========================
7       x^7 + x^6 + 1       127           PRBS7
9       x^9 + x^5 + 1       511           PRBS9
11      x^11 + x^9 + 1      2047          PRBS11
15      x^15 + x^14 + 1     32767         PRBS15
23      x^23 + x^18 + 1     8388607       PRBS23
31      x^31 + x^28 + 1     2147483647    PRBS31
======  ==================  ============  =========================

Every one of these polynomials is primitive over GF(2), which is what makes the
period exactly ``2^n - 1`` rather than some divisor of it.  ``tests/`` verifies
that by computing the order of the companion matrix, so the table is checked
rather than trusted.
"""


_PRBS_MAX_CACHE = 1 << 24
"""Largest number of PRBS bits ever materialised (16 Mibit, 16 MB as uint8).

Orders up to 23 have a shorter period than this and are cached whole.  Order 31
would need 2 GB for a full period, so it is generated lazily and refuses to run
past this many bits --- at 1 Gb/s that is 16.8 ms of simulated time, which is
several orders of magnitude more than any transient field solve will reach.
"""


# ==========================================================================
# Internal helpers
# ==========================================================================
def _as_time(t: float | np.ndarray) -> tuple[np.ndarray, bool]:
    """Coerce ``t`` [s] to a float array, reporting whether it was scalar."""
    arr = np.asarray(t, dtype=float)
    return arr, arr.ndim == 0


def _out(value: np.ndarray, scalar: bool,
         shape: tuple[int, ...] = ()) -> float | np.ndarray:
    """Return a Python float for scalar input, an array of ``shape`` otherwise."""
    arr = np.asarray(value, dtype=float)
    if scalar:
        return float(arr)
    if arr.shape != shape:
        arr = np.broadcast_to(arr, shape).copy()
    return arr


def _edge_profile(u: np.ndarray, shape: str) -> np.ndarray:
    """Normalised 0-to-1 edge profile on ``u`` in [0, 1] (dimensionless)."""
    if shape == "linear":
        return u
    if shape == "smooth":
        return u * u * (3.0 - 2.0 * u)
    if shape == "smoother":
        return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))
    raise ValueError(f"unknown edge shape {shape!r}; expected one of {EDGE_SHAPES}")


def _ramp01(dt: np.ndarray, width: float, shape: str) -> np.ndarray:
    """0 before the edge, 1 after it, the profile in between.

    Parameters
    ----------
    dt
        Time relative to the start of the edge [s].
    width
        Edge duration [s]; ``0`` gives an ideal step at ``dt == 0``.
    """
    if width <= 0.0:
        return (dt >= 0.0).astype(float)
    return _edge_profile(np.clip(dt / width, 0.0, 1.0), shape)


def _require_finite(name: str, value: float) -> float:
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


def _require_nonneg(name: str, value: float) -> float:
    v = _require_finite(name, value)
    if v < 0.0:
        raise ValueError(f"{name} must be >= 0 s, got {v!r}")
    return v


def _require_positive(name: str, value: float) -> float:
    v = _require_finite(name, value)
    if v <= 0.0:
        raise ValueError(f"{name} must be > 0, got {v!r}")
    return v


def _require_shape(shape: str) -> str:
    if shape not in EDGE_SHAPES:
        raise ValueError(f"unknown edge shape {shape!r}; expected one of {EDGE_SHAPES}")
    return shape


# ==========================================================================
# Elementary waveforms
# ==========================================================================
def step(t0: float, v0: float = 0.0, v1: float = 1.0, trise: float = 0.0,
         shape: str = "linear") -> TimeFunc:
    """A level change from ``v0`` to ``v1`` at time ``t0``.

    Parameters
    ----------
    t0
        Time at which the edge *starts* [s].  With ``trise > 0`` the waveform
        reaches ``v1`` at ``t0 + trise``, so the 50% crossing is at
        ``t0 + trise/2``.
    v0, v1
        Levels before and after the edge [V] (or [A]).
    trise
        Edge duration, 0% to 100% [s].  ``0`` gives an ideal discontinuous
        step.  **This is the default only for compatibility with the textbook
        definition of a step response; a finite ``trise`` is almost always what
        you want.**  An ideal step has unbounded bandwidth, so in a transient
        solve everything above ``1/(2*dt)`` aliases: backward Euler smears the
        edge over a step, trapezoidal rings at the Nyquist frequency forever.
        Use ``trise >= 20*dt`` and the problem disappears.
    shape
        Edge profile, one of :data:`EDGE_SHAPES`.  Ignored when ``trise == 0``.

    Returns
    -------
    callable
        ``f(t)`` with ``t`` in seconds; scalar in, float out; array in, array
        of the same shape out.

    Examples
    --------
    >>> f = step(1e-9, v0=0.0, v1=1.8, trise=100e-12)
    >>> round(f(1.05e-9), 6)
    0.9
    """
    t0 = _require_finite("t0", t0)
    v0 = _require_finite("v0", v0)
    v1 = _require_finite("v1", v1)
    trise = _require_nonneg("trise", trise)
    shape = _require_shape(shape)

    def f(t: float | np.ndarray) -> float | np.ndarray:
        arr, scalar = _as_time(t)
        return _out(v0 + (v1 - v0) * _ramp01(arr - t0, trise, shape),
                    scalar, arr.shape)

    return f


def pulse(t0: float, width: float, v0: float = 0.0, v1: float = 1.0,
          trise: float = 0.0, tfall: float = 0.0,
          period: float | None = None, shape: str = "linear") -> TimeFunc:
    """A single pulse or a periodic pulse train, SPICE ``PULSE`` semantics.

    The segment layout inside one period, measured from ``t0``:
    rise over ``trise``, flat top at ``v1`` for ``width``, fall over ``tfall``,
    then ``v0`` for the remainder.  ``width`` is therefore the *flat-top* width,
    not the width at 50% (which is ``width + trise/2 + tfall/2``), exactly as in
    SPICE.  Before ``t0`` the output is ``v0``.

    Parameters
    ----------
    t0
        Delay before the first rising edge starts [s].
    width
        Flat-top duration at ``v1`` [s].  May be 0.
    v0, v1
        Low and high levels [V] (or [A]).
    trise, tfall
        Edge durations, 0% to 100% [s].  See :func:`step` for why 0 is a poor
        choice in a transient solve.
    period
        Repetition period [s], or ``None`` for a single pulse.  Must be at
        least ``trise + width + tfall``.
    shape
        Edge profile, one of :data:`EDGE_SHAPES`.

    Returns
    -------
    callable
        ``f(t)``, vectorised over ``t``.

    Raises
    ------
    ValueError
        If any duration is negative, or if ``period`` cannot contain one pulse.
    """
    t0 = _require_finite("t0", t0)
    width = _require_nonneg("width", width)
    v0 = _require_finite("v0", v0)
    v1 = _require_finite("v1", v1)
    trise = _require_nonneg("trise", trise)
    tfall = _require_nonneg("tfall", tfall)
    shape = _require_shape(shape)

    t_fall_start = trise + width
    t_low_start = t_fall_start + tfall
    if period is not None:
        period = _require_positive("period", period)
        if period < t_low_start * (1.0 - 1e-12):
            raise ValueError(
                f"period {period!r} s is shorter than one pulse "
                f"(trise + width + tfall = {t_low_start!r} s)")

    def f(t: float | np.ndarray) -> float | np.ndarray:
        arr, scalar = _as_time(t)
        if period is None:
            tau = arr - t0
        else:
            # A negative sentinel keeps the pre-t0 region low, matching SPICE,
            # instead of extending the train backwards in time.
            tau = np.where(arr < t0, -1.0, np.mod(arr - t0, period))
        rising = v0 + (v1 - v0) * _ramp01(tau, trise, shape)
        falling = v1 + (v0 - v1) * _ramp01(tau - t_fall_start, tfall, shape)
        return _out(np.where(tau < t_fall_start, rising, falling),
                    scalar, arr.shape)

    return f


def sine(freq: float, amp: float = 1.0, phase: float = 0.0,
         offset: float = 0.0) -> TimeFunc:
    """A continuous sinusoid ``offset + amp * sin(2*pi*freq*t + phase)``.

    Parameters
    ----------
    freq
        Frequency [Hz].  ``0`` gives the constant ``offset + amp*sin(phase)``.
    amp
        Amplitude [V] (or [A]), zero-to-peak.
    phase
        Phase at ``t = 0`` [rad].  Use ``phase = pi/2`` for a cosine.
    offset
        DC offset [V] (or [A]).

    Returns
    -------
    callable
        ``f(t)``, vectorised over ``t``.

    Notes
    -----
    Unlike SPICE's ``SIN`` source this does not switch on at a delay and does
    not damp; it exists for all ``t``, including ``t < 0``.  A steady-state AC
    answer is far cheaper from :class:`~fieldspice.solvers.ac.ACSolver` than
    from a transient run of this, which has to ring down its turn-on transient
    first.
    """
    freq = _require_finite("freq", freq)
    amp = _require_finite("amp", amp)
    phase = _require_finite("phase", phase)
    offset = _require_finite("offset", offset)
    omega = 2.0 * np.pi * freq

    def f(t: float | np.ndarray) -> float | np.ndarray:
        arr, scalar = _as_time(t)
        return _out(offset + amp * np.sin(omega * arr + phase),
                    scalar, arr.shape)

    return f


def pwl(times: Sequence[float] | np.ndarray,
        values: Sequence[float] | np.ndarray) -> TimeFunc:
    """Piecewise-linear waveform through ``(times, values)``.

    Parameters
    ----------
    times
        Breakpoint times [s].  Need not be sorted; they are sorted internally
        (stably, so the relative order of equal times is preserved).
    values
        Breakpoint amplitudes [V] (or [A]), same length as ``times``.

    Returns
    -------
    callable
        ``f(t)``, vectorised over ``t``.

    Raises
    ------
    ValueError
        If the two sequences have different lengths, if either is empty, or if
        either contains a non-finite entry.

    Notes
    -----
    Outside ``[min(times), max(times)]`` the waveform **holds** the nearest
    endpoint value rather than extrapolating.  Extrapolating a linear segment
    off the end of a measured waveform is how a source ends up at 10 kV in step
    9000, so it is not offered.

    Two identical times express a deliberate instantaneous jump (the usual
    SPICE idiom); the value exactly at that instant is the later of the two.
    Everything :func:`step` says about ideal edges applies to that jump too.
    """
    tt = np.asarray(times, dtype=float).ravel()
    vv = np.asarray(values, dtype=float).ravel()
    if tt.size != vv.size:
        raise ValueError(
            f"times and values must have the same length, got {tt.size} and {vv.size}")
    if tt.size == 0:
        raise ValueError("pwl needs at least one breakpoint")
    if not np.all(np.isfinite(tt)):
        raise ValueError("times must all be finite")
    if not np.all(np.isfinite(vv)):
        raise ValueError("values must all be finite")

    order = np.argsort(tt, kind="stable")
    tt, vv = tt[order], vv[order]

    def f(t: float | np.ndarray) -> float | np.ndarray:
        arr, scalar = _as_time(t)
        return _out(np.interp(arr, tt, vv), scalar, arr.shape)

    return f


def gaussian(t0: float, tau: float, amp: float = 1.0) -> TimeFunc:
    """Gaussian pulse ``amp * exp(-0.5 * ((t - t0)/tau)^2)``.

    Parameters
    ----------
    t0
        Time of the peak [s].
    tau
        Standard deviation of the pulse in time [s].  The full width at half
        maximum is ``2*sqrt(2*ln 2)*tau = 2.3548*tau``.
    amp
        Peak amplitude [V] (or [A]).

    Returns
    -------
    callable
        ``f(t)``, vectorised over ``t``.

    Raises
    ------
    ValueError
        If ``tau <= 0``.

    Notes
    -----
    The Fourier transform is Gaussian with standard deviation
    ``sigma_f = 1/(2*pi*tau)`` in Hz, so the spectrum is down by ``exp(-2)``
    (-17.4 dB) at ``f = 1/(pi*tau)`` and is negligible above ``f ~ 1/tau``.
    That bounded, exactly-known bandwidth is why this is the standard probe for
    broadband extraction: start the run at ``t0 - 5*tau`` (amplitude 3.7e-6 of
    peak, an acceptable discontinuity) and divide output by input spectrum.

    Unlike :func:`step` there is no discontinuity anywhere, so this waveform
    never aliases provided ``dt <= tau/5``.
    """
    t0 = _require_finite("t0", t0)
    tau = _require_positive("tau", tau)
    amp = _require_finite("amp", amp)

    def f(t: float | np.ndarray) -> float | np.ndarray:
        arr, scalar = _as_time(t)
        u = (arr - t0) / tau
        return _out(amp * np.exp(-0.5 * u * u), scalar, arr.shape)

    return f


def ramp(slope: float, t0: float = 0.0) -> TimeFunc:
    """A linear ramp starting at ``t0``: 0 before, ``slope*(t - t0)`` after.

    Parameters
    ----------
    slope
        Rate of change [V/s] (or [A/s]).
    t0
        Time at which the ramp starts [s].

    Returns
    -------
    callable
        ``f(t)``, vectorised over ``t``.

    Notes
    -----
    Unbounded in amplitude by construction.  Its usual job is a slow bias sweep
    for an IV curve, where the ramp must be slow enough that the device is
    quasi-static: ``slope * C_total << I`` of interest, or the measured current
    is displacement current from the sweep itself.  This is a real experimental
    artifact, not just a simulation one.
    """
    slope = _require_finite("slope", slope)
    t0 = _require_finite("t0", t0)

    def f(t: float | np.ndarray) -> float | np.ndarray:
        arr, scalar = _as_time(t)
        return _out(slope * np.maximum(arr - t0, 0.0), scalar, arr.shape)

    return f


# ==========================================================================
# Digital patterns
# ==========================================================================
def prbs_sequence(order: int = 7, n: int | None = None,
                  seed: int = 0) -> np.ndarray:
    """The first ``n`` bits of a maximal-length PRBS as ``uint8`` 0/1.

    Parameters
    ----------
    order
        LFSR order; a key of :data:`PRBS_TAPS`.  Dimensionless.
    n
        Number of bits to generate.  Defaults to one full period,
        ``2**order - 1``, which is refused above 24 bits of order because a
        full period no longer fits in memory.
    seed
        Initial LFSR state, dimensionless.  Any nonzero value works; ``0`` is
        mapped to the conventional all-ones state because the all-zero state is
        the LFSR's fixed point and would output nothing but zeros.

    Returns
    -------
    np.ndarray
        Shape ``(n,)``, dtype ``uint8``, values 0 or 1.

    Raises
    ------
    ValueError
        If ``order`` is unsupported, ``n`` is negative, or ``n`` exceeds the
        cache limit.

    Notes
    -----
    Because the sequence is maximal length, every nonzero seed lands on the
    *same* cycle: the seed selects a phase, never a different pattern.  Two runs
    with the same seed are bit-identical, which is what makes a BER or eye
    comparison between two solver configurations meaningful.
    """
    if order not in PRBS_TAPS:
        raise ValueError(
            f"unsupported PRBS order {order!r}; supported: {sorted(PRBS_TAPS)}")
    period = (1 << order) - 1
    if n is None:
        n = period
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n > _PRBS_MAX_CACHE:
        raise ValueError(
            f"refusing to materialise {n} PRBS bits; limit is {_PRBS_MAX_CACHE}")

    nn, kk = PRBS_TAPS[order]
    mask = period  # 2**order - 1 is also the all-ones bit mask
    state = (int(seed) & mask) or mask

    bits = np.empty(n, dtype=np.uint8)
    hi, lo = nn - 1, kk - 1
    for i in range(n):
        fb = ((state >> hi) ^ (state >> lo)) & 1
        state = ((state << 1) | fb) & mask
        bits[i] = fb
    return bits


def prbs(bit_period: float, order: int = 7, v0: float = 0.0, v1: float = 1.0,
         seed: int = 0, trise: float | None = None,
         shape: str = "linear") -> TimeFunc:
    """Non-return-to-zero pseudo-random bit stream from a maximal-length LFSR.

    Bit ``k`` occupies ``[k*bit_period, (k+1)*bit_period)``; the transition into
    it happens at the *start* of its interval, so the eye is centred at
    ``(k + 0.5)*bit_period``.  The pattern repeats every ``2**order - 1`` bits
    and is extended periodically to negative time.

    Parameters
    ----------
    bit_period
        Unit interval [s].  The bit rate is ``1/bit_period``.
    order
        LFSR order; a key of :data:`PRBS_TAPS` (7, 9, 11, 15, 23, 31).
    v0, v1
        Levels for a 0 and a 1 bit [V] (or [A]).
    seed
        LFSR seed, dimensionless.  ``0`` means the all-ones state.  See
        :func:`prbs_sequence`.
    trise
        Edge duration, 0% to 100% [s], used for both edges.  ``None`` selects
        ``0.2 * bit_period``, a 20% edge, which is a realistic bandwidth-limited
        driver and keeps the eye open.  Must not exceed ``bit_period``.
    shape
        Edge profile, one of :data:`EDGE_SHAPES`.

    Returns
    -------
    callable
        ``f(t)``, vectorised over ``t``.

    Raises
    ------
    ValueError
        If ``bit_period <= 0``, ``order`` is unsupported, ``trise`` is negative
        or longer than a bit, or an evaluation lands past bit
        ``2**24`` of an order-31 pattern.

    Notes
    -----
    A PRBS is the right stimulus for interconnect because it contains every
    ``order``-bit sub-pattern exactly once per period, so it exercises the
    worst-case inter-symbol interference (the long run of identical bits
    followed by a lone transition) without anyone having to construct it by
    hand.  Its power spectral density is ``sinc^2(f*bit_period)`` sampled on a
    ``1/(period*bit_period)`` comb, i.e. essentially flat to the first null at
    ``1/bit_period``.

    Order matters: PRBS7 has a longest run of 7 identical bits, PRBS31 of 31.
    If the answer depends on baseline wander or on a low-frequency pole, a short
    pattern will hide it.
    """
    bit_period = _require_positive("bit_period", bit_period)
    if order not in PRBS_TAPS:
        raise ValueError(
            f"unsupported PRBS order {order!r}; supported: {sorted(PRBS_TAPS)}")
    v0 = _require_finite("v0", v0)
    v1 = _require_finite("v1", v1)
    shape = _require_shape(shape)
    trise = 0.2 * bit_period if trise is None else _require_nonneg("trise", trise)
    if trise > bit_period:
        raise ValueError(
            f"trise {trise!r} s exceeds bit_period {bit_period!r} s; "
            "an edge longer than a unit interval has no meaning")

    period_bits = (1 << order) - 1
    nn, kk = PRBS_TAPS[order]
    mask = period_bits
    hi, lo = nn - 1, kk - 1

    # Lazily grown bit cache.  Orders up to 23 fit whole; order 31 does not, so
    # bits are produced only as far as the caller actually evaluates.
    cache: dict[str, object] = {
        "bits": np.empty(0, dtype=np.uint8),
        "state": (int(seed) & mask) or mask,
    }

    def _ensure(n_needed: int) -> np.ndarray:
        bits: np.ndarray = cache["bits"]  # type: ignore[assignment]
        if n_needed <= bits.size:
            return bits
        if n_needed > _PRBS_MAX_CACHE:
            raise ValueError(
                f"prbs evaluation needs bit {n_needed - 1} of the pattern, past "
                f"the {_PRBS_MAX_CACHE}-bit generation limit "
                f"({_PRBS_MAX_CACHE * bit_period:.4g} s of pattern)")
        target = min(max(n_needed, 2 * bits.size, 4096),
                     period_bits, _PRBS_MAX_CACHE)
        target = max(target, n_needed)
        grown = np.empty(target, dtype=np.uint8)
        grown[:bits.size] = bits
        state = int(cache["state"])  # type: ignore[arg-type]
        for i in range(bits.size, target):
            fb = ((state >> hi) ^ (state >> lo)) & 1
            state = ((state << 1) | fb) & mask
            grown[i] = fb
        cache["bits"] = grown
        cache["state"] = state
        return grown

    def f(t: float | np.ndarray) -> float | np.ndarray:
        arr, scalar = _as_time(t)
        u = arr / bit_period
        k = np.floor(u)
        frac = (u - k) * bit_period                     # time into the bit [s]
        idx = np.mod(k, period_bits).astype(np.int64)   # wraps negative t too
        prev = np.mod(k - 1.0, period_bits).astype(np.int64)
        bits = _ensure(int(max(idx.max(initial=0), prev.max(initial=0))) + 1)
        b_now = np.where(bits[idx] > 0, v1, v0)
        b_prev = np.where(bits[prev] > 0, v1, v0)
        out = b_prev + (b_now - b_prev) * _ramp01(frac, trise, shape)
        return _out(out, scalar, arr.shape)

    return f


def trapezoid_clock(period: float, duty: float = 0.5,
                    trise: float | None = None, tfall: float | None = None,
                    v0: float = 0.0, v1: float = 1.0,
                    shape: str = "linear") -> TimeFunc:
    """Periodic trapezoidal clock with finite edges and a specified duty cycle.

    One period, starting at ``t = 0``: rise over ``trise``, high for
    ``duty*period - (trise + tfall)/2``, fall over ``tfall``, low for the rest.

    Parameters
    ----------
    period
        Clock period [s].  Frequency is ``1/period``.
    duty
        Duty cycle in [0, 1], **measured between the 50% crossings** of the
        rising and falling edges, which is the convention every datasheet uses.
        Defining it this way makes the mean value exactly
        ``duty*v1 + (1 - duty)*v0`` regardless of the edge times, so changing
        the edge rate does not move the DC operating point of whatever this
        drives.
    trise, tfall
        Edge durations, 0% to 100% [s].  ``None`` selects
        ``0.1 * min(duty, 1 - duty) * period``, which is a fast but always
        feasible edge; ``tfall`` further defaults to whatever ``trise`` ended up
        being.
    v0, v1
        Low and high levels [V] (or [A]).
    shape
        Edge profile, one of :data:`EDGE_SHAPES`.  ``"linear"`` preserves the
        exact mean above; so do the other two, because every profile in
        :data:`EDGE_SHAPES` is antisymmetric about the midpoint of its edge.

    Returns
    -------
    callable
        ``f(t)``, vectorised over ``t``, periodic in both directions.

    Raises
    ------
    ValueError
        If ``period <= 0``, ``duty`` is outside [0, 1], an edge time is
        negative, or the edges do not fit:
        ``(trise + tfall)/2 <= min(duty, 1 - duty)*period`` is required, since
        otherwise the flat top or bottom would have negative length.
    """
    period = _require_positive("period", period)
    duty = _require_finite("duty", duty)
    if not 0.0 <= duty <= 1.0:
        raise ValueError(f"duty must be in [0, 1], got {duty!r}")
    v0 = _require_finite("v0", v0)
    v1 = _require_finite("v1", v1)
    shape = _require_shape(shape)

    if trise is None:
        trise = 0.1 * min(duty, 1.0 - duty) * period
    trise = _require_nonneg("trise", trise)
    tfall = trise if tfall is None else _require_nonneg("tfall", tfall)

    tol = 1e-12 * period
    high_flat = duty * period - 0.5 * (trise + tfall)
    low_flat = (1.0 - duty) * period - 0.5 * (trise + tfall)
    if high_flat < -tol or low_flat < -tol:
        raise ValueError(
            f"edges do not fit: (trise + tfall)/2 = {0.5 * (trise + tfall):.6g} s "
            f"exceeds min(duty, 1-duty)*period = "
            f"{min(duty, 1.0 - duty) * period:.6g} s")
    high_flat = max(high_flat, 0.0)

    t_fall_start = trise + high_flat

    def f(t: float | np.ndarray) -> float | np.ndarray:
        arr, scalar = _as_time(t)
        tau = np.mod(arr, period)
        rising = v0 + (v1 - v0) * _ramp01(tau, trise, shape)
        falling = v1 + (v0 - v1) * _ramp01(tau - t_fall_start, tfall, shape)
        return _out(np.where(tau < t_fall_start, rising, falling),
                    scalar, arr.shape)

    return f


# ==========================================================================
# Composition
# ==========================================================================
class Waveform:
    """A callable ``f(t)`` with arithmetic, delay and composition.

    Wrapping is deliberate rather than automatic: the generators above return
    plain closures (per ``docs/CONTRACTS.md``), and this class adds algebra when
    it is wanted, without every solver having to import it.

    Parameters
    ----------
    func
        Any callable taking time [s] and returning an amplitude [V] or [A].
        It may be scalar-only; :meth:`__call__` broadcasts the result to the
        shape of ``t``, so a sloppy ``lambda t: 1.0`` still behaves.
    name
        Optional label used in :meth:`__repr__` only.

    Examples
    --------
    >>> w = Waveform(sine(1e9, amp=2.0)) + 0.5
    >>> round(w(0.25e-9), 12)
    2.5
    >>> round((2 * w).delay(0.25e-9)(0.5e-9), 12)
    5.0
    """

    __slots__ = ("func", "name")

    def __init__(self, func: Callable[..., float | np.ndarray],
                 name: str | None = None):
        if not callable(func):
            raise ValueError(f"Waveform needs a callable, got {type(func).__name__}")
        self.func = func
        self.name = name or getattr(func, "__name__", "waveform")

    # -- evaluation --------------------------------------------------------
    def __call__(self, t: float | np.ndarray) -> float | np.ndarray:
        """Evaluate at time ``t`` [s]; scalar in, float out, array in, array out."""
        arr, scalar = _as_time(t)
        return _out(np.asarray(self.func(t), dtype=float), scalar, arr.shape)

    def sampled(self, t: Sequence[float] | np.ndarray) -> np.ndarray:
        """Evaluate on a time axis ``t`` [s], always returning an ``ndarray``."""
        arr = np.asarray(t, dtype=float)
        return np.atleast_1d(np.asarray(self(arr), dtype=float))

    # -- transforms --------------------------------------------------------
    def delay(self, dt: float) -> "Waveform":
        """Shift later in time by ``dt`` [s]: the result is ``self(t - dt)``."""
        dt = _require_finite("dt", dt)
        return Waveform(lambda t: self.func(np.asarray(t, dtype=float) - dt),
                        f"{self.name}.delay({dt:g})")

    def compose(self, fn: Callable[[float | np.ndarray], float | np.ndarray]
                ) -> "Waveform":
        """Post-compose: return ``fn(self(t))``, e.g. a saturating driver."""
        if not callable(fn):
            raise ValueError("compose needs a callable")
        return Waveform(lambda t: fn(self(t)),
                        f"{getattr(fn, '__name__', 'fn')}({self.name})")

    def clip(self, lo: float, hi: float) -> "Waveform":
        """Clamp the amplitude to ``[lo, hi]`` [V] (or [A])."""
        lo = _require_finite("lo", lo)
        hi = _require_finite("hi", hi)
        if hi < lo:
            raise ValueError(f"clip needs lo <= hi, got {lo!r} and {hi!r}")
        return Waveform(lambda t: np.clip(self(t), lo, hi), f"clip({self.name})")

    # -- algebra -----------------------------------------------------------
    @staticmethod
    def _operand(other: object) -> Callable[..., float | np.ndarray] | None:
        """Coerce an operand to a callable, or ``None`` if it is not numeric."""
        if isinstance(other, Waveform):
            return other.func
        if callable(other):
            return other
        if isinstance(other, (int, float, np.integer, np.floating)):
            value = float(other)
            return lambda t: value
        return None

    def __add__(self, other: object) -> "Waveform":
        g = self._operand(other)
        if g is None:
            return NotImplemented
        return Waveform(lambda t: np.add(self(t), g(t)), f"({self.name}+)")

    __radd__ = __add__

    def __sub__(self, other: object) -> "Waveform":
        g = self._operand(other)
        if g is None:
            return NotImplemented
        return Waveform(lambda t: np.subtract(self(t), g(t)), f"({self.name}-)")

    def __rsub__(self, other: object) -> "Waveform":
        g = self._operand(other)
        if g is None:
            return NotImplemented
        return Waveform(lambda t: np.subtract(g(t), self(t)), f"(-{self.name})")

    def __mul__(self, other: object) -> "Waveform":
        g = self._operand(other)
        if g is None:
            return NotImplemented
        return Waveform(lambda t: np.multiply(self(t), g(t)), f"({self.name}*)")

    __rmul__ = __mul__

    def __truediv__(self, other: object) -> "Waveform":
        g = self._operand(other)
        if g is None:
            return NotImplemented
        return Waveform(lambda t: np.divide(self(t), g(t)), f"({self.name}/)")

    def __neg__(self) -> "Waveform":
        return Waveform(lambda t: -np.asarray(self(t), dtype=float),
                        f"(-{self.name})")

    def __pos__(self) -> "Waveform":
        return self

    # -- constructors ------------------------------------------------------
    @classmethod
    def constant(cls, value: float) -> "Waveform":
        """A time-independent level [V] (or [A])."""
        v = _require_finite("value", value)

        def f(t: float | np.ndarray) -> float | np.ndarray:
            arr, scalar = _as_time(t)
            return _out(np.full(arr.shape, v), scalar, arr.shape)

        return cls(f, f"const({v:g})")

    @classmethod
    def wrap(cls, obj: "Waveform | Callable[..., float] | float") -> "Waveform":
        """Accept a :class:`Waveform`, a bare callable, or a number."""
        if isinstance(obj, Waveform):
            return obj
        if callable(obj):
            return cls(obj)
        return cls.constant(float(obj))  # type: ignore[arg-type]

    def __repr__(self) -> str:
        return f"<Waveform {self.name}>"
