"""Persistence and interchange: result files, grid files, VTK, Touchstone.

Four jobs, three file families:

1. **Native round-trip** (:func:`save_result` / :func:`load_result`,
   :func:`save_grid` / :func:`load_grid`).  A :class:`~fieldspice.solvers.base.Result`
   is a graph of NumPy arrays plus free-form provenance, and it must come back
   *bit-identical* --- a result you cannot reload is a result you cannot audit.
   HDF5 is used when ``h5py`` is importable and a compressed ``.npz`` container
   otherwise.  The fallback is silent (it never blocks a run) but never
   invisible: the backend is written into the file itself and reappears as
   ``Result.meta["io_backend"]`` on load, and :func:`load_result` sniffs the
   file's magic bytes rather than trusting its extension, so a ``.h5`` file that
   actually holds ``.npz`` bytes still loads.

2. **Visualisation** (:func:`export_vtk`).  Legacy ``.vtk`` and XML ``.vtr``
   rectilinear grids for ParaView / VisIt.

3. **RF interchange** (:func:`to_touchstone`).  ``.sNp`` files for ADS, AWR,
   scikit-rf, and every VNA made since 1990.

Units
-----
This module never rescales anything.  Grid coordinates are metres, times are
seconds, frequencies written to Touchstone are converted from hertz into the
requested header unit and nothing else.

Byte order and index order
--------------------------
Two conventions collide here and both are silent when you get them wrong:

* VTK stores structured points with **x varying fastest** (Fortran order) while
  every fieldspice array is C-ordered with z fastest.  Every array handed to
  VTK therefore goes through :func:`_vtk_flat`, which ravels in ``order="F"``.
* Legacy VTK binary is **big-endian by definition**, XML VTK carries an
  explicit ``byte_order`` attribute.  Both are written explicitly here rather
  than inheriting the host's native order.

Optional dependencies
---------------------
``h5py`` only, imported inside the function that needs it.  Importing this
module pulls in nothing beyond NumPy and the standard library.
"""

from __future__ import annotations

import base64
import dataclasses
import importlib
import json
import os
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .grid import RectilinearGrid
from .solvers.base import Result

__all__ = [
    "save_result", "load_result", "save_grid", "load_grid",
    "export_vtk", "to_touchstone", "h5py_available",
]

# --------------------------------------------------------------------------
# Container constants
# --------------------------------------------------------------------------
_FORMAT = "fieldspice"
_SCHEMA = 1

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
_ZIP_MAGIC = b"PK\x03\x04"

_HDF5_SUFFIXES = frozenset({".h5", ".hdf5", ".he5", ".hdf"})
_NPZ_SUFFIXES = frozenset({".npz"})

_META_KEY = "__meta__"
_FORMAT_KEY = "__format__"

# Stand-in for an empty field/terminal name; see _esc.
_EMPTY_NAME = "%."

# Numeric kinds that both backends can store without an encoder.
_NUMERIC_KINDS = frozenset("biufc")

# Modules a loaded file is allowed to import when rebuilding a type reference.
# Without this whitelist, opening a result file would be arbitrary-import.
_SAFE_MODULES = frozenset({"builtins", "numpy", "fieldspice"})

# Single-key dicts using one of these names are ambiguous with an encoded
# value, so a user dict that happens to look like one gets wrapped explicitly.
_RESERVED = frozenset({
    "__tuple__", "__set__", "__frozenset__", "__complex__", "__bytes__",
    "__ndarray__", "__npscalar__", "__dict__", "__type__", "__dataclass__",
    "__repr__",
})

# Keys that load_* injects into Result.meta.  Documented so that a strict
# round-trip test knows exactly what to ignore.
_INJECTED_META = ("io_backend", "io_format_version", "io_path",
                  "io_degraded_keys")


# ==========================================================================
# Optional backend detection
# ==========================================================================
def h5py_available() -> bool:
    """Whether the HDF5 backend can be used.

    Returns
    -------
    bool
        ``True`` if ``h5py`` imports.  Never raises; a broken or partially
        installed h5py counts as unavailable, because a save must not fail on
        an optional dependency.
    """
    return _import_h5py() is not None


def _import_h5py():
    """Import h5py, or return ``None``.  Imported here, not at module scope."""
    try:
        import h5py  # noqa: PLC0415  (deliberately lazy: optional dependency)
    except Exception:  # pragma: no cover - depends on the host environment
        return None
    return h5py


# ==========================================================================
# Name escaping
# ==========================================================================
def _esc(name: str) -> str:
    """Percent-encode a user name so it is a legal HDF5/zip member name.

    Field, scalar and terminal names are user data and may contain ``/``,
    which would silently create a nested HDF5 group and a nested zip path.
    Escaping keeps the stored key a single, unambiguous component.

    The empty name gets its own token because HDF5 rejects an empty path
    component outright, and ``_esc`` never otherwise emits ``%`` followed by a
    non-hex character, so the token cannot collide with a real escape.
    """
    if not isinstance(name, str):
        raise ValueError(f"names must be str, got {type(name).__name__}")
    if name == "":
        return _EMPTY_NAME
    out = []
    for ch in name:
        if ch.isalnum() and ch.isascii() or ch in "._-":
            out.append(ch)
        else:
            out.extend(f"%{b:02X}" for b in ch.encode("utf-8"))
    return "".join(out)


def _unesc(name: str) -> str:
    """Inverse of :func:`_esc`."""
    if name == _EMPTY_NAME:
        return ""
    raw = bytearray()
    i = 0
    n = len(name)
    while i < n:
        if (name[i] == "%" and i + 2 < n
                and all(c in "0123456789abcdefABCDEF" for c in name[i + 1:i + 3])):
            raw.append(int(name[i + 1:i + 3], 16))
            i += 3
        else:
            raw.extend(name[i].encode("utf-8"))
            i += 1
    return raw.decode("utf-8")


def _as_path(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise ValueError(f"path must be str or PathLike, got {type(path).__name__}")
    return Path(os.fspath(path))


# ==========================================================================
# Array validation
# ==========================================================================
def _check_array(value: Any, where: str) -> np.ndarray:
    """Validate that ``value`` is a storable numeric array.

    Raises
    ------
    ValueError
        If the value is not array-like or is not numeric.  Strings, callables
        and Python objects belong in ``meta``, which has a real encoder; the
        array containers do not silently accept them.
    """
    arr = np.asarray(value)
    if arr.dtype.kind not in _NUMERIC_KINDS:
        raise ValueError(
            f"{where}: arrays must be numeric (bool/int/float/complex), "
            f"got dtype {arr.dtype!r}. Put non-numeric data in Result.meta.")
    return arr


# ==========================================================================
# meta encoding / decoding
# ==========================================================================
def _encode_meta(obj: Any, arrays: dict[str, np.ndarray],
                 keypath: str, degraded: list[str]) -> Any:
    """Convert one meta value into a JSON-safe form.

    Large arrays are *not* base64'd into the JSON; they are added to ``arrays``
    under ``meta_arrays/<n>`` and replaced by a reference, so a field snapshot
    parked in ``meta`` costs the same as one parked in ``fields``.
    """
    if obj is None:
        return None
    # NumPy types are tested first on purpose: np.float64 subclasses float and
    # np.complex128 subclasses complex, so a later test would silently demote
    # them to Python scalars and break the round-trip of dtype.
    if isinstance(obj, np.ndarray):
        key = f"meta_arrays/{len(arrays)}"
        arrays[key] = _check_array(obj, f"meta[{keypath}]")
        return {"__ndarray__": key}
    if isinstance(obj, np.generic):
        return {"__npscalar__": {"dtype": obj.dtype.str,
                                 "b64": base64.b64encode(obj.tobytes()
                                                         ).decode("ascii")}}
    if isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, float)):
        return obj
    if isinstance(obj, complex):
        return {"__complex__": [obj.real, obj.imag]}
    if isinstance(obj, (bytes, bytearray)):
        return {"__bytes__": base64.b64encode(bytes(obj)).decode("ascii")}
    if isinstance(obj, tuple):
        return {"__tuple__": [_encode_meta(v, arrays, f"{keypath}[{i}]", degraded)
                              for i, v in enumerate(obj)]}
    if isinstance(obj, (set, frozenset)):
        tag = "__frozenset__" if isinstance(obj, frozenset) else "__set__"
        items = sorted(obj, key=repr)
        return {tag: [_encode_meta(v, arrays, f"{keypath}{{{i}}}", degraded)
                      for i, v in enumerate(items)]}
    if isinstance(obj, list):
        return [_encode_meta(v, arrays, f"{keypath}[{i}]", degraded)
                for i, v in enumerate(obj)]
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj):
            enc = {k: _encode_meta(v, arrays, f"{keypath}.{k}", degraded)
                   for k, v in obj.items()}
            # A user dict of exactly one reserved-looking key would decode as an
            # encoded value, so it is wrapped explicitly instead.
            if not (len(enc) == 1 and next(iter(enc)) in _RESERVED):
                return enc
            return {"__dict__": [[k, v] for k, v in enc.items()]}
        return {"__dict__": [
            [_encode_meta(k, arrays, f"{keypath}.<key>", degraded),
             _encode_meta(v, arrays, f"{keypath}.{k!r}", degraded)]
            for k, v in obj.items()]}
    if isinstance(obj, type):
        return {"__type__": f"{obj.__module__}:{obj.__qualname__}"}
    if dataclasses.is_dataclass(obj):
        cls = type(obj)
        return {"__dataclass__": {
            "type": f"{cls.__module__}:{cls.__qualname__}",
            "fields": {f.name: _encode_meta(getattr(obj, f.name), arrays,
                                            f"{keypath}.{f.name}", degraded)
                       for f in dataclasses.fields(obj)}}}
    degraded.append(keypath)
    return {"__repr__": repr(obj)}


def _decode_meta(obj: Any, arrays: Mapping[str, np.ndarray]) -> Any:
    if isinstance(obj, list):
        return [_decode_meta(v, arrays) for v in obj]
    if not isinstance(obj, dict):
        return obj
    if len(obj) == 1:
        (tag, payload), = obj.items()
        if tag == "__complex__":
            return complex(payload[0], payload[1])
        if tag == "__bytes__":
            return base64.b64decode(payload.encode("ascii"))
        if tag == "__ndarray__":
            return arrays[payload]
        if tag == "__npscalar__":
            buf = base64.b64decode(payload["b64"].encode("ascii"))
            return np.frombuffer(buf, dtype=np.dtype(payload["dtype"]))[0]
        if tag == "__tuple__":
            return tuple(_decode_meta(v, arrays) for v in payload)
        if tag == "__set__":
            return set(_decode_meta(v, arrays) for v in payload)
        if tag == "__frozenset__":
            return frozenset(_decode_meta(v, arrays) for v in payload)
        if tag == "__dict__":
            return {_decode_meta(k, arrays): _decode_meta(v, arrays)
                    for k, v in payload}
        if tag == "__type__":
            resolved = _resolve(payload)
            return resolved if resolved is not None else payload
        if tag == "__repr__":
            return payload
        if tag == "__dataclass__":
            fields = {k: _decode_meta(v, arrays)
                      for k, v in payload["fields"].items()}
            cls = _resolve(payload["type"])
            if cls is not None and dataclasses.is_dataclass(cls):
                try:
                    return cls(**fields)
                except TypeError:
                    return fields
            return fields
    return {k: _decode_meta(v, arrays) for k, v in obj.items()}


def _resolve(spec: str) -> Any:
    """Resolve ``"module:Qual.Name"`` to an object, or ``None``.

    Restricted to :data:`_SAFE_MODULES`: loading a data file must never be able
    to import an arbitrary module named inside that file.
    """
    mod_name, _, qual = spec.partition(":")
    if mod_name.split(".")[0] not in _SAFE_MODULES:
        return None
    try:
        obj: Any = importlib.import_module(mod_name)
    except Exception:
        return None
    for part in qual.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# ==========================================================================
# Low-level container write / read
# ==========================================================================
def _select_backend(path: Path, backend: str) -> str:
    if backend not in ("auto", "hdf5", "npz"):
        raise ValueError(
            f"backend must be 'auto', 'hdf5' or 'npz', got {backend!r}")
    if backend == "npz":
        return "npz"
    have = h5py_available()
    if backend == "hdf5":
        if not have:
            raise ValueError("backend='hdf5' requested but h5py is not installed")
        return "hdf5"
    suffix = path.suffix.lower()
    if suffix in _NPZ_SUFFIXES:
        return "npz"
    # Everything else prefers HDF5 and degrades quietly to npz.  The bytes,
    # not the extension, decide what load_result does later.
    return "hdf5" if have else "npz"


def _write_container(path: Path, kind: str, arrays: Mapping[str, np.ndarray],
                     meta_doc: dict[str, Any], backend: str,
                     compress: bool) -> None:
    meta_doc = dict(meta_doc, format=_FORMAT, schema=_SCHEMA, kind=kind,
                    backend=backend)
    blob = json.dumps(meta_doc, allow_nan=True)
    if backend == "hdf5":
        h5py = _import_h5py()
        with h5py.File(path, "w") as f:
            f.attrs["format"] = _FORMAT
            f.attrs["schema"] = _SCHEMA
            f.attrs["kind"] = kind
            f.attrs["backend"] = "hdf5"
            f.create_dataset(_META_KEY, data=blob,
                             dtype=h5py.string_dtype("utf-8"))
            for key, arr in arrays.items():
                if compress and arr.size > 1024:
                    f.create_dataset(key, data=arr, compression="gzip",
                                     compression_opts=4)
                else:
                    f.create_dataset(key, data=arr)
        return
    payload = {_META_KEY: _utf8_array(blob),
               _FORMAT_KEY: _utf8_array(f"{_FORMAT}/{_SCHEMA}/{kind}/npz")}
    payload.update(arrays)
    # np.savez appends '.npz' to a *filename*; a file object writes exactly
    # where the caller asked, which matters when h5py was unavailable and the
    # caller passed a '.h5' path.
    saver = np.savez_compressed if compress else np.savez
    with open(path, "wb") as fh:
        saver(fh, **payload)


def _utf8_array(text: str) -> np.ndarray:
    return np.frombuffer(text.encode("utf-8"), dtype=np.uint8)


def _read_container(path: Path) -> tuple[str, dict[str, np.ndarray],
                                         dict[str, Any], str]:
    """Return ``(kind, arrays, meta_doc, backend)`` from a fieldspice file."""
    if not path.exists():
        raise FileNotFoundError(str(path))
    with open(path, "rb") as fh:
        head = fh.read(8)
    if head.startswith(_HDF5_MAGIC):
        h5py = _import_h5py()
        if h5py is None:
            raise ValueError(
                f"{path} is an HDF5 file but h5py is not installed; "
                "install h5py or re-save with backend='npz'")
        arrays: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            if _META_KEY not in f:
                raise ValueError(f"{path} is not a fieldspice file "
                                 f"(no {_META_KEY} dataset)")
            raw = f[_META_KEY][()]
            blob = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)

            def _visit(name, obj):
                if isinstance(obj, h5py.Dataset) and name != _META_KEY:
                    arrays[name] = np.asarray(obj[()])

            f.visititems(_visit)
        backend = "hdf5"
    elif head.startswith(_ZIP_MAGIC):
        with open(path, "rb") as fh:
            with np.load(fh, allow_pickle=False) as z:
                if _META_KEY not in z.files:
                    raise ValueError(f"{path} is a .npz archive but not a "
                                     f"fieldspice file (no {_META_KEY})")
                blob = bytes(z[_META_KEY].tobytes()).decode("utf-8")
                arrays = {k: np.asarray(z[k]) for k in z.files
                          if k not in (_META_KEY, _FORMAT_KEY)}
        backend = "npz"
    else:
        raise ValueError(f"{path} is neither HDF5 nor .npz "
                         f"(magic {head[:4]!r}); not a fieldspice file")
    meta_doc = json.loads(blob)
    if meta_doc.get("format") != _FORMAT:
        raise ValueError(f"{path} is not a fieldspice file")
    if int(meta_doc.get("schema", 0)) > _SCHEMA:
        raise ValueError(f"{path} was written by a newer fieldspice "
                         f"(schema {meta_doc.get('schema')} > {_SCHEMA})")
    return str(meta_doc.get("kind", "")), arrays, meta_doc, backend


# ==========================================================================
# Grid serialisation
# ==========================================================================
def _grid_arrays(grid: RectilinearGrid) -> dict[str, np.ndarray]:
    """Node coordinates plus the collapsed-direction thicknesses.

    A grid is fully determined by its three node-coordinate arrays: an axis is
    flagged collapsed exactly when it has two nodes, so ``thickness`` is only
    ever consulted for an axis given as ``None``.  It is stored anyway (as the
    span of a collapsed axis, 1 m otherwise) so the file records the geometry
    the user asked for, and reconstruction passes all three arrays explicitly,
    which makes the round-trip exact regardless.
    """
    if not isinstance(grid, RectilinearGrid):
        raise ValueError(f"expected a RectilinearGrid, got {type(grid).__name__}")
    thick = []
    for axis, nodes in ((grid.ax, grid.xn), (grid.ay, grid.yn), (grid.az, grid.zn)):
        thick.append(float(nodes[-1] - nodes[0]) if axis.collapsed else 1.0)
    return {"grid/x": np.asarray(grid.xn, dtype=float),
            "grid/y": np.asarray(grid.yn, dtype=float),
            "grid/z": np.asarray(grid.zn, dtype=float),
            "grid/thickness": np.asarray(thick, dtype=float)}


def _grid_from_arrays(arrays: Mapping[str, np.ndarray]) -> RectilinearGrid:
    missing = [k for k in ("grid/x", "grid/y", "grid/z") if k not in arrays]
    if missing:
        raise ValueError(f"file has no grid (missing {missing})")
    thick = arrays.get("grid/thickness", np.ones(3))
    return RectilinearGrid(np.asarray(arrays["grid/x"], dtype=float),
                           np.asarray(arrays["grid/y"], dtype=float),
                           np.asarray(arrays["grid/z"], dtype=float),
                           thickness=(float(thick[0]), float(thick[1]),
                                      float(thick[2])))


def save_grid(grid: RectilinearGrid, path: str | os.PathLike[str], *,
              backend: str = "auto", compress: bool = False) -> None:
    """Write a grid to disk.

    Parameters
    ----------
    grid
        The grid to store.  Node coordinates are in metres.
    path
        Destination.  A ``.npz`` suffix forces the npz backend; anything else
        uses HDF5 when available.  The path is used exactly as given.
    backend
        ``"auto"`` (default), ``"hdf5"`` or ``"npz"``.  ``"hdf5"`` raises
        ``ValueError`` if h5py is missing; ``"auto"`` falls back silently.
    compress
        Compress array payloads (gzip for HDF5, deflate for npz).  Off by
        default because coordinate arrays are tiny.

    Raises
    ------
    ValueError
        If ``grid`` is not a :class:`~fieldspice.grid.RectilinearGrid` or an
        unknown backend is requested.
    """
    p = _as_path(path)
    chosen = _select_backend(p, backend)
    _write_container(p, "grid", _grid_arrays(grid), {}, chosen, compress)


def load_grid(path: str | os.PathLike[str]) -> RectilinearGrid:
    """Read a grid written by :func:`save_grid` (or embedded in a result file).

    Parameters
    ----------
    path
        File written by :func:`save_grid` or :func:`save_result`.  The backend
        is detected from the file's magic bytes, not its extension.

    Returns
    -------
    RectilinearGrid
        Node coordinates in metres, identical to the saved grid to the last
        bit.
    """
    p = _as_path(path)
    kind, arrays, _doc, _backend = _read_container(p)
    if kind not in ("grid", "result"):
        raise ValueError(f"{p} holds a {kind!r}, not a grid")
    return _grid_from_arrays(arrays)


# ==========================================================================
# Result serialisation
# ==========================================================================
def save_result(result: Result, path: str | os.PathLike[str], *,
                backend: str = "auto", compress: bool = False) -> None:
    """Write a :class:`~fieldspice.solvers.base.Result` to disk losslessly.

    Stored: the grid (as node coordinates in metres), the time vector [s], the
    ``fields``, ``terminals`` and ``scalars`` array trees, and ``meta``.

    ``meta`` is JSON-encoded with reversible tags for the types JSON lacks
    (tuple, set, complex, bytes, numpy scalar, type object, dataclass) and with
    NumPy arrays spilled to real datasets rather than base64.  A value that no
    encoder handles (a lambda, a live solver object) is stored as its ``repr``,
    its key is recorded in the file, and :func:`load_result` reports the list
    in ``meta["io_degraded_keys"]`` and warns.  Nothing is dropped in silence.

    Parameters
    ----------
    result
        The result to store.  Every array in ``fields``, ``scalars`` and
        ``terminals`` must be numeric (bool, int, float or complex).
    path
        Destination.  Used verbatim; no suffix is appended, so a ``.h5`` name
        keeps its name even when the npz backend had to be used.
    backend
        ``"auto"`` (default), ``"hdf5"`` or ``"npz"``.
    compress
        Compress array payloads.  Worth it for stored field histories, not for
        terminal traces.

    Raises
    ------
    ValueError
        If ``result`` is not a ``Result``, if any stored array is non-numeric,
        or if ``backend='hdf5'`` is requested without h5py.

    Notes
    -----
    Loading injects the provenance keys listed in :data:`_INJECTED_META` into
    ``meta``: ``io_backend``, ``io_format_version``, ``io_path`` and (only when
    something degraded) ``io_degraded_keys``.  Everything else compares equal,
    dtype and shape included.
    """
    if not isinstance(result, Result):
        raise ValueError(f"expected a Result, got {type(result).__name__}")
    p = _as_path(path)
    chosen = _select_backend(p, backend)

    arrays: dict[str, np.ndarray] = dict(_grid_arrays(result.grid))

    t = _check_array(result.t, "Result.t")
    if t.ndim > 1:
        raise ValueError(f"Result.t must be 1-D, got shape {t.shape}")
    arrays["t"] = t

    for group, mapping in (("fields", result.fields), ("scalars", result.scalars)):
        if not isinstance(mapping, Mapping):
            raise ValueError(f"Result.{group} must be a dict")
        for name, value in mapping.items():
            arrays[f"{group}/{_esc(name)}"] = _check_array(
                value, f"Result.{group}[{name!r}]")

    if not isinstance(result.terminals, Mapping):
        raise ValueError("Result.terminals must be a dict")
    terminal_keys: dict[str, list[str]] = {}
    for tname, sub in result.terminals.items():
        if not isinstance(sub, Mapping):
            raise ValueError(
                f"Result.terminals[{tname!r}] must be a dict of arrays, "
                f"got {type(sub).__name__}")
        terminal_keys[tname] = list(sub)
        for qty, value in sub.items():
            arrays[f"terminals/{_esc(tname)}/{_esc(qty)}"] = _check_array(
                value, f"Result.terminals[{tname!r}][{qty!r}]")

    degraded: list[str] = []
    meta_enc = _encode_meta(dict(result.meta), arrays, "meta", degraded)
    # Empty sub-dicts leave no dataset behind, so the key lists are what make
    # an empty terminal entry survive the round-trip.
    doc = {"meta": meta_enc, "degraded": degraded,
           "terminal_order": [[k, v] for k, v in terminal_keys.items()],
           "field_order": list(result.fields),
           "scalar_order": list(result.scalars)}
    _write_container(p, "result", arrays, doc, chosen, compress)


def load_result(path: str | os.PathLike[str]) -> Result:
    """Read a :class:`~fieldspice.solvers.base.Result` written by :func:`save_result`.

    Parameters
    ----------
    path
        File to read.  HDF5 and npz containers are distinguished by their magic
        bytes, so the extension is irrelevant.

    Returns
    -------
    Result
        Reconstructed result.  ``meta`` additionally carries ``io_backend``
        (``"hdf5"`` or ``"npz"``), ``io_format_version``, ``io_path``, and
        ``io_degraded_keys`` if any meta value could only be stored as a repr.

    Raises
    ------
    ValueError
        If the file is not a fieldspice result, or is an HDF5 file on a machine
        without h5py.
    """
    p = _as_path(path)
    kind, arrays, doc, backend = _read_container(p)
    if kind != "result":
        raise ValueError(f"{p} holds a {kind!r}, not a result")

    grid = _grid_from_arrays(arrays)
    t = arrays.get("t", np.zeros(0))

    fields: dict[str, np.ndarray] = {}
    scalars: dict[str, np.ndarray] = {}
    terminals: dict[str, dict[str, np.ndarray]] = {}
    for name in doc.get("field_order", []):
        fields[name] = arrays[f"fields/{_esc(name)}"]
    for name in doc.get("scalar_order", []):
        scalars[name] = arrays[f"scalars/{_esc(name)}"]
    for tname, qtys in doc.get("terminal_order", []):
        terminals[tname] = {q: arrays[f"terminals/{_esc(tname)}/{_esc(q)}"]
                            for q in qtys}

    meta = _decode_meta(doc.get("meta", {}), arrays)
    if not isinstance(meta, dict):
        raise ValueError(f"{p}: meta did not decode to a dict")
    degraded = list(doc.get("degraded", []))
    if degraded:
        warnings.warn(
            f"{p}: {len(degraded)} meta value(s) were stored as repr only: "
            + ", ".join(degraded[:8]), RuntimeWarning, stacklevel=2)
        meta["io_degraded_keys"] = degraded
    meta["io_backend"] = backend
    meta["io_format_version"] = int(doc.get("schema", _SCHEMA))
    meta["io_path"] = str(p)

    return Result(grid=grid, t=t, fields=fields, terminals=terminals,
                  scalars=scalars, meta=meta)


# ==========================================================================
# VTK export
# ==========================================================================
_XML_DTYPE = {np.dtype("float32"): "Float32", np.dtype("float64"): "Float64",
              np.dtype("int32"): "Int32", np.dtype("uint8"): "UInt8"}
_LEGACY_DTYPE = {np.dtype("float32"): "float", np.dtype("float64"): "double",
                 np.dtype("int32"): "int", np.dtype("uint8"): "unsigned_char"}


def _vtk_cast(arr: np.ndarray, name: str) -> np.ndarray:
    """Coerce to one of the four dtypes both VTK writers here support."""
    kind = arr.dtype.kind
    if kind == "b":
        return arr.astype(np.uint8)
    if kind in "iu":
        lo, hi = np.iinfo(np.int32).min, np.iinfo(np.int32).max
        if arr.size == 0 or (int(arr.min()) >= lo and int(arr.max()) <= hi):
            return arr.astype(np.int32)
        warnings.warn(f"field {name!r} exceeds int32; written as Float64",
                      RuntimeWarning, stacklevel=3)
        return arr.astype(np.float64)
    if kind == "f":
        return arr.astype(np.float32) if arr.dtype == np.float32 \
            else arr.astype(np.float64)
    raise ValueError(f"field {name!r}: dtype {arr.dtype!r} cannot be written "
                     "to VTK")


def _vtk_flat(arr: np.ndarray, ncomp: int) -> np.ndarray:
    """Flatten to VTK's ordering: components fastest, then x, then y, then z.

    This is the single most common VTK bug.  fieldspice arrays are C-ordered
    ``(i, j, k)`` with k fastest; VTK structured data is Fortran-ordered with i
    fastest and, for multi-component tuples, the components innermost.  Ravelling
    in ``order="F"`` after moving the component axis to the front produces
    exactly that sequence.
    """
    if ncomp == 1:
        return np.ravel(arr, order="F")
    return np.ravel(np.transpose(arr, (3, 0, 1, 2)), order="F")


def _classify_fields(grid: RectilinearGrid, fields: Mapping[str, Any]
                     ) -> tuple[list[tuple[str, np.ndarray, int]],
                                list[tuple[str, np.ndarray, int]]]:
    """Sort fields into (point, cell) lists of ``(name, array, ncomp)``.

    Accepts the shaped form ``(Nx+1, Ny+1, Nz+1)[, 3]`` / ``(Nx, Ny, Nz)[, 3]``
    and the flat solver form ``(n_nodes,)[, 3]`` / ``(n_cells,)[, 3]``.  The two
    are never ambiguous because a grid always has more nodes than cells.
    """
    if not isinstance(fields, Mapping):
        raise ValueError("fields must be a dict of name -> array")
    sn, sc = grid.shape_nodes, grid.shape_cells
    nn, nc = grid.n_nodes, grid.n_cells
    point: list[tuple[str, np.ndarray, int]] = []
    cell: list[tuple[str, np.ndarray, int]] = []
    for name, value in fields.items():
        if not isinstance(name, str):
            raise ValueError(f"field names must be str, got {type(name).__name__}")
        arr = np.asarray(value)
        if arr.dtype.kind == "c":
            # VTK has no complex type; split so both parts stay plottable.
            for suffix, part in (("_re", arr.real), ("_im", arr.imag)):
                p2, c2 = _classify_fields(grid, {name + suffix: part})
                point.extend(p2)
                cell.extend(c2)
            continue
        shape = arr.shape
        if shape == sn:
            point.append((name, arr, 1))
        elif shape == sn + (3,):
            point.append((name, arr, 3))
        elif shape == (nn,):
            point.append((name, arr.reshape(sn), 1))
        elif shape == (nn, 3):
            point.append((name, arr.reshape(sn + (3,)), 3))
        elif shape == sc:
            cell.append((name, arr, 1))
        elif shape == sc + (3,):
            cell.append((name, arr, 3))
        elif shape == (nc,):
            cell.append((name, arr.reshape(sc), 1))
        elif shape == (nc, 3):
            cell.append((name, arr.reshape(sc + (3,)), 3))
        else:
            raise ValueError(
                f"field {name!r} has shape {shape}, which is neither node-centred "
                f"({sn}, {sn + (3,)}, ({nn},), ({nn}, 3)) nor cell-centred "
                f"({sc}, {sc + (3,)}, ({nc},), ({nc}, 3))")
    return point, cell


def export_vtk(grid: RectilinearGrid, fields: Mapping[str, Any],
               path: str | os.PathLike[str], *, fmt: str = "binary",
               description: str = "fieldspice export") -> None:
    """Write a rectilinear grid and its fields for ParaView / VisIt.

    Parameters
    ----------
    grid
        Grid to export.  Node coordinates are written in metres, unscaled: a
        chip-scale model therefore spans ~1e-5 in ParaView, which is correct
        but needs the camera reset.
    fields
        ``{name: array}``.  Node-centred arrays have shape ``grid.shape_nodes``
        (or ``+ (3,)`` for a vector, or the flat ``(n_nodes,)`` /
        ``(n_nodes, 3)`` form); cell-centred arrays use ``grid.shape_cells``
        likewise.  Complex arrays are split into ``name_re`` and ``name_im``.
        Units are whatever the caller's arrays carry; VTK has no unit concept.
    path
        Output file.  ``.vtr`` writes the XML ``RectilinearGrid`` format,
        ``.vtk`` writes the legacy ``DATASET RECTILINEAR_GRID`` format.
    fmt
        ``"binary"`` (default) or ``"ascii"``.  XML binary uses a raw appended
        block with a ``UInt64`` size header and explicit little-endian data;
        legacy binary is big-endian, as that format mandates.
    description
        Header comment (legacy format only; truncated to 256 characters).

    Raises
    ------
    ValueError
        On an unknown suffix or format, or on a field whose shape matches
        neither the node nor the cell layout.

    Notes
    -----
    Index order is the trap.  VTK stores structured data with x varying
    fastest; fieldspice arrays are C-ordered with z fastest.  Every array is
    ravelled in Fortran order on the way out.  Legacy field names may not
    contain whitespace (a limitation of that format's ``SCALARS`` line), so
    such names raise rather than being silently mangled; use ``.vtr`` instead.
    """
    if fmt not in ("binary", "ascii"):
        raise ValueError(f"fmt must be 'binary' or 'ascii', got {fmt!r}")
    if not isinstance(grid, RectilinearGrid):
        raise ValueError(f"expected a RectilinearGrid, got {type(grid).__name__}")
    p = _as_path(path)
    suffix = p.suffix.lower()
    point, cell = _classify_fields(grid, fields)
    point = [(n, _vtk_cast(a, n), c) for n, a, c in point]
    cell = [(n, _vtk_cast(a, n), c) for n, a, c in cell]
    if suffix == ".vtr":
        _write_vtr(grid, point, cell, p, fmt)
    elif suffix == ".vtk":
        _write_legacy_vtk(grid, point, cell, p, fmt, description)
    else:
        raise ValueError(f"unsupported VTK suffix {suffix!r}; "
                         "use '.vtr' (XML) or '.vtk' (legacy)")


def _xml_attr(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def _ascii_numbers(arr: np.ndarray, per_line: int = 6) -> str:
    if arr.dtype.kind in "iub":
        toks = [str(int(v)) for v in arr]
    elif arr.dtype == np.float32:
        toks = [f"{float(v):.9g}" for v in arr]
    else:
        toks = [f"{float(v):.17g}" for v in arr]
    lines = [" ".join(toks[i:i + per_line]) for i in range(0, len(toks), per_line)]
    return "\n".join(lines) if lines else ""


def _write_vtr(grid: RectilinearGrid,
               point: Sequence[tuple[str, np.ndarray, int]],
               cell: Sequence[tuple[str, np.ndarray, int]],
               path: Path, fmt: str) -> None:
    """XML ``RectilinearGrid`` (.vtr), appended-raw or inline-ascii."""
    nx, ny, nz = grid.ncell
    extent = f"0 {nx} 0 {ny} 0 {nz}"          # extents count points, not cells
    coords = [np.asarray(grid.xn, dtype=np.float64),
              np.asarray(grid.yn, dtype=np.float64),
              np.asarray(grid.zn, dtype=np.float64)]

    blobs: list[bytes] = []
    offsets: list[int] = []
    total = 0

    def _declare(name: str, arr: np.ndarray, ncomp: int) -> str:
        nonlocal total
        flat = _vtk_flat(arr, ncomp)
        tname = _XML_DTYPE[flat.dtype]
        head = (f'<DataArray type="{tname}" Name="{_xml_attr(name)}" '
                f'NumberOfComponents="{ncomp}"')
        if fmt == "ascii":
            return head + ' format="ascii">\n' + _ascii_numbers(flat) + \
                "\n</DataArray>"
        raw = np.ascontiguousarray(flat, dtype=flat.dtype.newbyteorder("<")).tobytes()
        blobs.append(np.uint64(len(raw)).astype("<u8").tobytes() + raw)
        offsets.append(total)
        total += 8 + len(raw)
        return head + f' format="appended" offset="{offsets[-1]}"/>'

    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="RectilinearGrid" version="1.0" '
             'byte_order="LittleEndian" header_type="UInt64">',
             f'  <RectilinearGrid WholeExtent="{extent}">',
             f'    <Piece Extent="{extent}">',
             '      <Coordinates>']
    for name, arr in zip(("x", "y", "z"), coords):
        lines.append("        " + _declare(name, arr, 1))
    lines.append("      </Coordinates>")

    for tag, group in (("PointData", point), ("CellData", cell)):
        if not group:
            lines.append(f"      <{tag}/>")
            continue
        scalars = [n for n, _, c in group if c == 1]
        vectors = [n for n, _, c in group if c == 3]
        attrs = ""
        if scalars:
            attrs += f' Scalars="{_xml_attr(scalars[0])}"'
        if vectors:
            attrs += f' Vectors="{_xml_attr(vectors[0])}"'
        lines.append(f"      <{tag}{attrs}>")
        for name, arr, ncomp in group:
            lines.append("        " + _declare(name, arr, ncomp))
        lines.append(f"      </{tag}>")

    lines += ["    </Piece>", "  </RectilinearGrid>"]
    if fmt == "binary":
        lines.append('  <AppendedData encoding="raw">')
    lines.append("")
    header = "\n".join(lines)

    with open(path, "wb") as fh:
        fh.write(header.encode("utf-8"))
        if fmt == "binary":
            fh.write(b"   _")
            for blob in blobs:
                fh.write(blob)
            fh.write(b"\n  </AppendedData>\n")
        fh.write(b"</VTKFile>\n")


def _write_legacy_vtk(grid: RectilinearGrid,
                      point: Sequence[tuple[str, np.ndarray, int]],
                      cell: Sequence[tuple[str, np.ndarray, int]],
                      path: Path, fmt: str, description: str) -> None:
    """Legacy ``DATASET RECTILINEAR_GRID``.  Binary payloads are big-endian."""
    for name, _, _ in list(point) + list(cell):
        if any(ch.isspace() for ch in name):
            raise ValueError(
                f"legacy .vtk field names cannot contain whitespace: {name!r}; "
                "use the .vtr (XML) format instead")
    nx, ny, nz = grid.shape_nodes
    binary = fmt == "binary"

    def _payload(arr: np.ndarray) -> bytes:
        if not binary:
            return (_ascii_numbers(arr) + "\n").encode("ascii")
        # The legacy format is defined as big-endian regardless of host.
        return np.ascontiguousarray(
            arr, dtype=arr.dtype.newbyteorder(">")).tobytes() + b"\n"

    with open(path, "wb") as fh:
        w = lambda s: fh.write(s.encode("ascii"))  # noqa: E731
        w("# vtk DataFile Version 3.0\n")
        w(description.replace("\n", " ")[:256] + "\n")
        w("BINARY\n" if binary else "ASCII\n")
        w("DATASET RECTILINEAR_GRID\n")
        w(f"DIMENSIONS {nx} {ny} {nz}\n")
        for axis, nodes in (("X", grid.xn), ("Y", grid.yn), ("Z", grid.zn)):
            arr = np.asarray(nodes, dtype=np.float64)
            w(f"{axis}_COORDINATES {arr.size} double\n")
            fh.write(_payload(arr))
        for tag, group, count in (("POINT_DATA", point, grid.n_nodes),
                                  ("CELL_DATA", cell, grid.n_cells)):
            if not group:
                continue
            w(f"{tag} {count}\n")
            for name, arr, ncomp in group:
                flat = _vtk_flat(arr, ncomp)
                tname = _LEGACY_DTYPE[flat.dtype]
                if ncomp == 1:
                    w(f"SCALARS {name} {tname} 1\nLOOKUP_TABLE default\n")
                else:
                    w(f"VECTORS {name} {tname}\n")
                fh.write(_payload(flat))


# ==========================================================================
# Touchstone
# ==========================================================================
_FREQ_UNITS = {"HZ": 1.0, "KHZ": 1e3, "MHZ": 1e6, "GHZ": 1e9}


def to_touchstone(freqs: np.ndarray, s: np.ndarray,
                  path: str | os.PathLike[str], z0: float = 50.0, *,
                  fmt: str = "RI", unit: str = "HZ", precision: int = 12,
                  comments: Sequence[str] = ()) -> None:
    """Write S-parameters as a Touchstone ``.sNp`` file.

    Parameters
    ----------
    freqs
        Frequencies [Hz], shape ``(nf,)``, strictly increasing (the format
        requires ascending order).  They are divided by the ``unit`` scale on
        the way out and by nothing else.
    s
        Complex scattering parameters, shape ``(nf, N, N)`` with
        ``s[f, i, j] = S_ij``, i.e. ``S[out, in]``.  A 1-port may be passed as
        ``(nf,)``.  Dimensionless.
    path
        Output file.  If the suffix is absent it becomes ``.s{N}p``; if it is
        present it must agree with ``N``.
    z0
        Reference impedance [ohm] for every port.  Touchstone 1.x carries one
        real reference impedance, so per-port values are not expressible.
    fmt
        ``"RI"`` (real/imaginary, default), ``"MA"`` (linear magnitude and
        angle in degrees) or ``"DB"`` (20*log10 magnitude and angle in
        degrees).
    unit
        ``"HZ"`` (default), ``"KHZ"``, ``"MHZ"`` or ``"GHZ"``.
    precision
        Significant digits in the exponential number format.  The default of 12
        round-trips float64 S-parameters to better than 1e-12 relative.
    comments
        Extra ``!`` comment lines placed after the provenance header.

    Raises
    ------
    ValueError
        On a non-increasing frequency vector, a non-square or mismatched
        S-array, a bad format/unit keyword, or a filename whose ``.sNp`` suffix
        contradicts the port count.

    Notes
    -----
    **The two-port ordering wart.**  Touchstone orders the entries of a
    one-line-per-frequency record row-major for every port count *except two*:

    ====== =====================================================
    Ports  Order on the data line(s)
    ====== =====================================================
    1      ``S11``
    2      ``S11 S21 S12 S22``   (column-major: S21 before S12)
    >=3    ``S11 S12 S13 ... / S21 S22 S23 ... / ...`` row-major
    ====== =====================================================

    The two-port case is a historical accident that every tool implements, so
    getting it backwards silently transposes a device's gain and reverse
    isolation --- a two-port file that loads without complaint can still be
    wrong.  For three ports and up, one matrix row occupies its own line, and
    rows longer than four entries are wrapped at four complex pairs per line
    with the continuation indented.
    """
    fmt_u = str(fmt).upper()
    unit_u = str(unit).upper()
    if fmt_u not in ("RI", "MA", "DB"):
        raise ValueError(f"fmt must be 'RI', 'MA' or 'DB', got {fmt!r}")
    if unit_u not in _FREQ_UNITS:
        raise ValueError(f"unit must be one of {sorted(_FREQ_UNITS)}, got {unit!r}")
    if not np.isscalar(z0) or not np.isfinite(z0) or float(z0) <= 0.0:
        raise ValueError(f"z0 must be a positive finite scalar [ohm], got {z0!r}")
    if int(precision) < 1:
        raise ValueError("precision must be >= 1")

    f = np.asarray(freqs, dtype=float)
    if f.ndim != 1:
        raise ValueError(f"freqs must be 1-D [Hz], got shape {f.shape}")
    if f.size == 0:
        raise ValueError("freqs is empty")
    if np.any(f <= 0):
        raise ValueError("Touchstone frequencies must be positive")
    if f.size > 1 and np.any(np.diff(f) <= 0):
        raise ValueError("Touchstone requires strictly increasing frequencies")

    sm = np.asarray(s)
    if sm.ndim == 1:
        sm = sm.reshape(f.size, 1, 1)
    elif sm.ndim == 2 and sm.shape == (f.size, 1):
        sm = sm.reshape(f.size, 1, 1)
    if sm.ndim != 3:
        raise ValueError(f"s must have shape (nf, N, N), got {np.shape(s)}")
    if sm.shape[0] != f.size:
        raise ValueError(f"s has {sm.shape[0]} frequency rows but freqs has {f.size}")
    if sm.shape[1] != sm.shape[2]:
        raise ValueError(f"s must be square per frequency, got {sm.shape[1:]}")
    sm = sm.astype(np.complex128)
    n = int(sm.shape[1])

    p = _as_path(path)
    want = f".s{n}p"
    if p.suffix == "":
        p = p.with_name(p.name + want)
    elif p.suffix.lower() != want:
        raise ValueError(f"{p.name}: {n}-port data must use the suffix "
                         f"{want!r}, got {p.suffix!r}")

    scale = _FREQ_UNITS[unit_u]
    z0f = float(z0)
    z0s = f"{z0f:g}"
    num = f"%.{int(precision)}E"

    def _pair(z: complex) -> tuple[float, float]:
        if fmt_u == "RI":
            return z.real, z.imag
        mag = abs(z)
        ang = np.degrees(np.angle(z))
        if fmt_u == "MA":
            return mag, ang
        db = -np.inf if mag == 0.0 else 20.0 * np.log10(mag)
        return db, ang

    def _order(mat: np.ndarray) -> list[list[complex]]:
        """Entries grouped per output line, honouring the 2-port exception."""
        if n == 1:
            return [[mat[0, 0]]]
        if n == 2:
            return [[mat[0, 0], mat[1, 0], mat[0, 1], mat[1, 1]]]
        rows: list[list[complex]] = []
        for i in range(n):
            row = list(mat[i, :])
            for k in range(0, n, 4):
                rows.append(row[k:k + 4])
        return rows

    lines = [f"!fieldspice Touchstone 1.1 export, {n}-port S-parameters",
             f"!Reference impedance {z0s} ohm on every port; "
             f"frequency in {unit_u.lower()}",
             "!Entry order: " + ("S11" if n == 1 else
                                 "S11 S21 S12 S22 (Touchstone two-port "
                                 "column-major convention)" if n == 2 else
                                 "row-major, one matrix row per line")]
    lines += [f"!{c}" for c in comments]
    lines.append(f"# {unit_u} S {fmt_u} R {z0s}")

    for fi in range(f.size):
        groups = _order(sm[fi])
        head = num % (f[fi] / scale)
        for gi, group in enumerate(groups):
            vals = []
            for z in group:
                a, b = _pair(complex(z))
                vals.append(num % a)
                vals.append(num % b)
            prefix = head + " " if gi == 0 else " " * (len(head) + 1)
            lines.append(prefix + " ".join(vals))

    p.write_text("\n".join(lines) + "\n", encoding="ascii")
