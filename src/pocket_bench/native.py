"""Optional native (Rust) geometry kernels with a NumPy fallback.

The kernels are a bit-identical port of the NumPy reference loops, so enabling or
disabling them cannot change a result — only the wall-clock. Every reduction inside
them is a `min` or a boolean `any`, both order-independent, so thread scheduling is
not observable in the output.

Build (optional):
    cd native/geoaudit_kernels && CARGO_TARGET_DIR="$PWD/target" cargo build --release

If the shared library is absent, ``available()`` is False and callers transparently
use the NumPy path. Set ``GEOAUDIT_NO_NATIVE=1`` to force the fallback.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_CANDIDATES = (
    _ROOT / "native/geoaudit_kernels/target/release/libgeoaudit_kernels.dylib",
    _ROOT / "native/geoaudit_kernels/target/release/libgeoaudit_kernels.so",
)

_LIB: ctypes.CDLL | None = None
_TRIED = False


def _load() -> ctypes.CDLL | None:
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True
    if os.environ.get("GEOAUDIT_NO_NATIVE"):
        return None
    for path in _CANDIDATES:
        if not path.is_file():
            continue
        try:
            lib = ctypes.CDLL(str(path))
        except OSError:
            continue
        try:
            _bind(lib)
        except AttributeError:
            # Stale library built before a kernel was added. Fall back to NumPy
            # wholesale rather than exposing a half-bound module whose missing
            # attribute would only surface as a crash deep inside a detector.
            continue
        _LIB = lib
        break
    return _LIB


def _bind(lib: ctypes.CDLL) -> None:
    """Bind every required symbol. Raises AttributeError if the library is stale."""
    dbl = ctypes.POINTER(ctypes.c_double)
    u8 = ctypes.POINTER(ctypes.c_uint8)
    lib.gk_free_grid_mask.restype = None
    lib.gk_free_grid_mask.argtypes = [
        dbl, ctypes.c_size_t, dbl, ctypes.c_size_t,
        ctypes.c_double, ctypes.c_double, u8, u8,
    ]
    lib.gk_buriedness.restype = None
    lib.gk_buriedness.argtypes = [
        dbl, ctypes.c_size_t, dbl, ctypes.c_size_t, dbl, ctypes.c_size_t,
        ctypes.c_double, ctypes.c_double, dbl,
    ]
    lib.gk_local_free_enclosed.restype = ctypes.c_uint64
    lib.gk_local_free_enclosed.argtypes = [
        dbl, ctypes.c_size_t, dbl, ctypes.c_size_t,
        ctypes.c_double, ctypes.c_double, ctypes.c_uint64,
    ]


def available() -> bool:
    return _load() is not None


def _c(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def _p(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def free_grid_mask(pts: np.ndarray, coords: np.ndarray, atom_r: float,
                   near_r: float) -> tuple[np.ndarray, np.ndarray] | None:
    """(keep, near) boolean masks per grid point, or None if native is unavailable."""
    lib = _load()
    if lib is None:
        return None
    pts_c, coords_c = _c(pts), _c(coords)
    n_pts, n_atoms = len(pts_c), len(coords_c)
    keep = np.zeros(n_pts, dtype=np.uint8)
    near = np.zeros(n_pts, dtype=np.uint8)
    lib.gk_free_grid_mask(
        _p(pts_c), n_pts, _p(coords_c), n_atoms, float(atom_r), float(near_r),
        keep.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        near.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )
    return keep.astype(bool), near.astype(bool)


def buriedness(pts: np.ndarray, coords: np.ndarray, dirs: np.ndarray,
               cutoff: float, perp: float) -> np.ndarray | None:
    """Blocked-direction fraction per point, or None if native is unavailable."""
    lib = _load()
    if lib is None:
        return None
    pts_c, coords_c, dirs_c = _c(pts), _c(coords), _c(dirs)
    out = np.zeros(len(pts_c), dtype=np.float64)
    lib.gk_buriedness(
        _p(pts_c), len(pts_c), _p(coords_c), len(coords_c),
        _p(dirs_c), len(dirs_c), float(cutoff), float(perp) * float(perp),
        _p(out),
    )
    return out


def local_free_enclosed(pts: np.ndarray, coords: np.ndarray, atom_r: float,
                        enclose_cut: float, enclose_min: int) -> int | None:
    """# of probe points that are free and still enclosed, or None if unavailable."""
    lib = _load()
    if lib is None:
        return None
    pts_c, coords_c = _c(pts), _c(coords)
    return int(lib.gk_local_free_enclosed(
        _p(pts_c), len(pts_c), _p(coords_c), len(coords_c),
        float(atom_r), float(enclose_cut), int(enclose_min),
    ))
