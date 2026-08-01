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
    i8 = ctypes.POINTER(ctypes.c_int8)
    i32 = ctypes.POINTER(ctypes.c_int32)
    i64 = ctypes.POINTER(ctypes.c_int64)
    lib.gk_table_addresses.restype = ctypes.c_int32
    lib.gk_table_addresses.argtypes = [
        i8, ctypes.c_size_t, ctypes.c_size_t,
        i32, ctypes.c_size_t, ctypes.c_size_t,
        i64, ctypes.c_int64, i64,
    ]
    lib.gk_table_cell_counts.restype = ctypes.c_int32
    lib.gk_table_cell_counts.argtypes = [
        i8, ctypes.c_size_t, ctypes.c_size_t, u8,
        i32, ctypes.c_size_t, ctypes.c_size_t,
        i64, ctypes.c_int64, ctypes.c_size_t, i64, i64,
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


def table_addresses(D: np.ndarray, cols: np.ndarray, offsets: np.ndarray,
                    n_levels: int, a: int, b: int) -> np.ndarray | None:
    """``(b - a, n_tables)`` cell addresses, or None if native is unavailable.

    The one function every consumer of the counting field passes through:
    ``compile_cells`` calls it once per block, ``scatter_and_means`` twice and
    ``score`` once. For 10,144 tables at width 2 that is 20,288 integer
    multiply-accumulate passes over each block of 8,192 rows, and NumPy runs all
    of it on one core -- Accelerate is not involved, because there is no
    floating-point product in it.

    Bit-identical to the NumPy loop rather than approximately equal. That is a
    property of the port and not something to be checked with a tolerance: the
    result indexes a cell array, so an address off by one is a different cell
    entirely, with no numerical smallness to make the error visible.

    Returns None when the library is absent, when the tables are not all the same
    width, or when the kernel refuses a column index out of range -- in every
    case the caller falls back to NumPy rather than guessing.
    """
    lib = _load()
    if lib is None:
        return None
    cols = np.ascontiguousarray(cols, dtype=np.int32)
    if cols.ndim != 2:
        return None
    n_tables, width = cols.shape
    Dblk = np.ascontiguousarray(D[a:b], dtype=np.int8)
    n_rows, n_cols = Dblk.shape
    off = np.ascontiguousarray(offsets[:n_tables], dtype=np.int64)
    out = np.empty((n_rows, n_tables), dtype=np.int64)
    rc = lib.gk_table_addresses(
        Dblk.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        n_rows, n_cols,
        cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        n_tables, width,
        off.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        int(n_levels),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if rc != 0:
        return None
    return out


def table_cell_counts(D: np.ndarray, y: np.ndarray, cols: np.ndarray,
                      offsets: np.ndarray, n_levels: int, n_cells: int
                      ) -> tuple[np.ndarray, np.ndarray] | None:
    """``(total, positive)`` counts per cell over every row, or None.

    Fuses the addressing into the reduction, so the ``(n_rows, n_tables)`` address
    matrix is never built. NumPy materialises it and calls ``bincount`` twice: at
    8,192 rows and 10,144 tables that is 665 MB written once and read twice, with
    both reductions on one core.

    Bit-identical, and the reason it can be while splitting a sum across threads
    is specific rather than general. The positive count is a sum of the label,
    the label is exactly 0 or 1, so the quantity is an integer; it is accumulated
    as ``int64`` here and an integer sum does not depend on the order its terms
    arrive in. That makes the parallel result equal to the serial one exactly, and
    equal to NumPy's ``float64`` bincount exactly, for any count below ``2**53``.
    The division into frequencies stays in Python, so the one place a float enters
    is the one place it has to.
    """
    lib = _load()
    if lib is None:
        return None
    cols = np.ascontiguousarray(cols, dtype=np.int32)
    if cols.ndim != 2:
        return None
    n_tables, width = cols.shape
    Dc = np.ascontiguousarray(D, dtype=np.int8)
    n_rows, n_cols = Dc.shape
    yc = np.ascontiguousarray(y, dtype=np.uint8)
    if yc.shape[0] != n_rows or not np.isin(yc, (0, 1)).all():
        return None
    off = np.ascontiguousarray(offsets[:n_tables], dtype=np.int64)
    total = np.zeros(n_cells, dtype=np.int64)
    pos = np.zeros(n_cells, dtype=np.int64)
    rc = lib.gk_table_cell_counts(
        Dc.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)), n_rows, n_cols,
        yc.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        cols.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), n_tables, width,
        off.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), int(n_levels),
        int(n_cells),
        total.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        pos.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
    )
    if rc != 0:
        return None
    return total, pos
