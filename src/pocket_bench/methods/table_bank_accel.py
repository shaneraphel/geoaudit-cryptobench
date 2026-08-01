"""Multithreaded substitutes for the two hot integer loops of ``table_bank``.

Why this is a separate file, and not three edits inside ``table_bank.py``
------------------------------------------------------------------------
``table_field.code_sha256`` hashes the bytes of eight source files, and
``table_bank.py`` is one of them. ``TABLE_FIELD.json`` and ``GEOMETRY_FIELD.json``
both record ``fdfe2f27...`` for that digest, and every per-unit prediction under
``results/external/`` carries it as ``tool_version``. The digest exists so a reader
can fetch the exact code that produced a frozen score; it is over bytes, not over
behaviour, so a change that provably cannot move a number still breaks it.

The kernels were first written directly into ``table_bank.py``. The numbers were
bit-identical and the gate failed anyway, correctly: the frozen predictions then
claimed a version of the source that no longer existed. The accelerated paths live
here so the eight digested files stay byte-identical to what the frozen artifacts
pin, and ``check_frozen_field_digest`` in ``tools/verify_claims.py`` now fails with
the name of the offending file if any of them is touched again. ``scatter_and_means``
in ``tools/straddling_attachment.py`` had already reached the same conclusion and
copied its loop rather than importing it.

What this buys, and what it does not
------------------------------------
This is for the attachment harness, which compiles a bank per split per arm and so
runs these loops thousands of times. The deployed detector calls them once per chain
for about half a second and gains nothing worth this file. Measured on a full split
of ``geometry 624``: 162--165 s with the kernels against 189--219 s without, a
factor of 1.25 end to end, because 86% of the split is the K x K Gram matrix, which
Accelerate already spreads over every core. The two loops here were 6% of the split
and are now 0.5%; the remaining ceiling is the Gram matrix and is not an integer
problem.

Both substitutes are ports rather than reimplementations --- integer throughout,
same accumulation order within a row --- and are checked for bit-identity, not for
closeness. That distinction matters more here than the speed does: the output of
``addresses`` indexes a cell array, so an address off by one selects a different
cell and a different score, with no numerical smallness to make the error visible.
A tolerance could not establish correctness for this function, and none is used.
Every path falls back to the reference loop when the kernel declines, so a machine
with no library built produces the same numbers more slowly.
"""
from __future__ import annotations

import numpy as np

from pocket_bench import native
from pocket_bench.methods.table_bank import (  # noqa: F401  (re-exported)
    BLOCK,
    N_LEVELS,
    addresses as addresses_reference,
    compile_cells as compile_cells_reference,
)


def _column_matrix(tables) -> np.ndarray | None:
    """``(K, width)`` int32 column indices, or ``None`` for ragged banks.

    Derived per call rather than cached on the module. A cache keyed on nothing
    would hand one bank's column layout to another's tables the moment two banks
    exist in one process, which is what every arm of the attachment harness does.
    At 10,144 tables of width 2 the array is 81 kB, against the hundreds of
    megabytes the addresses themselves occupy.
    """
    if not tables:
        return None
    width = len(tables[0])
    if width < 2 or any(len(t) != width for t in tables):
        return None
    cols = np.empty((len(tables), width), dtype=np.int32)
    for k, t in enumerate(tables):
        cols[k] = t
    return cols


def addresses(D: np.ndarray, tables, offsets: np.ndarray,
              a: int, b: int) -> np.ndarray:
    """``(b - a, n_tables)`` addresses, identical to ``table_bank.addresses``.

    The work is integer multiply-accumulate, so the BLAS that makes the rest of the
    pipeline fast is not involved and NumPy runs it on one core. At 10,144 tables of
    width 2 that is 20,288 passes over each block of 8,192 rows. The kernel splits
    disjoint row ranges across threads, which changes nothing about the arithmetic
    because each row's address depends on that row alone.
    """
    cols = _column_matrix(tables)
    if cols is not None:
        got = native.table_addresses(D, cols, offsets, N_LEVELS, a, b)
        if got is not None:
            return got
    return addresses_reference(D, tables, offsets, a, b)


def compile_cells(D: np.ndarray, y: np.ndarray, tables,
                  offsets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell frequencies, identical to ``table_bank.compile_cells``.

    The fused kernel never builds the address matrix, which at 8,192 rows and
    10,144 tables is 665 MB written once and read twice, and it counts in int64 ---
    exact, order-independent, and therefore safe to split across threads. The label
    is 0 or 1, so the positive tally that the reference accumulates as a float sum
    is an integer one, and the two paths agree bit for bit rather than to a
    tolerance. Unseen addresses take the fold's base rate in both.
    """
    cols = _column_matrix(tables)
    if cols is not None:
        total = int(offsets[-1])
        got = native.table_cell_counts(D, y, cols, offsets, N_LEVELS, total)
        if got is not None:
            tot, pos = got
            rate = float(y.astype(np.float64).mean())
            frac = np.where(tot > 0, pos / np.maximum(tot, 1), rate)
            return frac, tot
    return compile_cells_reference(D, y, tables, offsets)
