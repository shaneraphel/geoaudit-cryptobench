#!/usr/bin/env python3.12
"""Quaternary digits of a wide cache, built without ever holding it whole.

Why this exists
---------------
Every tool on the wire axis starts the same way: load
``_wide_cache_train.npz``, take ``X``, and call ``chain_digits`` on it. ``X`` is
234,838 x 645 float32, 578 MB decompressed, and the digits it produces are int8
and 151 MB. Nothing downstream reads the floats -- the readouts, the Gram
matrices, the tables and the fan-out all consume digits -- so the 578 MB is
carried only to be thrown away, and on a 16 GB host with an IDE holding 6 GB it
is the allocation that gets the process killed. Three runs of
``collectability_screen.py --cross`` died exactly there, with no traceback,
because the kernel does not write one.

The npz members are deflate-compressed, so the array cannot be memory-mapped in
place. But ``zipfile`` will hand back a *stream*, and the digits of a chain
depend only on that chain: ``chain_digits`` ranks within each chain over that
chain's own residues and writes its rows independently of every other chain.
Chains are contiguous in row order. So the file can be consumed one chain at a
time and the digits written straight into a memory-mapped int8 output, with a
few megabytes resident rather than three-quarters of a gigabyte.

Identity, not approximation
---------------------------
This is a reimplementation of a piece of the pipeline, which in this repository
means it has to reproduce the canonical path rather than resemble it. Two
things make that easy to guarantee and it is still checked rather than argued:

* the per-chain loop calls ``chain_digits`` itself on one chain, so the ranking,
  the tie handling and the level boundaries are the deployed code and not a
  copy of it;
* ``--check`` digitises the first chains through both paths and requires the
  int8 arrays to be equal, which is the only claim this tool makes.

The cache is written under ``data/cryptobench_apo/`` where ``_*.npy`` is
gitignored, so it is a local derived file and not an artifact. It carries a
sidecar recording the source digest, the shape and the level count, and is
rebuilt rather than trusted whenever any of those disagree.

Usage:
  PYTHONPATH=tools:src python3.12 tools/digit_cache.py --check
  PYTHONPATH=tools:src python3.12 tools/digit_cache.py --rebuild
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path

import numpy as np

from pocket_bench.methods.table_bank import N_LEVELS, chain_digits
from pocket_bench.paths import ROOT

WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
DIGITS = ROOT / "data/cryptobench_apo/_wide_digits_train.npy"
SIDECAR = ROOT / "data/cryptobench_apo/_wide_digits_train.json"

# Read this many rows of the compressed stream at a time. The constraint is not
# speed, it is that a chain must be complete before it can be ranked, so the
# buffer has to hold at least the longest chain; 8,192 rows is 21 MB at 645
# float32 columns and the longest chain in the training fold is under 1,500.
STREAM_ROWS = 8192


def _digest(path: Path, n_bytes: int = 1 << 20) -> str:
    """Digest of the source, cheaply: size plus first and last megabyte.

    A full sha256 of 461 MB costs a second and buys nothing here. What this has
    to catch is the cache being rebuilt from a different wide cache, which
    changes the length and the header, not a targeted edit of the middle of a
    compressed member.
    """
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode())
    with path.open("rb") as fh:
        h.update(fh.read(n_bytes))
        if size > 2 * n_bytes:
            fh.seek(-n_bytes, 2)
            h.update(fh.read(n_bytes))
    return h.hexdigest()


def _stream_member(zf: zipfile.ZipFile, member: str):
    """Open one npy member of an npz and return (stream, dtype, shape)."""
    fh = zf.open(member, "r")
    # The public header readers, chosen on the version byte. numpy 2.x dropped
    # the private _check_version and _read_array_header this first reached for,
    # and a private helper is not something to depend on for a file format that
    # has a documented reader.
    major, minor = np.lib.format.read_magic(fh)
    if (major, minor) == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(fh)
    elif (major, minor) == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(fh)
    else:
        raise SystemExit(f"{member} is npy format {major}.{minor}, which this "
                         f"reader does not know how to parse")
    if fortran:
        raise SystemExit(f"{member} is Fortran-ordered; the row-block reader "
                         f"assumes C order and would silently transpose it")
    return fh, dtype, shape


def build(n_res_per: np.ndarray, member: str = "X.npy",
          src: Path = WIDE, out: Path = DIGITS) -> np.ndarray:
    """Digitise ``member`` chain by chain, straight into a memory-mapped int8."""
    t0 = time.perf_counter()
    with zipfile.ZipFile(src) as zf:
        fh, dtype, shape = _stream_member(zf, member)
        n_rows, n_cols = int(shape[0]), int(shape[1])
        if int(np.sum(n_res_per)) != n_rows:
            raise SystemExit(
                f"{member} has {n_rows} rows and n_res_per sums to "
                f"{int(np.sum(n_res_per))}; the chain boundaries do not "
                f"describe this array and every rank would be drawn over the "
                f"wrong residues")
        itemsize = int(dtype.itemsize)
        out.parent.mkdir(parents=True, exist_ok=True)
        dst = np.lib.format.open_memmap(out, mode="w+", dtype=np.int8,
                                        shape=(n_rows, n_cols))
        buf = np.empty((0, n_cols), dtype=dtype)
        off = 0
        with fh:
            for n in n_res_per:
                n = int(n)
                while buf.shape[0] < n:
                    want = max(STREAM_ROWS, n - buf.shape[0])
                    raw = fh.read(want * n_cols * itemsize)
                    if not raw:
                        raise SystemExit(
                            f"{member} ended after {off + buf.shape[0]} rows "
                            f"with a chain of {n} still to fill")
                    got = np.frombuffer(raw, dtype=dtype).reshape(-1, n_cols)
                    buf = got if buf.shape[0] == 0 else np.concatenate(
                        [buf, got], axis=0)
                # chain_digits itself, on one chain, so the ranking rule is the
                # deployed one rather than a restatement of it
                dst[off:off + n] = chain_digits(buf[:n], (n,))
                buf = buf[n:]
                off += n
        dst.flush()
    SIDECAR.write_text(json.dumps({
        "built_from": str(src.relative_to(ROOT)),
        "member": member,
        "source_digest": _digest(src),
        "shape": [n_rows, n_cols],
        "n_levels": N_LEVELS,
        "n_chains": int(len(n_res_per)),
        "seconds": round(time.perf_counter() - t0, 1),
        "what_it_is": "quaternary digits by within-chain rank, identical to "
                      "chain_digits(X, n_res_per), built one chain at a time "
                      "so the 578 MB float array is never resident",
    }, indent=1) + "\n")
    print(f"wrote {out.relative_to(ROOT)}  ({n_rows} x {n_cols} int8, "
          f"{dst.nbytes / 2**20:.0f} MB) in {time.perf_counter() - t0:.0f}s",
          flush=True)
    return dst


def load(n_res_per: np.ndarray, member: str = "X.npy", src: Path = WIDE,
         out: Path = DIGITS, rebuild: bool = False) -> np.ndarray:
    """The digit cache, built if absent or stale, returned memory-mapped."""
    want = {
        "built_from": str(src.relative_to(ROOT)),
        "member": member,
        "source_digest": _digest(src),
        "n_levels": N_LEVELS,
        "n_chains": int(len(n_res_per)),
    }
    if not rebuild and out.is_file() and SIDECAR.is_file():
        have = json.loads(SIDECAR.read_text())
        if all(have.get(k) == v for k, v in want.items()):
            return np.load(out, mmap_mode="r")
        print("digit cache is stale; rebuilding", flush=True)
    return build(n_res_per, member, src, out)


def check(n_res_per: np.ndarray, n_chains: int = 60) -> dict:
    """Require the streamed digits to equal the canonical whole-array path."""
    k = min(int(n_chains), len(n_res_per))
    n = int(np.sum(n_res_per[:k]))
    with zipfile.ZipFile(WIDE) as zf:
        fh, dtype, shape = _stream_member(zf, "X.npy")
        with fh:
            raw = fh.read(n * int(shape[1]) * int(dtype.itemsize))
    head = np.frombuffer(raw, dtype=dtype).reshape(n, int(shape[1]))
    want = chain_digits(head, n_res_per[:k])
    got = np.asarray(load(n_res_per)[:n])
    same = np.array_equal(want, got)
    if not same:
        bad = int((want != got).sum())
        raise SystemExit(
            f"the streamed digit cache differs from chain_digits on the first "
            f"{k} chains in {bad} of {want.size} entries; refusing to let a "
            f"cache stand in for the computation it is meant to reproduce")
    return {
        "checked_on": f"the first {k} chains, {n} residues, {shape[1]} columns",
        "identical_to_chain_digits": bool(same),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true",
                    help="rebuild even if the sidecar matches")
    ap.add_argument("--check", action="store_true",
                    help="require the streamed digits to equal chain_digits on "
                         "the leading chains, which is the tool's only claim")
    a = ap.parse_args(argv)

    with zipfile.ZipFile(WIDE) as zf:
        with zf.open("n_res_per.npy") as fh:
            n_res_per = np.lib.format.read_array(fh, allow_pickle=False)

    D = load(n_res_per, rebuild=a.rebuild)
    print(f"digits {D.shape} {D.dtype}, "
          f"{D.nbytes / 2**20:.0f} MB on disk, memory-mapped")
    if a.check:
        r = check(n_res_per)
        print(f"  identical to chain_digits on {r['checked_on']}: "
              f"{r['identical_to_chain_digits']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
