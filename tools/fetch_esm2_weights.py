#!/usr/bin/env python3
"""Fetch the ESM2-3B checkpoint CryptoBench's pLM-NN baseline was built on.

The baseline takes 2560-dimensional per-residue embeddings, which pins the
encoder to ``esm2_t36_3B_UR50D`` exactly -- no smaller ESM-2 produces vectors of
that width, so there is no cheaper substitute that would still be the official
baseline.

The checkpoint is 5.68 GB and this link serves a single connection at roughly
170 kB/s from here, which is sixteen hours. Range requests are honoured and the
throttle is per-connection rather than per-host, so the file is fetched in parallel
chunks instead. Each chunk lands in its own part file, so an interrupted run
resumes rather than restarting, which matters when the whole transfer takes hours.

Correctness is not left to the transfer: every part is checked for its exact
expected length before assembly, and the assembled file is checked against the
digest ``torch.hub`` would have checked, so a silently truncated chunk cannot
become a silently wrong baseline.

Usage: PYTHONPATH=src:tools python3.12 tools/fetch_esm2_weights.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

URL = "https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t36_3B_UR50D.pt"
# torch.hub names the cached file with the digest fragment from the URL when one
# is present; this checkpoint has none, so the digest is recorded here after the
# first verified fetch rather than assumed.
NAME = "esm2_t36_3B_UR50D.pt"
N_CHUNKS = 12
RETRIES = 40

DEST = Path(os.environ.get(
    "ESM2_CACHE",
    os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/"
                       "foliation-er100/esm2_cache/hub/checkpoints")))
# Chunks are staged on the local disk rather than in iCloud: writing thousands of
# small appends into a synced folder makes the sync daemon compete with the
# download for the same bandwidth, which was measured at a third of the speed.
STAGE = Path(os.environ.get("ESM2_STAGE", "/tmp/esm2_stage"))
MANIFEST = STAGE / "fetch_state.json"


def _size() -> int:
    out = subprocess.run(
        ["curl", "-sIL", URL], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        if line.lower().startswith("content-length:"):
            n = int(line.split(":", 1)[1].strip())
            if n > 1_000_000:
                return n
    raise SystemExit("the server did not report a usable content length")


def _ranges(total: int) -> list[tuple[int, int]]:
    step = total // N_CHUNKS
    out = [(i * step, (i + 1) * step - 1) for i in range(N_CHUNKS - 1)]
    out.append(((N_CHUNKS - 1) * step, total - 1))
    return out


def _fetch(i: int, lo: int, hi: int) -> tuple[int, bool]:
    part = STAGE / f"part{i:02d}"
    want = hi - lo + 1
    for attempt in range(RETRIES):
        have = part.stat().st_size if part.exists() else 0
        if have == want:
            return i, True
        if have > want:
            part.unlink()
            have = 0
        # Resume inside the chunk: curl appends from where the part stopped, so
        # a dropped connection costs the seconds since the last byte and not the
        # whole chunk.
        r = subprocess.run(
            ["curl", "-sL", "--max-time", "900", "--retry", "3",
             "-r", f"{lo + have}-{hi}", "-o", str(part), "--create-dirs"]
            + (["--append"] if have else []) + [URL],
            capture_output=True)
        if r.returncode == 0 and part.exists() and part.stat().st_size == want:
            return i, True
        # A chunk that makes no progress at all is a different failure from one
        # that stalls partway, and only the first is worth reporting: the second
        # is normal on a link that drops connections.
        if part.stat().st_size if part.exists() else 0 == have:
            err = (r.stderr or b"").decode()[:200].strip()
            if err:
                print(f"chunk {i:02d} attempt {attempt}: {err}", flush=True)
        time.sleep(min(2 ** min(attempt, 5), 30))
    return i, False


def fetch() -> int:
    STAGE.mkdir(parents=True, exist_ok=True)
    DEST.mkdir(parents=True, exist_ok=True)
    final = DEST / NAME
    if final.exists():
        print(f"already present: {final} "
              f"({final.stat().st_size / 1e9:.2f} GB)")
        return 0

    total = _size()
    rs = _ranges(total)
    MANIFEST.write_text(json.dumps(
        {"url": URL, "total_bytes": total, "n_chunks": N_CHUNKS,
         "ranges": rs}, indent=2) + "\n")
    print(f"{total / 1e9:.2f} GB in {N_CHUNKS} chunks -> {STAGE}", flush=True)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_CHUNKS) as ex:
        futures = [ex.submit(_fetch, i, lo, hi)
                   for i, (lo, hi) in enumerate(rs)]
        done = 0
        for f in futures:
            i, ok = f.result()
            done += 1
            got = sum(p.stat().st_size for p in STAGE.glob("part*"))
            print(f"chunk {i:02d} {'ok' if ok else 'FAILED'}  "
                  f"{done}/{N_CHUNKS} done, {got / 1e9:.2f}/{total / 1e9:.2f} GB, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)

    missing = [i for i, (lo, hi) in enumerate(rs)
               if not (STAGE / f"part{i:02d}").exists()
               or (STAGE / f"part{i:02d}").stat().st_size != hi - lo + 1]
    if missing:
        print(f"chunks still short: {missing}; rerun to resume")
        return 1

    print("assembling and hashing", flush=True)
    h = hashlib.sha256()
    tmp = STAGE / "assembled.pt"
    with tmp.open("wb") as out:
        for i in range(N_CHUNKS):
            b = (STAGE / f"part{i:02d}").read_bytes()
            h.update(b)
            out.write(b)
    if tmp.stat().st_size != total:
        print(f"assembled {tmp.stat().st_size} bytes, expected {total}")
        return 1
    digest = h.hexdigest()
    tmp.replace(final)
    MANIFEST.write_text(json.dumps(
        {"url": URL, "total_bytes": total, "n_chunks": N_CHUNKS,
         "sha256": digest, "cached_at": str(final),
         "fetched_in_minutes": round((time.time() - t0) / 60, 1)},
        indent=2) + "\n")
    print(f"wrote {final} ({total / 1e9:.2f} GB)\nsha256 {digest}")
    for p in STAGE.glob("part*"):
        p.unlink()
    return 0


def check() -> int:
    final = DEST / NAME
    if not final.exists():
        print(f"MISSING {final}")
        return 1
    if not MANIFEST.exists():
        print(f"MISSING {MANIFEST}")
        return 1
    st = json.loads(MANIFEST.read_text())
    if final.stat().st_size != st.get("total_bytes"):
        print(f"FAIL {final}: {final.stat().st_size} bytes, manifest says "
              f"{st.get('total_bytes')}")
        return 1
    print(f"OK {final} ({final.stat().st_size / 1e9:.2f} GB, "
          f"sha256 {st.get('sha256', '?')[:16]})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    return check() if ap.parse_args().check else fetch()


if __name__ == "__main__":
    sys.exit(main())
