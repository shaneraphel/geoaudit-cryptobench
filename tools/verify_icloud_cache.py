#!/usr/bin/env python3
"""Check that the raw PDBs said to be cached in iCloud are there and unaltered.

The manifest's own schema is named ``raw_pdb_icloud`` and its note says the raw
files are "bulk data cached in iCloud, not committed" -- but the entries recorded
only a PDB id, a SHA-256 and an RCSB URL. Nothing in the file said where in
iCloud, so the claim was unverifiable from the artifact and the location survived
only as folklore. A reader could not check it, and neither could a later run.

This tool resolves each entry under the recorded iCloud root, hashes the bytes
and compares against the manifest. It is the difference between saying the cache
exists and knowing it does.

The receptor-only ATOM PDBs and the labels are committed and are not affected;
this covers the bulk originals alone.

Usage:
  PYTHONPATH=src python3.12 tools/verify_icloud_cache.py
  PYTHONPATH=src python3.12 tools/verify_icloud_cache.py --quiet
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data/manifests/RAW_PDB_ICLOUD.json"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_root(doc: dict) -> Path | None:
    """The iCloud directory the manifest points at, expanded for this machine.

    Stored with ``~`` rather than an absolute path: the container lives under a
    home directory, and the repository's scope gate rejects those in published
    files for good reason.
    """
    rel = doc.get("icloud_root")
    return Path(rel).expanduser() if rel else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    doc = json.loads(MANIFEST.read_text())
    entries = doc.get("entries") or {}
    root = resolve_root(doc)
    if root is None:
        print("MANIFEST INCOMPLETE: no icloud_root recorded, so the cache "
              "location is not stated anywhere and cannot be checked")
        return 1
    if not root.exists():
        print(f"iCloud container not present on this machine: {doc['icloud_root']}")
        print(f"  {len(entries)} raw PDBs are therefore unverified here. They are "
              f"re-downloadable from RCSB; the receptor PDBs and labels this "
              f"paper depends on are committed and unaffected.")
        return 0

    ok = bad = missing = 0
    problems: list[str] = []
    for pdb_id, rec in sorted(entries.items()):
        f = root / f"{pdb_id}.pdb"
        if not f.is_file():
            missing += 1
            problems.append(f"{pdb_id}: absent from the cache")
            continue
        want = rec.get("raw_sha256")
        got = _sha256(f)
        if want and got != want:
            bad += 1
            problems.append(f"{pdb_id}: sha256 {got[:12]} on disk, "
                            f"{want[:12]} in the manifest")
        else:
            ok += 1

    if problems:
        print(f"iCloud raw-PDB cache FAILED: {ok} verified, {bad} altered, "
              f"{missing} absent")
        for p in problems[:20]:
            print(f"  - {p}")
        print("  re-fetch: PYTHONPATH=src python3.12 tools/build_labels.py "
              "--download")
        return 1
    if not args.quiet:
        print(f"iCloud raw-PDB cache verified: {ok} files under "
              f"{doc['icloud_root']}, every SHA-256 matches the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
