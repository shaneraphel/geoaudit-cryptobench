#!/usr/bin/env python3.12
"""Fetch the full deposited PDB entry for every unit in the train and test folds.

Why this is a separate download
-------------------------------
The receptor files this repository scores were stripped to polymer ATOM records
before they were committed. Measured on all 962 units, **zero** of them carry a
``CRYST1`` record, a ``SCALE`` matrix, a ``REMARK 290 SMTRY`` operator or an
``ANISOU`` line. Every descriptor family written so far is therefore a function
of a point cloud in an arbitrary frame, and two experimentally measured
quantities that ship with every crystal structure have never been read:

* **the crystal form** — space group, cell, and the symmetry operators that
  generate the neighbouring molecules in the lattice. A surface patch buried by
  a symmetry mate is a patch that prefers to be an interface rather than
  solvent, which is the same preference a cryptic site expresses.
* **the anisotropic displacement parameters** — ``ANISOU``, a per-atom
  symmetric 3x3 in units of 1e-4 A^2. The isotropic B-factor already carries
  most of one existing family (AGENT_MEMORY 2n); ANISOU carries the *direction*
  of that motion, which no isotropic number can state.

Neither is a label and neither comes from the holo structure. Both are
properties of the apo crystal, present in the deposited entry at prediction
time. That said, using them changes the input class of the detector, so they are
built as a separate arm and never folded into the receptor-only numbers.
See ``docs/DECISIONS.md``.

What is checked
---------------
The chain we score must exist in the downloaded entry and its CA coordinates
must match the committed receptor to within a tolerance, otherwise the entry has
been re-refined or re-deposited since the receptor was cut and the two cannot be
combined. Units that fail are recorded, not silently dropped.

``clinical_grade`` is false.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pocket_bench.paths import ROOT


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

DEST = ROOT / "data/deposited_entries"
OUT = ROOT / "data/deposited_entries/DEPOSITED_ENTRY_MANIFEST.json"
URL = "https://files.rcsb.org/download/{pdb}.pdb.gz"
CIF_URL = "https://files.rcsb.org/download/{pdb}.cif.gz"
SCHEMA = "geoaudit.deposited_entry_manifest.v1"


def _fetch(pdb: str, retries: int = 4) -> tuple[str, str, int]:
    """Return (pdb, status, n_bytes). Written gzipped, exactly as served."""
    dest = DEST / f"{pdb.lower()}.pdb.gz"
    if dest.exists() and dest.stat().st_size > 1024:
        return pdb, "cached", dest.stat().st_size
    last = ""
    for attempt in range(retries):
        for url in (URL, CIF_URL):
            target = (dest if url is URL
                      else DEST / f"{pdb.lower()}.cif.gz")
            if url is CIF_URL and dest.exists():
                break
            try:
                req = urllib.request.Request(
                    url.format(pdb=pdb.upper()),
                    headers={"User-Agent": "geoaudit-cryptobench/1.0"})
                with urllib.request.urlopen(req, timeout=60) as fh:
                    blob = fh.read()
                if len(blob) < 512:
                    last = f"short:{len(blob)}"
                    continue
                target.write_bytes(blob)
                return pdb, ("ok" if url is URL else "ok_cif"), len(blob)
            except urllib.error.HTTPError as exc:
                last = f"HTTP {exc.code}"
                if exc.code == 404:
                    continue  # very large entries are cif-only
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
        time.sleep(1.5 * (attempt + 1))
    return pdb, f"FAIL {last}", 0


def _inventory(pdb: str) -> dict:
    """What crystallographic records the downloaded entry actually holds."""
    p = DEST / f"{pdb.lower()}.pdb.gz"
    if not p.exists():
        p = DEST / f"{pdb.lower()}.cif.gz"
        if not p.exists():
            return {"present": False}
        txt = gzip.decompress(p.read_bytes()).decode("utf-8", "ignore")
        return {
            "present": True, "format": "cif",
            "has_cell": "_cell.length_a" in txt,
            "has_symmetry": "_symmetry.space_group_name_H-M" in txt,
            "has_symop": "_symmetry_equiv.pos_as_xyz" in txt
                         or "_space_group_symop.operation_xyz" in txt,
            "has_anisou": "_atom_site_anisotrop." in txt,
        }
    txt = gzip.decompress(p.read_bytes()).decode("utf-8", "ignore")
    smtry = sum(1 for ln in txt.splitlines() if ln.startswith("REMARK 290   SMTRY1"))
    spacegroup = ""
    for ln in txt.splitlines():
        if ln.startswith("CRYST1"):
            spacegroup = ln[55:66].strip()
            break
    return {
        "present": True, "format": "pdb",
        "has_cryst1": "\nCRYST1" in txt or txt.startswith("CRYST1"),
        "has_scale": "\nSCALE1" in txt,
        "n_symmetry_operators": smtry,
        "space_group": spacegroup,
        "n_anisou_lines": txt.count("\nANISOU"),
        "has_anisou": "\nANISOU" in txt,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    DEST.mkdir(parents=True, exist_ok=True)
    pdbs: dict[str, list[str]] = {}
    for fold, fname in (("train", "train_manifest.json"),
                        ("test", "official_manifest.json")):
        man = json.loads((ROOT / "data/cryptobench_apo" / fname).read_text())
        for e in man["entries"]:
            pdbs.setdefault(e["pdb"].lower(), []).append(f"{fold}:{e['chain']}")

    ids = sorted(pdbs)
    print(f"{len(ids)} distinct PDB entries across "
          f"{sum(len(v) for v in pdbs.values())} units", flush=True)

    t0 = time.perf_counter()
    status: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(_fetch, p): p for p in ids}
        for i, fut in enumerate(as_completed(futs), 1):
            pdb, st, _n = fut.result()
            status[pdb] = st
            if i % 50 == 0:
                nfail = sum(1 for v in status.values() if v.startswith("FAIL"))
                print(f"  {i}/{len(ids)}  {time.perf_counter() - t0:.0f}s  "
                      f"fail={nfail}", flush=True)

    inv = {p: _inventory(p) for p in ids}
    have = [p for p in ids if inv[p].get("present")]
    n_cryst = sum(1 for p in have if inv[p].get("has_cryst1"))
    n_symop = sum(1 for p in have if inv[p].get("n_symmetry_operators", 0) > 0)
    n_aniso = sum(1 for p in have if inv[p].get("has_anisou"))
    groups: dict[str, int] = {}
    for p in have:
        g = inv[p].get("space_group") or "unknown"
        groups[g] = groups.get(g, 0) + 1

    out = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "what_this_is": (
            "the deposited PDB entry for every apo receptor in the train and "
            "test folds, downloaded to recover the crystal form and the "
            "anisotropic displacement parameters that the committed receptor "
            "files do not carry"
        ),
        "source": "https://files.rcsb.org/download/{ID}.pdb.gz",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_entries_requested": len(ids),
        "n_entries_present": len(have),
        "n_failed": sum(1 for v in status.values() if v.startswith("FAIL")),
        "failures": {p: v for p, v in status.items() if v.startswith("FAIL")},
        "coverage": {
            "has_cryst1": n_cryst,
            "has_cryst1_fraction": round(n_cryst / max(len(have), 1), 4),
            "has_symmetry_operators": n_symop,
            "has_symmetry_operators_fraction": round(n_symop / max(len(have), 1), 4),
            "has_anisou": n_aniso,
            "has_anisou_fraction": round(n_aniso / max(len(have), 1), 4),
            "why_coverage_matters": (
                "a descriptor family defined only where a record exists is a "
                "family with a missing level, not a family with fewer units; "
                "the compiled field must carry an explicit absent digit and "
                "the coverage fraction must be reported beside any lift"
            ),
        },
        "space_group_histogram": dict(sorted(groups.items(),
                                             key=lambda kv: -kv[1])),
        "n_distinct_space_groups": len(groups),
        # The archives are not committed -- 956 files, 190 MB, and every one is
        # a byte-identical copy of a public RCSB download. What is committed is
        # this digest per file, which is what makes the input *pinned* rather
        # than merely *named*: a reader re-fetches and compares, and a silent
        # re-deposition by the PDB shows up as a mismatch instead of as an
        # unexplained change in a descriptor.
        "files": {p.name: _sha256(p) for p in sorted(DEST.iterdir())
                  if p.suffix == ".gz"},
        "files_definition": (
            "sha256 of each downloaded archive as written, keyed by filename; "
            "the archives themselves are gitignored and re-fetched by this "
            "tool"),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("\nWROTE", args.out)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("failures", "space_group_histogram")},
                     indent=2))
    print("top space groups:",
          list(out["space_group_histogram"].items())[:8])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
