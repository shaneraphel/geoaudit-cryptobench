#!/usr/bin/env python3
"""Fetch the official CryptoBench data from OSF (node pz4a9) and assemble the
manifest that ``adapters.load_official_test_fold()`` requires. Fail-closed.

Honesty contract (do not weaken):
* This script does NOT hard-code file GUIDs, download URLs, or SHA-256 hashes that
  it cannot obtain from the source. It discovers files via the public OSF API and
  verifies every download against the SHA-256 that OSF itself reports for that file
  (``attributes.extra.hashes.sha256``). If OSF does not report a sha256 for a file,
  the download is refused — never trusted unverified.
* It does not invent the official test-fold membership. The fold definition is read
  from the file you select on OSF (``--fold-file``); its per-structure cluster
  assignments are copied verbatim. If that file lacks explicit cluster ids, manifest
  assembly refuses (cluster-disjointness cannot be asserted from nothing).
* `clinical_grade=false`.

Typical use:
    # 1. see exactly what is published (names + sizes + OSF sha256):
    PYTHONPATH=src python3.12 tools/fetch_official_data.py --list
    # 2. download named artifacts (verified) into data/cryptobench_apo/_osf/:
    PYTHONPATH=src python3.12 tools/fetch_official_data.py --fetch <name> [<name> ...]
    # 3. build the loader manifest from the downloaded fold-definition file:
    PYTHONPATH=src python3.12 tools/fetch_official_data.py \
        --build-manifest --fold-file <downloaded_fold.json> \
        --labels-file <downloaded_label_dataset.json>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from pocket_bench.pdb_io import parse_pdb_atoms, write_receptor_only_pdb

ROOT = Path(__file__).resolve().parents[1]
OSF_NODE = "pz4a9"
OSF_API = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/"
RCSB = "https://files.rcsb.org/download/{}.pdb"
OSF_DIR = ROOT / "data/cryptobench_apo/_osf"
RECEPTOR_DIR = ROOT / "data/cryptobench_apo/official_receptors"
LABEL_DIR = ROOT / "data/cryptobench_apo/official_labels"
MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
_UA = {"User-Agent": "geoaudit-fetch/1.0"}
_TIMEOUT = 60


# --- transport (stdlib only) ------------------------------------------------- #
def _get(url: str, *, binary: bool, retries: int = 3) -> bytes | dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                data = r.read()
            return data if binary else json.loads(data.decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed after {retries} tries: {url} ({last})")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# --- OSF discovery ----------------------------------------------------------- #
def osf_walk(url: str = OSF_API) -> list[dict]:
    """Recursively list osfstorage files with name, size, sha256, download url."""
    out: list[dict] = []
    stack = [url]
    while stack:
        page = stack.pop()
        while page:
            payload = _get(page, binary=False)
            for item in payload.get("data", []):
                attr = item.get("attributes", {})
                kind = attr.get("kind")
                if kind == "folder":
                    rel = (item.get("relationships", {}).get("files", {})
                           .get("links", {}).get("related", {}).get("href"))
                    if rel:
                        stack.append(rel)
                elif kind == "file":
                    out.append({
                        "name": attr.get("name"),
                        "path": attr.get("materialized_path") or attr.get("name"),
                        "size": attr.get("size"),
                        "sha256": (attr.get("extra", {}).get("hashes", {})
                                   .get("sha256")),
                        "download": item.get("links", {}).get("download"),
                    })
            page = payload.get("links", {}).get("next")
    return out


def download_verify(entry: dict, dest: Path) -> str:
    """Download an OSF file and verify against the sha256 OSF reports for it."""
    expected = entry.get("sha256")
    if not expected:
        raise RuntimeError(
            f"refusing to download '{entry['name']}': OSF reports no sha256 "
            "(cannot verify integrity)"
        )
    if not entry.get("download"):
        raise RuntimeError(f"no download link for '{entry['name']}'")
    blob = _get(entry["download"], binary=True)
    got = sha256_bytes(blob)
    if got != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for '{entry['name']}': OSF {expected}, got {got}"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    return got


# --- manifest assembly ------------------------------------------------------- #
def _fold_entries(fold_obj: object) -> list[dict]:
    """Normalize a fold-definition file into [{pdb, chain, cluster_id}].

    Accepts either a list of records or a mapping keyed by pdb. Each record MUST
    carry a cluster id under one of {cluster_id, cluster, mmseqs_cluster}; if none
    is present, we refuse (cluster-disjointness cannot be asserted without it).
    """
    def one(pdb: str, rec: dict) -> dict:
        cid = rec.get("cluster_id") or rec.get("cluster") or rec.get("mmseqs_cluster")
        if cid is None:
            raise RuntimeError(
                f"fold entry '{pdb}' has no cluster id; cannot assert "
                "cluster-disjointness. Inspect the OSF fold file and re-run."
            )
        return {"pdb": str(pdb).lower(),
                "chain": str(rec.get("chain") or rec.get("apo_chain") or "A"),
                "cluster_id": str(cid)}

    rows: list[dict] = []
    if isinstance(fold_obj, list):
        for rec in fold_obj:
            if not isinstance(rec, dict) or "pdb" not in rec:
                raise RuntimeError("fold list entries must be dicts with a 'pdb' key")
            rows.append(one(rec["pdb"], rec))
    elif isinstance(fold_obj, dict):
        for pdb, rec in fold_obj.items():
            rows.append(one(pdb, rec if isinstance(rec, dict) else {}))
    else:
        raise RuntimeError("unrecognized fold-file structure (need list or dict)")
    return rows


def _cryptic_residues(labels_obj: dict, pdb: str, chain: str) -> list[int]:
    """Pull cryptic-residue numbers for (pdb, chain) from a CryptoBench label file."""
    key_candidates = [pdb, pdb.upper(), pdb.lower(), f"{pdb}_{chain}"]
    rec = None
    for k in key_candidates:
        if k in labels_obj:
            rec = labels_obj[k]
            break
    if rec is None:
        raise RuntimeError(f"no label record for {pdb} in labels file")
    resids: list[int] = []
    # tolerate a few common shapes; refuse if none match
    if isinstance(rec, dict) and "residues" in rec:
        resids = [int(r) for r in rec["residues"]]
    elif isinstance(rec, list):
        for r in rec:
            if isinstance(r, dict) and "residue" in r:
                resids.append(int(r["residue"]))
            elif isinstance(r, (int, str)):
                resids.append(int(r))
    if not resids:
        raise RuntimeError(f"could not parse cryptic residues for {pdb}")
    return sorted(set(resids))


def build_manifest(fold_file: Path, labels_file: Path) -> Path:
    fold = _fold_entries(json.loads(fold_file.read_text()))
    labels = json.loads(labels_file.read_text())
    labels_sha = sha256_bytes(labels_file.read_bytes())
    entries = []
    for row in fold:
        pdb, chain, cid = row["pdb"], row["chain"], row["cluster_id"]
        raw = _get(RCSB.format(pdb.upper()), binary=True)
        atoms = parse_pdb_atoms(raw.decode("utf-8", "ignore"))
        rec_path = RECEPTOR_DIR / f"{pdb}_{chain}_receptor.pdb"
        write_receptor_only_pdb(atoms, rec_path, chain=chain)
        rec_sha = sha256_bytes(rec_path.read_bytes())
        resids = _cryptic_residues(labels, pdb, chain)
        lab = {"schema": "cryptobench.official_label.v1", "clinical_grade": False,
               "pdb_id": pdb, "chain": chain, "cryptic_residues": resids,
               "labels_source_sha256": labels_sha}
        lab_path = LABEL_DIR / f"{pdb}_{chain}_labels.json"
        lab_path.parent.mkdir(parents=True, exist_ok=True)
        lab_path.write_text(json.dumps(lab, indent=2) + "\n")
        entries.append({
            "pdb": pdb, "chain": chain, "cluster_id": cid,
            "receptor_path": str(rec_path.relative_to(ROOT)),
            "receptor_sha256": rec_sha,
            "label_path": str(lab_path.relative_to(ROOT)),
            "label_sha256": sha256_bytes(lab_path.read_bytes()),
        })
    manifest = {
        "schema": "cryptobench.official_test_fold.v1",
        "clinical_grade": False,
        "fold": "test",
        "clustering": {"method": "mmseqs2", "sequence_identity_threshold": 0.10,
                       "coverage": 0.8},
        "source_url": f"https://osf.io/{OSF_NODE}/",
        "labels_source_sha256": labels_sha,
        "n_entries": len(entries),
        "entries": entries,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    return MANIFEST


# --- CLI --------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true",
                    help="list OSF pz4a9 files (name, size, sha256)")
    ap.add_argument("--fetch", nargs="+", metavar="NAME",
                    help="download named OSF files (verified) into _osf/")
    ap.add_argument("--build-manifest", action="store_true",
                    help="assemble official_manifest.json from downloaded fold data")
    ap.add_argument("--fold-file", type=Path,
                    help="path to the downloaded official test-fold definition")
    ap.add_argument("--labels-file", type=Path,
                    help="path to the downloaded CryptoBench label file")
    args = ap.parse_args()

    if not any([args.list, args.fetch, args.build_manifest]):
        ap.print_help()
        return 2

    if args.list:
        files = osf_walk()
        for f in sorted(files, key=lambda x: x["path"] or ""):
            print(f"{(f['sha256'] or 'NO-SHA256'):64s}  "
                  f"{(f['size'] or 0):>10}  {f['path']}")
        print(f"# {len(files)} files on OSF node {OSF_NODE}")

    if args.fetch:
        files = {f["name"]: f for f in osf_walk()}
        for name in args.fetch:
            if name not in files:
                raise SystemExit(f"'{name}' not found on OSF (run --list)")
            dest = OSF_DIR / name
            got = download_verify(files[name], dest)
            print(f"OK  {got}  -> {dest.relative_to(ROOT)}")

    if args.build_manifest:
        if not args.fold_file or not args.labels_file:
            raise SystemExit("--build-manifest requires --fold-file and --labels-file")
        for f in (args.fold_file, args.labels_file):
            if not f.is_file():
                raise SystemExit(f"missing input file: {f}")
        out = build_manifest(args.fold_file, args.labels_file)
        print(f"wrote {out.relative_to(ROOT)}; validate with "
              "adapters.load_official_test_fold()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
