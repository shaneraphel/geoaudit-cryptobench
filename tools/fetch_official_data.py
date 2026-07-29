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
import http.client
import json
import re
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
# Accept-Encoding: identity avoids a gzip/content-length mismatch through the
# upstream CDN proxy that truncates the stream (observed IncompleteRead on the OSF
# API); Connection: close keeps each fetch on a fresh socket.
_UA = {"User-Agent": "geoaudit-fetch/1.0",
       "Accept-Encoding": "identity",
       "Connection": "close"}
_TIMEOUT = 120


# --- transport (stdlib only) ------------------------------------------------- #
def _get(url: str, *, binary: bool, retries: int = 4) -> bytes | dict:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                data = r.read()
            return data if binary else json.loads(data.decode("utf-8"))
        except http.client.IncompleteRead as exc:
            # retry from scratch; a partial body is never returned as if complete
            last = exc
            time.sleep(1.5 * (attempt + 1))
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


# --- manifest assembly (official CryptoBench schema) ------------------------- #
# CryptoBench folds/test.json is keyed by APO pdb id -> [record...]; each record
# carries {uniprot_id, apo_chain, apo_pocket_selection:["<chain>_<resnum>", ...]}.
# The cryptic label for the apo structure is the union of apo_pocket_selection
# residues on apo_chain; the sequence cluster (for split-disjointness) is uniprot_id.
_SEL_RE = re.compile(r"^([A-Za-z0-9]+)_(-?\d+)")


def _parse_selection(tok: str) -> tuple[str, int] | None:
    """'A_258' -> ('A', 258); tolerate icodes ('A_258A' -> 258). None if unparsable."""
    m = _SEL_RE.match(str(tok).strip())
    if not m:
        return None
    return m.group(1), int(m.group(2))


def _fold_units(fold_obj: dict) -> tuple[dict[tuple[str, str], dict], list[dict]]:
    """Group the fold file into (apo_pdb, apo_chain) units with residues + uniprot.

    Returns ``(single_chain_units, excluded_multichain)``. CryptoBench records a
    compound ``apo_chain`` (e.g. ``"M-O-P"``) when the cryptic pocket spans a
    multi-chain assembly. The per-residue metric pipeline keys residues by integer
    resseq (chain-agnostic), so a multi-chain unit would yield ambiguous labels;
    such units are EXCLUDED and reported (never silently dropped, never merged).
    Refuses (raises) if a unit has no cluster surrogate (uniprot_id).
    """
    if not isinstance(fold_obj, dict):
        raise RuntimeError("CryptoBench fold file must be a dict keyed by apo pdb id")
    units: dict[tuple[str, str], dict] = {}
    excluded: dict[tuple[str, str], str] = {}
    for pdb, records in fold_obj.items():
        if not isinstance(records, list):
            raise RuntimeError(f"fold entry '{pdb}' is not a record list")
        for rec in records:
            chain = str(rec.get("apo_chain") or "").strip()
            uni = str(rec.get("uniprot_id") or "").strip()
            sel = rec.get("apo_pocket_selection") or []
            if not chain:
                raise RuntimeError(f"{pdb}: record without apo_chain")
            if not uni:
                raise RuntimeError(
                    f"{pdb}/{chain}: no uniprot_id (no cluster surrogate; refusing)"
                )
            if "-" in chain:  # compound multi-chain assembly
                excluded[(str(pdb).lower(), chain)] = uni
                continue
            key = (str(pdb).lower(), chain)
            u = units.setdefault(key, {"uniprot": uni, "residues": set()})
            for tok in sel:
                parsed = _parse_selection(tok)
                if parsed and parsed[0] == chain:
                    u["residues"].add(parsed[1])
    single = {k: v for k, v in units.items() if v["residues"]}
    excl = [{"pdb": p, "apo_chain": c, "cluster_id": u,
             "reason": "multi_chain_assembly_apo_chain_chain_agnostic_resseq_ambiguous"}
            for (p, c), u in sorted(excluded.items())]
    return single, excl


def build_manifest(fold_file: Path, labels_file: Path | None = None,
                   *, limit: int = 0) -> Path:
    """Assemble official_manifest.json from the CryptoBench test-fold file.

    Labels are taken from the fold file's apo_pocket_selection (labels_file, if
    given, must be byte-identical or a superset key set; it is recorded for
    provenance). Receptors are fetched from RCSB and written chain-scoped,
    ligand-stripped. RCSB fetch failures are collected and reported, never silently
    imputed.
    """
    fold_obj = json.loads(fold_file.read_text())
    fold_sha = sha256_bytes(fold_file.read_bytes())
    labels_sha = sha256_bytes(labels_file.read_bytes()) if labels_file else fold_sha
    units, excluded_multichain = _fold_units(fold_obj)
    keys = sorted(units)
    if limit:
        keys = keys[:limit]

    entries: list[dict] = []
    skipped: list[dict] = []
    for i, (pdb, chain) in enumerate(keys, 1):
        u = units[(pdb, chain)]
        resids = sorted(u["residues"])
        rec_path = RECEPTOR_DIR / f"{pdb}_{chain}_receptor.pdb"
        try:
            if rec_path.is_file() and rec_path.stat().st_size > 0:
                # resume: reuse the already-fetched, chain-scoped receptor on disk
                rec_sha = sha256_bytes(rec_path.read_bytes())
            else:
                raw = _get(RCSB.format(pdb.upper()), binary=True)
                atoms = parse_pdb_atoms(raw.decode("utf-8", "ignore"))
                if not any(a["record"] == "ATOM" and a["chain"] == chain for a in atoms):
                    raise RuntimeError(f"chain {chain} absent in RCSB {pdb}")
                write_receptor_only_pdb(atoms, rec_path, chain=chain)
                rec_sha = sha256_bytes(rec_path.read_bytes())
        except Exception as exc:  # noqa: BLE001 -- record + continue, never impute
            skipped.append({"pdb": pdb, "chain": chain, "reason": str(exc)})
            print(f"  [{i}/{len(keys)}] SKIP {pdb}_{chain}: {exc}", file=sys.stderr)
            continue
        lab = {"schema": "cryptobench.official_label.v1", "clinical_grade": False,
               "pdb_id": pdb, "chain": chain, "cryptic_residues": resids,
               "binding_residues": resids,  # alias for core residue_f1 (keys on this)
               "labels_source_sha256": labels_sha}
        lab_path = LABEL_DIR / f"{pdb}_{chain}_labels.json"
        lab_path.parent.mkdir(parents=True, exist_ok=True)
        lab_path.write_text(json.dumps(lab, indent=2) + "\n")
        entries.append({
            "pdb": pdb, "chain": chain, "cluster_id": u["uniprot"],
            "receptor_path": str(rec_path.relative_to(ROOT)),
            "receptor_sha256": rec_sha,
            "label_path": str(lab_path.relative_to(ROOT)),
            "label_sha256": sha256_bytes(lab_path.read_bytes()),
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(keys)}] built {pdb}_{chain} "
                  f"({len(resids)} cryptic residues)", file=sys.stderr)
    manifest = {
        "schema": "cryptobench.official_test_fold.v1",
        "clinical_grade": False,
        "fold": "test",
        "clustering": {"method": "mmseqs2", "sequence_identity_threshold": 0.10,
                       "coverage": 0.8,
                       "cluster_id_semantics": "uniprot_id sequence-cluster surrogate; "
                       "test/train disjointness is defined by folds.json"},
        "source_url": f"https://osf.io/{OSF_NODE}/",
        "fold_file_sha256": fold_sha,
        "labels_source_sha256": labels_sha,
        "cryptobench_test_apo_pdbs": 222,
        "n_fold_units": len(units),
        "n_excluded_multichain": len(excluded_multichain),
        "excluded_multichain": excluded_multichain,
        "n_entries": len(entries),
        "n_skipped": len(skipped),
        "skipped": skipped,
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
                    help="download OSF files (verified) into _osf/, by file name "
                         "or by full OSF path when the name is not unique")
    ap.add_argument("--build-manifest", action="store_true",
                    help="assemble official_manifest.json from downloaded fold data")
    ap.add_argument("--fold-file", type=Path,
                    help="path to the downloaded official test-fold definition")
    ap.add_argument("--labels-file", type=Path,
                    help="optional label file for provenance (defaults to fold file)")
    ap.add_argument("--limit", type=int, default=0,
                    help="build only the first N fold units (0 = all; smoke tests)")
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
        files = osf_walk()
        for name in args.fetch:
            hits = [f for f in files
                    if f["name"] == name or (f["path"] or "").lstrip("/") == name
                    or f["path"] == name]
            if not hits:
                raise SystemExit(f"'{name}' not found on OSF (run --list)")
            # The deposit publishes the trained and the untrained network under
            # the same file names in different folders, so a basename alone can
            # name two different models. Fetching the wrong one would produce a
            # baseline that runs and is silently untrained.
            if len(hits) > 1:
                paths = "\n  ".join(sorted(str(f["path"]) for f in hits))
                raise SystemExit(
                    f"'{name}' names {len(hits)} files on OSF; give the full "
                    f"path instead:\n  {paths}")
            # The file lands where it was named: a bare name stays flat in _osf/,
            # a full OSF path keeps that path underneath it.
            dest = OSF_DIR / name.lstrip("/")
            got = download_verify(hits[0], dest)
            print(f"OK  {got}  -> {dest.relative_to(ROOT)}")

    if args.build_manifest:
        if not args.fold_file:
            raise SystemExit("--build-manifest requires --fold-file")
        if not args.fold_file.is_file():
            raise SystemExit(f"missing input file: {args.fold_file}")
        if args.labels_file and not args.labels_file.is_file():
            raise SystemExit(f"missing input file: {args.labels_file}")
        out = build_manifest(args.fold_file, args.labels_file, limit=args.limit)
        m = json.loads(out.read_text())
        print(f"wrote {out.relative_to(ROOT)}: {m['n_entries']} entries, "
              f"{m['n_skipped']} skipped of {m['n_fold_units']} units; "
              "validate with adapters.load_official_test_fold()")
    return 0


if __name__ == "__main__":
    sys.exit(main())
