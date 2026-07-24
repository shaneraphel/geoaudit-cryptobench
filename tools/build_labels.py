#!/usr/bin/env python3
"""Deterministically (re)build ESR1 receptor-only inputs and chain-scoped labels.

Fixes the label-merge defect: labels are built from a SINGLE chain-scoped ligand
instance (pdb_io.ligand_heavy_coords), not every crystallographic copy sharing a
resname. Raw PDBs are cached in iCloud (bulk, not committed) with SHA-256 pins;
receptor-only ATOM PDBs and label JSONs are written into the paper tree.

Usage:
  PYTHONPATH=src python3.12 tools/build_labels.py            # use cached raw
  PYTHONPATH=src python3.12 tools/build_labels.py --download  # fetch missing raw
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.request
from pathlib import Path

from pocket_bench.dataset.catalog import CURATED_ENTRIES
from pocket_bench.pdb_io import (
    binding_residues,
    ligand_centroid,
    ligand_heavy_coords,
    parse_pdb_atoms,
    sha256_file,
    write_receptor_only_pdb,
)

ROOT = Path(__file__).resolve().parents[1]
RECEPTOR_DIR = ROOT / "data/receptors"
LABEL_DIR = ROOT / "data/labels"
RAW_MANIFEST = ROOT / "data/manifests/RAW_PDB_ICLOUD.json"
RCSB_URL = "https://files.rcsb.org/download/{pdb}.pdb"


def _icloud_raw_dir() -> Path:
    base = Path(
        os.environ.get(
            "ER100_ICLOUD_DIR",
            Path.home() / "Library/Mobile Documents/com~apple~CloudDocs",
        )
    )
    d = base / "foliation-er100" / "raw_pdb"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(pdb: str, dest: Path, download: bool, attempts: int = 4) -> None:
    if dest.exists() and dest.stat().st_size > 5000:
        return
    if not download:
        raise FileNotFoundError(
            f"raw {pdb} missing at {dest}; rerun with --download"
        )
    url = RCSB_URL.format(pdb=pdb)
    last_err: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "er100-labels/1.0"}
            )
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
                data = resp.read()
            if len(data) < 5000:
                raise ValueError(f"{pdb} download too small ({len(data)} bytes)")
            dest.write_bytes(data)
            return
        except Exception as err:  # noqa: BLE001 (retry any transient read error)
            last_err = err
            print(f"  retry {pdb} ({attempt}/{attempts}): {type(err).__name__}")
    raise RuntimeError(f"failed to fetch {pdb} after {attempts} attempts: {last_err}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()

    raw_dir = _icloud_raw_dir()
    RECEPTOR_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)

    raw_manifest: dict[str, dict] = {
        "schema": "foliation.pocket_bench.raw_pdb_icloud.v1",
        "note": "Raw PDBs are bulk data cached in iCloud, not committed. "
        "Receptor-only ATOM PDBs and labels ARE committed.",
        "source": "RCSB",
        "entries": {},
    }
    summary: list[dict] = []

    for e in CURATED_ENTRIES:
        pdb = e["pdb_id"]
        chain = e["chain"]
        raw = raw_dir / f"{pdb}.pdb"
        _fetch(pdb, raw, args.download)
        atoms = parse_pdb_atoms(raw.read_text())

        rec_path = RECEPTOR_DIR / f"{pdb}_{chain}_receptor.pdb"
        write_receptor_only_pdb(atoms, rec_path, chain=chain)

        rec = {
            "pdb_id": pdb,
            "chain": chain,
            "genotype": e["genotype"],
            "state": e["state"],
            "raw_sha256": sha256_file(raw),
            "raw_url": RCSB_URL.format(pdb=pdb),
            "receptor_sha256": sha256_file(rec_path),
        }

        resname = e.get("ligand_resname")
        if resname:
            coords = ligand_heavy_coords(atoms, resname, chain=chain)
            centroid = ligand_centroid(coords)
            residues = binding_residues(atoms, coords, chain=chain)
            label = {
                "schema": "foliation.pocket_bench.label.v1",
                "clinical_grade": False,
                "pdb_id": pdb,
                "chain": chain,
                "ligand_resname": resname,
                "ligand_heavy_coords": coords,
                "ligand_centroid": centroid,
                "binding_residues": residues,
                "truth_boundary": "Ligand coordinates are EVALUATION LABELS "
                "only. Forbidden as Foliation / baseline pocket-finder inputs.",
            }
            lab_path = LABEL_DIR / f"{pdb}_{chain}_labels.json"
            lab_path.write_text(json.dumps(label, indent=2) + "\n")
            rec["ligand_resname"] = resname
            rec["label_n_heavy"] = len(coords)
            rec["label_sha256"] = sha256_file(lab_path)
            summary.append({"pdb": pdb, "resname": resname, "n_heavy": len(coords)})
        else:
            rec["ligand_resname"] = None
            summary.append({"pdb": pdb, "resname": None, "n_heavy": 0})

        raw_manifest["entries"][pdb] = rec

    RAW_MANIFEST.write_text(json.dumps(raw_manifest, indent=2) + "\n")
    print(json.dumps({"rebuilt": summary, "raw_dir": str(raw_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
