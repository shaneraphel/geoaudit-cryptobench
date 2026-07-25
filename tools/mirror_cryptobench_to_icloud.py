#!/usr/bin/env python3
"""Mirror a pinned CryptoBench-apo subset to iCloud and build receptor+labels.

Source of truth (real, non-synthetic):
- Cryptic-residue labels: skrhakv/CryptoBench `label_dataset.json`
  (dict: apo_pdb_id -> ["<chain>_<resseq>", ...]), Škrhák et al., Bioinformatics
  2025, doi:10.1093/bioinformatics/btae745. Dataset: https://osf.io/pz4a9/ .
- Apo structures: RCSB `https://files.rcsb.org/download/<PDBID>.pdb`.

We select a DETERMINISTIC subset (fixed stride over sorted keys) so the sample is
reproducible and disclosed, not cherry-picked. Raw apo PDBs are written to iCloud;
receptor-only PDBs, chain-scoped cryptic-residue labels, and a SHA-256 manifest
are written under the repo (data/cryptobench_apo/). No scores here.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/cryptobench_apo"
RECEPTORS = OUT / "receptors"
LABELS = OUT / "labels"
ICLOUD = (
    Path.home()
    / "Library/Mobile Documents/com~apple~CloudDocs"
    / "Foliation-Engine-Archive/cryptobench_apo"
)
RAW = ICLOUD / "raw_pdb"
LABELS_URL = (
    "https://raw.githubusercontent.com/skrhakv/CryptoBench/master/"
    "src/F-statistics/conservation/label_dataset.json"
)
N_TARGET = 15


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fetch(url: str, tries: int = 4, timeout: int = 120) -> bytes:
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"fetch failed {url}: {last}")


def parse_atoms(text: str) -> list[dict]:
    atoms = []
    for ln in text.splitlines():
        rec = ln[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        try:
            atoms.append(
                {
                    "record": rec,
                    "name": ln[12:16].strip(),
                    "resname": ln[17:20].strip(),
                    "chain": ln[21:22].strip(),
                    "resseq": int(ln[22:26]),
                    "x": float(ln[30:38]),
                    "y": float(ln[38:46]),
                    "z": float(ln[46:54]),
                    "element": (ln[76:78].strip() or ln[12:16].strip()[:1]),
                    "line": ln,
                }
            )
        except ValueError:
            continue
    return atoms


def main() -> int:
    for d in (RECEPTORS, LABELS, RAW):
        d.mkdir(parents=True, exist_ok=True)

    labels_raw = fetch(LABELS_URL)
    (ICLOUD / "label_dataset.json").write_bytes(labels_raw)
    labels = json.loads(labels_raw)
    keys = sorted(labels)
    stride = max(1, len(keys) // (N_TARGET * 2))
    picks = keys[::stride]  # oversample; keep first N that succeed

    manifest = {
        "schema": "gf4cc.cryptobench_apo.manifest.v1",
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {
            "labels": {
                "url": LABELS_URL,
                "sha256": sha256_bytes(labels_raw),
                "bytes": len(labels_raw),
                "citation": "Skrhak et al., Bioinformatics 2025, "
                "doi:10.1093/bioinformatics/btae745; dataset https://osf.io/pz4a9/",
            },
            "structures": "RCSB https://files.rcsb.org/download/<PDBID>.pdb",
        },
        "selection": {
            "rule": "deterministic stride over sorted label_dataset.json keys",
            "stride": stride,
            "n_target": N_TARGET,
        },
        "entries": [],
        "skipped": [],
    }

    got = 0
    for pdb in picks:
        if got >= N_TARGET:
            break
        residues = labels[pdb]
        chains = {r.split("_")[0] for r in residues}
        chain = sorted(chains)[0]  # primary labeled chain
        want = {
            int(r.split("_")[1])
            for r in residues
            if r.split("_")[0] == chain and r.split("_")[1].lstrip("-").isdigit()
        }
        try:
            raw = fetch(f"https://files.rcsb.org/download/{pdb.upper()}.pdb")
        except Exception as exc:  # noqa: BLE001
            manifest["skipped"].append({"pdb": pdb, "reason": f"download:{exc}"[:120]})
            continue
        atoms = parse_atoms(raw.decode(errors="ignore"))
        rec_lines = [
            a["line"]
            for a in atoms
            if a["record"] == "ATOM" and a["chain"] == chain and a["element"] != "H"
        ]
        lab_coords = [
            [a["x"], a["y"], a["z"]]
            for a in atoms
            if a["record"] == "ATOM"
            and a["chain"] == chain
            and a["resseq"] in want
            and a["element"] != "H"
        ]
        if len(rec_lines) < 100 or len(lab_coords) < 5:
            manifest["skipped"].append(
                {"pdb": pdb, "reason": f"rec={len(rec_lines)} lab={len(lab_coords)}"}
            )
            continue

        (RAW / f"{pdb}.pdb").write_bytes(raw)
        rec_text = "".join(l + "\n" for l in rec_lines) + "END\n"
        rec_path = RECEPTORS / f"{pdb}_{chain}_receptor.pdb"
        rec_path.write_text(rec_text)
        centroid = [sum(c[i] for c in lab_coords) / len(lab_coords) for i in range(3)]
        label = {
            "pdb_id": pdb,
            "chain": chain,
            "cryptic_residues": sorted(want),
            "n_label_atoms": len(lab_coords),
            "label_centroid": centroid,
            "ligand_centroid": centroid,  # scorer key (DCC to cryptic-site centroid)
            "ligand_heavy_coords": lab_coords,  # scored as DCA to cryptic-site atoms
            "source": "CryptoBench label_dataset.json (cryptic binding residues)",
            "clinical_grade": False,
        }
        (LABELS / f"{pdb}_{chain}_labels.json").write_text(json.dumps(label, indent=2))
        manifest["entries"].append(
            {
                "pdb": pdb,
                "chain": chain,
                "raw_sha256": sha256_bytes(raw),
                "raw_bytes": len(raw),
                "receptor_sha256": sha256_bytes(rec_text.encode()),
                "receptor_atoms": len(rec_lines),
                "n_cryptic_residues": len(want),
                "n_label_atoms": len(lab_coords),
            }
        )
        got += 1
        print(f"  {pdb} chain {chain}: rec={len(rec_lines)} label_atoms={len(lab_coords)}")

    (OUT / "PREDICTION_INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nmirrored {got}/{N_TARGET} apo structures; raw in iCloud: {RAW}")
    print(f"manifest: {OUT / 'PREDICTION_INPUT_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
