#!/usr/bin/env python3
"""Refresh prediction-input + smoke manifests and structure provenance.

Fixes stale paths (data/pocket_bench/esr1/receptors -> data/receptors),
refreshes receptor SHA-256 from the actual committed files, pins the exact
baseline tool environment/commands, and writes authoritative RCSB provenance
(full SHA-256, byte count, retrieval date) for non-ESR1 structures the paper
references (KRAS G12D 9BL0, hERG 5VA1).
"""
from __future__ import annotations

import json
from pathlib import Path

from pocket_bench.dataset.catalog import CURATED_ENTRIES
from pocket_bench.pdb_io import sha256_file

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "data/receptors"
MAN = ROOT / "data/manifests"

# Authoritative RCSB provenance, retrieved 2026-07-25. RCSB re-versions
# coordinate files, so we pin the exact SHA-256 + byte count we used.
STRUCTURE_PROVENANCE = {
    "schema": "gf4cc.structure_provenance.v1",
    "clinical_grade": False,
    "source": "RCSB",
    "retrieved": "2026-07-25",
    "note": "RCSB updates/re-versions coordinate files over time; these SHA-256 "
    "digests pin the exact files used here. Mismatch with a later RCSB fetch "
    "means RCSB re-released the entry, not that our record is wrong.",
    "entries": {
        "9BL0": {
            "role": "KRAS G12D noncovalent primary docking geometry",
            "url": "https://files.rcsb.org/download/9BL0.pdb",
            "sha256": "719c4a6c728a85c577cf61d87786934f49068e90bf1c1b3e255b65ba9b508fe0",
            "bytes": 289737,
        },
        "5VA1": {
            "role": "hERG cryo-EM (cardiotoxicity geometric-compatibility probe)",
            "url": "https://files.rcsb.org/download/5VA1.pdb",
            "sha256": "946f89bb727f063d07e6df1b15d7130a914597d6524cd364e3fdeea83fc7f472",
            "bytes": 402732,
        },
    },
}

BASELINE_ENV = {
    "schema": "gf4cc.baseline_env.v1",
    "clinical_grade": False,
    "python": "3.12",
    "tools": {
        "fpocket": {"version": "4.0 (git tag 4.2.3)", "build": "from source, Discngine/fpocket"},
        "p2rank": {"version": "2.5.1", "java": "OpenJDK 17"},
        "deeppocket": {"version": None, "status": "TOOL_UNAVAILABLE"},
    },
    "commands": {
        "labels": "PYTHONPATH=src python3.12 tools/build_labels.py --download",
        "benchmark": "PYTHONPATH=src python3.12 tools/run_pilot.py --split all",
    },
    "report": "results/pilot/REGENERATED_PILOT_REPORT.json",
    "note": "Prior P2Rank 6/6 was on merged (buggy) labels and is invalidated. "
    "Current numbers are on corrected chain-scoped labels; splits are NOT "
    "cluster-disjoint, so no comparative-superiority claim is made.",
}


def cluster_of(pdb: str, ledger: dict) -> str | None:
    for a in ledger.get("assignments") or []:
        if a.get("pdb_id") == pdb:
            return a.get("cluster_id")
    return None


def main() -> int:
    ledger = json.loads((MAN / "STRUCTURE_CLUSTER_LEDGER.json").read_text())
    entries = []
    for e in CURATED_ENTRIES:
        pdb, chain = e["pdb_id"], e["chain"]
        rec = REC / f"{pdb}_{chain}_receptor.pdb"
        entries.append(
            {
                "pdb_id": pdb,
                "genotype": e["genotype"],
                "split": e["split"],
                "structure_cluster_id": cluster_of(pdb, ledger),
                "receptor_pdb": f"data/receptors/{pdb}_{chain}_receptor.pdb",
                "receptor_sha256": sha256_file(rec) if rec.exists() else None,
            }
        )
    pim = {
        "schema": "foliation.pocket_bench.prediction_inputs.v1",
        "clinical_grade": False,
        "ligand_fields_present": False,
        "n_entries": len(entries),
        "entries": entries,
    }
    (MAN / "PREDICTION_INPUT_MANIFEST.json").write_text(json.dumps(pim, indent=2) + "\n")

    smoke = {
        "schema": "foliation.pocket_bench.smoke.v1",
        "clinical_grade": False,
        "superseded_by": "results/pilot/REGENERATED_PILOT_REPORT.json",
        "note": "Full benchmark (all 14 labeled structures, all methods) is in "
        "the regenerated report; this smoke file only records the corrected "
        "single-structure path and baseline environment.",
        "pdb_id": "3ERT",
        "receptor_pdb": "data/receptors/3ERT_A_receptor.pdb",
        "baseline_env": BASELINE_ENV,
    }
    (ROOT / "results/pilot/SMOKE_BASELINES.json").write_text(json.dumps(smoke, indent=2) + "\n")

    (MAN / "STRUCTURE_PROVENANCE.json").write_text(json.dumps(STRUCTURE_PROVENANCE, indent=2) + "\n")
    (MAN / "BASELINE_ENV.json").write_text(json.dumps(BASELINE_ENV, indent=2) + "\n")
    print(json.dumps({"prediction_inputs": len(entries), "provenance": list(STRUCTURE_PROVENANCE["entries"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
