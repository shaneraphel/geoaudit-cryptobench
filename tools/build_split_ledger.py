#!/usr/bin/env python3
"""Build the authoritative SPLIT_LEDGER.json (deterministic, reproducible).

Two benchmark groups, each with explicit cluster/split assignments:

* ``esr1_receptor_only_pilot`` — the Appendix A holo pilot. It is honestly NOT
  cluster-disjoint (one conserved ESR1 LBD fold spans all splits), so it declares
  ``cluster_disjoint_required=false`` + ``split_integrity_passed=false`` + a
  reason. The verifier forbids comparative-superiority claims on such a group.
* ``cryptobench_apo`` — the out-of-distribution apo/cryptic set. It declares
  ``cluster_disjoint_required=true``; the CI gate then fails closed if any
  ``cluster_id`` appears in more than one split.

Run: PYTHONPATH=src python3.12 tools/build_split_ledger.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLUSTER_LEDGER = ROOT / "data/manifests/STRUCTURE_CLUSTER_LEDGER.json"
APO_LABELS = ROOT / "data/cryptobench_apo/labels"
OUT = ROOT / "data/manifests/SPLIT_LEDGER.json"


def _esr1_group() -> dict:
    led = json.loads(CLUSTER_LEDGER.read_text())
    assignments = [
        {
            "pdb_id": a["pdb_id"],
            "split": a["split"],
            "cluster_id": a["cluster_id"],
        }
        for a in led.get("assignments", [])
    ]
    return {
        "cluster_disjoint_required": False,
        "split_integrity_passed": bool(led.get("split_integrity_passed", False)),
        "reason_not_disjoint": (
            "Single conserved ESR1 LBD fold spans development/validation/"
            "locked_test; comparative-superiority claims are disallowed for this "
            "group (verify_claims enforces comparative_claim_allowed=false)."
        ),
        "cluster_source": (
            "STRUCTURE_CLUSTER_LEDGER.json (Kabsch Cα RMSD <=1.5A + shared "
            "primary-citation DOI, connected components)."
        ),
        "assignments": assignments,
    }


def _cryptobench_group() -> dict:
    rows = []
    for f in sorted(APO_LABELS.glob("*_labels.json")):
        lab = json.loads(f.read_text())
        pdb = str(lab["pdb_id"]).lower()
        rows.append(
            {
                "pdb_id": pdb,
                "chain": lab.get("chain"),
                "split": "test",
                # Conservative singleton clustering: each distinct apo protein is
                # its own cluster. Singletons cannot overlap across splits, so this
                # can never *hide* leakage. Real MMseqs2 @10% seq-id is TODO.
                "cluster_id": f"cbapo-{pdb}",
            }
        )
    return {
        "cluster_disjoint_required": True,
        "is_official_cryptobench_test_fold": False,
        "cluster_source": (
            "singleton_pending_mmseqs2 — each distinct apo protein is its own "
            "cluster; conservative (cannot hide leakage). Replace with MMseqs2 "
            "@10% sequence identity to match the CryptoBench cluster-disjoint "
            "test fold before any comparative claim."
        ),
        "assignments": rows,
    }


def build() -> dict:
    return {
        "schema": "geoaudit.split_ledger.v1",
        "clinical_grade": False,
        "roles": {
            "development": "threshold / hyperparameter tuning ONLY",
            "validation": "model / variant selection ONLY",
            "test": "cluster-disjoint where required; evaluated once; frozen",
        },
        "clustering": {
            "target_seq_id_threshold": 0.10,
            "coverage": 0.8,
            "note": "CryptoBench test folds are cluster-disjoint at 10% identity.",
        },
        "invariants": {
            "disjoint_required_groups_are_cluster_disjoint": True,
            "non_disjoint_groups_must_declare_split_integrity_false": True,
        },
        "groups": {
            "esr1_receptor_only_pilot": _esr1_group(),
            "cryptobench_apo": _cryptobench_group(),
        },
    }


def main() -> int:
    OUT.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
