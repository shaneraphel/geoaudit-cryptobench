#!/usr/bin/env python3
"""Fail closed if primary claims contradict the quarantine/correction inventory."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


FORBIDDEN_PRIMARY = re.compile(
    r"(?i)\b(40/0*40\s*PASS|FDA\s+superior|hidden[- ]pocket\s+superior|"
    r"patient\s+scRNA|ER100\s+is\s+100\s+independent|beats\s+FDA)\b"
)
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])/(?:Users|home|private)/")
CREDENTIAL = re.compile(
    r"(?i)(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*[\"']?[^\s,\"']{8,}"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    checks: dict[str, bool] = {}

    inventory = json.loads((root / "CORRECTION_INVENTORY.json").read_text())
    quarantine = json.loads(
        (root / "legacy/quarantine/2026-07-22/QUARANTINE_INDEX.json").read_text()
    )
    claims = json.loads((root / "results/pilot/CLAIMS.json").read_text())
    pilot = json.loads(
        (root / "results/pilot/RETROSPECTIVE_PILOT_REPORT.json").read_text()
    )
    clusters = json.loads(
        (root / "data/manifests/STRUCTURE_CLUSTER_LEDGER.json").read_text()
    )

    checks["inventory_schema"] = (
        inventory.get("schema") == "foliation.evidence_correction_inventory.v1"
    )
    checks["quarantine_indexed"] = quarantine.get("record_count", 0) >= 200
    checks["all_inventory_claims_invalid_or_unsupported"] = all(
        row.get("corrected_status")
        in {"invalid_for_primary_claim", "unsupported", "not_supported_by_pilot"}
        for row in inventory.get("claims", [])
    )
    checks["pilot_not_locked"] = (
        claims.get("evidence_level") == "retrospective_pilot_only"
        and claims.get("comparative_claim_allowed") is False
        and pilot.get("not_a_locked_test") is True
        and clusters.get("split_integrity_passed") is False
    )
    checks["clinical_grade_false"] = (
        inventory.get("clinical_grade") is False
        and claims.get("clinical_grade") is False
        and pilot.get("clinical_grade") is False
    )

    primary_text = "\n".join(
        [
            json.dumps(claims),
            json.dumps(
                {
                    k: pilot.get(k)
                    for k in (
                        "evidence_level",
                        "comparative_claim_allowed",
                        "truth_boundary",
                        "primary_metric",
                        "summaries",
                    )
                }
            ),
        ]
    )
    checks["no_forbidden_primary_phrases"] = FORBIDDEN_PRIMARY.search(primary_text) is None

    readme = (root / "README.md").read_text(errors="ignore")
    checks["readme_marks_legacy_invalid"] = (
        "invalid_for_primary_claim" in json.dumps(inventory)
        and "retrospective" in readme.lower()
        and "clinical_grade=false" in readme.lower()
    )

    # Quarantine digests must still match moved files.
    digest_ok = True
    for row in quarantine.get("records", []):
        path = root / row["path"]
        if not path.is_file() or sha256(path) != row["sha256"]:
            digest_ok = False
            break
    checks["quarantine_digests_match"] = digest_ok

    tree_text = "\n".join(
        path.read_text(errors="ignore")
        for path in [
            root / "README.md",
            root / "CORRECTION_INVENTORY.json",
            root / "results/pilot/CLAIMS.json",
        ]
        if path.is_file()
    )
    checks["no_local_absolute_paths_in_primary_docs"] = (
        ABSOLUTE_PATH.search(tree_text) is None
    )
    checks["no_credential_patterns_in_primary_docs"] = (
        CREDENTIAL.search(tree_text) is None
    )

    failed = sorted(name for name, ok in checks.items() if not ok)
    print(json.dumps({"ok": not failed, "checks": checks, "failed": failed}, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
