#!/usr/bin/env python3
"""Fail closed if the ER100 paper tree exceeds its ESR1 benchmark scope."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FORBIDDEN_PRIMARY = re.compile(
    r"(?i)\b(hidden[- ]pocket\s+superior|therapeutic\s+superiority|"
    r"clinical[- ]grade|clinical\s+readiness)\b"
)
OUT_OF_SCOPE = re.compile(
    r"(?i)\b(GPT[- ]?5\.6|silicon|EDA|IND\s+readiness|retina|kidney|"
    r"uremia|leukemia|FDA\s+comparison)\b"
)
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])/(?:Users|home|private)/")
CREDENTIAL = re.compile(
    r"(?i)(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*[\"']?[^\s,\"']{8,}"
)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()
    checks: dict[str, bool] = {}

    claims = json.loads((root / "results/pilot/CLAIMS.json").read_text())
    pilot = json.loads(
        (root / "results/pilot/RETROSPECTIVE_PILOT_REPORT.json").read_text()
    )
    clusters = json.loads(
        (root / "data/manifests/STRUCTURE_CLUSTER_LEDGER.json").read_text()
    )

    checks["esr1_only_tree"] = not (root / "legacy").exists()
    checks["pilot_not_locked"] = (
        claims.get("evidence_level") == "retrospective_pilot_only"
        and claims.get("comparative_claim_allowed") is False
        and pilot.get("not_a_locked_test") is True
        and clusters.get("split_integrity_passed") is False
    )
    checks["clinical_grade_false"] = (
        claims.get("clinical_grade") is False
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
    checks["readme_scopes_retrospective_pilot"] = (
        "retrospective" in readme.lower()
        and "clinical_grade=false" in readme.lower()
        and "ESR1" in readme
    )

    primary_files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.suffix.lower() in {".md", ".json", ".py", ".yml", ".yaml"}
    ]
    tree_text = "\n".join(path.read_text(errors="ignore") for path in primary_files)
    scope_text = "\n".join(
        path.read_text(errors="ignore")
        for path in primary_files
        if path.name != "verify_claims.py"
    )
    checks["no_out_of_scope_paper_material"] = OUT_OF_SCOPE.search(scope_text) is None
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
