#!/usr/bin/env python3
"""Fail closed if the ER100 paper tree exceeds its declared scope."""
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
    r"(?i)\b(silicon|EDA|IND\s+readiness|retina|kidney|"
    r"uremia|leukemia|FDA\s+comparison)\b"
)
# Campaign/generator names may appear only as companion release directory
# pointers, never as primary scientific claims in this paper tree.
CAMPAIGN_CLAIM = re.compile(
    r"(?i)\b(GPT[- ]?5\.6\s+campaign|free[- ]reasoning\s+campaign)\b"
)
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z])/(?:Users|home|private)/")
CREDENTIAL = re.compile(
    r"(?i)(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*[\"']?[^\s,\"']{8,}"
)
TARGET_PANEL = {"ESR1", "KRAS", "FLT3", "PIM1", "PIK3CA", "CDK4/6"}


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
    scope = json.loads((root / "contracts/ER100_PAPER_SCOPE.json").read_text())
    companion = json.loads(
        (root / "data/manifests/COMPANION_EVIDENCE.json").read_text()
    )

    checks["no_legacy_showcase_tree"] = not (root / "legacy").exists()
    checks["appendix_pilot_not_locked"] = (
        claims.get("evidence_level") == "retrospective_pilot_only"
        and claims.get("comparative_claim_allowed") is False
        and pilot.get("not_a_locked_test") is True
        and clusters.get("split_integrity_passed") is False
        and (scope.get("appendices") or {}).get("A", {}).get("evidence_level")
        == "retrospective_pilot_only"
    )
    checks["clinical_grade_false"] = (
        claims.get("clinical_grade") is False
        and pilot.get("clinical_grade") is False
        and scope.get("clinical_grade") is False
        and companion.get("clinical_grade") is False
    )
    checks["multitarget_paper_scope"] = (
        scope.get("paper_id") == "er100-multitarget-multimodal"
        and set(scope.get("target_panel") or []) == TARGET_PANEL
        and len(scope.get("structure_defined_modalities") or []) == 4
    )
    checks["companion_pointer_present"] = (
        "foliation-er100-multimodal-chemistry"
        in str(companion.get("companion_repo") or "")
        and companion.get("counts", {}).get("chemistry_ready") == 4000
        and set((companion.get("counts") or {}).get("targets") or {}) == TARGET_PANEL
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
    checks["readme_scopes_paper_and_appendix"] = (
        "retrospective" in readme.lower()
        and "clinical_grade=false" in readme.lower()
        and "ESR1" in readme
        and "multitarget" in readme.lower()
        and "foliation-er100-multimodal-chemistry" in readme
        and "Appendix A" in readme
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
    checks["no_campaign_claim_language"] = CAMPAIGN_CLAIM.search(scope_text) is None
    checks["no_local_absolute_paths_in_primary_docs"] = (
        ABSOLUTE_PATH.search(tree_text) is None
    )
    checks["no_credential_patterns_in_primary_docs"] = (
        CREDENTIAL.search(tree_text) is None
    )
    checks["no_bulk_candidate_dump_in_paper_tree"] = not any(
        name in path.as_posix()
        for path in primary_files
        for name in (
            "PHASE2_CHEMISTRY_READY_RECORDS.json",
            "PHASE1_SLOT_RECORDS.json",
            "BILINGUAL_CANDIDATE_EVIDENCE.json",
        )
    )

    failed = sorted(name for name, ok in checks.items() if not ok)
    print(json.dumps({"ok": not failed, "checks": checks, "failed": failed}, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
