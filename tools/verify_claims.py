#!/usr/bin/env python3
"""Fail closed if the GeoAudit paper tree exceeds its declared scope."""
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
    r"uremia|leukemia|FDA\s+comparison|systems\s+biology|"
    r"cross[- ]disease)\b"
)
# Campaign/generator names may appear only as companion release directory
# pointers, never as primary scientific claims in this paper tree.
CAMPAIGN_CLAIM = re.compile(
    r"(?i)\b(GPT[- ]?5\.6\s+campaign|free[- ]reasoning\s+campaign)\b"
)
# Proprietary engine names must stay LOCAL only; public docs use abstract
# descriptors ("geometric manifold prior", "exact-form topological filter",
# "discrete conformal rescaling"). This gate hard-fails if a brand leaks back in.
PROPRIETARY_ENGINE = re.compile(
    r"(?i)\b(NCGD|PINEKF|PINEFK|Non[- ]Commutative\s+Geometric|"
    r"Conformal\s+Squeeze|Kähler|Kahler)\b"
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
    scope = json.loads((root / "contracts/GEOAUDIT_PAPER_SCOPE.json").read_text())
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
    # Science-honesty gate: the retrospective pilot numbers were produced with a
    # now-fixed label-merge defect (ligand selected by resname only). Until the
    # labels are regenerated per chain, every pilot number is non-citable and
    # MUST be marked invalidated in both the claims file and the report.
    checks["pilot_marked_invalidated_pending_regeneration"] = (
        claims.get("status") == "INVALIDATED_PENDING_LABEL_REGENERATION"
        and bool(claims.get("invalidation_reason"))
        and pilot.get("invalidated") is True
        and bool(pilot.get("invalidation_reason"))
    )
    checks["clinical_grade_false"] = (
        claims.get("clinical_grade") is False
        and pilot.get("clinical_grade") is False
        and scope.get("clinical_grade") is False
        and companion.get("clinical_grade") is False
    )
    checks["multitarget_paper_scope"] = (
        scope.get("paper_id") == "geoaudit-multitarget-multimodal"
        and set(scope.get("target_panel") or []) == TARGET_PANEL
        and len(scope.get("structure_defined_modalities") or []) == 4
    )
    checks["companion_pointer_present"] = (
        "gf4-allele-conditioned-evidence"
        in str(companion.get("companion_repo") or "")
        and companion.get("counts", {}).get("chemistry_ready") == 4000
        and set((companion.get("counts") or {}).get("targets") or {}) == TARGET_PANEL
        and re.fullmatch(
            r"[0-9a-f]{40}", str(companion.get("companion_git_sha_v9") or "")
        )
        is not None
    )
    protocol = companion.get("v8_1_protocol_requalification") or {}
    scale = companion.get("requested_1000_per_target_per_modality_scale") or {}
    expansion = companion.get("v8_1_expansion_readiness") or {}
    checks["protocol_requalification_truth_boundary"] = (
        protocol.get("records_requalified") == 4000
        and protocol.get("candidate_geometry_route_ready") == 167
        and protocol.get("structural_protocol_blocked") == 3833
        and protocol.get("method_template_records_quarantined") == 334
        and protocol.get("pdb_5t35_panel_geometry_eligible") is False
        and protocol.get("pdb_5fqd_panel_geometry_eligible") is False
        and protocol.get("target_pose_computed") is False
        and scale.get("requested_record_slots") == 24000
        and scale.get("druggable_candidates_established") == 0
        and expansion.get("cells_ready_for_1000") == 0
        and expansion.get("internal_diversity_seeds") == 875
        and expansion.get("diversity_and_protocol_seeds") == 52
        and expansion.get("global_novelty_claim") is False
        and expansion.get("druggability_claim") is False
    )
    v9_slots = companion.get("v9_design_slot_ledger") or {}
    v9_sources = companion.get("v9_real_source_snapshot") or {}
    v9_routes = companion.get("v9_genotype_isoform_and_modality_routes") or {}
    checks["v9_source_and_expansion_truth_boundary"] = (
        v9_slots.get("total_design_slots") == 24000
        and v9_slots.get("v7_v8_lineage_references") == 4000
        and v9_slots.get("empty_design_slots") == 20000
        and v9_slots.get("v9_identity_ready") == 0
        and v9_slots.get("druggable_candidates_established") == 0
        and v9_sources.get("chembl_db_version") == "ChEMBL_37"
        and v9_sources.get("declared_activity_query_fully_paginated") is True
        and v9_sources.get("raw_payloads") == 420
        and v9_sources.get("raw_payload_bytes") == 244652474
        and v9_sources.get("synthetic_assay_or_patient_data_used") is False
        and v9_sources.get("chembl_reported_reference_identities") == 26543
        and v9_sources.get("rdkit_parseable_reference_smiles") == 26543
        and v9_sources.get("rdkit_chembl_inchikey_match") == 26502
        and v9_sources.get("rdkit_chembl_inchikey_mismatch_retained") == 41
        and v9_sources.get("null_variant_inferred_as_wild_type") is False
        and v9_sources.get("source_backed_diverse_chemotype_anchors") == 9002
        and v9_sources.get("icloud_account_sync_completion_claimed") is False
        and v9_routes.get("flt3_itd_structure_route") == "BLOCKED"
        and v9_routes.get("reference_structure_activates_arbitrary_candidate")
        is False
        and v9_routes.get("cells_safe_expansion_ready") == 0
    )
    openreview = companion.get("v9_openreview_and_lineage_migration") or {}
    checks["v9_openreview_blocks_unsafe_migration"] = (
        openreview.get("recommendation") == "MAJOR_REVISION"
        and openreview.get("historical_lineage_records") == 4000
        and openreview.get("v9_identity_migrated") == 0
        and openreview.get("lineage_migration_blocked") == 4000
        and openreview.get("cdk_5l2i_route_drift_records") == 664
        and openreview.get("dual_geometry_truth_conflict_records") == 1002
        and openreview.get("records_with_nonrecomputable_hard_gate") == 4000
        and openreview.get("biology_exact_unique_values") == 24
        and openreview.get("pharmacology_exact_unique_values") == 24
        and openreview.get("expansion_authorized") is False
    )
    v91 = companion.get(
        "v9_1_kras_g12d_targeted_small_molecule_computational_priority"
    ) or {}
    checks["v9_1_computational_priority_truth_boundary"] = (
        v91.get("clinical_grade") is False
        and v91.get("campaign_scope") == "KRAS_G12D_targeted_small_molecule_only"
        and v91.get("paper_scope_preserved") is True
        and v91.get("priority_label") == "computational_priority"
        and v91.get("primary_geometry_pdb") == "9BL0"
        and v91.get("covalent_observation_only_pdb") == "9GBJ"
        and v91.get("accepted_count") == 494
        and int(v91.get("representative_count") or 0)
        < int(v91.get("accepted_count") or 0)
        and v91.get("druggability_claim") is False
        and v91.get("global_novelty_claim") is False
        and v91.get("clinical_readiness_claim") is False
        and v91.get("target_pose_is_affinity_claim") is False
        and v91.get("surechembl_full_15gb_snapshot_claimed") is False
        and str(v91.get("records_jsonl") or "").endswith(
            "ACCEPTED_CANDIDATES.jsonl.gz"
        )
        and str(v91.get("records_parquet") or "").endswith(".parquet")
        and str(v91.get("structures_sdf") or "").endswith(".sdf.gz")
        and bool(
            re.fullmatch(r"[0-9a-f]{40}", str(v91.get("companion_git_sha_v9_1") or ""))
        )
    )

    # --- Science-invariant gate -------------------------------------------
    # Reviewer #3: CI previously only checked scope wording, never a scientific
    # invariant, so a 232-heavy-atom merged label passed. These checks re-read
    # the actual artifacts and fail closed on the specific defect classes.
    label_dir = root / "data/labels"
    label_files = sorted(label_dir.glob("*_labels.json")) if label_dir.exists() else []
    label_ok = len(label_files) >= 14
    for lf in label_files:
        lab = json.loads(lf.read_text())
        n_heavy = len(lab.get("ligand_heavy_coords") or [])
        # A single small-molecule ligand instance is ~10-70 heavy atoms; the
        # merge bug inflated OHT to 232 (~8 copies). Reject anything >80 or <3.
        if not (3 <= n_heavy <= 80) or len(lab.get("ligand_centroid") or []) != 3:
            label_ok = False
    checks["labels_single_instance_physical"] = label_ok

    regen_path = root / "results/pilot/REGENERATED_PILOT_REPORT.json"
    if regen_path.exists():
        regen = json.loads(regen_path.read_text())
        per = regen.get("per_method") or {}
        methods_present = {"foliation_pocket_ro", "fpocket", "p2rank", "random_bbox"} <= set(per)
        denom_ok = all(
            (m.get("ok", 0) + m.get("unavailable", 0) + m.get("crash_empty", 0))
            == regen.get("n_structures")
            for m in per.values()
        )
        checks["regenerated_report_honest"] = (
            regen.get("n_structures") == 14
            and regen.get("labels") == "chain_scoped_single_instance_corrected"
            and regen.get("clinical_grade") is False
            and methods_present
            and denom_ok
        )
    else:
        checks["regenerated_report_honest"] = False

    prov_path = root / "data/manifests/STRUCTURE_PROVENANCE.json"
    if prov_path.exists():
        prov = json.loads(prov_path.read_text()).get("entries") or {}
        checks["structure_provenance_pinned"] = all(
            re.fullmatch(r"[0-9a-f]{64}", str((prov.get(p) or {}).get("sha256") or ""))
            and int((prov.get(p) or {}).get("bytes") or 0) > 5000
            for p in ("9BL0", "5VA1")
        )
    else:
        checks["structure_provenance_pinned"] = False

    # Every receptor path referenced by the input/smoke manifests must exist and
    # must not use the retired data/pocket_bench/esr1/receptors/ location.
    pim = json.loads((root / "data/manifests/PREDICTION_INPUT_MANIFEST.json").read_text())
    smoke = json.loads((root / "results/pilot/SMOKE_BASELINES.json").read_text())
    referenced = [e.get("receptor_pdb") for e in pim.get("entries") or []]
    referenced.append(smoke.get("receptor_pdb"))
    checks["manifest_receptor_paths_resolve"] = bool(referenced) and all(
        rp
        and "data/pocket_bench/esr1/receptors" not in rp
        and (root / rp).exists()
        for rp in referenced
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
        and "gf4-allele-conditioned-evidence" in readme
        and "Appendix A" in readme
    )

    def _is_local_only(path: Path) -> bool:
        # Never-published local-only material (internal changelog, hardware/
        # accelerator notes, proprietary engine source, peer-review transcripts).
        # Gitignored via
        # *.local.* and _local/ ; excluded here so it cannot trip scope gates.
        parts = set(path.parts)
        return "_local" in parts or ".local" in "".join(path.suffixes)

    primary_files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and not _is_local_only(path)
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
    checks["no_proprietary_engine_names_public"] = (
        PROPRIETARY_ENGINE.search(scope_text) is None
    )
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

    # Fail-closed cluster-disjoint split ledger: groups that require disjointness
    # must be disjoint; groups that are not disjoint must declare it (no hiding).
    ledger_path = root / "data/manifests/SPLIT_LEDGER.json"
    if ledger_path.exists():
        ledger = json.loads(ledger_path.read_text())
        groups = ledger.get("groups") or {}
        split_ok = ledger.get("schema") == "geoaudit.split_ledger.v1" and bool(groups)
        for g in groups.values():
            cmap: dict[str, set] = {}
            for a in g.get("assignments") or []:
                cmap.setdefault(a.get("cluster_id"), set()).add(a.get("split"))
            overlap = {c for c, s in cmap.items() if len(s) > 1}
            if g.get("cluster_disjoint_required"):
                if overlap:
                    split_ok = False
            else:
                if g.get("split_integrity_passed") is not False or not str(
                    g.get("reason_not_disjoint") or ""
                ).strip():
                    split_ok = False
        checks["split_ledger_cluster_disjoint_where_required"] = split_ok
    else:
        checks["split_ledger_cluster_disjoint_where_required"] = False

    # Zero-leakage firewall: GeoAudit predictors are decorated, and no predictor
    # module imports the scorer (predictor is physically blind to labels).
    import ast

    methods_dir = root / "src/pocket_bench/methods"
    forbidden = {"pocket_bench.metrics", "pocket_bench.scoring"}
    imports_clean = True
    for py in methods_dir.glob("*.py"):
        tree = ast.parse(py.read_text())
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            if any(m in forbidden or m.split(".")[-1] == "scoring" for m in mods):
                imports_clean = False
    gf_src = (methods_dir / "geometric_foundation.py").read_text()
    fs_src = (methods_dir / "fstar_pocket.py").read_text()
    guarded = (
        'ligand_leak_guard("geometric_foundation")' in gf_src
        and 'ligand_leak_guard("fstar_pocket")' in fs_src
    )
    checks["leakage_firewall_enforced"] = imports_clean and guarded

    failed = sorted(name for name, ok in checks.items() if not ok)
    print(json.dumps({"ok": not failed, "checks": checks, "failed": failed}, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
