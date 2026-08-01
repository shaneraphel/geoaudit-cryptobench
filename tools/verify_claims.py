#!/usr/bin/env python3
"""Fail closed if the GeoAudit paper tree exceeds its declared scope."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
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
# One absolute path ships, and it cannot be removed without doing more damage than
# it does. The external validation set records why each candidate was dropped, and
# the receptor writer names its destination in the message it raises, so the reason
# for dropping 9lym_CA carries the checkout root of whoever built the set.
#
# The builder now strips the root, but this artifact is frozen: the preregistered
# plan pins its SHA-256, and the read refuses to run unless the hash still matches.
# Redacting the string would change the hash, which would mean regenerating the plan
# and rerunning the only confirmatory comparison in the paper -- and the new plan's
# commit would then postdate the scores, destroying the ordering evidence that makes
# the result confirmatory in the first place. A username in a diagnostic is a
# smaller cost than that, so it stays and is named here instead.
#
# The exemption is one occurrence in one file. A second occurrence there, or one
# anywhere else, still fails.
ABSOLUTE_PATH_EXEMPT = {
    "results/external/EXTERNAL_SET.json": {
        "n_permitted": 1,
        "field": "receptors.dropped[0].why",
        "why_it_cannot_be_redacted": (
            "the preregistered plan pins this file's SHA-256 and the confirmatory "
            "read verifies it; changing a byte would require rewriting the plan "
            "after the scores existed"),
    },
}
CREDENTIAL = re.compile(
    r"(?i)(?:password|passwd|api[_-]?key|secret|token)\s*[:=]\s*[\"']?[^\s,\"']{8,}"
)
TARGET_PANEL = {"ESR1", "KRAS", "FLT3", "PIM1", "PIK3CA", "CDK4/6"}


def frozen_field_digest_checks(root: Path) -> list[str]:
    """Complaints if a compiled field's recorded ``code_sha256`` is no longer live.

    ``table_field.code_sha256`` hashes the bytes of ten source files. Each compiled
    field records the value it was built under, and every per-unit prediction under
    ``results/external/`` carries the same string as ``tool_version``. The digest is
    the reader's route from a frozen number back to the exact code that produced it,
    so it is over bytes and not over behaviour: an edit that provably cannot move a
    number still severs the link.

    Two Rust kernels were once written straight into ``table_bank.py``. They were
    bit-identical ports and the numbers did not move by so much as an ulp, and the
    edit was still wrong, because ``TABLE_FIELD.json``, ``GEOMETRY_FIELD.json`` and
    57 frozen external predictions all then pointed at source that no longer
    existed. The failure surfaced as a per-unit version mismatch on one external
    prediction --- true, but it names a unit rather than a cause, and it only fires
    at all because Set A happens to have been read. This check states the rule
    directly and names the file, and it fires on a fresh tree with no reads in it.

    The fix is to revert the digested file and put the change beside it; see
    ``src/pocket_bench/methods/table_bank_accel.py``. Recompiling the field instead
    would move the digest to match the new code and quietly redefine what the frozen
    external read was a read of.
    """
    import sys as _sys
    src = root / "src"
    if str(src) not in _sys.path:
        _sys.path.insert(0, str(src))
    try:
        from pocket_bench.methods.table_field import code_sha256
    except Exception as exc:  # pragma: no cover - import failure is its own alarm
        return [f"cannot import table_field to recompute the digest: {exc}"]

    live = code_sha256()
    problems: list[str] = []
    for rel in ("data/cryptobench_apo/TABLE_FIELD.json",
                "data/cryptobench_apo/GEOMETRY_FIELD.json"):
        path = root / rel
        if not path.exists():
            continue
        recorded = (json.loads(path.read_text()).get("code_sha256") or "")
        if recorded and recorded != live:
            problems.append(
                f"{rel} was compiled under code_sha256 {recorded[:16]}... but the "
                f"eight digested sources now hash to {live[:16]}...; revert the "
                f"edited file and put the change in a module the digest does not "
                f"cover, or the frozen reads no longer name their own code")
    return problems


def candidate_showcase_checks(
        root: Path, primary_files: list[Path]) -> tuple[list[str], list[str]]:
    """Which files are candidate evidence, and which admitted ones are incomplete.

    Returns ``(offenders, problems)``: files holding candidate evidence that no
    registry entry admits, and complaints about the admitted ones. Both empty is
    the passing state.

    This lives outside ``main`` so it can be run against a constructed tree, and
    that is not tidiness. The rule it replaced was green for its whole life while
    doing nothing --- the showcase it was written to admit sat outside every
    pattern it matched --- and nobody noticed, because the only way to exercise
    it was to run the whole verifier over the real repository, where the answer
    is "no offenders" whether the gate works or not. A gate that cannot be
    pointed at a synthetic failure cannot be shown to catch a real one.
    """
    # Candidate evidence belongs to the companion repository, and the scope
    # contract says so. This used to be a denylist of three filenames, which is
    # the wrong shape for the rule it is meant to enforce: it passed for months
    # while evidence/kras_g12d_sm/ sat in the tree with curated candidates, a
    # DFT metrics table and a toxicology report, none of which happened to be
    # named one of the three. A structural rule instead -- no candidate-evidence
    # directory, and no file that declares itself a candidate record.
    candidate_dirs = {"evidence", "candidates"}
    candidate_markers = (
        '"schema": "foliation.er100.candidate',
        '"chemistry_ready_records"',
        '"slot_records"',
        "CURATED_TOP",
    )
    # The third rule, and the one that makes the other two nearly redundant.
    #
    # The two above are still denylists: a directory name and four strings. They
    # were written after a denylist of three filenames passed for months while
    # curated candidates sat in the tree, and the replacement had the same shape
    # as the thing it replaced -- so it failed the same way. Removing the ESR1
    # showcase from the registry and re-running left this gate green, because
    # the showcase is not under evidence/ and carries none of the four markers.
    # The exception admitting it had therefore never done any work; the file was
    # simply never seen. The probe that found this also found
    # data/appendix_esr1/SHOWCASE_INPUT.json, six molecules that no completeness
    # gate had ever read.
    #
    # A structural rule instead, keyed on the thing that actually makes a file
    # candidate evidence: it names a molecule. A SMILES string is a molecule's
    # identity, so a file holding one is candidate evidence whatever it is
    # called and wherever it sits, and has to be registered. A tool that names
    # the field without holding a value is not caught, which is the intended
    # boundary -- tools/emit_esr1_showcase.py mentions all three identifier
    # fields and stays clean, while the same tool with one molecule hard-coded
    # into it would not.
    smiles_value = re.compile(
        r'"(?:isomeric_|canonical_|input_|parent_)?smiles"\s*:\s*"([^"]{4,})"',
        re.IGNORECASE)

    def _names_a_molecule(text: str) -> bool:
        return bool(smiles_value.search(text))
    # The exception, and it is a registry rather than a path in this file.
    #
    # The prohibition above exists because a bulk dump of thousands of candidate
    # records is unreviewable and because its presence invites the affinity and
    # efficacy claims this paper's scope contract forbids. Both objections are
    # about volume and about claims. Neither applies to a small, field-complete,
    # individually audited set admitted for one stated purpose, so such a set is
    # admitted -- and gated harder than the rule it relaxes.
    #
    # This was one hard-coded path until a second showcase had to be admitted.
    # Hard-coding the second would have made the third easy and the tenth
    # invisible, so the admitted set now lives in
    # contracts/CANDIDATE_SHOWCASES.json: adding one is a diff to a contract
    # that has to say what the showcase demonstrates, what it may not claim, and
    # how many records it may hold. A file not in that registry fails exactly as
    # any other candidate file does, including a showcase at a new path.
    #
    # What the exception rests on, and it is the only irreversible thing here:
    # the repository is private, so nothing in it is a publication and no prior
    # art is created against a composition claim. Every showcase must declare
    # that, and the registry records when the declaration was last checked
    # against the remote instead of believed.
    registry_path = root / "contracts/CANDIDATE_SHOWCASES.json"
    registry: dict = {}
    registry_problems: list[str] = []
    if registry_path.exists():
        try:
            registry = json.loads(registry_path.read_text())
        except Exception as exc:  # noqa: BLE001
            registry_problems.append(f"registry is not valid JSON: {exc}")
    else:
        registry_problems.append(
            "contracts/CANDIDATE_SHOWCASES.json is missing; with no registry "
            "no showcase is admitted and the prohibition applies to all of them")
    admitted = {s["path"]: s for s in (registry.get("showcases") or [])}
    glob_rules = registry.get("global") or {}

    def _is_admitted_showcase(path: Path) -> bool:
        rel = path.relative_to(root).as_posix()
        entry = admitted.get(rel)
        if entry is None:
            return False
        try:
            doc = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            return False
        return doc.get("schema") == entry.get("schema")

    offenders = []
    for path in primary_files:
        if _is_admitted_showcase(path):
            continue
        rel = path.relative_to(root).as_posix()
        if rel == "tools/verify_claims.py":
            continue  # this file quotes the patterns it searches for
        text = path.read_text(errors="ignore")
        if (candidate_dirs & set(path.relative_to(root).parts[:-1])
                or any(mark in text[:4000] for mark in candidate_markers)
                or _names_a_molecule(text)):
            offenders.append(rel)

    # Each admitted showcase's own gate. It fails closed on every clause: a
    # missing field, a record over its cap, the tree over the global cap, a
    # claim the scope contract forbids, or a required declaration absent. A
    # showcase that is registered and missing from disk is not an error -- the
    # registry may run ahead of the tree -- but one that is present and
    # incomplete is.
    showcase_problems: list[str] = list(registry_problems)
    total_records = 0
    for rel, entry in sorted(admitted.items()):
        path = root / rel
        if not path.exists():
            continue
        try:
            doc = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            showcase_problems.append(f"{rel} is not valid JSON: {exc}")
            continue
        if doc.get("schema") != entry.get("schema"):
            showcase_problems.append(
                f"{rel}: schema is {doc.get('schema')!r}, not "
                f"{entry.get('schema')!r}")
        for key, want in (glob_rules.get("required_declarations") or {}).items():
            if doc.get(key) is not want:
                showcase_problems.append(
                    f"{rel}: {key} is {doc.get(key)!r}, not {want!r}")
        for key, want in (entry.get("additional_required_declarations")
                          or {}).items():
            if doc.get(key) is not want:
                showcase_problems.append(
                    f"{rel}: {key} is {doc.get(key)!r}, not {want!r}")
        records = doc.get("records") or []
        total_records += len(records)
        if not records:
            showcase_problems.append(f"{rel}: carries no records")
        cap = int(entry.get("record_cap", 0))
        if len(records) > cap:
            showcase_problems.append(
                f"{rel}: {len(records)} records exceeds its cap of {cap}; the "
                f"exception was argued for a showcase, not for a dump")
        # An entry may state its own field list, because an input file carries
        # the same molecules at an earlier stage than a finished showcase and
        # demanding a bond graph of it would either fail an honest file or push
        # its author to fabricate the field. Overriding costs a written reason:
        # a list without why_its_own_fields is a silent relaxation.
        own_fields = entry.get("required_fields_on_every_record")
        if own_fields is not None and not entry.get("why_its_own_fields"):
            showcase_problems.append(
                f"{rel}: states its own required_fields_on_every_record without "
                f"why_its_own_fields; an override with no reason is how the "
                f"default becomes optional")
        required = tuple(
            own_fields
            if own_fields is not None
            else glob_rules.get("default_required_fields_on_every_record") or ()
        ) + tuple(entry.get("additional_required_fields") or ())
        # The audit keys are required of every role. Which file a molecule sits
        # in does not change the reasons to withdraw it.
        audit_field = entry.get("audit_field", "structural_audit")
        audit_keys = tuple(glob_rules.get("what_structural_audit_must_carry")
                           or ())
        for i, rec in enumerate(records):
            missing = [k for k in required if not rec.get(k)]
            if missing:
                showcase_problems.append(
                    f"{rel} record {i} ({rec.get('candidate_id')}) is missing "
                    f"{missing}")
            audit = rec.get(audit_field)
            if not isinstance(audit, dict):
                showcase_problems.append(
                    f"{rel} record {i} ({rec.get('candidate_id')}): {audit_field}"
                    f" is absent or is not an object")
                continue
            gaps = [k for k in audit_keys if k not in audit]
            if gaps:
                showcase_problems.append(
                    f"{rel} record {i} ({rec.get('candidate_id')}): "
                    f"{audit_field} does not state {gaps}")
    total_cap = int(glob_rules.get("total_record_cap", 0))
    if total_records > total_cap:
        showcase_problems.append(
            f"{total_records} candidate records across all showcases exceeds "
            f"the global cap of {total_cap}; per-showcase caps do not bound the "
            f"tree and the reviewability argument is about the total")
    return offenders, showcase_problems


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
    # The paper claim is the CryptoBench receptor-only benchmark. Allele-conditioning
    # and candidate generation are appendix / future work and must NOT re-enter the
    # primary claim.
    bench = scope.get("primary_benchmark") or {}
    checks["benchmark_paper_scope"] = (
        scope.get("paper_id") == "geoaudit-cryptobench-zero-shot"
        and bench.get("input_contract") == "receptor_only_apo"
        and "cryptobench" in str(bench.get("name") or "").lower()
        and "mmseqs2" in str(bench.get("split") or "").lower()
        and set(bench.get("primary_metrics") or []) == {
            "residue_auc", "residue_pr_auc", "residue_mcc", "residue_f1"
        }
        and (scope.get("appendices") or {}).get("B", {}).get(
            "excluded_from_primary_claim"
        ) is True
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
    # The KRAS G12D candidate-generation campaign is out of scope for this
    # repository's CryptoBench claim, so its counts are no longer carried here or
    # gated. The gate is replaced by its negation: the manifest must NOT
    # reintroduce a candidate-generation claim into the paper repo.
    checks["no_out_of_scope_candidate_campaign_claim"] = not any(
        k.startswith("v9_1_kras") for k in companion
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
    # The README must lead with the benchmark claim and keep ESR1 / allele
    # conditioning demarcated in appendices.
    #
    # This used to require the literal words "zero-shot", and the README passed
    # while the sentence carrying them was false: five detectors compile counts
    # on the training fold. A gate that pins a word cannot tell a true use of it
    # from a false one, so it pins the fact instead. Every detector holding a
    # compiled artifact must be named in the README, and the paths below must
    # resolve, so renaming an artifact fails here rather than going unnoticed.
    #
    # The names are required inside the training-exposure bullet, not anywhere
    # in the file: the autogenerated results table lists every method, so a
    # whole-document search passed even with a detector struck from the
    # declaration, which is the one place the reader looks for it.
    low = readme.lower()
    compiled = {
        "quaternary_lut": "data/cryptobench_apo/RESOLUTION_FIELD.json",
        "quaternary_lut_seq": "data/cryptobench_apo/RESOLUTION_FIELD_B.json",
        "algebraic_field": "data/cryptobench_apo/ALGEBRAIC_FIELD.json",
        "algebraic_field_linear": "data/cryptobench_apo/ALGEBRAIC_FIELD_LINEAR.json",
        "table_field": "data/cryptobench_apo/TABLE_FIELD.json",
    }
    m = re.search(r"^- What is compiled on the training fold.*?(?=^- )",
                  readme, re.S | re.M)
    exposure = m.group(0).lower() if m else ""
    checks["readme_names_every_compiled_detector"] = bool(exposure) and all(
        (root / p).is_file() and name in exposure for name, p in compiled.items()
    )
    checks["readme_scopes_benchmark_and_appendix"] = (
        "cryptobench" in low
        and "training fold" in low
        and "receptor-only" in low
        and "retrospective" in low
        and "clinical_grade=false" in low
        and "ESR1" in readme
        and "Appendix A" in readme
        and "Appendix B" in readme
    )

    def _is_local_only(path: Path) -> bool:
        # Never-published local-only material (internal changelog, hardware/
        # accelerator notes, proprietary engine source, peer-review transcripts).
        # Gitignored via
        # *.local.* and _local/ ; excluded here so it cannot trip scope gates.
        parts = set(path.parts)
        return "_local" in parts or ".local" in "".join(path.suffixes)

    def _tracked() -> list[Path] | None:
        # These gates ask what this repository publishes, so the enumeration has
        # to be what the repository contains rather than what happens to be
        # sitting in the directory. Walking the filesystem read third-party
        # site-packages out of a local virtualenv and failed the credential, path
        # and scope gates on numpy's docstrings and pygments' token types --- a
        # gate that fires on somebody else's vendored code is telling the reader
        # nothing about this paper.
        #
        # --others --exclude-standard adds files that are untracked but not
        # ignored, and it is there because the tracked-only version had an
        # ordering hole: writing a candidate dump, running make verify green and
        # then committing passed, because at verify time the file was untracked
        # and at commit time nothing re-ran. An untracked, unignored file is one
        # git add away from being published, and AGENTS.md records the commit
        # where git add -A swept 600 files and 292 MB into the history. Ignored
        # paths stay excluded, so the virtualenv that motivated this function is
        # still out.
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"], cwd=root, capture_output=True)
        if proc.returncode != 0:
            return None
        return [root / p for p in proc.stdout.decode().split("\0") if p]

    candidates = _tracked()
    if candidates is None:
        candidates = list(root.rglob("*"))
    primary_files = [
        path
        for path in candidates
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
    # Per file rather than over the concatenation, so that the one exempt
    # occurrence can be counted rather than the rule being weakened for everyone.
    absolute_path_offenders = {}
    for path in primary_files:
        rel = str(path.relative_to(root))
        n = len(ABSOLUTE_PATH.findall(path.read_text(errors="ignore")))
        allowed = ABSOLUTE_PATH_EXEMPT.get(rel, {}).get("n_permitted", 0)
        if n > allowed:
            absolute_path_offenders[rel] = {"found": n, "permitted": allowed}
    checks["no_local_absolute_paths_in_primary_docs"] = not absolute_path_offenders
    if absolute_path_offenders:
        checks["absolute_path_offenders"] = absolute_path_offenders
    checks["no_credential_patterns_in_primary_docs"] = (
        CREDENTIAL.search(tree_text) is None
    )
    offenders, showcase_problems = candidate_showcase_checks(root, primary_files)
    checks["no_bulk_candidate_dump_in_paper_tree"] = not offenders
    if offenders:
        checks["_candidate_evidence_offenders"] = offenders[:10]
    checks["candidate_showcases_are_registered_and_complete"] = (
        not showcase_problems)
    if showcase_problems:
        checks["_candidate_showcase_problems"] = showcase_problems[:10]

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

    # No learned model anywhere on the detector's datapath.
    #
    # The repository's central claim is that the shipped detector contains no
    # trained network and no fitted real-valued transform, and until now that
    # was a property of the code that nobody checked -- true, but promised
    # rather than enforced. pLM-NN is rebuilt here and reported as a baseline,
    # which is the honest thing to do with a method that beats us; the risk is
    # that a future session, reaching for the 0.0243 deficit, quietly lets an
    # embedding into a wire builder and the claim silently stops holding. This
    # walks every module under src/ and fails on an import of a learning
    # framework or an embedding library, and separately on any reference to the
    # cached encoder or its artifacts.
    #
    # scipy is permitted and named rather than pattern-matched: it is used for
    # the confidence intervals in the reporting path, not on the datapath, and
    # a blanket rule would have to either allow every numerical package or ban
    # the one that computes a Wilson interval.
    learned = {"torch", "tensorflow", "jax", "flax", "keras", "esm",
               "transformers", "sklearn", "xgboost", "lightgbm", "catboost",
               "fairseq", "sentencepiece", "onnxruntime"}
    embedding_refs = ("esm2_t36_3B", "PLMNN_WEIGHTS", "_plmnn", "ESM2_CACHE")
    detector_clean = True
    offenders: list[str] = []
    for py in sorted((root / "src").rglob("*.py")):
        text = py.read_text()
        rel = str(py.relative_to(root))
        for node in ast.walk(ast.parse(text)):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods = [node.module]
            for m in mods:
                if m.split(".")[0] in learned:
                    detector_clean = False
                    offenders.append(f"{rel} imports {m}")
        # A docstring may name ESM-2 to say it is excluded, which is exactly
        # what sequence_wires.py does and is worth keeping. Code may not.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if any(ref in stripped for ref in embedding_refs):
                detector_clean = False
                offenders.append(f"{rel}: {stripped[:60]}")
    checks["detector_reads_no_learned_model"] = detector_clean
    if offenders:
        checks["_detector_learned_model_offenders"] = offenders

    digest_problems = frozen_field_digest_checks(root)
    checks["frozen_fields_name_their_own_code"] = not digest_problems
    if digest_problems:
        checks["_frozen_field_digest_problems"] = digest_problems

    failed = sorted(name for name, ok in checks.items() if not ok)
    print(json.dumps({"ok": not failed, "checks": checks, "failed": failed}, indent=2))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
