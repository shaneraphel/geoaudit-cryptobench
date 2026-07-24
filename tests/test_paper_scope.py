"""Paper-scope tests for the ER100 multitarget multimodal repository."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestPaperScope(unittest.TestCase):
    def test_scope_contract_and_companion_pointer(self) -> None:
        scope = json.loads((ROOT / "contracts/ER100_PAPER_SCOPE.json").read_text())
        companion = json.loads(
            (ROOT / "data/manifests/COMPANION_EVIDENCE.json").read_text()
        )
        self.assertEqual(scope["paper_id"], "er100-multitarget-multimodal")
        self.assertEqual(
            set(scope["target_panel"]),
            {"ESR1", "KRAS", "FLT3", "PIM1", "PIK3CA", "CDK4/6"},
        )
        self.assertEqual(companion["counts"]["chemistry_ready"], 4000)
        self.assertEqual(companion["v8_structure_assets"]["sdf_records"], 4000)
        self.assertEqual(
            companion["v8_structure_assets"]["ligand_feature_count"], 67686
        )
        self.assertIs(
            companion["v8_structure_assets"]["target_pose_computed"], False
        )
        self.assertIs(
            companion["v8_internal_diversity_audit"]["global_novelty_claim"],
            False,
        )
        public_exact = companion["v8_public_exact_match_snapshot"]
        self.assertEqual(public_exact["public_exact_matches"], 81)
        self.assertEqual(
            public_exact["no_exact_match_in_queried_snapshots"], 3919
        )
        self.assertIs(public_exact["all_queries_complete"], True)
        self.assertIs(public_exact["global_novelty_claim"], False)
        self.assertIs(public_exact["patentability_claim"], False)
        self.assertIs(public_exact["freedom_to_operate_claim"], False)
        accelerator = companion["v8_bounded_accelerator_sidecar"]
        self.assertEqual(accelerator["representative_ligand_only_3d_records"], 24)
        self.assertEqual(accelerator["gcu_integer_reference_parity"], 24)
        self.assertEqual(accelerator["ncgd_converged"], 0)
        self.assertIs(accelerator["physical_hardware_executed"], False)
        self.assertIs(accelerator["new_aig_or_netlist_created"], False)
        protocol = companion["v8_1_protocol_requalification"]
        self.assertEqual(protocol["records_requalified"], 4000)
        self.assertEqual(protocol["candidate_geometry_route_ready"], 167)
        self.assertEqual(protocol["structural_protocol_blocked"], 3833)
        self.assertEqual(protocol["method_template_records_quarantined"], 334)
        self.assertIs(protocol["pdb_5t35_panel_geometry_eligible"], False)
        self.assertIs(protocol["pdb_5fqd_panel_geometry_eligible"], False)
        self.assertIs(protocol["target_pose_computed"], False)
        scale = companion["requested_1000_per_target_per_modality_scale"]
        self.assertEqual(scale["requested_record_slots"], 24000)
        self.assertEqual(scale["druggable_candidates_established"], 0)
        expansion = companion["v8_1_expansion_readiness"]
        self.assertEqual(expansion["cells_ready_for_1000"], 0)
        self.assertEqual(expansion["internal_diversity_seeds"], 875)
        self.assertEqual(expansion["diversity_and_protocol_seeds"], 52)
        self.assertIs(expansion["global_novelty_claim"], False)
        self.assertIs(expansion["druggability_claim"], False)
        self.assertEqual(
            sum(
                companion[
                    "v8_planned_slots_not_materialized_candidates"
                ].values()
            ),
            3000,
        )
        v9_slots = companion["v9_design_slot_ledger"]
        self.assertEqual(v9_slots["total_design_slots"], 24000)
        self.assertEqual(v9_slots["v7_v8_lineage_references"], 4000)
        self.assertEqual(v9_slots["empty_design_slots"], 20000)
        self.assertEqual(v9_slots["v9_identity_ready"], 0)
        self.assertEqual(v9_slots["druggable_candidates_established"], 0)
        v9_sources = companion["v9_real_source_snapshot"]
        self.assertEqual(v9_sources["chembl_db_version"], "ChEMBL_37")
        self.assertIs(v9_sources["declared_activity_query_fully_paginated"], True)
        self.assertEqual(v9_sources["raw_payloads"], 420)
        self.assertEqual(v9_sources["raw_payload_bytes"], 244652474)
        self.assertEqual(
            v9_sources["chembl_reported_reference_identities"], 26543
        )
        self.assertEqual(
            v9_sources["source_backed_diverse_chemotype_anchors"], 9002
        )
        self.assertEqual(v9_sources["rdkit_chembl_inchikey_match"], 26502)
        self.assertEqual(
            v9_sources["rdkit_chembl_inchikey_mismatch_retained"], 41
        )
        self.assertIs(v9_sources["null_variant_inferred_as_wild_type"], False)
        self.assertIs(v9_sources["synthetic_assay_or_patient_data_used"], False)
        self.assertIs(v9_sources["icloud_account_sync_completion_claimed"], False)
        v9_routes = companion["v9_genotype_isoform_and_modality_routes"]
        self.assertEqual(v9_routes["flt3_itd_structure_route"], "BLOCKED")
        self.assertEqual(v9_routes["esr1_mgprotac_reference_pdb"], "9SV3")
        self.assertEqual(
            set(v9_routes["kras_protac_reference_pdbs"]),
            {"8QU8", "9RKE", "9RKN", "9RKC"},
        )
        self.assertIs(
            v9_routes["reference_structure_activates_arbitrary_candidate"], False
        )
        self.assertEqual(v9_routes["cells_safe_expansion_ready"], 0)
        openreview = companion["v9_openreview_and_lineage_migration"]
        self.assertEqual(openreview["recommendation"], "MAJOR_REVISION")
        self.assertEqual(openreview["lineage_migration_blocked"], 4000)
        self.assertEqual(openreview["dual_geometry_truth_conflict_records"], 1002)
        self.assertEqual(openreview["biology_exact_unique_values"], 24)
        self.assertIs(openreview["expansion_authorized"], False)
        v91 = companion["v9_1_kras_g12d_targeted_small_molecule_computational_priority"]
        self.assertIs(v91["clinical_grade"], False)
        self.assertEqual(
            v91["campaign_scope"], "KRAS_G12D_targeted_small_molecule_only"
        )
        self.assertIs(v91["paper_scope_preserved"], True)
        self.assertEqual(v91["priority_label"], "computational_priority")
        self.assertEqual(v91["primary_geometry_pdb"], "9BL0")
        self.assertEqual(v91["covalent_observation_only_pdb"], "9GBJ")
        self.assertEqual(v91["accepted_count"], 494)
        self.assertLess(v91["representative_count"], v91["accepted_count"])
        self.assertIs(v91["druggability_claim"], False)
        self.assertIs(v91["global_novelty_claim"], False)
        self.assertIs(v91["clinical_readiness_claim"], False)
        self.assertIs(v91["target_pose_is_affinity_claim"], False)
        self.assertIs(v91["surechembl_full_15gb_snapshot_claimed"], False)
        self.assertTrue(
            str(v91["records_jsonl"]).endswith("ACCEPTED_CANDIDATES.jsonl.gz")
        )
        self.assertTrue(str(v91["records_parquet"]).endswith(".parquet"))
        self.assertTrue(str(v91["structures_sdf"]).endswith(".sdf.gz"))
        self.assertEqual(len(v91["companion_git_sha_v9_1"]), 40)
        int(v91["companion_git_sha_v9_1"], 16)
        self.assertIn(
            "foliation-er100-multimodal-chemistry", companion["companion_repo"]
        )
        self.assertEqual(len(companion["companion_git_sha_v9"]), 40)
        int(companion["companion_git_sha_v9"], 16)
        self.assertFalse(
            (ROOT / "releases").exists(),
            "bulk candidate releases must stay in the companion evidence repo",
        )


if __name__ == "__main__":
    unittest.main()
