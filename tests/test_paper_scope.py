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
        self.assertIn(
            "foliation-er100-multimodal-chemistry", companion["companion_repo"]
        )
        self.assertFalse(
            (ROOT / "releases").exists(),
            "bulk candidate releases must stay in the companion evidence repo",
        )


if __name__ == "__main__":
    unittest.main()
