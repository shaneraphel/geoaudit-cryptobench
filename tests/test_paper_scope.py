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
        self.assertIn(
            "foliation-er100-multimodal-chemistry", companion["companion_repo"]
        )
        self.assertFalse(
            (ROOT / "releases").exists(),
            "bulk candidate releases must stay in the companion evidence repo",
        )


if __name__ == "__main__":
    unittest.main()
