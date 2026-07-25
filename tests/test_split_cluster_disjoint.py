"""Fail-closed cluster-disjoint split gate.

For every group that declares ``cluster_disjoint_required=true`` the gate asserts
that no ``cluster_id`` appears in more than one split. A group that is honestly
NOT disjoint must instead declare ``split_integrity_passed=false`` with a stated
reason — it can never silently omit the flag. Both paths are enforced here so a
future edit that introduces leakage (or hides it) fails CI.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "data/manifests/SPLIT_LEDGER.json"


def cluster_split_map(assignments: list[dict]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for a in assignments:
        out.setdefault(a["cluster_id"], set()).add(a["split"])
    return out


def overlapping_clusters(assignments: list[dict]) -> dict[str, set[str]]:
    return {c: s for c, s in cluster_split_map(assignments).items() if len(s) > 1}


class TestSplitClusterDisjoint(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = json.loads(LEDGER.read_text())
        self.groups = self.ledger["groups"]

    def test_ledger_shape(self) -> None:
        self.assertEqual(self.ledger["schema"], "geoaudit.split_ledger.v1")
        self.assertIs(self.ledger["clinical_grade"], False)
        self.assertTrue(self.groups)

    def test_disjoint_required_groups_are_disjoint(self) -> None:
        for name, g in self.groups.items():
            if not g.get("cluster_disjoint_required"):
                continue
            overlap = overlapping_clusters(g["assignments"])
            self.assertEqual(
                overlap, {},
                f"group '{name}' requires cluster-disjoint splits but these "
                f"cluster_ids span multiple splits: {overlap}",
            )

    def test_non_disjoint_groups_declare_integrity_false(self) -> None:
        # A group that is not disjoint may NOT hide it: it must declare
        # split_integrity_passed=false AND give a reason.
        for name, g in self.groups.items():
            if g.get("cluster_disjoint_required"):
                continue
            self.assertIs(
                g.get("split_integrity_passed"), False,
                f"non-disjoint group '{name}' must declare split_integrity_passed=false",
            )
            self.assertTrue(
                str(g.get("reason_not_disjoint") or "").strip(),
                f"non-disjoint group '{name}' must state reason_not_disjoint",
            )

    def test_every_assignment_has_required_fields(self) -> None:
        for name, g in self.groups.items():
            for a in g["assignments"]:
                for field in ("pdb_id", "split", "cluster_id"):
                    self.assertIn(field, a, f"group '{name}' assignment missing {field}")


if __name__ == "__main__":
    unittest.main()
