"""Regression tests for the deterministic geometric-foundation pocket detector."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from pocket_bench.methods import geometric_foundation
from pocket_bench.metrics import score_prediction

ROOT = Path(__file__).resolve().parents[1]


class TestGeometricFoundation(unittest.TestCase):
    def _write_shell_pdb(self, tmp: Path) -> Path:
        """A hollow sphere of CA atoms => a clear enclosed central cavity."""
        pts = []
        for i in range(200):
            v = np.array(
                [np.cos(i) * np.sin(i * 1.3), np.sin(i) * np.sin(i * 1.3), np.cos(i * 1.3)]
            )
            v = v / (np.linalg.norm(v) + 1e-9) * 9.0
            pts.append(v)
        p = tmp / "shell.pdb"
        with p.open("w") as fh:
            for k, v in enumerate(pts, start=1):
                fh.write(
                    f"ATOM  {k:5d}  CA  ALA A{k:4d}    "
                    f"{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}  1.00  0.00           C\n"
                )
            fh.write("END\n")
        return p

    def test_output_contract_and_determinism(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            pdb = self._write_shell_pdb(Path(d))
            a = geometric_foundation.predict(pdb, pdb_id="SHEL")
            b = geometric_foundation.predict(pdb, pdb_id="SHEL")
        self.assertEqual(a["status"], "OK")
        self.assertGreaterEqual(len(a["pockets"]), 1)
        # deterministic: identical top-1 centre across runs (no RNG)
        self.assertEqual(a["pockets"][0]["center_xyz"], b["pockets"][0]["center_xyz"])
        # top pocket should sit in the hollow interior (radius-9 shell), not on it
        c = np.array(a["pockets"][0]["center_xyz"])
        self.assertLess(float(np.linalg.norm(c)), 6.0)

    def test_recovers_esr1_pocket_top1(self):
        rec = ROOT / "data/receptors/3ERT_A_receptor.pdb"
        lab = ROOT / "data/labels/3ERT_A_labels.json"
        if not (rec.exists() and lab.exists()):
            self.skipTest("ESR1 receptor/label not present")
        pred = geometric_foundation.predict(rec, pdb_id="3ERT")
        scored = score_prediction(pred, json.loads(lab.read_text()))
        top1 = scored.get("top1") or {}
        self.assertIs(top1.get("success"), True)
        self.assertLessEqual(float(top1["best_dca"]), 4.0)


if __name__ == "__main__":
    unittest.main()
