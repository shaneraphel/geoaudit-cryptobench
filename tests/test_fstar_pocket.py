"""Contract + determinism tests for the F*-breathing apo pocket ablation.

fstar_pocket is retained as a documented NEGATIVE-RESULT ablation (it
under-performs the rigid detector on the CryptoBench-apo pilot). These tests only
assert it runs deterministically and honours the output contract — no hit claim.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from pocket_bench.methods import fstar_pocket


def _shell_pdb(tmp: Path) -> Path:
    p = tmp / "shell.pdb"
    with p.open("w") as fh:
        for k in range(200):
            v = np.array([np.cos(k) * np.sin(k * 1.3), np.sin(k) * np.sin(k * 1.3), np.cos(k * 1.3)])
            v = v / (np.linalg.norm(v) + 1e-9) * 9.0
            fh.write(
                f"ATOM  {k+1:5d}  CA  ALA A{k+1:4d}    "
                f"{v[0]:8.3f}{v[1]:8.3f}{v[2]:8.3f}  1.00  0.00           C\n"
            )
        fh.write("END\n")
    return p


class TestFStarPocket(unittest.TestCase):
    def test_contract_and_determinism(self):
        with tempfile.TemporaryDirectory() as d:
            pdb = _shell_pdb(Path(d))
            a = fstar_pocket.predict(pdb, pdb_id="SHEL")
            b = fstar_pocket.predict(pdb, pdb_id="SHEL")
        self.assertEqual(a["status"], "OK")
        self.assertGreaterEqual(len(a["pockets"]), 1)
        self.assertEqual(a["pockets"][0]["center_xyz"], b["pockets"][0]["center_xyz"])


if __name__ == "__main__":
    unittest.main()
