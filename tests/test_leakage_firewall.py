"""Zero-leakage firewall gates.

1. Import-graph gate: no predictor module may import the scorer, so a predictor
   is physically unable to reach label-joining code.
2. Decorator gate: a receptor carrying ligand atoms is refused (hard CRASH); a
   clean ATOM-only receptor runs and is stamped with firewall provenance.
"""
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METHODS_DIR = ROOT / "src" / "pocket_bench" / "methods"

# Any module in this set means the predictor could reach the label. Forbidden.
FORBIDDEN_SCORER_MODULES = {"pocket_bench.metrics", "pocket_bench.scoring"}


def _imports_of(py: Path) -> set[str]:
    tree = ast.parse(py.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module)
    return mods


class TestImportGraphFirewall(unittest.TestCase):
    def test_no_method_imports_the_scorer(self) -> None:
        offenders: dict[str, set[str]] = {}
        for py in sorted(METHODS_DIR.glob("*.py")):
            bad = _imports_of(py) & FORBIDDEN_SCORER_MODULES
            # also catch any `*.scoring` module regardless of package
            bad |= {m for m in _imports_of(py) if m.split(".")[-1] == "scoring"}
            if bad:
                offenders[py.name] = bad
        self.assertEqual(
            offenders, {}, f"predictor(s) import label-joining scorer: {offenders}"
        )


class TestLigandLeakGuard(unittest.TestCase):
    _HEADER = "ATOM      1  N   ALA A   1      11.104  13.207  10.000  1.00  0.00           N\n"

    def _write(self, tmp: Path, body: str) -> Path:
        p = tmp / "rec.pdb"
        p.write_text(self._HEADER + body + "END\n")
        return p

    def test_hetatm_ligand_triggers_crash(self) -> None:
        from pocket_bench.methods import geometric_foundation as gf
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            lig = "HETATM 2000  C1  OHT A 900      12.0    12.0    12.0  1.00  0.00           C\n"
            rec = self._write(tmp, lig)
            out = gf.predict(rec, pdb_id="leaky")
        self.assertEqual(out["status"], "CRASH")
        self.assertIn("ligand_leak_guard", out.get("error", ""))
        self.assertIn("OHT", out.get("error", ""))

    def test_smuggled_atom_ligand_triggers_crash(self) -> None:
        from pocket_bench.methods import fstar_pocket as fs
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            # a ligand resname (STI) hidden inside an ATOM record
            smuggled = "ATOM   2001  C1  STI A 900      12.0    12.0    12.0  1.00  0.00           C\n"
            rec = self._write(tmp, smuggled)
            out = fs.predict(rec, pdb_id="smuggled")
        self.assertEqual(out["status"], "CRASH")
        self.assertIn("non_polymer_atom_resnames", out.get("error", ""))

    def test_predictors_are_marked_firewalled(self) -> None:
        from pocket_bench.methods import geometric_foundation as gf
        from pocket_bench.methods import fstar_pocket as fs

        self.assertTrue(getattr(gf.predict, "__ligand_firewalled__", False))
        self.assertTrue(getattr(fs.predict, "__ligand_firewalled__", False))

    def test_clean_receptor_runs_and_is_stamped(self) -> None:
        from pocket_bench.methods import geometric_foundation as gf
        import tempfile

        # small clean poly-ALA shell so the predictor executes end-to-end
        body = ""
        for i in range(2, 60):
            body += (
                f"ATOM  {i:5d}  CA  ALA A{i:4d}      "
                f"{(i%7)*3.0:8.3f}{(i%5)*3.0:8.3f}{(i%3)*3.0:8.3f}  1.00  0.00           C\n"
            )
        with tempfile.TemporaryDirectory() as d:
            rec = self._write(Path(d), body)
            out = gf.predict(rec, pdb_id="clean")
        # The firewall must NOT block a clean receptor (no ligand_leak_guard error),
        # and every returned record must carry atom-only provenance regardless of
        # whether the detector itself found a pocket.
        self.assertNotIn("ligand_leak_guard", out.get("error") or "")
        self.assertFalse(out.get("input_has_hetatm"))
        self.assertTrue(out.get("input_atom_only_verified"))
        self.assertIn("input_receptor_sha256", out)


if __name__ == "__main__":
    unittest.main()
