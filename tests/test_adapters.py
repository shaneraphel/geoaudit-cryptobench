"""Fail-closed adapter gates: official CryptoBench fold + PocketMiner baseline."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pocket_bench.adapters import (
    DataUnavailable,
    load_official_test_fold,
    load_pocketminer_scores,
    official_fold_available,
    pocketminer_available,
)


def _entries() -> list[dict]:
    return [
        {"pdb": "aaaa", "chain": "A", "cluster_id": "P00001"},
        {"pdb": "bbbb", "chain": "B", "cluster_id": "P00002"},
    ]


def _write_manifest(root: Path, entries: list[dict], *,
                    receptor_sha: str | None = None,
                    extra: dict | None = None) -> Path:
    """Materialize a syntactically valid official-fold manifest under ``root``."""
    base = root / "data/cryptobench_apo"
    base.mkdir(parents=True, exist_ok=True)
    out = []
    for e in entries:
        stem = f"{e['pdb']}_{e['chain']}"
        rec = base / f"{stem}_receptor.pdb"
        lab = base / f"{stem}_labels.json"
        rec.write_text(f"ATOM  {stem}\n")
        lab.write_text(json.dumps({"pdb_id": e["pdb"], "chain": e["chain"],
                                   "cryptic_residues": [1, 2]}) + "\n")
        row = dict(e)
        row.update({
            "receptor_path": str(rec.relative_to(root)),
            "receptor_sha256": receptor_sha or hashlib.sha256(rec.read_bytes()).hexdigest(),
            "label_path": str(lab.relative_to(root)),
            "label_sha256": hashlib.sha256(lab.read_bytes()).hexdigest(),
        })
        out.append(row)
    manifest = {
        "schema": "cryptobench.official_test_fold.v1",
        "fold": "test",
        "clustering": {"method": "mmseqs2", "sequence_identity_threshold": 0.10},
        "entries": out,
    }
    manifest.update(extra or {})
    (base / "official_manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


class TestOfficialFold(unittest.TestCase):
    def test_absent_manifest_fails_closed_with_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertFalse(official_fold_available(root))
            with self.assertRaises(DataUnavailable) as ctx:
                load_official_test_fold(root)
            msg = str(ctx.exception)
            self.assertIn("official_manifest.json", msg)
            self.assertIn("mmseqs2", msg)
            self.assertIn("0.10", msg)

    def test_wrong_threshold_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "data/cryptobench_apo").mkdir(parents=True)
            (root / "data/cryptobench_apo/official_manifest.json").write_text(
                json.dumps({
                    "schema": "cryptobench.official_test_fold.v1",
                    "fold": "test",
                    "clustering": {"method": "mmseqs2",
                                   "sequence_identity_threshold": 0.30},
                    "entries": [{"pdb": "x", "chain": "A", "cluster_id": "c1",
                                 "receptor_path": "r", "receptor_sha256": "0",
                                 "label_path": "l", "label_sha256": "0"}],
                })
            )
            with self.assertRaises(ValueError):
                load_official_test_fold(root)

    def test_hash_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = _write_manifest(Path(d), _entries(), receptor_sha="deadbeef")
            with self.assertRaises(ValueError) as ctx:
                load_official_test_fold(root, verify_hashes=True)
            self.assertIn("SHA-256", str(ctx.exception))

    def test_clean_single_split_manifest_loads(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = _write_manifest(Path(d), _entries())
            manifest = load_official_test_fold(root, verify_hashes=True)
            self.assertEqual(len(manifest["entries"]), 2)

    def test_cluster_spanning_two_splits_rejected(self) -> None:
        """Deliberate cross-split overlap: same cluster_id in test AND train."""
        entries = _entries()
        entries[0]["split"] = "test"
        entries[1]["split"] = "train"
        entries[1]["cluster_id"] = entries[0]["cluster_id"]  # inject the overlap
        with tempfile.TemporaryDirectory() as d:
            root = _write_manifest(Path(d), entries)
            with self.assertRaises(ValueError) as ctx:
                load_official_test_fold(root, verify_hashes=True)
            msg = str(ctx.exception)
            self.assertIn("spans splits", msg)
            self.assertIn("leakage", msg)

    def test_cluster_declared_in_foreign_split_rejected(self) -> None:
        """A test cluster also declared in the train split must fail closed."""
        entries = _entries()
        with tempfile.TemporaryDirectory() as d:
            root = _write_manifest(
                Path(d), entries,
                extra={"foreign_split_clusters": {entries[0]["cluster_id"]: "train"}},
            )
            with self.assertRaises(ValueError) as ctx:
                load_official_test_fold(root, verify_hashes=True)
            self.assertIn("spans splits", str(ctx.exception))


class TestPocketMiner(unittest.TestCase):
    def test_absent_dir_fails_closed_with_instructions(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.assertFalse(pocketminer_available(root))
            with self.assertRaises(DataUnavailable) as ctx:
                load_pocketminer_scores("3mo7", "A", root)
            self.assertIn("PocketMiner", str(ctx.exception))

    def test_json_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pm = root / "data/baselines/pocketminer"
            pm.mkdir(parents=True)
            (pm / "3mo7_A.json").write_text(
                json.dumps({"pdb": "3mo7", "chain": "A",
                            "residue_scores": {"10": 0.9, "11": 0.1}})
            )
            scores = load_pocketminer_scores("3mo7", "A", root)
            self.assertEqual(scores, {10: 0.9, 11: 0.1})

    def test_csv_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pm = root / "data/baselines/pocketminer"
            pm.mkdir(parents=True)
            (pm / "3mo7_A.csv").write_text("resseq,score\n10,0.8\n11,0.2\n")
            scores = load_pocketminer_scores("3mo7", "A", root)
            self.assertEqual(scores, {10: 0.8, 11: 0.2})

    def test_out_of_range_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pm = root / "data/baselines/pocketminer"
            pm.mkdir(parents=True)
            (pm / "x_A.json").write_text(
                json.dumps({"residue_scores": {"1": 1.7}})
            )
            with self.assertRaises(ValueError):
                load_pocketminer_scores("x", "A", root)


if __name__ == "__main__":
    unittest.main()
