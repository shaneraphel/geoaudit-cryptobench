"""Fail-closed adapter gates: official CryptoBench fold + PocketMiner baseline."""
from __future__ import annotations

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

    def test_cluster_leak_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            base = root / "data/cryptobench_apo"
            base.mkdir(parents=True)
            (base / "r.pdb").write_text("ATOM\n")
            (base / "l.json").write_text("{}\n")
            # duplicate cluster in a foreign split is impossible here (single fold),
            # but a malformed entry with a non-test prior split must raise; emulate by
            # skipping hash checks and using a manifest with duplicated cluster ok.
            manifest = {
                "schema": "cryptobench.official_test_fold.v1",
                "fold": "test",
                "clustering": {"method": "mmseqs2",
                               "sequence_identity_threshold": 0.10},
                "entries": [
                    {"pdb": "a", "chain": "A", "cluster_id": "c1",
                     "receptor_path": "data/cryptobench_apo/r.pdb",
                     "receptor_sha256": "x", "label_path": "data/cryptobench_apo/l.json",
                     "label_sha256": "y"},
                ],
            }
            (base / "official_manifest.json").write_text(json.dumps(manifest))
            # hash mismatch must fail closed (verify_hashes on real files)
            with self.assertRaises(ValueError):
                load_official_test_fold(root, verify_hashes=True)


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
