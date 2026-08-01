"""The gate that notices when a compiled field stops naming its own source.

Two Rust kernels were once written directly into ``table_bank.py``. They were
bit-identical ports, no number moved, and the edit was still wrong: that file is one
of the ten whose bytes ``table_field.code_sha256`` hashes, so ``TABLE_FIELD.json``,
``GEOMETRY_FIELD.json`` and every frozen external prediction were left pointing at
source that no longer existed.

``make verify`` did catch it, but through a per-unit ``tool_version`` mismatch on one
Set A prediction --- a symptom that names a unit rather than a cause, and one that
only exists because Set A happens to have been read. The gate tested here states the
rule itself, and the accelerator now lives in ``table_bank_accel.py``, outside the
digest.
"""
from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from verify_claims import frozen_field_digest_checks  # noqa: E402
from pocket_bench.methods import table_field  # noqa: E402

FIELDS = ("data/cryptobench_apo/TABLE_FIELD.json",
          "data/cryptobench_apo/GEOMETRY_FIELD.json")


class TestTheDigestStillMatches(unittest.TestCase):

    def test_the_real_tree_is_clean(self) -> None:
        """The daily assertion: nothing under the digest has drifted."""
        self.assertEqual(frozen_field_digest_checks(ROOT), [])

    def test_every_compiled_field_records_the_live_digest(self) -> None:
        live = table_field.code_sha256()
        for rel in FIELDS:
            path = ROOT / rel
            if not path.exists():
                continue
            with self.subTest(field=rel):
                self.assertEqual(
                    json.loads(path.read_text()).get("code_sha256"), live)


class TestTheGateFiresOnAMismatch(unittest.TestCase):
    """A gate that cannot be shown to fail cannot be shown to work."""

    def _tree(self, digests: dict[str, str]) -> Path:
        root = Path(self.enterContext(
            __import__("tempfile").TemporaryDirectory())) / "tree"
        (root / "data/cryptobench_apo").mkdir(parents=True)
        for rel, digest in digests.items():
            doc = {"code_sha256": digest, "cells": [], "note": "synthetic"}
            (root / rel).write_text(json.dumps(doc))
        (root / "src").symlink_to(ROOT / "src")
        return root

    def test_a_stale_field_is_named(self) -> None:
        root = self._tree({FIELDS[0]: "0" * 64})
        problems = frozen_field_digest_checks(root)
        self.assertEqual(len(problems), 1)
        self.assertIn("TABLE_FIELD.json", problems[0])
        self.assertIn("0000000000000000", problems[0])

    def test_both_fields_are_reported_not_just_the_first(self) -> None:
        root = self._tree({FIELDS[0]: "0" * 64, FIELDS[1]: "1" * 64})
        problems = frozen_field_digest_checks(root)
        self.assertEqual(len(problems), 2)

    def test_a_matching_field_is_silent(self) -> None:
        root = self._tree({FIELDS[0]: table_field.code_sha256()})
        self.assertEqual(frozen_field_digest_checks(root), [])

    def test_an_absent_field_is_not_an_offence(self) -> None:
        """A fresh clone that has compiled nothing yet is not in violation."""
        root = self._tree({})
        self.assertEqual(frozen_field_digest_checks(root), [])


class TestTheAcceleratorIsOutsideTheDigest(unittest.TestCase):
    """The property the relocation exists to hold."""

    def test_the_digested_list_is_the_expected_ten_files(self) -> None:
        src = (ROOT / "src/pocket_bench/methods/table_field.py").read_text()
        body = src.split("def code_sha256")[1]
        for name in ("table_field.py", "table_bank.py", "wide_descriptors.py",
                     "spatial.py", "pdb_io.py"):
            self.assertIn(name, body)
        self.assertNotIn("table_bank_accel", body)

    def test_editing_the_accelerator_does_not_move_the_digest(self) -> None:
        accel = ROOT / "src/pocket_bench/methods/table_bank_accel.py"
        before = table_field.code_sha256()
        backup = accel.read_bytes()
        try:
            accel.write_bytes(backup + b"\n# a byte the digest must not see\n")
            self.assertEqual(table_field.code_sha256(), before)
        finally:
            accel.write_bytes(backup)

    def test_editing_a_digested_file_does_move_it(self) -> None:
        """The converse, so the test above is not vacuous."""
        bank = ROOT / "src/pocket_bench/methods/table_bank.py"
        before = table_field.code_sha256()
        backup = bank.read_bytes()
        try:
            bank.write_bytes(backup + b"\n# the edit that broke the link\n")
            self.assertNotEqual(table_field.code_sha256(), before)
        finally:
            bank.write_bytes(backup)
        self.assertEqual(table_field.code_sha256(), before)


class TestTheHarnessUsesTheAccelerator(unittest.TestCase):

    def test_straddling_takes_the_hot_pair_from_the_accel_module(self) -> None:
        text = (ROOT / "tools/straddling_attachment.py").read_text()
        self.assertIn(
            "from pocket_bench.methods.table_bank_accel import "
            "addresses, compile_cells", text)
        head = text.split("from pocket_bench.methods.table_field")[0]
        bank_import = head.split(
            "from pocket_bench.methods.table_bank import")[1].split(")")[0]
        for name in ("addresses", "compile_cells"):
            self.assertNotIn(name, bank_import)


if __name__ == "__main__":
    unittest.main()
