"""Every training-fold artifact declares itself, and the gate knows six spellings.

The audit behind this file found three things, and the tests below fix each so
that it cannot come back:

* twenty-seven artifacts under the training-fold directory carried no
  ``reads_test_fold`` at all, including the whole ``COUNTERATTACK_*`` series;
* several of them had in fact read the held-out fold, and were invisible to the
  test-fold access ledger because its rule looked for a per-unit table keyed
  ``unit_id`` with a column containing ``auc``, or a declared read index;
* the tree spells the question eighteen ways, six of which answer this one, so a
  gate that greps for a single field name reports honest artifacts as silent.

The last is the general lesson and the one most likely to recur, so it is tested
directly: the synonym list must contain every spelling actually present in
``results/``, computed from the tree rather than from this docstring.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import build_test_fold_ledger as ledger  # noqa: E402
import train_fold_declarations as tfd  # noqa: E402


def _artifacts() -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(tfd.SWEEP / "*.json")))
            if not os.path.basename(p).startswith("_")]


class TestTheAuditIsClosed(unittest.TestCase):

    def test_every_sweep_artifact_declares_both_fields(self) -> None:
        missing = []
        for p in _artifacts():
            d = json.loads(p.read_text())
            if "clinical_grade" not in d:
                missing.append((p.name, "clinical_grade"))
            if not any(s in d for s in tfd.READ_SYNONYMS):
                missing.append((p.name, "reads_test_fold"))
        self.assertEqual(missing, [])

    def test_the_gate_passes_on_the_tree(self) -> None:
        self.assertEqual(tfd.audit(), [])

    def test_nothing_the_ledger_records_claims_it_did_not_read(self) -> None:
        """The one error this tool could make that would be worse than silence."""
        for p in _artifacts():
            if not tfd.in_the_ledger(p):
                continue
            d = json.loads(p.read_text())
            for s in tfd.READ_SYNONYMS:
                if s in d and isinstance(d[s], bool):
                    self.assertTrue(
                        d[s],
                        f"{p.name} is listed in the test-fold access ledger and "
                        f"declares {s}=false")

    def test_the_sweep_artifacts_that_read_the_fold_say_so(self) -> None:
        expected = {
            "BANK_FUSION.json", "DUAL_TRACK_AB.json",
            "FEATURE_CEILING_DIAGNOSIS.json", "FULL_EXPANSION.json",
            "MULTISCALE_GATE.json", "THRESHOLD_GATE.json",
            "WIDE_BUS_BANK.json",
            # Plan for the resolution-stratified pLM-NN deficit read; the read
            # itself is RESOLUTION_READ.json under official_fold/.
            "PREREGISTERED_RESOLUTION.json",
        }
        true_ones = set()
        for p in _artifacts():
            d = json.loads(p.read_text())
            if d.get("reads_test_fold") is True:
                true_ones.add(p.name)
        self.assertEqual(true_ones, expected)


class TestTheSynonymListCoversTheTree(unittest.TestCase):
    """A gate that greps one spelling measures the spelling."""

    def _spellings_present(self) -> set[str]:
        found = set()
        for p in glob.glob(str(ROOT / "results/**/*.json"), recursive=True):
            if "p2rank_raw" in p:
                continue
            try:
                d = json.loads(Path(p).read_text())
            except Exception:  # noqa: BLE001
                continue
            if not isinstance(d, dict):
                continue
            for k in d:
                if k in ("reads_test_fold", "test_fold_touched",
                         "test_fold_read", "test_fold_reads",
                         "reads_our_test_fold", "reads_cryptobench_test_fold"):
                    found.add(k)
        return found

    def test_every_spelling_in_the_tree_is_in_the_list(self) -> None:
        self.assertTrue(self._spellings_present() <= set(tfd.READ_SYNONYMS))

    def test_the_canonical_spelling_is_first(self) -> None:
        self.assertEqual(tfd.READ_SYNONYMS[0], "reads_test_fold")

    def test_the_list_has_no_duplicates(self) -> None:
        self.assertEqual(len(set(tfd.READ_SYNONYMS)), len(tfd.READ_SYNONYMS))


class TestTheLedgersThirdSignal(unittest.TestCase):

    def test_an_aggregate_only_fold_metric_is_an_access(self) -> None:
        d = {"schema": "x", "n_test_units": 192,
             "results": [{"roc_auc": 0.7}]}
        self.assertTrue(ledger._reports_a_fold_metric(Path("X.json"), d))

    def test_a_metric_on_a_different_fold_is_not(self) -> None:
        d = {"schema": "x", "n_units": 770, "results": [{"roc_auc": 0.7}]}
        self.assertFalse(ledger._reports_a_fold_metric(Path("X.json"), d))

    def test_the_fold_size_without_a_metric_is_not(self) -> None:
        d = {"schema": "x", "n_units": 192, "note": "the fold has 192 units"}
        self.assertFalse(ledger._reports_a_fold_metric(Path("X.json"), d))

    def test_the_named_exemptions_carry_a_reason(self) -> None:
        for name, why in ledger._NOT_AN_ACCESS.items():
            self.assertTrue(name.endswith(".json"), name)
            self.assertGreater(len(why), 40,
                               f"{name} is exempted without a reason worth "
                               f"reading")

    def test_an_exempted_file_is_refused_by_the_third_signal(self) -> None:
        d = {"schema": "x", "n_units": 192, "auc": 0.8}
        for name in ledger._NOT_AN_ACCESS:
            self.assertFalse(ledger._reports_a_fold_metric(Path(name), d))

    def test_the_row_rule_accepts_the_spelling_that_escaped(self) -> None:
        """DUAL_TRACK_AB spells the key `unit` and the metric `A_resolved`."""
        rows = [{"unit": None, "A_resolved": 0.8} for _ in range(192)]
        keys = set(rows[0])
        self.assertTrue(any(k in keys for k in ("unit_id", "unit", "pdb")))
        self.assertTrue(any(("auc" in k or "resolved" in k) for k in keys))

    def test_the_ledger_does_not_list_itself(self) -> None:
        led = json.loads(tfd.LEDGER.read_text())
        names = {r["artifact"] for r in led["standalone_probe_artifacts"]}
        self.assertNotIn(str(tfd.LEDGER.relative_to(ROOT)), names)

    def test_the_probe_count_matches_the_list(self) -> None:
        led = json.loads(tfd.LEDGER.read_text())
        self.assertEqual(led["n_standalone_probes"],
                         len(led["standalone_probe_artifacts"]))


class TestGeneratorDetection(unittest.TestCase):
    """A reader that binds a path is not the tool that wrote the artifact."""

    def test_a_reader_is_not_counted_as_a_generator(self) -> None:
        gens = tfd.generators_of(tfd.SWEEP / "GAP_DECOMPOSITION.json")
        self.assertNotIn("emit_frozen_numbers.py", [g.name for g in gens])

    def test_the_writer_is_found(self) -> None:
        gens = [g.name for g in
                tfd.generators_of(tfd.SWEEP / "GAP_DECOMPOSITION.json")]
        self.assertIn("gap_decomposition.py", gens)

    def test_a_generator_naming_a_fold_path_is_reported(self) -> None:
        marks = tfd.touches_the_fold(ROOT / "tools/run_full_expansion.py")
        self.assertIn("_cascade_cache_test", marks)

    def test_a_selection_provenance_mention_is_not_membership(self) -> None:
        """COUNTERATTACK_QUOTIENT.json chose an architecture; it read nothing.

        Its path appears inside a probe's selection_provenance, so a substring
        search over the ledger calls it an access and the gate then demands it
        declare true. It correctly declares false.
        """
        p = tfd.SWEEP / "COUNTERATTACK_QUOTIENT.json"
        self.assertTrue(p.is_file())
        self.assertIn(str(p.relative_to(ROOT)), tfd.LEDGER.read_text())
        self.assertFalse(tfd.in_the_ledger(p))
        self.assertIs(json.loads(p.read_text())["reads_test_fold"], False)


if __name__ == "__main__":
    unittest.main()
