"""The architecture choice must not be an artefact of the split that made it.

Nine candidates were compared on one seeded half-split of the training fold and
the winner was frozen. That keeps the test fold out of the choice, but on its
own it does not distinguish a selected architecture from a lucky draw: the
winner carries the maximum of nine estimates and the top few are close.

``tools/crossvalidate_architecture.py`` repeats the selection on CryptoBench's
own four training folds -- the benchmark authors' MMseqs2 10 % partition, not a
proxy for it -- and on 25 further accession-disjoint halves. These tests hold
the artifact to its own tables, and hold the arithmetic behind the headline to
cases where the answer is known by hand.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import crossvalidate_architecture as cv  # noqa: E402

ARTIFACT = ROOT / "results/architecture_sweep/REPEATED_TRAIN_SELECTION.json"
FROZEN = ROOT / "results/architecture_sweep/TRAIN_ONLY_SELECTION.json"


def _table(pairs: list[tuple[str, float]]) -> list[dict]:
    ordered = sorted(pairs, key=lambda kv: -kv[1])
    return [{"architecture": n, "roc_auc": a, "rank": i}
            for i, (n, a) in enumerate(ordered, 1)]


class TestTheArithmeticBehindTheHeadline(unittest.TestCase):
    """Known-answer cases, so the summary is not merely self-consistent."""

    def test_a_candidate_that_always_wins_is_reported_as_always_winning(self):
        tables = [_table([("a", 0.70 + i / 1000), ("b", 0.60), ("c", 0.50)])
                  for i in range(5)]
        v = cv.verdict_for("a", tables, cv.summarise(tables))
        self.assertEqual((v["n_first"], v["n_splits"], v["worst_rank"]),
                         (5, 5, 1))

    def test_a_candidate_that_loses_once_is_not_reported_as_unbeaten(self):
        tables = [_table([("a", 0.70), ("b", 0.60)]) for _ in range(4)]
        tables.append(_table([("a", 0.60), ("b", 0.70)]))
        v = cv.verdict_for("a", tables, cv.summarise(tables))
        self.assertEqual(v["n_first"], 4)
        self.assertEqual(v["worst_rank"], 2)

    def test_the_margin_is_against_the_best_alternative_not_the_worst(self):
        tables = [_table([("a", 0.80), ("b", 0.79), ("c", 0.10)])]
        v = cv.verdict_for("a", tables, cv.summarise(tables))
        self.assertAlmostEqual(v["mean_margin_over_runner_up"], 0.01, places=9)

    def test_a_negative_margin_is_reported_as_negative(self):
        # A loss must not be laundered into a small win by an absolute value.
        tables = [_table([("b", 0.80), ("a", 0.70)])]
        v = cv.verdict_for("a", tables, cv.summarise(tables))
        self.assertAlmostEqual(v["worst_margin_over_runner_up"], -0.10,
                               places=9)

    def test_architectures_are_ordered_by_mean_rank(self):
        tables = [_table([("a", 0.70), ("b", 0.60), ("c", 0.50)]),
                  _table([("a", 0.70), ("c", 0.65), ("b", 0.50)])]
        self.assertEqual([r["architecture"] for r in cv.summarise(tables)],
                         ["a", "b", "c"])


class TestTheArtifactHoldsUp(unittest.TestCase):
    def setUp(self) -> None:
        if not ARTIFACT.exists():
            self.skipTest(f"{ARTIFACT.name} not present")
        self.rec = json.loads(ARTIFACT.read_text())

    def test_the_audit_passes(self) -> None:
        self.assertEqual(
            cv.audit(), 0,
            "the cross-validation artifact does not agree with its own tables "
            "or no longer names the frozen architecture")

    def test_it_did_not_read_the_test_fold(self) -> None:
        self.assertIs(self.rec["reads_test_fold"], False)
        self.assertIs(self.rec["clinical_grade"], False)

    def test_it_cross_validates_the_architecture_that_was_frozen(self) -> None:
        frozen = json.loads(FROZEN.read_text())["selected"]["architecture"]
        self.assertEqual(self.rec["frozen_choice"]["architecture"], frozen)

    def test_the_cluster_level_folds_are_cryptobenchs_own(self) -> None:
        # The point of this block is that the partition is the benchmark's,
        # under its MMseqs2 clustering, rather than an accession proxy. If it
        # ever stops holding out whole published folds it stops being that.
        block = self.rec["cluster_level_cv"]
        held = [f["held_out_fold"] for f in block["folds"]]
        self.assertEqual(held, list(cv.TRAIN_FOLDS))
        self.assertEqual(block["n_folds"], len(cv.TRAIN_FOLDS))

    def test_every_split_ranks_the_same_candidates(self) -> None:
        expected = {c["architecture"]
                    for c in json.loads(FROZEN.read_text())["candidates"]}
        for block in ("cluster_level_cv", "repeated_halves"):
            b = self.rec[block]
            for entry in (b.get("folds") or b.get("splits")):
                self.assertEqual(
                    {r["architecture"] for r in entry["ranking"]}, expected,
                    f"{block}: a split compares a different candidate set")

    def test_the_margin_over_the_runner_up_is_recorded(self) -> None:
        # Not that it is large. It is not: the runner-up is the same bank under
        # the other fusion rule and sits a few thousandths behind. The claim
        # this artifact supports is that the ordering is stable, and a reader
        # can only weigh that against the margin if the margin is stated.
        for block in ("cluster_level_cv", "repeated_halves"):
            v = self.rec[block]["frozen_choice"]
            for field in ("mean_margin_over_runner_up",
                          "worst_margin_over_runner_up"):
                self.assertIsInstance(v[field], float, f"{block}: no {field}")


if __name__ == "__main__":
    unittest.main()
