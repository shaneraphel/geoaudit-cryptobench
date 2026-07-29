"""The pLM-NN baseline has to be theirs, and the comparison has to be able to lose.

Three things could go wrong here in ways no metric would reveal.

The network could be the wrong tensors. The published checkpoint holds the Adam
moments as well as the weights, three times as much data under adjacent names, and
a network assembled from a moment tensor runs and predicts noise. So the export is
checked against the shapes and the activations the graph itself states.

The encoder could be read at the wrong depth. The deposit never says which layer
its example embedding came from, and the final layer -- the obvious reading -- is
a hundredfold smaller in magnitude and matches nothing. So the recovered layer is
checked against the authors' own worked example.

The comparison could be written so that it can only win. So the plan is checked
for a sentence to print when the language model beats the counting field, and the
read is checked for printing the sentence its own outcome selects rather than a
sentence chosen afterwards.
"""
from __future__ import annotations

import json
import unittest

import numpy as np

from pocket_bench.paths import ROOT

import export_plmnn_weights as EW
import plmnn_embed as PE
import preregister_plmnn as PP

NETWORK = ROOT / "results/baselines/PLMNN_NETWORK.json"
SCORES = ROOT / "results/baselines/PLMNN_SCORES.json"
PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_PLMNN.json"
READ = ROOT / "results/official_fold/PLMNN_READ.json"
EXAMPLE = ROOT / "data/cryptobench_apo/_osf/cryptobench/scripts/data/7w19A.npy"
PER_STRUCTURE = ROOT / "results/official_fold/PER_STRUCTURE.json"


def _json(p):
    return json.loads(p.read_text())


class TheGatesPass(unittest.TestCase):
    def test_the_network_gate(self):
        self.assertEqual(EW.check(), 0)

    @unittest.skipUnless(SCORES.exists(), "the baseline has not been run here")
    def test_the_scores_gate(self):
        self.assertEqual(PE.check(), 0)

    @unittest.skipUnless(PLAN.exists(), "the plan has not been written here")
    def test_the_plan_gate(self):
        self.assertEqual(PP._check(), 0)


class TheNetworkIsTheirs(unittest.TestCase):
    def test_the_architecture_is_the_published_one(self):
        a = _json(NETWORK)["architecture"]
        self.assertEqual(a["widths"], [2560, 256, 256, 2])
        self.assertEqual(a["activations"], ["Relu", "Relu", "Softmax"])

    def test_the_input_width_pins_the_encoder(self):
        # No ESM-2 other than the 3B model emits 2560-wide vectors, so this width
        # is what makes the encoder choice a fact rather than a preference.
        self.assertEqual(_json(NETWORK)["architecture"]["widths"][0], 2560)

    def test_the_graph_holds_no_op_that_would_change_the_forward_pass(self):
        # A dropout live at inference, or a normalisation, would make the numpy
        # forward pass in this repository a different network from the one whose
        # weights it loads.
        ops = _json(NETWORK)["ops_found_in_the_graph"]
        allowed = {"MatMul", "BiasAdd", "ReadVariableOp", "kernel", "bias",
                   "Relu", "Softmax"}
        for layer, found in ops.items():
            self.assertTrue(set(found) <= allowed,
                            f"{layer} has unexpected ops: {set(found) - allowed}")

    @unittest.skipUnless(EW.NPZ.exists(), "the weights are not exported here")
    def test_a_moment_tensor_would_not_pass_as_a_weight(self):
        # The checkpoint's Adam moments have the same shapes as the weights, so
        # shape alone cannot catch a misnamed read. What catches it is that the
        # exported arrays hash to the digest recorded when they were read at the
        # offsets the checkpoint's index stated.
        w = EW.load()
        self.assertEqual(sorted(w), sorted(
            f"{n}_{r}" for n in EW.LAYERS for r in ("kernel", "bias")))

    @unittest.skipUnless(EW.NPZ.exists() and EXAMPLE.exists(),
                         "weights or the worked example are absent here")
    def test_the_forward_pass_is_a_probability_over_two_classes(self):
        p = EW.forward(np.load(EXAMPLE).astype(np.float64), EW.load())
        self.assertEqual(p.shape[1], 2)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-12)
        self.assertTrue(((p >= 0.0) & (p <= 1.0)).all())


@unittest.skipUnless(SCORES.exists(), "the baseline has not been run here")
class TheEncoderWasReadAtTheRightDepth(unittest.TestCase):
    def test_the_layer_was_recovered_and_not_assumed(self):
        v = _json(SCORES)["validation_against_the_published_example"]
        by_layer = v["mean_cosine_against_the_published_embedding_by_layer"]
        best = max(by_layer, key=lambda k: by_layer[k])
        self.assertEqual(int(best), v["layer_recovered"])
        self.assertEqual(int(best), _json(SCORES)["encoder"]["layer"])

    def test_the_match_is_not_ambiguous(self):
        # Neighbouring layers of a transformer are correlated, so the recovered
        # layer has to win by a margin rather than by a hair.
        v = _json(SCORES)["validation_against_the_published_example"]
        self.assertGreater(v["mean_cosine_at_that_layer"],
                           v["second_best_mean_cosine"] + 0.01)

    def test_the_final_layer_is_not_what_the_authors_used(self):
        # The obvious reading of "the ESM2-3B embedding" is the last layer, and
        # taking it would be a baseline scored on the wrong features. This records
        # that the obvious reading was checked and is wrong.
        by_layer = _json(SCORES)["validation_against_the_published_example"][
            "mean_cosine_against_the_published_embedding_by_layer"]
        self.assertNotEqual(max(by_layer, key=lambda k: by_layer[k]),
                            str(max(int(k) for k in by_layer)))

    def test_the_ranking_of_residues_agrees_with_the_authors(self):
        # Per-unit ROC-AUC reads only the order of the scores within a chain, so
        # this is the fidelity the comparison actually rests on.
        a = _json(SCORES)["validation_against_the_published_example"][
            "predicted_probability_agreement"]
        self.assertGreaterEqual(a["spearman"], PE.MIN_SPEARMAN)

    def test_the_residual_disagreement_is_declared_rather_than_hidden(self):
        v = _json(SCORES)["validation_against_the_published_example"]
        self.assertLess(v["mean_cosine_at_that_layer"], 1.0)
        self.assertIn("what_remains", v["alternatives_tested_and_excluded"])


@unittest.skipUnless(SCORES.exists(), "the baseline has not been run here")
class TheBaselineWasScoredOnOurUniverse(unittest.TestCase):
    def test_every_unit_covers_the_frozen_universe_exactly(self):
        per = {f"{r['pdb']}_{r['chain']}": r["n_universe"]
               for r in _json(PER_STRUCTURE)}
        for u in _json(SCORES)["units"]:
            self.assertEqual(len(u["scores"]), per[u["unit_id"]],
                             f"{u['unit_id']} was scored on a different universe")

    def test_the_insertion_code_chain_was_collapsed_by_a_declared_rule(self):
        # 2v6m_D has two embedding rows sharing one resseq. Whatever rule resolves
        # that has to be written down, because it is a choice about a rival
        # method's score.
        d = _json(SCORES)["residues_sharing_a_resseq"]
        self.assertIn("2v6m_D", d["units"])
        self.assertTrue(d["rule"].strip())


@unittest.skipUnless(PLAN.exists(), "the plan has not been written here")
class ThePlanCanLose(unittest.TestCase):
    def test_a_sentence_is_written_for_the_baseline_winning(self):
        s = _json(PLAN)["what_will_be_written_under_each_outcome"]
        self.assertIn("the_baseline_is_ahead", s)
        self.assertTrue(s["the_baseline_is_ahead"].strip())

    def test_losing_costs_the_abstract_and_the_plan_says_so(self):
        r = _json(PLAN)["decision_rules"][
            "if_the_baseline_is_ahead_and_the_interval_excludes_zero"]
        self.assertIn("abstract", r)

    def test_a_broken_reproduction_is_not_allowed_to_look_like_a_win(self):
        p = _json(PLAN)["numeric_parameters"]
        self.assertGreater(p["baseline_auc_floor"], 0.5)
        self.assertIn("the_reproduction_fails_its_floor",
                      _json(PLAN)["what_will_be_written_under_each_outcome"])

    def test_the_functional_is_the_one_already_declared_primary(self):
        # Switching to a robust summary for a new baseline, after the mean failed
        # to resolve against the old one, is the move this repository retracted.
        self.assertIn("mean", _json(PLAN)["statistic"]["primary_functional"])

    def test_the_plan_pins_the_scores_it_will_read(self):
        import hashlib
        self.assertEqual(
            _json(PLAN)["the_baseline"]["scores_artifact_sha256"],
            hashlib.sha256(SCORES.read_bytes()).hexdigest())

    def test_no_threshold_is_tuned_on_the_test_fold(self):
        p = _json(PLAN)["numeric_parameters"]
        self.assertEqual(p["top_q"], 0.09)
        self.assertEqual(p["plmnn_threshold"], 0.95)


@unittest.skipUnless(READ.exists(), "the tenth read has not been run here")
class TheReadPrintsTheSentenceItsOutcomeSelects(unittest.TestCase):
    def test_the_sentence_is_the_planned_one_for_this_outcome(self):
        d = _json(READ)
        self.assertEqual(
            d["sentence_fixed_in_advance"],
            _json(PLAN)["what_will_be_written_under_each_outcome"][d["outcome"]])

    def test_the_outcome_follows_from_the_interval(self):
        d = _json(READ)
        c = d["primary_comparison"]
        if not d["reproduction_gate"]["passes"]:
            self.assertEqual(d["outcome"], "the_reproduction_fails_its_floor")
        elif not c["excludes_zero"]:
            self.assertEqual(d["outcome"], "the_interval_crosses_zero")
        else:
            self.assertEqual(d["outcome"], "the_field_is_ahead"
                             if c["mean"] > 0 else "the_baseline_is_ahead")

    def test_the_read_is_exploratory(self):
        self.assertEqual(_json(READ)["status"], "exploratory")

    def test_the_corrected_interval_is_at_least_as_wide(self):
        for key in ("primary_comparison", "context_comparison"):
            c = _json(READ)[key]
            self.assertLessEqual(c["ci_bonferroni"][0], c["ci"][0] + 1e-12)
            self.assertGreaterEqual(c["ci_bonferroni"][1], c["ci"][1] - 1e-12)

    def test_the_win_loss_tie_counts_cover_every_paired_unit(self):
        for key in ("primary_comparison", "context_comparison"):
            c = _json(READ)[key]
            self.assertEqual(
                c["n_first_ahead"] + c["n_second_ahead"] + c["n_tied"], c["n"])

    def test_both_baselines_are_reported_against_each_other(self):
        # If P2Rank beats pLM-NN on this fold, that bears on how strong the
        # earlier comparison was, and it has to be on the record either way.
        self.assertIn("context_comparison", _json(READ))


if __name__ == "__main__":
    unittest.main()
