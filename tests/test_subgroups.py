"""The subgroup analysis must stay descriptive and stay honest about its bands.

Three properties matter and none is enforced by the arithmetic. The covariates
must be computable without opening a score, or the groups were chosen with the
answers visible. The plan must keep calling itself exploratory, because a
subgroup analysis after an unresolved primary endpoint is exploratory whatever it
finds. And any band that clears a corrected threshold has to carry its own trend
test, so the number cannot travel into prose without the thing that qualifies it.
"""
from __future__ import annotations

import json
import unittest

import numpy as np

from pocket_bench.paths import ROOT

import subgroup_covariates as SC
import preregister_subgroups as SP
import subgroup_read as SR

COV = ROOT / "results/official_fold/SUBGROUP_COVARIATES.json"
PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_SUBGROUPS.json"
READ = ROOT / "results/official_fold/SUBGROUP_READ.json"
LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
PREREG = ROOT / "results/official_fold/PREREGISTERED_READ.json"


def _load(p):
    return json.loads(p.read_text())


class TheGatesPass(unittest.TestCase):
    def test_covariates(self):
        self.assertEqual(SC.check(), 0)

    def test_plan(self):
        self.assertEqual(SP._check(), 0)

    def test_read(self):
        self.assertEqual(SR.check(), 0)


class CovariatesTouchNoScore(unittest.TestCase):
    def test_the_builder_reads_no_prediction_artifact(self):
        # The covariates are only preregisterable because they cannot see the
        # answers. Any of these paths appearing in the module would mean they
        # could.
        src = (ROOT / "tools/subgroup_covariates.py").read_text()
        for forbidden in ("TELEMETRY", "OFFICIAL_MULTI_METHOD",
                          "residue_auc", "p2rank", "table_field"):
            self.assertNotIn(forbidden, src)

    def test_it_claims_no_read_index(self):
        self.assertIsNone(_load(COV)["test_fold_read_index"])

    def test_every_unit_has_every_covariate(self):
        d = _load(COV)
        self.assertEqual(d["n_units"], 192)
        for r in d["rows"]:
            for c in d["covariates"]:
                self.assertIsNotNone(r[c], f"{r['unit_id']} has no {c}")

    def test_positive_rate_follows_from_the_other_two(self):
        for r in _load(COV)["rows"]:
            self.assertAlmostEqual(r["positive_rate"],
                                   r["n_true"] / r["chain_length"], places=5)

    def test_tertiles_split_into_three_nonempty_groups(self):
        d = _load(COV)
        for c in d["covariates"]:
            s = d["distributions"][c]
            self.assertEqual(sum(s["group_sizes"]), s["n_defined"])
            for n in s["group_sizes"]:
                self.assertGreater(n, 0)
            # Order-statistic tertiles should be close to even; a covariate
            # with enough ties to break that would silently change the power of
            # its three tests.
            self.assertLess(max(s["group_sizes"]) - min(s["group_sizes"]),
                            0.15 * s["n_defined"])

    def test_prmsd_is_the_deposits_own_number(self):
        osf = json.loads(
            (ROOT / "data/cryptobench_apo/_osf/test.json").read_text())
        for r in _load(COV)["rows"][:20]:
            pdb, chain = r["unit_id"].split("_")
            want = max(p["pRMSD"] for p in osf[pdb]
                       if p.get("apo_chain") == chain)
            self.assertAlmostEqual(r["prmsd"], want, places=4)


class ThePlanFixesWhatCannotBeFixedLater(unittest.TestCase):
    def test_it_is_declared_exploratory(self):
        self.assertEqual(_load(PLAN)["status_declared_in_advance"],
                         "exploratory")

    def test_it_forbids_a_subgroup_claim(self):
        self.assertTrue(_load(PLAN)["decision_rules"][
            "no_subgroup_may_become_a_claim"])

    def test_it_pins_the_covariate_artifact_by_hash(self):
        import hashlib
        self.assertEqual(_load(PLAN)["covariate_artifact"]["sha256"],
                         hashlib.sha256(COV.read_bytes()).hexdigest())

    def test_bonferroni_matches_the_number_of_bands(self):
        p = _load(PLAN)
        n = len(p["covariates"]) * len(p["bands"])
        self.assertEqual(p["multiplicity"]["n_subgroup_tests"], n)
        self.assertAlmostEqual(p["multiplicity"]["corrected_level"],
                               0.05 / n, places=6)

    def test_it_names_a_sentence_for_an_unfavourable_outcome(self):
        s = _load(PLAN)["what_will_be_written_under_each_outcome"]
        self.assertIn("nothing_survives_correction", s)
        self.assertIn("a_band_or_trend_favours_p2rank", s)

    def test_the_plan_does_not_read_the_fold(self):
        self.assertFalse(_load(PLAN)["reads_test_fold"])

    def test_chain_length_is_carried_as_a_negative_control(self):
        # It was already stratified on at read five, where it bought nothing.
        # Dropping it after the fact would be selecting covariates on results.
        self.assertIn("chain_length",
                      [c["id"] for c in _load(PLAN)["covariates"]])


class TheReadReproducesWhatIsAlreadyPublished(unittest.TestCase):
    def test_it_reproduces_read_fives_mean_and_counts(self):
        c = _load(READ)["calibration_against_read_five"]
        self.assertTrue(c["reproduces"])
        self.assertAlmostEqual(c["mean"], c["mean_published_at_read_five"],
                               places=6)
        self.assertEqual(c["n_field_ahead"], c["n_field_ahead_published"])
        self.assertEqual(c["n_baseline_ahead"], c["n_baseline_ahead_published"])

    def test_the_distribution_accounts_for_every_chain(self):
        p = _load(READ)["per_chain_distribution"]
        self.assertEqual(p["n_field_ahead"] + p["n_baseline_ahead"]
                         + p["n_tied"], p["n"])
        self.assertEqual(p["n"], 192)

    def test_quantiles_are_ordered(self):
        q = _load(READ)["per_chain_distribution"]["quantiles"]
        ks = sorted(q, key=float)
        vals = [q[k] for k in ks]
        self.assertEqual(vals, sorted(vals))

    def test_bands_partition_the_units_for_every_covariate(self):
        d = _load(READ)
        for name, v in d["by_covariate"].items():
            self.assertEqual(sum(b["n"] for b in v["bands"]),
                             d["n_paired_units"], name)

    def test_band_means_reconstruct_the_overall_mean(self):
        d = _load(READ)
        want = d["per_chain_distribution"]["mean"]
        for name, v in d["by_covariate"].items():
            got = (sum(b["mean"] * b["n"] for b in v["bands"])
                   / sum(b["n"] for b in v["bands"]))
            self.assertAlmostEqual(got, want, places=6, msg=name)

    def test_band_sizes_match_the_preregistered_ones(self):
        plan = {c["id"]: c["group_sizes"] for c in _load(PLAN)["covariates"]}
        for name, v in _load(READ)["by_covariate"].items():
            self.assertEqual([b["n"] for b in v["bands"]], plan[name], name)

    def test_the_corrected_interval_is_never_narrower_than_the_uncorrected(self):
        for name, v in _load(READ)["by_covariate"].items():
            for b in v["bands"]:
                w95 = b["ci95"][1] - b["ci95"][0]
                wbf = b["ci_bonferroni"][1] - b["ci_bonferroni"][0]
                self.assertGreaterEqual(wbf, w95 - 1e-9,
                                        f"{name} {b['band']}")

    def test_a_corrected_exclusion_implies_an_uncorrected_one(self):
        for name, v in _load(READ)["by_covariate"].items():
            for b in v["bands"]:
                if b["excludes_zero_corrected"]:
                    self.assertTrue(b["excludes_zero_uncorrected"],
                                    f"{name} {b['band']}")


class TheReadStaysExploratory(unittest.TestCase):
    def test_it_reports_itself_exploratory(self):
        self.assertEqual(_load(READ)["status"], "exploratory")

    def test_the_outcome_is_a_preregistered_sentence(self):
        d = _load(READ)
        self.assertEqual(
            d["outcome"],
            _load(PLAN)["what_will_be_written_under_each_outcome"][
                d["outcome_key"]])

    def test_every_surviving_band_carries_its_trend_test(self):
        d = _load(READ)
        for b in d["bands_excluding_zero_after_correction"]:
            for k in ("supported_by_a_trend", "covariate_is_monotone",
                      "covariate_spearman_rho", "covariate_spearman_p"):
                self.assertIn(k, b)

    def test_a_band_is_only_trend_supported_if_the_trend_survived(self):
        d = _load(READ)
        trends = set(d["trends_surviving_correction"])
        for b in d["bands_excluding_zero_after_correction"]:
            self.assertEqual(b["supported_by_a_trend"],
                             b["covariate_is_monotone"]
                             and b["covariate"] in trends)

    def test_it_records_what_may_not_be_concluded(self):
        self.assertTrue(_load(READ)["what_may_not_be_concluded"])

    def test_it_is_indexed_as_read_eight(self):
        d = _load(READ)
        self.assertEqual(d["test_fold_read_index"], 8)
        self.assertFalse(d["rescored_anything"])
        seq = _load(LEDGER)["indexed_read_sequence"]
        self.assertIn("results/official_fold/SUBGROUP_READ.json",
                      [r["artifact"] for r in seq])


class TheSpearmanIsTheRealOne(unittest.TestCase):
    def test_it_matches_scipy_on_the_actual_covariates(self):
        from scipy import stats
        cov = {r["unit_id"]: r for r in _load(COV)["rows"]}
        d, units = SR._paired()
        for name in _load(COV)["covariates"]:
            x = np.array([cov[u][name] for u in units], dtype=np.float64)
            self.assertAlmostEqual(SR._spearman(x, d),
                                   float(stats.spearmanr(x, d).statistic),
                                   places=9, msg=name)

    def test_it_averages_tied_ranks(self):
        # Pocket size has heavy ties; ordering them by array position instead
        # of averaging would give a different rho for the same data.
        x = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
        y = np.array([5.0, 1.0, 3.0, 4.0, 2.0])
        from scipy import stats
        self.assertAlmostEqual(SR._spearman(x, y),
                               float(stats.spearmanr(x, y).statistic),
                               places=9)


class TheGateRejectsATamperedRead(unittest.TestCase):
    def _with(self, mutate):
        d = _load(READ)
        mutate(d)
        original = READ.read_text()
        try:
            READ.write_text(json.dumps(d, indent=2) + "\n")
            return SR.check()
        finally:
            READ.write_text(original)

    def test_promoting_it_out_of_exploratory_fails(self):
        def m(d):
            d["status"] = "confirmatory"
        self.assertEqual(self._with(m), 1)

    def test_a_band_losing_its_trend_context_fails(self):
        def m(d):
            for b in d["bands_excluding_zero_after_correction"]:
                b.pop("supported_by_a_trend", None)
        self.assertEqual(self._with(m), 1)

    def test_claiming_trend_support_without_a_trend_fails(self):
        def m(d):
            for b in d["bands_excluding_zero_after_correction"]:
                b["supported_by_a_trend"] = True
        self.assertEqual(self._with(m), 1)

    def test_a_band_dropped_from_a_partition_fails(self):
        def m(d):
            d["by_covariate"]["n_true"]["bands"][0]["n"] -= 1
        self.assertEqual(self._with(m), 1)

    def test_an_outcome_sentence_that_was_not_preregistered_fails(self):
        def m(d):
            d["outcome"] = "the field is better on medium-sized pockets"
        self.assertEqual(self._with(m), 1)

    def test_a_broken_calibration_fails(self):
        def m(d):
            d["calibration_against_read_five"]["reproduces"] = False
        self.assertEqual(self._with(m), 1)

    def test_ties_that_do_not_add_up_fail(self):
        def m(d):
            d["per_chain_distribution"]["n_tied"] = 5
        self.assertEqual(self._with(m), 1)

    def test_the_gate_restored_the_file(self):
        self.assertEqual(SR.check(), 0)


if __name__ == "__main__":
    unittest.main()
