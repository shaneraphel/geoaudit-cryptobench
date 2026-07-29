"""The pLM-NN alignment has to match CryptoBench's own indexing, not ours.

Their baseline indexes one embedding row per observed residue, zero-based, in
coordinate order. Ours indexes the sorted set of integer resseq. Those two
disagree on the chains that number residues out of order and on the one that
carries insertion codes, and a mismatch would produce a baseline whose scores are
permuted relative to its labels -- which no metric would reveal, because a
permuted baseline still looks like a plausible weak detector.

The deposit ships one worked example, ``7w19A``, with an embedding of known length
and an annotation naming the amino acid at each binding index. That is enough to
pin the convention exactly, and this file does.
"""
from __future__ import annotations

import json
import unittest

from pocket_bench.paths import ROOT

import plmnn_sequences as PS

ART = ROOT / "results/baselines/PLMNN_SEQUENCES.json"
PER_STRUCTURE = ROOT / "results/official_fold/PER_STRUCTURE.json"

# From data/annotation.txt in the deposit's scripts folder: the binding residues
# of 7w19A as {chain}_{one-letter}{zero-based index}. Copied here because it is
# the only external check on the indexing convention available, and the file it
# came from is not part of the dataset download this repository verifies.
DEPOSIT_7W19A = (
    "A_N22 A_M24 A_G25 A_I26 A_D27 A_F28 A_V30 A_F32 A_L42 A_R44 A_Y86 A_I88 "
    "A_L89 A_D91 A_N92 A_P93 A_N96 A_D192 A_G196 A_H197 A_L199 A_I210 A_D211")
DEPOSIT_7W19A_N_ROWS = 293


def _load():
    return json.loads(ART.read_text())


def _row(uid):
    return next(r for r in _load()["rows"] if r["unit_id"] == uid)


class TheGatePasses(unittest.TestCase):
    def test_check(self):
        self.assertEqual(PS.check(), 0)


class TheConventionMatchesTheDeposit(unittest.TestCase):
    def test_the_example_chain_has_the_length_the_deposit_embedded(self):
        # The deposit's 7w19A.npy is (293, 2560). A different residue count here
        # would mean our sequence is not the sequence their model was given.
        self.assertEqual(_row("7w19_A")["n_residues"], DEPOSIT_7W19A_N_ROWS)

    def test_every_annotated_residue_lands_on_its_own_amino_acid(self):
        # This is the whole test. If the indexing were off by even one, most of
        # these twenty-three would name the wrong residue.
        seq = _row("7w19_A")["sequence"]
        for token in DEPOSIT_7W19A.split():
            _, rest = token.split("_")
            aa, idx = rest[0], int(rest[1:])
            self.assertEqual(seq[idx], aa,
                             f"index {idx} is {seq[idx]}, the deposit says {aa}")

    def test_the_annotation_is_not_trivially_satisfiable(self):
        # A sequence of one repeated letter would pass the test above for free.
        # Check the annotated positions actually discriminate: shifting the
        # sequence by one must break it.
        seq = _row("7w19_A")["sequence"]
        for shift in (-1, 1):
            wrong = sum(
                1 for t in DEPOSIT_7W19A.split()
                for aa, idx in [(t.split("_")[1][0], int(t.split("_")[1][1:]))]
                if 0 <= idx + shift < len(seq) and seq[idx + shift] != aa)
            self.assertGreater(wrong, 10,
                               f"a shift of {shift} would still match; the "
                               f"alignment test has no power")


class TheJoinToOurUniverseIsExact(unittest.TestCase):
    def test_every_row_maps_to_a_resseq(self):
        for r in _load()["rows"]:
            self.assertEqual(len(r["resseq_per_row"]), r["n_residues"])
            self.assertEqual(len(r["sequence"]), r["n_residues"])

    def test_the_resseq_map_reproduces_the_scored_universe(self):
        per = {f"{x['pdb']}_{x['chain']}": x
               for x in json.loads(PER_STRUCTURE.read_text())}
        for r in _load()["rows"]:
            self.assertEqual(len(set(r["resseq_per_row"])),
                             per[r["unit_id"]]["n_universe"], r["unit_id"])

    def test_the_awkward_chains_are_named_rather_than_hidden(self):
        d = _load()
        # These are the chains where a positional join would silently break.
        # They are asserted by name so that a future receptor refresh that
        # changes them fails here rather than in the baseline's numbers.
        self.assertEqual(d["units_with_insertion_codes"], ["2v6m_D"])
        self.assertEqual(sorted(d["units_numbered_out_of_order"]),
                         ["1k47_D", "1vsn_A"])
        self.assertEqual(d["units_where_rows_outnumber_the_universe"],
                         ["2v6m_D"])

    def test_only_the_insertion_code_chain_has_more_rows_than_resseq(self):
        for r in _load()["rows"]:
            more = r["n_residues"] > len(set(r["resseq_per_row"]))
            self.assertEqual(more, r["unit_id"] == "2v6m_D", r["unit_id"])

    def test_out_of_order_chains_really_are_out_of_order(self):
        # Guards the guard: if these two became monotone, a positional join
        # would start working by accident and the test above would stop having
        # anything to catch.
        for uid in ("1k47_D", "1vsn_A"):
            rs = _row(uid)["resseq_per_row"]
            self.assertNotEqual(rs, sorted(rs), uid)


class TheSequencesAreProteins(unittest.TestCase):
    def test_all_192_chains_are_present(self):
        d = _load()
        self.assertEqual(d["n_units"], 192)
        self.assertEqual(len({r["unit_id"] for r in d["rows"]}), 192)

    def test_no_residue_became_an_unknown(self):
        # Selenomethionine is the only modified residue in this fold and it maps
        # to methionine. Any X would mean a residue ESM cannot embed properly.
        self.assertEqual(_load()["n_non_standard_residues_written_as_X"], 0)

    def test_the_alphabet_is_the_standard_twenty(self):
        seen = set("".join(r["sequence"] for r in _load()["rows"]))
        self.assertTrue(seen <= set("ACDEFGHIKLMNPQRSTVWY"), sorted(seen))

    def test_chain_lengths_are_plausible(self):
        d = _load()
        self.assertEqual(d["n_residues_total"],
                         sum(r["n_residues"] for r in d["rows"]))
        self.assertGreaterEqual(d["shortest_chain"], 20)
        self.assertLessEqual(d["longest_chain"], 3000)


class ItIsNotAReadOfTheFold(unittest.TestCase):
    def test_it_claims_no_read_index(self):
        self.assertIsNone(_load()["test_fold_read_index"])

    def test_the_builder_opens_no_label_or_prediction(self):
        src = (ROOT / "tools/plmnn_sequences.py").read_text()
        for forbidden in ("cryptic_residues", "label_path", "TELEMETRY",
                          "residue_auc"):
            self.assertNotIn(forbidden, src)


if __name__ == "__main__":
    unittest.main()
