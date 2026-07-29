"""The external validation set: is it actually external, and is it actually a set?

Every read of CryptoBench's test fold in this repository is exploratory, and the
paper says so. This set is the only thing that can carry a confirmatory claim, and
it can only do that if three properties hold: nothing in it was available to
CryptoBench, nothing in it is a close relative of anything CryptoBench had, and its
labels were fixed before any method was pointed at it.

None of the three is self-evident from looking at the file, so each is tested here.
The last one is the easiest to lose by accident and the most expensive to lose, so
it is tested twice: once on the set's own declaration and once on the presence of
anything score-shaped inside it.
"""
from __future__ import annotations

import json

import pytest

from pocket_bench.paths import ROOT

SET = ROOT / "results/external/EXTERNAL_SET.json"
UNIREF = ROOT / "data/external/UNIREF50.json"
MANIFEST = ROOT / "data/external/external_manifest.json"


@pytest.fixture(scope="module")
def ext() -> dict:
    if not SET.is_file():
        pytest.skip("external set absent")
    return json.loads(SET.read_text())


@pytest.fixture(scope="module")
def uniref() -> dict:
    if not UNIREF.is_file():
        pytest.skip("uniref map absent")
    return json.loads(UNIREF.read_text())


def test_no_method_has_been_run_on_it(ext):
    assert ext["no_method_has_been_run"] is True
    assert ext["reads_test_fold"] is False


def test_nothing_score_shaped_is_stored_in_the_set(ext):
    """A label file may not carry a prediction, however convenient that would be.

    If a score ever appears in this artifact, the set stops being something that
    can be read once under a plan and becomes something that has already been read.
    """
    banned = ("roc_auc", "score", "prediction", "auc", "f1", "mcc", "delta")
    for unit in ext["units"]:
        for key in unit:
            assert not any(b in key.lower() for b in banned), key


def test_every_apo_structure_postdates_cryptobench(ext):
    cutoff = ext["selection"]["cutoff"]
    for u in ext["units"]:
        assert u["released"] > cutoff, (u["apo_pdb"], u["released"])


def test_no_unit_shares_an_accession_or_a_cluster_with_cryptobench(ext, uniref):
    """Sequence externality, not just temporal.

    A structure deposited after the cutoff can still be a 99%-identical relative of
    something in CryptoBench's training fold, and a model that learned the fold
    from the relative is not being tested on anything new.
    """
    acc = set(uniref["cryptobench"])
    clusters = set(uniref["cryptobench"].values())
    for u in ext["units"]:
        assert u["uniprot"] not in acc, u["uniprot"]
        assert u["cluster"] not in clusters, u["cluster"]


def test_one_unit_per_cluster_so_the_units_are_independent(ext):
    """The bootstrap resamples units; if two units were relatives it would lie."""
    clusters = [u["cluster"] for u in ext["units"]]
    assert len(set(clusters)) == len(clusters)


def test_the_set_is_large_enough_to_be_worth_reading(ext):
    assert ext["n_units_with_a_cryptic_pocket"] >= 30
    assert sum(len(u["residues"]) for u in ext["units"]) >= 200


def test_every_unit_has_at_least_one_cryptic_pair_behind_its_labels(ext):
    for u in ext["units"]:
        assert u["residues"]
        assert any(p["verdict"] == "cryptic" for p in u["pairs"]), u["apo_pdb"]


def test_the_guard_band_leaves_borderline_pairs_unlabelled(ext):
    """The band is the price of a pRMSD we can only reproduce to a tolerance.

    It has to actually be paid: if no pair ever lands inside it, the band is
    decorative and the reproduction tail is still in the labels.
    """
    v = ext["pair_verdicts"]
    inside = [k for k in v if "guard band" in k]
    assert inside and v[inside[0]] > 0


def test_reading_mmcif_agrees_with_reading_pdb(ext):
    """A quarter of these structures exist only as mmCIF, so the second reader
    has to be shown equivalent rather than assumed so."""
    e = ext["reading_mmcif_gives_what_reading_pdb_gives"]
    assert e["n_compared"] >= 10
    assert e["n_identical"] == e["n_compared"]
    assert e["worst_coordinate_difference"] <= e["tolerance"]


def test_rendering_mmcif_as_pdb_loses_no_atom(ext):
    rt = ext["receptors"]["round_trip"]
    assert rt["n_identical"] == rt["n_compared"]


def test_the_receptors_come_from_the_repositorys_own_writer(ext):
    """If the external inputs were prepared differently from the CryptoBench ones,
    a difference in score could be the preparation rather than the protein."""
    assert (ext["receptors"]["writer"]
            == "pocket_bench.pdb_io.write_receptor_only_pdb")
    assert ext["receptors"]["n"] == len(ext["units"])


def test_the_manifest_matches_the_units(ext):
    if not MANIFEST.is_file():
        pytest.skip("manifest absent")
    m = json.loads(MANIFEST.read_text())
    assert m["n_entries"] == len(ext["units"])
    assert m["fold"] == "external"
    assert m["clustering"]["sequence_identity_threshold"] == 0.50
    named = {(e["pdb"], e["chain"]) for e in m["entries"]}
    assert named == {(u["apo_pdb"], u["apo_chain"]) for u in ext["units"]}


def test_collapsing_insertion_codes_did_not_merge_two_residues(ext):
    """The harness indexes residues by integer number, so a labelled 60 and 60A
    would collide. It has bitten this repository once already."""
    if not MANIFEST.is_file():
        pytest.skip("manifest absent")
    m = json.loads(MANIFEST.read_text())
    assert m["n_units_with_a_collapse_collision"] == 0, m["collisions"]


def test_the_ways_the_set_is_easier_are_written_down(ext):
    """Stated before any result exists, so they cannot be produced afterwards to
    explain one."""
    c = ext["choices_that_shrink_the_set"]
    assert {"ligand_chemistry", "one_unit_per_cluster", "guard_band"} <= set(c)
    assert "easier" in c["guard_band"]
