"""The labelling rule, recovered from the benchmark rather than assumed.

An external validation set is only external if its labels mean the same thing as
the benchmark's. Nothing in CryptoBench's release states the rule in executable
form, so it was recovered by fitting candidate rules against 510 of its own
training pair records and keeping the one that reproduces them. These tests guard
that recovery: that it reproduces the deposit, that the alternatives were actually
run and lost rather than dismissed, and that the places where it still disagrees
are places the external build refuses to label at all.

The recovery reads the training fold only. That is why it costs no indexed read.
"""
from __future__ import annotations

import json

import pytest

from pocket_bench.paths import ROOT

RULE = ROOT / "results/external/CRYPTOBENCH_RULE.json"


@pytest.fixture(scope="module")
def rule() -> dict:
    if not RULE.is_file():
        pytest.skip("rule recovery absent")
    return json.loads(RULE.read_text())


def test_the_recovery_never_touches_the_test_fold(rule):
    """A rule fitted on the test fold would leak it into the external labels."""
    assert rule["reads_test_fold"] is False
    assert rule["test_fold_read_index"] is None
    assert rule["why_no_read_is_spent"]


def test_the_contact_rule_misses_nothing_the_deposit_labels(rule):
    """Missing labelled residues would make the external set easier than it is.

    Added residues are tolerated where the apo chain does not have them, because
    the deposit's two selections are the same length and so are already restricted
    to residues that mapped. Missed residues have no such excuse.
    """
    c = rule["contact_rule"]
    assert c["residues_we_missed"] == 0
    assert c["compared"] >= 500
    assert c["exact"] / c["compared"] >= 0.98


def test_every_residue_we_add_is_one_the_apo_chain_lacks(rule):
    c = rule["contact_rule"]
    assert c["added_absent_from_apo"] >= c["residues_we_added"] - 1


def test_the_rejected_contact_variants_were_run_and_lost(rule):
    """Ignoring hydrogens and pooling ligand copies are the two obvious readings.

    Both are plausible enough that a reader would assume one of them, so both are
    executed against the deposit and reported with their scores rather than being
    argued away.
    """
    c, r = rule["contact_rule"], rule["rejected_variants"]
    assert set(r) >= {"ignoring_deposited_hydrogens",
                      "pooling_every_copy_of_the_ligand"}
    for name, arm in r.items():
        if isinstance(arm, dict):
            assert arm["exact"] < c["exact"], name


def test_the_pocket_frame_beats_the_chain_frame_for_the_prmsd(rule):
    """Superposing on the chain is the reading a reader would guess first."""
    p = rule["prmsd"]
    assert p["chain_frame_within_tolerance"] < p["within_tolerance"]
    assert p["chain_frame_within_tolerance"] == 0


def test_the_prmsd_reproduces_the_deposited_value(rule):
    p = rule["prmsd"]
    assert p["compared"] >= 500
    assert p["correlation"] >= 0.99
    assert p["within_tolerance"] / p["compared"] >= 0.93


def test_the_guard_band_covers_every_pair_the_residual_would_reclassify(rule):
    """This is what makes the residual harmless instead of unresolved.

    The pRMSD reproduction has a tail. A tail matters only if it moves a pair
    across the cryptic threshold, so the external build refuses a band around
    that threshold, and the band is only sufficient if every disagreement on the
    training fold falls inside it.
    """
    p = rule["prmsd"]
    assert p["flips_inside_the_guard_band"] == p["would_change_inclusion"]
    band = rule["external_build_constraints"]["prmsd_guard_band_angstrom"]
    for flip in p["changed"]:
        assert abs(flip["theirs"] - rule["recovered_rule"]["prmsd_floor"]) <= band


def test_the_guard_band_states_what_it_costs(rule):
    """Refusing borderline pairs removes the hardest ones, and that is a cost."""
    b = rule["external_build_constraints"]
    assert b["prmsd_guard_band_angstrom"] > 0
    assert "easier" in b["cost"]


def test_the_label_is_the_union_over_every_holo_partner(rule):
    """The alternative -- the main partner alone -- is wrong on half the units."""
    a = rule["aggregation"]
    assert a["shipped_equals_union_over_holos"] == a["units_compared"]
    assert a["shipped_equals_main_holo_only"] < a["shipped_equals_union_over_holos"]


def test_modified_residues_count_as_protein(rule):
    """Selenomethionine and friends are HETATM records the benchmark still labels.

    Reading a chain as its ATOM records alone drops them, which is what three of
    the recovery's last four disagreements were.
    """
    assert rule["recovered_rule"]["polymer_includes_seqres_hetatm"] is True
