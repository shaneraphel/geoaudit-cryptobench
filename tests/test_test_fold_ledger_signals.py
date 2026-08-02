"""The test-fold ledger must be able to see an access it has never seen before.

AGENTS.md §2: a gate that has never failed is indistinguishable from one that
cannot fail. This ledger is worse than that shape, because it did not fail --- it
returned a number, and the number was wrong, and it had been wrong for the whole
seam programme.

What went wrong. The ledger had three signals for "this artifact took a number
off the held-out fold", and all three keyed on how the artifact *spells* its
metric: a per-unit table with a key containing ``auc``, a declared
``test_fold_read_index``, or the literal substring ``auc`` somewhere in the
document beside a unit count of 192. The seam probes store each method's
per-unit ROC-AUC under the method's own name --- ``per_unit[i]["seam_msa_field"]``
--- and never write those three letters. Eighteen artifacts were invisible,
including ``GEO_SEAM_EQUALZ_FUSION_VS_PLMNN.json``, whose architecture is the
best this project has. The ledger reported 13 architectures over 27 artifacts;
the true figures were 20 over 44, and the manuscript printed the 13.

Every one of the eighteen carried ``reads_test_fold: true``, which
``.cursor/rules/00-evidence-discipline.mdc`` requires of any artifact touching a
held-out fold. The ledger was not reading the field the rule exists to produce.
That is the fourth signal these tests hold in place.

The violations planted, each in a temporary tree, nothing on disk touched:

  1. an artifact that declares ``reads_test_fold`` and spells its only metric
     by the method's name --- the exact shape of the eighteen. The ledger must
     count it and must name its architecture.
  2. the same artifact with the fourth signal disabled, which must make it
     vanish. Without this the first test would still pass if some other signal
     happened to catch the artifact, and the new code would be dead.
  3. an artifact evaluating several architectures in one pass, which must
     contribute all of them and not one.
  4. an artifact that names no architecture, which must be counted as an access
     and must *not* invent one. The old default called it "table field
     variant", asserting something the artifact never said.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_test_fold_ledger as ledger  # noqa: E402

N = ledger.N_OFFICIAL_UNITS


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """An empty results tree with a minimal frozen telemetry beside it."""
    results = tmp_path / "results"
    (results / "cryptobench_official").mkdir(parents=True)
    (results / "official_fold").mkdir(parents=True)
    telemetry = results / "cryptobench_official/TELEMETRY.json"
    telemetry.write_text(json.dumps({"rows": [
        {"unit_id": f"u{i}", "method": m, "residue_auc": 0.5}
        for i in range(N) for m in ("table_field", "p2rank")]}))
    monkeypatch.setattr(ledger, "RESULTS", results)
    monkeypatch.setattr(ledger, "TELEMETRY", telemetry)
    monkeypatch.setattr(ledger, "ROOT", tmp_path)
    return results


def _write(results: Path, name: str, doc: dict) -> None:
    (results / "official_fold" / name).write_text(json.dumps(doc))


SEAM_SHAPED = {
    "schema": "geoaudit.seam_probe.v1",
    "clinical_grade": False,
    "reads_test_fold": True,
    "method": "seam_msa_field",
    "n_units_compared": N,
    # The metric, spelled by the method's name. No "auc" anywhere in the
    # document, which is what made eighteen real artifacts invisible.
    "mean_seam_msa_field": 0.8262,
    "per_unit": [{"unit": f"u{i}", "seam_msa_field": 0.8, "plmnn": 0.79}
                 for i in range(N)],
}


def test_a_metric_spelled_by_method_name_is_still_an_access(tree):
    """Signal four: the artifact says it read the fold, so it read the fold."""
    _write(tree, "SEAM_MSA_VS_PLMNN_PROBE.json", SEAM_SHAPED)
    led = ledger.build()

    assert led["n_standalone_probes"] == 1, (
        "an artifact declaring reads_test_fold must be counted whatever it "
        "calls its metric")
    row = led["standalone_probe_artifacts"][0]
    assert row["architectures"] == ["seam_msa_field"]
    assert "seam_msa_field" in led["distinct_architectures_evaluated"]
    assert "fourth signal" in row["kind"], (
        "the row must record which signal found it, so that a reader can tell "
        "this was not caught by the original three")


def test_without_the_fourth_signal_it_vanishes(tree, monkeypatch):
    """The new code must be load-bearing, not merely present.

    If some other signal already caught this shape, test one above would pass
    with the fourth signal deleted and the fix would be decoration.
    """
    _write(tree, "SEAM_MSA_VS_PLMNN_PROBE.json", SEAM_SHAPED)
    monkeypatch.setattr(ledger, "_declares_a_read", lambda path, d: False)
    led = ledger.build()

    assert led["n_standalone_probes"] == 0, (
        "with signal four disabled this artifact must be invisible; if it is "
        "not, the eighteen were missed for some other reason and this fix "
        "does not address it")


def test_one_artifact_can_evaluate_many_architectures(tree):
    """GRAND_BASELINE_READ.json scores ten methods in one pass.

    Reading only a top-level ``method`` field would record that as one
    architecture, or --- since it has no such field --- as none.
    """
    _write(tree, "GRAND_BASELINE_READ.json", {
        "schema": "geoaudit.grand_baseline_read.v1",
        "reads_test_fold": True,
        "n_units_in_manifest": N,
        "summary": {m: {"mean_per_unit_roc_auc": 0.8} for m in (
            "geometry_field", "seam_geometry_field", "geo_seam_equalz_r14",
            "p2rank", "plmnn", "pocketminer")},
    })
    led = ledger.build()

    got = set(led["distinct_architectures_evaluated"])
    assert {"geometry_field", "seam_geometry_field",
            "geo_seam_equalz_r14"} <= got
    assert not ({"p2rank", "plmnn", "pocketminer"} & got), (
        "a rival we ran as a baseline is not one of our architectures; "
        "counting it would inflate the number this ledger exists to keep "
        "honest")


def test_an_unnamed_architecture_is_not_invented(tree):
    """Counted as an access, reported as unnamed, never guessed at."""
    _write(tree, "POLARITY_SWITCH_SCREEN.json", {
        "schema": "geoaudit.screen.v1",
        "reads_test_fold": True,
        "n_units": N,
        "mean_delta": 0.01,
    })
    led = ledger.build()

    assert led["n_standalone_probes"] == 1
    row = led["standalone_probe_artifacts"][0]
    assert row["architectures"] == []
    assert row["architecture_is_named"] is False
    assert row["method"] is None, (
        "the old default was the string 'table field variant', which turned "
        "'this artifact does not say' into a claim about which architecture ran")
    assert led["n_artifacts_with_unnamed_architecture"] == 1
    assert "lower bound" in led["honest_summary"], (
        "with unnamed artifacts present the architecture total is a lower "
        "bound and the summary sentence has to say so")


def test_the_real_tree_counts_every_declared_read():
    """No artifact on disk declares a read the ledger does not list.

    The regression that matters. The eighteen were found by comparing the
    ledger against ``reads_test_fold``; this keeps that comparison running, so
    that a nineteenth cannot appear the same way.
    """
    out = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
    if not out.is_file():
        pytest.skip("ledger not built")
    listed = {a["artifact"]
              for a in json.loads(out.read_text())["standalone_probe_artifacts"]}
    declared = set()
    for p in sorted((ROOT / "results").rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if (isinstance(d, dict) and d.get("reads_test_fold") is True
                and p.name not in ledger._NOT_AN_ACCESS):
            declared.add(str(p.relative_to(ROOT)))
    missing = sorted(declared - listed)
    assert not missing, (
        "these artifacts say they read the held-out fold and the ledger does "
        "not list them: " + ", ".join(missing))
