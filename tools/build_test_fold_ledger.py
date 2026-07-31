#!/usr/bin/env python3
"""Count every evaluation this project has run against the official test fold.

Why this is generated rather than written. The manuscript disclosed that the
test fold was read three times, and that number was typed by hand from memory of
the three probes that produced the table field. It was wrong. A fourth
evaluation, ``THRESHOLD_FIELD_TEST.json``, sits in the same directory carrying
192 per-unit metrics and a paired test against P2Rank; its own header says "the
first and only evaluation of this architecture on the official test fold", which
is true of that architecture and says nothing about the programme. On top of
those, every detector in the frozen telemetry was at some point a new
architecture scored on the same 192 units.

A hand-maintained count of how often you looked at the held-out data is the
least trustworthy sentence in any paper, because the incentive runs one way and
the author is the only witness. So it is counted from the artifacts: anything
carrying per-unit metrics over the official units is an access, and the ledger
lists it whether or not it flattered the method.

The distinction the ledger keeps, because it is the one that matters for
selection bias:

  a **fold read** scores a NEW architecture, chosen on the training fold, and
  is an opportunity to learn something about the test set;

  a **re-score** re-runs an architecture already frozen -- to attach an
  environment digest, to check reproducibility, to fill in a metric -- and
  cannot leak anything that was not already leaked, provided the numbers do not
  move. The ledger records whether they moved.

Usage:
  PYTHONPATH=src python3.12 tools/build_test_fold_ledger.py
  PYTHONPATH=src python3.12 tools/build_test_fold_ledger.py --check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TELEMETRY = RESULTS / "cryptobench_official/TELEMETRY.json"
OUT = RESULTS / "official_fold/TEST_FOLD_ACCESS_LEDGER.json"

# Detectors that are not ours: a published tool run as a baseline is not a look
# at the test set on our behalf, it is the comparison the test set exists for.
EXTERNAL = {"p2rank", "random_bbox"}

# The official fold's size. An artifact that reports a metric and names this
# count has taken a number off it.
N_OFFICIAL_UNITS = 192
_UNIT_KEYS = ("n_test_units", "n_units", "n_paired_units",
              "n_units_scored", "n_units_in_manifest", "n_units_compared")
# Named exemptions from the third signal, each with the reason. A rule broad
# enough to excuse these by shape would also excuse a real access.
_NOT_AN_ACCESS = {
    "PLMNN_SCORES.json": "a rival's per-residue predictions; no label is opened "
                         "and no metric is computed, which the artifact argues "
                         "in why_this_is_not_an_indexed_read",
    "ENDPOINT_STATUS.json": "the endpoint declaration, which summarises reads "
                            "this ledger already counts",
    "TEST_FOLD_ACCESS_LEDGER.json": "this file. It quotes the fold's size and "
                                    "the metrics of the reads it lists, so the "
                                    "third signal matches it and it would "
                                    "otherwise enter its own inventory",
}


def _reports_a_fold_metric(path: Path, d: dict) -> bool:
    if path.name in _NOT_AN_ACCESS:
        return False
    if not any(d.get(k) == N_OFFICIAL_UNITS for k in _UNIT_KEYS):
        return False
    return "auc" in json.dumps(d).lower()


def _per_unit_artifacts() -> list[dict]:
    """Artifacts under results/ that are an access to the official fold.

    Three things qualify. One is carrying per-unit metrics over the official
    units, which is what scoring the fold produces. The other is declaring a
    read index, which catches an artifact that draws a fresh inference from
    per-unit numbers an earlier read already froze. The second kind re-scores
    nothing, and leaving it out would let the fold be used again for free every
    time a new summary of the same numbers is wanted.

    The third was added after an audit found seven artifacts that satisfied
    neither. ``FULL_EXPANSION.json`` is the clearest: ``run_full_expansion.py``
    loads ``_cascade_cache_test.npz``, takes its labels, and reports twelve
    ROC-AUCs over ``n_test_units: 192``. It stores no per-unit table and claims
    no index, so both original signals missed it, and the fold had been read.
    An artifact that reports a metric while naming the fold's unit count has
    taken a number off the held-out set whatever shape it stored it in, so that
    is now the third signal. It is deliberately blunt: it will also catch an
    artifact that merely *quotes* a fold number, and being listed with
    ``kind: reports a fold metric`` and no index is the correct outcome for
    that too -- the ledger's job is to make every look at the fold visible, not
    to grade them.

    Two artifacts match the third signal and are not accesses, and they are
    named rather than filtered by a rule that would also excuse a real one:
    ``PLMNN_SCORES.json`` is a rival's predictions with no label opened, which
    the file itself argues, and ``ENDPOINT_STATUS.json`` is the endpoint
    declaration that summarises reads already counted here.
    """
    found = []
    for p in sorted(RESULTS.rglob("*.json")):
        if p.resolve() == TELEMETRY.resolve():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict):
            continue
        rows = None
        for key in ("per_structure", "per_unit"):
            v = d.get(key)
            # The identifier and the metric are each spelled more than one way.
            # DUAL_TRACK_AB.json carries 192 per-unit ROC-AUCs on the official
            # fold under the keys "unit" and "A_resolved", and matched neither
            # half of the original rule; it is also the artifact whose "unit" is
            # null on every row, so it scored the fold without recording which
            # units it scored.
            if (isinstance(v, list) and len(v) >= 150
                    and isinstance(v[0], dict)
                    and any(k in v[0] for k in ("unit_id", "unit", "pdb"))
                    and any(("auc" in k or "resolved" in k) for k in v[0])):
                rows = v
                break
        declares = d.get("test_fold_read_index") is not None
        reports = _reports_a_fold_metric(p, d)
        if rows is None and not declares and not reports:
            continue
        found.append({
            "artifact": str(p.relative_to(ROOT)),
            # A re-summarising read reports its own unit count, and the two
            # spellings both occur. Missing one left the ledger printing a
            # blank where the size of the read belonged.
            "n_units": (len(rows) if rows is not None
                        else d.get("n_paired_units") or d.get("n_units")),
            "method": d.get("method") or "table field variant",
            "read_index": d.get("test_fold_read_index"),
            "kind": "scored the fold" if rows is not None
            else ("new inference over per-unit numbers an earlier read froze"
                  if declares else
                  "reports a fold metric and declares no index; found by the "
                  "audit that added the third signal"),
            "mean_residue_auc": d.get("residue_auc_mean")
            or (d.get("means") or {}).get("residue_auc")
            or d.get("mean_method"),
            "selection_provenance": (
                d.get("selection_provenance") or d.get("selected_on")
                or (d.get("provenance_of_the_choice") or {}).get("committed_in")),
        })
    return found


def build() -> dict:
    tel = json.loads(TELEMETRY.read_text())
    rows = tel["rows"] if isinstance(tel, dict) and "rows" in tel else tel
    methods = sorted({r["method"] for r in rows})
    ours = [m for m in methods if m not in EXTERNAL]

    probes = _per_unit_artifacts()
    # A frozen detector and a probe of the same architecture are one access, not
    # two: the probe is what produced the number the frozen run reproduces.
    probe_methods = {p["method"] for p in probes}

    # Reads are indexed across the whole counterattack programme, not per
    # lineage, so the fourth read is a counting-field probe while the first
    # three were table-field ones. Counting them by lineage keeps the summary
    # sentence true as further lineages are read.
    indexed = sorted((p for p in probes if p.get("read_index") is not None),
                     key=lambda p: p["read_index"])
    # The lineage is written "table field" in some artifacts and "table_field"
    # in others, and matching only the spaced form silently undercounted the
    # main architecture's readings.
    n_table = sum(1 for p in indexed
                  if "table field" in p["method"].replace("_", " "))
    n_scored = sum(1 for p in probes if p["kind"] == "scored the fold")
    n_resummary = len(probes) - n_scored

    return {
        "schema": "geoaudit.test_fold_access_ledger.v1",
        "clinical_grade": False,
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "n_units": len({r["unit_id"] for r in rows}),
        "question": "how many times has this project scored the held-out fold, "
                    "and with what",
        "counting_rule": (
            "Every artifact carrying per-unit metrics over the official units "
            "is an access, and so is every artifact that draws a new inference "
            "from per-unit numbers an earlier read froze, even though the "
            "second kind scores nothing again: a fresh summary of the same "
            "numbers is a fresh use of the fold and is indexed as one. "
            "Detectors in the frozen telemetry count once each. "
            "Baselines we did not design (p2rank, random_bbox) are not counted "
            "as looks at the test set on our behalf. Re-scoring a frozen "
            "detector is recorded separately and is not a fold read, provided "
            "its numbers do not move; tools/run_cryptobench_apo.py --merge has "
            "reproduced all of them to four decimals."),
        "detectors_in_frozen_telemetry": methods,
        "our_detectors_scored_on_the_fold": ours,
        "n_our_detectors": len(ours),
        "standalone_probe_artifacts": probes,
        "n_standalone_probes": len(probes),
        "n_distinct_architectures_evaluated": len(set(ours) | probe_methods),
        "indexed_read_sequence": indexed,
        "n_indexed_reads": len(indexed),
        "n_probes_that_scored_the_fold": n_scored,
        "n_probes_that_only_resummarised": n_resummary,
        "honest_summary": (
            f"{len(ours)} of our detectors appear in the frozen telemetry. "
            f"{len(probes)} standalone artifacts are further accesses: "
            f"{n_scored} of them scored the 192 units again, and "
            f"{n_resummary} drew a new inference from per-unit numbers an "
            f"earlier read had already frozen, which scores nothing but uses "
            f"the fold again and is indexed as such. {len(indexed)} carry a "
            f"read index, {n_table} of them readings of the architecture "
            f"reported as the main result and {len(indexed) - n_table} of a "
            f"different lineage, and every one is reported with the reason it "
            f"was taken. The wider programme has looked at this fold more "
            f"often than that, which is what this ledger is for."),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    led = build()
    text = json.dumps(led, indent=2, allow_nan=False) + "\n"

    if args.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        if json.loads(OUT.read_text()) != led:
            print("STALE TEST_FOLD_ACCESS_LEDGER.json: the artifacts on disk "
                  "record a different set of test-fold evaluations")
            return 1
        print(f"test-fold ledger current: "
              f"{led['n_distinct_architectures_evaluated']} architectures, "
              f"{led['n_standalone_probes']} standalone probes")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"our detectors in frozen telemetry: {led['n_our_detectors']}  "
          f"({', '.join(led['our_detectors_scored_on_the_fold'])})")
    print(f"standalone probe artifacts: {led['n_standalone_probes']}")
    for p in led["standalone_probe_artifacts"]:
        auc = p["mean_residue_auc"]
        print(f"  {p['artifact']:52s} {p['method']:22s} "
              f"{'' if auc is None else f'{auc:.4f}'}")
    print(f"distinct architectures evaluated on the fold: "
          f"{led['n_distinct_architectures_evaluated']}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
