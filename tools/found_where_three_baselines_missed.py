#!/usr/bin/env python3
"""The recovery count with pLM-NN as a third baseline, on the training fold.

Why this exists
---------------
``RECOVERED_UNITS_TRAIN.json`` counts the training units where the counting
field ranks a chain's cryptic residues well while the published baselines rank
them at chance. It carries a field named ``plmnn_absent`` saying what it is not:

    pLM-NN has never been run on the training fold; it is a five-hour encoder
    pass rather than a lookup. Every count here is against two baselines, not
    three, and the official-fold version that includes it is a read of the
    held-out set and needs a plan and an index first.

So "P2Rank and pLM-NN both missed it" has had no support anywhere. On the
training fold pLM-NN had never been run; on the official fold it had, and the
preregistered read came back **0 recoveries against 1 mirror**
(``RECOVERY_READ.json``), which is the answer and is negative.

**This tool cannot supply the missing arm, and the reason is recorded here
because it is the reason somebody will try again.**

The plan was to run pLM-NN over the training fold and join those per-unit AUCs
in as a third baseline. It costs no test-fold read, so it looked free. It is not
a measurement. The head is CryptoBench's published ``best_trained``, fitted by
the authors on their training folds, and ``TRAIN_MANIFEST.json`` says our
training partition *is* ``train-0`` through ``train-3``. The 770 chains are the
model's own fitting set.

The numbers say it plainly: pLM-NN scores **0.8235** mean per-unit ROC-AUC on
the official fold (``PLMNN_READ.json``) and **0.98** on the chains it was fitted
on. The recovery rule asks whether a baseline sits *at chance* on a unit, and a
model that has memorised a chain is never at chance on it. Feeding these scores
in would drive the recovery count to zero — not because the four cases fail
against a third method, but because the third method was shown the answer key.
That is a confidently wrong answer, which is worse than no answer, so the tool
refuses rather than producing it. See ``AGENT_MEMORY`` 2h-bis.

**What would make the measurement possible.** A pLM-NN whose fitting set is
disjoint from the units being scored. Three routes, none of them free: retrain
the published architecture on a split that excludes the units of interest, which
makes it our model and not the published baseline; use the external sets, which
are frozen and spending one destroys the confirmatory result it exists for; or
accept ``RECOVERY_READ.json``'s official-fold answer, which was preregistered,
already read, and negative. The third is the honest default.

The rule is read, not restated
------------------------------
Every threshold, the statistic, and the ladder come out of
``RECOVERED_UNITS_TRAIN.json``. Nothing is redeclared in this file. That is the
whole reason it can be trusted: a third baseline changes the counts, and an
author who both chooses the thresholds and sees the counts is searching. The
rule was fixed before the two-baseline counts were read, it is the same rule
here, and if the artifact's ``fixed_before_the_counts_were_read`` is ever false
this tool refuses to run.

What the answer can be, written before running it
-------------------------------------------------
Adding a baseline to a conjunction can only remove recoveries, never add one:
a unit needs *all* baselines below the missed threshold, so the count is
monotone non-increasing in the number of baselines. The mirror count moves the
other way -- it needs *any* baseline above the found threshold -- so it is
monotone non-decreasing. **The honest expectation is therefore that the
asymmetry shrinks.** The interesting quantity is not whether it shrinks but
whether the four clean cases survive, because a case that survives a third and
architecturally unrelated baseline is a different claim from one that survives
two.

If none survives, that is the result and it is reportable: it would say the
two-baseline table was two methods failing together rather than a site only this
detector sees, and §2l's official-fold negative would stop looking like a
power problem.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR / "src"), str(ROOT_DIR / "tools")]

from pocket_bench.paths import ROOT                                # noqa: E402

SCHEMA = "geoaudit.found_where_three_baselines_missed.v1"
TWO = ROOT / "results/architecture_sweep/RECOVERED_UNITS_TRAIN.json"
CKPT = ROOT / "results/baselines/_plmnn_train_checkpoint.jsonl"
OUT = ROOT / "results/architecture_sweep/RECOVERED_UNITS_TRAIN_THREE.json"

BASELINES = ("p2rank", "pocketminer", "plmnn")


FIT_SET_REFUSAL = """\
This tool will not run, and the refusal is the result.

It was written to add pLM-NN as a third baseline to the training-fold recovery
count. That cannot be done with the checkpoint it reads, because the pLM-NN head
is CryptoBench's published best_trained, fitted by the authors on their training
folds, and data/cryptobench_apo/TRAIN_MANIFEST.json records our training
partition as exactly those folds: train-0 through train-3. Every chain in the
checkpoint is a chain the model was fitted on.

  pLM-NN, official test fold   0.8235   (results/official_fold/PLMNN_READ.json)
  pLM-NN, its own fitting set  {mean:.4f}   ({n} chains scored)
  difference                   {gap:+.4f}

The recovery rule asks whether a baseline sits at chance on a unit. A model that
has memorised a chain is never at chance on it, so these scores would drive the
count to zero for a reason that has nothing to do with the method under test.
Producing that number would be worse than producing none.

What would make the measurement possible, none of it free:

  * retrain the published architecture on a split excluding the units scored,
    which makes it our model rather than the published baseline;
  * use an external set, which is frozen, and spending one destroys the
    confirmatory result it exists for;
  * accept results/official_fold/RECOVERY_READ.json, which asked this question
    under a preregistered plan against all three baselines and returned
    0 recoveries against 1 mirror. It is negative and it is the honest default.
"""


def _refuse_on_fit_set(plm: dict[str, float]) -> None:
    """Stop before turning a memorised baseline into a recovery count."""
    if not plm:
        return
    mean = sum(plm.values()) / len(plm)
    raise SystemExit(FIT_SET_REFUSAL.format(
        mean=mean, n=len(plm), gap=mean - 0.823469))


def _plmnn() -> dict[str, float]:
    """Per-unit pLM-NN ROC-AUC from the training-fold checkpoint.

    Units whose AUC is undefined -- no cryptic residue, or every residue
    cryptic -- are dropped rather than defaulted. A unit with no positives has
    not been "missed" by pLM-NN at chance; it has no ranking problem to get
    right, and scoring it 0.5 would manufacture recoveries out of degenerate
    labels.
    """
    if not CKPT.exists():
        raise SystemExit(
            f"{CKPT.relative_to(ROOT)} does not exist. Run\n"
            f"  PYTHONPATH=src:tools python3.12 tools/plmnn_by_stratum.py "
            f"--embed\n"
            f"first; it is one ESM2-3B pass over the training chains and takes "
            f"about four and a half hours. Nothing here is a default: without "
            f"that file the third baseline is absent, not zero.")
    out: dict[str, float] = {}
    for line in CKPT.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        auc = r.get("auc")
        if auc is None or auc != auc:  # NaN: undefined, not chance
            continue
        out[r["unit_id"]] = float(auc)
    return out


def _count(rows: list[dict], found: float, missed: float,
           baselines: tuple[str, ...]) -> tuple[list[dict], list[dict]]:
    rec, mir = [], []
    for r in rows:
        bs = [r[b] for b in baselines]
        if r["ours"] >= found and all(b <= missed for b in bs):
            rec.append(r)
        if r["ours"] <= missed and any(b >= found for b in bs):
            mir.append(r)
    return rec, mir


def build(write: bool) -> int:
    two = json.loads(TWO.read_text())
    rule = two["rule"]
    if not rule.get("fixed_before_the_counts_were_read"):
        raise SystemExit(
            "the two-baseline artifact does not declare its rule as fixed "
            "before the counts were read, so re-running it with a third "
            "baseline would be a search over thresholds")
    if two.get("reads_test_fold") is not False:
        raise SystemExit("the source artifact reads the test fold")

    plm = _plmnn()
    _refuse_on_fit_set(plm)
    joined, unmatched = [], []
    for r in two["per_unit"]:
        if r["unit"] in plm:
            joined.append({**r, "plmnn": plm[r["unit"]]})
        else:
            unmatched.append(r["unit"])

    found = float(rule["found_at_or_above"])
    missed = float(rule["missed_at_or_below"])
    rec3, mir3 = _count(joined, found, missed, BASELINES)
    rec2, mir2 = _count(joined, found, missed, ("p2rank", "pocketminer"))

    ladder = []
    for row in two["threshold_ladder"]:
        f, m = float(row["found_at_or_above"]), float(row["missed_at_or_below"])
        r3, m3 = _count(joined, f, m, BASELINES)
        r2, m2 = _count(joined, f, m, ("p2rank", "pocketminer"))
        ladder.append({
            "found_at_or_above": f, "missed_at_or_below": m,
            "n_recovered_three_baselines": len(r3),
            "n_mirror_three_baselines": len(m3),
            "difference_three": len(r3) - len(m3),
            "n_recovered_two_baselines_same_units": len(r2),
            "n_mirror_two_baselines_same_units": len(m2),
        })

    # What happened to each unit the two-baseline table named. This is the
    # question the tool exists for and it is reported unit by unit, including
    # the ones that did not survive, because reporting only survivors is the
    # selection the ladder above exists to prevent.
    named = [r["unit"] for r in two["recovered"]]
    fate = []
    for unit in named:
        row = next((j for j in joined if j["unit"] == unit), None)
        if row is None:
            fate.append({"unit": unit, "outcome": "no pLM-NN score",
                         "why": "not in the checkpoint; the join lost it"})
            continue
        survives = row in rec3
        fate.append({
            "unit": unit, "ours": row["ours"], "p2rank": row["p2rank"],
            "pocketminer": row["pocketminer"], "plmnn": round(row["plmnn"], 4),
            "n_cryptic": row["n_cryptic"], "n_residues": row["n_residues"],
            "survives_a_third_baseline": survives,
            "why_not": (None if survives else
                        f"pLM-NN scores {row['plmnn']:.4f}, above the "
                        f"missed-at-or-below threshold of {missed}"),
        })

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": (
            "on how many training units does the counting field rank the "
            "cryptic binding residues well while all three published baselines "
            "rank them at chance"),
        "rule": {**rule, "source": TWO.relative_to(ROOT).as_posix(),
                 "restated_here": False,
                 "why_read_rather_than_restated": (
                     "a third baseline changes the counts, and an author who "
                     "both picks the thresholds and sees the counts is "
                     "searching. These are the thresholds fixed before the "
                     "two-baseline counts were read")},
        "monotonicity": {
            "recoveries_cannot_increase": True,
            "mirrors_cannot_decrease": True,
            "why": (
                "a recovery needs every baseline below the missed threshold and "
                "a mirror needs any baseline above the found threshold, so "
                "adding a baseline tightens the first conjunction and loosens "
                "the second. The count going down is not evidence against the "
                "method and the count staying up is the informative outcome"),
            "stated_before_the_run": True,
        },
        "join": {
            "n_units_in_the_two_baseline_table": len(two["per_unit"]),
            "n_with_a_plmnn_score": len(joined),
            "n_without": len(unmatched),
            "first_unmatched": unmatched[:10],
            "why_units_are_dropped_not_defaulted": (
                "a unit whose pLM-NN AUC is undefined has no positives or no "
                "negatives; it has not been missed at chance, and scoring it "
                "0.5 would manufacture recoveries out of degenerate labels"),
        },
        "n_recovered_three_baselines": len(rec3),
        "n_mirror_three_baselines": len(mir3),
        "n_recovered_two_baselines_on_the_same_units": len(rec2),
        "n_mirror_two_baselines_on_the_same_units": len(mir2),
        "why_the_two_baseline_count_is_recomputed_here": (
            "so the comparison is on one set of units. The published "
            "two-baseline count is over all units in that table, and any unit "
            "the pLM-NN join lost would otherwise show up as a loss caused by "
            "the third baseline when it was caused by the join"),
        "recovered": rec3,
        "mirror": mir3,
        "what_became_of_the_two_baseline_cases": fate,
        "threshold_ladder": ladder,
        "what_a_recovery_is_not": two["what_a_recovery_is_not"],
        "what_this_does_not_do": (
            "it does not read the official fold. RECOVERY_READ.json is the "
            "preregistered official-fold read of this same rule and it returned "
            "0 recoveries against 1 mirror; nothing here revises that, and a "
            "training-fold count is exploratory whichever way it falls"),
    }

    print(f"units joined      {len(joined)} of {len(two['per_unit'])}"
          f"  ({len(unmatched)} without a pLM-NN score)")
    print(f"three baselines   {len(rec3)} recovered, {len(mir3)} mirror")
    print(f"two, same units   {len(rec2)} recovered, {len(mir2)} mirror")
    print("\nthe four named cases:")
    for f in fate:
        if "ours" not in f:
            print(f"  {f['unit']}  {f['outcome']}")
            continue
        mark = "survives" if f["survives_a_third_baseline"] else "falls"
        print(f"  {f['unit']}  ours {f['ours']:.3f}  p2rank {f['p2rank']:.3f}  "
              f"pocketminer {f['pocketminer']:.3f}  plmnn {f['plmnn']:.3f}  "
              f"-> {mark}")
    print("\nladder (found / missed -> three, two):")
    for row in ladder:
        print(f"  {row['found_at_or_above']:.2f} / "
              f"{row['missed_at_or_below']:.2f}   "
              f"{row['n_recovered_three_baselines']:2d} vs "
              f"{row['n_mirror_three_baselines']:2d}"
              f"   ({row['n_recovered_two_baselines_same_units']:2d} vs "
              f"{row['n_mirror_two_baselines_same_units']:2d} on two)")

    if write:
        OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    return build(ap.parse_args(argv).write)


if __name__ == "__main__":
    raise SystemExit(main())
