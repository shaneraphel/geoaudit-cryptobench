#!/usr/bin/env python3
"""Read 13: chains the counting field ranks and all three baselines do not.

Runs only under ``results/architecture_sweep/PREREGISTERED_RECOVERY.json``, and
refuses until that plan is committed and is an ancestor of HEAD. Everything this
file decides -- the thresholds, the ladder, the mirror, the exclusion -- is read
out of the plan rather than written here, so the plan is the specification and
this is the execution.

The one thing worth restating: the rule was fixed on the *training* fold, in
``tools/found_where_baselines_missed.py``, and both that tool and its artifact
are hashed into the plan. If either has moved since, this refuses to run, because
then the thresholds could have been chosen after seeing the fold and nobody could
prove otherwise.

Usage: PYTHONPATH=src:tools python3.12 tools/recovery_read.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from pocket_bench.metrics import roc_auc
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.recovery_read.v1"
READ_INDEX = 13
PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_RECOVERY.json"
OUT = ROOT / "results/official_fold/RECOVERY_READ.json"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
PLMNN = ROOT / "results/baselines/PLMNN_SCORES.json"
POCKETMINER = ROOT / "results/baselines/POCKETMINER_SCORES.json"
PM_SCORES = ROOT / "data/baselines/pocketminer"
LABELS = ROOT / "data/cryptobench_apo/official_labels"
TRAIN_TOOL = ROOT / "tools/found_where_baselines_missed.py"
TRAIN_ARTIFACT = ROOT / "results/architecture_sweep/RECOVERED_UNITS_TRAIN.json"

BASELINES = ("p2rank", "plmnn", "pocketminer")


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _resnum(x) -> int | None:
    if isinstance(x, int):
        return x
    s, digits, negative = str(x), "", False
    for ch in reversed(s):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            negative = ch == "-"
            break
    if not digits:
        return None
    return -int(digits) if negative else int(digits)


def guards() -> dict:
    """Every condition the plan lists, checked before a label is opened."""
    rel = str(PLAN.relative_to(ROOT))
    if _git("status", "--porcelain", "--", rel):
        raise SystemExit(f"{rel} has uncommitted changes; the plan must be "
                         f"committed before the read")
    sha = _git("log", "-1", "--format=%H", "--", rel)
    if not sha:
        raise SystemExit(f"{rel} is not committed")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha, "HEAD"],
                      cwd=ROOT).returncode != 0:
        raise SystemExit("the plan's commit is not an ancestor of HEAD")

    plan = json.loads(PLAN.read_text())
    src = plan["the_rule"]["where_the_rule_comes_from"]
    for key, path in (("tool_sha256", TRAIN_TOOL),
                      ("artifact_sha256", TRAIN_ARTIFACT)):
        if src[key] != _sha(path):
            raise SystemExit(
                f"{path.name} has changed since the plan was written, so the "
                f"rule this read would apply is not the rule that was fixed on "
                f"the training fold. Refusing.")
    w = plan["where_each_number_comes_from"]
    for key, path in (("telemetry_sha256", TELEMETRY),
                      ("plmnn_scores_sha256", PLMNN),
                      ("pocketminer_scores_sha256", POCKETMINER)):
        if w[key] != _sha(path):
            raise SystemExit(f"{path.name} changed between the plan and the "
                             f"read")
    return {"plan": {"artifact": rel, "commit": sha,
                     "committed_at": _git("log", "-1", "--format=%cI", sha),
                     "subject": _git("log", "-1", "--format=%s", sha),
                     "is_ancestor_of_head": True},
            "plan_sha256": _sha(PLAN),
            "inputs_unchanged_since_the_plan": True,
            "rule_source_unchanged_since_the_plan": True}


def frozen_per_unit() -> tuple[dict[str, float], dict[str, float]]:
    tel = json.loads(TELEMETRY.read_text())
    rows = tel["rows"] if isinstance(tel, dict) and "rows" in tel else tel
    ours, p2 = {}, {}
    for r in rows:
        v = r.get("residue_auc")
        if v is None:
            continue
        if r["method"] == "table_field":
            ours[r["unit_id"]] = float(v)
        elif r["method"] == "p2rank":
            p2[r["unit_id"]] = float(v)
    return ours, p2


def truth_of(unit: str) -> set[int]:
    d = json.loads((LABELS / f"{unit}_labels.json").read_text())
    return {r for r in (_resnum(x) for x in
                        (d.get("cryptic_residues")
                         or d.get("binding_residues") or []))
            if r is not None}


def auc_from_scores(by_res: dict[int, float], truth: set[int]) -> float | None:
    """The telemetry's own statistic: roc_auc over the sorted-resseq universe."""
    keys = sorted(by_res)
    if not keys or not (truth & set(keys)):
        return None
    s = [by_res[k] for k in keys]
    y = [1 if k in truth else 0 for k in keys]
    return roc_auc(s, y)


def plmnn_per_unit() -> dict[str, float]:
    out = {}
    for u in json.loads(PLMNN.read_text())["units"]:
        by = {int(k): float(v) for k, v in u["scores"].items()}
        a = auc_from_scores(by, truth_of(u["unit_id"]))
        if a is not None:
            out[u["unit_id"]] = float(a)
    return out


def pocketminer_per_unit() -> tuple[dict[str, float], dict[str, dict]]:
    flags = {}
    for u in json.loads(POCKETMINER.read_text())["units"]:
        flags[u["unit"]] = {
            "pocketminer_agrees_with_official_featurisation":
                u.get("agrees_with_official_featurisation"),
            "n_dropped_by_pocketminer": len(u.get("dropped") or []),
            "n_resseq_collisions": len(
                u.get("resseq_collisions_from_insertion_codes") or []),
        }
    out = {}
    for f in sorted(PM_SCORES.glob("*.json")):
        unit = f.stem
        raw = json.loads(f.read_text())["residue_scores"]
        by = {r: float(v) for r, v in
              ((_resnum(k), v) for k, v in raw.items()) if r is not None}
        a = auc_from_scores(by, truth_of(unit))
        if a is not None:
            out[unit] = float(a)
    return out, flags


def build() -> dict:
    g = guards()
    plan = json.loads(PLAN.read_text())
    rule = plan["the_rule"]
    found = 0.80
    missed = 0.55
    ladder = [tuple(x) for x in rule["ladder"]]

    ours, p2 = frozen_per_unit()
    pl = plmnn_per_unit()
    pm, flags = pocketminer_per_unit()
    shared = sorted(set(ours) & set(p2) & set(pl) & set(pm))

    def clean(u: str) -> bool:
        f = flags.get(u, {})
        return (f.get("pocketminer_agrees_with_official_featurisation") is True
                and not f.get("n_dropped_by_pocketminer")
                and not f.get("n_resseq_collisions"))

    rec, mir, caveat = [], [], []
    for u in shared:
        o = ours[u]
        b = {"p2rank": p2[u], "plmnn": pl[u], "pocketminer": pm[u]}
        row = {"unit": u, "ours": round(o, 4),
               **{k: round(v, 4) for k, v in b.items()},
               "best_baseline": round(max(b.values()), 4),
               "margin_over_the_best_baseline": round(o - max(b.values()), 4),
               "n_cryptic": len(truth_of(u)),
               "every_method_verified_on_the_same_residues": clean(u),
               **flags.get(u, {})}
        if o >= found and all(v <= missed for v in b.values()):
            (rec if clean(u) else caveat).append(row)
        if all(v >= found for v in b.values()) and o <= missed:
            mir.append(row)
    rec.sort(key=lambda r: -r["margin_over_the_best_baseline"])
    caveat.sort(key=lambda r: -r["margin_over_the_best_baseline"])
    mir.sort(key=lambda r: r["margin_over_the_best_baseline"])

    lad = []
    for f, m in ladder:
        nr = sum(1 for u in shared
                 if ours[u] >= f and max(p2[u], pl[u], pm[u]) <= m)
        nm = sum(1 for u in shared
                 if min(p2[u], pl[u], pm[u]) >= f and ours[u] <= m)
        lad.append({"found_at_or_above": f, "missed_at_or_below": m,
                    "n_recovered": nr, "n_mirror": nm, "difference": nr - nm})

    every = all(r["difference"] > 0 for r in lad)
    if every and len(rec) + len(caveat) > len(mir):
        outcome, key = "recoveries_exceed_mirrors", "recoveries_exceed_mirrors"
    elif len(mir) > len(rec) + len(caveat):
        outcome, key = "mirrors_exceed_recoveries", "mirrors_exceed_recoveries"
    else:
        outcome, key = "comparable", "comparable"

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "test_fold_read_index": READ_INDEX,
        "status": "exploratory",
        "method": "table_field",
        "question": plan["question"],
        **g,
        "rescored_anything": False,
        "why_it_is_indexed_anyway": (
            "no model is re-run and no threshold is re-fitted, but the labels "
            "are opened and a new statistic is formed over the held-out units. "
            "A ledger that counted only re-runs would make an unlimited number "
            "of such statements free"),
        "rule_applied": {
            "found_at_or_above": found, "missed_at_or_below": missed,
            "recovery": rule["recovery"], "mirror": rule["mirror"],
            "fixed_on": "the training fold, in "
                        + str(TRAIN_TOOL.relative_to(ROOT)),
        },
        "n_units_compared": len(shared),
        "n_recovered": len(rec),
        "n_recovered_with_a_parsing_caveat": len(caveat),
        "n_mirror": len(mir),
        "threshold_ladder": lad,
        "difference_positive_at_every_setting": every,
        "recovered": rec,
        "recovered_but_the_file_is_parsed_differently": caveat,
        "mirror": mir,
        "outcome": outcome,
        "conclusion": plan["what_will_be_written_under_each_outcome"][key],
        "what_this_cannot_show": plan["what_this_cannot_show"],
        "per_unit": [{"unit": u, "ours": round(ours[u], 4),
                      "p2rank": round(p2[u], 4), "plmnn": round(pl[u], 4),
                      "pocketminer": round(pm[u], 4)} for u in shared],
    }


def _report(d: dict) -> None:
    print(f"read {d['test_fold_read_index']} ({d['status']}), "
          f"{d['n_units_compared']} units")
    print(f"  ours >= {d['rule_applied']['found_at_or_above']} and all three "
          f"baselines <= {d['rule_applied']['missed_at_or_below']}: "
          f"{d['n_recovered']} units"
          + (f" (+{d['n_recovered_with_a_parsing_caveat']} set aside for "
             f"parsing)" if d['n_recovered_with_a_parsing_caveat'] else ""))
    print(f"  the mirror: {d['n_mirror']} units")
    print(f"\n  {'found>=':>8s} {'missed<=':>9s} {'ours':>6s} {'mirror':>7s} "
          f"{'diff':>6s}")
    for r in d["threshold_ladder"]:
        print(f"  {r['found_at_or_above']:8.2f} {r['missed_at_or_below']:9.2f} "
              f"{r['n_recovered']:6d} {r['n_mirror']:7d} {r['difference']:+6d}")
    if d["recovered"]:
        print(f"\n  {'unit':10s} {'cryptic':>8s} {'ours':>7s} {'p2rank':>7s} "
              f"{'plmnn':>7s} {'pocketm':>8s}")
        for r in d["recovered"]:
            print(f"  {r['unit']:10s} {r['n_cryptic']:8d} {r['ours']:7.3f} "
                  f"{r['p2rank']:7.3f} {r['plmnn']:7.3f} "
                  f"{r['pocketminer']:8.3f}")
    for r in d["recovered_but_the_file_is_parsed_differently"]:
        print(f"  {r['unit']:10s} set aside: parsed differently")
    if d["mirror"]:
        print(f"\n  the mirror:")
        for r in d["mirror"]:
            print(f"  {r['unit']:10s} {r['n_cryptic']:8d} {r['ours']:7.3f} "
                  f"{r['p2rank']:7.3f} {r['plmnn']:7.3f} "
                  f"{r['pocketminer']:8.3f}")
    print(f"\n  outcome: {d['outcome']}")


def check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    plan = json.loads(PLAN.read_text())
    if d.get("schema") != SCHEMA:
        print(f"FAILED: schema {d.get('schema')}")
        return 1
    if d.get("test_fold_read_index") != READ_INDEX:
        print(f"FAILED: the artifact does not index itself as read {READ_INDEX}")
        return 1
    if d.get("status") != "exploratory":
        print(f"FAILED: status {d.get('status')}")
        return 1
    if d["conclusion"] not in plan[
            "what_will_be_written_under_each_outcome"].values():
        print("FAILED: the reported conclusion is not one the plan wrote")
        return 1
    if [[r["found_at_or_above"], r["missed_at_or_below"]]
            for r in d["threshold_ladder"]] != plan["the_rule"]["ladder"]:
        print("FAILED: the ladder is not the one the plan fixed")
        return 1
    _report(d)
    print(f"OK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        return check()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
