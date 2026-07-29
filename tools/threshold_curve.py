#!/usr/bin/env python3
"""The twelfth read: precision, recall, F1 and MCC at every calling fraction.

Nothing is rescored. Every score this reads was frozen by an earlier read; what is
new is that they are binarised at 39 cut points instead of three, which is a new
statement about the held-out fold and is indexed as one.

The plan forbids selecting anything from the result, and this file records that
its output feeds no configuration. What it does report, besides the curves, is the
one property of a curve that a chosen point cannot fake: the span of cut points
over which the paired difference keeps a single sign.

At q = 0.09 the curve recomputes a number the seventh read published, and it has
to reproduce it to six decimals or the read is void -- because if the two disagree
there, the curve is measuring something other than what the paper reports.

Usage: PYTHONPATH=src:tools python3.12 tools/threshold_curve.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

import numpy as np

from matched_full_read import (_resnum, confusion, load_truth, metric_of,
                              top_q_call)
from plmnn_read import _interval
from pocket_bench.paths import ROOT

PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_THRESHOLD_CURVE.json"
PREDS = ROOT / "results/cryptobench_official/predictions"
POCKETMINER_DIR = ROOT / "data/baselines/pocketminer"
PLMNN_SCORES = ROOT / "results/baselines/PLMNN_SCORES.json"
FULL_READ = ROOT / "results/official_fold/MATCHED_FULL_READ.json"
OUT = ROOT / "results/official_fold/THRESHOLD_CURVE.json"

SCHEMA = "geoaudit.threshold_curve.v1"
READ_INDEX = 12
METHOD = "table_field"
METRICS = ("precision", "recall", "positive_class_f1", "mcc")


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _plan() -> tuple[dict, dict]:
    rel = str(PLAN.relative_to(ROOT))
    if _git("rev-parse", "--is-shallow-repository") == "true":
        raise SystemExit("shallow clone; the plan's ordering cannot be checked")
    if _git("status", "--porcelain", "--", rel):
        raise SystemExit(f"{rel} is modified or untracked; this read is refused")
    sha = _git("log", "-1", "--format=%H", "--", rel)
    if not sha:
        raise SystemExit(f"{rel} has no commit; this read is refused")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha,
                       _git("rev-parse", "HEAD")], cwd=ROOT).returncode != 0:
        raise SystemExit(f"{sha[:12]} is not an ancestor of HEAD")
    return json.loads(PLAN.read_text()), {
        "artifact": rel, "committed_in": sha,
        "committed_at": _git("log", "-1", "--format=%cI", sha),
        "subject": _git("log", "-1", "--format=%s", sha),
        "is_ancestor_of_head": True}


def _keyed(raw: dict) -> dict[int, float]:
    """Residue numbers through the harness's own parser, so a chain with insertion
    codes lands on the same keys here as in the reads this curve has to reproduce."""
    out = {}
    for k, v in raw.items():
        r = _resnum(k)
        if r is not None:
            out[r] = float(v)
    return out


def _from_predictions(name: str) -> dict[str, dict[int, float]]:
    f = PREDS / f"{name}.json"
    if not f.is_file():
        return {}
    return {u["unit_id"]: _keyed(u.get("residue_scores") or {})
            for u in json.loads(f.read_text())["units"]}


def _pocketminer() -> dict[str, dict[int, float]]:
    if not POCKETMINER_DIR.is_dir():
        return {}
    out = {}
    for f in sorted(POCKETMINER_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        out[f.stem] = _keyed(d["residue_scores"])
    return out


def _plmnn() -> dict[str, dict[int, float]]:
    if not PLMNN_SCORES.is_file():
        return {}
    return {u["unit_id"]: _keyed(u["scores"])
            for u in json.loads(PLMNN_SCORES.read_text())["units"]}


def _grid(plan: dict) -> list[float]:
    g = plan["grid"]
    n = int(g["n"])
    return [round(g["low"] + i * g["step"], 2) for i in range(n)]


def _sign_span(rows: list[dict], metric: str) -> dict:
    """The stretch of cut points over which the difference keeps one sign.

    A property of the whole curve, which is what the plan allows to be reported;
    the best point of a curve is what it does not.
    """
    signs = [(r["q"], (1 if r[metric]["mean"] > 0 else
                       -1 if r[metric]["mean"] < 0 else 0)) for r in rows]
    pos = [q for q, s in signs if s > 0]
    neg = [q for q, s in signs if s < 0]
    crossings = sum(1 for a, b in zip(signs, signs[1:])
                    if a[1] * b[1] < 0)
    return {"n_cut_points": len(signs),
            "n_where_ours_is_higher": len(pos),
            "n_where_theirs_is_higher": len(neg),
            "sign_changes": crossings,
            "one_sign_throughout": crossings == 0,
            "first_cut_point_where_ours_is_higher": min(pos) if pos else None,
            "last_cut_point_where_ours_is_higher": max(pos) if pos else None}


def _arm(ours: dict[str, dict[int, float]], theirs: dict[str, dict[int, float]],
         truth: dict[str, set[int]], grid: list[float], seed: int,
         n_boot: int) -> dict:
    """One baseline, every cut point, on the residues both methods scored."""
    units = []
    not_shared = 0
    for uid in sorted(set(ours) & set(theirs)):
        pos = truth.get(uid) or set()
        keys = sorted(set(ours[uid]) & set(theirs[uid]))
        if len(keys) != len(ours[uid]):
            not_shared += 1
        if not keys or not (pos & set(keys)):
            continue
        units.append((np.array([ours[uid][k] for k in keys]),
                      np.array([theirs[uid][k] for k in keys]),
                      np.array([k in pos for k in keys], dtype=bool)))
    rows = []
    for i, q in enumerate(grid):
        conf_o = [confusion(top_q_call(a, q), y) for a, _, y in units]
        conf_t = [confusion(top_q_call(b, q), y) for _, b, y in units]
        row = {"q": q}
        for k, name in enumerate(METRICS):
            vo = [metric_of(name, c) for c in conf_o]
            vt = [metric_of(name, c) for c in conf_t]
            d = np.array([a - b for a, b in zip(vo, vt)
                          if a is not None and b is not None])
            blk = _interval(d, seed + 1000 * i + k, n_boot, 0.05)
            blk.pop("ci_bonferroni", None)
            row[name] = {
                "ours": round(float(np.mean([v for v in vo if v is not None])), 6),
                "theirs": round(float(np.mean([v for v in vt if v is not None])), 6),
                **blk}
        rows.append(row)
    return {"n_units": len(units),
            "n_units_where_the_residue_sets_differ": not_shared,
            "interval_kind": "pointwise 95%, uncorrected, on a display",
            "curve": rows,
            "sign_span": {m: _sign_span(rows, m) for m in METRICS}}


def build() -> dict:
    plan, prov = _plan()
    grid = _grid(plan)
    st = plan["statistic"]
    n_boot, seed = int(st["n_boot"]), int(st["seed"])
    truth = load_truth()
    ours = _from_predictions("table_field")
    baselines = {"p2rank": _from_predictions("p2rank"),
                 "pocketminer": _pocketminer(),
                 "plmnn": _plmnn()}
    arms, missing = {}, []
    for i, (name, sc) in enumerate(baselines.items()):
        if not sc:
            missing.append(name)
            continue
        arms[name] = _arm(ours, sc, truth, grid, seed + 7919 * i, n_boot)

    g = plan["reproduction_guard"]
    q = float(g["at"])
    at = next(r for r in arms["p2rank"]["curve"] if abs(r["q"] - q) < 1e-9)
    for side, want in (("ours", g["positive_class_f1_ours"]),
                       ("theirs", g["positive_class_f1_theirs"])):
        got = at["positive_class_f1"][side]
        if abs(got - float(want)) > 1e-6:
            raise SystemExit(
                f"at q={q} this curve gives {side} positive-class F1 {got} but "
                f"read seven published {want}; the curve is measuring something "
                f"else and this read is void")

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "is_official_mmseqs2_10pct_test_fold": True,
        "test_fold_read_index": READ_INDEX,
        "status": plan["status_declared_in_advance"],
        "method": METHOD,
        "question": plan["question"],
        "rescored_anything": False,
        "why_it_is_indexed_anyway": (
            "the scores were frozen by earlier reads; binarising them at 39 cut "
            "points instead of three is a new statement about the fold"),
        "plan": prov,
        "grid": plan["grid"],
        "selects_nothing": {
            "consumed_by_any_configuration": False,
            "deployed_q_unchanged": float(plan["reproduction_guard"]["at"]),
            "what_was_committed": plan["nothing_is_selected_from_this_curve"],
        },
        "reproduced_read_seven_at_the_deployed_point": {
            "q": q,
            "positive_class_f1_ours": at["positive_class_f1"]["ours"],
            "positive_class_f1_theirs": at["positive_class_f1"]["theirs"],
            "tolerance": 1e-6,
            "agrees": True,
        },
        "baselines_missing": missing,
        "why_missing_is_not_zero": (
            "a baseline whose scores are absent is recorded as absent. Scoring it "
            "as zeros would manufacture an advantage"),
        "arms": arms,
        "multiplicity": plan["multiplicity"],
        "conclusion_key": _conclusion_key(arms),
        "conclusion": plan["what_will_be_written_under_each_outcome"][
            _conclusion_key(arms)],
    }


def _conclusion_key(arms: dict) -> str:
    s = arms["p2rank"]["sign_span"]["positive_class_f1"]
    if s["sign_changes"] == 0 and s["n_where_ours_is_higher"] == s["n_cut_points"]:
        return "our_curve_is_above_theirs_throughout"
    if s["sign_changes"] == 0 and s["n_where_theirs_is_higher"] == s["n_cut_points"]:
        return "their_curve_is_above_ours_throughout"
    return "the_curves_cross"


def _report(d: dict) -> None:
    print(f"read {d['test_fold_read_index']} ({d['status']}), "
          f"{d['grid']['n']} cut points")
    if d["baselines_missing"]:
        print(f"  absent baselines: {', '.join(d['baselines_missing'])}")
    for name, arm in d["arms"].items():
        s = arm["sign_span"]["positive_class_f1"]
        print(f"  vs {name}: {arm['n_units']} units, F1 difference positive at "
              f"{s['n_where_ours_is_higher']}/{s['n_cut_points']} cut points, "
              f"{s['sign_changes']} sign change(s)")
        excl = [r["q"] for r in arm["curve"]
                if r["positive_class_f1"]["excludes_zero"]]
        print(f"     pointwise interval excludes zero at "
              f"{len(excl)} cut points" + (f": {excl[0]}\u2013{excl[-1]}"
                                           if excl else ""))
    print(f"  outcome: {d['conclusion_key']}")


def check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    if d.get("schema") != SCHEMA:
        print(f"FAILED: schema {d.get('schema')}")
        return 1
    if d.get("test_fold_read_index") != READ_INDEX:
        print("FAILED: the artifact does not index itself as read "
              f"{READ_INDEX}")
        return 1
    if d["selects_nothing"]["consumed_by_any_configuration"]:
        print("FAILED: the artifact admits feeding a configuration, which the "
              "plan forbids")
        return 1
    if not d["reproduced_read_seven_at_the_deployed_point"]["agrees"]:
        print("FAILED: the curve does not reproduce read seven at the deployed q")
        return 1
    plan = json.loads(PLAN.read_text())
    if d["conclusion"] not in plan[
            "what_will_be_written_under_each_outcome"].values():
        print("FAILED: the reported conclusion is not one the plan wrote")
        return 1
    for name, arm in d["arms"].items():
        qs = [r["q"] for r in arm["curve"]]
        if qs != sorted(qs):
            print(f"FAILED: the {name} curve is not ordered by q")
            return 1
        for r in arm["curve"]:
            for m in METRICS:
                lo, hi = r[m]["ci"]
                if lo > hi:
                    print(f"FAILED: inverted interval at q={r['q']} for {m}")
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
