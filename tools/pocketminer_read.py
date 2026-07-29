#!/usr/bin/env python3
"""The eleventh read: the counting field against PocketMiner, under its plan.

Nothing is rescored. PocketMiner's per-residue probabilities were produced and
committed before ``PREREGISTERED_POCKETMINER.json`` was written, and that plan
pins them by hash; this file reads labels against numbers that already exist.

Every metric here is computed by a function some other read already uses --
``residue_auc_pr`` for the curves, ``metric_of`` for the confusion summaries,
``top_q_call`` for the budget rule -- because a second implementation of a
statistic is a way to lose a comparison without noticing. The one number this
file computes twice on purpose is our own mean per-unit ROC-AUC: once as the
frozen telemetry has it, and once through the same call PocketMiner's column goes
through. If those disagree the read stops, because then the two columns would be
different quantities wearing the same name.

PocketMiner does not score quite the same residues we do. Seven residues across
four chains have an incomplete backbone, which its featurisation cannot represent,
and one chain uses insertion codes that ``resseq`` keys cannot distinguish. Every
paired comparison is therefore taken on the residues both methods scored, and the
count that removes is reported rather than absorbed.

Usage: PYTHONPATH=src:tools python3.12 tools/pocketminer_read.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

import numpy as np

from matched_full_read import aligned, confusion, load_truth, metric_of, top_q_call
from plmnn_read import _boot, _interval, _quantiles  # noqa: F401
from pocket_bench.metrics import residue_auc_pr
from pocket_bench.paths import ROOT

PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_POCKETMINER.json"
SCORES = ROOT / "results/baselines/POCKETMINER_SCORES.json"
SELFTEST = ROOT / "results/baselines/POCKETMINER_SELFTEST.json"
TRAIN_OP = ROOT / "results/architecture_sweep/POCKETMINER_TRAIN_OPERATING_POINT.json"
SCORE_DIR = ROOT / "data/baselines/pocketminer"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
PREDS = ROOT / "results/cryptobench_official/predictions"
OUT = ROOT / "results/official_fold/POCKETMINER_READ.json"

SCHEMA = "geoaudit.pocketminer_read.v1"
READ_INDEX = 11
METHOD = "table_field"
BASELINE = "pocketminer"
TIE_ATOL = 1e-12
METRICS = ("precision", "recall", "positive_class_f1", "mcc")


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _plan() -> tuple[dict, dict]:
    rel = str(PLAN.relative_to(ROOT))
    if _git("rev-parse", "--is-shallow-repository") == "true":
        raise SystemExit(
            "this is a shallow clone, so the commit that fixed the plan may not "
            "be present and the ordering cannot be checked. Fetch the full "
            "history (actions/checkout with fetch-depth: 0) and retry.")
    if _git("status", "--porcelain", "--", rel):
        raise SystemExit(
            f"{rel} is modified or untracked. A plan that can still be edited is "
            f"not one, so this read is refused.")
    sha = _git("log", "-1", "--format=%H", "--", rel)
    if not sha:
        raise SystemExit(f"{rel} has no commit; this read is refused.")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha,
                       _git("rev-parse", "HEAD")], cwd=ROOT).returncode != 0:
        raise SystemExit(f"{sha[:12]} is not an ancestor of HEAD")
    plan = json.loads(PLAN.read_text())
    for path, key, where in (
            (SCORES, "scores_artifact_sha256", "the_baseline"),
            (SELFTEST, "selftest_artifact_sha256",
             "how_faithful_the_reproduction_is"),
            (TRAIN_OP, "train_operating_point_sha256", "numeric_parameters")):
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        want = plan[where][key]
        if got != want:
            raise SystemExit(
                f"{path.name} hashes to {got[:12]} but the plan pinned "
                f"{want[:12]}; it was regenerated after the plan was written")
    return plan, {"artifact": rel, "committed_in": sha,
                  "committed_at": _git("log", "-1", "--format=%cI", sha),
                  "subject": _git("log", "-1", "--format=%s", sha),
                  "is_ancestor_of_head": True}


def _ours() -> dict[str, dict]:
    return json.loads((PREDS / "table_field.json").read_text())["units"]


def _theirs() -> dict[str, dict[int, float]]:
    out = {}
    for f in sorted(SCORE_DIR.glob("*.json")):
        d = json.loads(f.read_text())
        out[f.stem] = {int(k): float(v) for k, v in d["residue_scores"].items()}
    return out


def _p2rank() -> dict[str, dict]:
    return json.loads((PREDS / "p2rank.json").read_text())["units"]


def _frozen_aucs() -> dict[str, dict[str, float]]:
    by: dict[str, dict[str, float]] = {}
    for r in json.loads(TELEMETRY.read_text())["rows"]:
        if r.get("residue_auc") is not None:
            by.setdefault(r["method"], {})[r["unit_id"]] = float(r["residue_auc"])
    return by


def _auc_on(keys: list[int], scores: dict[int, float], pos: set[int],
            called: list[int]) -> float | None:
    """One per-unit ROC-AUC, always through the harness's own function."""
    res = residue_auc_pr(
        [], sorted(pos), sorted(keys),
        {"residue_scores": {str(k): scores[k] for k in keys},
         "residue_positive": called})
    v = res.get("residue_auc")
    return None if v is None else float(v)


def _calibrate(frozen: dict[str, float]) -> dict:
    """Our own per-unit ROC-AUC, recomputed through the call the baseline gets."""
    truth = load_truth()
    worst, worst_unit, n = 0.0, None, 0
    for uid, u in _ours().items():
        if uid not in frozen:
            continue
        keys = sorted(int(k) for k in u["residue_scores"])
        got = _auc_on(keys, {int(k): float(v) for k, v in
                             u["residue_scores"].items()},
                      truth.get(uid) or set(),
                      [int(k) for k in (u.get("residue_positive") or [])])
        if got is None:
            continue
        n += 1
        gap = abs(got - frozen[uid])
        if gap > worst:
            worst, worst_unit = gap, uid
    if worst > 1e-6:
        raise SystemExit(
            f"recomputing the counting field's own per-unit ROC-AUC through the "
            f"call the baseline gets disagrees with the frozen telemetry by "
            f"{worst:.2e} on {worst_unit}")
    return {"n_units_recomputed": n,
            "largest_absolute_disagreement": float(f"{worst:.3e}"),
            "on_unit": worst_unit,
            "why": ("the baseline's ROC-AUC is produced by driving "
                    "residue_auc_pr from a prediction dictionary. Doing the same "
                    "for the counting field has to reproduce what the harness "
                    "froze, or the two columns are different quantities")}


def _shared() -> dict[str, dict]:
    """Per unit: the residues both methods scored, with labels and both scores.

    The intersection, not either method's own set, because a paired difference
    over two different residue universes is not paired.
    """
    truth, ours, theirs, p2 = load_truth(), _ours(), _theirs(), _p2rank()
    out, lost = {}, []
    for uid in sorted(set(ours) & set(theirs)):
        u = ours[uid]
        pos = truth.get(uid) or set()
        ours_scores = {int(k): float(v) for k, v in
                       (u.get("residue_scores") or {}).items()}
        keys = sorted(set(ours_scores) & set(theirs[uid]))
        dropped = len(ours_scores) - len(keys)
        if dropped:
            lost.append({"unit": uid, "n_residues_not_shared": dropped,
                         "n_ours": len(ours_scores),
                         "n_theirs": len(theirs[uid])})
        if not keys or not (pos & set(keys)):
            continue
        p2u = p2.get(uid) or {}
        p2_scores = {int(k): float(v) for k, v in
                     (p2u.get("residue_scores") or {}).items()}
        out[uid] = {
            "keys": keys,
            "y": np.array([k in pos for k in keys], dtype=bool),
            "pos": pos,
            "ours": ours_scores,
            "theirs": theirs[uid],
            "p2rank": p2_scores if set(keys) <= set(p2_scores) else None,
            "ours_called": [int(k) for k in (u.get("residue_positive") or [])],
            "p2rank_called": [int(k) for k in (p2u.get("residue_positive") or [])],
        }
    return {"units": out, "not_shared": lost}


def _paired(a: dict[str, float], b: dict[str, float], seed: int, n_boot: int,
            corrected: float) -> dict:
    shared = sorted(set(a) & set(b))
    d = np.array([a[u] - b[u] for u in shared])
    out = _interval(d, seed, n_boot, corrected)
    out["n_first_ahead"] = int((d > TIE_ATOL).sum())
    out["n_second_ahead"] = int((d < -TIE_ATOL).sum())
    out["n_tied"] = int((np.abs(d) <= TIE_ATOL).sum())
    out["level_first"] = round(float(np.mean([a[u] for u in shared])), 6)
    out["level_second"] = round(float(np.mean([b[u] for u in shared])), 6)
    return out


def _aucs(units: dict, keep: set[str] | None = None) -> dict[str, dict[str, float]]:
    """Per-unit ROC-AUC for all three methods, on the shared residues."""
    out: dict[str, dict[str, float]] = {"table_field": {}, "pocketminer": {},
                                        "p2rank": {}}
    for uid, u in units.items():
        if keep is not None and uid not in keep:
            continue
        v = _auc_on(u["keys"], u["ours"], u["pos"], u["ours_called"])
        if v is not None:
            out["table_field"][uid] = v
        v = _auc_on(u["keys"], u["theirs"], u["pos"],
                    [k for k, s in u["theirs"].items() if s >= 0.5])
        if v is not None:
            out["pocketminer"][uid] = v
        if u["p2rank"] is not None:
            v = _auc_on(u["keys"], u["p2rank"], u["pos"], u["p2rank_called"])
            if v is not None:
                out["p2rank"][uid] = v
    return out


def _thresholded(units: dict, q: float, cut_f1: float, cut_mcc: float,
                 budget_f1: float, seed: int, n_boot: int, corrected: float,
                 keep: set[str] | None = None) -> dict:
    """The three binarisation conventions the plan named, on shared residues."""
    arms = {"common_budget": {"ours": [], "theirs": []},
            "their_trained_budget": {"ours": [], "theirs": []},
            "their_trained_cut": {"ours": [], "theirs": []}}
    rates = []
    for uid, u in units.items():
        if keep is not None and uid not in keep:
            continue
        y = u["y"]
        s_o = np.array([u["ours"][k] for k in u["keys"]])
        s_t = np.array([u["theirs"][k] for k in u["keys"]])
        own = confusion(top_q_call(s_o, q), y)
        arms["common_budget"]["ours"].append(own)
        arms["common_budget"]["theirs"].append(confusion(top_q_call(s_t, q), y))
        arms["their_trained_budget"]["ours"].append(own)
        arms["their_trained_budget"]["theirs"].append(
            confusion(top_q_call(s_t, budget_f1), y))
        arms["their_trained_cut"]["ours"].append(own)
        cut = s_t >= cut_f1
        rates.append(float(cut.mean()))
        arms["their_trained_cut"]["theirs"].append(confusion(cut, y))
    out = {"q_ours": q, "their_trained_budget": budget_f1,
           "their_trained_cut": cut_f1,
           "their_trained_cut_for_mcc": cut_mcc,
           "n_units": len(arms["common_budget"]["ours"]),
           "mean_positive_rate_at_their_trained_cut":
               round(float(np.mean(rates)), 6) if rates else None,
           "note": ("the field's own rule is the same top-q budget in all three "
                    "arms, because that budget was fixed on the training fold and "
                    "is what it deploys; the arms differ only in how PocketMiner "
                    "is binarised")}
    for i, (arm, sides) in enumerate(arms.items()):
        block = {}
        for k, name in enumerate(METRICS):
            vals = {side: [metric_of(name, c) for c in cs]
                    for side, cs in sides.items()}
            block[name] = {
                side: round(float(np.mean([v for v in vs if v is not None])), 6)
                for side, vs in vals.items()}
            d = np.array([o - t for o, t in zip(vals["ours"], vals["theirs"])
                          if o is not None and t is not None])
            block[name]["paired_difference_ours_minus_theirs"] = _interval(
                d, seed + 100 * i + k, n_boot, corrected)
        out[arm] = block
    return out


def build() -> dict:
    plan, prov = _plan()
    p = plan["numeric_parameters"]
    q = float(p["top_q"])
    st = plan["statistic"]
    n_boot, seed, corrected = (int(st["n_boot"]), int(st["seed"]),
                               float(plan["multiplicity"]["corrected_level"]))
    frozen = _frozen_aucs()
    cal = _calibrate(frozen[METHOD])
    sh = _shared()
    units = sh["units"]
    aucs = _aucs(units)
    contaminated = set(plan["contamination"]["entries_in_pocketminers_own_data"])
    clean = {u for u in units if u.split("_")[0].lower() not in contaminated}
    removed = sorted(set(units) - clean)
    aucs_clean = _aucs(units, clean)

    d = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "is_official_mmseqs2_10pct_test_fold": True,
        "test_fold_read_index": READ_INDEX,
        "method": METHOD,
        "baseline": BASELINE,
        "status": plan["status_declared_in_advance"],
        "question": plan["question"],
        "rescored_anything": False,
        "why_it_is_indexed_anyway": (
            "the baseline's probabilities were committed before the plan, but "
            "reading labels against them is a new statement about the held-out "
            "fold and a ledger that counted only re-runs would make an unlimited "
            "number of such statements free"),
        "plan": prov,
        "reproduction_pinned_by_the_plan": {
            "published_roc_auc":
                plan["how_faithful_the_reproduction_is"]["published_roc_auc"],
            "ours_on_their_test_set":
                plan["how_faithful_the_reproduction_is"][
                    "our_roc_auc_on_their_test_set"],
            "residue_counts_match_exactly": True,
            "why_it_matters_here": (
                "whatever this read finds, it cannot be explained by a baseline "
                "that was rebuilt wrongly, and the evidence for that was frozen "
                "before the labels were touched"),
        },
        "calibration_of_our_own_column": cal,
        "residue_universe": {
            "rule": "the intersection, per unit, of the residues both methods "
                    "scored",
            "why": "a paired difference over two different residue sets is not "
                   "paired",
            "n_units_where_the_two_sets_differ": len(sh["not_shared"]),
            "units_where_they_differ": sh["not_shared"],
            "n_units_compared": len(units),
        },
        "n_units": len(units),
        "levels": {m: round(float(np.mean(list(v.values()))), 6)
                   for m, v in aucs.items() if v},
        "primary": {
            "functional": st["primary_functional"],
            "table_field_minus_pocketminer": _paired(
                aucs["table_field"], aucs["pocketminer"], seed, n_boot, corrected),
            "p2rank_minus_pocketminer": _paired(
                aucs["p2rank"], aucs["pocketminer"], seed + 1, n_boot, corrected),
        },
        "contamination_arm": {
            "entries_removed": sorted(contaminated),
            "units_removed": removed,
            "n_units_left": len(clean),
            "direction": "their presence can only favour PocketMiner",
            "table_field_minus_pocketminer": _paired(
                aucs_clean["table_field"], aucs_clean["pocketminer"],
                seed + 2, n_boot, corrected),
            "p2rank_minus_pocketminer": _paired(
                aucs_clean["p2rank"], aucs_clean["pocketminer"],
                seed + 3, n_boot, corrected),
        },
        "thresholded": _thresholded(
            units, q, float(p["pocketminer_trained_cut_f1"]),
            float(p["pocketminer_trained_cut_mcc"]),
            float(p["pocketminer_trained_budget_f1"]),
            seed, n_boot, corrected),
        "multiplicity": plan["multiplicity"],
        "decision_rules": plan["decision_rules"],
    }

    pr = d["primary"]["table_field_minus_pocketminer"]
    lvl = d["levels"]
    near_chance = lvl.get("pocketminer", 0.0) < 0.55
    if near_chance:
        key = "pocketminer_lands_near_chance"
    elif not pr["excludes_zero"]:
        key = "the_interval_crosses_zero"
    elif pr["mean"] > 0:
        key = "the_field_is_ahead"
    else:
        key = "pocketminer_is_ahead"
    d["outcome_key"] = key
    d["conclusion"] = plan["what_will_be_written_under_each_outcome"][key]
    d["outcome_was_named_in_the_plan"] = True
    return d


def _report(d: dict) -> None:
    print(f"read {d['test_fold_read_index']} ({d['status']}), "
          f"{d['n_units']} units")
    print("  mean per-unit ROC-AUC: " + "  ".join(
        f"{m} {v:.4f}" for m, v in d["levels"].items()))
    for name, blk in (("ours - pocketminer",
                       d["primary"]["table_field_minus_pocketminer"]),
                      ("p2rank - pocketminer",
                       d["primary"]["p2rank_minus_pocketminer"])):
        print(f"  {name:22s} {blk['mean']:+.4f}  95% [{blk['ci'][0]:+.4f}, "
              f"{blk['ci'][1]:+.4f}]  "
              f"{'excludes zero' if blk['excludes_zero'] else 'crosses zero'}  "
              f"win/loss/tie {blk['n_first_ahead']}/{blk['n_second_ahead']}/"
              f"{blk['n_tied']}")
    c = d["contamination_arm"]["table_field_minus_pocketminer"]
    print(f"  without its own {len(d['contamination_arm']['units_removed'])} "
          f"entries: {c['mean']:+.4f}  95% [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]")
    t = d["thresholded"]
    for arm in ("common_budget", "their_trained_budget", "their_trained_cut"):
        b = t[arm]["positive_class_f1"]
        i = b["paired_difference_ours_minus_theirs"]
        print(f"  F1 {arm:22s} ours {b['ours']:.4f} theirs {b['theirs']:.4f}  "
              f"delta {i['mean']:+.4f} 95% [{i['ci'][0]:+.4f}, {i['ci'][1]:+.4f}]")
    print(f"  outcome: {d['outcome_key']}")


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
    if d.get("status") != "exploratory":
        print(f"FAILED: status {d.get('status')}")
        return 1
    plan = json.loads(PLAN.read_text())
    if d["conclusion"] not in plan[
            "what_will_be_written_under_each_outcome"].values():
        print("FAILED: the reported conclusion is not one the plan wrote")
        return 1
    if d["calibration_of_our_own_column"]["largest_absolute_disagreement"] > 1e-6:
        print("FAILED: our own column does not reproduce the frozen telemetry")
        return 1
    for blk in (d["primary"]["table_field_minus_pocketminer"],
                d["primary"]["p2rank_minus_pocketminer"]):
        lo, hi = blk["ci"]
        blo, bhi = blk["ci_bonferroni"]
        if not (blo <= lo <= hi <= bhi):
            print("FAILED: the Bonferroni interval is not wider than the "
                  "uncorrected one")
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
