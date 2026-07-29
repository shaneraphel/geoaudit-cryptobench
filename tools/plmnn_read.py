#!/usr/bin/env python3
"""The tenth read: the counting field against CryptoBench's own pLM-NN baseline.

Runs the plan in results/architecture_sweep/PREREGISTERED_PLMNN.json and refuses
to start until that plan is committed, clean and an ancestor of HEAD, and until
the baseline's scores still hash to what the plan pinned.

Two things are borrowed rather than reimplemented, deliberately. The per-unit
ROC-AUC comes from the same ``residue_auc_pr`` that produced every other method's,
so the three columns of the comparison are the same quantity. The thresholded
metrics come from the functions the matched-operating-point read already uses, so
the top-9% budget means here exactly what it means there. A second implementation
of either would be a way for the comparison to be wrong without anything failing.

The read can void itself. If the reproduced baseline scores below the floor the
plan set, no comparison is reported at all: a win over a baseline broken in
reproduction would be worse than a loss.

Usage: PYTHONPATH=src:tools python3.12 tools/plmnn_read.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

import numpy as np

from matched_full_read import (aligned, confusion, load_truth, metric_of,
                               top_q_call)
from pocket_bench.metrics import residue_auc_pr
from pocket_bench.paths import ROOT

PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_PLMNN.json"
SCORES = ROOT / "results/baselines/PLMNN_SCORES.json"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
PREDS = ROOT / "results/cryptobench_official/predictions"
OUT = ROOT / "results/official_fold/PLMNN_READ.json"

SCHEMA = "geoaudit.plmnn_read.v1"
READ_INDEX = 10
METHOD = "table_field"
P2RANK = "p2rank"
# A tie in a per-residue ROC-AUC over the same universe is exact equality of two
# rank statistics; the tolerance absorbs float representation, nothing more.
TIE_ATOL = 1e-12


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
            f"{rel} is modified or untracked. A plan that can still be edited "
            f"is not one, so this read is refused.")
    sha = _git("log", "-1", "--format=%H", "--", rel)
    if not sha:
        raise SystemExit(
            f"{rel} has no commit. The comparison has to be in the history "
            f"before it is run; this read is refused.")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha,
                       _git("rev-parse", "HEAD")], cwd=ROOT).returncode != 0:
        raise SystemExit(f"{sha[:12]} is not an ancestor of HEAD")
    plan = json.loads(PLAN.read_text())
    got = hashlib.sha256(SCORES.read_bytes()).hexdigest()
    want = plan["the_baseline"]["scores_artifact_sha256"]
    if got != want:
        raise SystemExit(
            f"the baseline's scores hash to {got[:12]} but the plan pinned "
            f"{want[:12]}; the baseline was rerun after the plan was written, "
            f"so either restore it or write a new plan")
    return plan, {"artifact": rel, "committed_in": sha,
                  "committed_at": _git("log", "-1", "--format=%cI", sha),
                  "subject": _git("log", "-1", "--format=%s", sha),
                  "is_ancestor_of_head": True}


def _ours() -> dict[str, dict]:
    return json.loads((PREDS / "table_field.json").read_text())["units"]


def _plmnn_units() -> dict[str, dict[int, float]]:
    return {u["unit_id"]: {int(k): float(v) for k, v in u["scores"].items()}
            for u in json.loads(SCORES.read_text())["units"]}


def _plmnn_aucs(thr: float) -> dict[str, float]:
    """Per-unit ROC-AUC of the baseline, through the harness's own function.

    The baseline's own positive call is its published threshold, which is what
    ``residue_auc_pr`` means by a natively residue-level predictor stating its own
    operating point. It is the same treatment P2Rank gets.
    """
    truth = load_truth()
    out = {}
    for uid, scores in _plmnn_units().items():
        res = residue_auc_pr(
            [], sorted(truth.get(uid, ())), sorted(scores),
            {"residue_scores": {str(k): v for k, v in scores.items()},
             "residue_positive": [k for k, v in scores.items() if v > thr]})
        if res.get("residue_auc") is not None:
            out[uid] = float(res["residue_auc"])
    return out


def _calibrate(frozen: dict[str, float]) -> dict:
    """Our own per-unit ROC-AUC, recomputed here by the same call the baseline
    gets, and compared against the frozen telemetry.

    This is what makes the comparison apples to apples. If ``residue_auc_pr``
    driven from a prediction dictionary reproduces the numbers the harness froze
    for the counting field, then driving it the same way for pLM-NN measures the
    same quantity. If it does not, the baseline's column would be a different
    statistic wearing the same name.
    """
    truth = load_truth()
    worst, worst_unit, n = 0.0, None, 0
    for uid, u in _ours().items():
        if uid not in frozen:
            continue
        res = residue_auc_pr([], sorted(truth.get(uid, ())),
                             sorted(int(k) for k in u["residue_scores"]), u)
        if res.get("residue_auc") is None:
            continue
        n += 1
        gap = abs(float(res["residue_auc"]) - frozen[uid])
        if gap > worst:
            worst, worst_unit = gap, uid
    if worst > 1e-6:
        raise SystemExit(
            f"recomputing the counting field's own per-unit ROC-AUC through the "
            f"same call the baseline gets disagrees with the frozen telemetry by "
            f"{worst:.2e} on {worst_unit}; the baseline's column would not be "
            f"the same statistic as the others")
    return {"n_units_recomputed": n,
            "largest_absolute_disagreement": float(f"{worst:.3e}"),
            "on_unit": worst_unit,
            "why": ("the baseline's ROC-AUC is produced by driving "
                    "residue_auc_pr from a prediction dictionary. Doing the same "
                    "for the counting field has to reproduce what the harness "
                    "froze, or the two columns are different quantities")}


def _frozen_aucs() -> dict[str, dict[str, float]]:
    by: dict[str, dict[str, float]] = {}
    for r in json.loads(TELEMETRY.read_text())["rows"]:
        if r.get("residue_auc") is not None:
            by.setdefault(r["method"], {})[r["unit_id"]] = float(r["residue_auc"])
    return by


def _boot(d: np.ndarray, seed: int, n_boot: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)


def _quantiles(boot: np.ndarray, level: float) -> list[float]:
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [a, 1.0 - a])
    return [round(float(lo), 6), round(float(hi), 6)]


def _interval(d: np.ndarray, seed: int, n_boot: int, corrected: float) -> dict:
    """Both intervals from one resampling, so the only difference between them is
    the quantile read off it rather than a second draw."""
    if len(d) == 0:
        return {"n": 0, "mean": None, "median": None, "ci": [None, None],
                "ci_bonferroni": [None, None], "excludes_zero": None}
    boot = _boot(d, seed, n_boot)
    ci = _quantiles(boot, 0.95)
    return {"n": int(len(d)),
            "mean": round(float(d.mean()), 6),
            "median": round(float(np.median(d)), 6),
            "ci": ci,
            "ci_bonferroni": _quantiles(boot, 1.0 - corrected),
            "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)}


def _paired_auc(a: dict[str, float], b: dict[str, float], seed: int,
                n_boot: int, corrected: float) -> dict:
    shared = sorted(set(a) & set(b))
    d = np.array([a[u] - b[u] for u in shared])
    out = _interval(d, seed, n_boot, corrected)
    out["n_first_ahead"] = int((d > TIE_ATOL).sum())
    out["n_second_ahead"] = int((d < -TIE_ATOL).sum())
    out["n_tied"] = int((np.abs(d) <= TIE_ATOL).sum())
    out["level_first"] = round(float(np.mean([a[u] for u in shared])), 6)
    out["level_second"] = round(float(np.mean([b[u] for u in shared])), 6)
    return out


def _thresholded(q: float, thr: float, seed: int, n_boot: int,
                 corrected: float) -> dict:
    """F1, MCC, precision and recall under a common budget and under each
    method's own rule, on the units where both methods scored the same residues."""
    truth = load_truth()
    ours, plm = _ours(), _plmnn_units()
    arms = {"common_budget": {"ours": [], "plmnn": []},
            "each_methods_own_rule": {"ours": [], "plmnn": []}}
    rates = []
    for uid in sorted(set(ours) & set(plm)):
        pos = truth.get(uid) or set()
        al = aligned(ours[uid], pos)
        if al is None:
            continue
        s_ours, _, y = al
        keys = sorted(plm[uid])
        if keys != sorted(int(k) for k in (ours[uid]["residue_scores"] or {})):
            raise SystemExit(
                f"{uid}: the baseline and the field were scored on different "
                f"residues, so no paired metric on this unit would be paired")
        s_plm = np.array([plm[uid][k] for k in keys])
        if not np.array_equal(y, np.array([k in pos for k in keys], dtype=bool)):
            raise SystemExit(f"{uid}: the two arms disagree about the labels")

        budget_ours = confusion(top_q_call(s_ours, q), y)
        arms["common_budget"]["ours"].append(budget_ours)
        arms["common_budget"]["plmnn"].append(confusion(top_q_call(s_plm, q), y))
        # The field's own rule is that same budget: it was fixed on the training
        # fold and is what it deploys, so this arm differs only on the baseline.
        arms["each_methods_own_rule"]["ours"].append(budget_ours)
        native = s_plm > thr
        rates.append(float(native.mean()))
        arms["each_methods_own_rule"]["plmnn"].append(confusion(native, y))

    out = {"q": q, "threshold": thr,
           "n_units": len(arms["common_budget"]["ours"]),
           "mean_positive_rate_of_the_baseline_at_its_own_threshold":
               round(float(np.mean(rates)), 6),
           "note_on_the_second_arm": (
               "the field's own rule is the same top-q budget, so the two arms "
               "differ only in how the baseline is binarised")}
    for arm, sides in arms.items():
        block = {}
        for k, name in enumerate(("precision", "recall", "positive_class_f1",
                                  "mcc")):
            vals = {side: [metric_of(name, c) for c in cs]
                    for side, cs in sides.items()}
            block[name] = {
                side: round(float(np.mean([v for v in vs if v is not None])), 6)
                for side, vs in vals.items()}
            d = np.array([o - p for o, p in zip(vals["ours"], vals["plmnn"])
                          if o is not None and p is not None])
            block[name]["paired_difference_ours_minus_plmnn"] = _interval(
                d, seed + 1000 * (arm == "common_budget") + k, n_boot, corrected)
        out[arm] = block
    return out


def build() -> dict:
    plan, prov = _plan()
    p = plan["numeric_parameters"]
    n_boot = plan["statistic"]["n_boot"]
    seed = plan["statistic"]["seed"]
    corrected = plan["multiplicity"]["corrected_level"]

    frozen = _frozen_aucs()
    ours, p2 = frozen[METHOD], frozen[P2RANK]
    calibration = _calibrate(ours)
    plm = _plmnn_aucs(p["plmnn_threshold"])

    mean_plm = float(np.mean([plm[u] for u in sorted(plm)]))
    voided = mean_plm < p["baseline_auc_floor"]

    vs_ours = _paired_auc(ours, plm, seed, n_boot, corrected)
    vs_p2 = _paired_auc(p2, plm, seed + 1, n_boot, corrected)
    thresholded = (None if voided else
                   _thresholded(p["top_q"], p["plmnn_threshold"], seed, n_boot,
                                corrected))

    rate_ok = (voided or
               thresholded["mean_positive_rate_of_the_baseline_at_its_own_threshold"]
               < p["max_positive_rate_at_the_baseline_threshold"])
    if not rate_ok:
        raise SystemExit(
            "at its published threshold the baseline calls a majority of "
            "residues positive, which the plan set as evidence that the "
            "probabilities are not the ones the authors calibrated")

    if voided:
        outcome = "the_reproduction_fails_its_floor"
    elif not vs_ours["excludes_zero"]:
        outcome = "the_interval_crosses_zero"
    elif vs_ours["mean"] > 0:
        outcome = "the_field_is_ahead"
    else:
        outcome = "the_baseline_is_ahead"

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "test_fold_read_index": READ_INDEX,
        "status": plan["status_declared_in_advance"],
        "question": plan["question"],
        "plan": prov,
        "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "baseline_scores_sha256": hashlib.sha256(SCORES.read_bytes()).hexdigest(),

        "same_statistic_check": calibration,

        "reproduction_gate": {
            "baseline_mean_per_unit_roc_auc": round(mean_plm, 6),
            "floor": p["baseline_auc_floor"],
            "passes": not voided,
            "why_a_floor": (
                "a published supervised model cannot score near chance on the "
                "benchmark it was fitted for. Below the floor the reproduction "
                "is broken and this read reports nothing rather than reporting "
                "a win"),
        },

        "primary_comparison": {
            "what": "mean paired per-unit ROC-AUC, counting field minus our "
                    "reproduction of pLM-NN",
            **vs_ours,
        },
        "context_comparison": {
            "what": "the same functional, P2Rank minus our reproduction of "
                    "pLM-NN, which places the two baselines against each other",
            **vs_p2,
        },
        "thresholded_metrics": thresholded,

        "outcome": outcome,
        "sentence_fixed_in_advance":
            plan["what_will_be_written_under_each_outcome"][outcome],
        "multiplicity": plan["multiplicity"],
        "what_this_does_not_show": (
            "this is our reproduction of their baseline, agreeing with their "
            "one published example to a mean cosine of about 0.9987 rather than "
            "exactly, and it is exploratory because nine indexed reads preceded "
            "the plan. It is not a comparison against a number they reported"),
    }


def _report(d: dict) -> None:
    g = d["reproduction_gate"]
    c0 = d["same_statistic_check"]
    print(f"read {d['test_fold_read_index']} ({d['status']})")
    print(f"  our own AUC recomputed through the baseline's call agrees to "
          f"{c0['largest_absolute_disagreement']:.1e} over "
          f"{c0['n_units_recomputed']} units")
    print(f"  baseline mean per-unit ROC-AUC "
          f"{g['baseline_mean_per_unit_roc_auc']:.4f} "
          f"(floor {g['floor']}, {'passes' if g['passes'] else 'VOID'})")
    for key in ("primary_comparison", "context_comparison"):
        c = d[key]
        print(f"  {c['level_first']:.4f} against {c['level_second']:.4f}: "
              f"delta {c['mean']:+.4f} CI [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]"
              f" {'excludes zero' if c['excludes_zero'] else 'crosses zero'}"
              f"  win/loss/tie {c['n_first_ahead']}/{c['n_second_ahead']}"
              f"/{c['n_tied']}")
    t = d.get("thresholded_metrics")
    if t:
        print(f"  baseline positive rate at {t['threshold']}: "
              f"{t['mean_positive_rate_of_the_baseline_at_its_own_threshold']:.4f}")
        for arm in ("common_budget", "each_methods_own_rule"):
            for name in ("positive_class_f1", "mcc"):
                m = t[arm][name]
                pd = m["paired_difference_ours_minus_plmnn"]
                print(f"  {arm:22s} {name:17s} {m['ours']:.4f} against "
                      f"{m['plmnn']:.4f}, delta {pd['mean']:+.4f} "
                      f"CI [{pd['ci'][0]:+.4f}, {pd['ci'][1]:+.4f}]")
    print(f"  outcome: {d['outcome']}")


def check() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    have = json.loads(OUT.read_text())
    bad = []
    if have.get("schema") != SCHEMA:
        bad.append("unexpected schema")
    if have.get("test_fold_read_index") != READ_INDEX:
        bad.append(f"read index {have.get('test_fold_read_index')}")
    try:
        live = build()
    except SystemExit as e:
        print(f"FAIL {OUT.relative_to(ROOT)}: {e}")
        return 1
    for key in ("primary_comparison", "context_comparison", "reproduction_gate",
                "outcome", "thresholded_metrics", "same_statistic_check"):
        if have.get(key) != live.get(key):
            bad.append(f"{key} does not reproduce")
    fixed = json.loads(PLAN.read_text())[
        "what_will_be_written_under_each_outcome"]
    if have.get("sentence_fixed_in_advance") != fixed.get(have.get("outcome")):
        bad.append("the reported sentence is not the one the plan fixed for "
                   "this outcome")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if bad:
        return 1
    _report(have)
    print(f"\nOK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        return check()
    d = build()
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
