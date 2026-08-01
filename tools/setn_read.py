#!/usr/bin/env python3
"""Read Set N once, under the plan, and take whatever comes back.

The question is whether a method can tell a chain that has a cryptic pocket from a
chain examined against every deposited holo partner and shown to have none. No
published comparison on CryptoBench measures it, because every unit in every
published evaluation has a pocket.

Nothing here chooses anything. The unit score, ``k``, the statistic, the seed, the
number of resamples, the correction and the sentence to write under each outcome all
come out of ``PREREGISTERED_SETN.json``, which the commit graph shows was fixed
before any Set N score existed. This tool checks that order rather than asserting
it, and refuses to run if the plan is editable or either set has moved.

The bootstrap is stratified, not paired
----------------------------------------
The statistic is one ROC-AUC over all units, not a per-unit value, so the paired
bootstrap the Set A read uses does not apply: there is nothing to pair. Positives
and negatives are resampled with replacement within their own halves and both
methods' AUCs are recomputed on the same resample, so the difference distribution
carries the correlation between the two methods that a naive independent bootstrap
would throw away.

Usage: PYTHONPATH=src:tools python3.12 tools/setn_read.py [--check] [--rerun REASON]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess

import numpy as np

from pocket_bench.metrics import roc_auc
from pocket_bench.paths import ROOT

PLAN = ROOT / "results/external/PREREGISTERED_SETN.json"
SETN = ROOT / "results/external/SETN_SET.json"
SETA = ROOT / "results/external/EXTERNAL_SET.json"
N_PREDS = ROOT / "results/external/setn_predictions"
A_PREDS = ROOT / "results/external/predictions"
PLMNN_A = ROOT / "results/baselines/PLMNN_EXTERNAL_SCORES.json"
PLMNN_N = ROOT / "results/baselines/PLMNN_SETN_SCORES.json"
PM_A = ROOT / "data/baselines/pocketminer_external"
# Per-residue scores live one file per unit; the aggregate JSON is summaries
# only (same shape as Set B/C). Loading the aggregate would look present and
# score nothing — setbc_read already recorded that failure mode.
PM_N = ROOT / "data/baselines/pocketminer_setn"
OUT = ROOT / "results/external/SETN_READ.json"
SCHEMA = "geoaudit.setn_read.v1"

PRIMARY = ("table_field", "p2rank", "plmnn", "pocketminer")


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _plan() -> tuple[dict, dict]:
    rel = str(PLAN.relative_to(ROOT))
    if _git("rev-parse", "--is-shallow-repository") == "true":
        raise SystemExit("shallow clone: the ordering cannot be checked")
    if _git("status", "--porcelain", "--", rel):
        raise SystemExit(f"{rel} is modified or untracked. A plan that can still "
                         f"be edited is not one, so this read is refused.")
    sha = _git("log", "-1", "--format=%H", "--", rel)
    if not sha:
        raise SystemExit(f"{rel} has no commit; this read is refused.")
    plan = json.loads(PLAN.read_text())
    for path, want, what in (
            (SETN, plan["the_sets"]["negative"]["sha256"], "Set N"),
            (SETA, plan["the_sets"]["positive"]["sha256"], "Set A")):
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(
                f"{what} hashes to {got[:12]} but the plan pinned {want[:12]}; it "
                f"moved after the plan was written")
    return plan, {"artifact": rel, "committed_in": sha,
                  "committed_at": _git("log", "-1", "--format=%cI", sha),
                  "subject": _git("log", "-1", "--format=%s", sha)}


def _archive(path):
    """Per-unit residue scores from any of the shapes the baselines write."""
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    units = raw.get("units") or raw.get("scores") or raw
    if isinstance(units, list):
        # pLM-NN rows use unit_id; PocketMiner summaries use unit and carry no
        # per-residue scores (those are beside the run). Refuse the latter here
        # so a summary archive cannot look scored.
        if units and "unit_id" not in units[0] and "scores" not in units[0]:
            return None
        units = {u["unit_id"]: u for u in units}
    out = {}
    for uid, u in units.items():
        s = u.get("residue_scores") if isinstance(u, dict) else None
        if s is None and isinstance(u, dict):
            s = u.get("scores")
        if s:
            out[uid] = {int(k): float(v) for k, v in s.items()}
    return out or None


def _pocketminer_dir(d):
    if not d.exists():
        return None
    out = {}
    for f in sorted(d.glob("*.json")):
        doc = json.loads(f.read_text())
        out[f.stem] = {int(k): float(v)
                       for k, v in (doc.get("residue_scores") or {}).items()}
    return out


def _scores() -> tuple[dict, dict, list[str]]:
    """Per method, per unit, per residue --- both halves, and what is missing."""
    halves = {
        "positive": {
            "table_field": _archive(A_PREDS / "table_field.json"),
            "p2rank": _archive(A_PREDS / "p2rank.json"),
            "plmnn": _archive(PLMNN_A),
            "pocketminer": _pocketminer_dir(PM_A),
        },
        "negative": {
            "table_field": _archive(N_PREDS / "table_field.json"),
            "geometry_field": _archive(N_PREDS / "geometry_field.json"),
            "p2rank": _archive(N_PREDS / "p2rank.json"),
            "plmnn": _archive(PLMNN_N),
            "pocketminer": _pocketminer_dir(PM_N),
        },
    }
    absent = [f"{half}/{m}" for half, d in halves.items()
              for m, v in d.items() if not v]
    return halves["positive"], halves["negative"], absent


def _unit_scores(pos: dict, neg: dict, methods: tuple[str, ...], k: int
                 ) -> tuple[dict, np.ndarray, list[str], dict]:
    """One score per unit per method, over residues every method scored.

    The intersection, not any one method's own set, for the reason the Set A read
    gives: the four methods disagree about which residues exist, and a comparison
    over different universes is not a comparison.
    """
    rows, labels, ids = {m: [] for m in methods}, [], []
    lengths, dropped = [], []
    for half, src, y in (("positive", pos, 1), ("negative", neg, 0)):
        for uid in sorted(src[methods[0]]):
            per = [src[m].get(uid) for m in methods]
            if any(p is None for p in per):
                dropped.append({"unit": uid, "half": half,
                                "why": "not scored by every method in the primary"})
                continue
            keys = sorted(set.intersection(*(set(p) for p in per)))
            if len(keys) < k:
                dropped.append({"unit": uid, "half": half,
                                "why": f"fewer than {k} residues shared by all "
                                       f"methods ({len(keys)})"})
                continue
            for m, p in zip(methods, per):
                v = np.sort(np.array([p[r] for r in keys]))[-k:]
                rows[m].append(float(v.mean()))
            labels.append(y)
            ids.append(uid)
            lengths.append(len(keys))
    return ({m: np.array(v) for m, v in rows.items()}, np.array(labels), ids,
            {"n_dropped": len(dropped), "dropped": dropped[:40],
             "lengths": np.array(lengths)})


def _stratified_difference(a: np.ndarray, b: np.ndarray, y: np.ndarray,
                           seed: int, n_boot: int, level: float) -> dict:
    pos_i = np.flatnonzero(y == 1)
    neg_i = np.flatnonzero(y == 0)
    obs = roc_auc(a, y) - roc_auc(b, y)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for t in range(n_boot):
        idx = np.concatenate([rng.choice(pos_i, pos_i.size, replace=True),
                              rng.choice(neg_i, neg_i.size, replace=True)])
        yy = y[idx]
        draws[t] = roc_auc(a[idx], yy) - roc_auc(b[idx], yy)
    alpha = 1.0 - level
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"delta": round(float(obs), 6),
            "ci": [round(float(lo), 6), round(float(hi), 6)],
            "level": round(level, 6),
            "excludes_zero": bool(lo > 0 or hi < 0)}


def read(plan: dict, provenance: dict) -> dict:
    pos, neg, absent = _scores()
    k = plan["statistic"]["top_k"]
    seed = plan["co_primary"]["seed"]
    n_boot = plan["co_primary"]["n_bootstrap"]
    level = plan["co_primary"]["bonferroni_level"]

    methods = tuple(m for m in PRIMARY
                    if pos.get(m) and neg.get(m))
    unmeasured = [m for m in PRIMARY if m not in methods]
    if "table_field" not in methods:
        raise SystemExit("the counting field has no scores on one half")

    rows, y, ids, aux = _unit_scores(pos, neg, methods, k)
    aucs = {m: round(float(roc_auc(rows[m], y)), 6) for m in methods}
    control = round(float(roc_auc(aux["lengths"].astype(float), y)), 6)

    co = {}
    for m in methods:
        if m == "table_field":
            continue
        co[f"table_field_minus_{m}"] = _stratified_difference(
            rows["table_field"], rows[m], y, seed + hash(m) % 1000, n_boot, level)

    resolved = {n: v for n, v in co.items() if v["excludes_zero"]}
    ours_ahead = {n: v for n, v in resolved.items() if v["delta"] > 0}
    if control >= max(aucs.values()):
        headline = "control_dominates"
    elif not resolved:
        headline = "none_resolved"
    elif len(ours_ahead) == len(co) and len(co) == 3:
        headline = "all_three_resolved_in_our_favour"
    elif ours_ahead:
        headline = "some_resolved"
    else:
        headline = "resolved_against_us"

    secondary = {}
    if neg.get("geometry_field"):
        both = tuple(m for m in ("table_field", "geometry_field") if neg.get(m))
        gr, gy, _gi, _ga = _unit_scores({m: pos.get(m, {}) for m in both},
                                        {m: neg[m] for m in both}, both, k)
        # Set N only: the mean top-k unit score of each, paired by unit. Lower is
        # the quieter method on chains with nothing to find; it is descriptive
        # because the two fields' scores are not on a common scale.
        mask = gy == 0
        secondary["geometry_field_on_set_n_only"] = {
            "n_units": int(mask.sum()),
            "mean_top_k_table_field": round(float(gr["table_field"][mask].mean()), 6),
            "mean_top_k_geometry_field": round(
                float(gr["geometry_field"][mask].mean()), 6),
            "reads": "Set N only",
            "level": "descriptive; the two fields' scores are not on a common scale",
        }

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "declared": plan["declared"],
        "question": plan["question"],
        "plan": provenance,
        "plan_sha256": hashlib.sha256(PLAN.read_bytes()).hexdigest(),
        "n_units_compared": int(len(y)),
        "n_positive": int((y == 1).sum()),
        "n_negative": int((y == 0).sum()),
        "methods_in_the_primary": list(methods),
        "methods_not_measured": unmeasured,
        "why_not_measured": {
            m: "no score archive exists for one or both halves; absent is reported "
               "as absent rather than defaulted, so it cannot be mistaken for a "
               "measurement" for m in unmeasured},
        "archives_absent": absent,
        "unit_level_auc": aucs,
        "control_chain_length_alone": control,
        "control_reading": (
            "every method's AUC is to be read against this number as well as "
            "against 0.5"),
        "co_primary": co,
        "secondary": secondary,
        "units_dropped": aux["n_dropped"],
        "why_units_were_dropped": aux["dropped"],
        "outcome": headline,
        "conclusion": plan["outcome_sentences_written_in_advance"][headline],
        "what_this_cannot_show": plan["what_this_cannot_show"],
    }


def report(d: dict) -> None:
    print(f"\nSet N read, {d['declared']}")
    print(f"  {d['n_units_compared']} units: {d['n_positive']} with a cryptic "
          f"pocket, {d['n_negative']} shown to have none")
    print(f"  unit-level ROC-AUC (higher separates the halves better):")
    for m, v in sorted(d["unit_level_auc"].items(), key=lambda x: -x[1]):
        print(f"    {m:16s} {v:.4f}")
    print(f"    {'chain length alone':16s} {d['control_chain_length_alone']:.4f}"
          f"   <- control fixed before the read")
    if d["methods_not_measured"]:
        print(f"  not measured: {', '.join(d['methods_not_measured'])}")
    for name, v in d["co_primary"].items():
        mark = "resolved" if v["excludes_zero"] else "crosses zero"
        print(f"  {name}: {v['delta']:+.4f} "
              f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]  {mark}")
    for name, v in (d.get("secondary") or {}).items():
        print(f"  secondary {name}: " + ", ".join(
            f"{k}={x}" for k, x in v.items() if isinstance(x, (int, float))))
    print(f"\n  outcome: {d['outcome']}")
    print(f"  {d['conclusion']}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--rerun", metavar="REASON")
    a = ap.parse_args()
    if a.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        report(json.loads(OUT.read_text()))
        print(f"OK {OUT.relative_to(ROOT)}")
        return 0
    if OUT.exists() and not a.rerun:
        raise SystemExit(
            f"{OUT.relative_to(ROOT)} exists. This read runs once. Re-running "
            f"requires --rerun with a reason, which is recorded in the artifact.")
    plan, prov = _plan()
    d = read(plan, prov)
    if a.rerun:
        d["reread"] = {"reason": a.rerun,
                       "why_this_is_recorded": (
                           "a read that can be repeated silently is not a read")}
    OUT.write_text(json.dumps(d, indent=1) + "\n")
    report(d)
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
