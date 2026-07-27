#!/usr/bin/env python3.12
"""Recompute the headline comparison from the raw predictions, from scratch.

Why this exists
---------------
A reviewer asked to be able to recalculate the main result independently from
the download directory. Committing the per-residue predictions made that
*possible*; nothing in the repository made it *demonstrated*, and no gate would
have noticed if the frozen means stopped following from the raw scores.

The point of this script is that it shares no code with the harness. It imports
nothing from ``pocket_bench``, uses only the standard library, and re-derives
ROC-AUC, PR-AUC, MCC and F1 from their definitions rather than calling the
implementations under test. Agreement therefore says the frozen numbers are
right, not merely that the pipeline is deterministic.

What it reads, and nothing else:
  data/cryptobench_apo/official_labels/*.json      ground truth
  results/cryptobench_official/predictions/*.json  per-residue scores and calls

What it checks:
  1. every per-unit metric in TELEMETRY.json, for every method it can recompute
  2. the per-method means in the frozen bootstrap report
  3. the paired bootstrap confidence interval against P2Rank, resampled again

Five detectors return pockets rather than per-residue scores. Their residue
metrics are a function of the receptor residue universe, and the receptors are
the one input this repository does not commit -- 185 MB, re-downloadable, pinned
by SHA-256 in the manifest. Those five are reported as not-recomputable-here
rather than silently skipped, because the difference matters: it is a property
of what is committed, not of whether the numbers are correct.

  PYTHONDONTWRITEBYTECODE=1 python3.12 tools/recompute_from_raw.py
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "data/cryptobench_apo/official_labels"
PREDS = ROOT / "results/cryptobench_official/predictions"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
FROZEN = ROOT / "results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json"
OUT = ROOT / "results/official_fold/INDEPENDENT_RECOMPUTATION.json"

BASELINE = "p2rank"
HEADLINE = "table_field"
N_BOOT = 10000
SEED = 20260725
CI = 0.95
# Exact agreement is the expectation: same inputs, same arithmetic, IEEE doubles.
# The tolerance absorbs the last bit or two of summation-order difference, which
# is real because this script sums in a different order than the harness does.
TOL = 1e-9


# --- metrics, written from the definitions ---------------------------------

def roc_auc(pairs: list[tuple[float, int]]) -> float | None:
    """Area under the ROC curve as the Mann-Whitney statistic, ties at mid-rank.

    U counts, over all positive/negative pairs, the fraction where the positive
    scores higher, crediting a half to each tie; AUC is U normalised by the pair
    count. Ranking with mid-ranks is what makes the tie handling come out right,
    and it matters here because a detector that emits integer scores produces
    very large tie blocks.
    """
    n_pos = sum(y for _, y in pairs)
    n_neg = len(pairs) - n_pos
    if not n_pos or not n_neg:
        return None
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and pairs[order[j + 1]][0] == pairs[order[i]][0]:
            j += 1
        mid = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if pairs[order[k]][1] == 1:
                rank_sum_pos += mid
        i = j + 1
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def pr_auc(pairs: list[tuple[float, int]]) -> float | None:
    """Average precision: precision at each positive, weighted by recall gained.

    This is the step-wise estimator, not the trapezoid, and not interpolated
    precision. On a fold where positives are about 8 per cent of residues the
    three differ by more than the effect being measured, so the choice is part
    of the claim and is stated here rather than inherited.
    """
    n_pos = sum(y for _, y in pairs)
    if not n_pos:
        return None
    order = sorted(range(len(pairs)), key=lambda i: -pairs[i][0])
    tp = seen = 0
    ap = 0.0
    prev_recall = 0.0
    for idx in order:
        seen += 1
        if pairs[idx][1] == 1:
            tp += 1
        recall = tp / n_pos
        ap += (tp / seen) * (recall - prev_recall)
        prev_recall = recall
    return ap


def confusion(pairs: list[tuple[float, int]], called: list[int]) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for (_, y), c in zip(pairs, called):
        if c and y:
            tp += 1
        elif c and not y:
            fp += 1
        elif not c and y:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def mcc_of(tp: int, fp: int, tn: int, fn: int) -> float | None:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / denom) if denom else None


def f1_of(tp: int, fp: int, tn: int, fn: int) -> float | None:
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else None


# --- inputs -----------------------------------------------------------------

def _resnum(x) -> int | None:
    """Residue number from an int, or from a string ending in one.

    Reading the trailing run of digits with its sign, rather than every digit in
    the string, is what makes 'A:ALA123' give 123 and '-1' give -1. Both matter:
    the first is the identifier form the label files use, and the second is an
    expression-tag residue, which eleven of the official structures have.
    """
    if isinstance(x, int):
        return x
    s = str(x)
    digits = ""
    negative = False
    for ch in reversed(s):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            negative = ch == "-"
            break
    if not digits:
        return None
    return -int(digits) if negative else int(digits)


def load_truth() -> dict[str, set[int]]:
    truth: dict[str, set[int]] = {}
    for f in sorted(LABELS.glob("*_labels.json")):
        d = json.loads(f.read_text())
        unit = f"{d['pdb_id']}_{d['chain']}"
        res = d.get("cryptic_residues") or d.get("binding_residues") or []
        truth[unit] = {r for r in (_resnum(r) for r in res) if r is not None}
    return truth


def unit_metrics(pred: dict, positives: set[int]) -> dict[str, float | None] | None:
    """The four metrics for one structure, or None if this unit is not scorable."""
    scores = pred.get("residue_scores")
    if not scores:
        return None
    by_res: dict[int, float] = {}
    for k, v in scores.items():
        rn = _resnum(k)
        if rn is not None:
            by_res[rn] = float(v)
    if not by_res or not (positives & set(by_res)):
        return None
    keys = sorted(by_res)
    pairs = [(by_res[k], 1 if k in positives else 0) for k in keys]

    # The operating point is the detector's own positive call. No threshold is
    # chosen here; inventing one would compare the detectors at a point none of
    # them was built for, and would make MCC and F1 a property of this script.
    native = pred.get("residue_positive")
    called_set = {r for r in (_resnum(r) for r in (native or [])) if r is not None}
    called = [1 if k in called_set else 0 for k in keys]

    tp, fp, tn, fn = confusion(pairs, called)
    return {
        "residue_auc": roc_auc(pairs),
        "residue_pr_auc": pr_auc(pairs),
        "residue_mcc": mcc_of(tp, fp, tn, fn),
        "residue_f1": f1_of(tp, fp, tn, fn),
    }


# --- aggregation and the paired bootstrap ------------------------------------

def mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def paired_ci(a: list[float], b: list[float]) -> tuple[float, float, float]:
    """Bootstrap the paired difference a - b over structures.

    Resampling is over structures, with both arms taking the same draw, because
    the two detectors are evaluated on the same proteins; an unpaired interval
    would carry the between-protein variance that the pairing removes and would
    be roughly twice as wide for no reason.
    """
    n = len(a)
    rng = random.Random(SEED)
    deltas = []
    for _ in range(N_BOOT):
        pick = [rng.randrange(n) for _ in range(n)]
        deltas.append(sum(a[i] for i in pick) / n - sum(b[i] for i in pick) / n)
    deltas.sort()
    lo = deltas[int((1 - CI) / 2 * N_BOOT)]
    hi = deltas[min(N_BOOT - 1, int((1 + CI) / 2 * N_BOOT))]
    return (sum(a) / n - sum(b) / n), lo, hi


# --- the check ----------------------------------------------------------------

def close(x, y) -> bool:
    if x is None and y is None:
        return True
    if x is None or y is None:
        return False
    return abs(float(x) - float(y)) <= TOL


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="record the recomputation as an artifact")
    args = ap.parse_args(argv)

    truth = load_truth()
    telemetry = json.loads(TELEMETRY.read_text())["rows"]
    frozen = json.loads(FROZEN.read_text())

    tel: dict[tuple[str, str], dict] = {}
    for row in telemetry:
        tel[(row["method"], row["unit_id"])] = row

    recomputed: dict[str, dict[str, dict]] = {}
    not_recomputable: list[str] = []
    problems: list[str] = []

    for f in sorted(PREDS.glob("*.json")):
        if f.name == "INDEX.json":
            continue
        doc = json.loads(f.read_text())
        method = doc["method"]
        per_unit: dict[str, dict] = {}
        for unit, pred in doc["units"].items():
            got = unit_metrics(pred, truth.get(unit, set()))
            if got is not None:
                per_unit[unit] = got
        if not per_unit:
            not_recomputable.append(method)
            continue
        recomputed[method] = per_unit

        for unit, got in per_unit.items():
            row = tel.get((method, unit))
            if row is None:
                problems.append(f"{method}/{unit}: no telemetry row to compare against")
                continue
            for key, val in got.items():
                if not close(val, row.get(key)):
                    problems.append(
                        f"{method}/{unit}/{key}: recomputed {val!r} "
                        f"but telemetry says {row.get(key)!r}")

    # The published per-method means, recomputed.
    means: dict[str, dict[str, float | None]] = {}
    for method, per_unit in recomputed.items():
        means[method] = {}
        for key in ("residue_auc", "residue_pr_auc", "residue_mcc", "residue_f1"):
            vals = [u[key] for u in per_unit.values() if u[key] is not None]
            means[method][key] = mean(vals)
            pub = ((frozen["metrics"].get(key) or {}).get("per_method") or {}).get(method)
            if pub and not close(means[method][key], pub.get("point")):
                problems.append(
                    f"{method}/{key}: recomputed mean {means[method][key]!r} "
                    f"but the frozen report publishes {pub.get('point')!r}")

    # The headline interval, resampled.
    headline: dict[str, dict] = {}
    if HEADLINE in recomputed and BASELINE in recomputed:
        for key in ("residue_auc", "residue_pr_auc", "residue_mcc", "residue_f1"):
            shared = sorted(
                u for u in recomputed[HEADLINE]
                if u in recomputed[BASELINE]
                and recomputed[HEADLINE][u][key] is not None
                and recomputed[BASELINE][u][key] is not None)
            a = [recomputed[HEADLINE][u][key] for u in shared]
            b = [recomputed[BASELINE][u][key] for u in shared]
            point, lo, hi = paired_ci(a, b)
            headline[key] = {
                "n_paired_structures": len(shared),
                "difference": point,
                "ci_low": lo,
                "ci_high": hi,
                "excludes_zero": (lo > 0) or (hi < 0),
            }
            pub = ((frozen["metrics"].get(key) or {}).get("paired_vs_baseline") or {}
                   ).get(HEADLINE)
            if pub:
                for ours, theirs in (("difference", "difference"),
                                     ("ci_low", "ci_low"), ("ci_high", "ci_high")):
                    if theirs in pub and not close(headline[key][ours], pub[theirs]):
                        problems.append(
                            f"paired {key}/{ours}: recomputed {headline[key][ours]!r} "
                            f"but the frozen report publishes {pub[theirs]!r}")
    else:
        problems.append(
            f"cannot recompute the headline: {HEADLINE} or {BASELINE} has no "
            "per-residue predictions committed")

    n_checked = sum(len(v) for v in recomputed.values()) * 4
    print(f"recomputed {n_checked} per-unit metric values across "
          f"{len(recomputed)} detectors, from labels and raw scores only")
    for m in sorted(recomputed):
        mm = means[m]
        print(f"  {m:26} auc {mm['residue_auc']:.4f}  pr {mm['residue_pr_auc']:.4f}  "
              f"mcc {mm['residue_mcc']:.4f}  f1 {mm['residue_f1']:.4f}")
    if not_recomputable:
        print(f"\nnot recomputable from the committed files ({len(not_recomputable)} "
              "pocket-only detectors; their residue metrics need the receptor "
              "universe, and receptors are fetched, not committed):")
        for m in sorted(not_recomputable):
            print(f"  {m}")
    if headline:
        print(f"\n{HEADLINE} minus {BASELINE}, paired bootstrap resampled here "
              f"({N_BOOT} draws, seed {SEED}):")
        for key, h in headline.items():
            star = "excludes 0" if h["excludes_zero"] else "includes 0"
            print(f"  {key:16} {h['difference']:+.4f}  "
                  f"[{h['ci_low']:+.4f}, {h['ci_high']:+.4f}]  {star}  "
                  f"n={h['n_paired_structures']}")

    if problems:
        print(f"\nRECOMPUTATION DISAGREES WITH THE FROZEN ARTIFACTS "
              f"({len(problems)} discrepancies):")
        for p in problems[:25]:
            print(f"  - {p}")
        if len(problems) > 25:
            print(f"  ... and {len(problems) - 25} more")
        return 1
    print("\nevery recomputed value agrees with the frozen artifacts")

    if args.write:
        OUT.write_text(json.dumps({
            "schema": "geoaudit.independent_recomputation.v1",
            "clinical_grade": False,
            "purpose": "the frozen headline, rederived from raw predictions and "
                       "labels by code that shares nothing with the harness",
            "recomputed_from": [
                "data/cryptobench_apo/official_labels/*.json",
                "results/cryptobench_official/predictions/*.json",
            ],
            "checked_against": [
                "results/cryptobench_official/TELEMETRY.json",
                "results/official_fold/"
                "OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json",
            ],
            "tolerance": TOL,
            "n_per_unit_values_checked": n_checked,
            "detectors_recomputed": sorted(recomputed),
            "detectors_not_recomputable_from_the_clone": sorted(not_recomputable),
            "why_not_recomputable": "pocket-only detectors score a residue by "
                                    "whether it falls in a returned pocket, which "
                                    "needs the receptor residue universe; "
                                    "receptors are fetched from RCSB and OSF "
                                    "against committed SHA-256 pins, not committed",
            "means": means,
            "headline_paired_vs_p2rank": headline,
            "agrees": True,
        }, indent=2, allow_nan=False) + "\n")
        print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
