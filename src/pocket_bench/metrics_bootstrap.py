"""Paired bootstrap confidence intervals for residue-level pocket metrics.

Replaces bare "X/15 vs Y/15" ranking talk with proper uncertainty: a paired
bootstrap over structures gives each method a CI on its mean metric AND a CI on the
paired difference Δ = method − baseline (same resampled structures for both, so the
comparison is paired and the between-method correlation is preserved).

Metrics supported directly: ROC-AUC and average precision (PR-AUC) are reused from
``pocket_bench.metrics``; MCC and F1 are provided here from a confusion count or
from (scores, labels, threshold). `clinical_grade=false`.

Honesty boundaries (do not remove):
* The CryptoBench-apo pilot is n=15 and is a deterministic stride over the label
  set — it is NOT the official MMseqs2 10%-identity cluster-disjoint test fold.
  Running the official fold requires that fold's structure list + labels, which are
  not present in this repo; the loader here refuses to relabel a pilot as the
  official fold.
* Only baselines with data actually present (declared in ``BASELINE_ENV.json``) are
  compared. A baseline that is absent (e.g. PocketMiner, if not installed) is
  reported as unavailable, never imputed.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Sequence

from pocket_bench.metrics import average_precision, roc_auc


def mcc_from_counts(tp: int, fp: int, tn: int, fn: int) -> float | None:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return None
    return (tp * tn - fp * fn) / denom


def f1_from_counts(tp: int, fp: int, fn: int) -> float | None:
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else None


def _confusion(scores: Sequence[float], labels: Sequence[int], thr: float):
    tp = fp = tn = fn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= thr else 0
        if pred and y:
            tp += 1
        elif pred and not y:
            fp += 1
        elif not pred and y:
            fn += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def mcc(scores: Sequence[float], labels: Sequence[int], thr: float = 0.5) -> float | None:
    return mcc_from_counts(*_confusion(scores, labels, thr))


def f1(scores: Sequence[float], labels: Sequence[int], thr: float = 0.5) -> float | None:
    tp, fp, _tn, fn = _confusion(scores, labels, thr)
    return f1_from_counts(tp, fp, fn)


def _mean(vals: Sequence[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def _percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_xs[lo]
    frac = pos - lo
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def paired_bootstrap(
    values_by_method: dict[str, list[float | None]],
    *,
    baseline: str,
    n_boot: int = 10000,
    seed: int = 20260725,
    ci: float = 0.95,
) -> dict[str, Any]:
    """Paired bootstrap over structures (same resample indices for all methods).

    ``values_by_method[m][i]`` is method ``m``'s metric on structure ``i`` (or None
    if it could not be computed there). Returns per-method mean + CI and, for each
    non-baseline method, the paired Δ = method − baseline mean + CI + a two-sided
    bootstrap p-value (fraction of resamples whose Δ crosses 0, doubled).
    """
    methods = list(values_by_method)
    n = len(next(iter(values_by_method.values())))
    for m, vals in values_by_method.items():
        if len(vals) != n:
            raise ValueError(f"method {m} has {len(vals)} values, expected {n}")
    if baseline not in values_by_method:
        raise ValueError(f"baseline {baseline} not among methods {methods}")

    rng = random.Random(seed)
    boot_means: dict[str, list[float]] = {m: [] for m in methods}
    boot_delta: dict[str, list[float]] = {m: [] for m in methods if m != baseline}
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        means = {}
        for m in methods:
            mv = _mean([values_by_method[m][i] for i in idx])
            means[m] = mv
        base = means[baseline]
        for m in methods:
            if means[m] is not None:
                boot_means[m].append(means[m])
            if m != baseline and means[m] is not None and base is not None:
                boot_delta[m].append(means[m] - base)

    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    per_method = {}
    for m in methods:
        s = sorted(boot_means[m])
        per_method[m] = {
            "point": _mean(values_by_method[m]),
            "ci_low": _percentile(s, lo_q),
            "ci_high": _percentile(s, hi_q),
            "n_structures_scored": sum(
                1 for v in values_by_method[m] if v is not None
            ),
        }
    paired = {}
    for m, deltas in boot_delta.items():
        s = sorted(deltas)
        n_ge0 = sum(1 for d in deltas if d >= 0)
        frac = n_ge0 / len(deltas) if deltas else float("nan")
        p_two = 2 * min(frac, 1 - frac) if deltas else float("nan")
        paired[m] = {
            "delta_point": (per_method[m]["point"] - per_method[baseline]["point"])
            if per_method[m]["point"] is not None
            and per_method[baseline]["point"] is not None
            else None,
            "delta_ci_low": _percentile(s, lo_q),
            "delta_ci_high": _percentile(s, hi_q),
            "p_two_sided_bootstrap": p_two,
            "crosses_zero": (_percentile(s, lo_q) <= 0 <= _percentile(s, hi_q)),
        }
    return {
        "schema": "geoaudit.bootstrap_ci.v1",
        "clinical_grade": False,
        "baseline": baseline,
        "n_structures": n,
        "n_boot": n_boot,
        "seed": seed,
        "ci_level": ci,
        "per_method": per_method,
        "paired_vs_baseline": paired,
    }


def per_structure_values(
    rows: list[dict[str, Any]], metric_key: str
) -> dict[str, list[float | None]]:
    """Align a telemetry metric into {method: [value per structure]} by pdb order."""
    pdbs = sorted({r["pdb"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    index = {p: i for i, p in enumerate(pdbs)}
    out = {m: [None] * len(pdbs) for m in methods}
    for r in rows:
        out[r["method"]][index[r["pdb"]]] = r.get(metric_key)
    return out


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    telem = json.loads((root / "results/cryptobench_apo/TELEMETRY.json").read_text())
    rows = telem["rows"]
    baseline = "p2rank"  # the ML SOTA baseline present on this run
    report: dict[str, Any] = {
        "schema": "geoaudit.bootstrap_report.v1",
        "clinical_grade": False,
        "dataset": "cryptobench_apo_pilot_n15",
        "is_official_mmseqs2_10pct_test_fold": False,
        "official_fold_note": (
            "n=15 deterministic stride, NOT the official CryptoBench cluster-disjoint "
            "test fold; official-fold rerun requires that fold's structure list + "
            "labels (absent in this repo)."
        ),
        "metrics": {},
    }
    for metric_key in ("residue_auc", "residue_pr_auc"):
        vals = per_structure_values(rows, metric_key)
        report["metrics"][metric_key] = paired_bootstrap(vals, baseline=baseline)
    out = root / "results/cryptobench_apo/BOOTSTRAP_CI.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    for mk, res in report["metrics"].items():
        print(f"== {mk} (baseline={baseline}, n={res['n_structures']}, "
              f"boot={res['n_boot']}) ==")
        for m, s in res["per_method"].items():
            print(f"  {m:22s} {s['point']:.3f}  "
                  f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]")
        for m, d in res["paired_vs_baseline"].items():
            print(f"    Δ({m} - {baseline}) = {d['delta_point']:+.3f} "
                  f"[{d['delta_ci_low']:+.3f}, {d['delta_ci_high']:+.3f}]  "
                  f"p≈{d['p_two_sided_bootstrap']:.3f} "
                  f"{'(CI crosses 0)' if d['crosses_zero'] else '(CI excludes 0)'}")
    print("-> results/cryptobench_apo/BOOTSTRAP_CI.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
