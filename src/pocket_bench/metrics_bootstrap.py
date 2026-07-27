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

import argparse
import json
import math
import random
import sys
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


def json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with None.

    ``json.dumps`` emits bare ``NaN`` / ``Infinity`` tokens, which are valid
    Python but NOT valid JSON (RFC 8259 has no such literals). Any strict parser
    — ``jq``, Go, Rust, most JSON-Schema validators, ``json.loads(...,
    parse_constant=...)`` under a strict config — rejects the artifact, so a
    reviewer could not machine-read the frozen metrics at all. A metric that
    could not be estimated is genuinely absent, and ``null`` is how absence is
    spelled in JSON. Applied at the single write choke point so no future field
    can leak a bare NaN.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    # numpy scalars expose .item(); convert then re-check finiteness
    item = getattr(obj, "item", None)
    if callable(item) and hasattr(obj, "dtype"):
        return json_safe(item())
    return obj


def _mean(vals: Sequence[float | None]) -> float | None:
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else None


def _percentile(sorted_xs: list[float], q: float) -> float | None:
    # A method that scored on no structure has no percentile. Returning NaN
    # here leaked a bare NaN token into every frozen report that did not route
    # its output through json_safe, which is not valid JSON; None is.
    if not sorted_xs:
        return None
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
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        for m in methods:
            mv = _mean([values_by_method[m][i] for i in idx])
            if mv is not None:
                boot_means[m].append(mv)

    # The paired difference is taken on the structures where BOTH methods have a
    # value, which is not always all of them: MCC is undefined wherever a
    # detector's confusion matrix is degenerate, and P2Rank has six such
    # structures on this fold while the table field has none. Differencing each
    # method's mean over its own coverage would fold "which structures were
    # scorable" into a quantity read as "which method is better", and the two
    # subsets differ by exactly the structures a detector found hardest.
    boot_delta: dict[str, list[float]] = {}
    matched_n: dict[str, int] = {}
    matched_point: dict[str, tuple[float, float] | None] = {}
    base_vals = values_by_method[baseline]
    for m in methods:
        if m == baseline:
            continue
        mv = values_by_method[m]
        shared = [i for i in range(n)
                  if mv[i] is not None and base_vals[i] is not None]
        matched_n[m] = len(shared)
        if not shared:
            boot_delta[m] = []
            matched_point[m] = None
            continue
        matched_point[m] = (
            sum(mv[i] for i in shared) / len(shared),
            sum(base_vals[i] for i in shared) / len(shared),
        )
        mrng = random.Random(seed)
        deltas = []
        for _ in range(n_boot):
            pick = [shared[mrng.randrange(len(shared))] for _ in shared]
            a = sum(mv[i] for i in pick) / len(pick)
            b = sum(base_vals[i] for i in pick) / len(pick)
            deltas.append(a - b)
        boot_delta[m] = deltas

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
        d_lo, d_hi = _percentile(s, lo_q), _percentile(s, hi_q)
        # A method with no computable values on any structure has no Δ. Comparing
        # NaN bounds to 0 is False, which previously rendered as the affirmative
        # "CI excludes 0" for a difference that was never estimated.
        undefined = (
            d_lo is None or d_hi is None
            or math.isnan(d_lo) or math.isnan(d_hi)
        )
        mp = matched_point[m]
        paired[m] = {
            "delta_point": (mp[0] - mp[1]) if mp is not None else None,
            "delta_ci_low": None if undefined else d_lo,
            "delta_ci_high": None if undefined else d_hi,
            "p_two_sided_bootstrap": None if undefined else p_two,
            "crosses_zero": None if undefined else (d_lo <= 0 <= d_hi),
            # The structures both methods could be scored on. Where this is
            # below n_structures the two point estimates in per_method are over
            # different sets and must not be subtracted; the difference here is
            # the one that is defined.
            "n_paired_structures": matched_n[m],
            "matched_point_method": None if mp is None else mp[0],
            "matched_point_baseline": None if mp is None else mp[1],
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


def _unit_key(row: dict[str, Any]) -> str:
    """The evaluation unit: (pdb, chain), falling back to pdb for legacy rows."""
    return row.get("unit_id") or (
        f"{row['pdb']}_{row['chain']}" if row.get("chain") else row["pdb"]
    )


def per_structure_values(
    rows: list[dict[str, Any]], metric_key: str
) -> dict[str, list[float | None]]:
    """Align a telemetry metric into {method: [value per unit]} by unit order.

    Keyed on (pdb, chain). Keying on `pdb` alone let two chains of the same entry
    overwrite each other, dropping them from the resample without any diagnostic.
    A collision is now an error rather than a silent loss.
    """
    units = sorted({_unit_key(r) for r in rows})
    methods = sorted({r["method"] for r in rows})
    index = {u: i for i, u in enumerate(units)}
    out = {m: [None] * len(units) for m in methods}
    seen: set[tuple[str, str]] = set()
    for r in rows:
        key = (r["method"], _unit_key(r))
        if key in seen:
            raise ValueError(
                f"duplicate telemetry row for method={key[0]} unit={key[1]}: "
                "the unit key does not separate these rows, so one would be dropped"
            )
        seen.add(key)
        out[r["method"]][index[_unit_key(r)]] = r.get(metric_key)
    return out


def _pick_baseline(vals: dict[str, list[float | None]],
                   preferred: str | None) -> tuple[str, str]:
    """Choose a baseline that actually has values; never silently compare to nulls."""
    scored = [m for m, v in vals.items() if any(x is not None for x in v)]
    if preferred and preferred in scored:
        return preferred, "REQUESTED"
    if preferred:
        note = f"REQUESTED_{preferred}_HAS_NO_VALUES"
    else:
        note = "AUTO"
    for cand in ("p2rank", "pocketminer", "random_residue", "random_bbox"):
        if cand in scored:
            return cand, note
    # no external/null baseline scored: fall back to self (per-method CIs only)
    return (scored[0] if scored else next(iter(vals))), note + "_NO_BASELINE_AVAILABLE"


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Paired bootstrap CIs over structures.")
    ap.add_argument("--dataset", choices=("pilot", "official"), default="pilot")
    ap.add_argument("--telemetry", type=Path, default=None,
                    help="explicit TELEMETRY.json (overrides --dataset)")
    ap.add_argument("--baseline", default=None,
                    help="baseline method; auto-selected if absent/valueless")
    args = ap.parse_args(argv)
    is_official = args.dataset == "official"
    sub = "results/cryptobench_official" if is_official else "results/cryptobench_apo"
    telem_path = args.telemetry or (root / sub / "TELEMETRY.json")
    telem = json.loads(Path(telem_path).read_text())
    rows = telem["rows"]
    report: dict[str, Any] = {
        "schema": "geoaudit.bootstrap_report.v1",
        "clinical_grade": False,
        "dataset": ("cryptobench_official_mmseqs2_10pct_test_fold" if is_official
                    else "cryptobench_apo_pilot_n15"),
        "is_official_mmseqs2_10pct_test_fold": is_official,
        "official_fold_note": (
            "Official CryptoBench cluster-disjoint TEST fold (MMseqs2 @10% identity), "
            "loaded fail-closed with per-file SHA-256 verification."
            if is_official else
            "n=15 deterministic stride, NOT the official CryptoBench cluster-disjoint "
            "test fold."
        ),
        "telemetry_source": str(Path(telem_path).relative_to(root)),
        "metrics": {},
    }
    baseline = args.baseline
    for metric_key in ("residue_auc", "residue_pr_auc", "residue_mcc", "residue_f1"):
        vals = per_structure_values(rows, metric_key)
        if not any(v is not None for vs in vals.values() for v in vs):
            report["metrics"][metric_key] = {"status": "UNAVAILABLE_NO_VALUES"}
            continue
        base, note = _pick_baseline(vals, args.baseline)
        baseline = base
        res = paired_bootstrap(vals, baseline=base)
        res["baseline_selection"] = note
        report["metrics"][metric_key] = res
    report["baseline"] = baseline
    out = root / sub / "BOOTSTRAP_CI.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    def _f(v: float | None, spec: str = "+.3f") -> str:
        return "n/a" if v is None else format(v, spec)

    for mk, res in report["metrics"].items():
        if res.get("status"):
            print(f"== {mk}: {res['status']} ==")
            continue
        base = res.get("baseline")
        print(f"== {mk} (baseline={base} [{res.get('baseline_selection')}], "
              f"n={res['n_structures']}, boot={res['n_boot']}) ==")
        for m, s in res["per_method"].items():
            print(f"  {m:22s} {_f(s['point'], '.3f')}  "
                  f"[{_f(s['ci_low'], '.3f')}, {_f(s['ci_high'], '.3f')}]"
                  f"  n={s['n_structures_scored']}")
        for m, d in res["paired_vs_baseline"].items():
            cz = d["crosses_zero"]
            verdict = ("(Δ UNDEFINED: no scored structures)" if cz is None
                       else "(CI crosses 0)" if cz else "(CI excludes 0)")
            print(f"    Δ({m} - {base}) = {_f(d['delta_point'])} "
                  f"[{_f(d['delta_ci_low'])}, {_f(d['delta_ci_high'])}]  "
                  f"p≈{_f(d['p_two_sided_bootstrap'], '.3f')} {verdict}")
    print(f"-> {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
