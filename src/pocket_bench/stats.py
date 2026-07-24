"""Bootstrap CI, paired tests, random surface, label permutation, apo–holo controls."""
from __future__ import annotations

import math
import random
from typing import Any, Sequence

import numpy as np


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, Any]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(sample.mean()))
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return {
        "mean": float(arr.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "n": int(arr.size),
        "n_boot": n_boot,
        "alpha": alpha,
    }


def mcnemar_paired(success_a: Sequence[bool], success_b: Sequence[bool]) -> dict[str, Any]:
    """Exact two-sided McNemar/binomial test on paired per-structure success."""
    a = list(success_a)
    b = list(success_b)
    if len(a) != len(b) or not a:
        return {"n": 0, "b01": None, "b10": None, "p_approx": None}
    b01 = sum(1 for x, y in zip(a, b) if (not x) and y)  # A fail B success
    b10 = sum(1 for x, y in zip(a, b) if x and (not y))  # A success B fail
    n_disc = b01 + b10
    if n_disc == 0:
        return {"n": len(a), "b01": b01, "b10": b10, "p_exact": 1.0}
    smaller = min(b01, b10)
    tail = sum(math.comb(n_disc, k) for k in range(smaller + 1)) / (2**n_disc)
    return {
        "n": len(a),
        "b01": b01,
        "b10": b10,
        "p_exact": min(1.0, 2.0 * tail),
    }


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> dict[str, Any]:
    if total <= 0:
        return {"successes": successes, "n": total, "rate": None, "ci_low": None, "ci_high": None}
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {
        "successes": successes,
        "n": total,
        "rate": rate,
        "ci_low": max(0.0, center - margin),
        "ci_high": min(1.0, center + margin),
        "method": "Wilson score 95% interval",
    }


def paired_bootstrap_delta(
    success_a: Sequence[bool],
    success_b: Sequence[bool],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    a = np.asarray(success_a, dtype=float)
    b = np.asarray(success_b, dtype=float)
    if a.size == 0 or a.size != b.size:
        return {"delta_mean": None, "ci_low": None, "ci_high": None}
    delta = a - b
    rng = np.random.default_rng(seed)
    boots = [float(rng.choice(delta, size=delta.size, replace=True).mean()) for _ in range(n_boot)]
    boots.sort()
    return {
        "delta_mean": float(delta.mean()),
        "ci_low": boots[int(0.025 * n_boot)],
        "ci_high": boots[int(0.975 * n_boot) - 1],
        "n": int(a.size),
    }


def random_surface_centers(
    receptor_coords: Sequence[Sequence[float]],
    *,
    n_pockets: int = 5,
    seed: int = 13,
) -> list[dict[str, Any]]:
    """Baseline: random heavy-atom positions as fake pocket centers (surface proxy)."""
    if len(receptor_coords) < n_pockets:
        raise ValueError("not enough receptor atoms for random surface baseline")
    rng = random.Random(seed)
    idxs = list(range(len(receptor_coords)))
    rng.shuffle(idxs)
    pockets = []
    for rank, i in enumerate(idxs[:n_pockets], start=1):
        pockets.append(
            {
                "rank": rank,
                "center_xyz": [float(x) for x in receptor_coords[i]],
                "score": float(n_pockets - rank + 1),
            }
        )
    return pockets


def permute_labels(
    labels_by_pdb: dict[str, dict[str, Any]],
    *,
    seed: int = 7,
) -> dict[str, dict[str, Any]]:
    """Shuffle ligand labels across PDBs within the provided set."""
    keys = sorted(labels_by_pdb.keys())
    vals = [labels_by_pdb[k] for k in keys]
    rng = random.Random(seed)
    shuffled = vals[:]
    rng.shuffle(shuffled)
    return {k: shuffled[i] for i, k in enumerate(keys)}


def aggregate_primary(
    per_structure: list[dict[str, Any]],
    *,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    """Report conditional and intention-to-evaluate rates without tool/miss conflation."""
    eligible = [r for r in per_structure if r.get("eligible_for_primary")]
    fails = [r for r in per_structure if not r.get("eligible_for_primary")]
    succ = [bool(r["top1"]["success"]) for r in eligible if r.get("top1")]
    succ3 = [bool(r["top3"]["success"]) for r in eligible if r.get("top3")]
    unavailable = [r for r in per_structure if r.get("status") == "TOOL_UNAVAILABLE"]
    intention_rows = [
        r for r in per_structure if r.get("status") != "TOOL_UNAVAILABLE"
    ]
    intention_success = sum(
        bool((r.get("top1") or {}).get("success")) for r in intention_rows
    )
    rate = float(np.mean(succ)) if succ else None
    rate3 = float(np.mean(succ3)) if succ3 else None
    ci = bootstrap_mean_ci([1.0 if s else 0.0 for s in succ], seed=bootstrap_seed)
    dccs = [float(r["dcc_top1"]) for r in eligible if r.get("dcc_top1") is not None]
    runtimes = [
        float(r["runtime_s"])
        for r in per_structure
        if r.get("runtime_s") is not None
    ]
    f1s = [
        float(r["residue_f1"]["f1"])
        for r in eligible
        if (r.get("residue_f1") or {}).get("available") and r["residue_f1"].get("f1") is not None
    ]
    return {
        "n_structures_scored": len(per_structure),
        "n_eligible_primary": len(eligible),
        "n_tool_failures": len(fails),
        "failure_rate": (len(fails) / len(per_structure)) if per_structure else None,
        "top1_success_rate": rate,
        "top1_intention_to_evaluate": wilson_interval(
            intention_success, len(intention_rows)
        ),
        "top3_success_rate": rate3,
        "mean_dcc_top1": float(np.mean(dccs)) if dccs else None,
        "mean_runtime_s": float(np.mean(runtimes)) if runtimes else None,
        "median_runtime_s": float(np.median(runtimes)) if runtimes else None,
        "mean_residue_f1": float(np.mean(f1s)) if f1s else None,
        "n_residue_f1": len(f1s),
        "top1_bootstrap_95ci": ci,
        "top1_wilson_95ci": wilson_interval(sum(succ), len(succ)),
        "n_tool_unavailable": len(unavailable),
        "note": (
            "TOOL_UNAVAILABLE makes the environment incomplete and is not a miss. "
            "CRASH/EMPTY count as failures in intention-to-evaluate; the conditional "
            "rate is reported separately."
        ),
    }
