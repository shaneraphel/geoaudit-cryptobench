#!/usr/bin/env python3
"""PocketMiner's threshold, chosen on the training fold and frozen there.

PocketMiner publishes no operating point. It emits a probability per residue and
the paper reports ROC-AUC, so there is no native binarisation to compare against
the way P2Rank's pocket assignment can be. A thresholded comparison therefore has
to give it one, and the only defensible place to get it is the training fold --
which PocketMiner has never seen either, since it was fitted on molecular
dynamics simulations of 38 unrelated systems.

Two grids, because a reviewer asked for both and they answer different questions:

  A per-chain top-q budget, on the same 0.02 to 0.40 grid the other two methods
  were tuned on. This is the arm that makes the three-way comparison matched:
  every method calls the same number of residues.

  A probability cut on the raw score. This is what someone deploying PocketMiner
  would actually set, and it lets the method call a different number of residues
  on different chains -- which is the freedom P2Rank's native assignment has and
  a fixed budget does not.

Pooled F1 and pooled MCC are both maximised, for the reason
``train_operating_points.py`` gives: the two do not peak together, and a
comparison that only ever optimises F1 cannot say whether the ordering survives a
different summary of the same confusion counts.

Precision and recall are recorded at every grid point, not just at the argmax,
because the curve is what answers whether one method's advantage over another is
a property of the ranking or of where the two were cut.

Nothing here touches the held-out fold, and ``--check`` asserts it.

Usage: python3.12 tools/pocketminer_train_operating_point.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "data/baselines/pocketminer_train"
MANIFEST_SCORES = ROOT / "results/baselines/POCKETMINER_TRAIN_SCORES.json"
LAB = ROOT / "data/cryptobench_apo/train_labels"
OUT = ROOT / "results/architecture_sweep/POCKETMINER_TRAIN_OPERATING_POINT.json"

SCHEMA = "geoaudit.pocketminer_train_operating_point.v1"
Q_GRID = [round(float(q), 2) for q in np.arange(0.02, 0.41, 0.01)]
P_GRID = [round(float(p), 2) for p in np.arange(0.05, 1.00, 0.05)]


def _resnum(x) -> int | None:
    """``pocket_bench.residue_id``'s convention, duplicated on purpose."""
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


def _units() -> list[dict]:
    out = []
    for f in sorted(SCORES.glob("*.json")):
        unit = f.stem
        lab = LAB / f"{unit}_labels.json"
        if not lab.is_file():
            raise SystemExit(f"no training labels for {unit}")
        d = json.loads(lab.read_text())
        pos = {r for r in (_resnum(r) for r in
                           (d.get("cryptic_residues")
                            or d.get("binding_residues") or []))
               if r is not None}
        raw = json.loads(f.read_text())["residue_scores"]
        by_res = {}
        for k, v in raw.items():
            r = _resnum(k)
            if r is not None:
                by_res[r] = float(v)
        keys = sorted(by_res)
        if not keys or not (pos & set(keys)):
            continue
        out.append({"unit": unit,
                    "s": np.array([by_res[k] for k in keys], dtype=np.float64),
                    "y": np.array([k in pos for k in keys], dtype=bool)})
    return out


def _top_q(s: np.ndarray, q: float) -> np.ndarray:
    """The budget rule the counting field deploys, character for character."""
    n = len(s)
    k = max(1, int(round(q * n)))
    out = np.zeros(n, dtype=bool)
    out[np.argsort(-s, kind="stable")[:k]] = True
    return out


def _pooled(calls: list[np.ndarray], truth: list[np.ndarray]) -> dict:
    tp = fp = fn = tn = 0
    for c, y in zip(calls, truth):
        tp += int((c & y).sum())
        fp += int((c & ~y).sum())
        fn += int((~c & y).sum())
        tn += int((~c & ~y).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    den = math.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / den) if den else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n_called": tp + fp,
            "pooled_precision": round(prec, 6),
            "pooled_recall": round(rec, 6),
            "pooled_f1": round(f1, 6),
            "pooled_mcc": round(mcc, 6)}


def build() -> dict:
    units = _units()
    if not units:
        raise SystemExit(f"no scored training units under {SCORES}")
    truth = [u["y"] for u in units]
    q_rows = []
    for q in Q_GRID:
        r = _pooled([_top_q(u["s"], q) for u in units], truth)
        r["q"] = q
        q_rows.append(r)
    p_rows = []
    for p in P_GRID:
        r = _pooled([u["s"] >= p for u in units], truth)
        r["threshold"] = p
        p_rows.append(r)

    def pick(rows: list[dict], key: str, tag: str) -> dict:
        best = max(rows, key=lambda r: r[key])
        ties = [r[tag] for r in rows if r[key] == best[key]]
        return {tag: best[tag], "value": best[key], "n_called": best["n_called"],
                "pooled_precision": best["pooled_precision"],
                "pooled_recall": best["pooled_recall"],
                "ties_at": ties}

    n_res = int(sum(len(u["y"]) for u in units))
    n_pos = int(sum(int(u["y"].sum()) for u in units))
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": "where should PocketMiner be cut, decided on training data "
                    "alone, since it publishes no operating point",
        "test_fold_touched": False,
        "reads_test_fold": False,
        "why_the_training_fold_is_fair_to_it": (
            "PocketMiner was fitted on molecular dynamics simulations of 38 "
            "systems and has seen neither of CryptoBench's folds, so a threshold "
            "chosen on the training fold is out-of-sample for it in the same way "
            "P2Rank's was"),
        "scores": {
            "artifact": str(MANIFEST_SCORES.relative_to(ROOT)),
            "artifact_sha256": hashlib.sha256(
                MANIFEST_SCORES.read_bytes()).hexdigest(),
            "n_units_with_a_positive_residue": len(units),
            "n_residues": n_res,
            "n_positive_residues": n_pos,
            "positive_rate": round(n_pos / n_res, 6),
        },
        "rule": {
            "budget": "per-chain top-q by score, ties broken by a stable sort of "
                      "the negated score, identical to the counting field's",
            "cut": "score at or above a fixed probability, applied to every chain",
            "objective": "pooled over all units: the TP/FP/TN/FN are summed and "
                         "the summary computed once, not averaged per unit",
            "q_grid": {"low": Q_GRID[0], "high": Q_GRID[-1], "step": 0.01,
                       "n": len(Q_GRID)},
            "p_grid": {"low": P_GRID[0], "high": P_GRID[-1], "step": 0.05,
                       "n": len(P_GRID)},
            "argmax_tie_break": "the smallest value attaining the maximum",
        },
        "selected": {
            "budget/pooled_f1": pick(q_rows, "pooled_f1", "q"),
            "budget/pooled_mcc": pick(q_rows, "pooled_mcc", "q"),
            "cut/pooled_f1": pick(p_rows, "pooled_f1", "threshold"),
            "cut/pooled_mcc": pick(p_rows, "pooled_mcc", "threshold"),
        },
        "curves": {"budget": q_rows, "cut": p_rows},
    }


def _report(d: dict) -> None:
    s = d["scores"]
    print(f"{s['n_units_with_a_positive_residue']} training units, "
          f"{s['n_residues']} residues, {s['n_positive_residues']} positive "
          f"({s['positive_rate']:.4f})")
    for k, v in d["selected"].items():
        tag = "q" if k.startswith("budget") else "threshold"
        print(f"  {k:22s} {tag}={v[tag]:<5} value {v['value']:.4f}  "
              f"precision {v['pooled_precision']:.4f}  "
              f"recall {v['pooled_recall']:.4f}  called {v['n_called']}")


def _check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    have = json.loads(OUT.read_text())
    if have.get("schema") != SCHEMA:
        print(f"FAILED: schema {have.get('schema')}")
        return 1
    if have.get("reads_test_fold") or have.get("test_fold_touched"):
        print("FAILED: the artifact claims to have touched the held-out fold")
        return 1
    for name, rows in have["curves"].items():
        tag = "q" if name == "budget" else "threshold"
        if any(r["pooled_f1"] < 0 or r["pooled_f1"] > 1 for r in rows):
            print(f"FAILED: {name} curve has an F1 outside [0,1]")
            return 1
        if [r[tag] for r in rows] != sorted(r[tag] for r in rows):
            print(f"FAILED: {name} curve is not ordered by {tag}")
            return 1
    for k, v in have["selected"].items():
        rows = have["curves"]["budget" if k.startswith("budget") else "cut"]
        key = "pooled_f1" if k.endswith("f1") else "pooled_mcc"
        if v["value"] != max(r[key] for r in rows):
            print(f"FAILED: {k} is not the maximum of its own curve")
            return 1
    _report(have)
    print(f"OK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        return _check()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
