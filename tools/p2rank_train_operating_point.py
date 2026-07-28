"""Give P2Rank the operating point our own method was given.

The published F1 comparison binarises the two methods by different rules. Ours
takes the top ``q`` fraction of residues in each chain, with ``q`` chosen on the
training fold to maximise pooled training F1. P2Rank is scored at its own pocket
assignment, which is a rule its authors fixed and nobody tuned on CryptoBench.
Those are not the same kind of object, and a reviewer is entitled to ask whether
our margin is a property of the scores or of the threshold.

Answering that fairly needs P2Rank's operating point chosen the way ours was:
same grid, same pooled-F1 objective, same training receptors, and fixed before
the held-out fold is consulted. ``run_p2rank_on_train`` already scored the 770
training receptors but kept only per-unit summaries, so the per-residue scores
that a threshold acts on are not on disk. This tool re-runs P2Rank on the same
receptors, keeps the per-residue table, and reports the whole F1-against-q curve
rather than only its argmax, so that the choice can be seen rather than trusted.

Two things make the re-run auditable. P2Rank 2.5.1 is deterministic given a
receptor, so every per-unit F1 recomputed here from the retained table must equal
the committed ``P2RANK_TRAIN_FOLD.json`` value; the artifact records the largest
disagreement and refuses to be written if one is material. And the grid search is
a transcription of ``table_field._best_operating_point``, checked against that
function on the retained arrays rather than assumed to match it.

The test fold is not opened. The receptors and labels read here are the training
partition only, which is why this artifact is filed under the architecture sweep
and not in the fold-access ledger.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pocket_bench import residue_id
from pocket_bench.methods import p2rank_wrap, table_field
from pocket_bench.paths import ROOT

REC = ROOT / "data/cryptobench_apo/train_receptors"
LAB = ROOT / "data/cryptobench_apo/train_labels"
SUMMARY = ROOT / "results/architecture_sweep/P2RANK_TRAIN_FOLD.json"
OUT = ROOT / "results/architecture_sweep/P2RANK_TRAIN_OPERATING_POINT.json"
# Underscore-prefixed and untracked, like the wide-bus cache: 231k per-residue
# scores are an intermediate, and the decision they support is what belongs in
# the repository. Regenerating it needs the JVM, exactly as the committed
# P2Rank summaries already do.
CACHE = ROOT / "results/architecture_sweep/_p2rank_train_residues.npz"

SCHEMA = "geoaudit.p2rank_train_operating_point.v1"
# The grid table_field searched, quoted from its source rather than retyped.
Q_GRID = np.arange(0.02, 0.41, 0.01)
# A recomputed per-unit F1 may differ from the committed one only by float
# formatting in JSON. Anything larger means the re-run is not the same run.
F1_TOLERANCE = 1e-9


def _universe(rec: Path, chain: str | None) -> list[int]:
    """Residue numbers of the scored chain.

    Duplicated from ``run_p2rank_on_train`` for the same reason it duplicated
    it: this runs in eight worker processes and the harness module imports a
    heavy graph at load time.
    """
    seen: list[int] = []
    got = set()
    for line in rec.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if chain and line[21] != chain:
            continue
        try:
            n = int(line[22:26])
        except ValueError:
            continue
        if n not in got:
            got.add(n)
            seen.append(n)
    return seen


def _one(label_path: str) -> dict:
    """P2Rank's per-residue table for one training unit.

    The alignment is ``metrics.native_residue_scores``: keys are the residue
    numbers of the universe in ascending order, a residue P2Rank did not mention
    scores zero, and the native call is P2Rank's own ``pocket > 0`` set
    restricted to the universe. Reproducing that here rather than importing it
    keeps the worker's import graph small, and the audit against the committed
    summaries is what proves the reproduction is faithful.
    """
    lab = json.loads(Path(label_path).read_text())
    pdb, ch = lab["pdb_id"], lab.get("chain")
    rec = REC / f"{pdb}_{ch}_receptor.pdb"
    unit = f"{pdb}_{ch}"
    if not rec.is_file():
        return {"unit_id": unit, "status": "MISSING_RECEPTOR"}
    pred = p2rank_wrap.predict(rec, pdb_id=pdb, chain=ch)
    keys = sorted({r for r in (residue_id.resseq(u)
                               for u in _universe(rec, ch)) if r is not None})
    if not keys:
        return {"unit_id": unit, "status": "EMPTY_UNIVERSE"}
    index = {k: i for i, k in enumerate(keys)}
    score = np.zeros(len(keys), dtype=np.float64)
    for rid, val in (pred.get("residue_scores") or {}).items():
        r = residue_id.resseq(rid)
        if r is not None and r in index:
            score[index[r]] = float(val)
    native = np.zeros(len(keys), dtype=bool)
    for rid in pred.get("residue_positive") or []:
        r = residue_id.resseq(rid)
        if r is not None and r in index:
            native[index[r]] = True
    truth = np.zeros(len(keys), dtype=bool)
    for rid in (lab.get("cryptic_residues") or lab.get("binding_residues") or []):
        r = residue_id.resseq(rid)
        if r is not None and r in index:
            truth[index[r]] = True
    return {
        "unit_id": unit,
        "status": pred.get("status", "OK"),
        "resseq": np.asarray(keys, dtype=np.int64),
        "score": score,
        "native": native,
        "truth": truth,
        "runtime_s": pred.get("runtime_s"),
    }


def _pooled_f1(call: np.ndarray, truth: np.ndarray) -> float:
    tp = int((call & truth).sum())
    fp = int((call & ~truth).sum())
    fn = int((~call & truth).sum())
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0


def top_q_call(s: np.ndarray, q: float) -> np.ndarray:
    """``TableField.positive_call``, for whichever method's scores are passed.

    The tie-break is the one our method already uses: a stable sort of the
    negated score, so residues holding equal scores are called in ascending
    residue-number order. Naming it here matters because a matched comparison
    that broke ties differently for the two methods would not be matched.
    """
    n = len(s)
    k = max(1, int(round(q * n)))
    call = np.zeros(n, dtype=bool)
    call[np.argsort(-np.asarray(s, dtype=np.float64), kind="stable")[:k]] = True
    return call


def f1_against_q(scores: list[np.ndarray], truths: list[np.ndarray]) -> list[dict]:
    """Pooled training F1 at every q on the grid, per-chain top-q rule."""
    rows = []
    for q in Q_GRID:
        tp = fp = fn = 0
        for s, t in zip(scores, truths):
            call = top_q_call(s, float(q))
            tp += int((call & t).sum())
            fp += int((call & ~t).sum())
            fn += int((~call & t).sum())
        d = 2 * tp + fp + fn
        rows.append({"q": round(float(q), 2),
                     "pooled_train_f1": round((2 * tp / d) if d else 0.0, 6),
                     "n_called": int(tp + fp)})
    return rows


def _verify_grid(scores: list[np.ndarray], truths: list[np.ndarray],
                 rows: list[dict]) -> None:
    """The grid search must be table_field's, not a lookalike."""
    flat_s = np.concatenate(scores)
    flat_y = np.concatenate([t.astype(np.int64) for t in truths])
    n_res = [len(s) for s in scores]
    q_theirs, f1_theirs = table_field._best_operating_point(flat_s, flat_y, n_res)
    best = max(rows, key=lambda r: r["pooled_train_f1"])
    if abs(best["q"] - q_theirs) > 1e-9:
        raise SystemExit(
            f"the grid here peaks at q={best['q']} but "
            f"table_field._best_operating_point returns q={q_theirs}; the "
            "matched rule would not be the published rule")
    if abs(best["pooled_train_f1"] - round(f1_theirs, 6)) > 1e-6:
        raise SystemExit(
            f"pooled F1 disagrees with table_field at the same q: "
            f"{best['pooled_train_f1']} here, {f1_theirs} there")


def _audit_against_summary(units: list[dict]) -> dict:
    """Every per-unit F1 recomputed here must equal the committed one.

    P2Rank is deterministic, so this is not a tolerance test on a stochastic
    method: a disagreement means the re-run scored something other than what
    the committed summary describes, and the matched analysis would then be
    resting on different numbers than the paper's training-fold statement.
    """
    if not SUMMARY.is_file():
        return {"checked": False, "why": f"{SUMMARY.name} absent"}
    want = {r["unit_id"]: r.get("residue_f1")
            for r in json.loads(SUMMARY.read_text())["rows"]}
    worst, worst_unit, n = 0.0, None, 0
    for u in units:
        ref = want.get(u["unit_id"])
        if ref is None:
            continue
        mine = _pooled_f1(u["native"], u["truth"])
        n += 1
        if abs(mine - ref) > worst:
            worst, worst_unit = abs(mine - ref), u["unit_id"]
    return {"checked": True, "n_units_compared": n,
            "largest_disagreement": worst, "at_unit": worst_unit,
            "tolerance": F1_TOLERANCE,
            "agrees": bool(worst <= F1_TOLERANCE)}


def _save_cache(units: list[dict]) -> str:
    np.savez_compressed(
        CACHE,
        unit_id=np.array([u["unit_id"] for u in units]),
        n_res=np.array([len(u["score"]) for u in units], dtype=np.int64),
        resseq=np.concatenate([u["resseq"] for u in units]),
        score=np.concatenate([u["score"] for u in units]),
        native=np.concatenate([u["native"] for u in units]),
        truth=np.concatenate([u["truth"] for u in units]),
    )
    return hashlib.sha256(CACHE.read_bytes()).hexdigest()


def load_cache() -> list[dict]:
    with np.load(CACHE, allow_pickle=False) as z:
        offs = np.concatenate([[0], np.cumsum(z["n_res"])])
        return [{"unit_id": str(u), "status": "OK",
                 "resseq": z["resseq"][a:b], "score": z["score"][a:b],
                 "native": z["native"][a:b], "truth": z["truth"][a:b]}
                for u, a, b in zip(z["unit_id"], offs[:-1], offs[1:])]


def build(workers: int = 8, limit: int | None = None,
          reuse_cache: bool = False) -> dict:
    t0 = time.perf_counter()
    if reuse_cache and CACHE.is_file():
        units = load_cache()
        cache_sha = hashlib.sha256(CACHE.read_bytes()).hexdigest()
        print(f"reusing {CACHE.name}: {len(units)} units", flush=True)
    else:
        labels = sorted(glob.glob(str(LAB / "*_labels.json")))
        if limit:
            labels = labels[:limit]
        got: list[dict] = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, u in enumerate(pool.map(_one, labels, chunksize=4), 1):
                got.append(u)
                if i % 50 == 0 or i == len(labels):
                    print(f"  {i}/{len(labels)}  "
                          f"{time.perf_counter() - t0:.0f}s", flush=True)
        units = [u for u in got if u.get("status") == "OK" and "score" in u]
        cache_sha = _save_cache(units) if not limit else "not-written"

    scores = [u["score"] for u in units]
    truths = [u["truth"] for u in units]
    rows = f1_against_q(scores, truths)
    _verify_grid(scores, truths, rows)
    best = max(rows, key=lambda r: r["pooled_train_f1"])
    native_f1 = _pooled_f1(np.concatenate([u["native"] for u in units]),
                           np.concatenate(truths))
    ours_q = float(json.loads(
        (ROOT / "data/cryptobench_apo/TABLE_FIELD.json").read_text()
    )["operating_point"]["q"])
    at_ours = next(r for r in rows if abs(r["q"] - ours_q) < 1e-9)
    audit = _audit_against_summary(units)
    if audit.get("checked") and not audit["agrees"]:
        raise SystemExit(
            "the re-run disagrees with the committed training summary by "
            f"{audit['largest_disagreement']:.3g} at {audit['at_unit']}; "
            "refusing to write an artifact built on different numbers")
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "question": (
            "what operating point P2Rank would have been given if it had been "
            "tuned the way our method was: same per-chain top-q rule, same "
            "grid, same pooled-F1 objective, same training receptors"
        ),
        "test_fold_touched": False,
        "reads_test_fold": False,
        "rule": {
            "binarisation": "per-chain top-q by score",
            "tie_break": "stable sort of the negated score, so equal scores "
                         "are called in ascending residue number",
            "objective": "pooled F1 over all training units, the TP/FP/FN "
                         "summed across chains rather than F1 averaged over "
                         "them, which is what table_field._best_operating_point "
                         "maximises",
            "grid": [round(float(Q_GRID[0]), 2), round(float(Q_GRID[-1]), 2),
                     0.01],
            "transcribed_from": "pocket_bench.methods.table_field."
                                "_best_operating_point",
        },
        "receptor_dir": str(REC.relative_to(ROOT)),
        "label_dir": str(LAB.relative_to(ROOT)),
        "p2rank_version": p2rank_wrap._version(),
        "n_units": len(units),
        "n_residues": int(sum(len(s) for s in scores)),
        "n_positives": int(sum(int(t.sum()) for t in truths)),
        "residue_cache": {
            "path": str(CACHE.relative_to(ROOT)),
            "tracked": False,
            "sha256": cache_sha,
            "why_untracked": "231k per-residue scores are an intermediate; the "
                             "curve below is the decision they support, and "
                             "regenerating them needs the JVM exactly as the "
                             "committed P2Rank summaries do",
        },
        "f1_against_q": rows,
        "p2rank_selected_q": best["q"],
        "p2rank_pooled_train_f1_at_selected_q": best["pooled_train_f1"],
        "p2rank_pooled_train_f1_at_native_call": round(native_f1, 6),
        "our_q": ours_q,
        "p2rank_pooled_train_f1_at_our_q": at_ours["pooled_train_f1"],
        "tuning_is_worth_to_p2rank": round(
            best["pooled_train_f1"] - native_f1, 6),
        "grid_verified_against_table_field": True,
        "audit_against_committed_summary": audit,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
    }


def _report(d: dict) -> None:
    print(f"\nP2Rank {d['p2rank_version']} operating point on the training fold")
    print(f"  units              {d['n_units']}  "
          f"({d['n_residues']} residues, {d['n_positives']} positive)")
    print(f"  native pocket call pooled train F1 "
          f"{d['p2rank_pooled_train_f1_at_native_call']:.4f}")
    print(f"  best top-q         q={d['p2rank_selected_q']:.2f}  "
          f"F1 {d['p2rank_pooled_train_f1_at_selected_q']:.4f}")
    print(f"  at our q={d['our_q']:.2f}        "
          f"F1 {d['p2rank_pooled_train_f1_at_our_q']:.4f}")
    print(f"  tuning is worth    "
          f"{d['tuning_is_worth_to_p2rank']:+.4f} to P2Rank")
    a = d["audit_against_committed_summary"]
    if a.get("checked"):
        print(f"  re-run agrees with the committed summary on "
              f"{a['n_units_compared']} units "
              f"(worst {a['largest_disagreement']:.3g})")
    print("  test fold          untouched")


def _check() -> int:
    """Audit the committed artifact without the JVM."""
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    if d.get("schema") != SCHEMA:
        print(f"FAILED: schema {d.get('schema')}")
        return 1
    if d.get("test_fold_touched") is not False:
        print("FAILED: artifact does not declare the test fold untouched")
        return 1
    rows = d["f1_against_q"]
    best = max(rows, key=lambda r: r["pooled_train_f1"])
    if abs(best["q"] - d["p2rank_selected_q"]) > 1e-9:
        print(f"FAILED: recorded q {d['p2rank_selected_q']} is not the argmax "
              f"of the committed curve ({best['q']})")
        return 1
    if abs(best["pooled_train_f1"]
           - d["p2rank_pooled_train_f1_at_selected_q"]) > 1e-9:
        print("FAILED: recorded F1 at the selected q is not the curve's peak")
        return 1
    want = round(d["p2rank_pooled_train_f1_at_selected_q"]
                 - d["p2rank_pooled_train_f1_at_native_call"], 6)
    if abs(want - d["tuning_is_worth_to_p2rank"]) > 1e-6:
        print("FAILED: the recorded value of tuning is not the difference of "
              "the two recorded F1 values")
        return 1
    ours = float(json.loads(
        (ROOT / "data/cryptobench_apo/TABLE_FIELD.json").read_text()
    )["operating_point"]["q"])
    if abs(ours - d["our_q"]) > 1e-9:
        print(f"FAILED: artifact says our q is {d['our_q']} but the shipped "
              f"field says {ours}")
        return 1
    if not d.get("audit_against_committed_summary", {}).get("agrees", False):
        print("FAILED: the run that produced this did not agree with the "
              "committed training-fold summary")
        return 1
    if len(rows) != len(Q_GRID):
        print(f"FAILED: curve has {len(rows)} points, grid has {len(Q_GRID)}")
        return 1
    _report(d)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N units (smoke test)")
    ap.add_argument("--reuse-cache", action="store_true",
                    help="recompute the curve from the local per-residue "
                         "cache instead of re-running P2Rank")
    ap.add_argument("--check", action="store_true",
                    help="audit the committed artifact; no JVM needed")
    a = ap.parse_args(argv)
    if a.check:
        return _check()
    d = build(workers=a.workers, limit=a.limit, reuse_cache=a.reuse_cache)
    if not a.limit:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
