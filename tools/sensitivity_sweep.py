#!/usr/bin/env python3
"""Is the field's accuracy an accident of three constants nobody varied?

Three numbers were fixed early and never moved: four quantisation levels, ranks
taken within the chain rather than across the fold, and a fan-out cap of 32. A
reader is entitled to ask whether any of them is load-bearing, and the honest
way to answer is to vary each one on the training partition and report what
happens -- including the cases where something else would have been better.

Nothing here is a selection. The published configuration was frozen before this
was written and is not revised by it; the sweep is a robustness statement about
that configuration, and the held-out fold is not read.

What is varied
--------------
**Levels.** Three, four and five bands per wire. More levels resolve a wire
more finely and give every table more cells to populate from the same fold, so
the two effects run against each other and the answer is not obvious in advance.

**Where the rank is taken.** Within the chain, which is what the method does, or
across the pooled training partition. The within-chain choice is not a tuning
decision but a structural one -- it is what lets a chain be scored on its own,
with no constant carried from training -- so the pooled variant is reported to
show what that costs, not as an alternative to adopt. A pooled rank would also
have to carry twenty-odd quantile boundaries into inference, which the method
deliberately does not do.

**Fan-out cap.** Sixteen, thirty-two, sixty-four. The cap bounds the integer
multiplicities, so it sets how much of the continuous ridge direction survives
rounding. It changes only the fusion step, so the compiled cells are shared
across the three and the sweep is nearly free.

Why the digitiser is re-implemented here
----------------------------------------
``table_bank`` fixes ``N_LEVELS`` as a module constant and the shipped artifact
is compiled against it, so parameterising it in place would change the hash of a
module the frozen field is committed to. The level-aware versions live here
instead, and ``_verify_equivalence`` asserts that at four levels and within-chain
ranking they reproduce ``table_bank.chain_digits`` and ``table_bank.addresses``
exactly. The duplicate is therefore checked rather than trusted.

Usage: PYTHONPATH=src:tools python3.12 tools/sensitivity_sweep.py [--check]
"""
from __future__ import annotations

import argparse
import gc
import json
import time

import numpy as np

from pocket_bench.methods import table_bank
from pocket_bench.methods.table_bank import partition_tables
from pocket_bench.paths import ROOT

from counterattack_ridge import spread_matched_gate
from counterattack_select import SEED, per_unit_auc

CACHE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/SENSITIVITY_SWEEP.json"
TABFIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
SCHEMA = "geoaudit.sensitivity_sweep.v1"

# The published configuration. Everything below varies exactly one of these.
FROZEN = {"levels": 4, "ranking": "within-chain", "cap": 32, "ridge": 0.03,
          "basis": "pairs x16", "rounds": 16, "width": 2}
LEVELS = (3, 4, 5)
RANKINGS = ("within-chain", "pooled")
CAPS = (16, 32, 64)
GATES = ((14.0, 1.0), (18.0, 0.5), (18.0, 1.0))


def _rank_digits(x: np.ndarray, levels: int) -> np.ndarray:
    """Mid-rank banding of one vector into ``levels`` bands."""
    n = len(x)
    order = np.argsort(x, kind="stable")
    r = np.empty(n)
    i = 0
    while i < n:
        k = i
        while k + 1 < n and x[order[k + 1]] == x[order[i]]:
            k += 1
        r[order[i:k + 1]] = 0.5 * (i + k)
        i = k + 1
    return np.clip(np.floor(r / max(n - 1, 1) * levels), 0, levels - 1)


def digits(F: np.ndarray, n_res_per, levels: int, ranking: str) -> np.ndarray:
    """Banded digits, either within each chain or over the pooled partition."""
    F = np.asarray(F, dtype=np.float64)
    out = np.empty(F.shape, dtype=np.int8)
    if ranking == "pooled":
        for j in range(F.shape[1]):
            out[:, j] = _rank_digits(F[:, j], levels)
        return out
    off = 0
    for n in n_res_per:
        n = int(n)
        for j in range(F.shape[1]):
            out[off:off + n, j] = _rank_digits(F[off:off + n, j], levels)
        off += n
    return out


def cell_offsets(tables, levels: int) -> np.ndarray:
    return np.concatenate(
        [[0], np.cumsum([levels ** len(t) for t in tables])]).astype(np.int64)


def addresses(D, tables, offsets, a: int, b: int, levels: int) -> np.ndarray:
    out = np.empty((b - a, len(tables)), dtype=np.int64)
    for k, cols in enumerate(tables):
        acc = np.zeros(b - a, dtype=np.int64)
        for t, c in enumerate(cols):
            acc += D[a:b, c].astype(np.int64) * (levels ** t)
        out[:, k] = acc + offsets[k]
    return out


def compile_cells(D, y, tables, offsets, levels: int):
    total = int(offsets[-1])
    tot = np.zeros(total, dtype=np.int64)
    pos = np.zeros(total, dtype=np.float64)
    yf = y.astype(np.float64)
    for a in range(0, D.shape[0], table_bank.BLOCK):
        b = min(a + table_bank.BLOCK, D.shape[0])
        flat = addresses(D, tables, offsets, a, b, levels).ravel()
        tot += np.bincount(flat, minlength=total)
        pos += np.bincount(flat, weights=np.repeat(yf[a:b], len(tables)),
                           minlength=total)
    rate = float(yf.mean())
    return np.where(tot > 0, pos / np.maximum(tot, 1), rate), tot


def blocks(D, tables, offsets, frac, levels: int):
    for a in range(0, D.shape[0], table_bank.BLOCK):
        b = min(a + table_bank.BLOCK, D.shape[0])
        yield a, b, frac[addresses(D, tables, offsets, a, b, levels)]


def integer_fanout(D, y, tables, offsets, frac, ridge: float, cap: int,
                   levels: int) -> np.ndarray:
    """``table_bank.integer_fanout`` with the level fixed by the caller."""
    K = len(tables)
    s1 = np.zeros(K)
    s0 = np.zeros(K)
    pos = y == 1
    n1 = int(pos.sum())
    n0 = int(len(y) - n1)
    for a, b, v in blocks(D, tables, offsets, frac, levels):
        p = pos[a:b]
        s1 += v[p].sum(0)
        s0 += v[~p].sum(0)
    mu1, mu0 = s1 / max(n1, 1), s0 / max(n0, 1)
    S = np.zeros((K, K))
    for a, b, v in blocks(D, tables, offsets, frac, levels):
        p = pos[a:b]
        c = np.where(p[:, None], v - mu1, v - mu0)
        S += c.T @ c
    S /= max(len(y) - 2, 1)
    S.flat[::K + 1] += ridge * float(np.trace(S)) / K + 1e-12
    w = np.linalg.solve(S, mu1 - mu0)
    peak = float(np.abs(w).max())
    if peak <= 0:
        return np.zeros(K, dtype=np.int64)
    return np.round(w / peak * cap).astype(np.int64)


def score(D, tables, offsets, frac, mult, levels: int) -> np.ndarray:
    out = np.empty(D.shape[0], dtype=np.float64)
    m = mult.astype(np.float64)
    for a, b, v in blocks(D, tables, offsets, frac, levels):
        out[a:b] = v @ m
    return out


def _verify_equivalence(F, n_res_per) -> None:
    """At the published setting the local code must be the shipped code."""
    sub = F[:4000, :24]
    n_sub, taken = [], 0
    for n in n_res_per:
        if taken + int(n) > 4000:
            break
        n_sub.append(int(n))
        taken += int(n)
    sub = sub[:taken]
    mine = digits(sub, n_sub, table_bank.N_LEVELS, "within-chain")
    theirs = table_bank.chain_digits(sub, n_sub)
    if not np.array_equal(mine, theirs):
        raise SystemExit("the level-aware digitiser does not reproduce "
                         "table_bank.chain_digits at four levels; the sweep "
                         "would not be measuring the published method")
    tables = [[0, 1], [2, 3, 4]]
    off_mine = cell_offsets(tables, table_bank.N_LEVELS)
    off_theirs = table_bank.cell_offsets(tables)
    if not np.array_equal(off_mine, off_theirs):
        raise SystemExit("the level-aware cell offsets disagree with "
                         "table_bank.cell_offsets at four levels")
    a_mine = addresses(theirs, tables, off_mine, 0, 200, table_bank.N_LEVELS)
    a_theirs = table_bank.addresses(theirs, tables, off_theirs, 0, 200)
    if not np.array_equal(a_mine, a_theirs):
        raise SystemExit("the level-aware addressing disagrees with "
                         "table_bank.addresses at four levels")
    # The fusion and the scoring too, since those are what the sweep reports.
    L = table_bank.N_LEVELS
    yv = (np.arange(len(theirs)) % 7 == 0).astype(np.float64)
    f_mine, _ = compile_cells(theirs, yv, tables, off_mine, L)
    f_theirs, _ = table_bank.compile_cells(theirs, yv, tables, off_theirs)
    if not np.allclose(f_mine, f_theirs, atol=0, rtol=0):
        raise SystemExit("the level-aware cell compilation disagrees with "
                         "table_bank.compile_cells at four levels")
    m_mine = integer_fanout(theirs, yv, tables, off_mine, f_mine, 0.03, 32, L)
    m_theirs = table_bank.integer_fanout(theirs, yv, tables, off_theirs,
                                         f_theirs, 0.03, 32)
    if not np.array_equal(m_mine, m_theirs):
        raise SystemExit("the level-aware fan-out disagrees with "
                         "table_bank.integer_fanout at four levels")
    s_mine = score(theirs, tables, off_mine, f_mine, m_mine, L)
    s_theirs = table_bank.score(theirs, tables, off_theirs, f_theirs, m_theirs)
    if not np.array_equal(s_mine, s_theirs):
        raise SystemExit("the level-aware scoring disagrees with "
                         "table_bank.score at four levels")


def _split(units, n_res):
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"]
                  for e in json.loads(MANIFEST.read_text())["entries"]}
    clusters = sorted({cluster_of[u] for u in units})
    rng = np.random.default_rng(SEED)
    rng.shuffle(clusters)
    fit = set(clusters[:len(clusters) // 2])
    is_fit = np.array([cluster_of[u] in fit for u in units])
    row_unit = np.repeat(np.arange(len(units)), n_res)
    return is_fit, is_fit[row_unit], ~is_fit[row_unit]


def _gated_auc(S, ctr_pick, n_pick, ypick):
    best, gname = -1.0, ""
    for r, wt in GATES:
        a = per_unit_auc(spread_matched_gate(S, ctr_pick, n_pick, r, wt),
                         ypick, n_pick)
        if a > best:
            best, gname = a, f"r{int(r)} w{wt}"
    return best, gname


def _load_wires() -> np.ndarray:
    """Just the wire matrix, so the caller can drop it as soon as it has the
    digits."""
    with np.load(CACHE, allow_pickle=False) as z:
        return z["X"]


def _summarise(rows: list[dict]) -> dict:
    base = next((r for r in rows if r["is_published_configuration"]), None)
    best = max(rows, key=lambda r: r["pick_half_roc_auc"]) if rows else None
    spread = (max(r["pick_half_roc_auc"] for r in rows)
              - min(r["pick_half_roc_auc"] for r in rows)) if rows else 0.0
    done = sorted({(r["levels"], r["ranking"]) for r in rows})
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "question": "whether the published quantisation and fan-out constants "
                    "are load-bearing, measured on the training partition "
                    "with the published configuration frozen and not revised",
        "frozen_configuration": FROZEN,
        "swept": {"levels": list(LEVELS), "ranking": list(RANKINGS),
                  "cap": list(CAPS)},
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED},
        "rows": rows,
        "cells_compiled": [list(x) for x in done],
        "complete": len(done) == len(LEVELS) * len(RANKINGS),
        "published_pick_half_roc_auc": base["pick_half_roc_auc"] if base
        else None,
        "best_pick_half_roc_auc": best["pick_half_roc_auc"] if best else None,
        "best_configuration": {k: best[k] for k in
                               ("levels", "ranking", "cap")} if best else None,
        "range_over_all_settings": round(float(spread), 6),
        "published_is_best": bool(best and best["is_published_configuration"]),
        "selection_note": "the published configuration was frozen before this "
                          "sweep and is not revised by it, whichever row is "
                          "highest",
        "reads_test_fold": False,
    }


def _write(rows: list[dict]) -> dict:
    """Checkpoint after every compilation.

    Each (levels, ranking) pair costs a full recompilation of the cell array,
    and this machine has killed three long jobs for memory. Writing as we go
    means a kill costs one configuration rather than the sweep, and a re-run
    picks up where it stopped.
    """
    doc = _summarise(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
    return doc


def _resume() -> list[dict]:
    if not OUT.exists():
        return []
    d = json.loads(OUT.read_text())
    if d.get("schema") != SCHEMA or d.get("frozen_configuration") != FROZEN:
        return []
    rows = d.get("rows") or []
    if rows:
        done = sorted({(r["levels"], r["ranking"]) for r in rows})
        print(f"resuming: {len(rows)} settings already measured over "
              f"{len(done)} compilations", flush=True)
    return rows


def build() -> dict:
    with np.load(CACHE, allow_pickle=False) as z:
        y, n_res, ctr = z["y"], z["n_res_per"], z["ctr"]
        units = [str(u) for u in z["units"]]
        X = z["X"]
        n_wires = int(X.shape[1])
        _verify_equivalence(X[:4000], n_res)
        del X
    gc.collect()

    is_fit, fm, pm = _split(units, n_res)
    n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
    yfit, ypick, ctr_pick = y[fm], y[pm], ctr[pm]
    tables = partition_tables(n_wires, FROZEN["width"], FROZEN["rounds"], SEED)
    print(f"{n_wires} wires, {len(tables)} tables, fit {int(fm.sum())} rows, "
          f"pick {int(pm.sum())} rows", flush=True)

    rows = _resume()
    already = {(r["levels"], r["ranking"]) for r in rows}
    for levels in LEVELS:
        for ranking in RANKINGS:
            if (levels, ranking) in already:
                continue
            # The cap sweep varies only the fusion, so it rides along on the
            # published cell compilation rather than repeating it.
            caps = CAPS if (levels == FROZEN["levels"]
                            and ranking == FROZEN["ranking"]) \
                else (FROZEN["cap"],)
            t0 = time.perf_counter()
            # The wire matrix is 1.2 GB of float64 and is needed only to
            # produce the digits. Holding it through the K x K solve as well is
            # what pushed this machine into swap and had it kill the job, so it
            # is reloaded per configuration and dropped before the solve.
            X = _load_wires()
            D = digits(X, n_res, levels, ranking)
            del X
            gc.collect()
            Dfit = np.ascontiguousarray(D[fm])
            Dpick = np.ascontiguousarray(D[pm])
            del D
            gc.collect()
            offsets = cell_offsets(tables, levels)
            frac, tot = compile_cells(Dfit, yfit, tables, offsets, levels)
            empty = int((tot == 0).sum())
            for cap in caps:
                m = integer_fanout(Dfit, yfit, tables, offsets, frac,
                                   FROZEN["ridge"], cap, levels)
                S = score(Dpick, tables, offsets, frac, m, levels)
                raw = per_unit_auc(S, ypick, n_pick)
                gated, gname = _gated_auc(S, ctr_pick, n_pick, ypick)
                is_frozen = (levels == FROZEN["levels"]
                             and ranking == FROZEN["ranking"]
                             and cap == FROZEN["cap"])
                rows.append({
                    "levels": levels, "ranking": ranking, "cap": cap,
                    "ridge": FROZEN["ridge"], "n_cells": int(len(frac)),
                    "n_cells_never_addressed": empty,
                    "fraction_never_addressed": round(empty / len(frac), 6),
                    "total_fan_out": int(np.abs(m).sum()),
                    "n_tables_used": int((m != 0).sum()),
                    "pick_half_roc_auc_raw": raw,
                    "pick_half_roc_auc": gated,
                    "gate": gname,
                    "is_published_configuration": is_frozen,
                })
                mark = "  <- published" if is_frozen else ""
                print(f"  levels {levels}  {ranking:<12s} cap {cap:<3d} "
                      f"raw {raw:.4f}  gated {gated:.4f}  "
                      f"empty {100 * empty / len(frac):5.2f}%{mark}",
                      flush=True)
            del Dfit, Dpick, frac, tot
            gc.collect()
            _write(rows)
            print(f"    ({time.perf_counter() - t0:.0f}s, checkpointed)",
                  flush=True)

    return _write(rows)


def check() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    bad = []
    if d.get("schema") != SCHEMA:
        bad.append("unexpected schema")
    if d.get("reads_test_fold"):
        bad.append("this sweep must not read the held-out fold")
    if not d.get("complete"):
        bad.append("the sweep is a checkpoint, not a finished sweep: "
                   f"{len(d.get('cells_compiled') or [])} of "
                   f"{len(LEVELS) * len(RANKINGS)} compilations done")
    rows = d.get("rows") or []
    frozen = d.get("frozen_configuration") or {}
    published = [r for r in rows if r["is_published_configuration"]]
    if len(published) != 1:
        bad.append(f"{len(published)} rows claim to be the published "
                   f"configuration; exactly one must")
    elif any(published[0][k] != frozen[k] for k in ("levels", "ranking", "cap")):
        bad.append("the row marked published does not match the frozen "
                   "configuration")
    # The sweep is only about the shipped method if what it calls published is
    # what was in fact shipped. Both constants live elsewhere and could move
    # without this file noticing.
    if frozen.get("levels") != table_bank.N_LEVELS:
        bad.append(f"the sweep froze {frozen.get('levels')} levels but "
                   f"table_bank ships {table_bank.N_LEVELS}")
    if TABFIELD.exists():
        shipped = json.loads(TABFIELD.read_text())
        for key, ours in (("fan_out_cap", "cap"), ("ridge", "ridge")):
            if key in shipped and shipped[key] != frozen.get(ours):
                bad.append(f"the sweep froze {ours} {frozen.get(ours)} but the "
                           f"compiled field ships {key} {shipped[key]}")
    for k, seen in (("levels", {r["levels"] for r in rows}),
                    ("ranking", {r["ranking"] for r in rows}),
                    ("cap", {r["cap"] for r in rows})):
        want = set(d["swept"][k])
        if not want <= seen:
            bad.append(f"{k} {sorted(want - seen)} declared swept but absent")
    if rows:
        spread = (max(r["pick_half_roc_auc"] for r in rows)
                  - min(r["pick_half_roc_auc"] for r in rows))
        if abs(spread - d.get("range_over_all_settings", -1)) > 5e-6:
            bad.append("the reported range does not follow from the rows")
        best = max(rows, key=lambda r: r["pick_half_roc_auc"])
        if best["is_published_configuration"] != d.get("published_is_best"):
            bad.append("the artifact disagrees with its own rows about "
                       "whether the published configuration is the best one")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if bad:
        return 1
    print(f"OK {OUT.relative_to(ROOT)}: {len(rows)} settings, published "
          f"{d['published_pick_half_roc_auc']:.4f}, range "
          f"{d['range_over_all_settings']:.4f}, test fold unread")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        return check()
    doc = build()
    if not doc["complete"]:
        print("\nincomplete: not every (levels, ranking) pair was measured")
    print(f"\npublished {doc['published_pick_half_roc_auc']:.4f}, "
          f"best {doc['best_pick_half_roc_auc']:.4f} at "
          f"{doc['best_configuration']}, range "
          f"{doc['range_over_all_settings']:.4f}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
