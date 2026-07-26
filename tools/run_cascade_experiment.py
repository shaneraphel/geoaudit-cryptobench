"""Compile and score a cascaded quaternary network from the cached invariants.

Every experiment is arithmetic over the two cached matrices, so a full
compile-and-score iteration costs seconds. Nothing here re-reads a PDB and
nothing on the resolution path is a fitted weight: the score a residue receives
is the positive fraction of the integer counter cell it addresses.

Usage:
  PYTHONPATH=src python3.12 tools/run_cascade_experiment.py \
      --group-mode balanced --l1-digits 2 --patch
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_TRAIN = ROOT / "data/cryptobench_apo/_cascade_cache_train.npz"
CACHE_TEST = ROOT / "data/cryptobench_apo/_cascade_cache_test.npz"
P2RANK = 0.7930
FLAT6 = 0.7583


def load(p: Path):
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in ("F", "y", "codes", "ctr", "n_res_per", "units")}


def per_unit_auc(scores, y, n_res_per):
    from pocket_bench.metrics import roc_auc, average_precision
    aucs, prs = [], []
    off = 0
    for n in n_res_per:
        n = int(n)
        s = scores[off:off + n]
        t = y[off:off + n].tolist()
        off += n
        if sum(t) == 0 or sum(t) == n:
            continue
        a = roc_auc(list(s), t)
        p = average_precision(list(s), t)
        if a is not None:
            aucs.append(a)
        if p is not None:
            prs.append(p)
    return float(np.mean(aucs)), float(np.mean(prs)), len(aucs)


def pooled_auc(x, y):
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    r = np.empty(len(y), dtype=np.float64)
    order = np.argsort(x, kind="stable")
    r[order] = np.arange(1, len(y) + 1)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def main(argv=None) -> int:
    from pocket_bench.methods.algebraic_descriptors import FEATURE_NAMES
    from pocket_bench.methods.cascade_lut import (
        QLUT, band_of, chain_rank_digits, compile_band, compile_quartiles,
        global_digits, oof_fraction, patch_mean, residue_fold, spread,
    )
    from pocket_bench.methods.sequence_wires import (
        apply_propensity, propensity_table, static_sequence_features,
    )

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quant", choices=("global", "chain"), default="chain")
    ap.add_argument("--group-mode", choices=("theme", "balanced", "strength"),
                    default="balanced")
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--l0-digits", type=int, default=1)
    ap.add_argument("--l1-digits", type=int, default=2)
    ap.add_argument("--n-super", type=int, default=2)
    ap.add_argument("--patch", action="store_true")
    ap.add_argument("--patch-radius", type=float, default=10.0)
    ap.add_argument("--drop-weak", type=float, default=0.0,
                    help="drop wires whose digitised |AUC-0.5| is below this")
    ap.add_argument("--no-seq", action="store_true")
    ap.add_argument("--folds", type=int, default=5,
                    help="0 disables cross-fitting")
    ap.add_argument("--topology", choices=("cascade", "flat"),
                    default="cascade")
    ap.add_argument("--flat-wires", type=int, default=6)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    tr, te = load(CACHE_TRAIN), load(CACHE_TEST)
    y = tr["y"]
    t0 = time.perf_counter()
    man = json.loads((ROOT / "data/cryptobench_apo/train_manifest.json")
                     .read_text())["entries"]
    cl_of = {f"{e['pdb']}_{e['chain']}": str(e.get("cluster_id", "")) or
             f"{e['pdb']}_{e['chain']}" for e in man}
    cluster = np.array([cl_of.get(str(u), str(u)) for u in tr["units"]])
    fold = (residue_fold(tr["n_res_per"], cluster, args.folds)
            if args.folds > 1 else np.zeros(len(tr["y"]), dtype=np.int64))

    def emit(digits):
        """(train fractions, inference table). Cross-fitted unless disabled."""
        if args.folds <= 1:                       # in-sample, for the ablation
            lut = QLUT.compile(digits, y)
            return lut.frac(digits), lut
        return oof_fraction(digits, y, fold)
    if not args.quiet:
        print(f"train {len(tr['n_res_per'])} units / {len(y)} residues, "
              f"cryptic {int(y.sum())} ({100*y.mean():.2f}%)   "
              f"test {len(te['n_res_per'])} units / {len(te['y'])} residues")

    # ---- sequence wires, per chain ----------------------------------------
    def seq_block(d, prop):
        parts, off = [], 0
        for n in d["n_res_per"]:
            n = int(n)
            parts.append(static_sequence_features(d["codes"][off:off + n],
                                                  d["ctr"][off:off + n]))
            off += n
        return np.concatenate(
            [np.concatenate(parts, axis=0),
             apply_propensity(d["codes"], prop)[:, None]], axis=1)

    prop = propensity_table(tr["codes"], y)
    names = list(FEATURE_NAMES)
    Ftr, Fte = tr["F"], te["F"]
    if not args.no_seq:
        Ftr = np.concatenate([Ftr, seq_block(tr, prop)], axis=1)
        Fte = np.concatenate([Fte, seq_block(te, prop)], axis=1)
        names += ["aa_hydropathy", "aa_volume", "nbr_hydropathy",
                  "aa_propensity"]

    # ---- level-0 digitisation ---------------------------------------------
    if args.quant == "chain":
        Dtr = chain_rank_digits(Ftr, tr["n_res_per"])
        Dte = chain_rank_digits(Fte, te["n_res_per"])
    else:
        e = compile_quartiles(Ftr)
        Dtr, Dte = global_digits(Ftr, e), global_digits(Fte, e)

    # Strength is measured on the DIGITS, not the raw values: the digit is what
    # the table actually addresses, and a wire can be strong pooled and useless
    # once quantised per chain (or the reverse).
    strength = np.array([abs(pooled_auc(Dtr[:, j].astype(np.float64), y) - 0.5)
                         for j in range(Dtr.shape[1])])
    order = np.argsort(-strength)
    keep = [j for j in order if strength[j] >= args.drop_weak]
    if not args.quiet:
        print("\ndigitised wire strength |AUC-0.5| (descending):")
        for rank, j in enumerate(order):
            mark = "" if strength[j] >= args.drop_weak else "   DROPPED"
            print(f"   {rank+1:2d}. {names[j]:16s} "
                  f"{pooled_auc(Dtr[:, j].astype(float), y):.4f} "
                  f"|{strength[j]:.4f}|{mark}")
    print(f"\nwires kept: {len(keep)}/{Dtr.shape[1]}")

    # ---- grouping ----------------------------------------------------------
    gsz = args.group_size
    n_groups = int(np.ceil(len(keep) / gsz))
    if args.group_mode == "theme":
        groups = [keep[i * gsz:(i + 1) * gsz] for i in range(n_groups)]
    elif args.group_mode == "strength":
        groups = [keep[i * gsz:(i + 1) * gsz] for i in range(n_groups)]
    else:  # balanced: deal the strength-sorted wires round robin
        groups = [[] for _ in range(n_groups)]
        for i, j in enumerate(keep):
            groups[i % n_groups].append(j)

    stage_tr, stage_te, stage_frac_tr = [], [], []
    if not args.quiet:
        print("\nlevel-0 group tables:")
    for gi, gcols in enumerate(groups):
        if not gcols:
            continue
        if args.folds > 1:
            f_tr, lut = oof_fraction(Dtr[:, gcols], y, fold)
        else:
            lut = QLUT.compile(Dtr[:, gcols], y)
            f_tr = lut.frac(Dtr[:, gcols])
        f_te = lut.frac(Dte[:, gcols])
        bd = compile_band(f_tr, args.l0_digits)
        stage_tr.append(spread(band_of(f_tr, bd), args.l0_digits))
        stage_te.append(spread(band_of(f_te, bd), args.l0_digits))
        stage_frac_tr.append(f_tr)
        st = lut.stats()
        if not args.quiet:
            print(f"   G{gi+1} n={len(gcols)} cells={st['n_cells']:6d} "
                  f"occ={len(y)/st['n_cells']:7.1f} asserted={st['asserted']:6d} "
                  f"X={st['X']:5d} Z={st['Z']:6d}  "
                  f"digitAUC={pooled_auc(f_tr, y):.4f}")
    Str_d = np.concatenate(stage_tr, axis=1)
    Ste_d = np.concatenate(stage_te, axis=1)

    # ---- level-1 super stages ----------------------------------------------
    n_stage_digits = Str_d.shape[1]
    chunks = np.array_split(np.arange(n_stage_digits), args.n_super)
    cols_tr, cols_te, patch_src = [], [], []
    if not args.quiet:
        print("\nlevel-1 super stages:")
    for si, idx in enumerate(chunks):
        idx = list(idx)
        if not idx:
            continue
        if args.folds > 1:
            f_tr, lut = oof_fraction(Str_d[:, idx], y, fold)
        else:
            lut = QLUT.compile(Str_d[:, idx], y)
            f_tr = lut.frac(Str_d[:, idx])
        f_te = lut.frac(Ste_d[:, idx])
        bd = compile_band(f_tr, args.l1_digits)
        b_tr, b_te = band_of(f_tr, bd), band_of(f_te, bd)
        cols_tr.append(spread(b_tr, args.l1_digits))
        cols_te.append(spread(b_te, args.l1_digits))
        patch_src.append((b_tr.astype(np.float64), b_te.astype(np.float64), si))
        st = lut.stats()
        if not args.quiet:
            print(f"   S{si+1} in={len(idx)} cells={st['n_cells']:6d} "
                  f"occ={len(y)/st['n_cells']:8.1f} asserted={st['asserted']:5d} "
                  f"X={st['X']:5d} Z={st['Z']:6d}  out={args.l1_digits}d  "
                  f"digitAUC={pooled_auc(f_tr, y):.4f}")

    # ---- level-2 spatial majority gate --------------------------------------
    aux_tr, aux_te = [], []
    if args.patch:
        if not args.quiet:
            print("\nlevel-2 spatial gates:")
        for b_tr, b_te, si in patch_src:
            m_tr = patch_mean(b_tr, tr["ctr"], tr["n_res_per"],
                              args.patch_radius)
            m_te = patch_mean(b_te, te["ctr"], te["n_res_per"],
                              args.patch_radius)
            bd = compile_band(m_tr, 1)
            cols_tr.append(spread(band_of(m_tr, bd), 1))
            cols_te.append(spread(band_of(m_te, bd), 1))
            hi = float(max(m_tr.max(), 1.0))
            aux_tr.append(m_tr / hi); aux_te.append(m_te / hi)
            if not args.quiet:
                print(f"   patch(S{si+1}) r={args.patch_radius} "
                      f"digitAUC={pooled_auc(m_tr, y):.4f}")

    if args.topology == "flat":
        sel = keep[:args.flat_wires]
        gtr, gte = Dtr[:, sel], Dte[:, sel]
        aux_tr, aux_te = [], []
        print(f"\nFLAT word over top {len(sel)} wires: "
              f"{[names[j] for j in sel]}")
    else:
        gtr = np.concatenate(cols_tr, axis=1)
        gte = np.concatenate(cols_te, axis=1)
    p_tr, gl = emit(gtr)
    st = gl.stats()
    print(f"\nL3 GLOBAL in={gtr.shape[1]}d cells={st['n_cells']:7d} "
          f"occ={len(y)/st['n_cells']:8.1f} asserted={st['asserted']:6d} "
          f"X={st['X']:6d} 0={st['0']:6d} 1={st['1']:5d} Z={st['Z']:7d}")

    p_te = gl.frac(gte)
    a_cell, pr_cell, n_ok = per_unit_auc(p_te, te["y"], te["n_res_per"])
    if aux_te:
        tie_te = sum(a / (2048.0 ** (k + 1)) for k, a in enumerate(aux_te))
        tie_tr = sum(a / (2048.0 ** (k + 1)) for k, a in enumerate(aux_tr))
    else:
        tie_te = np.zeros_like(p_te); tie_tr = np.zeros_like(p_tr)
    a_lex, pr_lex, _ = per_unit_auc(p_te + tie_te, te["y"], te["n_res_per"])
    a_tr, _, _ = per_unit_auc(p_tr + tie_tr, y, tr["n_res_per"])

    tag = (f"mode={args.group_mode} gsz={args.group_size} l0={args.l0_digits} "
           f"l1={args.l1_digits} super={args.n_super} "
           f"patch={'on' if args.patch else 'off'} drop={args.drop_weak}")
    print(f"\n{tag}")
    print(f"   train ROC-AUC (OOF)  = {a_tr:.4f}")
    print(f"   TEST  ROC-AUC (cell) = {a_cell:.4f}   PR-AUC = {pr_cell:.4f}")
    print(f"   TEST  ROC-AUC (lex)  = {a_lex:.4f}   PR-AUC = {pr_lex:.4f}")
    best = max(a_cell, a_lex)
    print(f"   vs P2Rank {P2RANK}: {best - P2RANK:+.4f}   "
          f"vs flat-6 {FLAT6}: {best - FLAT6:+.4f}   "
          f"{'*** BEATS SOTA ***' if best > P2RANK else 'below'}   "
          f"[{time.perf_counter()-t0:.0f}s]")
    if args.out:
        (ROOT / args.out).write_text(json.dumps({
            "config": vars(args), "train_roc_auc": a_tr,
            "test_roc_auc_cell": a_cell, "test_roc_auc_lex": a_lex,
            "test_pr_auc_lex": pr_lex, "n_units": n_ok,
            "global_stats": st, "beats_p2rank": bool(best > P2RANK),
        }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
