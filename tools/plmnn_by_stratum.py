#!/usr/bin/env python3
"""Does the supervised sequence baseline collapse on small pockets too?

THIS TOOL CANNOT ANSWER THAT, AND THE REASON IS THE POINT
---------------------------------------------------------
Read this before running anything below. The plan was to run pLM-NN over the
training fold, where scoring costs nothing from the test-fold ledger, and
stratify it by cryptic-pocket size. The cost argument was correct. It was also
doing the work of a validity argument that nobody had made.

**pLM-NN's head is CryptoBench's published ``best_trained``, fitted by the
authors on their training folds, and ``TRAIN_MANIFEST.json`` records our
training partition as exactly those folds --- ``train-0`` through ``train-3``.
The 770 chains here are the model's own fitting set.** It scores 0.8235 mean
per-unit ROC-AUC on the official test fold and about 0.98 on these. A model
evaluated on what it was fitted on does not collapse on a hard stratum; it has
memorised it, and the profile that came out would have described fit rather than
generalisation while looking exactly like an answer.

The tell was in the first five lines of output --- 0.9946, 0.9551, 0.9760,
0.9907, 0.9783, for a method whose published number on this benchmark is 0.82.
A rival that suddenly looks far better than its own paper is measuring something
else. The check was one field in a manifest and one number already on disk, and
it was available before the run rather than four minutes into it.

``--embed`` and ``--fit-set-audit`` are retained because the audit is worth
having as an artifact: it is the evidence for ``AGENT_MEMORY`` 2h-bis and for the
refusal in ``found_where_three_baselines_missed.py``, which would otherwise have
turned a memorising baseline into a recovery count of zero and read it as a
result about our method. ``--stratify`` is retained and refuses.

**The question in the title is still open.** What is closed is this route to it.
Three remain, none free: retrain the published architecture on a split excluding
the units scored, which makes it our model rather than the published baseline and
needs its own preregistration; spend an external set, which destroys the
confirmatory result it exists for; or stratify the official-fold comparison,
which is a ledger read for an exploratory question and is the thing 2h's own
method note records avoiding.

What follows is the original argument, kept because the reasoning was sound
everywhere except at the step nobody checked.

The question, and why it looked like the one worth the compute
---------------------------------------------------------------
``AGENT_MEMORY`` 2h settled that the small-pocket stratum is not headroom for
*this* detector: PocketMiner, a graph network sharing no architecture, no
featurisation and no fitting procedure with the counting field, scores 0.5985 on
the 0-9 stratum against our 0.5958, and its own fall from largest stratum to
smallest is -0.2008 against our -0.2796. It closed by naming exactly one thing it
does not settle:

    It does not locate the pLM-NN deficit. pLM-NN reads evolutionary information
    that neither method here has, and whether *its* profile also collapses on 0-9
    is unmeasured. That is now the question worth the compute.

Both answers change what to build next, which is the property a measurement
should have before it is run.

*If pLM-NN also sits near 0.60 on the small stratum*, its whole 0.0243 advantage
lives in the strata where we are already 0.076 ahead of PocketMiner. That is a
narrow and specific target: it says the deficit is not about the units we are
worst on, and it puts the remaining work on units the detector already handles
well, where a wire family has room to matter.

*If pLM-NN does not collapse there*, then a sequence model reads something on
small-pocket units that two structural methods both miss, 2h's reading that the
stratum is intrinsically hard is wrong, and the tail is headroom after all --
for a method with evolutionary input. That would reopen a closed direction,
which is worth more than confirming an open one.

What this costs, and what it does not
--------------------------------------
It costs one pass of ESM2-3B over the 770 training chains, about four and a half
hours on this machine at float16, and it costs **no read of the test fold**. The
tempting version of this measurement was to stratify the existing test-fold
comparison, which would spend a read from the ledger on an exploratory question;
2h records that mistake being avoided once already and it is avoided again here.
The encoder weights are the same checkpoint the test-fold run used, by sha256.

Why this is a separate file
---------------------------
``plmnn_embed.py`` and ``plmnn_sequences.py`` produce ``PLMNN_SCORES.json``,
which the manuscript cites. AGENTS.md's rule is that a tool feeding a pinned
artifact is not edited for an unrelated reason, so their model loading, their
embedding and their sequence construction are *imported* here and nothing in
either file changes. The one part that is copied rather than imported is the
recomputation of our own per-unit AUC, which lives inside
``baseline_by_stratum.main``: factoring it out would edit the tool that produces
``BASELINE_BY_STRATUM.json``. The copy carries the same reproduction check
against the frozen per-split means, so a copy that has drifted raises instead of
being compared.

Phases
------
``--embed`` runs the encoder and writes one line per chain to a checkpoint, so an
interrupted run resumes. ``--stratify`` joins those scores against our field and
against PocketMiner and writes the artifact. They are separate because the first
takes hours and the second takes minutes, and a mistake in the second should not
cost the first again.

Nothing here reads the test fold or any external unit.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR / "src"), str(ROOT_DIR / "tools")]

from pocket_bench.paths import ROOT                              # noqa: E402
from plmnn_sequences import THREE_TO_ONE, _chain_residues        # noqa: E402
from failure_tail import auc_per_unit                            # noqa: E402
from baseline_by_stratum import (                                # noqa: E402
    paired_stats, pocketminer_per_unit, strata_edges,
)

SCHEMA = "geoaudit.plmnn_by_stratum.v1"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
LABELS = ROOT / "data/cryptobench_apo/train_labels"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
CKPT = ROOT / "results/baselines/_plmnn_train_checkpoint.jsonl"
OUT = ROOT / "results/architecture_sweep/PLMNN_BY_STRATUM.json"

# Written before the run, and left here whatever the run says. 2h's own
# prediction block is the model for this: a prediction that is edited after the
# fact is a description.
PREDICTION = {
    "committed_before_the_run": True,
    "expected": (
        "pLM-NN also collapses on the 0-9 stratum, to somewhere between 0.60 "
        "and 0.68, because a unit with seven cryptic residues in three hundred "
        "is an ambiguous labelling problem before it is a detection problem and "
        "that is not a property of the input representation"),
    "if_it_collapses": (
        "the 0.0243 deficit lives in the strata where we already lead "
        "PocketMiner by 0.076, and the work goes to units the detector handles "
        "well rather than to the tail"),
    "if_it_does_not": (
        "evolutionary input reads something on small-pocket units that two "
        "structural methods both miss, 2h's reading is wrong, and the tail is "
        "headroom for a method with that input"),
    "what_would_make_this_unreadable": (
        "a join that loses the small stratum. Under ten units in a stratum and "
        "the block is omitted rather than reported, which is the rule "
        "baseline_by_stratum.py already applies"),
}


def _resnum(x) -> int | None:
    """The trailing integer of a residue key, or None.

    Copied from ``baseline_by_stratum.py`` for the reason recorded there: a
    change to the label reader elsewhere must not silently move this
    comparison's universe.
    """
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


def _labels(unit: str) -> set[int]:
    p = LABELS / f"{unit}_labels.json"
    if not p.is_file():
        raise SystemExit(f"no training labels for {unit}")
    d = json.loads(p.read_text())
    raw = d.get("cryptic_residues") or d.get("binding_residues") or []
    return {r for r in (_resnum(r) for r in raw) if r is not None}


def _sequences() -> list[dict]:
    """One row per training chain: sequence, and the resseq each row belongs to.

    Built by the same two functions ``plmnn_sequences.py`` uses on the official
    fold, so the sequence a chain becomes and the mapping back to residues are
    identical on both folds by construction rather than by inspection.
    """
    rows = []
    for e in json.loads(MANIFEST.read_text())["entries"]:
        uid = f"{e['pdb']}_{e['chain']}"
        res = _chain_residues((ROOT / e["receptor_path"]).read_text(),
                              e["chain"])
        rows.append({
            "unit_id": uid,
            "sequence": "".join(THREE_TO_ONE.get(n, "X") for _, _, n in res),
            "resseq_per_row": [r for r, _, _ in res],
        })
    rows.sort(key=lambda r: r["unit_id"])
    return rows


def _done() -> dict[str, dict]:
    if not CKPT.exists():
        return {}
    out = {}
    for line in CKPT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["unit_id"]] = r
    return out


def embed(limit: int = 0) -> int:
    """Run the encoder and the trained head over the training fold."""
    from plmnn_embed import LAYER, _embed, _model, forward, load_head

    rows = _sequences()
    if limit:
        rows = rows[:limit]
    head = load_head()
    done = _done()
    model = batch = None
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, r in enumerate(rows):
        if r["unit_id"] in done:
            continue
        if model is None:
            print("loading ESM2-3B", flush=True)
            model, batch = _model()
        t1 = time.time()
        rep = _embed(model, batch, r["sequence"], [LAYER])[LAYER]
        p = forward(rep.astype(np.float64), head)[:, 1]
        # The same tie rule the test-fold run uses: where two embedding rows
        # share one resseq the larger probability is kept, which is the reading
        # most favourable to the rival.
        by_res: dict[int, float] = {}
        for resseq, prob in zip(r["resseq_per_row"], p):
            by_res[resseq] = max(by_res.get(resseq, -1.0), float(prob))
        pos = _labels(r["unit_id"])
        keys = sorted(by_res)
        s = np.array([by_res[q] for q in keys], dtype=np.float64)
        y = np.array([q in pos for q in keys], dtype=np.int64)
        n_p = int(y.sum())
        auc = (float("nan") if n_p in (0, len(y))
               else float(auc_per_unit(s, y, np.array([len(y)]))[0]))
        rec = {"unit_id": r["unit_id"], "n_res": len(keys), "n_pos": n_p,
               "auc": auc, "seconds": round(time.time() - t1, 3)}
        with CKPT.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[r["unit_id"]] = rec
        print(f"{k + 1}/{len(rows)} {r['unit_id']} {len(keys)} residues "
              f"auc {auc:.4f} {rec['seconds']:.1f}s "
              f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)
    print(f"\n{len(done)} chains scored", flush=True)
    return 0


def _ours_per_unit(n_splits: int) -> tuple[dict[str, float], np.ndarray,
                                           list[str], np.ndarray, float]:
    """Our field's mean per-unit AUC over the splits each unit sits out on.

    Copied from ``baseline_by_stratum.main`` rather than imported, because
    factoring it out would edit the tool that produces a cited artifact. The
    reproduction check below is what makes the copy safe: it recomputes the
    frozen per-split means and refuses to return if they have moved.
    """
    import digit_cache
    from baseline_by_stratum import (
        COUNTING, FAN_OUT_CAP, PARTITION_ROUNDS, PARTITION_SEED, RIDGE, SEED,
        TABLE_WIDTH, apply_gate, cell_offsets, cluster_half_split,
        compile_cells, lean_integer_fanout, partition_tables, score,
    )

    cdoc = json.loads(COUNTING.read_text())
    n_splits = n_splits or int(cdoc["protocol"]["n_splits"])
    by_width = {int(k.split()[-2]): v for k, v in cdoc["per_split"].items()}

    z = np.load(WIDE, allow_pickle=False)
    y, n_res, ctr = z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    z.close()
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"]
                  for e in json.loads(MANIFEST.read_text())["entries"]}

    D = digit_cache.load(n_res)
    n_wires = int(D.shape[1])
    frozen = np.asarray(by_width[n_wires], dtype=float)[:n_splits]
    tabs = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                            PARTITION_SEED)
    offs = cell_offsets(tabs)
    row = np.repeat(np.arange(len(n_res)), n_res)

    n_pos = np.zeros(len(n_res), dtype=np.int64)
    off = 0
    for i, n in enumerate(n_res):
        n = int(n)
        n_pos[i] = int((y[off:off + n] == 1).sum())
        off += n

    seen = np.zeros(len(n_res), dtype=np.int64)
    total = np.zeros(len(n_res), dtype=np.float64)
    per_split: list[float] = []
    t0 = time.perf_counter()
    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        Dfit = D[fit]
        frac, _t = compile_cells(Dfit, y[fit], tabs, offs)
        mult = lean_integer_fanout(Dfit, y[fit], tabs, offs, frac, RIDGE,
                                   FAN_OUT_CAP)
        del Dfit
        gated = apply_gate(score(D[pick], tabs, offs, frac, mult),
                           ctr[pick], n_pick)
        per = auc_per_unit(gated, y[pick], n_pick)
        idx = np.flatnonzero(~is_fit)
        ok = ~np.isnan(per)
        seen[idx[ok]] += 1
        total[idx[ok]] += per[ok]
        per_split.append(float(np.nanmean(per)))
        print(f"  split {s + 1}/{n_splits}  ours {per_split[-1]:.4f}  "
              f"frozen {frozen[s]:.4f}  {time.perf_counter() - t0:.0f}s",
              flush=True)

    repro = float(np.abs(np.asarray(per_split) - frozen).max())
    if repro >= 5e-4:
        raise SystemExit(
            f"our recomputed per-split means differ from the frozen ones by "
            f"{repro:.2e}; this copy of the deployed pipeline has drifted and "
            f"the comparison would not be against the detector that ships")

    ours = {u: float(total[i] / seen[i]) for i, u in enumerate(units)
            if seen[i] > 0}
    return ours, n_pos, units, n_res, repro


OFFICIAL_PLMNN_AUC = 0.823469   # results/official_fold/PLMNN_READ.json
FIT_SET = ROOT / "results/architecture_sweep/PLMNN_TRAIN_IS_ITS_OWN_FIT_SET.json"


def fit_set_audit(write: bool) -> int:
    """Compare pLM-NN on chains it was fitted on against its published number.

    This is the whole of what the training-fold pass can honestly produce. It is
    written as an artifact rather than left in a commit message because two other
    tools refuse on the strength of it, and a refusal whose evidence is not on
    disk is an assertion.
    """
    rows = list(_done().values())
    aucs = sorted(r["auc"] for r in rows if r["auc"] == r["auc"])
    if len(aucs) < 20:
        raise SystemExit(
            f"only {len(aucs)} chains scored; run --embed --limit 60 first. "
            f"Twenty is the floor at which the gap below is worth writing down")
    n = len(aucs)
    mean = sum(aucs) / n
    doc = {
        "schema": "geoaudit.plmnn_train_is_its_own_fit_set.v1",
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": (
            "whether pLM-NN can be run on our training fold to answer an "
            "exploratory question about it, given that the published head was "
            "fitted somewhere"),
        "answer": "no, because it was fitted here",
        "why": {
            "head": ("CryptoBench's published best_trained, read out of the "
                     "authors' SavedModel; see results/baselines/"
                     "PLMNN_NETWORK.json"),
            "our_training_partition": ("data/cryptobench_apo/TRAIN_MANIFEST.json "
                                       "records fold = train-0..train-3, which "
                                       "are the folds that head was fitted on"),
            "so": "every chain scored below is in the model's own fitting set",
        },
        "measured": {
            "n_chains_scored": n,
            "mean_per_unit_roc_auc_on_its_fit_set": round(mean, 6),
            "median": round(aucs[n // 2], 6),
            "min": round(aucs[0], 6),
            "max": round(aucs[-1], 6),
            "n_at_or_above_0_95": sum(1 for a in aucs if a >= 0.95),
            "n_below_0_80": sum(1 for a in aucs if a < 0.80),
            "published_mean_on_the_official_test_fold": OFFICIAL_PLMNN_AUC,
            "gap": round(mean - OFFICIAL_PLMNN_AUC, 6),
            "source_of_the_official_number":
                "results/official_fold/PLMNN_READ.json, reproduction_gate",
        },
        "why_the_sample_is_partial": (
            f"the run was stopped at {n} chains of 770 once the gap was "
            f"established. Continuing would have cost four more hours to "
            f"measure a quantity that cannot be used, and the gap is not a "
            f"marginal effect that a larger sample could overturn"),
        "why_the_spread_does_not_rescue_it": (
            "not every chain is memorised equally and some sit well below the "
            "mean, so the distribution is reported rather than the mean alone. "
            "That does not make the set usable: the recovery rule and the "
            "stratification both ask where a baseline is weak, and a fit set "
            "answers where it happened to fit less well, which is a different "
            "question wearing the same units"),
        "what_this_is_evidence_for": [
            "docs/AGENT_MEMORY.md section 2h-bis",
            "the refusal in tools/found_where_three_baselines_missed.py",
            "the refusal in tools/plmnn_by_stratum.py --stratify",
        ],
        "what_remains_open": (
            "whether pLM-NN's advantage is concentrated in the large-pocket "
            "strata. Routes: retrain the architecture on a split excluding the "
            "units scored, which makes it our model and needs its own "
            "preregistration; spend an external set, which destroys the "
            "confirmatory result; or stratify the official-fold comparison, "
            "which is a ledger read for an exploratory question"),
    }
    print(f"pLM-NN on {n} chains of its own fitting set")
    print(f"  mean   {mean:.4f}   median {aucs[n // 2]:.4f}   "
          f"min {aucs[0]:.4f}   max {aucs[-1]:.4f}")
    print(f"  at or above 0.95: {doc['measured']['n_at_or_above_0_95']}"
          f"   below 0.80: {doc['measured']['n_below_0_80']}")
    print(f"  published on the official test fold: {OFFICIAL_PLMNN_AUC:.4f}")
    print(f"  gap: {mean - OFFICIAL_PLMNN_AUC:+.4f}")
    if write:
        FIT_SET.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        print(f"\nwrote {FIT_SET.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


def stratify(n_splits: int, out: str, write: bool) -> int:
    raise SystemExit(
        "--stratify will not run. The scores in the checkpoint are pLM-NN "
        "evaluated on its own fitting set: the head is CryptoBench's published "
        "best_trained, fitted on train-0..train-3, and TRAIN_MANIFEST.json "
        "records our training partition as exactly those folds. It scores "
        f"{OFFICIAL_PLMNN_AUC:.4f} on the official test fold and about 0.98 "
        "here. A stratification of a fit set describes where a model happened "
        "to fit less well, not where it is weak, and the two are not "
        "distinguishable after the fact. Run --fit-set-audit for the evidence, "
        "and read AGENT_MEMORY 2h-bis for what would make the real measurement "
        "possible.")


def _stratify_unreachable(n_splits: int, out: str, write: bool) -> int:
    plm = _done()
    if len(plm) < 100:
        raise SystemExit(
            f"only {len(plm)} chains are in the checkpoint; run --embed first")

    ours, n_pos, units, n_res, repro = _ours_per_unit(n_splits)
    pm = pocketminer_per_unit()
    edges = strata_edges()
    idx_of = {u: i for i, u in enumerate(units)}

    rows, absent_plm, absent_pm, mism = [], [], [], []
    for u, o in sorted(ours.items()):
        r = plm.get(u)
        if r is None:
            absent_plm.append(u)
            continue
        i = idx_of[u]
        if int(r["n_res"]) != int(n_res[i]) or int(r["n_pos"]) != int(n_pos[i]):
            mism.append(u)
            continue
        if np.isnan(r["auc"]):
            continue
        q = pm.get(u)
        if q is None or np.isnan(q["auc"]):
            absent_pm.append(u)
        rows.append({"unit": u, "ours": o, "plm": float(r["auc"]),
                     "pm": (None if q is None or np.isnan(q["auc"])
                            else float(q["auc"])),
                     "n_pos": int(n_pos[i])})

    def block(sel: list[dict]) -> dict:
        o = np.array([r["ours"] for r in sel], float)
        p = np.array([r["plm"] for r in sel], float)
        npos = np.array([r["n_pos"] for r in sel], int)
        by = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (npos >= lo) & (npos < hi)
            if m.sum() < 10:
                continue
            by.append({
                "n_cryptic_from": int(lo), "n_cryptic_below": int(hi),
                "n_units": int(m.sum()),
                "ours_mean_auc": round(float(o[m].mean()), 6),
                "plmnn_mean_auc": round(float(p[m].mean()), 6),
                "paired_ours_minus_plmnn": paired_stats(o[m] - p[m]),
            })
        first, last = by[0], by[-1]
        return {
            "n_units": len(sel),
            "ours_pooled_mean_auc": round(float(o.mean()), 6),
            "plmnn_pooled_mean_auc": round(float(p.mean()), 6),
            "paired_pooled": paired_stats(o - p),
            "by_stratum": by,
            "profile_across_strata": {
                "ours_small_minus_large": round(
                    first["ours_mean_auc"] - last["ours_mean_auc"], 6),
                "plmnn_small_minus_large": round(
                    first["plmnn_mean_auc"] - last["plmnn_mean_auc"], 6),
                "gap_small_minus_gap_large": round(
                    first["paired_ours_minus_plmnn"]["mean"]
                    - last["paired_ours_minus_plmnn"]["mean"], 6),
                "how_to_read_it": (
                    "if plmnn_small_minus_large is close to "
                    "ours_small_minus_large then both methods fall by the same "
                    "amount on small pockets and the deficit does not live "
                    "there; if it is much smaller then the sequence model "
                    "holds up where the structural methods do not"),
            },
        }

    three = [r for r in rows if r["pm"] is not None]
    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": (
            "whether CryptoBench's own supervised sequence baseline collapses "
            "on training units with few cryptic residues in the way this "
            "detector and PocketMiner both do, which locates where its "
            "official-fold advantage is earned"),
        "why_now": (
            "BASELINE_BY_STRATUM.json closes by naming this as the one thing "
            "it does not settle and as the question worth the compute"),
        "prediction": PREDICTION,
        "rival": {
            "method": "pLM-NN, CryptoBench's own baseline, rebuilt here",
            "encoder": "esm2_t36_3B_UR50D, layer 33, float16",
            "weights": "results/baselines/PLMNN_WEIGHTS.npz",
            "tie_rule": "max over the embedding rows sharing a resseq",
            "biases_and_their_direction": (
                "the tie rule favours the rival; the network was fitted by the "
                "benchmark's authors on this fold's own training partition, "
                "which favours the rival on these units and is the reason this "
                "is a reading about where its advantage lives and not a "
                "head-to-head accuracy claim"),
        },
        "protocol": {
            "n_splits": n_splits,
            "ours": (
                "mean within-unit ROC-AUC over the splits in which the unit "
                "sits on the pick side, gate applied as deployed"),
            "plmnn": "one forward pass per chain; no split, no fitting here",
            "why_that_is_comparable": (
                "both are within-unit rankings over the same residue universe, "
                "and the join checks residue and positive counts agree before "
                "either is subtracted"),
            "strata_from": "results/architecture_sweep/FAILURE_TAIL.json",
        },
        "join": {
            "n_units_ours": len(ours),
            "n_units_joined": len(rows),
            "n_units_plmnn_absent": len(absent_plm),
            "n_units_pocketminer_absent": len(absent_pm),
            "n_units_count_disagrees": len(mism),
            "units_where_count_disagrees": sorted(mism)[:20],
        },
        "all_joined_units": block(rows),
        "units_all_three_methods_cover": block(three) if three else None,
        "reproduction_check": {
            "max_absolute_difference_from_frozen_per_split": round(repro, 8),
            "reproduces_the_frozen_arm": bool(repro < 5e-4),
        },
    }

    b = doc["all_joined_units"]
    print(f"\n{'stratum':>10} {'n':>5} {'ours':>8} {'pLM-NN':>8} {'diff':>9}")
    for s in b["by_stratum"]:
        d = s["paired_ours_minus_plmnn"]
        print(f"{s['n_cryptic_from']:>4}-{s['n_cryptic_below'] - 1:<5} "
              f"{s['n_units']:>5} {s['ours_mean_auc']:>8.4f} "
              f"{s['plmnn_mean_auc']:>8.4f} {d['mean']:>+9.4f}")
    pf = b["profile_across_strata"]
    print(f"\nfall from largest stratum to smallest: ours "
          f"{pf['ours_small_minus_large']:+.4f}, pLM-NN "
          f"{pf['plmnn_small_minus_large']:+.4f}")

    if write:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {p.relative_to(ROOT)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--embed", action="store_true",
                    help="run the encoder over the training fold (hours)")
    ap.add_argument("--stratify", action="store_true",
                    help="refuses; the checkpoint is the model's own fit set")
    ap.add_argument("--fit-set-audit", action="store_true",
                    help="write the evidence that it is")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--splits", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    if a.embed:
        return embed(a.limit)
    if a.stratify:
        return stratify(a.splits, a.out, a.write)
    if a.fit_set_audit:
        return fit_set_audit(a.write)
    ap.error("choose --embed, --fit-set-audit or --stratify")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
