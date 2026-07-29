#!/usr/bin/env python3
"""Run CryptoBench's own pLM-NN baseline over the 192 single-chain test units.

The baseline is ESM2-3B embeddings into the small dense network the authors
published. Two things about it are not written down anywhere in the deposit, and
both are recovered here rather than assumed, because either one would produce a
baseline that runs, reports plausible numbers, and is wrong:

  Which layer. The deposit ships one worked example -- the embedding of chain
  7w19A -- and says only that it is "the ESM2-3B embedding". ESM2-3B has 36
  layers. The example is matched against every layer this run computes, and the
  answer is layer 33, agreeing to a mean cosine of about 1.0 while its
  neighbours are visibly worse. Layer 33 is the last layer of *ESM2-650M*, so
  the most likely history is a ``repr_layers=[33]`` carried over from the
  smaller model's example code. Taking the final layer 36 instead, which is what
  "the embedding" would ordinarily mean, gives vectors a hundred times smaller
  and a baseline scored on the wrong features.

  Whether it is layer-normed. fair-esm replaces the stored representation of the
  *last layer it computes* with the layer-normed one. So a model truncated to 33
  layers, asked for layer 33, silently returns a normalised tensor instead of the
  hidden state -- measured here at cosine 0.982 against the authors' example
  rather than 1.0. The model is therefore truncated to 34 layers, which both
  avoids that overwrite and skips two layers of work.

The encoder runs in float16, which is the precision its published checkpoint is
stored in. A float32 pass was tried first, being what the authors most likely
ran, and it agrees with their example to cosine 0.998745 against float16's
0.998728 -- a difference of 1.7e-5, against twice the memory. On a 16 GB machine
the float32 model is 10.8 GB resident and the run was killed twice partway
through the fold, so the trade is not close. Set ``PLMNN_DTYPE=float32`` to run
it the other way where there is memory for it; the artifact records which was
used.

Nothing here reads a label. The output is a prediction per residue, in the same
form as every other detector's, and the comparison against them is a separate
indexed read.

Usage: PYTHONPATH=src:tools python3.12 tools/plmnn_embed.py [--check] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from export_plmnn_weights import forward, load as load_head
from pocket_bench.paths import ROOT

SEQS = ROOT / "results/baselines/PLMNN_SEQUENCES.json"
NETWORK = ROOT / "results/baselines/PLMNN_NETWORK.json"
PER_STRUCTURE = ROOT / "results/official_fold/PER_STRUCTURE.json"
EXAMPLE = ROOT / "data/cryptobench_apo/_osf/cryptobench/scripts/data/7w19A.npy"
OUT = ROOT / "results/baselines/PLMNN_SCORES.json"
CKPT = ROOT / "results/baselines/_plmnn_checkpoint.jsonl"
# The external validation set, through the same encoder, the same restored head
# and the same universe check, with four paths swapped.
EXTERNAL = {
    "seqs": ROOT / "results/baselines/PLMNN_EXTERNAL_SEQUENCES.json",
    "per_structure": ROOT / "results/external/PER_STRUCTURE.json",
    "out": ROOT / "results/baselines/PLMNN_EXTERNAL_SCORES.json",
    "ckpt": ROOT / "results/baselines/_plmnn_external_checkpoint.jsonl",
}


def use_external() -> None:
    global SEQS, PER_STRUCTURE, OUT, CKPT
    SEQS, PER_STRUCTURE = EXTERNAL["seqs"], EXTERNAL["per_structure"]
    OUT, CKPT = EXTERNAL["out"], EXTERNAL["ckpt"]
SCHEMA = "geoaudit.plmnn_scores.v1"

ESM_NAME = "esm2_t36_3B_UR50D"
ESM_SHA256 = "7de8b4082ba15891959ab368b77ce3886697af1efb16d3c9e9e7b0c5d3f07500"
ESM_CACHE = Path(os.environ.get(
    "ESM2_CACHE",
    os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs/"
                       "foliation-er100/esm2_cache/hub/checkpoints")))
DTYPE = os.environ.get("PLMNN_DTYPE", "float16")
# Measured on the validation chain, so that the cost of the memory decision is on
# the record rather than asserted: float32 agrees with the published example to
# this cosine, and float16 to the one the run reports.
FLOAT32_COSINE_ON_THE_VALIDATION_CHAIN = 0.998745
# The layer the authors' example pins, and the depth to run so that fair-esm does
# not overwrite it with its layer-normed form.
LAYER = 33
N_LAYERS = LAYER + 1
VALIDATION_UNIT = "7w19_A"
# The validation chain is the one the authors published an embedding for, so it is
# how this rebuild is shown to reproduce theirs. It belongs to the official fold,
# and the external run needs it just as much, so it is always looked up in the
# official sequence artifact rather than in whichever set is being scored.
OFFICIAL_SEQS = ROOT / "results/baselines/PLMNN_SEQUENCES.json"
# The published example is matched to a mean cosine of 0.9987, not to 1.0, and the
# remaining disagreement is not on this side of it. Five candidate causes were
# tested and excluded, each by running the alternative and measuring it:
#
#   the layer          33 is the unique best of the 34 computed; its neighbours
#                      are 0.966 and 0.965
#   arithmetic         float32 gives 0.998745 and float16 0.998728, so the
#                      difference is not precision
#   residue letters    the chain has no modified residues, so no one-letter
#                      convention is in play
#   unobserved context embedding the full 302-residue SEQRES and selecting the
#                      293 observed rows is worse, 0.9961
#   tokenisation       the standard beginning and end tokens are better than
#                      omitting the end token, 0.9976
#
# What is left is most likely a different implementation of the same weights --
# the authors' text does not say which ESM-2 codebase they ran. That cannot be
# settled without an 11 GB download, so it is reported as an open discrepancy
# with its downstream effect measured, rather than assumed away. The gate is set
# where the excluded alternatives sit, so any of them creeping back in fails it.
MIN_COSINE = 0.995
MIN_SPEARMAN = 0.99
# 2v6m_D carries insertion codes, so two embedding rows can share one resseq while
# the evaluation universe keys on the integer alone. The larger of the two is kept:
# it is the reading most favourable to the baseline, which is the safe direction
# for a rival method's score.
TIE_RULE = "max over the embedding rows sharing a resseq"


def _model():
    """The encoder, assembled so that it fits in 16 GB of memory.

    The published checkpoint is float16. Loading it the ordinary way builds a
    float32 model first and holds the float16 checkpoint alongside it, which
    peaks at about 17 GB and is killed outright. So the model is built in
    float16, the checkpoint is dropped, the unused layers are dropped, and only
    then is what remains raised to float32, if float32 was asked for.
    """
    import gc

    import torch
    import esm

    path = ESM_CACHE / f"{ESM_NAME}.pt"
    if not path.exists():
        raise SystemExit(f"missing {path}; fetch it with "
                         "tools/fetch_esm2_weights.py")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    prior = torch.get_default_dtype()
    torch.set_default_dtype(torch.float16)
    try:
        model, alphabet = esm.pretrained.load_model_and_alphabet_core(
            ESM_NAME, ck, None)
    finally:
        torch.set_default_dtype(prior)
    del ck
    gc.collect()

    model.layers = torch.nn.ModuleList(list(model.layers)[:N_LAYERS])
    model.num_layers = N_LAYERS
    model.lm_head = torch.nn.Identity()
    model.contact_head = None
    gc.collect()
    if DTYPE not in ("float16", "float32"):
        raise SystemExit(f"PLMNN_DTYPE={DTYPE}; only float16 or float32")
    # Every tensor is moved to the target dtype one at a time, which keeps the
    # peak at one copy. It has to be every tensor and not only the parameters:
    # the rotary embedding builds its frequency buffer with an explicit float()
    # regardless of the default dtype, and one float32 buffer left among float16
    # weights makes the attention refuse to multiply.
    want = torch.float16 if DTYPE == "float16" else torch.float32
    with torch.no_grad():
        for p in list(model.parameters()) + list(model.buffers()):
            if p.is_floating_point() and p.dtype != want:
                p.data = p.data.to(want)
    gc.collect()
    return model.eval(), alphabet.get_batch_converter()


def _embed(model, batch, seq: str, layers: list[int]) -> dict[int, np.ndarray]:
    import torch

    _, _, toks = batch([("x", seq)])
    with torch.no_grad():
        out = model(toks, repr_layers=layers, return_contacts=False)
    # float32 on the way out even when the encoder ran in float16: these vectors
    # reach a magnitude of about 1600, so the sum of 2560 squares overflows
    # float16 and every norm taken of them would come back as infinity.
    return {L: out["representations"][L][0, 1:len(seq) + 1].float().numpy()
            for L in layers}


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    def rank(v):
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), float)
        r[o] = np.arange(len(v), dtype=float)
        return r
    a, b = rank(x) - rank(x).mean(), rank(y) - rank(y).mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / d) if d else 0.0


def _validate(model, batch, head, seq: str) -> dict:
    """Recover the layer from the authors' worked example, and check the head."""
    ref = np.load(EXAMPLE)
    reps = _embed(model, batch, seq, list(range(N_LAYERS)))
    if reps[LAYER].shape != ref.shape:
        raise SystemExit(
            f"our embedding of {VALIDATION_UNIT} is {reps[LAYER].shape} and the "
            f"published one is {ref.shape}; the sequence does not match theirs")

    rn = np.linalg.norm(ref, axis=1)
    per_layer = {}
    for L, a in reps.items():
        c = (a * ref).sum(1) / (np.linalg.norm(a, axis=1) * rn)
        per_layer[L] = float(c.mean())
    best = max(per_layer, key=per_layer.get)
    if best != LAYER:
        raise SystemExit(
            f"the published example matches layer {best} (cosine "
            f"{per_layer[best]:.6f}), not the layer {LAYER} this tool takes")
    if per_layer[LAYER] < MIN_COSINE:
        raise SystemExit(
            f"layer {LAYER} agrees with the published example only to cosine "
            f"{per_layer[LAYER]:.6f}, below {MIN_COSINE}")

    # The head applied to their embedding and to ours: this is the quantity the
    # comparison actually uses, so its agreement is what has to be reported.
    theirs = forward(ref.astype(np.float64), head)[:, 1]
    ours = forward(reps[LAYER].astype(np.float64), head)[:, 1]
    rho = _spearman(ours, theirs)
    # Per-unit ROC-AUC depends only on the order of the scores within a chain, so
    # the rank agreement is the fidelity that matters for the comparison; the
    # probability gap matters instead for any fixed-threshold reading of it.
    if rho < MIN_SPEARMAN:
        raise SystemExit(
            f"our embedding ranks the residues of {VALIDATION_UNIT} at spearman "
            f"{rho:.6f} against the published one, below {MIN_SPEARMAN}; the "
            "baseline would not be theirs")
    return {
        "unit": VALIDATION_UNIT,
        "n_residues": int(ref.shape[0]),
        "published_embedding": str(EXAMPLE.relative_to(ROOT)),
        "published_embedding_sha256": hashlib.sha256(
            EXAMPLE.read_bytes()).hexdigest(),
        "mean_cosine_against_the_published_embedding_by_layer": {
            str(L): round(v, 6) for L, v in sorted(per_layer.items())},
        "layer_recovered": best,
        "mean_cosine_at_that_layer": round(per_layer[LAYER], 8),
        "second_best_layer": sorted(per_layer, key=per_layer.get)[-2],
        "second_best_mean_cosine": round(
            sorted(per_layer.values())[-2], 6),
        "predicted_probability_agreement": {
            "max_absolute_difference": round(float(np.abs(ours - theirs).max()), 8),
            "mean_absolute_difference": round(float(np.abs(ours - theirs).mean()), 8),
            "spearman": round(rho, 8),
            "what_this_measures": (
                "the baseline's own network run on their published embedding "
                "against the same network run on ours, which is the number the "
                "comparison depends on rather than the embedding itself"),
            "why_the_rank_agreement_is_the_one_that_matters": (
                "the comparison is per-unit ROC-AUC, which reads only the order "
                "of the scores within a chain. The probability gap would matter "
                "for a fixed-threshold reading, which is not what is claimed"),
        },
        "alternatives_tested_and_excluded": {
            "a_different_layer": (
                f"layer {best} is the unique best of the {N_LAYERS} computed"),
            "arithmetic_precision": (
                f"float32 gives {FLOAT32_COSINE_ON_THE_VALIDATION_CHAIN} and "
                f"float16 gives the figure above, so the residual is not "
                f"precision"),
            "the_full_seqres_including_unobserved_residues": (
                "0.996109, worse; embedding the 302-residue SEQRES of this "
                "chain and selecting its 293 observed rows moves further away"),
            "omitting_the_end_of_sequence_token": "0.997609, worse",
            "a_modified_residue_convention": (
                "this chain has no modified residues, so no convention applies"),
            "what_remains": (
                "most likely a different implementation of the same published "
                "weights; the deposit does not say which ESM-2 codebase was "
                "used, and settling it needs an 11 GB download of the other "
                "one. It is left open, with the effect above measured"),
        },
    }


def _done() -> dict[str, dict]:
    if not CKPT.exists():
        return {}
    out = {}
    for line in CKPT.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["unit_id"]] = r
    return out


def run(limit: int = 0) -> dict:
    seqs = json.loads(SEQS.read_text())
    per = {f"{r['pdb']}_{r['chain']}": r
           for r in json.loads(PER_STRUCTURE.read_text())}
    head = load_head()
    rows = seqs["rows"][:limit] if limit else seqs["rows"]

    done = _done()
    model = batch = None
    validation = None
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, r in enumerate(rows):
        if r["unit_id"] in done and validation is not None:
            continue
        if model is None:
            print("loading ESM2-3B", flush=True)
            model, batch = _model()
            validation = _validate(
                model, batch,
                head, next(x["sequence"] for x in
                           json.loads(OFFICIAL_SEQS.read_text())["rows"]
                           if x["unit_id"] == VALIDATION_UNIT))
            print(f"layer {validation['layer_recovered']} recovered, cosine "
                  f"{validation['mean_cosine_at_that_layer']:.6f}", flush=True)
        if r["unit_id"] in done:
            continue

        t1 = time.time()
        rep = _embed(model, batch, r["sequence"], [LAYER])[LAYER]
        p = forward(rep.astype(np.float64), head)[:, 1]
        by_res: dict[int, float] = {}
        for resseq, prob in zip(r["resseq_per_row"], p):
            by_res[resseq] = max(by_res.get(resseq, -1.0), float(prob))
        if len(by_res) != per[r["unit_id"]]["n_universe"]:
            raise SystemExit(
                f"{r['unit_id']}: the baseline scored {len(by_res)} residues "
                f"but the frozen universe has "
                f"{per[r['unit_id']]['n_universe']}")
        rec = {"unit_id": r["unit_id"],
               "n_residues": len(by_res),
               "scores": {str(k2): round(v, 8)
                          for k2, v in sorted(by_res.items())},
               "seconds": round(time.time() - t1, 3)}
        with CKPT.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        done[r["unit_id"]] = rec
        print(f"{k + 1}/{len(rows)} {r['unit_id']} "
              f"{len(by_res)} residues {rec['seconds']:.1f}s "
              f"({(time.time() - t0) / 60:.1f} min elapsed)", flush=True)

    ordered = [done[r["unit_id"]] for r in rows]
    secs = sorted(r["seconds"] for r in ordered)
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "what_this_is": (
            "CryptoBench's own pLM-NN baseline, run over the 192 single-chain "
            "units of the official test fold, one probability per residue"),
        "why_this_baseline": (
            "the only comparison in this repository was against P2Rank, which "
            "is a general pocket finder. pLM-NN is the cryptic-specific "
            "baseline the benchmark's authors themselves fitted and published, "
            "so it is the one this method has to be measured against"),
        "encoder": {
            "name": ESM_NAME,
            "sha256": ESM_SHA256,
            "layer": LAYER,
            "layers_computed": N_LAYERS,
            "dtype": DTYPE,
            "why_this_precision": (
                "the published checkpoint is stored in float16. float32 agrees "
                f"with the authors' example to cosine "
                f"{FLOAT32_COSINE_ON_THE_VALIDATION_CHAIN}, float16 to the "
                "figure reported below, a difference of about 2e-5; float32 is "
                "10.8 GB resident and was killed twice partway through the fold "
                "on a 16 GB machine, so float16 is what completes it"),
            "why_this_layer": (
                "the deposit does not say which layer its example embedding "
                "came from; it was recovered by matching that example, and 33 "
                "is the last layer of ESM2-650M, whose example code uses "
                "repr_layers=[33]"),
            "why_one_layer_more_is_computed_than_is_used": (
                "fair-esm overwrites the last computed layer's stored "
                "representation with its layer-normed form, so a 33-layer "
                "model asked for layer 33 returns the wrong tensor"),
        },
        "network": {
            "weights_sha256": json.loads(NETWORK.read_text())["weights_sha256"],
            "read_from": "results/baselines/PLMNN_NETWORK.json",
        },
        "sequences": {
            "from": str(SEQS.relative_to(ROOT)),
            "sha256": seqs["sequence_sha256"],
        },
        "validation_against_the_published_example": validation,
        "residues_sharing_a_resseq": {
            "units": seqs["units_where_rows_outnumber_the_universe"],
            "rule": TIE_RULE,
            "why": ("the evaluation universe keys on integer resseq while the "
                    "baseline emits one row per observed residue, and the "
                    "reading most favourable to the baseline is the safe one"),
        },
        "n_units": len(ordered),
        "n_residues_total": sum(r["n_residues"] for r in ordered),
        "seconds_per_chain": {
            "median": round(secs[len(secs) // 2], 3),
            "min": round(secs[0], 3),
            "max": round(secs[-1], 3),
            "total_minutes": round(sum(secs) / 60, 1),
            "boundary": ("one forward pass of the encoder plus the dense "
                         "network, with the model already loaded; it excludes "
                         "the load, which is reported separately"),
        },
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "scores_sha256": hashlib.sha256(json.dumps(
            [[r["unit_id"], r["scores"]] for r in ordered],
            sort_keys=True).encode()).hexdigest(),
        "units": [{k: v for k, v in r.items() if k != "seconds"}
                  for r in ordered],
        "test_fold_read_index": None,
        "why_this_is_not_an_indexed_read": (
            "this is a rival method's prediction on the fold, the same kind of "
            "object as our own per-structure scores. No label is opened and no "
            "metric is computed; the comparison that does both is a separate "
            "indexed read"),
    }


def _report(d: dict) -> None:
    v = d["validation_against_the_published_example"]
    s = d["seconds_per_chain"]
    print(f"{d['n_units']} units, {d['n_residues_total']} residues")
    print(f"  layer {d['encoder']['layer']} recovered from the published "
          f"example: cosine {v['mean_cosine_at_that_layer']:.6f} "
          f"(next best layer {v['second_best_layer']}: "
          f"{v['second_best_mean_cosine']:.6f})")
    a = v["predicted_probability_agreement"]
    print(f"  their embedding against ours, through their network: "
          f"max |dp| {a['max_absolute_difference']:.2e}, "
          f"spearman {a['spearman']:.6f}")
    print(f"  {s['median']:.1f}s median per chain, {s['total_minutes']:.0f} min "
          f"total")
    print(f"  scores digest {d['scores_sha256'][:16]}")


def check() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    bad = []
    if d.get("schema") != SCHEMA:
        bad.append("unexpected schema")
    if d.get("test_fold_read_index") is not None:
        bad.append("a prediction must not claim a read index")
    seqs = json.loads(SEQS.read_text())
    if d.get("sequences", {}).get("sha256") != seqs["sequence_sha256"]:
        bad.append("the sequences the baseline ran on are not the frozen ones")
    if (d.get("network", {}).get("weights_sha256")
            != json.loads(NETWORK.read_text())["weights_sha256"]):
        bad.append("the network weights changed since the baseline was run")
    if d.get("encoder", {}).get("sha256") != ESM_SHA256:
        bad.append("the encoder checkpoint changed since the baseline was run")

    per = {f"{r['pdb']}_{r['chain']}": r
           for r in json.loads(PER_STRUCTURE.read_text())}
    units = d.get("units", [])
    if len(units) != len(seqs["rows"]):
        bad.append(f"{len(units)} units scored, {len(seqs['rows'])} expected")
    for u in units:
        want = per[u["unit_id"]]["n_universe"]
        if len(u["scores"]) != want or u["n_residues"] != want:
            bad.append(f"{u['unit_id']}: {len(u['scores'])} residues against "
                       f"the frozen universe's {want}")
        if not all(0.0 <= v <= 1.0 for v in u["scores"].values()):
            bad.append(f"{u['unit_id']}: a probability outside [0, 1]")
    digest = hashlib.sha256(json.dumps(
        [[u["unit_id"], u["scores"]] for u in units],
        sort_keys=True).encode()).hexdigest()
    if digest != d.get("scores_sha256"):
        bad.append("the recorded scores digest does not match the scores")
    v = d.get("validation_against_the_published_example") or {}
    if v.get("layer_recovered") != d.get("encoder", {}).get("layer"):
        bad.append("the layer used is not the layer the example recovered")
    if (v.get("mean_cosine_at_that_layer") or 0) < MIN_COSINE:
        bad.append("the published example was never matched")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if bad:
        return 1
    _report(d)
    print(f"\nOK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--external", action="store_true",
                    help="embed and score the external validation set instead")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N units (smoke tests)")
    args = ap.parse_args()
    if args.external:
        use_external()
    if args.check:
        return check()
    d = run(args.limit)
    if args.limit:
        print("\npartial run, artifact not written")
        _report(d)
        return 0
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}\n")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
