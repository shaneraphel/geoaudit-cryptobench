#!/usr/bin/env python3
"""Run the official PocketMiner network on the same 192 single-chain test units.

PocketMiner is the baseline this comparison has been missing. CryptoBench's own
paper treats it as the representative method for cryptic-site detection, while
P2Rank -- the only external baseline scored here so far -- is a general pocket
predictor that was never trained to find pockets that are absent from the apo
structure. A cryptic-pocket paper whose sole comparison is P2Rank is answering an
easier question than the one it claims to answer.

Nothing about the network is reimplemented. The published checkpoint is restored
into the authors' own ``MQAModel``, at the pinned commit, with the hyperparameters
their README gives, and the forward pass is theirs. Three things around it are
this file's own work, because each is a place where a run that looks fine returns
numbers that mean nothing:

  A restore that does nothing. ``util.load_checkpoint`` calls
  ``ckpt.restore(path)`` and neither asserts nor returns the status object, so a
  name mismatch leaves a randomly initialised network that still predicts, still
  produces probabilities in [0,1], and still writes a plausible artifact. Keras 3
  renames the variables of subclassed models, so on a current TensorFlow this is
  not a hypothetical. ``TF_USE_LEGACY_KERAS`` is set before the import and the
  restore is asserted; then, because an assertion inside TensorFlow's own
  bookkeeping cannot detect a fault in that bookkeeping, the whole thing is
  checked end to end against the authors' labels (see ``--selftest``).

  A reshape that silently misaligns. ``process_strucs`` selects backbone atoms
  and calls ``xyz.reshape(l, 4, 3)``, which assumes every residue contributes
  exactly N, CA, C and O, in that order. Four of the 192 receptors have a residue
  that does not -- seven residues in all -- and there the reshape would not fail,
  it would shift every subsequent residue's coordinates onto the wrong residue.
  Backbone atoms are therefore gathered per residue by name, incomplete residues
  are dropped and named in the artifact, and the result is asserted equal to the
  authors' own tensor on every structure where the two are supposed to agree.

  A comparison against a method that has seen the answer. Six of our 190 test
  PDB entries appear in PocketMiner's own data: 1rtc among the 38 systems it was
  trained on, 3rwv and 5uxa in the set it was selected on, and 1kx9, 3nx1 and
  3ugk in the set it was published on. The read that consumes these scores
  reports the fold with and without them.

Coordinates are in nanometres, which is what mdtraj hands back and what the
network was trained on; feeding Angstroms produces a confidently wrong answer
rather than an error.

This runs in ``.venv-pocketminer``, not the project interpreter, because
TensorFlow pins numpy below the version the frozen pipeline uses and installing
it alongside would move numbers that are already published.

Usage:
  .venv-pocketminer/bin/python tools/pocketminer_run.py --selftest
  .venv-pocketminer/bin/python tools/pocketminer_run.py --predict
  python3.12 tools/pocketminer_run.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PM = ROOT / "third_party/pocketminer"
PM_SRC = PM / "src"
CKPT = PM / "models/pocketminer"
RECEPTORS = ROOT / "data/cryptobench_apo/official_receptors"
MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
SCORE_DIR = ROOT / "data/baselines/pocketminer"
OUT = ROOT / "results/baselines/POCKETMINER_SCORES.json"
SELFTEST_OUT = ROOT / "results/baselines/POCKETMINER_SELFTEST.json"
# The training fold, so this baseline can be given a threshold chosen the same
# way the other two methods' were: on training data, then frozen.
TRAIN_RECEPTORS = ROOT / "data/cryptobench_apo/train_receptors"
TRAIN_SCORE_DIR = ROOT / "data/baselines/pocketminer_train"
TRAIN_OUT = ROOT / "results/baselines/POCKETMINER_TRAIN_SCORES.json"
# The external validation set: structures released after CryptoBench's newest, no
# UniRef50 cluster shared with either of its folds. Scored by exactly this code,
# this checkpoint and this featurisation, so a difference between the two folds
# cannot be a difference in how the baseline was run.
EXTERNAL_RECEPTORS = ROOT / "data/external/receptors"
EXTERNAL_SCORE_DIR = ROOT / "data/baselines/pocketminer_external"
EXTERNAL_OUT = ROOT / "results/baselines/POCKETMINER_EXTERNAL_SCORES.json"

SCHEMA = "geoaudit.pocketminer_scores.v1"
SELFTEST_SCHEMA = "geoaudit.pocketminer_selftest.v1"

SOURCE = ("https://github.com/Mickdub/gvp/tree/pocket_pred (Meller et al., "
          "Nat. Commun. 2023, doi:10.1038/s41467-023-36699-3)")
PM_COMMIT = "187062df3c94127e991669768009141a08fd5d8b"
WEIGHT_SHA256 = {
    "models/pocketminer.index":
        "09e36a62a987b14bc49cf7a1fa53fa734f85d85d088777f98c6aca253b47bf46",
    "models/pocketminer.data-00000-of-00001":
        "6f5ab62b9fb38b54040053ac321f3362d616bae9b31569249a032dbb50890b70",
}

# The README's own numbers for the published network. Changing any of these
# builds a different network, which would restore partially or not at all.
NODE_FEATURES = (8, 50)
EDGE_FEATURES = (1, 32)
HIDDEN_DIM = (16, 100)
NUM_LAYERS = 4
DROPOUT_RATE = 0.1

# The self-test guard, and it is tighter than a floor. The paper reports
# "563 residues that form cryptic pockets and 1283 residues that do not in our
# test set" and a residue-level ROC-AUC of 0.87. Those two counts are asserted
# exactly, because they pin the label alignment: any off-by-one in the residue
# order, any structure paired with the wrong labels, moves them. The AUC is then
# asserted to round to the published figure. A failed restore sits at 0.5.
PUBLISHED = {
    "roc_auc": 0.87,
    "n_positive": 563,
    "n_negative": 1283,
    "quote": "In total, there were 563 residues that form cryptic pockets and "
             "1283 residues that do not form cryptic pockets in our test set. "
             "... our final model, referred to as PocketMiner, achieves very good "
             "performance at discriminating residues that form cryptic pockets "
             "from those that do not (ROC AUC: 0.87)",
    "where": "Meller et al., Nat. Commun. 14, 1177 (2023)",
}
SELFTEST_MIN_ROC_AUC = 0.75
BACKBONE = ("N", "CA", "C", "O")
# Two residues can share a chain, a residue number and a residue name for two
# unrelated reasons, and they have to be told apart before the graph is built.
# An alternate conformer is a second copy of the same residue and its CA sits on
# top of the first: across the receptors here, within 1 A. An insertion-coded
# residue is a different residue that PDB numbering cannot distinguish, and its CA
# is a peptide bond away: 3.8 A in the one test chain that has any. A conformer
# copy left in the input does real damage -- 4m7p_A carries twenty of them, which
# would hand the network 7800 residues where the protein has 390, and every
# nearest-neighbour edge in the graph would be to a copy of the residue itself.
ALTERNATE_CONFORMER_CA_ANGSTROM = 2.0


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _verify_checkout() -> dict:
    """The pinned commit and the two weight files, by hash, before anything runs."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PM,
                          capture_output=True, text=True).stdout.strip()
    if head != PM_COMMIT:
        raise SystemExit(f"third_party/pocketminer is at {head}, expected {PM_COMMIT}")
    got = {}
    for rel, want in WEIGHT_SHA256.items():
        f = PM / rel
        if not f.is_file():
            raise SystemExit(f"missing {rel}")
        got[rel] = _sha256(f)
        if got[rel] != want:
            raise SystemExit(f"{rel} hashes {got[rel]}, expected {want}")
    return {"source": SOURCE, "commit": PM_COMMIT, "weight_sha256": got,
            "architecture": {"node_features": list(NODE_FEATURES),
                             "edge_features": list(EDGE_FEATURES),
                             "hidden_dim": list(HIDDEN_DIM),
                             "num_layers": NUM_LAYERS,
                             "dropout": DROPOUT_RATE,
                             "k_neighbors": 30,
                             "output": "Dense(1, activation='sigmoid') per residue"}}


def _import_pocketminer():
    """Their modules, on their own flat import path, with Keras 2 semantics.

    Set before importing tensorflow: with Keras 3 the variables of a subclassed
    model are named differently and the checkpoint restores nothing.
    """
    os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, str(PM_SRC))
    import mdtraj as md  # noqa: E402
    import tensorflow as tf  # noqa: E402
    from models import MQAModel  # noqa: E402
    from validate_performance_on_xtals import (  # noqa: E402
        abbrev, lookup, process_strucs)
    return md, tf, MQAModel, process_strucs, abbrev, lookup


def _load_model(tf, MQAModel):
    """Build, restore, and refuse to proceed on an unmatched restore.

    ``util.load_checkpoint`` restores ``Checkpoint(optimizer=Adam(), model=model)``.
    The checkpoint stores a pre-2.11 Keras optimizer, and a current Adam raises
    on contact with it, so only the ``model`` subtree is restored here. Nothing is
    lost: the optimizer slots are training state, this is inference, and the
    assertion below is over exactly the variables that carry the network.
    """
    model = MQAModel(node_features=NODE_FEATURES, edge_features=EDGE_FEATURES,
                     hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS,
                     dropout=DROPOUT_RATE)
    # ``read`` rather than ``restore``: ``restore`` also expects to own a
    # ``save_counter``, which is the writing Checkpoint's bookkeeping and not
    # part of the network, and its absence would fail the assertion below for a
    # reason that has nothing to do with the weights.
    status = tf.train.Checkpoint(model=model).read(str(CKPT))
    return model, status


def _assert_restored(status, model, tf) -> dict:
    """Every variable in the object graph got a value from the checkpoint file.

    ``assert_existing_objects_matched`` is TensorFlow's own statement; the
    variable count against the checkpoint's ``model/`` entries is a second,
    independent one, and the self-test is the third and only end-to-end one.
    """
    status.assert_existing_objects_matched()
    n_vars = len(model.variables)
    if n_vars == 0:
        raise SystemExit("model has no variables; the forward pass did not build it")
    live = {}
    for v in model.variables:
        a = v.numpy()
        live[hashlib.sha256(a.tobytes()).hexdigest()] = live.get(
            hashlib.sha256(a.tobytes()).hexdigest(), 0) + 1
    on_disk = {}
    n_file = 0
    for name, _ in tf.train.list_variables(str(CKPT)):
        if not name.startswith("model/"):
            continue
        n_file += 1
        a = tf.train.load_variable(str(CKPT), name)
        k = hashlib.sha256(a.tobytes()).hexdigest()
        on_disk[k] = on_disk.get(k, 0) + 1
    unbacked = [k for k, n in live.items() if on_disk.get(k, 0) < n]
    if unbacked:
        raise SystemExit(f"{len(unbacked)} of {n_vars} live tensors are not "
                         "byte-identical to anything in the checkpoint file")
    total = sum(int(v.numpy().size) for v in model.variables)
    absmean = float(sum(abs(v.numpy()).sum() for v in model.variables) / total)
    if absmean == 0.0:
        raise SystemExit("restored weights are all zero")
    return {"assert_existing_objects_matched": True,
            "every_live_tensor_is_byte_identical_to_the_checkpoint": True,
            "why_both": "the assertion is TensorFlow's own bookkeeping; the byte "
                        "comparison does not depend on it",
            "n_model_variables": n_vars,
            "n_checkpoint_entries_under_model": n_file,
            "n_scalars": total,
            "mean_abs_weight": round(absmean, 8)}


def _backbone(md, traj, abbrev, lookup):
    """One structure to (X, S, resseq, dropped), gathering N/CA/C/O by name.

    The authors' ``reshape(l, 4, 3)`` is correct exactly when every residue has
    all four backbone atoms; where it is not, this drops the residue rather than
    shifting the ones after it. Where a residue is a second conformer of one
    already taken, it is dropped too, for the reason given at
    ``ALTERNATE_CONFORMER_CA_ANGSTROM``.
    """
    import numpy as np
    keep, resseq, resname, dropped = [], [], [], []
    first_ca: dict[tuple, "np.ndarray"] = {}
    for r in traj.top.residues:
        if r.name not in abbrev:
            dropped.append({"resseq": int(r.resSeq), "resname": r.name,
                            "why": "not one of the network's 20 residue types"})
            continue
        idx = {}
        for a in r.atoms:
            if a.name in BACKBONE and a.name not in idx:
                idx[a.name] = a.index
        if len(idx) != len(BACKBONE):
            dropped.append({"resseq": int(r.resSeq), "resname": r.name,
                            "why": "incomplete backbone, has only "
                                   + "/".join(sorted(idx))})
            continue
        key = (r.chain.index, int(r.resSeq), r.name)
        ca = traj.xyz[0, idx["CA"]] * 10.0
        prior = first_ca.get(key)
        if prior is not None:
            d = float(np.linalg.norm(ca - prior))
            if d < ALTERNATE_CONFORMER_CA_ANGSTROM:
                dropped.append({"resseq": int(r.resSeq), "resname": r.name,
                                "why": "alternate conformer of a residue already "
                                       f"taken, CA {d:.2f} A away"})
                continue
        else:
            first_ca[key] = ca
        keep.extend(idx[n] for n in BACKBONE)
        resseq.append(int(r.resSeq))
        resname.append(r.name)
    if not resseq:
        raise SystemExit("no residue survived backbone selection")
    sub = traj.atom_slice(keep)
    L = len(resseq)
    X = sub.xyz.reshape(1, L, 4, 3).astype(np.float32)
    S = np.asarray([[lookup[abbrev[n]] for n in resname]], dtype=np.int32)
    mask = np.ones((1, L), dtype=np.float32)
    return X, S, mask, resseq, dropped


def _agrees_with_official(md, process_strucs, traj, X) -> bool:
    """On a structure with no dropped residue, their tensor and ours must match.

    This is what licenses the guarded path: it is the authors' featurisation
    everywhere it is well defined, and only differs where theirs is unsound.
    """
    import numpy as np
    Xo, _, _ = process_strucs([traj])
    if Xo.shape != X.shape:
        return False
    return bool(np.array_equal(Xo, X))


def _score(model, X, S, mask) -> list[float]:
    p = model(X, S, mask, train=False, res_level=True)
    v = p.numpy().reshape(-1).tolist()
    for x in v:
        if not (0.0 <= x <= 1.0):
            raise SystemExit(f"PocketMiner emitted {x}, not a probability")
    return [float(x) for x in v]


def _roc_auc(y: list[int], s: list[float]) -> float | None:
    """Rank AUC with ties averaged; None when one class is absent."""
    n1 = sum(y)
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = sorted(range(len(s)), key=lambda i: s[i])
    ranks = [0.0] * len(s)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and s[order[j + 1]] == s[order[i]]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    r1 = sum(ranks[i] for i in range(len(y)) if y[i] == 1)
    return (r1 - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _pr_auc(y: list[int], s: list[float]) -> float | None:
    """Average precision, the step-wise sum PR-AUC that avoids interpolation."""
    n1 = sum(y)
    if n1 == 0:
        return None
    order = sorted(range(len(s)), key=lambda i: -s[i])
    tp = 0
    total = 0.0
    for rank, i in enumerate(order, 1):
        if y[i] == 1:
            tp += 1
            total += tp / rank
    return total / n1


def selftest() -> int:
    """Reproduce PocketMiner on PocketMiner's own labelled apo structures.

    The one guard that can fail for every reason at once: wrong weights, a
    restore that did nothing, Angstroms for nanometres, a residue order off by
    one. Their labels are 0 (negative), 1 (cryptic) and 2 (unclassifiable); 2 is
    masked out, which is what their own validation script does.
    """
    import numpy as np
    prov = _verify_checkout()
    md, tf, MQAModel, process_strucs, abbrev, lookup = _import_pocketminer()
    model, status = _load_model(tf, MQAModel)
    d = PM / "data/pm-dataset"
    struc_dir = d / "apo-structures"
    report = {"schema": SELFTEST_SCHEMA, "clinical_grade": False,
              "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
              "question": "does the restored network reproduce PocketMiner's own "
                          "residue-level discrimination on PocketMiner's own "
                          "labelled apo structures",
              "reads_our_test_fold": False, "provenance": prov, "splits": {}}
    restore_checked = False
    for split in ("test", "val"):
        ids = [str(x) for x in np.load(d / f"{split}_apo_ids_with_chainids.npy")]
        labels = np.load(d / f"{split}_label_dictionary.npy",
                         allow_pickle=True).item()
        pooled_y: list[int] = []
        pooled_s: list[float] = []
        per_protein = []
        misaligned = []
        # The authors' own validation zips the label dictionary's values against
        # predictions made from the id list, so the two are paired by position.
        # One test id, ``5x8ua``, does not reduce to its key by any rule, which is
        # why position rather than a key lookup is the pairing that reproduces
        # their numbers. Each pair is still checked for a name that agrees.
        keys = [str(k) for k in labels]
        if len(keys) != len(ids):
            raise SystemExit(f"{split}: {len(ids)} ids against {len(keys)} labels")
        mismatched_names = [(e, k) for e, k in zip(ids, keys)
                            if k not in (e[:4].lower(), e.lower())]
        for e, key in zip(ids, keys):
            f = struc_dir / f"{e}_clean_h.pdb"
            if not f.is_file():
                f = struc_dir / f"{e.upper()}_clean_h.pdb"
            traj = md.load(str(f))
            X, S, mask = process_strucs([traj])
            p = _score(model, X, S, mask)
            if not restore_checked:
                report["restore"] = _assert_restored(status, model, tf)
                restore_checked = True
            lab = np.asarray(labels[key]).astype(int).tolist()
            if len(lab) != len(p):
                misaligned.append({"id": e, "n_labels": len(lab),
                                   "n_predictions": len(p)})
                continue
            y = [int(v) for v, l in zip(lab, p) if v != 2]
            sc = [l for v, l in zip(lab, p) if v != 2]
            pooled_y.extend(y)
            pooled_s.extend(sc)
            per_protein.append({"id": e, "n_scored": len(y),
                                "n_positive": sum(y),
                                "roc_auc": _roc_auc(y, sc)})
            print(f"  {split} {e} {len(p)} residues", flush=True)
        auc = _roc_auc(pooled_y, pooled_s)
        pa = [x["roc_auc"] for x in per_protein if x["roc_auc"] is not None]
        report["splits"][split] = {
            "n_structures": len(ids), "n_misaligned": len(misaligned),
            "misaligned": misaligned,
            "paired_by": "position in the id list and the label dictionary, as "
                         "the authors' validate_performance_on_xtal_residues does",
            "pairs_whose_names_do_not_agree": mismatched_names,
            "n_residues_scored": len(pooled_y),
            "n_positive": sum(pooled_y),
            "pooled_roc_auc": round(auc, 6) if auc is not None else None,
            "pooled_pr_auc": round(_pr_auc(pooled_y, pooled_s), 6),
            "mean_per_protein_roc_auc": round(sum(pa) / len(pa), 6) if pa else None,
            "per_protein": per_protein,
        }
    t = report["splits"]["test"]
    if t["n_misaligned"]:
        raise SystemExit(f"{t['n_misaligned']} test structures do not align with "
                         "their own labels; the featurisation is wrong")
    if t["pooled_roc_auc"] is None or t["pooled_roc_auc"] < SELFTEST_MIN_ROC_AUC:
        raise SystemExit(f"self-test ROC-AUC {t['pooled_roc_auc']} is below the "
                         f"{SELFTEST_MIN_ROC_AUC} floor; the restore is suspect")
    n_pos = t["n_positive"]
    n_neg = t["n_residues_scored"] - n_pos
    if (n_pos, n_neg) != (PUBLISHED["n_positive"], PUBLISHED["n_negative"]):
        raise SystemExit(
            f"self-test scored {n_pos} positive and {n_neg} negative residues; "
            f"the paper's test set is {PUBLISHED['n_positive']} and "
            f"{PUBLISHED['n_negative']}. The label alignment is not theirs.")
    if round(t["pooled_roc_auc"], 2) != PUBLISHED["roc_auc"]:
        raise SystemExit(f"self-test ROC-AUC {t['pooled_roc_auc']} does not round "
                         f"to the published {PUBLISHED['roc_auc']}")
    report["reproduction"] = {
        "published": PUBLISHED,
        "ours": {"roc_auc": t["pooled_roc_auc"],
                 "n_positive": n_pos, "n_negative": n_neg},
        "residue_counts_match_exactly": True,
        "roc_auc_rounds_to_the_published_value": True,
        "what_this_licenses":
            "the network that scores our fold is the published one, restored "
            "from the published weights, fed the featurisation the authors "
            "intended, with residues in the order their labels assume. The two "
            "residue counts are what pin the last of those: an alignment off by "
            "one residue anywhere would not reproduce 563 and 1283.",
        "floor_also_checked": SELFTEST_MIN_ROC_AUC,
    }
    SELFTEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    SELFTEST_OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {SELFTEST_OUT.relative_to(ROOT)}")
    for split, s in report["splits"].items():
        print(f"  {split}: pooled ROC-AUC {s['pooled_roc_auc']}  "
              f"PR-AUC {s['pooled_pr_auc']}  "
              f"mean per-protein {s['mean_per_protein_roc_auc']}  "
              f"({s['n_positive']}/{s['n_residues_scored']} positive)")
    return 0


def _units(where: Path | None = None) -> list[tuple[str, str]]:
    out = []
    for f in sorted((where or RECEPTORS).glob("*_receptor.pdb")):
        pdb, chain = f.name.replace("_receptor.pdb", "").rsplit("_", 1)
        out.append((pdb, chain))
    return out


def predict(limit: int = 0, train: bool = False,
            external: bool = False) -> int:
    """Score the receptors P2Rank and the counting field were given.

    ``train`` scores the 770 training units instead, which is not a read of
    anything held out: it exists so this baseline's threshold can be chosen on
    training data and frozen, the way the other two methods' were.

    ``external`` scores the external validation set. Producing scores is not
    reading it: no label is touched and no metric is computed here, which is what
    keeps the single read of that set under its plan.
    """
    if external:
        receptors, score_dir, out_path = (EXTERNAL_RECEPTORS,
                                          EXTERNAL_SCORE_DIR, EXTERNAL_OUT)
    elif train:
        receptors, score_dir, out_path = (TRAIN_RECEPTORS, TRAIN_SCORE_DIR,
                                          TRAIN_OUT)
    else:
        receptors, score_dir, out_path = RECEPTORS, SCORE_DIR, OUT
    prov = _verify_checkout()
    md, tf, MQAModel, process_strucs, abbrev, lookup = _import_pocketminer()
    model, status = _load_model(tf, MQAModel)
    score_dir.mkdir(parents=True, exist_ok=True)
    units = _units(receptors)
    if limit:
        units = units[:limit]
    rows = []
    restore = None
    t0 = time.time()
    for i, (pdb, chain) in enumerate(units, 1):
        f = receptors / f"{pdb}_{chain}_receptor.pdb"
        traj = md.load(str(f))
        X, S, mask, resseq, dropped = _backbone(md, traj, abbrev, lookup)
        agrees = None if dropped else _agrees_with_official(
            md, process_strucs, traj, X)
        if agrees is False:
            raise SystemExit(f"{pdb}_{chain}: our backbone tensor differs from "
                             "the authors' on a structure where it should not")
        p = _score(model, X, S, mask)
        if restore is None:
            restore = _assert_restored(status, model, tf)
        collisions = []
        scores: dict[str, float] = {}
        for r, v in zip(resseq, p):
            k = str(r)
            if k in scores:
                collisions.append(r)
                continue
            scores[k] = round(v, 6)
        (score_dir / f"{pdb}_{chain}.json").write_text(json.dumps({
            "pdb": pdb, "chain": chain, "source": SOURCE,
            "commit": PM_COMMIT,
            "residue_scores": scores}, indent=1) + "\n")
        rows.append({"unit": f"{pdb}_{chain}", "n_residues_in_file":
                     int(traj.top.n_residues), "n_scored": len(scores),
                     "agrees_with_official_featurisation": agrees,
                     "dropped": dropped,
                     "resseq_collisions_from_insertion_codes": collisions,
                     "max_score": round(max(p), 6),
                     "mean_score": round(sum(p) / len(p), 6)})
        print(f"{i}/{len(units)} {pdb}_{chain} {len(scores)} residues "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    n_drop = sum(1 for r in rows for x in r["dropped"]
                 if x["why"].startswith("incomplete"))
    n_conf = sum(1 for r in rows for x in r["dropped"]
                 if x["why"].startswith("alternate"))
    n_type = sum(1 for r in rows for x in r["dropped"]
                 if x["why"].startswith("not one"))
    n_coll = sum(len(r["resseq_collisions_from_insertion_codes"]) for r in rows)
    checked = [r for r in rows if r["agrees_with_official_featurisation"] is True]
    d = {
        "schema": SCHEMA, "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "fold": ("external" if external else
                 "train" if train else "official_mmseqs2_10pct_test"),
        "question": ("what does the official PocketMiner network predict for each "
                     f"residue of the {len(rows)} "
                     f"{'training' if train else 'single-chain test'} units"),
        "reads_a_label": False,
        "why_no_label_is_read": "this file produces predictions; the comparison "
                                "against ours and P2Rank's is a separate indexed "
                                "read of the test fold",
        "provenance": prov,
        "input": {"receptors": str(receptors.relative_to(ROOT)),
                  "identical_to_what_p2rank_and_the_field_were_given": True,
                  "coordinate_units": "nanometres, as mdtraj returns them"},
        "featurisation": {
            "gathered_by_atom_name": list(BACKBONE),
            "why_not_the_authors_reshape":
                "reshape(l, 4, 3) assumes four backbone atoms per residue; where "
                "that is false it shifts every later residue onto the wrong "
                "coordinates instead of failing",
            "n_units_asserted_identical_to_the_authors_tensor": len(checked),
            "n_units_where_it_could_not_be_asserted": len(rows) - len(checked),
            "n_residues_dropped_for_incomplete_backbone": n_drop,
            "n_residues_dropped_as_alternate_conformers": n_conf,
            "n_residues_dropped_for_residue_type": n_type,
            "alternate_conformer_ca_cutoff_angstrom":
                ALTERNATE_CONFORMER_CA_ANGSTROM,
            "why_conformers_are_dropped_before_featurisation":
                "a conformer copy is not a residue the protein has twice; left "
                "in, it takes nearest-neighbour edges away from the residue's "
                "real neighbours. 4m7p_A in the training fold carries twenty "
                "copies of all 390 of its residues",
            "n_resseq_collisions_from_insertion_codes": n_coll,
            "what_a_collision_means":
                "the frozen pipeline keys residues by resseq, so an insertion "
                "code cannot be represented; the first occurrence is kept and the "
                "residue number is recorded here",
        },
        "restore": restore,
        "overlap_with_pocketminers_own_data": {
            "trained_on": ["1rtc"],
            "selected_on": ["3rwv", "5uxa"],
            "published_on": ["1kx9", "3nx1", "3ugk"],
            "how_it_was_determined":
                "exact PDB entry match against the 38 simulation systems in "
                "data/task2 and the val/test id lists in data/pm-dataset",
            "caveat": "exact entry match is a floor, not a homology check; "
                      "PocketMiner's training set was not clustered against "
                      "CryptoBench's folds",
            "what_the_read_does": "reports the fold with and without these six "
                                  "entries",
        },
        "n_units": len(rows), "units": rows,
        "runtime_minutes": round((time.time() - t0) / 60, 2),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {out_path.relative_to(ROOT)} and {len(rows)} files in "
          f"{score_dir.relative_to(ROOT)}")
    print(f"  identical to the authors' tensor on {len(checked)}/{len(rows)} units; "
          f"{n_drop} residues dropped for backbone, {n_conf} as conformers; "
          f"{n_coll} resseq collisions")
    return 0


def check() -> int:
    """Validate the artifacts without TensorFlow, for the project interpreter."""
    if not OUT.is_file():
        print(f"{OUT.relative_to(ROOT)} absent; PocketMiner has not been run")
        return 0
    d = json.loads(OUT.read_text())
    if d["schema"] != SCHEMA:
        raise SystemExit("unexpected schema")
    prov = d["provenance"]
    if prov["commit"] != PM_COMMIT:
        raise SystemExit("artifact pins a different PocketMiner commit")
    for rel, want in WEIGHT_SHA256.items():
        if prov["weight_sha256"].get(rel) != want:
            raise SystemExit(f"artifact records a different hash for {rel}")
    if not d["restore"]["assert_existing_objects_matched"]:
        raise SystemExit("artifact records an unmatched restore")
    units = {r["unit"] for r in d["units"]}
    expect = {f"{p}_{c}" for p, c in _units()}
    if units != expect:
        raise SystemExit(f"artifact covers {len(units)} units, universe has "
                         f"{len(expect)}")
    for r in d["units"]:
        f = SCORE_DIR / f"{r['unit']}.json"
        if not f.is_file():
            raise SystemExit(f"missing {f.relative_to(ROOT)}")
        s = json.loads(f.read_text())["residue_scores"]
        if len(s) != r["n_scored"]:
            raise SystemExit(f"{r['unit']}: manifest says {r['n_scored']} scored, "
                             f"file has {len(s)}")
        for k, v in s.items():
            if not (0.0 <= float(v) <= 1.0):
                raise SystemExit(f"{r['unit']} residue {k} scores {v}")
    if SELFTEST_OUT.is_file():
        st = json.loads(SELFTEST_OUT.read_text())
        t = st["splits"]["test"]
        if t["pooled_roc_auc"] < SELFTEST_MIN_ROC_AUC:
            raise SystemExit("self-test is below its own floor")
        print(f"self-test pooled ROC-AUC {t['pooled_roc_auc']} on "
              f"{t['n_residues_scored']} of the authors' own labelled residues")
    else:
        print("self-test artifact absent")
    print(f"{len(units)} units scored, all probabilities in [0,1], "
          f"weights and commit as pinned")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true",
                    help="reproduce PocketMiner on its own labelled structures")
    ap.add_argument("--predict", action="store_true",
                    help="score the 192 single-chain units")
    ap.add_argument("--train", action="store_true",
                    help="with --predict, score the 770 training units instead, "
                         "so this baseline's threshold can be chosen off the "
                         "held-out fold")
    ap.add_argument("--external", action="store_true",
                    help="with --predict, score the external validation set")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.check:
        return check()
    if a.selftest:
        return selftest()
    if a.predict:
        return predict(a.limit, a.train, a.external)
    ap.error("one of --selftest, --predict, --check")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
