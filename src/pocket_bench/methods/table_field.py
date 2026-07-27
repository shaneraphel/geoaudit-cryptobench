"""The table field: a cryptic-site detector that is table lookups and integer adds.

What it computes
----------------
For a receptor chain, 645 wires (see ``wide_descriptors``), each ranked within
that chain and cut at quartiles into a quaternary digit. 5152 tables each take
two digits and address one of sixteen cells. Every cell carries two integers
counted on the official training fold -- how many residues landed there and how
many of those were cryptic -- and the score is

    S(i) = sum_k m_k * pos_k[a_k(i)] / tot_k[a_k(i)],     m_k integer in [-32, 32]
    F(i) = S(i) + G_18(i) * sd(S) / sd(G_18)

where a_k(i) is a two-digit address, m_k an integer fan-out, and G_18 the mean of
S over the residues within 18 A of i. Inference reads one cell per table, adds
the cells with integer weights, and takes one spatial mean. There is no
floating-point model: the only non-integer quantities are the compiled cell
ratios themselves and the final spatial average.

What is fitted, stated plainly
------------------------------
Two things are compiled on the training fold and both are in the artifact.

The cell counts are counts; nothing is estimated about them.

The integer fan-out ``m`` comes from one symmetric linear solve over the table
outputs, regularised by a ridge of 0.03 of the scatter trace, whose solution is
then rounded onto ``[-32, 32]``. That is a closed-form expression in the compiled
tables -- no gradient, no iteration, no automatic differentiation, and no test
residue enters it -- but it is a real-valued object at compile time and the
manuscript does not pretend otherwise. The ridge is not decoration: without it
the direction fell from 0.7844 to 0.6846 on the training split when the table
pool grew from 1032 to 1720, because random pairs repeat as the pool grows and
the scatter goes near-singular.

What is NOT compiled: the digitisation. Each wire is ranked against the other
residues of the same chain, so a test structure is quantised using only itself,
no threshold crosses the fold boundary, and a wire means the same thing in a
57-residue chain and a 307-residue one.

Provenance of every structural choice
-------------------------------------
Table width, pool size, ridge, fan-out cap, gate radius and gate weight were all
ranked on a cluster-disjoint half of the official training fold
(``results/architecture_sweep/COUNTERATTACK_WIDE2.json``), where this
configuration measured 0.8045. The official test fold was read three times in
total across the development of this method and all three readings are reported;
see ``results/official_fold/COUNTERATTACK_WIDE_PROBE.json``.

Where it stands
---------------
On the 192 official test units: ROC-AUC 0.8010, PR-AUC 0.3783, against P2Rank's
0.7935 and 0.3580. Nominally ahead on both, and the paired difference is NOT
statistically separable from P2Rank at 95% (ROC-AUC +0.0075, CI
[-0.0140, +0.0283]). It is separable from the fitted linear readout on PR-AUC
(+0.0189, p=0.0062) and from the previous counting field on both metrics.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pocket_bench.methods.algebraic_descriptors import (
    FEATURE_NAMES,
    algebraic_residue_features,
)
from pocket_bench.methods.table_bank import (
    N_LEVELS,
    cell_offsets,
    chain_digits,
    compile_cells,
    integer_fanout,
    partition_tables,
    score,
)
from pocket_bench.methods.wide_descriptors import build_wide

SCHEMA = "geoaudit.table_field.v1"

# Every one of these was ranked on a cluster-disjoint half of the training fold.
TABLE_WIDTH = 2
PARTITION_ROUNDS = 16
PARTITION_SEED = 20260725
RIDGE = 0.03
FAN_OUT_CAP = 32
GATE_RADIUS = 18.0
GATE_WEIGHT = 1.0


def _neighbourhood_mean(s: np.ndarray, ctr: np.ndarray,
                        radius: float) -> np.ndarray:
    n = len(s)
    out = np.empty(n, dtype=np.float64)
    r2 = radius * radius
    for i in range(0, n, 512):
        j = min(i + 512, n)
        d2 = ((ctr[i:j, None, :] - ctr[None, :, :]) ** 2).sum(-1)
        a = (d2 <= r2).astype(np.float64)
        out[i:j] = (a @ s) / np.maximum(a.sum(1), 1.0)
    return out


def apply_gate(s: np.ndarray, ctr: np.ndarray, n_res_per=None) -> np.ndarray:
    """Add back the neighbourhood mean, rescaled to the raw score's spread.

    A cryptic site is a contiguous patch, which is the one thing a per-residue
    function cannot state. Matching the standard deviation rather than the
    maximum before adding matters because the maximum of a score field over a
    chain is an order statistic of a handful of residues, so a max-normalised
    gate mixes in a different amount on every structure.
    """
    if n_res_per is None:
        n_res_per = [len(s)]
    out = np.empty(len(s), dtype=np.float64)
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = s[off:off + n]
        g = _neighbourhood_mean(blk, np.asarray(ctr[off:off + n], float),
                                GATE_RADIUS)
        sd_s, sd_g = float(np.std(blk)), float(np.std(g))
        out[off:off + n] = (blk if sd_g <= 0
                            else blk + GATE_WEIGHT * g * (sd_s / sd_g))
        off += n
    return out


def compile_field(X: np.ndarray, y: np.ndarray, ctr: np.ndarray, n_res_per,
                  names, propensity_table: np.ndarray, *,
                  code_sha256: str = "", train_manifest_sha256: str = "",
                  ) -> dict[str, Any]:
    """Count the cells and solve for the fan-out, on the training fold only."""
    D = chain_digits(np.asarray(X, dtype=np.float64), n_res_per)
    tables = partition_tables(D.shape[1], TABLE_WIDTH, PARTITION_ROUNDS,
                              PARTITION_SEED)
    offsets = cell_offsets(tables)

    yf = np.asarray(y, dtype=np.int64)
    total = int(offsets[-1])
    tot = np.zeros(total, dtype=np.int64)
    pos = np.zeros(total, dtype=np.int64)
    from pocket_bench.methods.table_bank import BLOCK, addresses
    for a in range(0, D.shape[0], BLOCK):
        b = min(a + BLOCK, D.shape[0])
        flat = addresses(D, tables, offsets, a, b).ravel()
        tot += np.bincount(flat, minlength=total)
        pos += np.bincount(flat, weights=np.repeat(yf[a:b], len(tables)),
                           minlength=total).astype(np.int64)

    frac, _tot = compile_cells(D, yf, tables, offsets)
    mult = integer_fanout(D, yf, tables, offsets, frac, RIDGE, FAN_OUT_CAP)

    s_all = apply_gate(score(D, tables, offsets, frac, mult), ctr, n_res_per)
    q, f1 = _best_operating_point(s_all, yf, n_res_per)

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "n_levels": N_LEVELS,
        "wire_names": [str(n) for n in names],
        "n_wires": int(D.shape[1]),
        "base_feature_names": list(FEATURE_NAMES),
        "propensity_table": [float(v) for v in np.asarray(propensity_table)],
        "tables": [[int(c) for c in t] for t in tables],
        "table_width": TABLE_WIDTH,
        "partition_rounds": PARTITION_ROUNDS,
        "partition_seed": PARTITION_SEED,
        "cell_total": [int(v) for v in tot],
        "cell_positive": [int(v) for v in pos],
        "n_cells": total,
        "n_cells_never_addressed": int((tot == 0).sum()),
        "unaddressed_cell_value": "training-fold base rate",
        "multiplicity": [int(v) for v in mult],
        "ridge": RIDGE,
        "fan_out_cap": FAN_OUT_CAP,
        "n_tables_with_nonzero_fanout": int((mult != 0).sum()),
        "total_fan_out": int(np.abs(mult).sum()),
        "gate": {"radius_angstrom": GATE_RADIUS, "weight": GATE_WEIGHT,
                 "rescaling": "neighbourhood mean matched to the raw score's "
                              "standard deviation",
                 "selected_on": "cluster-disjoint half of the training fold"},
        "operating_point": {"rule": "per-chain top-q by score", "q": q,
                            "train_f1_at_q": f1,
                            "selected_on": "training fold only"},
        "train": {"n_units": int(len(n_res_per)),
                  "n_residues": int(D.shape[0]),
                  "base_rate": float(np.mean(yf)),
                  "manifest_sha256": train_manifest_sha256},
        "code_sha256": code_sha256,
        "fitting": ("cell contents are counts; the integer fan-out is one "
                    "ridge-regularised closed-form solve over the table "
                    "outputs on the training fold, rounded to integers; no "
                    "gradient, no iteration, no test-fold selection"),
    }


def _best_operating_point(s_all, y, n_res_per):
    offs = np.concatenate([[0], np.cumsum(np.asarray(n_res_per, np.int64))])
    best_q, best_f1 = 0.10, -1.0
    for q in np.arange(0.02, 0.41, 0.01):
        tp = fp = fn = 0
        for a, b in zip(offs[:-1], offs[1:]):
            n = int(b - a)
            k = max(1, int(round(q * n)))
            call = np.zeros(n, dtype=bool)
            call[np.argsort(-s_all[a:b], kind="stable")[:k]] = True
            tr = np.asarray(y[a:b], dtype=bool)
            tp += int((call & tr).sum())
            fp += int((call & ~tr).sum())
            fn += int((~call & tr).sum())
        d = 2 * tp + fp + fn
        f1 = (2 * tp / d) if d else 0.0
        if f1 > best_f1:
            best_q, best_f1 = float(q), float(f1)
    return best_q, best_f1


class TableField:
    """A compiled field, ready to score any receptor chain on its own."""

    def __init__(self, doc: dict[str, Any]) -> None:
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"not a {SCHEMA} artifact: {doc.get('schema')}")
        self.doc = doc
        self.tables = [list(t) for t in doc["tables"]]
        self.offsets = cell_offsets(self.tables)
        tot = np.asarray(doc["cell_total"], dtype=np.float64)
        pos = np.asarray(doc["cell_positive"], dtype=np.float64)
        rate = float(doc["train"]["base_rate"])
        self.frac = np.where(tot > 0, pos / np.maximum(tot, 1.0), rate)
        self.mult = np.asarray(doc["multiplicity"], dtype=np.int64)
        self.prop = np.asarray(doc["propensity_table"], dtype=np.float64)
        self.q = float(doc["operating_point"]["q"])

    @staticmethod
    def load(path: str | Path) -> "TableField":
        return TableField(json.loads(Path(path).read_text()))

    def score_matrix(self, X: np.ndarray, ctr: np.ndarray,
                     n_res_per=None) -> np.ndarray:
        if n_res_per is None:
            n_res_per = [X.shape[0]]
        D = chain_digits(np.asarray(X, dtype=np.float64), n_res_per)
        S = score(D, self.tables, self.offsets, self.frac, self.mult)
        return apply_gate(S, np.asarray(ctr, dtype=np.float64), n_res_per)

    def positive_call(self, s: np.ndarray) -> np.ndarray:
        n = len(s)
        k = max(1, int(round(self.q * n)))
        call = np.zeros(n, dtype=bool)
        call[np.argsort(-np.asarray(s), kind="stable")[:k]] = True
        return call

    def score_receptor(self, receptor_pdb: str | Path,
                       chain: str | None = None):
        """``(resseq, score, positive)`` for one receptor chain, end to end."""
        resseq, F, codes, ctr = algebraic_residue_features(
            Path(receptor_pdb), chain=chain)
        n_res_per = np.asarray([len(resseq)], dtype=np.int64)
        X, _names = build_wide(F, codes, ctr, n_res_per,
                               tuple(FEATURE_NAMES), self.prop)
        s = self.score_matrix(X, ctr, n_res_per)
        return resseq, s, self.positive_call(s)


_FIELD_PATH = (Path(__file__).resolve().parents[3]
               / "data/cryptobench_apo/TABLE_FIELD.json")
_CACHED: dict[str, TableField] = {}


def load_field(path: str | Path | None = None) -> TableField:
    """Load the compiled field, fail-closed."""
    p = Path(path) if path else _FIELD_PATH
    key = str(p)
    if key not in _CACHED:
        if not p.exists():
            raise FileNotFoundError(
                f"compiled field missing: {p}\n"
                f"  action: PYTHONPATH=src python3.12 "
                f"tools/compile_table_field.py")
        _CACHED[key] = TableField.load(p)
    return _CACHED[key]


def predict(receptor_pdb: Path, *, pdb_id: str, chain: str | None = None,
            top_k: int = 5, **_ignored: Any) -> dict[str, Any]:
    """Runner-facing entry point, emitting a natively residue-level prediction."""
    import time

    from pocket_bench.methods import prediction
    from pocket_bench.paths import STATUS_CRASH, STATUS_OK

    t0 = time.perf_counter()
    try:
        field = load_field()
        resseq, s, call = field.score_receptor(receptor_pdb, chain)
        order = np.argsort(-s, kind="stable")
        pockets = [{"rank": r + 1, "center_xyz": [0.0, 0.0, 0.0],
                    "score": float(s[i]), "residues": [int(resseq[i])]}
                   for r, i in enumerate(order[:top_k])]
        return prediction(
            method="table_field", pdb_id=pdb_id, status=STATUS_OK,
            pockets=pockets, runtime_s=time.perf_counter() - t0,
            extra={
                "residue_scores": {str(int(r)): float(v)
                                   for r, v in zip(resseq, s)},
                "residue_positive": [int(r) for r, c in zip(resseq, call) if c],
                "n_residues": int(len(resseq)),
                "n_wires": int(field.doc["n_wires"]),
                "n_tables": int(len(field.tables)),
                "operating_q": field.q,
                "protocol": "quaternary_pair_tables_integer_fanout",
            },
        )
    except AssertionError as exc:
        return prediction(method="table_field", pdb_id=pdb_id,
                          status=STATUS_CRASH,
                          runtime_s=time.perf_counter() - t0,
                          error=f"ligand_leak_guard:{exc}")
    except Exception as exc:  # noqa: BLE001
        return prediction(method="table_field", pdb_id=pdb_id,
                          status=STATUS_CRASH,
                          runtime_s=time.perf_counter() - t0,
                          error=str(exc)[-400:])


def code_sha256() -> str:
    """SHA-256 of every source file the field's numbers depend on."""
    here = Path(__file__).resolve().parent
    rels = ["table_field.py", "table_bank.py", "wide_descriptors.py",
            "expanded_descriptors.py", "algebraic_descriptors.py",
            "density_topology.py", "geometric_foundation.py",
            "sequence_wires.py"]
    h = hashlib.sha256()
    for rel in rels:
        p = here / rel
        if p.exists():
            h.update(p.read_bytes())
    for rel in ("spatial.py", "pdb_io.py"):
        p = here.parent / rel
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()
