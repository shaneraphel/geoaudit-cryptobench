"""The algebraic field with a fitted linear readout.

Relationship to ``algebraic_field``
-----------------------------------
Same receptor, same quaternary quantization, same per-chain rank. The two
methods differ in two places, not one, and an earlier version of this docstring
claimed otherwise:

``algebraic_field``          35 invariants, integer-multiplicity sum of dense
                             table fractions
``algebraic_field_linear``   172 wires, one regularised linear functional of
                             the digits

The wire counts are the second difference. The 172 below are 43 quantities ---
the 35 invariants plus 7 published residue constants plus a training propensity
counter --- at four spatial scales, and the counting field sees none of the
extra 137. Measured on a cluster-disjoint half of the training fold, the
readout accounts for 59 percent of the distance between the two and the extra
wires for 41 percent (``results/architecture_sweep/GAP_DECOMPOSITION.json``).
The same-invariant comparison, and the one the manuscript reasons from, is
against the 35-feature Fisher discriminant in
``results/architecture_sweep/FEATURE_CEILING_DIAGNOSIS.json``.

The second is a strictly stronger model class and it is reported as such. It is
NOT training-free: the coefficient vector is fitted, by a single closed-form
solve, on the official training fold. What it does not do is iterate, take a
gradient, differentiate anything, or consult the test fold; the compiled artifact
is a deterministic function of the training bytes.

Why a linear readout beats the table bank here
----------------------------------------------
The obvious answer is cell noise: a width-six quaternary table has 4096 cells,
the training fold has 234838 residues at a 5.8 percent base rate, so a typical
cell holds around fifty residues and three positives and its fraction carries a
relative standard error near 60 percent. On this fold the readout trade is worth
0.024 ROC-AUC (``results/architecture_sweep/FINAL_READOUT_SELECTION.json``).

The obvious answer is not the operative one, and ``tools/gap_decomposition.py``
says why. Raising the tables' capacity by quotienting out a symmetry, which
moves the admissible width from 7 digits to 41, gains on every training split
and nothing on the test fold. Removing the capacity constraint altogether with
one-dimensional tables at 64 levels is far worse than six interaction tables.
Only 0.15 percent of held-out residues address a cell the fit half never
occupied, so the bank is not falling back to the base rate either. What does
account for the readout gap is the fan-out: replacing the algebraic field's
integer Gini rank with the solved integer fan-out ``table_field`` uses recovers
0.015 of it on the same half. The tables are not too small; the six numbers
that add them together are too coarse.

The wires
---------
172 = (35 algebraic and topological invariants + 7 published residue constants +
1 training propensity) x (the residue itself and its 6, 14 and 20 A
neighbourhood means). The propensity counter is compiled on the training fold
and carried into the artifact, so a test residue never contributes a count.

Quantization to four levels per chain, by the residue's own rank inside its own
structure, is kept from ``algebraic_field``: it makes a wire mean the same thing
in a 57-residue chain and a 307-residue chain, and it removes any dependence on
absolute units, which is why no per-structure normalisation constant has to be
carried between structures.
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
from pocket_bench.methods.algebraic_field import N_LEVELS, chain_digits
from pocket_bench.methods.expanded_descriptors import build_expanded

SCHEMA = "geoaudit.algebraic_field_linear.v1"
# Both fixed on a cluster-disjoint half of the training fold, before any test
# residue was scored; see tools/final_readout_select.py.
GATE_RADIUS = 18.0
GATE_WEIGHT = 0.5
RIDGE = 1e-6


def _gate(s: np.ndarray, ctr: np.ndarray, radius: float) -> np.ndarray:
    """Closed-neighbourhood mean of a score field over one chain."""
    n = len(s)
    out = np.empty(n, dtype=np.float64)
    r2 = radius * radius
    for i in range(0, n, 512):
        d2 = ((ctr[i:i + 512, None, :] - ctr[None, :, :]) ** 2).sum(-1)
        a = (d2 <= r2).astype(np.float64)
        out[i:i + 512] = (a @ s) / np.maximum(a.sum(1), 1.0)
    return out


def apply_gate(s: np.ndarray, ctr: np.ndarray) -> np.ndarray:
    """Add back the neighbourhood mean, rescaled to the raw score's spread.

    A cryptic site is a contiguous patch, which is the one fact a per-residue
    functional cannot express. Matching the spread before adding keeps the
    mixing weight meaningful across chains of different size, where the raw
    and smoothed fields have very different dynamic range.
    """
    g = _gate(s, ctr, GATE_RADIUS)
    sd_s, sd_g = float(np.std(s)), float(np.std(g))
    if sd_g <= 0.0:
        return s
    return s + GATE_WEIGHT * g * (sd_s / sd_g)


def _digits_by_chain(X: np.ndarray, n_res_per: np.ndarray) -> np.ndarray:
    offs = np.concatenate([[0], np.cumsum(np.asarray(n_res_per, dtype=np.int64))])
    D = np.empty(X.shape, dtype=np.int64)
    for a, b in zip(offs[:-1], offs[1:]):
        D[a:b] = chain_digits(X[a:b])
    return D


def compile_field(
    X: np.ndarray, y: np.ndarray, ctr: np.ndarray, n_res_per: np.ndarray,
    names: tuple[str, ...], propensity_table: np.ndarray,
    *, code_sha256: str = "", train_manifest_sha256: str = "",
) -> dict[str, Any]:
    """Fit the readout on the training fold: one symmetric solve, no iteration."""
    D = _digits_by_chain(np.asarray(X, dtype=np.float64), n_res_per).astype(float)
    mu, sd = D.mean(0), D.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    A = (D - mu) / sd
    t = np.asarray(y, dtype=np.float64)
    t = t - t.mean()
    G = A.T @ A
    G.flat[:: G.shape[0] + 1] += RIDGE * float(np.trace(G)) / G.shape[0] + 1e-9
    w = np.linalg.solve(G, A.T @ t)

    offs = np.concatenate([[0], np.cumsum(np.asarray(n_res_per, dtype=np.int64))])
    s_all = np.empty(len(y), dtype=np.float64)
    for a, b in zip(offs[:-1], offs[1:]):
        s_all[a:b] = apply_gate(A[a:b] @ w, np.asarray(ctr[a:b], dtype=float))

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

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "n_levels": N_LEVELS,
        "wire_names": [str(n) for n in names],
        "n_wires": int(X.shape[1]),
        "base_feature_names": list(FEATURE_NAMES),
        "propensity_table": [float(v) for v in np.asarray(propensity_table)],
        "digit_mean": [float(v) for v in mu],
        "digit_std": [float(v) for v in sd],
        "coefficients": [float(v) for v in w],
        "ridge": RIDGE,
        "gate": {"radius_angstrom": GATE_RADIUS, "weight": GATE_WEIGHT,
                 "selected_on": "cluster-disjoint half of the training fold"},
        "operating_point": {"rule": "per-chain top-q by score", "q": best_q,
                            "train_f1_at_q": best_f1,
                            "selected_on": "training fold only"},
        "train": {"n_units": int(len(n_res_per)), "n_residues": int(X.shape[0]),
                  "base_rate": float(np.mean(y)),
                  "manifest_sha256": train_manifest_sha256},
        "code_sha256": code_sha256,
        "fitting": ("one regularised closed-form solve of a 172x172 symmetric "
                    "system on the training fold; no gradient, no iteration, "
                    "no auto-differentiation, no test-fold selection"),
    }


class AlgebraicFieldLinear:
    """A compiled readout, ready to score any receptor chain independently."""

    def __init__(self, doc: dict[str, Any]) -> None:
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"not an {SCHEMA} artifact: {doc.get('schema')}")
        self.doc = doc
        self.mu = np.asarray(doc["digit_mean"], dtype=np.float64)
        self.sd = np.asarray(doc["digit_std"], dtype=np.float64)
        self.w = np.asarray(doc["coefficients"], dtype=np.float64)
        self.prop = np.asarray(doc["propensity_table"], dtype=np.float64)
        self.q = float(doc["operating_point"]["q"])

    @staticmethod
    def load(path: str | Path) -> "AlgebraicFieldLinear":
        return AlgebraicFieldLinear(json.loads(Path(path).read_text()))

    def score_matrix(self, X: np.ndarray, ctr: np.ndarray) -> np.ndarray:
        D = chain_digits(np.asarray(X, dtype=np.float64)).astype(np.float64)
        return apply_gate((D - self.mu) / self.sd @ self.w,
                          np.asarray(ctr, dtype=np.float64))

    def positive_call(self, score: np.ndarray) -> np.ndarray:
        n = len(score)
        k = max(1, int(round(self.q * n)))
        call = np.zeros(n, dtype=bool)
        call[np.argsort(-np.asarray(score), kind="stable")[:k]] = True
        return call

    def score_receptor(self, receptor_pdb: str | Path, chain: str | None = None):
        """``(resseq, score, positive)`` for one receptor chain, end to end."""
        resseq, F, codes, ctr = algebraic_residue_features(
            Path(receptor_pdb), chain=chain)
        n_res_per = np.asarray([len(resseq)], dtype=np.int64)
        X, _names, _p = build_expanded(F, codes, ctr, n_res_per,
                                       tuple(FEATURE_NAMES),
                                       prop_table=self.prop)
        s = self.score_matrix(X, ctr)
        return resseq, s, self.positive_call(s)


_FIELD_PATH = (Path(__file__).resolve().parents[3]
               / "data/cryptobench_apo/ALGEBRAIC_FIELD_LINEAR.json")
_CACHED: dict[str, AlgebraicFieldLinear] = {}


def load_field(path: str | Path | None = None) -> AlgebraicFieldLinear:
    """Load the compiled readout, fail-closed."""
    p = Path(path) if path else _FIELD_PATH
    key = str(p)
    if key not in _CACHED:
        if not p.exists():
            raise FileNotFoundError(
                f"compiled field missing: {p}\n"
                f"  action: PYTHONPATH=src python3.12 "
                f"tools/compile_algebraic_field_linear.py")
        _CACHED[key] = AlgebraicFieldLinear.load(p)
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
        resseq, score, call = field.score_receptor(receptor_pdb, chain)
        order = np.argsort(-score, kind="stable")
        pockets = [{"rank": r + 1, "center_xyz": [0.0, 0.0, 0.0],
                    "score": float(score[i]), "residues": [int(resseq[i])]}
                   for r, i in enumerate(order[:top_k])]
        return prediction(
            method="algebraic_field_linear", pdb_id=pdb_id, status=STATUS_OK,
            pockets=pockets, runtime_s=time.perf_counter() - t0,
            extra={
                "residue_scores": {str(int(r)): float(s)
                                   for r, s in zip(resseq, score)},
                "residue_positive": [int(r) for r, c in zip(resseq, call) if c],
                "n_residues": int(len(resseq)),
                "n_wires": int(len(field.w)),
                "operating_q": field.q,
                "protocol": "algebraic_invariants_quaternary_linear_readout",
            },
        )
    except AssertionError as exc:
        return prediction(method="algebraic_field_linear", pdb_id=pdb_id,
                          status=STATUS_CRASH,
                          runtime_s=time.perf_counter() - t0,
                          error=f"ligand_leak_guard:{exc}")
    except Exception as exc:  # noqa: BLE001
        return prediction(method="algebraic_field_linear", pdb_id=pdb_id,
                          status=STATUS_CRASH,
                          runtime_s=time.perf_counter() - t0,
                          error=str(exc)[-400:])


def code_sha256() -> str:
    """SHA-256 of every source file the readout's numbers depend on."""
    here = Path(__file__).resolve().parent
    rels = ["algebraic_field_linear.py", "algebraic_field.py",
            "algebraic_descriptors.py", "expanded_descriptors.py",
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
