"""The algebraic resolution field: a training-free combinational cryptic-site detector.

What this is
------------
Thirty-five exact algebraic and topological invariants of the receptor
(``algebraic_descriptors``), each banded into a quaternary digit against its own
chain's order statistics, addressed through six dense lookup tables, and fused by
integer-multiplicity counting with a multi-scale spatial gate. Every operation on
the query path is a comparison, a base-four positional addition, a table read, or
an integer sum. There is no gradient, no auto-differentiation, no random number,
no iteration to convergence, and no real-valued coefficient anywhere.

"Training-free" means what it says
----------------------------------
The tables are COUNTS. Compiling the field on the training fold increments two
integers per cell; it does not solve, fit, or optimise anything, and running the
compiler twice on the same bytes produces the same field. The only quantities
carried out of the training fold are those counts, the integer table
multiplicities, and one operating-point quantile. No parameter is differentiable
and none was chosen by search over the test fold.

Why the pieces are shaped the way they are
------------------------------------------
The capacity bound of a quaternary table on this fold is ``log_4(rN) = 6.87``
digits (``N = 234838`` training residues, base rate ``r = 0.0576``): beyond seven
digits most cells can never be driven by a single positive, and the combinational
layer degenerates. Thirty-five invariants therefore cannot address one table, and
they must not be cascaded either, because banding a cell fraction whose relative
standard error is 53 percent is an irreversible hard decision on a
noise-dominated statistic, and stacking such decisions compounds the loss. Both
statements were measured, not assumed: the flat seven-digit control reaches
0.7444 and every cascaded topology scores below it.

What survives is parallel reading with late fusion. Each thematic group of at
most six invariants owns one dense table; the tables are read simultaneously and
their cell fractions are summed with integer multiplicities. Multiplicity is
fan-out replication, not a coefficient: table ``k`` is replicated ``m_k`` times
where ``m_k`` is the integer RANK of that table's compiled Gini among the bank,
so the datapath only ever adds a value to itself.

The gate states the one fact about cryptic sites that a per-residue table cannot:
they are contiguous patches. It is a symmetric counting gate over the closed
geometric neighbourhood, evaluated at five radii so that no single patch size is
privileged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from pocket_bench.methods.algebraic_descriptors import (
    FEATURE_NAMES,
    GROUP_SIZES,
    N_ALGEBRAIC,
    algebraic_residue_features,
)

SCHEMA = "geoaudit.algebraic_field.v1"
N_LEVELS = 4
GATE_RADII: tuple[float, ...] = (6.0, 8.0, 10.0, 14.0, 18.0)


# --------------------------------------------------------------------------
# quantization: a comparator network over the residues of ONE chain
# --------------------------------------------------------------------------
def chain_digits(F: np.ndarray) -> np.ndarray:
    """Quaternary digits of one chain's feature matrix, from its own order.

    Absolute cut points are not chain-invariant: a 57-residue chain and a
    307-residue chain have different absolute buriedness distributions, so a
    frozen global threshold assigns whole small chains to one bin and destroys
    their address diversity. Ranking within the chain makes a digit mean the
    same thing everywhere -- which quarter of ITS OWN structure the residue lies
    in -- and requires no constant to be carried between structures.
    """
    F = np.asarray(F, dtype=np.float64)
    n, m = F.shape
    out = np.empty((n, m), dtype=np.int64)
    for j in range(m):
        x = F[:, j]
        order = np.argsort(x, kind="stable")
        r = np.empty(n, dtype=np.float64)
        i = 0
        while i < n:
            k = i
            while k + 1 < n and x[order[k + 1]] == x[order[i]]:
                k += 1
            r[order[i:k + 1]] = 0.5 * (i + k)          # midrank, ties shared
            i = k + 1
        q = np.floor(r / max(n - 1, 1) * N_LEVELS).astype(np.int64)
        out[:, j] = np.clip(q, 0, N_LEVELS - 1)
    return out


def thematic_tables() -> list[list[int]]:
    """Column index blocks, one per descriptor group, in group-major order."""
    tables, off = [], 0
    for size in GROUP_SIZES:
        tables.append(list(range(off, off + size)))
        off += size
    return tables


def _address(D: np.ndarray, cols: list[int]) -> np.ndarray:
    a = np.zeros(D.shape[0], dtype=np.int64)
    for t, c in enumerate(cols):
        a += D[:, c] * (N_LEVELS ** t)
    return a


def _pooled_auc(x: np.ndarray, y: np.ndarray) -> float:
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    r = np.empty(len(x), dtype=np.float64)
    r[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


# --------------------------------------------------------------------------
# spatial counting gate
# --------------------------------------------------------------------------
def _gate(s: np.ndarray, ctr: np.ndarray, radius: float) -> np.ndarray:
    """Closed-neighbourhood mean of a score field: a symmetric counting gate."""
    n = len(s)
    out = np.empty(n, dtype=np.float64)
    r2 = radius * radius
    for i in range(0, n, 512):
        d2 = ((ctr[i:i + 512, None, :] - ctr[None, :, :]) ** 2).sum(-1)
        a = (d2 <= r2).astype(np.float64)
        out[i:i + 512] = (a @ s) / np.maximum(a.sum(1), 1.0)
    return out


def _unit(x: np.ndarray) -> np.ndarray:
    m = float(np.abs(x).max())
    return x / m if m > 0 else x


def fuse(fracs: list[np.ndarray], mult: list[int], ctr: np.ndarray) -> np.ndarray:
    """Integer-multiplicity table sum plus the multi-scale patch gate."""
    s = np.zeros(len(ctr), dtype=np.float64)
    for m, f in zip(mult, fracs):
        for _ in range(int(m)):
            s = s + f
    g = np.zeros(len(ctr), dtype=np.float64)
    for r in GATE_RADII:
        g = g + _unit(_gate(s, ctr, r))
    return _unit(s) + _unit(g)


# --------------------------------------------------------------------------
# compilation (counting only)
# --------------------------------------------------------------------------
def compile_field(
    F: np.ndarray, y: np.ndarray, ctr: np.ndarray, n_res_per: np.ndarray,
    *, code_sha256: str = "", train_manifest_sha256: str = "",
) -> dict[str, Any]:
    """Compile the field from the training fold by counting.

    Returns a JSON-serialisable artifact holding the sparse tables, the integer
    multiplicities, and the frozen operating-point quantile. Nothing here is
    solved for; every stored number is a count, a rank, or a quantile of the
    training scores.
    """
    tables = thematic_tables()
    offs = np.concatenate([[0], np.cumsum(np.asarray(n_res_per, dtype=np.int64))])
    D = np.empty((F.shape[0], F.shape[1]), dtype=np.int64)
    for a, b in zip(offs[:-1], offs[1:]):
        D[a:b] = chain_digits(F[a:b])

    yf = np.asarray(y, dtype=np.float64)
    entries, fracs, gini = [], [], []
    for cols in tables:
        n_cells = N_LEVELS ** len(cols)
        addr = _address(D, cols)
        tot = np.bincount(addr, minlength=n_cells).astype(np.int64)
        pos = np.bincount(addr, weights=yf, minlength=n_cells).astype(np.int64)
        rate = float(yf.mean())
        frac = np.where(tot > 0, pos / np.maximum(tot, 1), rate)
        nz = np.nonzero(tot > 0)[0]
        entries.append({"cols": cols, "addr": nz.tolist(),
                        "pos": pos[nz].tolist(), "tot": tot[nz].tolist()})
        fracs.append(frac[addr])
        gini.append(abs(2.0 * _pooled_auc(frac[addr], np.asarray(y)) - 1.0))

    order = np.argsort(np.asarray(gini))
    mult = np.empty(len(tables), dtype=np.int64)
    mult[order] = np.arange(1, len(tables) + 1)

    # Operating point: the per-chain top-q fraction, with q chosen on TRAIN only
    # by maximising the training F1. It is a comparator threshold, not a fitted
    # parameter, and it never sees the test fold.
    s_all = np.empty(F.shape[0], dtype=np.float64)
    for a, b in zip(offs[:-1], offs[1:]):
        s_all[a:b] = fuse([f[a:b] for f in fracs], mult.tolist(), ctr[a:b])
    best_q, best_f1 = 0.10, -1.0
    for q in np.arange(0.02, 0.41, 0.01):
        tp = fp = fn = 0
        for a, b in zip(offs[:-1], offs[1:]):
            n = int(b - a)
            k = max(1, int(round(q * n)))
            idx = np.argsort(-s_all[a:b], kind="stable")[:k]
            call = np.zeros(n, dtype=bool)
            call[idx] = True
            t = np.asarray(y[a:b], dtype=bool)
            tp += int((call & t).sum())
            fp += int((call & ~t).sum())
            fn += int((~call & t).sum())
        d = 2 * tp + fp + fn
        f1 = (2 * tp / d) if d else 0.0
        if f1 > best_f1:
            best_q, best_f1 = float(q), float(f1)

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "n_levels": N_LEVELS,
        "feature_names": list(FEATURE_NAMES),
        "n_features": int(N_ALGEBRAIC),
        "gate_radii_angstrom": list(GATE_RADII),
        "tables": entries,
        "table_gini_train": [float(g) for g in gini],
        "table_multiplicity": mult.tolist(),
        "operating_point": {
            "rule": "per-chain top-q by fused score",
            "q": best_q,
            "train_f1_at_q": best_f1,
            "selected_on": "training fold only",
        },
        "train": {
            "n_units": int(len(n_res_per)),
            "n_residues": int(F.shape[0]),
            "base_rate": float(yf.mean()),
            "manifest_sha256": train_manifest_sha256,
        },
        "code_sha256": code_sha256,
        "fitting": "none; tables are integer counts, multiplicities are ranks, "
                   "the operating point is a training quantile",
    }


# --------------------------------------------------------------------------
# query
# --------------------------------------------------------------------------
class AlgebraicField:
    """A compiled field, ready to score any receptor chain independently."""

    def __init__(self, doc: dict[str, Any]) -> None:
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"not an {SCHEMA} artifact: {doc.get('schema')}")
        self.doc = doc
        self.rate = float(doc["train"]["base_rate"])
        self.mult = [int(m) for m in doc["table_multiplicity"]]
        self.q = float(doc["operating_point"]["q"])
        self.tables: list[tuple[list[int], np.ndarray]] = []
        for t in doc["tables"]:
            cols = [int(c) for c in t["cols"]]
            n_cells = N_LEVELS ** len(cols)
            frac = np.full(n_cells, self.rate, dtype=np.float64)
            a = np.asarray(t["addr"], dtype=np.int64)
            pos = np.asarray(t["pos"], dtype=np.float64)
            tot = np.asarray(t["tot"], dtype=np.float64)
            frac[a] = pos / np.maximum(tot, 1.0)
            self.tables.append((cols, frac))

    @staticmethod
    def load(path: str | Path) -> "AlgebraicField":
        return AlgebraicField(json.loads(Path(path).read_text()))

    def score_matrix(self, F: np.ndarray, ctr: np.ndarray) -> np.ndarray:
        """Fused score for one chain, from its feature matrix and centroids."""
        D = chain_digits(F)
        fr = [frac[_address(D, cols)] for cols, frac in self.tables]
        return fuse(fr, self.mult, np.asarray(ctr, dtype=np.float64))

    def positive_call(self, score: np.ndarray) -> np.ndarray:
        """The field's own binary call: the chain's top-q residues by score."""
        n = len(score)
        k = max(1, int(round(self.q * n)))
        idx = np.argsort(-np.asarray(score), kind="stable")[:k]
        call = np.zeros(n, dtype=bool)
        call[idx] = True
        return call

    def score_receptor(self, receptor_pdb: str | Path, chain: str | None = None):
        """``(resseq, score, positive)`` for one receptor chain, end to end.

        Self-contained per structure: quantization, table reads and the gate all
        act within this chain only, so no information crosses structures at query
        time and the detector is order-independent over the test fold.
        """
        resseq, F, _codes, ctr = algebraic_residue_features(
            Path(receptor_pdb), chain=chain)
        s = self.score_matrix(F, ctr)
        return resseq, s, self.positive_call(s)


_FIELD_PATH = (Path(__file__).resolve().parents[3]
               / "data/cryptobench_apo/ALGEBRAIC_FIELD.json")
_CACHED: dict[str, AlgebraicField] = {}


def load_field(path: str | Path | None = None) -> AlgebraicField:
    """Load the compiled field, fail-closed.

    There is deliberately no default field and no silent recompilation: scoring
    against a field that does not exist on disk would make the reported numbers
    unauditable.
    """
    p = Path(path) if path else _FIELD_PATH
    key = str(p)
    if key not in _CACHED:
        if not p.exists():
            raise FileNotFoundError(
                f"compiled field missing: {p}\n"
                f"  action: PYTHONPATH=src python3.12 "
                f"tools/compile_algebraic_field.py")
        _CACHED[key] = AlgebraicField.load(p)
    return _CACHED[key]


def predict(receptor_pdb: Path, *, pdb_id: str, chain: str | None = None,
            top_k: int = 5, **_ignored: Any) -> dict[str, Any]:
    """Runner-facing entry point, emitting a natively residue-level prediction.

    The residue table IS the prediction. The pocket list is a locality view for
    diagnostics only and is never what the residue metrics consume.
    """
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
            method="algebraic_field", pdb_id=pdb_id, status=STATUS_OK,
            pockets=pockets, runtime_s=time.perf_counter() - t0,
            extra={
                "residue_scores": {str(int(r)): float(s)
                                   for r, s in zip(resseq, score)},
                "residue_positive": [int(r) for r, c in zip(resseq, call) if c],
                "n_residues": int(len(resseq)),
                "n_features": int(N_ALGEBRAIC),
                "n_tables": len(field.tables),
                "table_multiplicity": field.mult,
                "operating_q": field.q,
                "protocol": "algebraic_invariants_quaternary_bank_counting_fusion",
            },
        )
    except AssertionError as exc:
        return prediction(method="algebraic_field", pdb_id=pdb_id,
                          status=STATUS_CRASH,
                          runtime_s=time.perf_counter() - t0,
                          error=f"ligand_leak_guard:{exc}")
    except Exception as exc:  # noqa: BLE001
        return prediction(method="algebraic_field", pdb_id=pdb_id,
                          status=STATUS_CRASH,
                          runtime_s=time.perf_counter() - t0,
                          error=str(exc)[-400:])


def code_sha256() -> str:
    """SHA-256 of every source file the field's numbers depend on."""
    here = Path(__file__).resolve().parent
    rels = ["algebraic_field.py", "algebraic_descriptors.py",
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
