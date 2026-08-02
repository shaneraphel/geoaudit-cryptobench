"""geometry + nonlocal-seam + Swiss-Prot MSA conservation counting field.

Development detector. clinical_grade = false.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pocket_bench.methods.algebraic_descriptors import (
    FEATURE_NAMES, algebraic_residue_features,
)
from pocket_bench.methods.geometry_wires import aggregate, geometry_columns
from pocket_bench.methods.nonlocal_seam import COLUMNS as SEAM_QTY
from pocket_bench.methods.seam_geometry_field import seam_columns
from pocket_bench.methods.table_field import TableField
from pocket_bench.methods.wide_descriptors import build_wide
from pocket_bench.paths import ROOT

_FIELD_PATH = ROOT / "data/cryptobench_apo/SEAM_MSA_FIELD.json"
_CONS_OFFICIAL = ROOT / "data/msa_cache/official_conservation_full.json"
_CACHED: dict[str, "SeamMsaField"] = {}

MSA_QTY = (
    "match_frac_x100",
    "n_hits_covering",
    "n_hits_matching",
    "best_span",
    "mean_neglog_e",
    "rank_match_frac",
    "rank_n_cover",
)
AGGREGATIONS = ("own", "contact", "walk2")


def _official_msa_raw(unit: str, resseq: list[int],
                      cons_doc: dict) -> np.ndarray:
    """Build the 7 raw MSA quantities for one official unit from full stats."""
    per = cons_doc.get(unit, {})
    n = len(resseq)
    X = np.zeros((n, len(MSA_QTY)), dtype=np.float64)
    for i, r in enumerate(resseq):
        v = per.get(str(int(r)))
        if isinstance(v, dict):
            for j, q in enumerate(MSA_QTY):
                X[i, j] = float(v.get(q, 0.0))
        elif v is not None:
            # legacy scalar [0,1]
            X[i, 0] = 100.0 * float(v)
    return X


def msa_columns(unit: str, receptor_pdb: Path, chain: str,
                resseq: np.ndarray, ctr: np.ndarray,
                cons_doc: dict) -> np.ndarray:
    raw = _official_msa_raw(unit, [int(r) for r in resseq], cons_doc)
    return aggregate(ctr, raw)


class SeamMsaField(TableField):
    def score_receptor(self, receptor_pdb: str | Path,
                       chain: str | None = None, *, unit: str | None = None):
        path = Path(receptor_pdb)
        if chain is None:
            raise ValueError("explicit chain required")
        if unit is None:
            raise ValueError("unit id required for MSA columns")
        resseq, F, codes, ctr = algebraic_residue_features(path, chain=chain)
        n_res_per = np.asarray([len(resseq)], dtype=np.int64)
        Xw, _ = build_wide(F, codes, ctr, n_res_per,
                           tuple(FEATURE_NAMES), self.prop)
        g_resseq, Xg = geometry_columns(path, chain)
        s_resseq, Xs = seam_columns(path, chain)
        if (not np.array_equal(np.asarray(resseq), g_resseq)
                or not np.array_equal(np.asarray(resseq), np.asarray(s_resseq))):
            raise AssertionError("residue universe mismatch")
        cons = json.loads(_CONS_OFFICIAL.read_text())
        Xm = msa_columns(unit, path, chain, np.asarray(resseq), ctr, cons)
        X = np.concatenate([np.asarray(Xw), Xg, Xs, Xm], axis=1)
        s = self.score_matrix(X, ctr, n_res_per)
        return resseq, s, self.positive_call(s)


def load_field(path: str | Path | None = None) -> SeamMsaField:
    p = Path(path) if path else _FIELD_PATH
    key = str(p)
    if key not in _CACHED:
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p}; run tools/compile_seam_msa_field.py")
        _CACHED[key] = SeamMsaField(json.loads(p.read_text()))
    return _CACHED[key]


def predict(receptor_pdb: Path, *, pdb_id: str, chain: str | None = None,
            unit: str | None = None, top_k: int = 5, **_ignored: Any) -> dict:
    import time
    from pocket_bench.methods import prediction
    from pocket_bench.paths import STATUS_CRASH, STATUS_OK

    t0 = time.perf_counter()
    uid = unit or f"{pdb_id}_{chain}"
    try:
        field = load_field()
        resseq, s, call = field.score_receptor(
            receptor_pdb, chain, unit=uid)
        order = np.argsort(-s, kind="stable")
        pockets = [{"rank": r + 1, "center_xyz": [0.0, 0.0, 0.0],
                    "score": float(s[i]), "residues": [int(resseq[i])]}
                   for r, i in enumerate(order[:top_k])]
        return prediction(
            method="seam_msa_field", pdb_id=pdb_id, status=STATUS_OK,
            pockets=pockets, runtime_s=time.perf_counter() - t0,
            extra={
                "residue_scores": {str(int(r)): float(v)
                                   for r, v in zip(resseq, s)},
                "residue_positive": [int(r) for r, c in zip(resseq, call) if c],
                "n_residues": int(len(resseq)),
                "n_wires": int(field.doc["n_wires"]),
                "n_tables": int(len(field.tables)),
                "operating_q": field.q,
            },
        )
    except Exception as exc:  # noqa: BLE001
        return prediction(method="seam_msa_field", pdb_id=pdb_id,
                          status=STATUS_CRASH,
                          runtime_s=time.perf_counter() - t0,
                          error=str(exc)[-400:])
