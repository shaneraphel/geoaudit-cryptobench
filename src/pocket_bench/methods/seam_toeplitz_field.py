"""geometry + spectral seam + toeplitz_3d counting field. clinical_grade=false."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from pocket_bench.methods.algebraic_descriptors import (
    FEATURE_NAMES, algebraic_residue_features,
)
from pocket_bench.methods.geometry_wires import aggregate, geometry_columns, residue_rows
from pocket_bench.methods.seam_geometry_field import seam_columns
from pocket_bench.methods.table_field import TableField
from pocket_bench.methods.toeplitz_3d import COLUMNS as TQ, compute as tz_compute
from pocket_bench.methods.wide_descriptors import build_wide
from pocket_bench.pdb_io import parse_pdb_atoms
from pocket_bench.paths import ROOT

_FIELD = ROOT / "data/cryptobench_apo/SEAM_TOEPLITZ_FIELD.json"
_CACHED: dict[str, "SeamToeplitzField"] = {}
SKIP = frozenset({"HOH", "WAT", "DOD"})
AGGS = ("own", "contact", "walk2")


def toeplitz_columns(path: Path, chain: str):
    atoms = parse_pdb_atoms(path.read_text())
    order, ctr, take = residue_rows(atoms, chain)
    poly = {}
    for a in atoms:
        if a["chain"] != chain or a["element"] == "H" or a["resname"] in SKIP:
            continue
        poly.setdefault((a["resseq"], a["icode"].strip()), []).append(a)
    raw = tz_compute([poly[k] for k in order], [k[0] for k in order])[take]
    X = aggregate(ctr, raw)
    resseq = np.asarray([r for r, _ in sorted({(r, "") for r, _ in order})],
                        dtype=np.int64)
    return resseq, X


class SeamToeplitzField(TableField):
    def score_receptor(self, receptor_pdb, chain=None):
        path = Path(receptor_pdb)
        if chain is None:
            raise ValueError("chain required")
        resseq, F, codes, ctr = algebraic_residue_features(path, chain=chain)
        n_res_per = np.asarray([len(resseq)], dtype=np.int64)
        Xw, _ = build_wide(F, codes, ctr, n_res_per, tuple(FEATURE_NAMES), self.prop)
        g_res, Xg = geometry_columns(path, chain)
        s_res, Xs = seam_columns(path, chain)
        t_res, Xt = toeplitz_columns(path, chain)
        if (not np.array_equal(np.asarray(resseq), g_res)
                or not np.array_equal(np.asarray(resseq), np.asarray(s_res))
                or not np.array_equal(np.asarray(resseq), t_res)):
            raise AssertionError("residue mismatch")
        X = np.concatenate([np.asarray(Xw), Xg, Xs, Xt], axis=1)
        s = self.score_matrix(X, ctr, n_res_per)
        return resseq, s, self.positive_call(s)


def load_field(path=None):
    p = Path(path) if path else _FIELD
    if str(p) not in _CACHED:
        _CACHED[str(p)] = SeamToeplitzField(json.loads(p.read_text()))
    return _CACHED[str(p)]


def predict(receptor_pdb: Path, *, pdb_id: str, chain=None, top_k=5, **_):
    import time
    from pocket_bench.methods import prediction
    from pocket_bench.paths import STATUS_CRASH, STATUS_OK
    t0 = time.perf_counter()
    try:
        field = load_field()
        # score once
        path = Path(receptor_pdb)
        resseq, F, codes, ctr = algebraic_residue_features(path, chain=chain)
        n_res_per = np.asarray([len(resseq)], dtype=np.int64)
        Xw, _ = build_wide(F, codes, ctr, n_res_per, tuple(FEATURE_NAMES), field.prop)
        _, Xg = geometry_columns(path, chain)
        _, Xs = seam_columns(path, chain)
        _, Xt = toeplitz_columns(path, chain)
        X = np.concatenate([np.asarray(Xw), Xg, Xs, Xt], axis=1)
        s = field.score_matrix(X, ctr, n_res_per)
        call = field.positive_call(s)
        order = np.argsort(-s, kind="stable")
        pockets = [{"rank": r + 1, "center_xyz": [0.0, 0.0, 0.0],
                    "score": float(s[i]), "residues": [int(resseq[i])]}
                   for r, i in enumerate(order[:top_k])]
        return prediction(
            method="seam_toeplitz_field", pdb_id=pdb_id, status=STATUS_OK,
            pockets=pockets, runtime_s=time.perf_counter() - t0,
            extra={"residue_scores": {str(int(r)): float(v) for r, v in zip(resseq, s)},
                   "n_residues": int(len(resseq)), "n_wires": int(field.doc["n_wires"])},
        )
    except Exception as exc:  # noqa: BLE001
        return prediction(method="seam_toeplitz_field", pdb_id=pdb_id,
                          status=STATUS_CRASH, runtime_s=time.perf_counter() - t0,
                          error=str(exc)[-400:])
