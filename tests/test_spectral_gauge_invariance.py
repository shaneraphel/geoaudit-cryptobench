"""The gauge self-test must catch a gauge-dependent column, or it proves nothing.

AGENT_MEMORY's rule about gates: a gate that has never failed is
indistinguishable from one that cannot fail. ``spectral_gauge.gauge_selftest``
returns ``gauge_invariant: True`` on the real family, and that statement is only
worth something if the same test returns ``False`` on a column that is genuinely
gauge-dependent.

The violation planted here is ``sign(v_2(i))`` — the Fiedler sign, which is the
canonical example of a quantity that flips wholesale when the eigensolver
returns the negated vector, and precisely the quantity this family was built to
avoid. It is planted in memory by wrapping ``compute``; no source file is
modified and nothing is written to disk.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pocket_bench.paths import ROOT
from pocket_bench.pdb_io import parse_pdb_atoms

SKIP = frozenset({"HOH", "WAT", "DOD"})
MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"


def _one_chain():
    man = json.loads(MANIFEST.read_text())
    e = man["entries"][0]
    atoms = parse_pdb_atoms((ROOT / e["receptor_path"]).read_text())
    poly: dict = {}
    for a in atoms:
        if (a["chain"] != e["chain"] or a["element"] == "H"
                or a["resname"] in SKIP):
            continue
        poly.setdefault((a["resseq"], a["icode"].strip()), []).append(a)
    order = sorted(poly)
    return [poly[k] for k in order], [k[0] for k in order]


@pytest.mark.skipif(not MANIFEST.exists(), reason="fold data not present")
def test_family_is_gauge_invariant():
    from pocket_bench.methods import spectral_gauge as sg
    abr, rs = _one_chain()
    st = sg.gauge_selftest(abr, rs, seed=7)
    assert st["gauge_invariant"], st["columns_failing_sign_flip"]
    assert st["max_abs_deviation"] == 0.0


@pytest.mark.skipif(not MANIFEST.exists(), reason="fold data not present")
def test_selftest_catches_a_planted_gauge_violation(monkeypatch):
    from pocket_bench.methods import spectral_gauge as sg
    abr, rs = _one_chain()
    original = sg.compute

    def planted(atoms_by_res, resseqs):
        X = original(atoms_by_res, resseqs)
        n = len(atoms_by_res)
        ca = []
        for res in atoms_by_res:
            hit = next((a for a in res
                        if (a.get("name") or "").strip().upper() == "CA"), None)
            if hit is None:
                return X
            ca.append((hit["x"], hit["y"], hit["z"]))
        pts = np.asarray(ca, dtype=float)
        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        A = (d <= 8.0).astype(float)
        w, V = np.linalg.eigh(np.diag(A.sum(1)) - A)
        # The Fiedler sign: flips wholesale with the eigenvector's sign.
        X[:, 3] = np.sign(V[:, int(np.argsort(w)[1])])
        assert len(X) == n
        return X

    monkeypatch.setattr(sg, "compute", planted)
    st = sg.gauge_selftest(abr, rs, seed=7)
    assert not st["gauge_invariant"], (
        "the self-test passed a column equal to sign(v_2), so it cannot "
        "detect gauge dependence and its True on the real family is worthless")
    assert sg.COLUMNS[3] in st["columns_failing_sign_flip"]
    assert st["max_abs_deviation"] >= 1.0
