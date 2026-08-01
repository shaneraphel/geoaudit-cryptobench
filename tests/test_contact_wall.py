"""Smoke tests for the contact-wall family (not yet attached)."""
from pocket_bench.methods.contact_wall import COLUMNS, N_COLUMNS, compute, consistency


def test_column_count_matches():
    assert len(COLUMNS) == N_COLUMNS == 36


def test_two_residue_toy_chain():
    atoms = [
        [{"name": "N", "element": "N", "x": 0, "y": 0, "z": 0},
         {"name": "CA", "element": "C", "x": 1.5, "y": 0, "z": 0},
         {"name": "CB", "element": "C", "x": 2.0, "y": 1.2, "z": 0},
         {"name": "CG", "element": "C", "x": 3.2, "y": 1.2, "z": 0.5}],
        [{"name": "N", "element": "N", "x": 0, "y": 3.5, "z": 0},
         {"name": "CA", "element": "C", "x": 1.5, "y": 3.5, "z": 0},
         {"name": "CB", "element": "C", "x": 2.0, "y": 2.3, "z": 0},
         {"name": "CG", "element": "C", "x": 3.0, "y": 2.3, "z": -0.5}],
    ]
    mat = compute(atoms, [10, 11])
    assert mat.shape == (2, N_COLUMNS)
    assert consistency(mat) == []
    assert (mat[:, 0] > 0).all()
