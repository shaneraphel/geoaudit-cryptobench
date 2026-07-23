"""Curated non-redundant ESR1 (P03372) LBD panel with genotype + fixed splits.

Ligand fields are for LABEL construction only — never Foliation inputs.
Splits are frozen: development / validation / locked_test.
Unit of analysis = one PDB (not pdb×radius).

Ligand resnames verified against RCSB HETATM (not guessed from literature aliases).
"""
from __future__ import annotations

# (pdb_id, chain, genotype, ligand_resname_or_None_for_apo, split, note)
CURATED_ENTRIES: list[dict] = [
    # —— development ——
    {
        "pdb_id": "1ERE",
        "chain": "A",
        "genotype": "WT",
        "ligand_resname": "EST",
        "split": "development",
        "state": "holo",
        "note": "estradiol WT holo",
    },
    {
        "pdb_id": "1GWR",
        "chain": "A",
        "genotype": "WT",
        "ligand_resname": "EST",
        "split": "development",
        "state": "holo",
        "note": "estradiol WT",
    },
    {
        "pdb_id": "1X7R",
        "chain": "A",
        "genotype": "WT",
        "ligand_resname": "GEN",
        "split": "development",
        "state": "holo",
        "note": "genistein WT",
    },
    {
        "pdb_id": "3UUD",
        "chain": "A",
        "genotype": "Y537S",
        "ligand_resname": "EST",
        "split": "development",
        "state": "holo",
        "note": "Y537S + estradiol",
    },
    {
        "pdb_id": "4Q13",
        "chain": "A",
        "genotype": "D538G",
        "ligand_resname": None,
        "split": "development",
        "state": "apo",
        "note": "D538G apo (receptor for apo–holo control; no ligand label)",
    },
    # —— validation (radius / hparam selection ONLY) ——
    {
        "pdb_id": "2IOK",
        "chain": "A",
        "genotype": "WT",
        "ligand_resname": "IOK",
        "split": "validation",
        "state": "holo",
        "note": "WT antagonist panel",
    },
    {
        "pdb_id": "3HQ5",
        "chain": "A",
        "genotype": "WT",
        "ligand_resname": "GKK",
        "split": "validation",
        "state": "holo",
        "note": "WT validation holo",
    },
    {
        "pdb_id": "5DXB",
        "chain": "A",
        "genotype": "Y537S",
        "ligand_resname": "EST",
        "split": "validation",
        "state": "holo",
        "note": "Y537S validation",
    },
    {
        "pdb_id": "4PXM",
        "chain": "A",
        "genotype": "D538G",
        "ligand_resname": "EST",
        "split": "validation",
        "state": "holo",
        "note": "D538G + estradiol validation",
    },
    # —— locked_test (frozen; no retuning) ——
    {
        "pdb_id": "3ERT",
        "chain": "A",
        "genotype": "WT",
        "ligand_resname": "OHT",
        "split": "locked_test",
        "state": "holo",
        "note": "canonical OHT WT — locked",
    },
    {
        "pdb_id": "3OS8",
        "chain": "A",
        "genotype": "WT",
        "ligand_resname": "KN0",
        "split": "locked_test",
        "state": "holo",
        "note": "WT locked",
    },
    {
        "pdb_id": "1UOM",
        "chain": "A",
        "genotype": "WT",
        "ligand_resname": "PTI",
        "split": "locked_test",
        "state": "holo",
        "note": "WT locked (THIQ ligand)",
    },
    {
        "pdb_id": "5U2B",
        "chain": "A",
        "genotype": "Y537S",
        "ligand_resname": "6WV",
        "split": "locked_test",
        "state": "holo",
        "note": "Y537S locked",
    },
    {
        "pdb_id": "6CHW",
        "chain": "A",
        "genotype": "Y537S",
        "ligand_resname": "F3D",
        "split": "locked_test",
        "state": "holo",
        "note": "Y537S covalent antagonist locked",
    },
    {
        "pdb_id": "4Q50",
        "chain": "A",
        "genotype": "D538G",
        "ligand_resname": "OHT",
        "split": "locked_test",
        "state": "holo",
        "note": "D538G + OHT locked",
    },
]

# Apo–holo control: predict on apo receptor, score against holo ligand labels.
APO_HOLO_PAIRS: list[dict] = [
    {
        "apo_pdb_id": "4Q13",
        "holo_pdb_id": "4Q50",
        "note": "D538G apo receptor vs D538G holo OHT label (leakage stress)",
    },
]
