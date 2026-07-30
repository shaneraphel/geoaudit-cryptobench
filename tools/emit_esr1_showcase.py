#!/usr/bin/env python3.12
"""The ESR1 appendix showcase: complete chemistry fields beside an exact identity.

`clinical_grade=false`. What this demonstrates is a property of the detector, not
a property of any molecule, and the distinction is the whole point of the file.

What is claimed
---------------
Every score this detector produces is a sum of integer table contributions. It can
be taken apart residue by residue and by descriptor family, and the parts add back
to the same number at machine precision --
``results/official_fold/AUDIT_DECOMPOSITION.json`` measures the worst relative
reconstruction error at 5.4e-16 over the four case units a committed tool had
already selected. A three-billion-parameter sequence encoder's score cannot be
taken apart that way. That is a structural difference and not a performance one,
and it is the only comparative statement anywhere near this appendix that no
measurement contradicts.

What is NOT claimed, and why the gate enforces it
-------------------------------------------------
Not that geometric interpretability beats statistical modelling. Candidates are
outputs, not comparisons, and the comparison that was actually run goes the other
way: pLM-NN leads this detector by 0.0243 on the official fold and 0.0340 on the
frozen external set. ``contracts/GEOAUDIT_PAPER_SCOPE.json`` sets
``comparative_claim_allowed`` false for Appendix A, so this document declares
``comparative_claim`` false in its own body and ``verify_claims.py`` fails if it
does not -- which stops the claim drifting back in through a caption.

Not affinity, potency, selectivity, efficacy or toxicity. Nothing in this
repository measures or predicts any of them.

Not that the ESR1 pilot's accuracy numbers are usable.
``results/pilot/RETROSPECTIVE_PILOT_REPORT.json`` declares itself invalidated: its
labels were built by ligand resname alone, merging every crystallographic copy, so
DCA could match the wrong one. That is why this showcase carries no accuracy
number for ESR1 at all. A decomposition is an algebraic identity and needs no
labels -- the score equals the sum of its parts whether or not the residue is a
pocket -- so the identity stands while the pilot's accuracy does not, and saying
which is which is the point.

Why the audit verdicts are carried and not recomputed
----------------------------------------------------
The rule tables behind the structural audit -- stability motifs, phase-I routes,
liability families, chemotype signatures -- live in the companion repository and
are covered by its tests, three of which exist because a pattern there was wrong
on first writing. Copying a SMARTS table into this repository would create two
copies that can drift, and a drifted screen is worse than an absent one. So the
verdict travels as pinned input with the commit that produced it, and this tool
recomputes only what needs no rule table: canonical SMILES, InChIKey, formula, the
bond graph, the topological pharmacophore and the stereocentres. Every one of
those is an RDKit primitive and every one is re-derived here rather than trusted,
so a stored field that no longer matches its own structure fails loudly.

rdkit is an optional dependency of this repository, declared in
``requirements.txt`` as a commented pin. Nothing the primary claim depends on
imports it, and this tool says so and exits if it is absent rather than raising an
import error from somewhere deep.

Usage: PYTHONPATH=src python3.12 tools/emit_esr1_showcase.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "data/appendix_esr1/SHOWCASE_INPUT.json"
OUT = ROOT / "results/appendix_esr1/DECOMPOSABILITY_SHOWCASE.json"
DECOMP = ROOT / "results/official_fold/AUDIT_DECOMPOSITION.json"
PILOT = ROOT / "results/pilot/RETROSPECTIVE_PILOT_REPORT.json"
SCOPE = ROOT / "contracts/GEOAUDIT_PAPER_SCOPE.json"
SCHEMA = "geoaudit.esr1_decomposability_showcase.v1"
SVG_W, SVG_H = 340, 260


def _require_rdkit():
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import Draw, rdMolDescriptors
    except ModuleNotFoundError:
        raise SystemExit(
            "rdkit is not installed. It is an optional dependency of this "
            "repository, needed only by this appendix tool, and is recorded in "
            "requirements.txt as a commented pin. Install that version to "
            "rebuild the showcase; nothing the primary claim depends on needs "
            "it.")
    RDLogger.DisableLog("rdApp.*")
    return Chem, Draw, rdMolDescriptors


def bond_graph_svg(Chem, Draw, mol) -> dict:
    """A valence-bond drawing, as SVG text rather than a raster.

    SVG because it is text: it goes into the JSON, into git, and into the
    manuscript without a rasteriser, a reader can diff two versions of it, and
    nothing about it depends on a font or a DPI. The bond orders and the wedge or
    hash of every stereocentre are what the drawing carries that a SMILES string
    does not put in front of the eye.
    """
    from rdkit.Chem import AllChem
    m = Chem.Mol(mol)
    AllChem.Compute2DCoords(m)
    d = Draw.rdMolDraw2D.MolDraw2DSVG(SVG_W, SVG_H)
    opts = d.drawOptions()
    opts.addStereoAnnotation = True
    d.DrawMolecule(m)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    return {
        "format": "svg",
        "width": SVG_W,
        "height": SVG_H,
        "stereo_annotated": True,
        "n_bytes": len(svg),
        "svg": svg,
    }


def topological_pharmacophore(Chem, mol) -> dict:
    """Feature counts and their separations in bonds.

    Separations are geodesics in the molecular graph. They are not conformational
    distances and no bound pose is implied: this repository asserts no binding
    mode for any molecule, and a graph geodesic is a property of the constitution.
    """
    feats = {
        "hbond_donor": "[$([NX3;H1,H2]),$([OX2H]),$([SX2H]),$([nX3;H1])]",
        "hbond_acceptor": (
            "[$([OX1]=[#6,#7,#16]),$([OX2;H0;!$(O[a]);!$(O=*)]),$([OX2H]),"
            "$([nX2;H0]),$([NX2]=[#6]),$([NX1]#[#6]),"
            "$([NX3;H0;!$(N[a]);!$(NC=O);!$(N=*);!$([N+])])]"),
        "ionisable_basic_N": "[NX3;H0,H1,H2;!$(NC=O);!$(N[a]);!$(N=*);!$(NS=O)]",
        "ionisable_acidic": "[$([CX3](=O)[OX2H1]),$([SX4](=O)(=O)[OX2H1])]",
        "aromatic_ring_atom": "[a]",
    }
    idx: dict[str, list[int]] = {}
    for name, sm in feats.items():
        q = Chem.MolFromSmarts(sm)
        if q is None:
            raise SystemExit(f"pharmacophore SMARTS did not compile: {name}")
        idx[name] = sorted({int(m[0]) for m in mol.GetSubstructMatches(q)})
    dm = Chem.GetDistanceMatrix(mol)
    pairs = {}
    keys = [k for k in idx if idx[k] and k != "aromatic_ring_atom"]
    for i, a in enumerate(keys):
        for b in keys[i:]:
            ds = [int(dm[p][q]) for p in idx[a] for q in idx[b] if p != q]
            if ds:
                pairs[f"{a}__{b}"] = {"min_bonds": min(ds), "max_bonds": max(ds)}
    return {
        "counts": {k: len(v) for k, v in idx.items()},
        "atom_indices": idx,
        "separations_in_bonds": pairs,
        "units": "bonds along the shortest path in the molecular graph",
        "is_a_pose": False,
        "why_topological": "no binding mode is asserted anywhere in this "
                           "repository, so a conformational distance would be a "
                           "claim this work does not support. A graph geodesic is "
                           "a property of the constitution",
    }


def stereochemistry(Chem, mol) -> dict:
    assigned, unassigned = [], []
    for i, label in Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False):
        a = mol.GetAtomWithIdx(i)
        rec = {"atom_index": int(i), "element": a.GetSymbol(),
               "in_ring": bool(a.IsInRing()),
               "neighbour_elements": sorted(n.GetSymbol()
                                            for n in a.GetNeighbors())}
        (unassigned if label == "?" else assigned).append(
            rec if label == "?" else {**rec, "cip_label": label})
    double = []
    for b in mol.GetBonds():
        if b.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if b.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if a1.GetIsAromatic() or a2.GetIsAromatic():
            continue
        subs = (a1.GetDegree() - 1) + (a2.GetDegree() - 1)
        double.append({
            "atoms": [a1.GetIdx(), a2.GetIdx()],
            "stereo": str(b.GetStereo()).replace("STEREO", ""),
            "n_substituents": subs,
            "note": ("a defined geometry on a fully substituted alkene; setting "
                     "it is the hardest step in this scaffold class"
                     if subs >= 4 else None),
        })
    n_u = len(unassigned)
    return {
        "assigned": assigned,
        "unassigned": unassigned,
        "n_assigned": len(assigned),
        "n_unassigned": n_u,
        "n_stereoisomers_this_identifier_covers": 2 ** n_u,
        "defined_double_bond_geometry": double,
        "why_unassigned_would_matter": (
            f"an undefined centre means the identifier names a mixture of "
            f"{2 ** n_u} stereoisomers rather than a compound") if n_u else None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.parse_args(argv)
    Chem, Draw, rdMD = _require_rdkit()

    src = json.loads(IN.read_text())
    decomp = json.loads(DECOMP.read_text())
    pilot = json.loads(PILOT.read_text())
    scope = json.loads(SCOPE.read_text())

    recon = decomp["reconstruction"]
    if not recon.get("agrees"):
        raise SystemExit(
            "AUDIT_DECOMPOSITION.json reports that the decomposition does not "
            "add back to the score; the only thing this showcase is admitted to "
            "demonstrate is not currently true")
    appendix_a = (scope.get("appendices") or {}).get("A") or {}
    if appendix_a.get("comparative_claim_allowed") is not False:
        raise SystemExit(
            "the scope contract no longer forbids comparative claims in "
            "Appendix A; this tool asserts that it does and must be rewritten "
            "deliberately rather than silently")

    out_records = []
    problems = []
    for rec in src["records"]:
        smi = rec["isomeric_smiles"]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            problems.append(f"{rec['candidate_id']}: SMILES did not parse")
            continue
        canonical = Chem.MolToSmiles(mol)
        out_records.append({
            "candidate_id": rec["candidate_id"],
            "modality": rec["modality"],
            "isomeric_smiles": smi,
            "canonical_smiles": canonical,
            "inchi_key": Chem.MolToInchiKey(mol),
            "formula": rdMD.CalcMolFormula(mol),
            "n_heavy_atoms": int(mol.GetNumHeavyAtoms()),
            "n_bonds": int(mol.GetNumBonds()),
            "elements": {a: sum(1 for x in mol.GetAtoms()
                                if x.GetSymbol() == a)
                         for a in sorted({x.GetSymbol()
                                          for x in mol.GetAtoms()})},
            "bond_graph_svg": bond_graph_svg(Chem, Draw, mol),
            "topological_pharmacophore": topological_pharmacophore(Chem, mol),
            "stereochemistry": stereochemistry(Chem, mol),
            "structural_audit": {
                **rec["audit_verdict"],
                "computed_by": src["companion_repository"]["tools"],
                "at_commit": src["companion_repository"]["commit"],
                "why_carried_not_recomputed": src[
                    "why_the_verdicts_are_carried_rather_than_recomputed"],
            },
            "non_claims": [
                "no affinity, potency, selectivity or efficacy is measured or "
                "predicted",
                "no toxicity in any organism is measured or predicted; the "
                "audit reports that no pattern from its named rule families "
                "matched, which is a different statement",
                "no binding mode, pose or docking score is asserted",
                "no accuracy number for ESR1 is offered: the pilot that would "
                "supply one declares itself invalidated",
                "this molecule demonstrates nothing about the detector's "
                "accuracy; the detector's decomposability is what the appendix "
                "is for",
            ],
        })
    if problems:
        raise SystemExit("input problems: " + "; ".join(problems))

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "comparative_claim": False,
        "efficacy_or_affinity_claim": False,
        "repository_is_private": True,
        "decomposition_reconstruction_error_is_recorded": True,
        "what_is_demonstrated": {
            "claim": "every score this detector produces is a sum of integer "
                     "table contributions, can be taken apart residue by "
                     "residue and by descriptor family, and the parts add back "
                     "to the same number at machine precision",
            "measured_worst_relative_error": recon["worst_relative_error"],
            "tolerance": recon["tolerance"],
            "source": str(DECOMP.relative_to(ROOT)),
            "n_case_units": decomp["cases_are_not_chosen_here"]["n_cases"],
            "case_units": decomp["cases_are_not_chosen_here"]["case_ids"],
            "families": decomp["family_assignment"]["families"],
            "why_this_is_structural_and_not_a_performance_claim":
                "a sequence encoder's score is not a sum over interpretable "
                "parts and cannot be decomposed this way at any accuracy. That "
                "is a difference in what the two constructions are, and it is "
                "the only comparative statement near this appendix that no "
                "measurement contradicts",
        },
        "what_is_not_demonstrated": {
            "not_that_geometry_beats_statistics": "candidates are outputs, not "
                "comparisons, and the comparison that was run goes the other "
                "way: pLM-NN leads this detector by 0.0243 on the official fold "
                "and 0.0340 on the frozen external set",
            "scope_contract": {
                "path": str(SCOPE.relative_to(ROOT)),
                "appendix_a_comparative_claim_allowed":
                    appendix_a.get("comparative_claim_allowed"),
                "appendix_a_evidence_level": appendix_a.get("evidence_level"),
            },
            "no_esr1_accuracy_number": {
                "why": "the pilot that would supply one declares itself "
                       "invalidated",
                "pilot": str(PILOT.relative_to(ROOT)),
                "invalidated": pilot.get("invalidated"),
                "invalidated_on": pilot.get("invalidated_on"),
                "reason": pilot.get("invalidation_reason"),
                "what_still_stands": "a decomposition is an algebraic identity "
                                     "and needs no labels: the score equals the "
                                     "sum of its parts whether or not the "
                                     "residue is a pocket. The identity stands "
                                     "while the pilot's accuracy does not",
            },
        },
        "input": {
            "path": str(IN.relative_to(ROOT)),
            "selection_rule": src["selection_rule"],
            "companion_repository": src["companion_repository"],
            "why_two_chemotypes": src["why_two_chemotypes_are_present"],
        },
        "fields_recomputed_here": [
            "canonical_smiles", "inchi_key", "formula", "elements",
            "bond_graph_svg", "topological_pharmacophore", "stereochemistry",
        ],
        "fields_carried_as_pinned_input": ["structural_audit"],
        "n_records": len(out_records),
        "records": out_records,
    }
    out = Path(OUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print(f"{len(out_records)} records")
    for r in out_records:
        st = r["stereochemistry"]
        ph = r["topological_pharmacophore"]["counts"]
        print(f"  {r['candidate_id']:26s} {r['formula']:16s} "
              f"stereo {st['n_assigned']}/{st['n_unassigned']}  "
              f"E-Z {len(st['defined_double_bond_geometry'])}  "
              f"don/acc {ph['hbond_donor']}/{ph['hbond_acceptor']}  "
              f"svg {r['bond_graph_svg']['n_bytes']}B")
    w = doc["what_is_demonstrated"]
    print(f"\n  decomposition worst relative error "
          f"{w['measured_worst_relative_error']:.2e} against a tolerance of "
          f"{w['tolerance']:g}, over {w['n_case_units']} units")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
