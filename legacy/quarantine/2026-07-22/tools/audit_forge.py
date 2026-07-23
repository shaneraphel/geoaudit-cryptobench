#!/usr/bin/env python3.12
"""One-click, deterministic descriptor audit for sourced reference SMILES.

This tool computes reproducible RDKit topology descriptors, PAINS catalog
matches, Murcko scaffolds, fingerprint hashes, and (when installed) the RDKit
Contrib synthetic-accessibility heuristic.

It intentionally does *not* claim patent clearance, freedom to operate,
biological activity, or clinical readiness. Those conclusions cannot be
derived from Fsp3, HAC, fingerprints, PAINS filters, or a similarity cutoff.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "inputs/public_reference_smiles.json"
DEFAULT_JSON = ROOT / "validation/W12_PUBLIC_REFERENCE_TOPOLOGY_AUDIT.json"
DEFAULT_MARKDOWN = ROOT / "Foliation_ER100_Public_Audit_Ledger.md"
NCBI_LOG = ROOT / "logs/ncbi_batch_fetch.json"
ACCESSION_MAP = ROOT / "inputs/w12_ncbi_refseq_100_accessions.json"
W12_LEDGER = ROOT / "validation/W12_COSMIC_100_LEDGER.json"
EXACT_FORM_LOG = ROOT / "tensors/exact_form_null.log"


def _pains_catalog() -> FilterCatalog:
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog(params)


def _sa_score(molecule: Chem.Mol) -> tuple[float | None, str]:
    try:
        from rdkit.Contrib.SA_Score import sascorer
    except ImportError:
        return None, "UNAVAILABLE"
    return round(float(sascorer.calculateScore(molecule)), 4), "HEURISTIC"


def _fingerprint_hashes(molecule: Chem.Mol) -> dict[str, str]:
    ecfp4 = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    ).GetFingerprint(molecule)
    maccs = MACCSkeys.GenMACCSKeys(molecule)
    return {
        "ecfp4_2048_sha256": hashlib.sha256(
            DataStructs.BitVectToBinaryText(ecfp4)
        ).hexdigest(),
        "maccs_167_sha256": hashlib.sha256(
            DataStructs.BitVectToBinaryText(maccs)
        ).hexdigest(),
    }


def _audit_record(
    record: dict[str, Any],
    catalog: FilterCatalog,
) -> dict[str, Any]:
    smiles = str(record["smiles"])
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {
            **record,
            "state": "INVALID_TOPOLOGY",
            "ip_clearance_state": "NOT_ASSESSED",
        }

    formula = rdMolDescriptors.CalcMolFormula(molecule)
    exact_mass = float(Descriptors.ExactMolWt(molecule))
    sa_score, sa_state = _sa_score(molecule)
    pains = tuple(
        sorted(
            match.GetDescription()
            for match in catalog.GetMatches(molecule)
        )
    )
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    scaffold_smiles = Chem.MolToSmiles(scaffold, isomericSmiles=True)
    result = {
        **record,
        "canonical_isomeric_smiles": Chem.MolToSmiles(
            molecule,
            isomericSmiles=True,
        ),
        "formula": formula,
        "fsp3": round(float(rdMolDescriptors.CalcFractionCSP3(molecule)), 4),
        "hac": int(molecule.GetNumHeavyAtoms()),
        "exact_mass": round(exact_mass, 8),
        "murcko_scaffold_smiles": scaffold_smiles,
        "pains_alert_count": len(pains),
        "pains_alerts": pains,
        "sa_score": sa_score,
        "sa_score_state": sa_state,
        "source_formula_match": formula == record.get("declared_formula"),
        "source_exact_mass_match": abs(
            exact_mass - float(record.get("declared_exact_mass", exact_mass))
        ) < 1e-6,
        "state": "TOPOLOGY_VERIFIED",
        "ip_clearance_state": "NOT_ASSESSED_REQUIRES_LEGAL_CLAIM_CHART",
        "similarity_clearance_state": "NOT_ASSESSED_NO_REFERENCE_SET",
        **_fingerprint_hashes(molecule),
    }
    return result


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Foliation ER100: Public Audit & Verification Ledger",
        "",
        "## Scope",
        "",
        "This ledger reports deterministic RDKit descriptors for sourced public "
        "reference compounds and reproducible NCBI/RefSeq provenance. It does "
        "**not** establish patent clearance, freedom to operate, efficacy, "
        "causality, or clinical readiness. `clinical_grade=false`.",
        "",
        "## Reference topology audit",
        "",
        "| Target | Compound | Formula | Fsp3 | HAC | Exact mass | PAINS | SA heuristic | State |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    lines.extend(
        "| {target} | [{compound}]({url}) | {formula} | {fsp3:.4f} | "
        "{hac} | {mass:.8f} | {pains} | {sa} | {state} |".format(
            target=row["target"],
            compound=row["compound"],
            url=row["source_url"],
            formula=row.get("formula", "n/a"),
            fsp3=float(row.get("fsp3", 0.0)),
            hac=int(row.get("hac", 0)),
            mass=float(row.get("exact_mass", 0.0)),
            pains=int(row.get("pains_alert_count", 0)),
            sa=(
                "n/a"
                if row.get("sa_score") is None
                else f"{float(row['sa_score']):.4f}"
            ),
            state=row["state"],
        )
        for row in report["records"]
    )
    lines.extend(
        [
            "",
            "## Representative RefSeq windows",
            "",
            "| Target label | RefSeq accession | Window strategy | Public source |",
            "|---|---|---|---|",
        ]
    )
    lines.extend(
        "| {target} | {accession} | {strategy} | [NCBI]({url}) |".format(
            target=row["target_name"],
            accession=row["refseq_accession"],
            strategy=row["window"]["strategy"],
            url=row["source_url"],
        )
        for row in report["provenance"]["refseq_examples"]
    )
    lines.extend(
        [
            "",
            "## Reproducible source artifacts",
            "",
            "- NCBI batch log: `logs/ncbi_batch_fetch.json`",
            "- RefSeq accession map: `inputs/w12_ncbi_refseq_100_accessions.json`",
            "- Fixed 100x48 FASTA: `inputs/w12_cosmic_top_100.fasta`",
            "- Algebraic settlement: `validation/W12_COSMIC_100_LEDGER.json`",
            "- Pre/post matrix log: `tensors/exact_form_null.log`",
            "- NCBI E-utilities documentation: "
            "https://www.ncbi.nlm.nih.gov/books/NBK25501/",
            "",
            "## Interpretation limits",
            "",
            "- PAINS catalogs detect only encoded substructure alerts.",
            "- Synthetic-accessibility values are heuristic, not exact proofs.",
            "- Fingerprint or Murcko similarity is not a legal non-infringement test.",
            "- RefSeq windows are reference transcripts, not tumour-allele sequences.",
            "- Algebraic complement and discrete closure are mathematical artifacts, "
            "not therapeutic validation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    accession_payload = _load_optional(ACCESSION_MAP) or {"rows": ()}
    catalog = _pains_catalog()
    records = tuple(
        _audit_record(record, catalog)
        for record in payload["records"]
    )
    report = {
        "schema": "foliation.public_topology_audit.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "clinical_grade": False,
        "tool": "RDKit",
        "input": str(args.input.relative_to(ROOT)),
        "records": records,
        "all_topologies_valid": all(
            row["state"] == "TOPOLOGY_VERIFIED" for row in records
        ),
        "all_source_formulas_match": all(
            bool(row.get("source_formula_match")) for row in records
        ),
        "all_source_exact_masses_match": all(
            bool(row.get("source_exact_mass_match")) for row in records
        ),
        "all_pains_clear": all(
            int(row.get("pains_alert_count", 0)) == 0 for row in records
        ),
        "ip_clearance_claimed": False,
        "affinity_claimed": False,
        "provenance": {
            "ncbi_batch_log_available": NCBI_LOG.is_file(),
            "w12_ledger_available": W12_LEDGER.is_file(),
            "exact_form_log_available": EXACT_FORM_LOG.is_file(),
            "ncbi_batch_log": _load_optional(NCBI_LOG),
            "refseq_examples": tuple(accession_payload.get("rows", ()))[:5],
        },
        "disclaimer": payload["disclaimer"],
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_out.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": bool(
                    report["all_topologies_valid"]
                    and report["all_source_formulas_match"]
                    and report["all_source_exact_masses_match"]
                ),
                "records": len(records),
                "all_pains_clear": report["all_pains_clear"],
                "ip_clearance_claimed": False,
                "json": str(args.json_out),
                "markdown": str(args.markdown_out),
            },
            sort_keys=True,
        )
    )
    return 0 if report["all_topologies_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
