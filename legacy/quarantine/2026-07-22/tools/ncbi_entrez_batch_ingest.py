#!/usr/bin/env python3.12
"""Resolve W12 target labels to public RefSeq mRNA windows in two Entrez calls.

NCBI network access is performed exclusively through Biopython Entrez:

1. one ESearch request for all unique gene symbols, with history enabled;
2. one EFetch request for the complete GenBank result set.

There is no sequential API polling and no per-target network request.
Entrez retries are disabled (max_tries=1). All transcript ranking, ambiguity
resolution, and 48-mer extraction are deterministic local operations.

Important scientific boundary:
  * NCBI RefSeq supplies reference transcripts, not tumour alleles.
  * Substitution-like labels are used only to center a reference window when
    the stated reference amino acid agrees with the selected CDS.
  * amplification/fusion/LOF/indel proxy labels use a CDS-midpoint reference
    window and are never represented as mutation-specific sequence evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

from Bio import Entrez, SeqIO
from Bio.SeqFeature import SeqFeature
from Bio.SeqRecord import SeqRecord

from cosmic_tensor_fetch import TARGETS

ROOT = Path(__file__).resolve().parents[1]
WIDTH = 48
DEFAULT_FASTA = ROOT / "inputs/w12_cosmic_top_100.fasta"
DEFAULT_MANIFEST = ROOT / "inputs/w12_cosmic_top_100_manifest.json"
DEFAULT_ACCESSIONS = ROOT / "inputs/w12_ncbi_refseq_100_accessions.json"
DEFAULT_LOG = ROOT / "logs/ncbi_batch_fetch.json"
NCBI_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
NCBI_DOCS = "https://www.ncbi.nlm.nih.gov/books/NBK25501/"

LABEL_CONTEXT = {
    "NOTCH1_L1601P": {
        "resolution": "MODEL_LABEL_NOT_HUMAN_REFSEQ_MATCH",
        "note": (
            "The published NOTCH1-L1601P-DeltaP construct is described as an "
            "oncogenic experimental model. Human RefSeq NM_017617.5 encodes H "
            "at residue 1601, so this label is retained only as a gene/locus "
            "reference and is not asserted as a human RefSeq allele."
        ),
        "source_url": "https://pubmed.ncbi.nlm.nih.gov/33659293/",
        "human_refseq_residue_1601": "H",
    }
}

POINT_VARIANT = re.compile(
    r"_(?P<ref>[A-Z])(?P<position>[1-9][0-9]*)"
    r"(?:(?P<alt>[A-Z*X])|(?P<frameshift>FS)|(?P<altfs>[A-Z])FS)"
    r"(?:_PROXY)?$"
)


@dataclass(frozen=True)
class VariantHint:
    reference_aa: str
    position: int
    alternate: str | None
    frameshift: bool


@dataclass(frozen=True)
class TranscriptCandidate:
    gene: str
    accession: str
    record: SeqRecord
    cds: SeqFeature | None
    translation: str
    base_score: tuple[int, int, int, int, int]


def _gene_from_label(label: str) -> str:
    return label.split("_", 1)[0]


def _variant_hint(label: str) -> VariantHint | None:
    match = POINT_VARIANT.search(label)
    if match is None:
        return None
    alt = match.group("alt") or match.group("altfs")
    return VariantHint(
        reference_aa=match.group("ref"),
        position=int(match.group("position")),
        alternate=alt,
        frameshift=bool(match.group("frameshift") or match.group("altfs")),
    )


def _first_feature(record: SeqRecord, feature_type: str) -> SeqFeature | None:
    return next(
        (feature for feature in record.features if feature.type == feature_type),
        None,
    )


def _feature_gene(feature: SeqFeature) -> str | None:
    genes = feature.qualifiers.get("gene", ())
    return str(genes[0]).upper() if genes else None


def _record_gene(record: SeqRecord) -> str | None:
    preferred = (
        _feature_gene(feature)
        for feature in record.features
        if feature.type in {"gene", "CDS", "source"}
    )
    return next((gene for gene in preferred if gene), None)


def _annotation_text(record: SeqRecord) -> str:
    fields = (
        str(record.description),
        " ".join(map(str, record.annotations.get("keywords", ()))),
        str(record.annotations.get("comment", "")),
    )
    return " ".join(fields).casefold()


def _translation(cds: SeqFeature | None) -> str:
    if cds is None:
        return ""
    values = cds.qualifiers.get("translation", ())
    return str(values[0]) if values else ""


def _candidate(record: SeqRecord) -> TranscriptCandidate | None:
    gene = _record_gene(record)
    if gene is None:
        return None
    cds = _first_feature(record, "CDS")
    text = _annotation_text(record)
    accession = str(record.id)
    base_score = (
        int("mane select" in text),
        int("refseq select" in text),
        int(accession.startswith("NM_")),
        int(cds is not None),
        len(_translation(cds)),
    )
    return TranscriptCandidate(
        gene=gene,
        accession=accession,
        record=record,
        cds=cds,
        translation=_translation(cds),
        base_score=base_score,
    )


def _variant_match(candidate: TranscriptCandidate, hint: VariantHint | None) -> int:
    if hint is None:
        return 0
    index = hint.position - 1
    if index < 0 or index >= len(candidate.translation):
        return -1
    return int(candidate.translation[index] == hint.reference_aa)


def _ranked_candidates(
    candidates: Iterable[TranscriptCandidate],
    hint: VariantHint | None,
) -> tuple[TranscriptCandidate, ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda item: (
                _variant_match(item, hint),
                *item.base_score,
                item.accession,
            ),
            reverse=True,
        )
    )


def _window(
    candidate: TranscriptCandidate,
    hint: VariantHint | None,
) -> tuple[str, dict[str, Any]]:
    sequence = str(candidate.record.seq).upper().replace("T", "U")
    cds = candidate.cds
    if cds is not None:
        cds_start = int(cds.location.start)
        cds_end = int(cds.location.end)
    else:
        cds_start = 0
        cds_end = len(sequence)

    variant_match = _variant_match(candidate, hint)
    if hint is not None and cds is not None and hint.position * 3 <= cds_end - cds_start:
        codon_start = cds_start + 3 * (hint.position - 1)
        center = codon_start + 1
        strategy = (
            "variant_locus_reference_window"
            if variant_match == 1
            else "variant_numbering_mismatch_reference_window"
        )
    else:
        center = cds_start + max(0, cds_end - cds_start) // 2
        strategy = "cds_midpoint_gene_reference_window"

    start = center - WIDTH // 2
    end = start + WIDTH
    left_padding = max(0, -start)
    right_padding = max(0, end - len(sequence))
    clipped = sequence[max(0, start):min(len(sequence), end)]
    fixed = ("A" * left_padding) + clipped + ("A" * right_padding)
    fixed = fixed[:WIDTH].ljust(WIDTH, "A")
    if len(fixed) != WIDTH:
        raise AssertionError("window normalization failed")

    metadata = {
        "strategy": strategy,
        "window_start_0based": start,
        "window_end_exclusive_0based": end,
        "left_padding_A": left_padding,
        "right_padding_A": right_padding,
        "variant_reference_match": (
            None if hint is None else bool(variant_match == 1)
        ),
        "cds_start_0based": cds_start if cds is not None else None,
        "cds_end_exclusive_0based": cds_end if cds is not None else None,
    }
    return fixed, metadata


def _candidate_public_summary(
    candidate: TranscriptCandidate,
    hint: VariantHint | None,
) -> dict[str, Any]:
    return {
        "accession": candidate.accession,
        "variant_reference_match": (
            None if hint is None else bool(_variant_match(candidate, hint) == 1)
        ),
        "mane_select": bool(candidate.base_score[0]),
        "refseq_select": bool(candidate.base_score[1]),
        "curated_nm": bool(candidate.base_score[2]),
        "has_cds": bool(candidate.base_score[3]),
        "translation_length": candidate.base_score[4],
    }


def _resolve_target(
    label: str,
    by_gene: dict[str, tuple[TranscriptCandidate, ...]],
) -> dict[str, Any]:
    gene = _gene_from_label(label).upper()
    hint = _variant_hint(label)
    ranked = _ranked_candidates(by_gene.get(gene, ()), hint)
    if not ranked:
        return {
            "target_name": label,
            "gene_symbol": gene,
            "status": "UNRESOLVED_NO_REFSEQ_MRNA",
            "source_url": (
                "https://www.ncbi.nlm.nih.gov/gene/?term="
                f"{quote(gene)}%5Bsym%5D+AND+human%5Borgn%5D"
            ),
        }

    selected = ranked[0]
    sequence, window = _window(selected, hint)
    top_score = (
        _variant_match(selected, hint),
        *selected.base_score,
    )
    equivalent_top = tuple(
        item
        for item in ranked
        if (_variant_match(item, hint), *item.base_score) == top_score
    )
    variant = (
        None
        if hint is None
        else {
            "reference_aa": hint.reference_aa,
            "protein_position": hint.position,
            "alternate": hint.alternate,
            "frameshift_label": hint.frameshift,
        }
    )
    return {
        "target_name": label,
        "gene_symbol": gene,
        "status": "RESOLVED",
        "refseq_accession": selected.accession,
        "reference_sequence_48mer": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "variant_hint": variant,
        "window": window,
        "candidate_count": len(ranked),
        "ambiguous_top_rank": len(equivalent_top) > 1,
        "equivalent_top_accessions": tuple(
            item.accession for item in equivalent_top[:10]
        ),
        "top_candidates": tuple(
            _candidate_public_summary(item, hint) for item in ranked[:5]
        ),
        "selection_policy": (
            "reference-AA match > MANE Select > RefSeq Select > curated NM_ "
            "> CDS present > translation length > accession lexical tie-break"
        ),
        "source_url": f"https://www.ncbi.nlm.nih.gov/nuccore/{selected.accession}",
        "label_context": LABEL_CONTEXT.get(label),
    }


def _fasta_record(row: dict[str, Any]) -> str:
    return (
        f">{row['target_name']} accession={row['refseq_accession']} "
        f"gene={row['gene_symbol']} strategy={row['window']['strategy']} "
        f"width={WIDTH} source=NCBI_RefSeq\n"
        f"{row['reference_sequence_48mer']}\n"
    )


def _sanitized_request(
    endpoint: str,
    *,
    database: str,
    method: str,
    query_sha256: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "endpoint": f"{NCBI_EUTILS}/{endpoint}.fcgi",
        "database": database,
        "method": method,
        "biopython_entrez": True,
    }
    if query_sha256 is not None:
        result["query_sha256"] = query_sha256
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", default=os.environ.get("NCBI_EMAIL"))
    parser.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"))
    parser.add_argument("--fasta-out", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--accessions-out", type=Path, default=DEFAULT_ACCESSIONS)
    parser.add_argument("--log-out", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--retmax", type=int, default=5000)
    args = parser.parse_args()

    if not args.email:
        parser.error("--email or NCBI_EMAIL is required by NCBI E-utilities policy")

    Entrez.email = args.email
    Entrez.tool = "foliation_engine_w12_refseq_batch"
    Entrez.api_key = args.api_key
    Entrez.max_tries = 1
    Entrez.sleep_between_tries = 0

    genes = tuple(sorted(set(map(_gene_from_label, TARGETS))))
    gene_term = " OR ".join(map(lambda gene: f'"{gene}"[Gene Name]', genes))
    query = (
        '"Homo sapiens"[Organism] AND srcdb_refseq[PROP] '
        f"AND biomol_mrna[PROP] AND ({gene_term})"
    )
    query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()

    # Network call 1/2: one batch search, storing all results on NCBI history.
    search_handle = Entrez.esearch(
        db="nuccore",
        term=query,
        retmax=args.retmax,
        usehistory="y",
    )
    search_result = Entrez.read(search_handle)
    search_handle.close()
    result_count = int(search_result["Count"])
    if result_count == 0:
        raise RuntimeError("NCBI batch search returned no RefSeq mRNA records")
    if result_count > args.retmax:
        raise RuntimeError(
            f"NCBI result count {result_count} exceeds one-fetch retmax {args.retmax}"
        )

    # Network call 2/2: one batch GenBank fetch through the history token.
    fetch_handle = Entrez.efetch(
        db="nuccore",
        query_key=search_result["QueryKey"],
        WebEnv=search_result["WebEnv"],
        rettype="gb",
        retmode="text",
        retstart=0,
        retmax=result_count,
    )
    records = tuple(SeqIO.parse(fetch_handle, "genbank"))
    fetch_handle.close()

    candidates = tuple(
        candidate
        for candidate in map(_candidate, records)
        if candidate is not None
    )
    grouped: dict[str, list[TranscriptCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.gene].append(candidate)
    by_gene = {
        gene: tuple(items)
        for gene, items in grouped.items()
    }

    rows = tuple(map(lambda label: _resolve_target(label, by_gene), TARGETS))
    unresolved = tuple(
        row for row in rows if row.get("status") != "RESOLVED"
    )
    resolved = tuple(
        row for row in rows if row.get("status") == "RESOLVED"
    )

    log = {
        "schema": "foliation.ncbi_entrez_batch_fetch.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "Biopython Entrez + SeqIO",
        "biopython_version": __import__("Bio").__version__,
        "official_api_documentation": NCBI_DOCS,
        "api_request_count": 2,
        "sequential_polling": False,
        "entrez_max_tries": 1,
        "api_key_used": bool(args.api_key),
        "contact_email_configured": True,
        "requests": [
            _sanitized_request(
                "esearch",
                database="nuccore",
                method="POST/GET selected by Biopython",
                query_sha256=query_sha256,
            ),
            _sanitized_request(
                "efetch",
                database="nuccore",
                method="history-backed batch fetch",
            ),
        ],
        "query": query,
        "unique_gene_count": len(genes),
        "ncbi_result_count": result_count,
        "genbank_records_parsed": len(records),
        "candidate_transcripts": len(candidates),
        "resolved_targets": len(resolved),
        "unresolved_targets": len(unresolved),
        "unresolved": unresolved,
    }

    args.log_out.parent.mkdir(parents=True, exist_ok=True)
    args.accessions_out.parent.mkdir(parents=True, exist_ok=True)
    args.log_out.write_text(
        json.dumps(log, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.accessions_out.write_text(
        json.dumps(
            {
                "schema": "foliation.w12.refseq_accession_map.v1",
                "generated_at": log["generated_at"],
                "source": "NCBI RefSeq",
                "source_documentation": NCBI_DOCS,
                "rows": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    if unresolved:
        print(
            json.dumps(
                {
                    "ok": False,
                    "resolved": len(resolved),
                    "unresolved": len(unresolved),
                    "log": str(args.log_out),
                },
                sort_keys=True,
            )
        )
        return 2

    fasta = "".join(map(_fasta_record, rows))
    manifest = {
        "schema": "foliation.w12.fixed48_input.v2",
        "generated_at": log["generated_at"],
        "mode": "ncbi_refseq_batch",
        "ncbi_derived": True,
        "cosmic_derived": False,
        "licensed_data_bundled": False,
        "clinical_grade": False,
        "n_targets": len(rows),
        "width": WIDTH,
        "all_rows_fixed_width": all(
            len(row["reference_sequence_48mer"]) == WIDTH for row in rows
        ),
        "padding_symbol": "A",
        "source_documentation": NCBI_DOCS,
        "batch_fetch_log": str(args.log_out.relative_to(ROOT)),
        "accession_map": str(args.accessions_out.relative_to(ROOT)),
        "rows": rows,
        "disclaimer": (
            "Public RefSeq reference windows only. Mutation labels provide locus "
            "hints where validated; these are not tumour-allele sequences, "
            "therapeutic designs, causal claims, or clinical evidence."
        ),
    }
    args.fasta_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.fasta_out.write_text(fasta, encoding="utf-8")
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "resolved": len(rows),
                "shape": [len(rows), WIDTH],
                "ambiguous": sum(bool(row["ambiguous_top_rank"]) for row in rows),
                "variant_reference_mismatches": sum(
                    row["window"]["variant_reference_match"] is False
                    for row in rows
                ),
                "fasta": str(args.fasta_out),
                "accessions": str(args.accessions_out),
                "log": str(args.log_out),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
