"""Fail-closed data adapters for external benchmark inputs.

Two adapters, both strictly fail-closed: they raise a hard, instructive error when
their required inputs are absent rather than silently degrading, imputing, or
relabelling a pilot as an official fold. `clinical_grade=false`.

* ``load_official_test_fold`` — the official CryptoBench MMseqs2 10%-identity
  cluster-disjoint TEST fold. Requires ``data/cryptobench_apo/official_manifest.json``.
* ``load_pocketminer_scores`` — per-residue cryptic-site probabilities emitted by
  PocketMiner. Requires ``data/baselines/pocketminer/<pdb>_<chain>.{json,csv}``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pocket_bench.paths import ROOT
from pocket_bench.pdb_io import sha256_file

OFFICIAL_FOLD_MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
POCKETMINER_DIR = ROOT / "data/baselines/pocketminer"

_OFFICIAL_SCHEMA = "cryptobench.official_test_fold.v1"
_CRYPTOBENCH_SOURCE = (
    "https://github.com/skrhakv/CryptoBench (dataset https://osf.io/pz4a9/; "
    "Skrhak et al., Bioinformatics 2025, doi:10.1093/bioinformatics/btae745)"
)
_POCKETMINER_SOURCE = (
    "https://github.com/Mickdub/gvp/tree/pocket_pred (Meller et al., "
    "Nat. Commun. 2023, doi:10.1038/s41467-023-36699-3)"
)


class DataUnavailable(FileNotFoundError):
    """Raised when a required external input is absent (hard fail, no fallback)."""


# --------------------------------------------------------------------------- #
# Adapter 1: official CryptoBench MMseqs2 10% cluster-disjoint TEST fold
# --------------------------------------------------------------------------- #
def official_fold_available(root: Path = ROOT) -> bool:
    return (root / "data/cryptobench_apo/official_manifest.json").is_file()


def load_official_test_fold(root: Path = ROOT, *, verify_hashes: bool = True) -> dict[str, Any]:
    """Load the official CryptoBench test fold; fail closed if absent or invalid.

    The manifest at ``data/cryptobench_apo/official_manifest.json`` must declare:
      { "schema": "cryptobench.official_test_fold.v1",
        "fold": "test",
        "clustering": {"method": "mmseqs2", "sequence_identity_threshold": 0.10,
                       "coverage": <float>},
        "source_url": <str>,
        "entries": [ {"pdb","chain","cluster_id",
                      "receptor_path","receptor_sha256",
                      "label_path","label_sha256"}, ... ] }

    Nothing here is generated or defaulted: a missing manifest, a wrong clustering
    threshold, a file whose SHA-256 does not match, or a cluster_id that appears in
    a foreign split all raise. A 15-structure stride pilot CANNOT be passed off as
    this fold.
    """
    path = root / "data/cryptobench_apo/official_manifest.json"
    if not path.is_file():
        raise DataUnavailable(
            "Official CryptoBench test fold manifest not found.\n"
            f"  expected: {path.relative_to(root)}\n"
            f"  source:   {_CRYPTOBENCH_SOURCE}\n"
            "  action:   fetch the official train/test split (MMseqs2 @10% identity),\n"
            "            write it as the schema below, and place per-structure\n"
            "            receptor+label files under data/cryptobench_apo/.\n"
            f"  schema:   {_OFFICIAL_SCHEMA} with keys "
            "{fold=='test', clustering.method=='mmseqs2', "
            "clustering.sequence_identity_threshold==0.10, entries[]}.\n"
            "  note:     the pinned n=15 stride pilot is NOT this fold and will not "
            "be substituted (clinical_grade=false)."
        )
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != _OFFICIAL_SCHEMA:
        raise ValueError(f"official fold: schema must be {_OFFICIAL_SCHEMA}")
    if manifest.get("fold") != "test":
        raise ValueError("official fold: 'fold' must be 'test'")
    clustering = manifest.get("clustering") or {}
    if clustering.get("method") != "mmseqs2":
        raise ValueError("official fold: clustering.method must be 'mmseqs2'")
    if float(clustering.get("sequence_identity_threshold", -1)) != 0.10:
        raise ValueError("official fold: sequence_identity_threshold must be 0.10")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("official fold: 'entries' must be a non-empty list")

    seen_cluster_split: dict[str, str] = {}
    for e in entries:
        for key in ("pdb", "chain", "cluster_id", "receptor_path", "receptor_sha256",
                    "label_path", "label_sha256"):
            if key not in e:
                raise ValueError(f"official fold: entry {e.get('pdb')} missing '{key}'")
        # a TEST-fold cluster_id must not also be declared in another split
        prior = seen_cluster_split.get(e["cluster_id"])
        if prior is not None and prior != "test":
            raise ValueError(
                f"official fold: cluster_id {e['cluster_id']} spans splits (leakage)"
            )
        seen_cluster_split[e["cluster_id"]] = "test"
        if verify_hashes:
            for path_key, sha_key in (("receptor_path", "receptor_sha256"),
                                      ("label_path", "label_sha256")):
                f = root / e[path_key]
                if not f.is_file():
                    raise DataUnavailable(
                        f"official fold: {path_key} missing for {e['pdb']}: {e[path_key]}"
                    )
                got = sha256_file(f)
                if got != e[sha_key]:
                    raise ValueError(
                        f"official fold: SHA-256 mismatch for {e[path_key]} "
                        f"(manifest {e[sha_key]}, file {got})"
                    )
    return manifest


# --------------------------------------------------------------------------- #
# Adapter 2: PocketMiner per-residue cryptic-site predictions
# --------------------------------------------------------------------------- #
def pocketminer_available(root: Path = ROOT) -> bool:
    return (root / "data/baselines/pocketminer").is_dir()


def load_pocketminer_scores(
    pdb: str, chain: str, root: Path = ROOT
) -> dict[int, float]:
    """Per-residue cryptic probabilities for one structure; fail closed if absent.

    Accepted standard PocketMiner output files (one per structure), searched in
    ``data/baselines/pocketminer/``:
      * ``<pdb>_<chain>.json``: {"pdb","chain","residue_scores": {"<resseq>": prob}}
      * ``<pdb>_<chain>.csv`` : header ``resseq,score`` (one row per residue)

    Scores must be finite probabilities in [0, 1]. A missing directory, a missing
    file, a malformed row, or an out-of-range score raises — never imputed to 0.
    """
    d = root / "data/baselines/pocketminer"
    if not d.is_dir():
        raise DataUnavailable(
            "PocketMiner baseline predictions not found.\n"
            f"  expected dir: {d.relative_to(root)}/<pdb>_<chain>.{{json,csv}}\n"
            f"  source:       {_POCKETMINER_SOURCE}\n"
            "  action:       run PocketMiner on each receptor and export per-residue\n"
            "                cryptic-site probabilities as <pdb>_<chain>.json "
            "({residue_scores:{resseq:prob}}) or <pdb>_<chain>.csv (resseq,score).\n"
            "  note:         absence is reported as an unavailable baseline; it is "
            "never scored as zeros (clinical_grade=false)."
        )
    pid = f"{pdb.lower()}_{chain}"
    fjson = d / f"{pid}.json"
    fcsv = d / f"{pid}.csv"
    scores: dict[int, float] = {}
    if fjson.is_file():
        raw = json.loads(fjson.read_text()).get("residue_scores") or {}
        for k, v in raw.items():
            scores[int(k)] = float(v)
    elif fcsv.is_file():
        with fcsv.open() as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "resseq" not in reader.fieldnames \
                    or "score" not in reader.fieldnames:
                raise ValueError(
                    f"PocketMiner CSV {fcsv.name} must have header 'resseq,score'"
                )
            for row in reader:
                scores[int(row["resseq"])] = float(row["score"])
    else:
        raise DataUnavailable(
            f"PocketMiner predictions for {pid} not found "
            f"(looked for {fjson.name} and {fcsv.name})."
        )
    if not scores:
        raise ValueError(f"PocketMiner predictions for {pid} are empty")
    for resseq, prob in scores.items():
        if not (0.0 <= prob <= 1.0):
            raise ValueError(
                f"PocketMiner score for {pid} residue {resseq} out of [0,1]: {prob}"
            )
    return scores
