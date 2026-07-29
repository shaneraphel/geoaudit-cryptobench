"""Score the counting field and P2Rank on the external set, and archive the raw output.

This produces per-residue scores and nothing else: no metric, no threshold, no
comparison. That separation is deliberate. Scoring is the step that has to touch
the external structures, and if it also computed a difference there would be no way
to show, afterwards, that the difference was computed once under the plan rather
than watched while something was adjusted.

Both methods are run exactly as they were run on CryptoBench: the same frozen
field, the same P2Rank version and configuration, the same receptor writer
upstream. P2Rank's own CSV output is archived per unit with its command, version,
JVM banner and file digests, because every P2Rank number in this repository is a
transformation of those two files and an unarchived baseline cannot be checked.

Usage: PYTHONPATH=src:tools python3.12 tools/external_score.py [--check]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

from pocket_bench.methods import p2rank_wrap, table_field
from pocket_bench.paths import ROOT
from pocket_bench.pdb_io import parse_pdb_atoms

MANIFEST = ROOT / "data/external/external_manifest.json"
PREDS = ROOT / "results/external/predictions"
ARCHIVE = ROOT / "results/external/p2rank_raw"
PER_STRUCTURE = ROOT / "results/external/PER_STRUCTURE.json"

SCHEMA = "geoaudit.raw_predictions.v1"
METHODS = ("table_field", "p2rank")


def universe(receptor: Path) -> list[int]:
    """Every residue of the chain in the receptor, which is the scoring universe.

    Taken from the receptor rather than from the label file, so that a residue with
    no label is still a negative rather than being quietly dropped, and so that
    every method is asked about the same set of residues.
    """
    seen: list[int] = []
    have: set[int] = set()
    for a in parse_pdb_atoms(receptor.read_text(errors="ignore")):
        if a["record"] != "ATOM":
            continue
        if a["resseq"] not in have:
            have.add(a["resseq"])
            seen.append(a["resseq"])
    return seen


def _row(pred: dict, universe_: list[int]) -> dict:
    scores = pred.get("residue_scores") or {}
    return {
        "status": pred.get("status"),
        "runtime_s": pred.get("runtime_s"),
        "input_receptor_sha256": pred.get("input_receptor_sha256"),
        "tool_version": (pred.get("tool_version")
                         or (pred.get("extra") or {}).get("tool_version")),
        "n_universe": len(universe_),
        "operating_q": pred.get("operating_q"),
        "residue_scores": {str(k): float(v) for k, v in scores.items()} or None,
        "residue_positive": pred.get("residue_positive"),
        "pockets": [{"rank": p.get("rank"), "center_xyz": p.get("center_xyz"),
                     "score": p.get("score")}
                    for p in (pred.get("pockets") or [])],
        "error": pred.get("error"),
    }


def run() -> dict:
    man = json.loads(MANIFEST.read_text())
    PREDS.mkdir(parents=True, exist_ok=True)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, dict]] = {m: {} for m in METHODS}
    failures: list[dict] = []

    for i, e in enumerate(man["entries"], start=1):
        unit = f"{e['pdb']}_{e['chain']}"
        rec = ROOT / e["receptor_path"]
        if hashlib.sha256(rec.read_bytes()).hexdigest() != e["receptor_sha256"]:
            raise SystemExit(
                f"{unit}: the receptor no longer matches the manifest digest. The "
                f"external set is frozen; scoring a changed input would make the "
                f"comparison meaningless")
        uni = universe(rec)
        for method in METHODS:
            if method == "table_field":
                pred = table_field.predict(rec, pdb_id=e["pdb"],
                                           chain=e["chain"])
            else:
                pred = p2rank_wrap.predict(rec, pdb_id=e["pdb"],
                                           chain=e["chain"],
                                           archive_dir=ARCHIVE)
            row = _row(pred, uni)
            if not row["residue_scores"]:
                failures.append({"unit": unit, "method": method,
                                 "status": row["status"],
                                 "error": row["error"]})
            out[method][unit] = row
        if i % 10 == 0 or i == len(man["entries"]):
            print(f"  scored {i}/{len(man['entries'])}", flush=True)

    index = {}
    for method, by_unit in out.items():
        payload = {"schema": SCHEMA, "clinical_grade": False, "method": method,
                   "n_units": len(by_unit),
                   "fold": "external",
                   "manifest_sha256": hashlib.sha256(
                       MANIFEST.read_bytes()).hexdigest(),
                   "units": dict(sorted(by_unit.items()))}
        p = PREDS / f"{method}.json"
        p.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        index[method] = {"file": p.name, "n_units": len(by_unit),
                         "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    # The residue universe per unit, in the shape the baseline tools already read
    # for the official fold. Emitting it here means pLM-NN and PocketMiner are
    # held to the same universe as the counting field by the same check, rather
    # than by a second one written for the external set.
    PER_STRUCTURE.write_text(json.dumps(
        [{"pdb": e["pdb"], "chain": e["chain"],
          "n_universe": out["table_field"][f"{e['pdb']}_{e['chain']}"]["n_universe"]}
         for e in man["entries"]], indent=1) + "\n")
    (PREDS / "INDEX.json").write_text(json.dumps({
        "schema": "geoaudit.raw_predictions_index.v1",
        "clinical_grade": False,
        "fold": "external",
        "carries_no_metric": True,
        "why": ("scoring is kept apart from comparing so that the comparison can "
                "be shown to have run once, under the plan"),
        "n_failures": len(failures), "failures": failures,
        "methods": index}, indent=2) + "\n")
    return {"index": index, "failures": failures,
            "n_units": len(man["entries"])}


def check() -> int:
    idx = PREDS / "INDEX.json"
    if not idx.is_file():
        print(f"MISSING {idx.relative_to(ROOT)}")
        return 1
    d = json.loads(idx.read_text())
    man = json.loads(MANIFEST.read_text())
    for method, info in d["methods"].items():
        p = PREDS / info["file"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != info["sha256"]:
            print(f"FAILED: {info['file']} has changed since it was indexed")
            return 1
        doc = json.loads(p.read_text())
        if doc.get("manifest_sha256") != hashlib.sha256(
                MANIFEST.read_bytes()).hexdigest():
            print(f"FAILED: {method} was scored against a different external "
                  f"manifest than the one on disk")
            return 1
        if doc["n_units"] != man["n_entries"]:
            print(f"FAILED: {method} covers {doc['n_units']} of "
                  f"{man['n_entries']} units")
            return 1
        scored = sum(1 for u in doc["units"].values() if u["residue_scores"])
        print(f"  {method}: {scored}/{doc['n_units']} units with per-residue "
              f"scores, version {next(iter(doc['units'].values()))['tool_version']}")
    if d["n_failures"]:
        print(f"  {d['n_failures']} unit-method pairs produced no scores: "
              f"{d['failures'][:3]}")
    print(f"OK {idx.relative_to(ROOT)}")
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return check()
    if not os.environ.get("JAVA_HOME"):
        print("note: JAVA_HOME is unset; P2Rank will report itself unavailable "
              "rather than being silently skipped", flush=True)
    t0 = time.perf_counter()
    got = run()
    print(f"\nscored {got['n_units']} external units in "
          f"{time.perf_counter() - t0:.0f}s")
    for method, info in got["index"].items():
        print(f"  {method}: {info['n_units']} units, {info['sha256'][:12]}")
    if got["failures"]:
        print(f"  {len(got['failures'])} unit-method pairs without scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
