#!/usr/bin/env python3
"""Declare what every frozen artifact is for, and refuse to let new ones drift in.

There are 46 JSON artifacts under ``results/``. Seven are cited by name in the
manuscript or the README; the rest are training-fold sweeps kept for provenance.
Undeclared, that asymmetry reads badly and reads correctly: a directory holding
two dozen architecture sweeps beside a headline number is what cherry-picking
looks like from outside, whether or not it is what happened.

So every artifact gets a class and a reason, and a gate fails on any file that
has neither. The classes:

  cited        a number in the paper or the README comes from this file, either
               directly or through tools/emit_frozen_numbers.py
  exploration  a sweep over the TRAINING fold; it informed a design decision and
               is kept so the decision can be re-derived, but no paper number
               comes from it
  fold_access  an evaluation on the official TEST fold; each one is also counted
               in TEST_FOLD_ACCESS_LEDGER.json, which is the honest register of
               how often the held-out data was scored
  superseded   an earlier frozen state, kept for the record and cited by nothing

The gate that matters most is the last consistency check below: an artifact
classified as ``exploration`` must not contain per-unit metrics over the
official test units. That is the mechanical form of "we did not quietly evaluate
on the test set and file it under sweeps", and it is checkable rather than
promised.

Usage:
  PYTHONPATH=src python3.12 tools/classify_artifacts.py
  PYTHONPATH=src python3.12 tools/classify_artifacts.py --check
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "ARTIFACT_MANIFEST.json"
LEDGER = RESULTS / "official_fold/TEST_FOLD_ACCESS_LEDGER.json"

# Files whose paths appear in the manuscript, the README, or the generators that
# turn artifacts into paper macros. Derived rather than typed where possible.
GENERATORS = ("tools/emit_frozen_numbers.py", "tools/render_results_section.py",
              "tools/check_report_consistency.py", "tools/freeze_bootstrap.py",
              "tools/build_test_fold_ledger.py")
PROSE = ("paper/MAIN_CRYPTOBENCH_GEOAUDIT.tex", "paper/appendix_b_gf4_ablation.tex",
         "README.md")

REASONS = {
    "cited": "a number in the paper or README derives from this file",
    "exploration": "a training-fold sweep; informed a design choice, cited by "
                   "no paper number",
    "fold_access": "an evaluation on the official test fold; also registered in "
                   "TEST_FOLD_ACCESS_LEDGER.json",
    "superseded": "an earlier frozen state, kept for the record",
}


def _referenced_paths() -> set[str]:
    """Artifact paths mentioned by the prose or by a macro generator."""
    pat = re.compile(r"results/[A-Za-z0-9_/\\]+?\.json")
    found: set[str] = set()
    for rel in GENERATORS + PROSE:
        p = ROOT / rel
        if not p.exists():
            continue
        for m in pat.findall(p.read_text(errors="ignore")):
            found.add(m.replace("\\_", "_"))
    return found


def _is_derived_summary(doc: dict) -> bool:
    """Recomputed from telemetry rather than produced by scoring the fold.

    The distinction is the whole point of the ledger. Running a detector over
    the 192 units is a look at held-out data; re-reducing the telemetry those
    runs wrote is arithmetic on data already seen, and counting it as a fresh
    access would inflate the register until nobody read it. Derived summaries
    say so by naming the telemetry they came from.
    """
    return bool(doc.get("telemetry_ref") or doc.get("telemetry_source"))


def _is_fold_access(doc: dict) -> bool:
    if _is_derived_summary(doc):
        return False
    if doc.get("is_official_mmseqs2_10pct_test_fold"):
        return True
    for key in ("per_structure", "per_unit"):
        v = doc.get(key)
        if (isinstance(v, list) and len(v) >= 150 and isinstance(v[0], dict)
                and "unit_id" in v[0]):
            return True
    return False


def build() -> dict:
    cited = _referenced_paths()
    entries = []
    for p in sorted(RESULTS.rglob("*.json")):
        rel = str(p.relative_to(ROOT))
        if "/predictions/" in rel or "/p2rank_raw/" in rel:
            continue
        if p.resolve() == OUT.resolve():
            continue
        try:
            doc = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            doc = {}
        doc = doc if isinstance(doc, dict) else {}

        if _is_fold_access(doc):
            cls = "fold_access"
        elif rel in cited:
            cls = "cited"
        elif rel.startswith("results/architecture_sweep/"):
            cls = "exploration"
        elif rel.startswith("results/cryptobench_apo/") or rel.startswith("results/pilot/"):
            cls = "cited" if rel in cited else "superseded"
        else:
            cls = "cited" if rel in cited else "superseded"

        entries.append({
            "artifact": rel,
            "class": cls,
            "reason": REASONS[cls],
            "schema": doc.get("schema"),
            "bytes": p.stat().st_size,
        })

    counts: dict[str, int] = {}
    for e in entries:
        counts[e["class"]] = counts.get(e["class"], 0) + 1
    return {
        "schema": "geoaudit.artifact_manifest.v1",
        "clinical_grade": False,
        "purpose": "every frozen artifact declares what it is for, so that a "
                   "directory of architecture sweeps beside a headline number "
                   "is a stated fact rather than an inference",
        "classes": REASONS,
        "counts": counts,
        "n_artifacts": len(entries),
        "artifacts": entries,
    }


def _consistency(man: dict) -> list[str]:
    problems = []
    for e in man["artifacts"]:
        p = ROOT / e["artifact"]
        if not p.exists():
            problems.append(f"{e['artifact']}: declared but absent")
            continue
        if e["class"] != "exploration":
            continue
        try:
            doc = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        if isinstance(doc, dict) and _is_fold_access(doc):
            problems.append(
                f"{e['artifact']}: classified exploration but carries official "
                f"test-fold per-unit metrics")

    if LEDGER.exists():
        led = json.loads(LEDGER.read_text())
        registered = {a["artifact"] for a in led["standalone_probe_artifacts"]}
        declared = {e["artifact"] for e in man["artifacts"]
                    if e["class"] == "fold_access"}
        # Bootstrap reports summarise the fold without being a fresh access.
        fresh = declared
        missing = fresh - registered
        if missing:
            problems.append(
                f"test-fold evaluations not in the access ledger: "
                f"{', '.join(sorted(missing))}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    man = build()
    problems = _consistency(man)
    if problems:
        print("artifact manifest FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    if args.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        if json.loads(OUT.read_text()) != man:
            print("STALE ARTIFACT_MANIFEST.json: results/ holds a different set "
                  "of artifacts than the manifest declares")
            return 1
        print(f"artifact manifest current: {man['n_artifacts']} artifacts "
              f"({', '.join(f'{v} {k}' for k, v in sorted(man['counts'].items()))})")
        return 0

    OUT.write_text(json.dumps(man, indent=2, allow_nan=False) + "\n")
    for k, v in sorted(man["counts"].items()):
        print(f"  {k:12s} {v:3d}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
