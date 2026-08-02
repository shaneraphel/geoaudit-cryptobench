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
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "ARTIFACT_MANIFEST.json"
LEDGER = RESULTS / "official_fold/TEST_FOLD_ACCESS_LEDGER.json"

# Files whose paths appear in the manuscript, the README, or the generators that
# turn artifacts into paper macros. Derived rather than typed where possible.
GENERATORS = ("tools/emit_frozen_numbers.py", "tools/render_results_section.py",
              "tools/check_report_consistency.py", "tools/freeze_bootstrap.py",
              "tools/build_test_fold_ledger.py", "tools/make_official_figures.py")
PROSE = ("paper/MAIN_CRYPTOBENCH_GEOAUDIT.tex", "paper/appendix_b_gf4_ablation.tex",
         "README.md")

REASONS = {
    "cited": "a number in the paper or README derives from this file",
    "exploration": "a training-fold sweep; informed a design choice, cited by "
                   "no paper number",
    "fold_access": "an evaluation on the official test fold; also registered in "
                   "TEST_FOLD_ACCESS_LEDGER.json",
    "superseded": "an earlier frozen state, kept for the record",
    # Added when the Set B pool inventory landed and the path fall-through
    # labelled it "an earlier frozen state, kept for the record". It is neither
    # earlier nor frozen: it is a forward-looking inventory of accessions a set
    # has not yet been built from, and describing it as a superseded frozen state
    # in a registered manifest is the kind of field defect this manifest exists to
    # prevent. Detected by the artifact's own declaration rather than by its path,
    # which is how reads_test_fold is already handled.
    "inventory": "a pinned inventory of candidate inputs; declares "
                 "is_a_frozen_set false, and nothing has been selected, "
                 "labelled, hashed or preregistered from it",
    # Added when Set B was frozen. See the comment at the assignment.
    "frozen_unread": "a frozen external set that no method has been run on. Its "
                     "one read is unspent; scoring a method that changed after "
                     "this digest was pinned destroys the confirmation rather "
                     "than producing a second one",
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


def _on_this_disk_only() -> set[str]:
    """Files under results/ that git ignores, and so are not distributed.

    The manifest describes the repository a reader receives, not the working
    tree of whoever last regenerated it. Counting a gitignored intermediate --
    a set of baseline weights, a checkpoint -- makes the manifest one that can
    never be green in a clean clone, which is where it was found: every gate
    passed here and `make verify` died on a stale manifest for anyone who
    cloned. An export has no .git and no ignored files either, so the empty set
    is the right answer there.
    """
    p = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--others", "--ignored",
         "--exclude-standard", "results/"], capture_output=True, text=True)
    if p.returncode != 0:
        return set()
    return {line.strip() for line in p.stdout.splitlines() if line.strip()}


def build() -> dict:
    cited = _referenced_paths()
    undistributed = _on_this_disk_only()
    entries = []
    for p in sorted(RESULTS.rglob("*.json")):
        rel = str(p.relative_to(ROOT))
        # setn_p2rank_raw / setbc_p2rank_raw share the stem; require the stem,
        # not the slash-bounded substring, or a live baseline run makes the
        # manifest permanently stale for anyone scoring Set N.
        if "/predictions/" in rel or "p2rank_raw" in rel:
            continue
        if rel in undistributed:
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
        elif doc.get("is_a_frozen_set") is False:
            cls = "inventory"
        elif (doc.get("is_a_frozen_set") is True
              and doc.get("no_method_has_been_run") is True
              and rel not in cited):
            # Added when Set B was frozen and the path fall-through labelled it
            # "an earlier frozen state, kept for the record" -- the same defect
            # the inventory class was added for, one stage later in the same
            # set's life. It is neither earlier nor superseded: it is a frozen
            # set that has never been read, which is the most consequential thing
            # an artifact here can be, and calling it superseded in a registered
            # manifest would invite spending it by accident. Set A is `cited`
            # instead because paper numbers derive from it, which is what having
            # been read looks like from this side.
            cls = "frozen_unread"
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


FIGURES = ROOT / "figures"
# One pair per family of figures: the committed generator that draws them and the
# provenance file it writes. A second pair exists because training-fold findings
# and official-fold results must not share a generator -- a figure drawn from a
# frozen test-fold artifact and one drawn from a sweep over training halves carry
# different licences, and keeping the generators apart keeps the licences apart.
# Every image in figures/ must still come from one of these and match the sha and
# the caption it recorded, so adding the pair widens the gate's knowledge and not
# its tolerance.
FIGURE_GENERATORS = (
    ("tools/make_official_figures.py",
     RESULTS / "official_fold/FIGURE_PROVENANCE.json"),
    ("tools/make_architecture_figures.py",
     RESULTS / "architecture_sweep/ARCHITECTURE_FIGURE_PROVENANCE.json"),
    # The three-baseline read. A third pair rather than a third family inside
    # an existing generator, for the same reason the second pair exists: this
    # one draws every method against every published baseline in a single pass,
    # and its licence -- one residue universe, one n_paired per row -- is a
    # property of that pass and should fail on its own if the pass is rerun
    # without the figures.
    ("tools/fig_grand_baseline.py",
     RESULTS / "official_fold/GRAND_BASELINE_FIGURE_PROVENANCE.json"),
)
README = ROOT / "README.md"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _figure_problems() -> list[str]:
    """Committed images must be generated by a committed script from frozen data.

    The scope gates read .md, .json, .py and .yml, so images were a blind spot,
    and one sat in the tree for weeks: a file named fig_baseline_comparison.png
    plotting the 14-structure ESR1 pilot on a split its own summary records as
    not cluster-disjoint, labelled with method names that appear nowhere in the
    manuscript, showing every one of our detectors at zero. Nothing could have
    caught it, because nothing was looking. A figure is read before the text and
    remembered after it, so it is held to the same standard as a number.
    """
    problems = []
    if not FIGURES.is_dir():
        return problems

    produced: dict[str, str] = {}
    recorded: dict[str, dict] = {}
    prov_name: dict[str, str] = {}
    for gen_path, prov_path in FIGURE_GENERATORS:
        gen = ROOT / gen_path
        if gen.exists():
            for name in re.findall(r'FIGDIR / "([^"]+)"', gen.read_text()):
                produced[name] = gen_path
        prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
        for name, rec in (prov.get("figures") or {}).items():
            recorded[name] = rec
            prov_name[name] = prov_path.name
        for path, sha in (prov.get("sources") or {}).items():
            p = ROOT / path
            if not p.exists():
                problems.append(
                    f"{path}: a figure was drawn from it and it is gone")
            elif _sha(p) != sha:
                problems.append(
                    f"{path}: changed since the figures were drawn, so the "
                    f"images no longer show the current numbers; run make "
                    f"figures")

    generators = " or ".join(g for g, _ in FIGURE_GENERATORS)
    for p in sorted(FIGURES.iterdir()):
        if p.name.startswith("."):
            continue
        if p.suffix.lower() in {".png", ".svg", ".pdf", ".jpg", ".jpeg"}:
            if p.name not in produced:
                problems.append(
                    f"figures/{p.name}: not produced by {generators}, so "
                    f"nothing ties it to a frozen artifact")
            elif p.name not in recorded:
                problems.append(
                    f"figures/{p.name}: absent from the provenance file of "
                    f"{produced[p.name]}, so nothing records which data it was "
                    f"drawn from")
            elif _sha(p) != recorded[p.name]["sha256"]:
                problems.append(
                    f"figures/{p.name}: bytes differ from the ones "
                    f"{prov_name[p.name]} recorded")
        elif p.suffix.lower() in {".json", ".py"}:
            problems.append(
                f"figures/{p.name}: figures/ holds images only; data and "
                f"generators live in results/ and tools/")

    problems.extend(_caption_problems(recorded, prov_name))
    return problems


def _caption_problems(recorded: dict, prov_name: dict) -> list[str]:
    """The caption under an image must be the one the generator emitted.

    Neither figure carries a title any more: the descriptive text is a caption,
    it sits under the image, and it states a structure count, a resample count,
    a seed and a standard error. That makes it as perishable as any other number
    here, so it is generated with the plot and checked like the plot. Checking
    it needs no plotting library, which is why it lives in this gate and not in
    ``make figures``.
    """
    problems: list[str] = []
    if not recorded or not README.exists():
        return problems
    text = README.read_text()
    for name, rec in sorted(recorded.items()):
        caption = (rec or {}).get("caption")
        where = prov_name.get(name, "the provenance file")
        if not caption:
            problems.append(
                f"figures/{name}: {where} records no caption, "
                f"so the text under the image is not tied to the artifacts")
        elif caption not in text:
            problems.append(
                f"figures/{name}: the caption in README.md is not the one "
                f"{where} recorded; run make figures")
    return problems


def _consistency(man: dict) -> list[str]:
    problems = _figure_problems()
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
