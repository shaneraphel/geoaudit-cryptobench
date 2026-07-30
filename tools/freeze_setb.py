#!/usr/bin/env python3.12
"""Freeze a built external set, after rechecking the properties that make it one.

What freezing is here
---------------------
Not a flag. The freeze is the commit: once the artifact is in history at a fixed
digest, a preregistration written in a later commit can pin that digest, and
``EXTERNAL_ORDER.json`` can prove by ``git merge-base`` that the plan came after
the set and before the read. This tool prepares the artifact for that commit and
prints the digest the plan will have to name.

Why it rechecks instead of trusting the build
---------------------------------------------
The builder already asserted these properties, and that is exactly why they are
checked again. A set is frozen once; if a later edit, a rerun with a different
cluster map, or a copied artifact has quietly broken cluster-disjointness, the
moment to find out is before the digest goes into history and a plan is written
against it. Set A is spent because it was read, and the read is only worth
anything if the set was what it claimed. So the checks below recompute from the
sources rather than reading the build's own summary:

  * no unit shares a UniRef50 cluster with CryptoBench,
  * no unit shares a cluster with Set A, whose read is already spent,
  * no unit's accession appears in CryptoBench,
  * one unit per cluster,
  * every unit carries at least one labelled residue,
  * no field anywhere in the artifact looks like a score, a prediction or a
    metric, because a set that has been read cannot be frozen as unread.

Any failure refuses to write. There is no ``--force``.

What this does not do
---------------------
It does not preregister and it does not read. Naming the comparisons, the
statistic and the sentence to write under each outcome is a separate artifact in a
separate commit, and it has to exist before any method touches these units.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

from pocket_bench.paths import ROOT

SCHEMA_SUFFIX = ".frozen"
SET_A = ROOT / "results/external/EXTERNAL_SET.json"
UNIREF_A = ROOT / "data/external/UNIREF50.json"
RULE = ROOT / "results/external/CRYPTOBENCH_RULE.json"
# Every external set already frozen. A new one must be disjoint from all of them,
# not only from Set A: two frozen sets sharing a cluster would make their two
# reads correlated, which is the one thing a second set exists to avoid. Listed
# by path and checked by recomputation, so adding a third set means adding it here.
ALREADY_FROZEN = (SET_A, ROOT / "results/external/SETB_SET.json")

# Substrings that must not appear as keys anywhere in a set being frozen. A set
# that carries a metric has been read, and freezing it as unread would make the
# order claim false.
FORBIDDEN_KEY = re.compile(
    r"auc|roc|f1|precision|recall|mcc|score|predict|margin|delta|bootstrap|"
    r"confidence|ci_|p_value", re.I)
# Keys that contain one of those substrings and are legitimate.
ALLOWED = {"n_positive_residues", "prediction_provenance"}


class NotFreezable(RuntimeError):
    """A property that makes this a valid external set does not hold."""


# The set being frozen, so the disjointness check does not compare it with itself
# and report every one of its own clusters as a collision.
_TARGET: dict[str, Path] = {}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_keys(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            yield f"{path}/{k}", k
            yield from _walk_keys(v, f"{path}/{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _walk_keys(v, f"{path}[{i}]")


def check_no_read(doc: dict) -> list[str]:
    bad = []
    for where, key in _walk_keys(doc):
        if key in ALLOWED:
            continue
        if FORBIDDEN_KEY.search(key):
            bad.append(f"{where}: key {key!r} looks like a metric or a prediction")
    if doc.get("no_method_has_been_run") is not True:
        bad.append("no_method_has_been_run is not True")
    return bad


def check_disjoint(doc: dict) -> list[str]:
    """Recompute the exclusions from their sources rather than trusting the build."""
    bad: list[str] = []
    units = doc.get("units") or []
    all_units = units + (doc.get("units_without_a_cryptic_pocket") or [])
    if not units:
        bad.append("the set has no unit carrying a labelled cryptic pocket")

    cb_clusters: set[str] = set()
    cb_acc: set[str] = set()
    if UNIREF_A.is_file():
        u = json.loads(UNIREF_A.read_text())
        cb_acc = set(u.get("cryptobench") or {})
        cb_clusters = set((u.get("cryptobench") or {}).values())
    spent: dict[str, set[str]] = {}
    for p in ALREADY_FROZEN:
        if not p.is_file() or p.resolve() == _TARGET.get("path"):
            continue
        other = json.loads(p.read_text())
        cl = {x["cluster"] for x in other["units"]}
        cl |= {x["cluster"] for x in other.get("units_without_a_cryptic_pocket", [])
               if "cluster" in x}
        spent[str(p.relative_to(ROOT))] = cl

    seen: dict[str, str] = {}
    for x in all_units:
        clu, acc = x.get("cluster"), x.get("uniprot")
        if clu is None or acc is None:
            bad.append(f"a unit carries no cluster or no accession: {x.get('apo')}")
            continue
        if clu in cb_clusters:
            bad.append(f"{acc}: cluster {clu} is shared with CryptoBench")
        for where, cl in spent.items():
            if clu in cl:
                bad.append(f"{acc}: cluster {clu} is shared with {where}, so the "
                           f"two sets' reads would be correlated")
        if acc in cb_acc:
            bad.append(f"{acc}: the accession itself appears in CryptoBench")
        if clu in seen and seen[clu] != acc:
            bad.append(f"cluster {clu} carries two accessions, {seen[clu]} and "
                       f"{acc}; one unit per cluster does not hold")
        seen[clu] = acc

    for x in units:
        if not x.get("residues"):
            bad.append(f"{x.get('uniprot')}: in units but carries no labelled "
                       f"residue")
    return bad


def head() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("set", type=Path, help="the built set artifact to freeze")
    ap.add_argument("--name", required=True,
                    help="what this set is called in prose, e.g. 'Set B'")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    path = a.set if a.set.is_absolute() else ROOT / a.set
    _TARGET["path"] = path.resolve()
    doc = json.loads(path.read_text())

    problems = check_no_read(doc) + check_disjoint(doc)
    if problems:
        for p in problems[:40]:
            print(f"  REFUSED: {p}")
        raise NotFreezable(
            f"{len(problems)} properties do not hold; nothing was written. There "
            f"is no --force: a set is frozen once and a plan will be written "
            f"against its digest")

    units = doc["units"]
    n_res = sum(len(u["residues"]) for u in units)
    clusters = sorted({u["cluster"] for u in units})
    doc["is_a_frozen_set"] = True
    doc.pop("why_not_frozen_yet", None)
    doc["frozen"] = {
        "name": a.name,
        "frozen_on": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "head_when_frozen": head(),
        "n_units": len(units),
        "n_units_without_a_cryptic_pocket": len(
            doc.get("units_without_a_cryptic_pocket") or []),
        "n_positive_residues": n_res,
        "n_uniref50_clusters": len(clusters),
        "one_unit_per_cluster": len(clusters) == len(units),
        "labelling_rule": str(RULE.relative_to(ROOT)) if RULE.is_file() else None,
        "labelling_rule_sha256": sha256_of(RULE) if RULE.is_file() else None,
        "checks_recomputed_at_freeze_time": [
            "no unit shares a UniRef50 cluster with CryptoBench",
            "no unit shares a cluster with Set A, whose read is spent",
            "no unit's accession appears in CryptoBench",
            "one unit per cluster",
            "every unit in units carries at least one labelled residue",
            "no key anywhere in the artifact looks like a score or a metric",
        ],
        "why_recomputed_rather_than_trusted": (
            "the builder asserted these too. A set is frozen once and a "
            "preregistration will be written against this digest, so a property "
            "broken by a later edit or a different cluster map has to be caught "
            "before the digest enters history, not after"),
        "what_freezing_does_not_do": (
            "it does not preregister and it does not read. The comparisons, the "
            "statistic and the sentence to write under each outcome are a separate "
            "artifact in a later commit, and that artifact must exist before any "
            "method touches these units"),
        "what_spends_this_set": (
            "one read. Scoring a method that was improved after this digest was "
            "pinned does not produce a second confirmation, it destroys the first"),
    }
    if a.write:
        path.write_text(json.dumps(doc, indent=1) + "\n")

    print(f"  {a.name}: {len(units)} units, {n_res} labelled residues, "
          f"{len(clusters)} clusters")
    print(f"  one unit per cluster: {doc['frozen']['one_unit_per_cluster']}")
    print(f"  every recomputed check passed")
    if a.write:
        print(f"\nwrote {path.relative_to(ROOT)}")
        print(f"  sha256 after writing: {sha256_of(path)}")
        print("  commit this file, then write the preregistration in the NEXT "
              "commit pinning that digest")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
