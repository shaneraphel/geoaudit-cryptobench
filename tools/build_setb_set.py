#!/usr/bin/env python3.12
"""Select the candidate units for Set B, a second external set from cryo-EM.

Why a second set has to exist, and why now
------------------------------------------
``results/external/EXTERNAL_SET.json`` is frozen, hashed, pinned by a
preregistration, and read. It is spent: scoring an improved method on it does not
produce a second confirmatory result, it destroys the first. A second set can only
serve that purpose if it is built and frozen *before* the method that will be read
on it is finalised. So it is built now, while every method experiment in flight is
returning null, rather than after one of them returns something.

This tool selects and does not read. No score, no prediction and no metric appears
in what it writes.

Why it is a separate file from build_external_set.py
---------------------------------------------------
``EXTERNAL_SET.json`` carries a ``code_sha256`` over ``build_external_set.py``
alone. Editing that file to add a second inventory would invalidate the hash on a
frozen artifact, so nothing here edits it: the labelling machinery is imported and
reused unchanged, which is what makes the two sets comparable, and only the
candidate selection is rewritten. Set A's receptors, labels and manifest are
written to different paths and are not touched.

What differs from Set A, and what deliberately does not
-------------------------------------------------------
Different: the inventory is single-particle cryo-EM rather than X-ray. That is the
point of the set --- a different experimental modality, so a confirmation on it is
not a confirmation on the same kind of data twice.

Also different: the exclusion is wider. Set A excludes CryptoBench's accessions and
their UniRef50 clusters. Set B must additionally exclude the clusters Set A itself
spent, because two reads that share sequence clusters are correlated and two
correlated reads are not two confirmations.

Deliberately identical: the ligand-relevance rule, the apo and holo definitions,
the one-unit-per-cluster rule, the apo chain choice, the partner cap and the
labelling. All of it is the imported code path.

The technique filter is not optional
------------------------------------
``experimental_method == "EM"`` admits electron crystallography and helical
reconstruction as well as single-particle imaging. ``SETB_POOL.json`` records this
and records that MicroED sits at the top of any resolution-ordered selection, which
is not hypothetical: this repository downloaded EMD-46871 as a cryo-EM example and
it is MicroED at 1.09 A. Selecting by resolution without the reconstruction-method
filter would draw a set disproportionately from the one technique the set does not
exist to test. The filter is applied to the inventory before anything else.

Usage:
  PYTHONPATH=src:tools python3.12 tools/build_setb_set.py --candidates
  PYTHONPATH=src:tools python3.12 tools/build_setb_set.py --candidates --write
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path

import build_external_set as A  # noqa: N812  -- the imported, unedited machinery
from external_setb_probe import _single_particle_ids  # noqa: E402

from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.setb_candidates.v1"
# The resolution ceiling is a parameter and not a constant, because at 2.5 A the
# candidate count is 40 and at 3.0 A it is several times that. Which one the set
# uses changes what the set means, so it is chosen explicitly and recorded in the
# artifact rather than baked in.
CEILING = 2.5
UNIREF_SETB = ROOT / "data/external/UNIREF50_SETB.json"
UNIREF_A = ROOT / "data/external/UNIREF50.json"
SET_A = ROOT / "results/external/EXTERNAL_SET.json"
OUT = ROOT / "results/external/SETB_CANDIDATES.json"


def probe_cache(ceiling: float) -> Path:
    return ROOT / f"data/external/_setb_probe_em_{ceiling:.1f}.json"


def sp_cache(ceiling: float) -> Path:
    return ROOT / f"data/external/_setb_single_particle_{ceiling:.1f}.json"


def inventory_path(ceiling: float) -> Path:
    return ROOT / f"data/external/INVENTORY_SETB_{ceiling:.1f}.json.gz"


def single_particle_ids(ceiling: float = CEILING,
                        refresh: bool = False) -> set[str]:
    """Entry IDs that are single-particle reconstructions, cached on disk.

    Cached because the selection has to be reproducible from the repository
    without assuming the archive is unchanged, and because RCSB's counts drift.
    The cache records the date it was taken.
    """
    cache = sp_cache(ceiling)
    if cache.is_file() and not refresh:
        got = json.loads(cache.read_text())
        if got.get("ceiling") != ceiling:
            raise SystemExit(
                f"{cache.relative_to(ROOT)} was taken at ceiling "
                f"{got.get('ceiling')} and is being read at {ceiling}. Filtering "
                f"a wider probe cache by a narrower id list silently collapses it "
                f"to the narrower set, which is how a 3.0 A count first came back "
                f"identical to the 2.5 A one")
        return {i.upper() for i in got["ids"]}
    ids = sorted({i.upper() for i in _single_particle_ids(ceiling)})
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "schema": "geoaudit.setb_single_particle_ids.v1",
        "taken_on": time.strftime("%Y-%m-%d"),
        "ceiling": ceiling,
        "attribute": "em_experiment.reconstruction_method",
        "value": "SINGLE PARTICLE",
        "why": "experimental_method == EM also admits electron crystallography "
               "and helical reconstruction; see SETB_POOL.json",
        "n_ids": len(ids),
        "ids": ids,
    }, indent=1) + "\n")
    return set(ids)


def inventory(ceiling: float = CEILING, refresh: bool = False) -> dict:
    """The probe cache, filtered to single particle, in the Set A inventory shape.

    Same keys as data/external/INVENTORY.json.gz, so the imported selection and
    labelling read it without knowing which modality produced it.
    """
    out = inventory_path(ceiling)
    if out.is_file() and not refresh:
        return json.loads(gzip.decompress(out.read_bytes()))
    cache = probe_cache(ceiling)
    if not cache.is_file():
        raise SystemExit(
            f"{cache.relative_to(ROOT)} is absent; run "
            f"tools/external_setb_probe.py to fetch the describe cache")
    blob = json.loads(cache.read_text())
    keep = single_particle_ids(ceiling, refresh)
    rows = [r for r in blob["rows"] if r["pdb"].upper() in keep]
    doc = {
        "schema": "geoaudit.inventory.v1",
        "modality": "single-particle cryo-EM",
        "cutoff": "2025-01-01",
        "max_resolution": ceiling,
        "reads_any_fold_labels": False,
        "n_entries": blob["n_entries"],
        "n_chains_before_method_filter": len(blob["rows"]),
        "reconstruction_method": "SINGLE PARTICLE",
        "chains": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.compress((json.dumps(doc, indent=1) + "\n").encode()))
    return doc


def spent_clusters() -> tuple[set[str], set[str]]:
    """(CryptoBench's clusters, Set A's clusters). Both are excluded from Set B."""
    cb: set[str] = set()
    if UNIREF_A.is_file():
        cb = set((json.loads(UNIREF_A.read_text()).get("cryptobench") or {}).values())
    a = json.loads(SET_A.read_text())
    a_clusters = {u["cluster"] for u in a["units"]}
    a_clusters |= {u["cluster"] for u in a.get("units_without_a_cryptic_pocket", [])
                   if "cluster" in u}
    return cb, a_clusters


def candidates(ceiling: float = CEILING, refresh: bool = False,
               exclude_clusters: set[str] | None = None,
               exclude_source: str | None = None
               ) -> tuple[list[dict], dict]:
    """Set A's selection rule, on the cryo-EM inventory, with Set A also excluded.

    The body mirrors build_external_set.candidates so that the two sets are
    selected the same way. The three differences are named in the returned meta:
    the inventory, the cluster map, and the extra exclusion.
    """
    inv = inventory(ceiling, refresh)
    uni_b = json.loads(UNIREF_SETB.read_text())
    cluster_of = uni_b["cluster_of"]
    cb_clusters, a_clusters = spent_clusters()
    cb_acc: set[str] = set()
    if UNIREF_A.is_file():
        cb_acc = set((json.loads(UNIREF_A.read_text()).get("cryptobench") or {}))
    accepted = {r["ligand"] for v in
                json.loads(A.CB_DATASET.read_text()).values() for r in v}

    by: dict[str, dict[str, list]] = {}
    for c in inv["chains"]:
        rel = sorted({l["code"] for l in c["ligands"]
                      if l["code"] in accepted and c["chain"] in l["chains"]})
        side = "holo" if rel else "apo"
        by.setdefault(c["uniprot"], {"holo": [], "apo": []})[side].append(
            dict(c, relevant=rel))

    reasons: dict[str, int] = {}

    def drop(why: str, n: int = 1) -> None:
        reasons[why] = reasons.get(why, 0) + n

    per_cluster: dict[str, list[tuple[str, dict]]] = {}
    for acc, v in sorted(by.items()):
        if not (v["holo"] and v["apo"]):
            drop("the accession has no apo chain and holo chain together")
            continue
        if acc in cb_acc:
            drop("the accession itself appears in CryptoBench")
            continue
        # Two different things, and conflating them is how a missing measurement
        # gets read as a negative one. The cached map was fetched for one pool; an
        # accession outside it is unmapped, not unmappable, and raising the
        # ceiling puts hundreds of accessions in that position. UniProt returning
        # no cluster is the real exclusion.
        if acc not in cluster_of:
            drop("the accession is absent from the cached UniRef50 map, which "
                 "is not the same as UniProt being unable to cluster it; rerun "
                 "tools/setb_uniref.py over this ceiling's pool")
            continue
        clu = cluster_of[acc]
        if clu is None:
            drop("UniProt cannot place the accession in a UniRef50 cluster")
            continue
        if clu in cb_clusters:
            drop("the accession shares a UniRef50 cluster with CryptoBench")
            continue
        if clu in a_clusters:
            drop("the accession shares a UniRef50 cluster with Set A, which "
                 "would make the two reads correlated")
            continue
        # A ceiling of 2.5 A selects a subset of what 3.0 A selects, so two sets
        # cut at two ceilings are not two sets -- the tighter one is inside the
        # wider one and reading both would read its units twice. Excluding the
        # clusters an earlier selection already took makes the second one
        # genuinely disjoint.
        if exclude_clusters and clu in exclude_clusters:
            drop(f"the cluster is already taken by {exclude_source}")
            continue
        per_cluster.setdefault(clu, []).append((acc, v))

    units: list[dict] = []
    for clu, group in sorted(per_cluster.items()):
        acc, v = sorted(group, key=lambda g: (
            A._pick_apo(g[1]["apo"])["resolution"] or 99.0, g[0]))[0]
        if len(group) > 1:
            drop("a second accession in an already-represented cluster",
                 len(group) - 1)
        apo = A._pick_apo(v["apo"])
        partners = sorted(v["holo"], key=lambda c: (
            c["resolution"] if c["resolution"] is not None else 99.0,
            c["pdb"], c["chain"]))[:A.MAX_HOLO_PARTNERS]
        units.append({"uniprot": acc, "cluster": clu, "apo": apo,
                      "partners": partners,
                      "n_partners_available": len(v["holo"])})
    meta = {
        "reasons": reasons,
        "accepted_ligand_codes": len(accepted),
        "n_clusters": len(per_cluster),
        "cutoff": inv["cutoff"],
        "max_resolution": inv["max_resolution"],
        "modality": inv["modality"],
        "how_this_differs_from_set_a": {
            "inventory": "single-particle cryo-EM rather than X-ray",
            "cluster_map": str(UNIREF_SETB.relative_to(ROOT)),
            "extra_exclusion": "the UniRef50 clusters Set A spent, so that a read "
                               "on this set is independent of the read on that one",
        },
        "what_is_identical_to_set_a": [
            "the accepted ligand codes, taken from CryptoBench's own dataset",
            "the apo and holo definitions",
            "one unit per UniRef50 cluster",
            "the apo chain choice and the partner cap, imported not reimplemented",
        ],
    }
    return units, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", action="store_true",
                    help="select and report; downloads nothing and labels nothing")
    ap.add_argument("--ceiling", type=float, default=CEILING,
                    help="apo resolution ceiling in Angstrom. 2.5 gives 40 "
                         "candidate units, 3.0 gives several times that at coarser "
                         "maps; the choice changes what the set means")
    ap.add_argument("--exclude-clusters-from", type=Path, default=None,
                    help="a SETB_CANDIDATES artifact whose clusters this "
                         "selection must not reuse. Needed because a tighter "
                         "ceiling selects a subset of a wider one, so without it "
                         "the two selections overlap and are not two sets")
    ap.add_argument("--refresh", action="store_true",
                    help="refetch the single-particle id list and rebuild the "
                         "inventory")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args(argv)
    if not a.candidates:
        raise SystemExit(
            "only --candidates is implemented. Selection is separated from "
            "fetching and labelling on purpose: the candidate count decides "
            "whether the set is worth building, and it costs no downloads")

    excl: set[str] = set()
    excl_src = None
    if a.exclude_clusters_from is not None:
        prior = json.loads(a.exclude_clusters_from.read_text())
        excl = {u["cluster"] for u in prior["units"]}
        excl_src = str(a.exclude_clusters_from.name)
    units, meta = candidates(a.ceiling, a.refresh, excl, excl_src)
    if excl:
        overlap = excl & {u["cluster"] for u in units}
        if overlap:
            raise SystemExit(
                f"{len(overlap)} clusters appear in both selections; the "
                f"exclusion did not take and the two sets are not disjoint")
        meta["disjoint_from"] = {
            "artifact": excl_src,
            "n_clusters_excluded": len(excl),
            "checked": "no cluster appears in both selections",
            "why_it_is_needed": "a 2.5 A ceiling selects a subset of what 3.0 A "
                                "selects, so two selections cut at two ceilings "
                                "share their tighter half. Reading both without "
                                "this exclusion would read those units twice and "
                                "the second read would not be independent",
        }
    res = sorted(u["apo"]["resolution"] for u in units
                 if u["apo"]["resolution"] is not None)
    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "is_a_frozen_set": False,
        "what_this_is": (
            "the candidate units for a second external set, selected from "
            "single-particle cryo-EM depositions by Set A's rule with Set A's own "
            "clusters additionally excluded"),
        "what_this_is_not": (
            "not a frozen set and not a read. Nothing here is labelled, hashed, "
            "preregistered or scored. Building the set is a separate step and "
            "reading it needs a preregistration that does not yet exist"),
        "why_it_is_built_before_the_method_is_final": (
            "a second set is only confirmatory if it is frozen before the method "
            "that will be read on it is finalised. Set A was read and is spent"),
        "n_candidate_units": len(units),
        "n_clusters": meta["n_clusters"],
        "apo_resolution_angstrom": {
            "min": res[0] if res else None,
            "median": res[len(res) // 2] if res else None,
            "max": res[-1] if res else None,
        },
        "n_partners_total": sum(len(u["partners"]) for u in units),
        "selection": meta,
        "units": units,
    }
    out = a.out if a.out.is_absolute() else ROOT / a.out
    if a.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1) + "\n")

    print(f"  inventory: {meta['modality']}, ceiling {meta['max_resolution']} A, "
          f"cutoff {meta['cutoff']}")
    print(f"  {len(units)} candidate units in {meta['n_clusters']} clusters")
    if res:
        print(f"  apo resolution: {res[0]:.2f} / {res[len(res) // 2]:.2f} / "
              f"{res[-1]:.2f} A  (min / median / max)")
    print("  dropped, by reason:")
    for why, n in sorted(meta["reasons"].items(), key=lambda kv: -kv[1]):
        print(f"    {n:6d}  {why}")
    if a.write:
        print(f"\nwrote {out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
