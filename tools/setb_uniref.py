#!/usr/bin/env python3.12
"""Cluster the Set B pool, and answer the question SETB_POOL.json leaves open.

The gap this closes
-------------------
``SETB_POOL.json`` reports 455 single-particle accessions with both an apo and a
holo form, none of them in CryptoBench, and then reports the cluster count as
``NOT MEASURED`` with 455 of 455 unmapped. That is not an empty axis -- it is an
absent measurement, and the artifact says so deliberately, because an earlier
version printed ``0 clusters remain`` and read as the opposite of what had been
found. The cached ``UNIREF50.json`` was fetched for Set A's X-ray accessions and
covers none of these.

Why this is a separate file and not ``--force``
-----------------------------------------------
``external_uniref.py --force`` would remap Set A's candidate inventory and
overwrite ``data/external/UNIREF50.json``. Set A is frozen, hashed and spent, and
``build_external_set.py`` reads that mapping, so overwriting it would change what
a rebuild of a frozen artifact produces. The rule in this repository is to check
what pins what before editing, and to prefer a small blast radius near a frozen
artifact over tidiness. So the pool is mapped into its own file and nothing Set A
reads is touched. The mapping call itself is imported from ``external_uniref``
rather than reimplemented, so the two files cluster identically.

What the answer means
---------------------
Set B exists to be a second confirmatory read, and two reads that share sequence
clusters are correlated rather than independent. So the number that decides
viability is not 455: it is how many of these accessions fall in UniRef50
clusters that neither CryptoBench nor Set A has already spent. That count is what
this writes, together with the accessions UniProt could not cluster, which are
excluded rather than assumed novel -- an accession that cannot be shown unrelated
to CryptoBench is not evidence of independence.

This selects nothing, freezes nothing and scores nothing. It reports how large a
cluster-disjoint Set B could be.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from external_uniref import BATCH, _map  # noqa: E402

from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.setb_uniref50.v1"
POOL = ROOT / "results/external/SETB_POOL.json"
SET_A = ROOT / "results/external/EXTERNAL_SET.json"
CB_ACC = ROOT / "data/external/CRYPTOBENCH_ACCESSIONS.json"
UNIREF_A = ROOT / "data/external/UNIREF50.json"
OUT = ROOT / "data/external/UNIREF50_SETB.json"


def spent_clusters() -> tuple[set[str], set[str]]:
    """(clusters Set A has spent, clusters CryptoBench occupies)."""
    a = json.loads(SET_A.read_text())
    a_clusters = {u["cluster"] for u in a["units"]}
    a_clusters |= {u["cluster"] for u in a.get("units_without_a_cryptic_pocket", [])
                   if "cluster" in u}
    cb_clusters: set[str] = set()
    if UNIREF_A.exists():
        uni = json.loads(UNIREF_A.read_text())
        cb_clusters = set((uni.get("cryptobench") or {}).values())
    return a_clusters, cb_clusters


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    pool_doc = json.loads(POOL.read_text())
    sp = pool_doc.get("single_particle_pool")
    if not sp:
        raise SystemExit(
            "SETB_POOL.json carries no single_particle_pool; run "
            "tools/external_setb_probe.py --write --from-cache --methods first")
    accessions = sorted(sp["pool_accessions"])
    a_clusters, cb_clusters = spent_clusters()

    print(f"mapping {len(accessions)} single-particle pool accessions to "
          f"UniRef50", flush=True)
    got: dict[str, str] = {}
    for i in range(0, len(accessions), BATCH):
        got.update(_map(accessions[i:i + BATCH]))
        print(f"  {len(got)}/{len(accessions)}", flush=True)

    unmapped = [x for x in accessions if x not in got]
    clusters = {}
    for acc, cl in got.items():
        clusters.setdefault(cl, []).append(acc)
    collide_a = {c for c in clusters if c in a_clusters}
    collide_cb = {c for c in clusters if c in cb_clusters}
    free = sorted(set(clusters) - a_clusters - cb_clusters)
    free_acc = sorted(x for c in free for x in clusters[c])

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "is_a_frozen_set": False,
        "what_this_is": (
            "the UniRef50 cluster of every accession in the single-particle Set B "
            "pool, and how many of those clusters neither CryptoBench nor Set A "
            "has already spent"),
        "what_this_is_not": (
            "not a set and not a selection. No accession here has been chosen, "
            "labelled, hashed or preregistered, and nothing has been scored"),
        "why_a_separate_file": (
            "external_uniref.py --force would remap Set A's inventory and "
            "overwrite data/external/UNIREF50.json, which build_external_set.py "
            "reads and which a frozen, hashed and spent artifact depends on. The "
            "mapping function is imported from that tool so the clustering is "
            "identical; only the destination differs"),
        "source": "UniProt id-mapping, UniProtKB_AC-ID to UniRef50",
        "identity": 0.5,
        "unmapped_are_kept_out": (
            "an accession UniProt cannot cluster cannot be shown unrelated to "
            "CryptoBench, so it is excluded rather than assumed novel"),
        "n_pool_accessions": len(accessions),
        "n_mapped": len(got),
        "n_unmapped": len(unmapped),
        "n_clusters": len(clusters),
        "n_clusters_colliding_with_set_a": len(collide_a),
        "n_clusters_colliding_with_cryptobench": len(collide_cb),
        "n_clusters_free": len(free),
        "n_accessions_in_free_clusters": len(free_acc),
        "largest_free_cluster": max((len(clusters[c]) for c in free), default=0),
        "how_to_read_the_headline": (
            "n_clusters_free is the ceiling on how many independent units a "
            "cluster-disjoint Set B could hold, not the size of Set B. Each unit "
            "still needs an apo form, a holo form and a cryptic pocket that "
            "survives the same labelling rule Set A used, and every one of those "
            "only removes candidates"),
        "unmapped_accessions": unmapped,
        "free_clusters": free,
        "accessions_in_free_clusters": free_acc,
        "cluster_of": got,
    }
    if a.write:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(doc, indent=1) + "\n")

    print(f"\n  mapped {len(got)} of {len(accessions)}; "
          f"{len(unmapped)} unclusterable and excluded")
    print(f"  {len(clusters)} clusters, of which "
          f"{len(collide_a)} collide with Set A and "
          f"{len(collide_cb)} with CryptoBench")
    print(f"  {len(free)} clusters free, holding {len(free_acc)} accessions")
    if a.write:
        print(f"\nwrote {a.out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
