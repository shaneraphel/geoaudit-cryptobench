"""Is there a second external set in cryo-EM, and at what resolution does it exist?

Set A used every X-ray cluster the cutoff allowed, so a second confirmatory read
needs a different axis rather than a relaxed threshold on the same one. Cryo-EM is
the natural one: CryptoBench is X-ray only, so an EM entry is external to it twice
over, by date and by method.

Whether that axis has enough on it is a question about the PDB, not a matter of
opinion, and it has to be answered before any architecture work starts --- if the
answer is thirty units the axis is wrong and the work would be aimed at a target
that cannot resolve anything.

This probe counts and does not build. It reads no label and writes no set.

Usage: PYTHONPATH=src:tools python3.12 tools/external_setb_probe.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CUTOFF = "2024-05-08"
SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query?json="
# Cryo-EM global resolution is an average over the map, and a cryptic pocket sits
# in exactly the kind of flexible region where local resolution is worst. So the
# ceiling is a decision with a cost either way, and the probe reports the curve
# rather than picking a point.
LADDER = (2.5, 3.0, 3.5, 4.0)
# The enum value is "EM". Spelling it "Electron Microscopy" returns zero rather
# than an error, which would have read as "this axis is empty" and sent the whole
# experiment down a different road on the strength of a wrong string.
EM = "EM"


def _count(method: str, max_resolution: float | None) -> int:
    nodes = [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_accession_info.initial_release_date",
            "operator": "greater", "value": CUTOFF}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
            "operator": "greater_or_equal", "value": 1}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.experimental_method",
            "operator": "exact_match", "value": method}},
    ]
    if max_resolution is not None:
        nodes.append({"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "less_or_equal", "value": max_resolution}})
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": nodes},
         "return_type": "entry",
         "request_options": {"paginate": {"start": 0, "rows": 1},
                             "results_content_type": ["experimental"]}}
    url = SEARCH + urllib.parse.quote(json.dumps(q))
    # Retried twelve times over a VPN that drops TLS mid-handshake, and a
    # failure raises rather than returning zero. A transient error
    # that reads as "no such entries" would answer the feasibility question in the
    # wrong direction and the answer would look like data.
    for attempt in range(12):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                body = r.read()
            return int(json.loads(body).get("total_count", 0)) if body else 0
        except Exception:                             # noqa: BLE001 - all transient
            if attempt == 11:
                raise
            time.sleep(min(2.0 * (attempt + 1), 15.0))
    raise AssertionError("unreachable")


def _inventory():
    """external_inventory, with its retries raised for a VPN that drops TLS.

    Patched here rather than in that module: it is what built Set A, and a tool
    that feeds a frozen artifact is not the place to make an unrelated edit, even
    one that cannot change a result. The five tries it ships with are enough on a
    good link and not enough on this one, and a describe of 1,706 entries makes
    twelve batched calls where any single failure loses the run.
    """
    import external_inventory as inv

    original = inv._post

    def patient(url: str, payload: dict, tries: int = 5) -> dict:
        last: Exception | None = None
        for attempt in range(12):
            try:
                return original(url, payload, tries=1)
            except Exception as exc:                   # noqa: BLE001 - transient
                last = exc
                time.sleep(min(2.0 * (attempt + 1), 15.0))
        assert last is not None
        raise last

    inv._post = patient
    return inv


def _pairable_from_cache(ceiling: float) -> dict:
    """The same count, computed from the cached describe without a network call.

    Why this path exists. The probe was print-only, so the pool size it found is
    quoted in AGENTS.md and in a commit message and stands behind no registered
    artifact -- a number in this repository's own law that a reader cannot check.
    Set B has to be drawn from that pool, and a set frozen against an unpinned
    inventory is not auditable, so the inventory is pinned first.

    The network path verifies the cache by comparing its entry count against a
    fresh search. This path cannot, so it records the cache's SHA-256 and says
    plainly that the entry count is the cached one. A reader who wants it
    reverified runs the probe without --from-cache.
    """
    cache = ROOT / f"data/external/_setb_probe_em_{ceiling:.1f}.json"
    if not cache.exists():
        raise SystemExit(
            f"{cache.relative_to(ROOT)} is absent; run the probe without "
            f"--from-cache to fetch it")
    raw = cache.read_bytes()
    blob = json.loads(raw)
    rows = blob["rows"]
    cb_accessions, a_clusters = _already_used()
    uni = json.loads((ROOT / "data/external/UNIREF50.json").read_text())
    cluster_of = uni.get("cluster_of") or uni.get("clusters") or {}

    apo: dict[str, set[str]] = {}
    holo: dict[str, set[str]] = {}
    for r in rows:
        on_this_chain = any(r["chain"] in lig["chains"] for lig in r["ligands"])
        (holo if on_this_chain else apo).setdefault(r["uniprot"], set()).add(
            f'{r["pdb"]}_{r["chain"]}')

    both = sorted(set(apo) & set(holo))
    fresh = [a for a in both if a not in cb_accessions]
    unmapped = [a for a in fresh if a not in cluster_of]
    mapped_clusters = sorted({cluster_of[a] for a in fresh if a in cluster_of}
                             - a_clusters)
    return {
        "ceiling": ceiling,
        "entry_count_source": "cache",
        "cache": str(cache.relative_to(ROOT)),
        "cache_sha256": hashlib.sha256(raw).hexdigest(),
        "n_entries": blob["n_entries"],
        "n_chains": len(rows),
        "n_accessions": len({r["uniprot"] for r in rows}),
        "n_with_apo_and_holo": len(both),
        "n_in_cryptobench": len(both) - len(fresh),
        "n_pool": len(fresh),
        "pool_accessions": fresh,
        "n_unmapped_to_uniref50": len(unmapped),
        "n_clusters_new_to_set_a": (len(mapped_clusters)
                                    if len(unmapped) < len(fresh) else None),
        "cluster_count_is_absent_not_zero": len(unmapped) == len(fresh),
        "what_to_run_for_the_cluster_count": (
            "tools/external_uniref.py over pool_accessions; the cached UniRef50 "
            "mapping was fetched for Set A's X-ray accessions and covers none of "
            "these, so the cluster count is unavailable rather than zero"
        ) if len(unmapped) == len(fresh) else None,
    }


def _pairable(ceiling: float) -> dict:
    """How many EM accessions could yield an apo-holo pair that Set A has not used.

    An upper bound, deliberately. A chain counts as apo-capable when the deposit
    assigns it no ligand and holo-capable when it assigns it one, which is looser
    than the real test: the builder goes on to require the ligand to be one
    CryptoBench accepted, to make heavy-atom contact within 4.5 A, and to move the
    pocket by at least 2.5 A. Each of those only removes candidates. So if this
    number is small the axis is dead, and that is the question being asked.
    """
    inv = _inventory()
    ids = _search_ids(EM, ceiling)

    # Cached per ceiling, because this link drops TLS often enough that a run
    # which restarts from nothing does not finish. The cache holds the PDB's
    # answer to a fixed query, so a stale one is only a risk if the PDB grows
    # underneath it, which would show up as a changed entry count.
    cache = ROOT / f"data/external/_setb_probe_em_{ceiling:.1f}.json"
    if cache.exists():
        blob = json.loads(cache.read_text())
        if blob.get("n_entries") == len(ids):
            print(f"    (reusing cached describe of {len(ids):,} entries)",
                  flush=True)
            rows = blob["rows"]
        else:
            blob = None
    else:
        blob = None
    if blob is None:
        print(f"    describing {len(ids):,} EM entries at {ceiling:.1f} A ...",
              flush=True)
        rows = inv.flatten(inv.describe_all(ids))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"n_entries": len(ids), "rows": rows}))

    cb_accessions, a_clusters = _already_used()
    uni = json.loads((ROOT / "data/external/UNIREF50.json").read_text())
    cluster_of = uni.get("cluster_of") or uni.get("clusters") or {}

    apo: dict[str, set[str]] = {}
    holo: dict[str, set[str]] = {}
    for r in rows:
        on_this_chain = any(r["chain"] in lig["chains"] for lig in r["ligands"])
        (holo if on_this_chain else apo).setdefault(r["uniprot"], set()).add(
            f'{r["pdb"]}_{r["chain"]}')

    both = sorted(set(apo) & set(holo))
    fresh_accession = [a for a in both if a not in cb_accessions]
    unmapped = [a for a in fresh_accession if a not in cluster_of]
    fresh_cluster = sorted({cluster_of[a] for a in fresh_accession
                            if a in cluster_of} - a_clusters)
    return {"ceiling": ceiling, "n_entries": len(ids), "n_chains": len(rows),
            "n_accessions": len({r["uniprot"] for r in rows}),
            "n_with_apo_and_holo": len(both),
            "n_in_cryptobench": len(both) - len(fresh_accession),
            "n_unmapped_to_uniref50": len(unmapped),
            "n_clusters_new_to_set_a": len(fresh_cluster)}


def _search_ids(method: str, ceiling: float) -> list[str]:
    inv = _inventory()
    q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_accession_info.initial_release_date",
            "operator": "greater", "value": CUTOFF}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
            "operator": "greater_or_equal", "value": 1}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.experimental_method",
            "operator": "exact_match", "value": method}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "less_or_equal", "value": ceiling}}]},
        "return_type": "entry",
        # 10000 is the page size the working inventory query uses. Asking for
        # 20000 does not error, it hangs, which spent an hour looking like a
        # bad VPN.
        "request_options": {"paginate": {"start": 0, "rows": 10000},
                            "results_content_type": ["experimental"]}}
    got = inv._post("https://search.rcsb.org/rcsbsearch/v2/query", q)
    return [r["identifier"] for r in got.get("result_set", [])]


def _already_used() -> tuple[set[str], set[str]]:
    """CryptoBench's accessions, and the UniRef50 clusters Set A has spent.

    Set B has to be external to CryptoBench and independent of Set A. Sharing a
    cluster with A would make the two reads correlated, and two correlated reads
    are not two confirmations.
    """
    a = json.loads((ROOT / "results/external/EXTERNAL_SET.json").read_text())
    a_clusters = {u["cluster"] for u in a["units"]}
    a_clusters |= {u["cluster"] for u in a.get("units_without_a_cryptic_pocket", [])
                   if "cluster" in u}
    cb: set[str] = set()
    cb_path = ROOT / "data/external/CRYPTOBENCH_ACCESSIONS.json"
    if cb_path.exists():
        cb = set(json.loads(cb_path.read_text()))
    return cb, a_clusters


SETB_POOL = ROOT / "results/external/SETB_POOL.json"
SETB_SCHEMA = "geoaudit.setb_pool.v1"

# What `experimental_method == "EM"` actually admits, counted rather than assumed.
# Recorded because the filter's name misled: it is not a cryo-EM filter, it is an
# electron-method filter, and three of the four techniques below are different
# sample classes. Measured live against RCSB on the date in the artifact; the
# counts are a property of the archive at that moment and will drift.
EM_METHODS = ("SINGLE PARTICLE", "CRYSTALLOGRAPHY", "HELICAL",
              "SUBTOMOGRAM AVERAGING")


def _method_mix(ceiling: float) -> dict:
    """Entry counts by reconstruction method at one resolution ceiling.

    The distinction is not pedantry. ``CRYSTALLOGRAPHY`` here means electron
    diffraction from nanocrystals -- MicroED -- which behaves like X-ray
    crystallography rather than like single-particle imaging, so it is not the
    different structural modality a second external set would be for. And the
    counts alone understate the hazard: MicroED is 2.7 per cent of the pool by
    entry and occupies the top of any resolution-ordered selection from it, so
    the first entries anyone picks as "the sharpest examples" are the ones that
    are not cryo-EM. Both of the two sharpest in this pool are MicroED, and one
    of those deposited no density map at all.
    """
    out: dict[str, int | None] = {}
    for meth in EM_METHODS:
        q = {"query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.experimental_method",
                "operator": "exact_match", "value": EM}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "greater", "value": CUTOFF}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal", "value": ceiling}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "em_experiment.reconstruction_method",
                "operator": "exact_match", "value": meth}},
        ]}, "return_type": "entry",
            "request_options": {"paginate": {"start": 0, "rows": 1}}}
        url = ("https://search.rcsb.org/rcsbsearch/v2/query?json="
               + urllib.parse.quote(json.dumps(q)))
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(
                        url, headers={"User-Agent": "geoaudit/1.0"}),
                    timeout=90) as r:
                out[meth] = json.loads(r.read()).get("total_count")
        except Exception:  # noqa: BLE001
            out[meth] = None
    return out


def _write_pool(ceilings: tuple[float, ...], methods: bool = False) -> int:
    """Pin the pool Set B would be drawn from, and nothing else.

    This artifact is an inventory. It is not a set, it is not frozen, and it
    carries no prediction, no score and no label. Recording that distinction in
    the artifact matters here more than usual: results/external/EXTERNAL_SET.json
    is frozen, hashed, pinned by a preregistration and spent, and a second file
    under the same directory naming external accessions must not be mistakable
    for a second frozen set.
    """
    rows = [_pairable_from_cache(c) for c in ceilings]
    mix = None
    if methods:
        counted = _method_mix(2.5)
        total = sum(v for v in counted.values() if v)
        cached = rows[0]["n_entries"] if rows else None
        mix = {
            "measured_on": time.strftime("%Y-%m-%d"),
            "ceiling": 2.5,
            "by_reconstruction_method": counted,
            "total_over_methods": total,
            "entries_in_the_cached_describe": cached,
            "cache_drift": (total - cached) if (total and cached) else None,
            "what_the_filter_actually_admits": (
                "experimental_method == 'EM' is an electron-method filter and "
                "not a cryo-EM filter. CRYSTALLOGRAPHY here means electron "
                "diffraction from nanocrystals -- MicroED -- which behaves like "
                "X-ray crystallography rather than like single-particle imaging "
                "and is therefore not the different structural modality a "
                "second external set would exist to provide. HELICAL is "
                "filament reconstruction, a third sample class"),
            "why_the_percentage_understates_it": (
                "MicroED is a small fraction by entry and occupies the top of "
                "any resolution-ordered selection. Both of the two sharpest "
                "entries in this pool are MicroED and one of them deposited no "
                "density map at all, so a selection made by picking the "
                "sharpest examples lands on the technique the set is not for"),
            "consequence_for_set_b": (
                "the pool counts are an upper bound in one more respect than "
                "was recorded: they mix techniques. A set built from them should "
                "filter on em_experiment.reconstruction_method and not on "
                "experimental_method, and the count after that filter has not "
                "been taken"),
        }
    doc = {
        "schema": SETB_SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "is_a_frozen_set": False,
        "what_this_is": "an inventory of the cryo-EM accessions a second "
                        "external set could be drawn from, pinned so that the "
                        "set can be built against a checkable pool",
        "what_this_is_not": [
            "not a frozen set: no accession here has been selected, labelled, "
            "hashed or preregistered",
            "not a measurement of anything about Set A, which remains spent",
            "not a claim that Set B is viable; the cluster count that would "
            "bound it is unavailable and the artifact says so",
        ],
        "why_it_exists": "the probe that found this pool was print-only, so the "
                         "pool size is quoted in AGENTS.md and in a commit "
                         "message and stands behind no registered artifact. A "
                         "set frozen against an unpinned inventory is not "
                         "auditable, and the order that makes Set B worth "
                         "building is inventory, freeze, hash, preregister, then "
                         "read once",
        "cutoff": CUTOFF,
        "method": EM,
        "apo_and_holo_test": "a chain counts as apo-capable when the deposit "
                             "assigns it no ligand and holo-capable when it "
                             "assigns it one. That is looser than the builder's "
                             "test, which additionally requires a CryptoBench-"
                             "accepted ligand, heavy-atom contact within 4.5 A "
                             "and a pocket movement of at least 2.5 A. Each of "
                             "those only removes candidates, so every count here "
                             "is an upper bound",
        "by_resolution_ceiling": rows,
        "reconstruction_method_mix": mix,
    }
    SETB_POOL.parent.mkdir(parents=True, exist_ok=True)
    SETB_POOL.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")
    for r in rows:
        print(f"EM at {r['ceiling']:.1f} A  entries {r['n_entries']:,} "
              f"(from {r['entry_count_source']})  chains {r['n_chains']:,}  "
              f"accessions {r['n_accessions']:,}")
        print(f"    apo+holo {r['n_with_apo_and_holo']:,}  "
              f"in CryptoBench {r['n_in_cryptobench']:,}  "
              f"pool {r['n_pool']:,}")
        if r["cluster_count_is_absent_not_zero"]:
            print(f"    clusters new to Set A: NOT MEASURED "
                  f"({r['n_unmapped_to_uniref50']:,} of {r['n_pool']:,} have no "
                  f"cached UniRef50 mapping)")
        else:
            print(f"    clusters new to Set A: "
                  f"{r['n_clusters_new_to_set_a']:,}")
    print(f"\nwrote {SETB_POOL.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true",
                    help="emit results/external/SETB_POOL.json")
    ap.add_argument("--from-cache", action="store_true",
                    help="compute from the cached describe without a network "
                         "call, recording the cache digest")
    ap.add_argument("--methods", action="store_true",
                    help="also count the pool by reconstruction method, which "
                         "needs the network and records the date it was taken")
    a = ap.parse_args()
    if a.write:
        if not a.from_cache:
            raise SystemExit(
                "--write currently requires --from-cache; the network path "
                "reverifies the entry count and has not been wired to the "
                "writer, and an artifact that mixes the two provenances without "
                "saying which is worse than one that says cache")
        return _write_pool((2.5, 3.0), methods=a.methods)

    print(f"protein entries released after {CUTOFF}\n")
    print(f"{'ceiling':>9}  {'EM':>10}  {'X-ray':>10}")
    for ceiling in LADDER:
        em = _count(EM, ceiling)
        xr = _count("X-ray", ceiling)
        print(f"{ceiling:>8.1f}A  {em:>10,}  {xr:>10,}")
    print(f"{'any':>9}  {_count(EM, None):>10,}  {_count('X-ray', None):>10,}")
    print("\nSet A used the 2.5 A X-ray row: 17,332 entries -> 33,725 chains -> "
          "4,858 accessions -> 561 clusters with an apo and a holo chain -> "
          "57 units with a cryptic pocket.")
    print("The yield from entries to units was 57/17,332, about 1 in 300, but "
          "extrapolating that ratio to EM would be guessing: cryo-EM is used for "
          "large assemblies rather than ligand series, so its entries concentrate "
          "on few proteins and an accession has to carry both an apo and a holo "
          "chain. So the pairing is counted rather than assumed.\n")

    for ceiling in (2.5, 3.0):
        p = _pairable(ceiling)
        print(f"EM at {p['ceiling']:.1f} A")
        print(f"  {p['n_entries']:>6,} entries -> {p['n_chains']:,} protein "
              f"chains -> {p['n_accessions']:,} accessions")
        print(f"  {p['n_with_apo_and_holo']:>6,} accessions carry a ligand-free "
              f"and a ligand-bearing chain")
        print(f"  {p['n_in_cryptobench']:>6,} of those are in CryptoBench itself")
        pool = p["n_with_apo_and_holo"] - p["n_in_cryptobench"]
        print(f"  {pool:>6,} of those are outside CryptoBench  <-- the pool "
              f"Set B would be drawn from")
        print(f"  {p['n_unmapped_to_uniref50']:>6,} of the pool have no UniRef50 "
              f"mapping cached")
        if p["n_unmapped_to_uniref50"] == pool:
            # The cached mapping was fetched for Set A's X-ray accessions and
            # covers none of these, so the cluster count below is not a small
            # number, it is an absent one. Printing it as though it were a
            # measurement would read as "this axis is empty" when nothing about
            # emptiness has been established.
            print(f"  {'?':>6}  clusters new to Set A: NOT MEASURED. The cached "
                  f"mapping covers Set A's X-ray accessions and none of these, "
                  f"so run tools/external_uniref.py over the pool before "
                  f"reading anything into this")
        else:
            print(f"  {p['n_clusters_new_to_set_a']:>6,} clusters remain that "
                  f"Set A has not spent  <-- the ceiling on Set B")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
