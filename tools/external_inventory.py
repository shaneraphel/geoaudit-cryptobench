"""Stage one of the external validation set: what the PDB has that CryptoBench could not.

CryptoBench's newest structure was released 2024-05-08. An apo-holo pair in which
both structures were released after that date cannot have reached the benchmark,
its splits, or any model trained on it, whatever the sequence similarity turns out
to be. That temporal boundary is the only part of externality that needs no
argument, so it is the part this stage rests on.

This stage downloads no structures and computes no labels. It builds an inventory
of entries, their chains, their UniProt accessions and their ligand content, then
groups it by accession so that the next stage has candidate pairs to look at. It
is separated out because it is the slow, network-bound, cacheable part, and
because keeping discovery apart from labelling makes it possible to state exactly
what was known before any label existed.

Reads no fold labels of any kind.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pocket_bench.paths import ROOT

OUT = ROOT / "data/external/INVENTORY.json.gz"

# The last day CryptoBench could have seen a structure. Established by taking the
# maximum initial release date over every entry it names, apo and holo alike.
CRYPTOBENCH_CUTOFF = "2024-05-08"

SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
GRAPHQL = "https://data.rcsb.org/graphql"

# X-ray only, because the cryptic filter is a distance between two crystal
# structures and comparing it across methods means comparing like with like. The
# resolution ceiling is CryptoBench's own.
MAX_RESOLUTION = 2.5
BATCH = 150
WORKERS = 6

_QUERY = """
query($ids:[String!]!){
  entries(entry_ids:$ids){
    rcsb_id
    rcsb_accession_info{initial_release_date}
    rcsb_entry_info{resolution_combined polymer_entity_count_protein}
    polymer_entities{
      rcsb_polymer_entity_container_identifiers{auth_asym_ids}
      uniprots{rcsb_id}
      entity_poly{rcsb_sample_sequence_length rcsb_entity_polymer_type}
    }
    nonpolymer_entities{
      nonpolymer_comp{chem_comp{id}}
      rcsb_nonpolymer_entity_container_identifiers{auth_asym_ids}
    }
  }
}"""


def _post(url: str, payload: dict, tries: int = 5) -> dict:
    """One request, retried, because a transient failure here silently shrinks
    the candidate pool and a smaller pool is not visibly wrong."""
    body = json.dumps(payload).encode()
    for attempt in range(tries):
        try:
            req = urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception as exc:                      # noqa: BLE001 - all transient
            if attempt == tries - 1:
                raise
            time.sleep(2.0 * (attempt + 1))
            _ = exc
    raise AssertionError("unreachable")


def released_after(cutoff: str = CRYPTOBENCH_CUTOFF) -> list[str]:
    """Every X-ray protein entry the PDB released after CryptoBench's last one."""
    q = {
        "query": {"type": "group", "logical_operator": "and", "nodes": [
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_accession_info.initial_release_date",
                "operator": "greater", "value": cutoff}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                "operator": "greater_or_equal", "value": 1}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.experimental_method",
                "operator": "exact_match", "value": "X-ray"}},
            {"type": "terminal", "service": "text", "parameters": {
                "attribute": "rcsb_entry_info.resolution_combined",
                "operator": "less_or_equal", "value": MAX_RESOLUTION}},
        ]},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": 10000},
                            "results_content_type": ["experimental"]},
    }
    ids: list[str] = []
    while True:
        q["request_options"]["paginate"]["start"] = len(ids)
        got = _post(SEARCH, q)
        rows = [h["identifier"] for h in got.get("result_set", [])]
        ids.extend(rows)
        total = got.get("total_count", 0)
        print(f"  search: {len(ids)}/{total}", flush=True)
        if not rows or len(ids) >= total:
            return ids


def _describe(ids: list[str]) -> list[dict]:
    got = _post(GRAPHQL, {"query": _QUERY, "variables": {"ids": ids}})
    return [e for e in (got.get("data") or {}).get("entries") or [] if e]


def describe_all(ids: list[str]) -> list[dict]:
    """Chains, accessions and ligand content for every entry, in batches."""
    chunks = [ids[i:i + BATCH] for i in range(0, len(ids), BATCH)]
    out: list[dict] = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for i, part in enumerate(pool.map(_describe, chunks), start=1):
            out.extend(part)
            if i % 10 == 0 or i == len(chunks):
                print(f"  describe: {i}/{len(chunks)} batches, "
                      f"{len(out)} entries", flush=True)
    return out


def flatten(entries: list[dict]) -> list[dict]:
    """One record per protein chain, carrying the entry's ligand inventory.

    Ligands are kept per auth chain where the deposition says so, because a
    ligand bound to a different chain of the same entry does not make this chain
    holo. The contact test in the next stage is what settles it, but carrying the
    chain here keeps obviously irrelevant pairs out of the download queue.
    """
    rows: list[dict] = []
    for e in entries:
        res = (e.get("rcsb_entry_info") or {}).get("resolution_combined") or []
        date = ((e.get("rcsb_accession_info") or {})
                .get("initial_release_date") or "")[:10]
        ligands = []
        for ne in e.get("nonpolymer_entities") or []:
            comp = ((ne.get("nonpolymer_comp") or {}).get("chem_comp") or {})
            code = comp.get("id")
            if not code:
                continue
            chains = ((ne.get("rcsb_nonpolymer_entity_container_identifiers")
                       or {}).get("auth_asym_ids") or [])
            ligands.append({"code": code, "chains": sorted(chains)})
        for pe in e.get("polymer_entities") or []:
            poly = pe.get("entity_poly") or {}
            if (poly.get("rcsb_entity_polymer_type") or "") != "Protein":
                continue
            accs = sorted(u["rcsb_id"] for u in (pe.get("uniprots") or [])
                          if u.get("rcsb_id"))
            if len(accs) != 1:
                # No accession means nothing to pair on; several means a chimera,
                # where "the same protein" is not a well-defined claim.
                continue
            chains = ((pe.get("rcsb_polymer_entity_container_identifiers")
                       or {}).get("auth_asym_ids") or [])
            for ch in sorted(chains):
                rows.append({
                    "pdb": e["rcsb_id"].lower(), "chain": ch,
                    "uniprot": accs[0], "released": date,
                    "resolution": min(res) if res else None,
                    "length": poly.get("rcsb_sample_sequence_length"),
                    "ligands": ligands,
                })
    return rows


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.is_file() and "--force" not in sys.argv:
        d = json.loads(gzip.decompress(OUT.read_bytes()))
        print(f"cached {OUT.relative_to(ROOT)}: {len(d['chains'])} chains from "
              f"{d['n_entries']} entries released after {d['cutoff']}")
        return 0
    print(f"entries released after {CRYPTOBENCH_CUTOFF}, X-ray, "
          f"<= {MAX_RESOLUTION} A:", flush=True)
    ids = released_after()
    entries = describe_all(ids)
    chains = flatten(entries)
    payload = {
        "schema": "geoaudit.external_inventory.v1",
        "cutoff": CRYPTOBENCH_CUTOFF,
        "max_resolution": MAX_RESOLUTION,
        "reads_any_fold_labels": False,
        "n_entries_searched": len(ids),
        "n_entries": len(entries),
        "chains": chains,
    }
    OUT.write_bytes(gzip.compress(json.dumps(payload).encode()))
    n_acc = len({c["uniprot"] for c in chains})
    print(f"wrote {OUT.relative_to(ROOT)}: {len(chains)} protein chains, "
          f"{n_acc} distinct accessions, from {len(entries)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
