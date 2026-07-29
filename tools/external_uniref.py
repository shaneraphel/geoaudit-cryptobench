"""UniRef50 clusters for every accession on both sides of the external boundary.

The temporal cutoff already guarantees that no external structure reached
CryptoBench. It does not guarantee that no *homologue* of an external protein was
in CryptoBench's training fold, and a model that learned a fold from a 2019
homologue is not being tested on something new. So externality needs a sequence
criterion as well as a date, and this is the part that cannot be established from
the PDB metadata alone.

No aligner is installed here and installing one to cluster two thousand sequences
would put an unpinned binary between the data and the labels. UniRef50 is a
published clustering of UniProtKB at 50% identity and 80% overlap, maintained by
UniProt, addressable by accession. Using it means the homology criterion is
someone else's frozen artefact rather than a threshold chosen here, which is the
weaker dependency of the two.

Reads no fold labels. Accessions are read from CryptoBench's dataset and test
files, but only as names to exclude; no residue, pocket or label is touched.
"""
from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from pocket_bench.paths import ROOT

INVENTORY = ROOT / "data/external/INVENTORY.json.gz"
CB_DATASET = ROOT / "data/cryptobench_apo/_osf/dataset.json"
CB_TEST = ROOT / "data/cryptobench_apo/_osf/test.json"
OUT = ROOT / "data/external/UNIREF50.json"

RUN = "https://rest.uniprot.org/idmapping/run"
STATUS = "https://rest.uniprot.org/idmapping/status/"
RESULTS = "https://rest.uniprot.org/idmapping/results/"
BATCH = 500


def _get(url: str, tries: int = 5) -> dict:
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())
        except Exception:                             # noqa: BLE001 - transient
            if attempt == tries - 1:
                raise
            time.sleep(3.0 * (attempt + 1))
    raise AssertionError("unreachable")


def _get_with_link(url: str, tries: int = 6) -> tuple[dict, str]:
    """A paged read that also hands back the Link header it needs to continue.

    Retried like every other call here. A truncated page would otherwise drop
    accessions from the cluster map, and an accession missing from the map is
    treated as unclusterable and excluded, so the failure would quietly shrink
    the external set instead of announcing itself.
    """
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"Accept": "application/json"}),
                    timeout=180) as r:
                return json.loads(r.read()), r.headers.get("Link", "")
        except Exception:                             # noqa: BLE001 - transient
            if attempt == tries - 1:
                raise
            time.sleep(3.0 * (attempt + 1))
    raise AssertionError("unreachable")


def _map(accessions: list[str]) -> dict[str, str]:
    """One id-mapping job, followed until it finishes, paged to the end."""
    req = urllib.request.Request(RUN, data=urllib.parse.urlencode(
        {"from": "UniProtKB_AC-ID", "to": "UniRef50",
         "ids": ",".join(accessions)}).encode())
    with urllib.request.urlopen(req, timeout=120) as r:
        jid = json.loads(r.read())["jobId"]
    for _ in range(120):
        st = _get(STATUS + jid)
        if st.get("jobStatus") == "RUNNING":
            time.sleep(2.0)
            continue
        break
    out: dict[str, str] = {}
    url = f"{RESULTS}{jid}?size=500"
    while url:
        body, link = _get_with_link(url)
        for row in body.get("results", []):
            out[row["from"]] = row["to"]
        url = (link.split(";")[0].strip("<> ")
               if link and 'rel="next"' in link else None)
    return out


def cryptobench_accessions() -> set[str]:
    """Every accession CryptoBench names, across its dataset and its test fold."""
    acc: set[str] = set()

    def walk(o: object) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "uniprot_id" and isinstance(v, str):
                    acc.add(v.strip())
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for f in (CB_DATASET, CB_TEST):
        walk(json.loads(f.read_text()))
    return {a for a in acc if a}


def main() -> int:
    if OUT.is_file() and "--force" not in sys.argv:
        d = json.loads(OUT.read_text())
        print(f"cached {OUT.relative_to(ROOT)}: "
              f"{len(d['cryptobench'])} CryptoBench accessions in "
              f"{len(set(d['cryptobench'].values()))} clusters, "
              f"{len(d['candidate'])} candidate accessions in "
              f"{len(set(d['candidate'].values()))}")
        return 0
    inv = json.loads(gzip.decompress(INVENTORY.read_bytes()))
    cand = sorted({c["uniprot"] for c in inv["chains"]})
    cb = sorted(cryptobench_accessions())
    print(f"mapping {len(cb)} CryptoBench and {len(cand)} candidate accessions "
          f"to UniRef50", flush=True)
    both: dict[str, dict[str, str]] = {}
    for name, ids in (("cryptobench", cb), ("candidate", cand)):
        got: dict[str, str] = {}
        for i in range(0, len(ids), BATCH):
            got.update(_map(ids[i:i + BATCH]))
            print(f"  {name}: {len(got)}/{len(ids)}", flush=True)
        both[name] = got
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.external_uniref50.v1",
        "source": "UniProt id-mapping, UniProtKB_AC-ID to UniRef50",
        "identity": 0.5,
        "reads_any_fold_labels": False,
        "why_accessions_from_the_test_fold_appear_here": (
            "they are excluded, not evaluated; a name is read and no residue, "
            "pocket or label is"),
        "unmapped_are_kept_out": (
            "an accession UniProt cannot cluster cannot be shown to be unrelated "
            "to CryptoBench, so it is dropped rather than assumed novel"),
        **both,
    }, indent=1) + "\n")
    cb_clusters = set(both["cryptobench"].values())
    hit = sum(1 for a, c in both["candidate"].items() if c in cb_clusters)
    print(f"wrote {OUT.relative_to(ROOT)}: {len(cb_clusters)} CryptoBench "
          f"clusters; {hit} of {len(both['candidate'])} candidate accessions "
          f"land in one of them")
    print(f"  unmapped: {len(cb) - len(both['cryptobench'])} CryptoBench, "
          f"{len(cand) - len(both['candidate'])} candidate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
