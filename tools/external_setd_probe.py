#!/usr/bin/env python3
"""Is a fourth external set buildable from X-ray structures Set A could not use?

Why ask
-------
The strongest training-fold result in this repository --- the four-family wire
stack at +0.0121 on 12 of 12 cluster-disjoint halvings --- has never been read on
held-out data, and a reviewer put that first. The frozen sets available to confirm
it are cryo-EM: Set B has 8 units and Set C has 38, and
``CRYOEM_LABEL_SENSITIVITY.json`` records that the recovered label rule declines
17.3 % of Set B's pairs and 31.9 % of Set C's against 9.2 % for X-ray Set A, so
resolution is a covariate of the label on both.

An X-ray set would carry no such covariate. Set A took X-ray entries released
after CryptoBench's newest at **2.5 Å or better** and used essentially every
UniRef50 cluster that offered an apo and a holo chain together, so that band is
spent. The band immediately below it is not: **X-ray at 2.5--3.0 Å has never been
inventoried here.** Whether it holds enough clusters to matter is a counting
question, and this tool counts rather than extrapolates.

That distinction is the reason this file exists at all. ``AGENTS.md`` records an
estimate of a second set's viability that was wrong by two orders of magnitude
because it multiplied a pool size by a yield measured on a different population.
The yield of cryptic units per free cluster is not transferable between resolution
bands --- a 2.9 Å structure has larger coordinate error, which moves the pocket
RMSD the label is cut on --- so this reports the pool and the free-cluster count
and stops. It does not predict a unit count.

Two harness checks, because a zero here would be believed
----------------------------------------------------------
``AGENTS.md``'s first rule is that a query returning zero has not told you there
is nothing there: ``rcsb_entry_info.experimental_method`` takes the value
``"EM"``, and spelling it ``"Electron Microscopy"`` returns ``total_count: 0``
with no error. So before the new band is queried, the tool re-runs the band Set A
used and requires it to return the entry count Set A's own inventory recorded. If
that fails, the report is that the harness is broken, not that the band is empty.

The second check is pagination. The same ``AGENTS.md`` section records ninety
minutes lost to a probe that asked for 20,000 rows where the working query asks
for 10,000: RCSB does not reject the larger page, it stops responding. This tool
uses the same page size as ``external_inventory.py`` and says so here so that a
future edit has to argue with the comment.

What this does not do
----------------------
It builds nothing and freezes nothing. It reads no label, no prediction and no
unit of any external set. Building Set D is a separate step that must come before
any preregistration, and reading it must come after --- the order in ``AGENTS.md``
is build, freeze, hash, preregister, read once, and this tool is upstream of all
five.

It also does not touch ``external_inventory.py``. That file feeds the frozen Set A
and ``AGENTS.md`` forbids editing a tool that feeds a frozen artifact for an
unrelated reason; the query is duplicated here instead, which is the same
judgement that put a retry-count fix in a probe rather than in the builder.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR / "src"), str(ROOT_DIR / "tools")]

from pocket_bench.paths import ROOT                                # noqa: E402

SCHEMA = "geoaudit.setd_probe.v1"
SEARCH = "https://search.rcsb.org/rcsbsearch/v2/query"
OUT = ROOT / "results/external/SETD_POOL.json"

# Identical to external_inventory.py, and identical on purpose: a different
# cutoff would make Set D's externality a different claim from Set A's.
CUTOFF = "2024-05-08"

# The band Set A used, re-queried as a harness check.
SET_A_CEILING = 2.5
# The band Set A could not use. 3.0 A is where this repository's own cryo-EM
# Set C stopped, so the two coarse sets are cut at the same place and a
# resolution effect is comparable across them.
SET_D_FLOOR = 2.5
SET_D_CEILING = 3.0

# 10000, because 20000 makes RCSB stop responding rather than refuse. See the
# module docstring; this is not a tuning parameter.
PAGE = 10000

INVENTORY_A = ROOT / "data/external/INVENTORY.json.gz"


def _post(url: str, payload: dict, tries: int = 6) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"})
    last: Exception | None = None
    for k in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=180) as fh:
                return json.loads(fh.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code == 204:      # no hits is a 204, not an error
                return {"total_count": 0, "result_set": []}
            last = exc
        except Exception as exc:     # noqa: BLE001
            last = exc
        time.sleep(1.5 * (k + 1))
    raise SystemExit(f"RCSB did not answer after {tries} tries: {last}")


def _query(lo: float | None, hi: float) -> dict:
    nodes = [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_accession_info.initial_release_date",
            "operator": "greater", "value": CUTOFF}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.polymer_entity_count_protein",
            "operator": "greater_or_equal", "value": 1}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.experimental_method",
            "operator": "exact_match", "value": "X-ray"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "less_or_equal", "value": hi}},
    ]
    if lo is not None:
        nodes.append({"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_entry_info.resolution_combined",
            "operator": "greater", "value": lo}})
    return {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": PAGE},
                            "results_content_type": ["experimental"]},
    }


def _ids(lo: float | None, hi: float, label: str) -> list[str]:
    q = _query(lo, hi)
    ids: list[str] = []
    while True:
        q["request_options"]["paginate"]["start"] = len(ids)
        got = _post(SEARCH, q)
        rows = [h["identifier"] for h in got.get("result_set", [])]
        ids.extend(rows)
        total = got.get("total_count", 0)
        print(f"  {label}: {len(ids)}/{total}", flush=True)
        if not rows or len(ids) >= total:
            return ids


def _harness_check() -> dict:
    """Re-query Set A's band and require it to match Set A's own inventory."""
    if not INVENTORY_A.is_file():
        raise SystemExit(
            f"{INVENTORY_A.relative_to(ROOT)} is absent, so the harness cannot "
            f"be checked against a count known to be right. Refusing to report "
            f"a pool size that could be a silent zero")
    with gzip.open(INVENTORY_A, "rt") as fh:
        inv = json.load(fh)
    want = inv.get("n_entries") or inv.get("entries") or 0
    if isinstance(want, list):
        want = len(want)
    got = _ids(None, SET_A_CEILING, f"harness, <= {SET_A_CEILING} A")
    ok = len(got) == want
    return {
        "band": f"<= {SET_A_CEILING} A",
        "entries_now": len(got),
        "entries_in_set_a_inventory": want,
        "agrees": ok,
        "why_this_runs_first": (
            "rcsb_entry_info.experimental_method takes the value 'EM', and "
            "spelling it 'Electron Microscopy' returns total_count 0 with no "
            "error. A zero from the new band would be believed unless a band "
            "with a known answer had been asked first"),
        "why_a_small_disagreement_is_not_a_failure": (
            "the PDB releases weekly, so a re-query after the inventory was "
            "cached returns more entries. Fewer would mean the query changed "
            "meaning, which is the failure this checks for"),
    }


def build(write: bool) -> int:
    print(f"harness check: re-running the band Set A used\n")
    harness = _harness_check()
    print(f"  now {harness['entries_now']}, "
          f"Set A's inventory {harness['entries_in_set_a_inventory']}")
    if harness["entries_now"] < harness["entries_in_set_a_inventory"]:
        raise SystemExit(
            f"the band Set A used now returns fewer entries "
            f"({harness['entries_now']}) than Set A's own inventory recorded "
            f"({harness['entries_in_set_a_inventory']}). The query has changed "
            f"meaning; nothing below would be trustworthy")

    print(f"\nthe band Set A could not use: "
          f"{SET_D_FLOOR}--{SET_D_CEILING} A, X-ray, after {CUTOFF}\n")
    ids = _ids(SET_D_FLOOR, SET_D_CEILING, f"{SET_D_FLOOR}-{SET_D_CEILING} A")

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "is_a_frozen_set": False,
        "what_this_is": (
            "a count of the X-ray entries a fourth external set could be drawn "
            "from: the resolution band immediately below the one Set A used, "
            "which has never been inventoried here"),
        "what_this_is_not": [
            "a set. Nothing is selected, labelled, frozen or hashed here",
            "a predicted unit count. The yield of cryptic units per free "
            "cluster is not transferable between resolution bands, because a "
            "2.9 A structure carries larger coordinate error and the label is "
            "cut on a pocket RMSD",
            "a read. No label, prediction or unit of any external set is opened",
        ],
        "why_x_ray_rather_than_more_cryo_em": (
            "CRYOEM_LABEL_SENSITIVITY.json records the recovered rule declining "
            "9.2 % of pairs for X-ray Set A, 17.3 % for Set B and 31.9 % for "
            "Set C, with the cryptic share rising alongside. Resolution is a "
            "covariate of the label on the cryo-EM sets and is not on an X-ray "
            "one"),
        "cutoff": CUTOFF,
        "why_the_same_cutoff": (
            "identical to external_inventory.py's. A different cutoff would "
            "make Set D's externality a different claim from Set A's"),
        "harness_check": harness,
        "band": {"floor_exclusive": SET_D_FLOOR, "ceiling_inclusive":
                 SET_D_CEILING},
        "n_entries_in_the_band": len(ids),
        "entry_ids": ids,
        "page_size": PAGE,
        "why_the_page_size_is_not_a_knob": (
            "20000 rows makes RCSB stop responding rather than refuse, and "
            "AGENTS.md records ninety minutes spent on retry logic and TLS "
            "diagnostics before the two requests were compared"),
        "what_has_to_happen_next_and_in_this_order": [
            "describe the entries and flatten to chains, as Set A's builder does",
            "map accessions to UniRef50 and remove every cluster shared with "
            "CryptoBench, Set A, Set B or Set C",
            "select one unit per free cluster and label with Set A's recovered "
            "rule, imported unedited so a read is comparable",
            "freeze and hash the set",
            "preregister the read, with the losing sentences written",
            "read once",
        ],
    }

    print(f"\n{len(ids)} X-ray entries in "
          f"{SET_D_FLOOR}-{SET_D_CEILING} A released after {CUTOFF}")
    print(f"  for scale: Set A drew 57 cryptic units from "
          f"{harness['entries_in_set_a_inventory']} entries at "
          f"<= {SET_A_CEILING} A")
    print("  no unit count is predicted from this; the yield does not transfer "
          "across resolution bands")

    if write:
        OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    return build(ap.parse_args(argv).write)


if __name__ == "__main__":
    raise SystemExit(main())
