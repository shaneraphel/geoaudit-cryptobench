#!/usr/bin/env python3
"""The ninth read: a ranked candidate-site list for the counting field, scored.

Runs the plan in results/architecture_sweep/PREREGISTERED_POCKETS.json and refuses
to start until it is committed, clean and an ancestor of HEAD.

The field is given a pocket stage here for the first time. Its residues above the
shipped operating point are clustered by single linkage at the method's own pinch
radius, clusters are ranked by the sum of their members' scores, and each is given
a score-weighted centre. P2Rank's candidates are its own committed centres in its
own order, untouched.

Both lists are then scored the same way. There is no ligand in an apo structure, so
the target is the labelled cryptic residue set and the distance is to its nearest
heavy atom; that quantity is called ``distance_to_labelled_site`` and never DCA,
because DCA is measured to a ligand and the two must not be compared.

Nothing here rescores a residue. The per-residue scores were frozen at read one and
this reads them from the committed prediction archive, which is also what lets the
whole read be recomputed by a third party without running either detector.

Usage: PYTHONPATH=src:tools python3.12 tools/pocket_read.py [--check]
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess

import numpy as np

from pocket_bench.paths import ROOT

PLAN = ROOT / "results/architecture_sweep/PREREGISTERED_POCKETS.json"
PRED = ROOT / "results/cryptobench_official/predictions"
MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
OUT = ROOT / "results/official_fold/POCKET_READ.json"

SCHEMA = "geoaudit.pocket_read.v1"
READ_INDEX = 9
METHOD = "table_field"
BASELINE = "p2rank"


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _plan_provenance() -> dict:
    rel = str(PLAN.relative_to(ROOT))
    if _git("rev-parse", "--is-shallow-repository") == "true":
        raise SystemExit(
            "this is a shallow clone, so the commit that fixed the pocket rules "
            "may not be present. Fetch the full history and retry.")
    if _git("status", "--porcelain", "--", rel):
        raise SystemExit(f"{rel} is modified or untracked; this read is refused")
    sha = _git("log", "-1", "--format=%H", "--", rel)
    if not sha:
        raise SystemExit(f"{rel} has no commit; this read is refused")
    if subprocess.run(["git", "merge-base", "--is-ancestor", sha,
                       _git("rev-parse", "HEAD")], cwd=ROOT).returncode != 0:
        raise SystemExit(f"{sha[:12]} is not an ancestor of HEAD")
    return {"artifact": rel, "committed_in": sha,
            "committed_at": _git("log", "-1", "--format=%cI", sha),
            "subject": _git("log", "-1", "--format=%s", sha),
            "is_ancestor_of_head": True}


def _residue_atoms(path) -> dict[int, np.ndarray]:
    """Heavy-atom coordinates per resseq, from the receptor as it was scored."""
    per: dict[int, list[list[float]]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[76:78].strip() == "H" or line[12:16].strip().startswith("H"):
            continue
        try:
            r = int(line[22:26])
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError:
            continue
        per.setdefault(r, []).append(xyz)
    return {r: np.asarray(v, dtype=np.float64) for r, v in per.items()}


def _single_linkage(pts: np.ndarray, cutoff: float) -> np.ndarray:
    """Cluster labels under single linkage at ``cutoff``, by union-find.

    scipy would do this in one call, but the pairwise matrix it wants is the one
    thing worth avoiding: these sets are small enough that an explicit union
    over the pairs within the cutoff is both cheaper and easier to check against
    the definition of single linkage, which is what the plan specifies.
    """
    n = len(pts)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    d2 = cutoff * cutoff
    for i in range(n):
        di = pts - pts[i]
        close = np.nonzero((di * di).sum(axis=1) <= d2)[0]
        for j in close:
            if j <= i:
                continue
            ra, rb = find(i), find(int(j))
            if ra != rb:
                parent[rb] = ra
    return np.array([find(i) for i in range(n)])


def _our_pockets(scores: dict, positives: list, atoms: dict,
                 cutoff: float) -> list[dict]:
    """Rank candidate sites from the field's own positive residues."""
    res = [int(r) for r in positives if int(r) in atoms]
    if not res:
        return []
    ctr = np.array([atoms[r].mean(axis=0) for r in res])
    w = np.array([float(scores[str(r)]) for r in res])
    lab = _single_linkage(ctr, cutoff)

    out = []
    for c in sorted(set(lab.tolist())):
        m = lab == c
        wm = w[m]
        # Scores here are unbounded and can be negative once a chain's mean is
        # subtracted, so the weighting is shifted to be non-negative within the
        # cluster before it is used as a weight. A negative weight would place a
        # centre outside the cluster it belongs to.
        ww = wm - wm.min() + 1.0
        out.append({
            "members": [res[i] for i in np.nonzero(m)[0]],
            "score_sum": float(wm.sum()),
            "score_mean": float(wm.mean()),
            "centre": (ctr[m] * ww[:, None]).sum(axis=0) / ww.sum(),
        })
    out.sort(key=lambda p: -p["score_sum"])
    for i, p in enumerate(out, 1):
        p["rank"] = i
    return out


P2RANK_RAW = ROOT / "results/cryptobench_official/p2rank_raw"


def _p2rank_pocket_residues(uid: str) -> list[list[int]]:
    """The residue list P2Rank assigns to each of its pockets, in its own order.

    Read from the archived ``*_predictions.csv`` rather than from the summarised
    prediction JSON, which keeps only centres. This exists to remove a confound
    that would otherwise decide the comparison: our candidate centre is the
    centroid of residues we predict, while P2Rank's is the centre of a geometric
    cavity, which by construction sits in the empty space a ligand would occupy
    and therefore *away* from any residue atom. Scoring both by distance to a
    labelled residue atom would penalise P2Rank for correctly locating a cavity.
    Given its residue lists, both methods can be scored on the same kind of
    object.
    """
    csv = P2RANK_RAW / uid / "rec.pdb_predictions.csv"
    if not csv.is_file():
        return []
    lines = csv.read_text().splitlines()
    if len(lines) < 2:
        return []
    head = [h.strip() for h in lines[0].split(",")]
    try:
        col = head.index("residue_ids")
    except ValueError:
        return []
    out = []
    for line in lines[1:]:
        if not line.strip():
            continue
        # residue_ids is a space-separated field inside a comma-separated file,
        # so a plain split on commas is enough as long as the column index holds.
        parts = line.split(",")
        if len(parts) <= col:
            out.append([])
            continue
        ids = []
        for tok in parts[col].split():
            tok = tok.strip()
            if not tok:
                continue
            # Tokens are {chain}_{resseq}; the universe keys on the integer.
            num = tok.rsplit("_", 1)[-1]
            try:
                ids.append(int(num))
            except ValueError:
                continue
        out.append(ids)
    return out


def _their_pockets(unit: dict, uid: str,
                   atoms: dict[int, np.ndarray]) -> list[dict]:
    """P2Rank's own candidates, in its own order, with both kinds of centre.

    ``centre`` is P2Rank's committed cavity centre, unmodified. ``centre_res`` is
    the centroid of the residues P2Rank itself assigned to that pocket, which is
    the same kind of object our own candidate centre is.
    """
    res_lists = _p2rank_pocket_residues(uid)
    out = []
    for i, p in enumerate(sorted(unit.get("pockets") or [],
                                 key=lambda p: p.get("rank", 10 ** 6))):
        c = p.get("center_xyz")
        if not c or all(abs(float(x)) < 1e-12 for x in c):
            continue
        members = [r for r in (res_lists[i] if i < len(res_lists) else [])
                   if r in atoms]
        centre_res = (np.array([atoms[r].mean(axis=0) for r in members]
                               ).mean(axis=0) if members else None)
        out.append({"rank": p.get("rank"), "score_sum": p.get("score"),
                    "centre": np.asarray([float(x) for x in c]),
                    "centre_res": centre_res,
                    "members": members or None})
    return out


def _min_distance(centre: np.ndarray, target: np.ndarray) -> float:
    d = target - centre
    return float(math.sqrt(float((d * d).sum(axis=1).min())))


def _paired_ci(a: list[float], b: list[float], n_boot: int, seed: int,
               level: float) -> dict:
    if not a:
        return {"n_paired": 0, "delta": None, "ci": [None, None],
                "excludes_zero": None}
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"n_paired": int(len(d)), "delta": round(float(d.mean()), 6),
            "ci": [round(float(lo), 6), round(float(hi), 6)],
            "excludes_zero": bool(lo > 0 or hi < 0)}


def _score_units(cutoff: float, plan: dict) -> dict:
    ours_all = json.loads((PRED / f"{METHOD}.json").read_text())["units"]
    theirs_all = json.loads((PRED / f"{BASELINE}.json").read_text())["units"]
    entries = json.loads(MANIFEST.read_text())["entries"]
    radii = plan["hit_radii_angstrom"]
    ks = plan["top_k"]

    rows, budget_mismatch = [], []
    n_no_res = 0
    for e in entries:
        uid = f"{e['pdb']}_{e['chain']}"
        atoms = _residue_atoms(ROOT / e["receptor_path"])
        lab = json.loads((ROOT / e["label_path"]).read_text())
        target = np.concatenate(
            [atoms[r] for r in lab["cryptic_residues"] if r in atoms])

        ou, tu = ours_all[uid], theirs_all[uid]
        pos = ou.get("residue_positive") or []
        # The plan promises the clustered set is the deployment rule's own
        # positive call, so the count is checked against the shipped q. The
        # expression is the scorer's own -- max(1, int(round(q*n))) -- rather
        # than a ceiling, which is what an earlier version of this guard assumed
        # and which disagreed on 104 of the 192 chains.
        q = plan["pocket_construction"]["step_1_residue_budget"]["q"]
        want = max(1, int(round(q * ou["n_universe"])))
        if len(pos) != want:
            budget_mismatch.append({"unit_id": uid, "have": len(pos),
                                    "want": want})

        ours = _our_pockets(ou["residue_scores"], pos, atoms, cutoff)
        theirs = _their_pockets(tu, uid, atoms)
        n_no_res += sum(1 for p in theirs if p.get("centre_res") is None)

        def _summary(pockets, centre_key="centre"):
            if not pockets:
                return {"n_candidates": 0, "top1_distance": None,
                        "hit": {str(r): {str(k): False for k in ks}
                                for r in radii},
                        "recall": {str(k): 0.0 for k in ks}}
            # A pocket with no residue list keeps its cavity centre rather than
            # dropping out, so the two P2Rank arms always compare the same
            # candidate set; n_p2rank_pockets_without_a_residue_list records how
            # often that fallback is used.
            def _c(p):
                v = p.get(centre_key)
                return p["centre"] if v is None else v

            dists = [_min_distance(_c(p), target) for p in pockets]
            hit = {str(r): {str(k): bool(min(dists[:k]) <= r) for k in ks}
                   for r in radii}
            # One geometric definition of coverage for both methods: a labelled
            # residue counts as covered when its centroid is within the linkage
            # cutoff of a top-K centre, using whichever centre this arm scores.
            recall = {}
            for k in ks:
                cov = 0
                for r in lab["cryptic_residues"]:
                    if r not in atoms:
                        continue
                    c = atoms[r].mean(axis=0)
                    if any(_min_distance(_c(p), c[None, :]) <= cutoff
                           for p in pockets[:k]):
                        cov += 1
                n = sum(1 for r in lab["cryptic_residues"] if r in atoms)
                recall[str(k)] = round(cov / n, 6) if n else 0.0
            return {"n_candidates": len(pockets),
                    "top1_distance": round(dists[0], 4),
                    "all_distances": [round(x, 4) for x in dists[:10]],
                    "hit": hit, "recall": recall}

        rows.append({"unit_id": uid, "n_labelled": len(lab["cryptic_residues"]),
                     "ours": _summary(ours),
                     "theirs": _summary(theirs),
                     "theirs_residue_centroid": _summary(theirs, "centre_res")})

    if budget_mismatch:
        raise SystemExit(
            f"{len(budget_mismatch)} chains' committed positive call is not the "
            f"shipped top-q budget the plan describes: {budget_mismatch[:3]}")
    return {"rows": rows, "radii": radii, "ks": ks,
            "n_p2rank_pockets_without_a_residue_list": n_no_res}


def build() -> dict:
    prov = _plan_provenance()
    plan = json.loads(PLAN.read_text())
    if plan["status_declared_in_advance"] != "exploratory":
        raise SystemExit("the plan no longer declares itself exploratory")

    st = plan["statistic"]
    cut = plan["pocket_construction"]["step_2_clustering"]
    lvl = 1.0 - plan["multiplicity"]["corrected_level"]

    cache: dict[float, dict] = {}

    def _arm(cutoff: float, tk: str = "theirs") -> dict:
        if cutoff not in cache:
            cache[cutoff] = _score_units(cutoff, plan)
        sc = cache[cutoff]
        rows = sc["rows"]
        both = [r for r in rows
                if r["ours"]["n_candidates"] and r[tk]["n_candidates"]]
        out = {
            "clustering_cutoff_angstrom": cutoff,
            "p2rank_centre_scored": (
                "its own cavity centre, as committed" if tk == "theirs"
                else "the centroid of the residues it assigned to the pocket"),
            "n_units": len(rows),
            "n_units_both_offer_a_candidate": len(both),
            "n_units_we_offer_none": sum(
                1 for r in rows if not r["ours"]["n_candidates"]),
            "n_units_p2rank_offers_none": sum(
                1 for r in rows if not r[tk]["n_candidates"]),
            "candidates_per_chain": {
                "ours_mean": round(float(np.mean(
                    [r["ours"]["n_candidates"] for r in rows])), 3),
                "ours_median": int(np.median(
                    [r["ours"]["n_candidates"] for r in rows])),
                "p2rank_mean": round(float(np.mean(
                    [r[tk]["n_candidates"] for r in rows])), 3),
                "p2rank_median": int(np.median(
                    [r[tk]["n_candidates"] for r in rows])),
            },
            "hit_rates": {},
            "recall": {},
        }
        for r in sc["radii"]:
            for k in sc["ks"]:
                a = [1.0 if x["ours"]["hit"][str(r)][str(k)] else 0.0
                     for x in both]
                b = [1.0 if x[tk]["hit"][str(r)][str(k)] else 0.0
                     for x in both]
                primary = (r == plan["primary_hit_radius_angstrom"])
                out["hit_rates"][f"{r}A/top{k}"] = {
                    "radius_angstrom": r, "k": k,
                    "is_a_corrected_test": primary,
                    "ours": round(float(np.mean(a)), 6),
                    "p2rank": round(float(np.mean(b)), 6),
                    "paired_95": _paired_ci(a, b, st["n_boot"], st["seed"],
                                            0.95),
                    "paired_bonferroni": (
                        _paired_ci(a, b, st["n_boot"], st["seed"], lvl)
                        if primary else None),
                }
        for k in sc["ks"]:
            a = [x["ours"]["recall"][str(k)] for x in both]
            b = [x[tk]["recall"][str(k)] for x in both]
            out["recall"][f"top{k}"] = {
                "ours": round(float(np.mean(a)), 6),
                "p2rank": round(float(np.mean(b)), 6),
                "paired_95": _paired_ci(a, b, st["n_boot"], st["seed"], 0.95),
            }
        d_ours = [r["ours"]["top1_distance"] for r in both]
        d_them = [r[tk]["top1_distance"] for r in both]
        out["top1_distance_to_labelled_site"] = {
            "ours_median": round(float(np.median(d_ours)), 4),
            "p2rank_median": round(float(np.median(d_them)), 4),
            "ours_mean": round(float(np.mean(d_ours)), 4),
            "p2rank_mean": round(float(np.mean(d_them)), 4),
            "paired_95": _paired_ci(d_ours, d_them, st["n_boot"], st["seed"],
                                    0.95),
            "sign_convention": "positive means our centre is further from the "
                               "labelled site than P2Rank's, so negative "
                               "favours us",
        }
        if tk == "theirs":
            out["per_unit"] = rows
        return out

    primary = _arm(cut["cutoff_angstrom"])
    sens = _arm(cut["sensitivity_cutoff_angstrom"])
    # The preregistered comparison scores P2Rank at its own cavity centre, which
    # was not a neutral choice: the target is the nearest heavy atom of a
    # labelled residue, and a cavity centre sits in the empty space a ligand
    # would occupy, so it is displaced from every residue atom by roughly a
    # pocket radius while our score-weighted residue centroid is not. The bias
    # runs in our favour and was noticed only after the read, so both are
    # reported and the claim is taken from the fairer one.
    fair = _arm(cut["cutoff_angstrom"], "theirs_residue_centroid")

    r0 = plan["primary_hit_radius_angstrom"]

    def _verdict(arm: dict) -> tuple[str, dict]:
        b = arm["hit_rates"][f"{r0}A/top1"]["paired_bonferroni"]
        if not b["excludes_zero"]:
            return "top1_unresolved", b
        return ("top1_favours_the_field" if b["delta"] > 0
                else "top1_favours_p2rank"), b

    # The plan's decision rule is scoped to the top-1 hit rate at the primary
    # radius, so that is what selects the outcome. The candidate shortfall is a
    # caveat on the larger K, not a competing verdict: an earlier version of this
    # let it override top-1, which would have buried the one result the read was
    # designed to produce.
    key_prereg, bon = _verdict(primary)
    key_fair_only, bon_fair = _verdict(fair)
    # The weaker of the two verdicts governs. "Weaker" means: if the fairer arm
    # fails to resolve, or reverses, the read does not get to keep the
    # preregistered arm's win.
    key = key_fair_only
    if key_prereg != key_fair_only and key == "top1_favours_the_field":
        key = "top1_unresolved"

    short = (primary["n_units_we_offer_none"] > 0
             or min(primary["candidates_per_chain"]["ours_median"],
                    primary["candidates_per_chain"]["p2rank_median"])
             < max(plan["top_k"]))
    extra = ([{"key": "the_field_offers_too_few_candidates",
               "sentence": plan["what_will_be_written_under_each_outcome"][
                   "the_field_offers_too_few_candidates"],
               "why_it_applies": (
                   f"the median chain has "
                   f"{primary['candidates_per_chain']['ours_median']} candidates "
                   f"from us and "
                   f"{primary['candidates_per_chain']['p2rank_median']} from "
                   f"P2Rank, both below the largest K compared "
                   f"({max(plan['top_k'])}), so a top-5 rate is mostly a top-3 "
                   f"rate for both and the two larger K are not independent of "
                   f"each other")}]
             if short else [])

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "is_official_mmseqs2_10pct_test_fold": True,
        "test_fold_read_index": READ_INDEX,
        # The ledger reports the size of each read from this field. The read
        # covers every chain in the fold; the paired comparisons run on the
        # subset where both methods offer a candidate, which the arms record.
        "n_units": primary["n_units"],
        "n_paired_units": primary["n_units_both_offer_a_candidate"],
        "status": plan["status_declared_in_advance"],
        "question": plan["question"],
        "rescored_anything": False,
        "why_it_is_indexed_anyway": (
            "no residue is rescored -- the per-residue scores come from the "
            "committed prediction archive read one froze -- but a ranked "
            "candidate list is a new quantity and comparing it against the "
            "labels draws a new inference from the fold"),
        "provenance_of_the_plan": prov,
        "method": METHOD,
        "baseline": BASELINE,
        "what_was_built_for_this_read": (
            "the pocket stage. Before this read the field had no candidate "
            "sites at all, only per-residue scores, and its pocket entries in "
            "the frozen predictions were placeholders at the origin"),
        "distance_convention": plan["why_the_distance_is_not_called_dca"],
        "primary": primary,
        "sensitivity_at_the_looser_cutoff": sens,
        "fairness_correction_p2rank_scored_at_its_own_residue_centroid": {
            "why": (
                "the preregistered arm scores P2Rank at its cavity centre, "
                "which sits in the empty space a ligand would occupy and is "
                "therefore displaced from every residue heavy atom by about a "
                "pocket radius, while our candidate centre is a centroid of "
                "residues. Against a target defined by residue atoms that bias "
                "runs in our favour. It was noticed after the read, so this arm "
                "is a correction rather than a plan, and the verdict is taken "
                "from whichever arm is less favourable to us"),
            "n_p2rank_pockets_without_a_residue_list": (
                cache[cut["cutoff_angstrom"]][
                    "n_p2rank_pockets_without_a_residue_list"]),
            "arm": fair,
            "verdict_of_this_arm": key_fair_only,
            "verdict_of_the_preregistered_arm": key_prereg,
            "the_two_arms_agree": key_prereg == key_fair_only,
        },
        "outcome_key": key,
        "outcome": plan["what_will_be_written_under_each_outcome"][key],
        "additional_preregistered_outcomes_that_apply": extra,
    }


def _report(d: dict) -> None:
    _report_arm(d, d["primary"], "preregistered: P2Rank at its cavity centre")
    f = d["fairness_correction_p2rank_scored_at_its_own_residue_centroid"]
    print()
    _report_arm(d, f["arm"], "corrected: P2Rank at its own residue centroid")
    print(f"\n  {d['outcome_key']}  (prereg arm said "
          f"{f['verdict_of_the_preregistered_arm']}, corrected arm said "
          f"{f['verdict_of_this_arm']})")
    for x in d.get("additional_preregistered_outcomes_that_apply") or []:
        print(f"  also: {x['key']}")


def _report_arm(d: dict, p: dict, title: str) -> None:
    print(f"read {d['test_fold_read_index']} ({d['status']}), cutoff "
          f"{p['clustering_cutoff_angstrom']} A -- {title}")
    c = p["candidates_per_chain"]
    print(f"  candidates per chain: ours median {c['ours_median']} "
          f"(mean {c['ours_mean']}), P2Rank median {c['p2rank_median']} "
          f"(mean {c['p2rank_mean']})")
    print(f"  {p['n_units_both_offer_a_candidate']} of {p['n_units']} chains "
          f"have a candidate from both; we offer none on "
          f"{p['n_units_we_offer_none']}, P2Rank on "
          f"{p['n_units_p2rank_offers_none']}")
    for name, h in p["hit_rates"].items():
        mark = "  *" if h["is_a_corrected_test"] else ""
        ci = h["paired_95"]
        print(f"  {name:<12s} ours {h['ours']:.3f}  P2Rank {h['p2rank']:.3f}  "
              f"delta {ci['delta']:+.4f} [{ci['ci'][0]:+.4f}, "
              f"{ci['ci'][1]:+.4f}]{mark}")
    t = p["top1_distance_to_labelled_site"]
    print(f"  top-1 distance to labelled site: ours median "
          f"{t['ours_median']:.2f} A, P2Rank {t['p2rank_median']:.2f} A, "
          f"paired {t['paired_95']['delta']:+.3f} "
          f"[{t['paired_95']['ci'][0]:+.3f}, {t['paired_95']['ci'][1]:+.3f}]")
    for k, r in p["recall"].items():
        print(f"  recall {k}: ours {r['ours']:.3f}, P2Rank {r['p2rank']:.3f}, "
              f"delta {r['paired_95']['delta']:+.4f}")


def check() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    plan = json.loads(PLAN.read_text())
    bad = []
    if d.get("schema") != SCHEMA:
        bad.append("unexpected schema")
    if d.get("test_fold_read_index") != READ_INDEX:
        bad.append(f"read index {d.get('test_fold_read_index')}")
    if d.get("status") != "exploratory":
        bad.append(f"the read reports itself {d.get('status')}")
    if d.get("outcome") != plan["what_will_be_written_under_each_outcome"].get(
            d.get("outcome_key")):
        bad.append("the stated outcome is not the sentence preregistered for "
                   "the key it reports")
    for x in d.get("additional_preregistered_outcomes_that_apply") or []:
        if x["sentence"] != plan[
                "what_will_be_written_under_each_outcome"].get(x["key"]):
            bad.append(f"the extra outcome {x['key']} is not the sentence "
                       f"preregistered for it")
    p = d.get("primary") or {}
    if p.get("clustering_cutoff_angstrom") != plan["pocket_construction"][
            "step_2_clustering"]["cutoff_angstrom"]:
        bad.append("the primary arm was not clustered at the preregistered "
                   "cutoff")
    fc = d.get("fairness_correction_p2rank_scored_at_its_own_residue_centroid")
    if not fc:
        bad.append("the read no longer carries the arm that scores P2Rank at "
                   "its own residue centroid, without which the headline "
                   "compares a residue centroid against a cavity centre and is "
                   "biased in our favour by construction")
    else:
        # The governing verdict is re-derived here rather than trusted, because
        # the whole point of the corrected arm is to be allowed to veto a win.
        r0 = plan["primary_hit_radius_angstrom"]

        def _v(arm):
            b = (arm["hit_rates"][f"{r0}A/top1"] or {})["paired_bonferroni"]
            if not b["excludes_zero"]:
                return "top1_unresolved"
            return ("top1_favours_the_field" if b["delta"] > 0
                    else "top1_favours_p2rank")

        kp, kf = _v(d["primary"]), _v(fc["arm"])
        want = kf
        if kp != kf and want == "top1_favours_the_field":
            want = "top1_unresolved"
        if d.get("outcome_key") != want:
            bad.append(f"the read reports {d.get('outcome_key')} but the two "
                       f"arms give {kp} and {kf}, which governs as {want}")
        if fc.get("the_two_arms_agree") != (kp == kf):
            bad.append("the read misreports whether the two arms agree")
        # Both P2Rank arms must rank the same candidate list, only measured from
        # a different point on each pocket. If the counts drift, the correction
        # changed the comparison instead of the measurement.
        for r in d["primary"].get("per_unit") or []:
            if (r["theirs"]["n_candidates"]
                    != r["theirs_residue_centroid"]["n_candidates"]):
                bad.append(f"{r['unit_id']}: the two P2Rank arms do not score "
                           f"the same candidate list")
    for arm_name in ("primary", "sensitivity_at_the_looser_cutoff"):
        arm = d.get(arm_name) or {}
        if arm_name == "primary" and fc:
            arms = [arm, fc["arm"]]
        else:
            arms = [arm]
        for r in arm.get("per_unit") or []:
            for side in ("ours", "theirs", "theirs_residue_centroid"):
                s = r[side]
                if s["n_candidates"] and s["top1_distance"] is None:
                    bad.append(f"{r['unit_id']}/{side}: a candidate with no "
                               f"distance to the labelled site")
        # A hit at a tighter radius must be a hit at a looser one, and a hit at
        # a smaller K must be a hit at a larger one. Both are properties of the
        # definition, so a violation is an arithmetic bug.
        for a in arms:
            radii = sorted({h["radius_angstrom"]
                            for h in (a.get("hit_rates") or {}).values()})
            ks = sorted({h["k"] for h in (a.get("hit_rates") or {}).values()})
            for side in ("ours", "p2rank"):
                for k in ks:
                    vals = [a["hit_rates"][f"{r}A/top{k}"][side] for r in radii]
                    if vals != sorted(vals):
                        bad.append(f"{arm_name}/{side}: the top-{k} hit rate "
                                   f"falls as the radius grows, which cannot "
                                   f"happen")
                for r in radii:
                    vals = [a["hit_rates"][f"{r}A/top{k}"][side] for k in ks]
                    if vals != sorted(vals):
                        bad.append(f"{arm_name}/{side}: the {r} A hit rate "
                                   f"falls as K grows, which cannot happen")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if bad:
        return 1
    _report(d)
    print(f"\nOK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        return check()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
