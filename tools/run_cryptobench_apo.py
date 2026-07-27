#!/usr/bin/env python3
"""Run pocket methods on the pinned CryptoBench-apo subset and write an honest report.

Metric: Top-1 DCA <= 4 A from the predicted pocket centre to the nearest atom of
the labelled cryptic binding residues (a pocket-LOCALIZATION proxy). NOTE this is
NOT the CryptoBench per-residue classification protocol (AUC/F1); a fully faithful
comparison to the CryptoBench baselines would use their residue-level metrics.

Methods: geometric_foundation (rigid), fstar_pocket (F*-breathing ablation),
sstar_pocket (anisotropic-shear S* oracle, dynamic spectral amplitudes),
p2rank (needs JAVA_HOME + P2Rank on PATH), random_bbox. TOOL_UNAVAILABLE is never
counted as a miss.

Usage: PYTHONPATH=src python3.12 tools/run_cryptobench_apo.py
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# Pin BLAS/OMP to a single thread PER WORKER before numpy/scipy are imported.
# The spectral step (ARPACK shift-invert on a 3N x 3N sparse Hessian) is the
# runtime hotspot; running P worker processes that each spawn a full BLAS thread
# pool oversubscribes the cores and is slower than serial. Parallelism is taken
# across structures instead, which is where it actually scales.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pocket_bench.adapters import load_official_test_fold
from pocket_bench.methods import (
    algebraic_field,
    algebraic_field_linear,
    fstar_pocket,
    geometric_foundation,
    p2rank_wrap,
    prediction,
    quaternary_lut,
    sstar_pocket,
    table_field,
    ultrametric_shear_oracle,
)
from pocket_bench.metrics import score_prediction
from pocket_bench.pdb_io import parse_pdb_atoms, sha256_file
from pocket_bench.telemetry import (
    aggregate,
    assert_denominator_discipline,
    declared_available_tools,
    telemetry_row,
)

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "data/cryptobench_apo/receptors"
LAB = ROOT / "data/cryptobench_apo/labels"
BASELINE_ENV = ROOT / "data/manifests/BASELINE_ENV.json"
# P2Rank's own CSVs, kept so the strongest baseline stays auditable without a JVM.
P2RANK_RAW = ROOT / "results/cryptobench_official/p2rank_raw"


def _receptor_residue_universe(rec: Path, chain: str | None) -> list[int]:
    resseqs: set[int] = set()
    for a in parse_pdb_atoms(rec.read_text()):
        if a["record"] != "ATOM":
            continue
        if chain is not None and a["chain"] != chain:
            continue
        resseqs.add(int(a["resseq"]))
    return sorted(resseqs)


def _archivable(pred: dict, universe: list[int]) -> dict:
    """The part of a prediction a third party needs to recompute our metrics.

    A frozen metric that cannot be recomputed from a stored prediction is an
    assertion, not a measurement: the reader has to trust the number. Keeping
    the per-residue score vector (or, for a pocket-only detector, the pocket
    centres it is derived from) makes every table in the paper checkable
    without re-running the detector -- which matters most for the one detector
    a reader may be unable to run at all, since P2Rank needs a JVM.
    """
    raw = pred.get("residue_scores") or {}
    scores = {str(r): float(raw[str(r)]) for r in universe if str(r) in raw}
    return {
        "status": pred.get("status"),
        "runtime_s": pred.get("runtime_s"),
        "input_receptor_sha256": pred.get("input_receptor_sha256"),
        "tool_version": (pred.get("tool_version")
                         or (pred.get("extra") or {}).get("tool_version")),
        "n_universe": len(universe),
        "residue_scores": scores or None,
        "residue_positive": pred.get("residue_positive"),
        "pockets": [{"rank": p.get("rank"), "center_xyz": p.get("center_xyz"),
                     "score": p.get("score")}
                    for p in (pred.get("pockets") or [])],
        "error": pred.get("error"),
    }


def _write_predictions(dest: Path, raw: dict[str, dict[str, dict]]) -> None:
    """One file per method, plus a SHA-256 index over the whole directory.

    Only methods scored in THIS run are rewritten; a file belonging to a method
    carried over by ``--merge`` is left untouched and still indexed, so the
    archive matches the telemetry rather than shrinking to the last run.
    """
    dest.mkdir(parents=True, exist_ok=True)
    for method, by_unit in sorted(raw.items()):
        payload = {"schema": "geoaudit.raw_predictions.v1",
                   "clinical_grade": False,
                   "method": method,
                   "n_units": len(by_unit),
                   "units": dict(sorted(by_unit.items()))}
        (dest / f"{method}.json").write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n")
    index = {}
    for path in sorted(dest.glob("*.json")):
        if path.name == "INDEX.json":
            continue
        doc = json.loads(path.read_text())
        index[doc.get("method", path.stem)] = {
            "file": path.name,
            "n_units": doc.get("n_units"),
            "rescored_in_last_run": doc.get("method") in raw,
            "sha256": sha256_file(path),
        }
    (dest / "INDEX.json").write_text(
        json.dumps({"schema": "geoaudit.raw_predictions_index.v1",
                    "clinical_grade": False, "methods": index},
                   indent=2) + "\n")
    print(f"raw per-residue predictions -> {dest.name}/ ({len(index)} methods)")


def _random_baseline(rec: Path, *, pdb_id: str, seed: int = 42, top_k: int = 5) -> dict:
    t0 = time.perf_counter()
    atoms = [a for a in parse_pdb_atoms(rec.read_text()) if a["record"] == "ATOM"]
    xs = [a["x"] for a in atoms]; ys = [a["y"] for a in atoms]; zs = [a["z"] for a in atoms]
    rng = random.Random(seed)
    pockets = [
        {"rank": r, "center_xyz": [rng.uniform(min(xs), max(xs)),
         rng.uniform(min(ys), max(ys)), rng.uniform(min(zs), max(zs))],
         "score": 1.0 / r, "residues": []}
        for r in range(1, top_k + 1)
    ]
    return prediction(method="random_bbox", pdb_id=pdb_id, status="OK",
                      pockets=pockets, runtime_s=time.perf_counter() - t0)


def _pilot_items() -> list[tuple[dict, Path]]:
    """The pinned n=15 stride pilot (NOT the official fold)."""
    items: list[tuple[dict, Path]] = []
    for lp in sorted(glob.glob(str(LAB / "*_labels.json"))):
        lab = json.loads(Path(lp).read_text())
        items.append((lab, REC / f"{lab['pdb_id']}_{lab['chain']}_receptor.pdb"))
    return items


def _official_items() -> list[tuple[dict, Path]]:
    """The official CryptoBench MMseqs2 10% cluster-disjoint TEST fold.

    Fail-closed by construction: ``load_official_test_fold`` raises if the manifest
    is absent or any SHA-256 mismatches. There is deliberately NO fallback to the
    pilot — silently substituting a 15-structure stride sample for the official fold
    is the exact failure mode this flag exists to prevent.
    """
    manifest = load_official_test_fold()
    items: list[tuple[dict, Path]] = []
    for e in manifest["entries"]:
        lab = json.loads((ROOT / e["label_path"]).read_text())
        items.append((lab, ROOT / e["receptor_path"]))
    return items


def _summaries_from_rows(rows: list[dict]) -> dict[str, dict]:
    """Per-method counts derived from telemetry rows alone.

    Used when merging, so the benchmark summary describes every method present
    in the telemetry rather than only the ones the current invocation re-scored.
    """
    from pocket_bench.paths import STATUS_OK, STATUS_TOOL_UNAVAILABLE

    out: dict[str, dict] = {}
    for r in rows:
        m = r["method"]
        a = out.setdefault(m, {"ok": 0, "top1_hits": 0, "unavailable": 0,
                               "crash_empty": 0, "rows": []})
        st = r.get("status")
        if st == STATUS_OK:
            a["ok"] += 1
        elif st == STATUS_TOOL_UNAVAILABLE:
            a["unavailable"] += 1
        else:
            a["crash_empty"] += 1
        if r.get("top1_success"):
            a["top1_hits"] += 1
    return {m: {"per_method": a,
                "summary": {
                    "top1_dca_le_4A_hits": a["top1_hits"],
                    "intention_to_evaluate_denominator": (a["ok"]
                                                          + a["crash_empty"]),
                    "tool_unavailable": a["unavailable"]}}
            for m, a in out.items()}


def _environment_sha() -> str | None:
    """The digest of the measured stack, from the committed environment lock."""
    p = ROOT / "ENVIRONMENT.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text()).get("environment_sha256")
    except Exception:  # noqa: BLE001
        return None


def _predict(method: str, rec: Path, pdb: str, chain: str | None = None) -> dict:
    if method == "geometric_foundation":
        return geometric_foundation.predict(rec, pdb_id=pdb)
    if method == "fstar_pocket":
        return fstar_pocket.predict(rec, pdb_id=pdb)
    if method == "sstar_pocket":
        return sstar_pocket.predict(rec, pdb_id=pdb)
    if method == "ultrametric_shear_oracle":
        return ultrametric_shear_oracle.predict(rec, pdb_id=pdb)
    if method == "algebraic_field":
        return algebraic_field.predict(rec, pdb_id=pdb, chain=chain)
    if method == "algebraic_field_linear":
        return algebraic_field_linear.predict(rec, pdb_id=pdb, chain=chain)
    if method == "table_field":
        return table_field.predict(rec, pdb_id=pdb, chain=chain)
    if method == "quaternary_lut":
        return quaternary_lut.predict(rec, pdb_id=pdb, chain=chain, track="A")
    if method == "quaternary_lut_seq":
        return quaternary_lut.predict(rec, pdb_id=pdb, chain=chain, track="B")
    if method == "p2rank":
        return p2rank_wrap.predict(rec, pdb_id=pdb, chain=chain,
                                   archive_dir=P2RANK_RAW)
    if method == "random_bbox":
        return _random_baseline(rec, pdb_id=pdb)
    raise KeyError(method)


METHOD_NAMES = ("geometric_foundation", "fstar_pocket", "sstar_pocket",
                "ultrametric_shear_oracle", "quaternary_lut",
                "quaternary_lut_seq", "algebraic_field",
                "algebraic_field_linear", "table_field", "p2rank",
                "random_bbox")


def _run_one(item: tuple[dict, Path],
             method_names: tuple[str, ...] = METHOD_NAMES,
             ) -> tuple[list[int], dict[str, tuple[dict, dict]]]:
    """Score every method on ONE structure. Module-level so it is picklable.

    Pure function of (label, receptor): no shared state, no RNG dependence on
    evaluation order (random_bbox is seeded per structure), so the parallel result
    is identical to the serial one.
    """
    lab, rec = item
    pdb, ch = lab["pdb_id"], lab["chain"]
    universe = _receptor_residue_universe(rec, ch)
    res: dict[str, tuple[dict, dict]] = {}
    for m in method_names:
        pred = _predict(m, rec, pdb, ch)
        if "ligand_heavy_coords" in lab:
            sc = score_prediction(pred, lab)
        else:
            # Official apo labels carry cryptic residues but no holo ligand
            # coordinates; DCA is undefined there, so it is reported null rather
            # than fabricated. Residue-level metrics remain valid.
            sc = {"method": m, "pdb_id": pdb, "status": pred.get("status", "OK"),
                  "runtime_s": pred.get("runtime_s"),
                  "primary_metric": "residue_level_only",
                  "clinical_grade": False, "top1": None, "top3": None,
                  "dcc_top1": None, "residue_f1": {"available": False}}
        res[m] = (pred, sc)
    return universe, res


class _Checkpoint:
    """Per-structure scoring cache so an interrupted run resumes where it stopped.

    A full official run is hours of work whose unit of progress is one structure,
    and losing all of it to a terminated shell is a reproducibility problem, not
    only an inconvenience: it pushes an operator towards partial or hand-patched
    freezes. Each structure is appended as one JSON line the moment it is scored.

    Correctness rests on _run_one being a pure function of (label, receptor) for
    a fixed method list. The header therefore pins the method list and the
    dataset, and a checkpoint written under a different method list is discarded
    rather than reused, so a resume can never silently mix two method sets. A
    truncated final line -- the normal result of a kill mid-write -- is dropped.
    """

    def __init__(self, path: Path, items: list[tuple[dict, Path]],
                 mnames: tuple[str, ...], *, enabled: bool) -> None:
        self.path = path
        self.enabled = enabled
        self.methods = list(mnames)
        self._cache: dict[str, tuple[list[int], dict]] = {}
        self._fh = None
        if not enabled:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            self._load()
        if not self._cache:
            path.write_text(json.dumps({"header": {"methods": self.methods}}) + "\n")
        self._fh = path.open("a")

    @staticmethod
    def _key(item: tuple[dict, Path]) -> str:
        lab, _ = item
        return f"{lab['pdb_id']}_{lab['chain']}"

    def _load(self) -> None:
        lines = self.path.read_text().splitlines()
        if not lines:
            return
        try:
            header = json.loads(lines[0]).get("header") or {}
        except json.JSONDecodeError:
            return
        if header.get("methods") != self.methods:
            print(f"  checkpoint {self.path.name} was written for "
                  f"{header.get('methods')}, not {self.methods}; ignoring it",
                  flush=True)
            self.path.unlink()
            return
        for line in lines[1:]:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                break        # a partial line is the tail of an interrupted write
            self._cache[rec["unit"]] = (
                rec["universe"],
                {m: (v["pred"], v["score"]) for m, v in rec["results"].items()},
            )

    @property
    def n_done(self) -> int:
        return len(self._cache)

    def get(self, item: tuple[dict, Path]):
        if not self.enabled:
            return None
        return self._cache.get(self._key(item))

    def put(self, item: tuple[dict, Path],
            result: tuple[list[int], dict]) -> None:
        if not self.enabled or self._fh is None:
            return
        universe, res = result
        self._fh.write(json.dumps({
            "unit": self._key(item),
            "universe": list(universe),
            "results": {m: {"pred": pred, "score": score}
                        for m, (pred, score) in res.items()},
        }, allow_nan=False) + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def clear(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self.enabled and self.path.exists():
            self.path.unlink()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=("pilot", "official"), default="pilot",
                    help="'official' = CryptoBench MMseqs2 10%% test fold "
                         "(fail-closed, never falls back to the pilot)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes over structures "
                         "(results are order-preserving and identical to --jobs 1)")
    ap.add_argument("--methods", default=",".join(METHOD_NAMES),
                    help="comma-separated subset of: " + ",".join(METHOD_NAMES))
    ap.add_argument("--merge", action="store_true",
                    help="merge these methods' rows into an existing TELEMETRY.json "
                         "instead of replacing it. Rows for methods NOT in --methods "
                         "are carried over verbatim, so re-scoring one detector "
                         "never silently drops a frozen baseline (e.g. P2Rank, "
                         "whose binary may be unavailable on this host).")
    ap.add_argument("--resume", action="store_true",
                    help="checkpoint each structure as it is scored and skip the "
                         "ones already present, so an interrupted run continues "
                         "instead of restarting. The checkpoint is keyed on the "
                         "method set, so it can never mix two method lists.")
    args = ap.parse_args(argv)
    mnames = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = set(mnames) - set(METHOD_NAMES)
    if unknown:
        raise SystemExit(f"unknown methods: {sorted(unknown)}")
    is_official = args.dataset == "official"
    items = _official_items() if is_official else _pilot_items()
    out = ROOT / ("results/cryptobench_official" if is_official
                  else "results/cryptobench_apo")

    # One BLAS/OpenMP thread per worker: the parallelism is across structures, so
    # nested threading would oversubscribe the cores and slow the run down.
    if args.jobs > 1:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[var] = "1"

    methods = {m: None for m in mnames}
    env = json.loads(BASELINE_ENV.read_text())
    p2rank_version = ((env.get("tools") or {}).get("p2rank") or {}).get("version")
    # Every row records the stack that produced it. This used to be hardcoded
    # None, which meant the telemetry carried a field named env_sha and no
    # environment, and a run under a different BLAS was indistinguishable from
    # one under the same BLAS.
    env_sha = _environment_sha()
    per = {m: {"ok": 0, "top1_hits": 0, "unavailable": 0, "crash_empty": 0, "rows": []} for m in methods}
    telem_rows: list[dict] = []
    n_attempted = {m: 0 for m in methods}
    t_start = time.perf_counter()
    ckpt = _Checkpoint(out / "_scoring_checkpoint.jsonl", items, mnames,
                       enabled=args.resume)
    if ckpt.enabled and ckpt.n_done:
        print(f"  resuming: {ckpt.n_done}/{len(items)} structures already "
              f"scored in {ckpt.path.name}", flush=True)

    results = None
    if args.jobs > 1 and not ckpt.enabled:
        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
                results = []
                stream = ex.map(_run_one, items, itertools.repeat(mnames),
                                chunksize=1)
                for done, r in enumerate(stream, 1):
                    results.append(r)
                    if done % 10 == 0 or done == len(items):
                        el = time.perf_counter() - t_start
                        print(f"  [{done}/{len(items)}] {el:.0f}s elapsed, "
                              f"~{el / done * (len(items) - done):.0f}s left",
                              flush=True)
        except (PermissionError, OSError, NotImplementedError) as exc:
            # Hardened runtimes deny the POSIX semaphore probe that
            # ProcessPoolExecutor performs at construction. _run_one is a pure
            # function of (label, receptor), so the sequential result is
            # identical; refusing to run at all would make the documented
            # command unreproducible on such a host for no scientific reason.
            print(f"  process pool unavailable ({type(exc).__name__}: {exc}); "
                  f"falling back to --jobs 1 (identical results)", flush=True)
            results = None
    if results is None:
        results = []
        n_fresh = 0
        for done, it in enumerate(items, 1):
            cached = ckpt.get(it)
            if cached is not None:
                results.append(cached)
                continue
            r = _run_one(it, mnames)
            ckpt.put(it, r)
            results.append(r)
            n_fresh += 1
            if n_fresh % 5 == 0 or done == len(items):
                el = time.perf_counter() - t_start
                left = len(items) - done
                print(f"  [{done}/{len(items)}] {el:.0f}s elapsed, "
                      f"~{el / max(n_fresh, 1) * left:.0f}s left", flush=True)
    print(f"  predictions done in {time.perf_counter() - t_start:.0f}s "
          f"(jobs={args.jobs})", flush=True)

    raw_preds: dict[str, dict[str, dict]] = {m: {} for m in methods}
    for idx, ((lab, rec), (universe, res)) in enumerate(zip(items, results), 1):
        pdb, ch = lab["pdb_id"], lab["chain"]
        for m in methods:
            pred, sc = res[m]
            raw_preds[m][f"{pdb}_{ch}"] = _archivable(pred, universe)
            agg = per[m]; st = sc["status"]; top1 = sc.get("top1") or {}
            n_attempted[m] += 1
            if st == "TOOL_UNAVAILABLE":
                agg["unavailable"] += 1
            elif st != "OK":
                agg["crash_empty"] += 1
            else:
                agg["ok"] += 1
                if top1.get("success") is True:
                    agg["top1_hits"] += 1
            agg["rows"].append({"pdb": pdb, "status": st,
                                "top1_success": top1.get("success"),
                                "best_dca": top1.get("best_dca")})
            telem_rows.append(
                telemetry_row(
                    method=m, pdb=pdb, chain=ch, split="test", status=st,
                    scored=sc, label=lab, prediction=pred,
                    universe_residues=universe,
                    tool_version=p2rank_version if m == "p2rank" else None,
                    env_sha=env_sha, seed=42 if m == "random_bbox" else 0,
                    runtime_s=pred.get("runtime_s"),
                )
            )
    summaries = {m: {"top1_dca_le_4A_hits": a["top1_hits"],
                     "intention_to_evaluate_denominator": a["ok"] + a["crash_empty"],
                     "tool_unavailable": a["unavailable"]} for m, a in per.items()}
    report = {
        "schema": "geoaudit.cryptobench.report.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "clinical_grade": False,
        "dataset": args.dataset,
        "is_official_mmseqs2_10pct_test_fold": is_official,
        "benchmark": (
            "CryptoBench official MMseqs2 10% cluster-disjoint TEST fold "
            "(Skrhak 2025)" if is_official else
            "CryptoBench apo (Skrhak 2025) — pinned deterministic subset"
        ),
        "n_structures": len(items),
        "primary_metric": ("residue_auc/residue_pr_auc/residue_mcc/residue_f1"
                           if is_official else
                           "top1_dca_le_4A_to_cryptic_residue_atoms"),
        "metric_caveat": (
            "Official apo labels carry cryptic residues but no holo ligand "
            "coordinates; DCA is undefined and reported null. Residue-level metrics "
            "are the primary readout." if is_official else
            "Pocket-localization proxy (DCA to cryptic-residue atoms); NOT the "
            "CryptoBench per-residue AUC/F1 protocol. DCA is more permissive for "
            "structures with many labelled residues."
        ),
        "splits_note": (
            "Official CryptoBench TEST fold: MMseqs2 clustering at 10% sequence "
            "identity, loaded fail-closed with per-file SHA-256 verification. "
            "clinical_grade=false." if is_official else
            "CryptoBench test splits are cluster-disjoint at 10% sequence identity; "
            "this pinned subset is a deterministic stride sample of the full label "
            "set (train+test), not exclusively the test fold. clinical_grade=false."
        ),
        "summaries": summaries,
        "per_method": per,
    }
    if args.merge:
        prior_path = out / "TELEMETRY.json"
        if not prior_path.exists():
            raise SystemExit(f"--merge requires an existing {prior_path}")
        prior = json.loads(prior_path.read_text())["rows"]
        rescored = set(mnames)
        carried = [r for r in prior if r.get("method") not in rescored]
        dropped = {r.get("method") for r in prior} & rescored
        telem_rows = carried + telem_rows

        # The benchmark summary must be merged too, and for a while it was not:
        # re-scoring one detector carried every telemetry row but overwrote
        # APO_BENCHMARK.json with only that detector, so the file declared 192
        # structures beside a single method and looked like an interrupted run.
        # Rebuilt from the merged rows rather than carried from the previous
        # report, because a carried report is only as complete as whatever the
        # run before it happened to score, and one truncated run would poison
        # every later merge.
        for m, v in _summaries_from_rows(telem_rows).items():
            if m not in rescored:
                report["per_method"].setdefault(m, v["per_method"])
                report["summaries"].setdefault(m, v["summary"])
        # The denominator is per method: a carried method keeps the number of
        # structures it was originally attempted on, so the fail-closed
        # denominator discipline still holds across the merge.
        for r in carried:
            n_attempted[r["method"]] = n_attempted.get(r["method"], 0) + 1
        print(f"merge: carried {len(carried)} frozen rows for "
              f"{sorted({r['method'] for r in carried})}; "
              f"replaced {sorted(dropped)}")

    # 0-masking telemetry: faithful per-residue metrics + fail-closed denominators.
    telemetry = aggregate(telem_rows, n_attempted)
    assert_denominator_discipline(telemetry, declared_available_tools(env))
    telemetry["rows"] = telem_rows
    report["telemetry_ref"] = "TELEMETRY.json"
    out.mkdir(parents=True, exist_ok=True)
    (out / "APO_BENCHMARK.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "TELEMETRY.json").write_text(json.dumps(telemetry, indent=2) + "\n")
    _write_predictions(out / "predictions", raw_preds)
    # The freeze is on disk, so the scratch checkpoint has no further claim to
    # being evidence; leaving it would let a later run resume from a method set
    # that has already been superseded.
    ckpt.clear()
    print(json.dumps({m: s["top1_dca_le_4A_hits"] for m, s in summaries.items()}, indent=2))
    # faithful metric availability (honest: null where universe/labels cannot join)
    avail = sum(1 for r in telem_rows if r["residue_metrics_available"])
    print(f"faithful residue metrics available on {avail}/{len(telem_rows)} rows")
    print(f"dataset={args.dataset} n={len(items)} -> {out.relative_to(ROOT)}/"
          "{APO_BENCHMARK,TELEMETRY}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
