"""Run P2Rank on the 770 training receptors.

P2Rank had been scored only on the official test fold, which left every
statistical statement about our margin over it unpreregistrable: the choice of
summary statistic could only ever be made after seeing the held-out numbers.
That is the exact failure this repository exists to avoid. Scoring P2Rank on
the training partition costs nothing from the test-fold budget and turns the
question "which functional of the paired differences should we report" into
something answerable before the held-out fold is consulted.

This tool reads ``data/cryptobench_apo/train_receptors`` and ``train_labels``
only. It never opens the official manifest, the official receptors, or the
official labels, and the artifact it writes is filed under the architecture
sweep rather than the fold-access ledger for that reason.

Per-unit metrics come from ``telemetry_row``, the same function the official
run used, so the training-fold and test-fold numbers are commensurable.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pocket_bench.methods import p2rank_wrap
from pocket_bench.paths import ROOT
from pocket_bench.metrics import score_prediction
from pocket_bench.telemetry import telemetry_row

REC = ROOT / "data/cryptobench_apo/train_receptors"
LAB = ROOT / "data/cryptobench_apo/train_labels"
OUT = ROOT / "results/architecture_sweep/P2RANK_TRAIN_FOLD.json"


def _universe(rec: Path, chain: str | None) -> list[int]:
    """Residue numbers of the scored chain, in file order.

    Duplicated from ``run_cryptobench_apo`` rather than imported because that
    module executes a heavy import graph at load time and this tool runs it in
    eight worker processes.
    """
    seen: list[int] = []
    got = set()
    for line in rec.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if chain and line[21] != chain:
            continue
        try:
            n = int(line[22:26])
        except ValueError:
            continue
        if n not in got:
            got.add(n)
            seen.append(n)
    return seen


def _one(label_path: str) -> dict:
    lab = json.loads(Path(label_path).read_text())
    pdb, ch = lab["pdb_id"], lab.get("chain")
    rec = REC / f"{pdb}_{ch}_receptor.pdb"
    if not rec.is_file():
        return {"unit_id": f"{pdb}_{ch}", "status": "MISSING_RECEPTOR"}
    pred = p2rank_wrap.predict(rec, pdb_id=pdb, chain=ch)
    universe = _universe(rec, ch)
    if "ligand_heavy_coords" in lab:
        scored = score_prediction(pred, lab)
    else:
        # The training labels carry cryptic residues but no holo ligand
        # coordinates, exactly as the official ones do, so DCA is undefined and
        # reported null rather than fabricated. This mirrors the official run
        # so the two sets of residue-level numbers stay commensurable.
        scored = {"method": "p2rank", "pdb_id": pdb,
                  "status": pred.get("status", "OK"),
                  "runtime_s": pred.get("runtime_s"),
                  "primary_metric": "residue_level_only",
                  "clinical_grade": False, "top1": None, "top3": None,
                  "dcc_top1": None, "residue_f1": {"available": False}}
    return telemetry_row(
        method="p2rank", pdb=pdb, chain=ch, split="train",
        status=pred.get("status"), scored=scored, label=lab, prediction=pred,
        universe_residues=universe, tool_version=None, env_sha=None, seed=0,
        runtime_s=pred.get("runtime_s"),
    )


def build(workers: int = 8, limit: int | None = None) -> dict:
    labels = sorted(glob.glob(str(LAB / "*_labels.json")))
    if limit:
        labels = labels[:limit]
    t0 = time.perf_counter()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(_one, labels, chunksize=4), 1):
            rows.append(row)
            if i % 50 == 0 or i == len(labels):
                print(f"  {i}/{len(labels)}  {time.perf_counter()-t0:.0f}s",
                      flush=True)
    ok = [r for r in rows if r.get("status") == "OK"]
    auc = [r["residue_auc"] for r in ok if r.get("residue_auc") is not None]
    return {
        "schema": "geoaudit.p2rank_train_fold.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "what_this_is": (
            "P2Rank 2.5.1 scored on the 770 CryptoBench training receptors. "
            "Exists so that the choice of summary statistic for the paired "
            "comparison against P2Rank can be fixed on the training partition "
            "before the held-out fold is consulted."
        ),
        "test_fold_touched": False,
        "receptor_dir": str(REC.relative_to(ROOT)),
        "label_dir": str(LAB.relative_to(ROOT)),
        "p2rank_version": p2rank_wrap._version(),
        "jvm_version": p2rank_wrap._jvm_version(),
        "n_units": len(rows),
        "n_ok": len(ok),
        "n_with_residue_auc": len(auc),
        "residue_auc_mean": sum(auc) / len(auc) if auc else None,
        "wall_clock_s": round(time.perf_counter() - t0, 1),
        "rows": rows,
    }


def _report(d: dict) -> None:
    print(f"\nP2Rank {d['p2rank_version']} on the training fold")
    print(f"  units          {d['n_units']}  (OK {d['n_ok']}, "
          f"with residue AUC {d['n_with_residue_auc']})")
    print(f"  residue AUC    {d['residue_auc_mean']:.4f}")
    print(f"  wall clock     {d['wall_clock_s']:.0f}s")
    print(f"  test fold      untouched")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                    help="score only the first N units (smoke test)")
    ap.add_argument("--audit", action="store_true",
                    help="check the committed artifact instead of rebuilding")
    a = ap.parse_args(argv)
    if a.audit:
        if not OUT.is_file():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        d = json.loads(OUT.read_text())
        if d.get("test_fold_touched") is not False:
            print("FAILED: artifact does not declare the test fold untouched")
            return 1
        _report(d)
        return 0
    d = build(workers=a.workers, limit=a.limit)
    if not a.limit:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(d, indent=2) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}  "
              f"sha256 {hashlib.sha256(OUT.read_bytes()).hexdigest()[:16]}")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
