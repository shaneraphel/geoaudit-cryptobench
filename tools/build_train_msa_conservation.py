#!/usr/bin/env python3.12
"""Build per-residue Swiss-Prot conservation for the CryptoBench training fold.

Extracts sequences from train receptors (9 cores), runs mmseqs2 easy-search
against the local Swiss-Prot DB, and writes
``data/msa_cache/train_conservation.json`` plus a wire cache
``data/cryptobench_apo/_msa_conserv_cache_train.npz`` (several distinct MSA
statistics × 3 aggregations).

This is not unit-level match-fraction fusion (already null). It prepares columns
for a counting-field compile. clinical_grade = false. Train receptors only.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pocket_bench.pdb_io import parse_pdb_atoms
from pocket_bench.paths import ROOT

N_JOBS = min(9, os.cpu_count() or 4)
MMSEQS = shutil.which("mmseqs") or "mmseqs"
SPROT = ROOT / "data/msa_cache/sprot_db/sprot"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
CONS_JSON = ROOT / "data/msa_cache/train_conservation.json"
QUERY_FA = ROOT / "data/msa_cache/train_queries.fasta"
M8 = ROOT / "data/msa_cache/train_vs_sprot.m8"
CACHE = ROOT / "data/cryptobench_apo/_msa_conserv_cache_train.npz"

CONTACT_RADIUS = 8.0
AGGREGATIONS = ("own", "contact", "walk2")
SKIP = frozenset({"HOH", "WAT", "DOD"})
AA3 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "SEC": "U", "PYL": "O",
}

def _one_seq(args):
    u, path, chain = args
    atoms = parse_pdb_atoms(Path(path).read_text())
    keep = [a for a in atoms
            if a["chain"] == chain and a["element"] != "H"
            and a["resname"] not in SKIP]
    # first CA per resseq
    order = []
    seen = set()
    for a in sorted(keep, key=lambda x: (x["resseq"], x["icode"])):
        if a["name"].strip() != "CA":
            continue
        r = int(a["resseq"])
        if r in seen:
            continue
        seen.add(r)
        order.append((r, AA3.get(a["resname"].strip().upper(), "X")))
    seq = "".join(aa for _, aa in order)
    resseq = [r for r, _ in order]
    return u, seq, resseq


def extract_sequences() -> dict[str, dict]:
    entries = json.loads(MANIFEST.read_text())["entries"]
    jobs = [(f"{e['pdb']}_{e['chain']}", str(ROOT / e["receptor_path"]), e["chain"])
            for e in entries]
    out = {}
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futs = [ex.submit(_one_seq, j) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            u, seq, resseq = fut.result()
            out[u] = {"sequence": seq, "resseq": resseq}
            if i % 100 == 0:
                print(f"  seq {i}/{len(jobs)} {time.perf_counter()-t0:.0f}s",
                      flush=True)
    return out


def write_fasta(seqs: dict[str, dict], path: Path) -> None:
    with path.open("w") as fh:
        for u, rec in sorted(seqs.items()):
            fh.write(f">{u}\n")
            s = rec["sequence"]
            for i in range(0, len(s), 80):
                fh.write(s[i:i + 80] + "\n")


def run_mmseqs(query_fa: Path, out_m8: Path) -> None:
    if not SPROT.exists() and not Path(str(SPROT) + ".dbtype").exists():
        # mmseqs DB may be prefix
        pass
    tmp = ROOT / "data/msa_cache/mmseqs_tmp_train"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    cmd = [
        MMSEQS, "easy-search", str(query_fa), str(SPROT), str(out_m8), str(tmp),
        "--threads", str(N_JOBS),
        "-s", "5.0",
        "--max-seqs", "50",
        "--format-output", "query,target,qstart,qend,tstart,tend,qaln,taln,evalue",
        "-e", "1e-3",
    ]
    print("RUN", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


COLUMN_QTY = (
    "match_frac_x100",
    "n_hits_covering",
    "n_hits_matching",
    "best_span",
    "mean_neglog_e",
    "rank_match_frac",
    "rank_n_cover",
)


def conservation_tables(m8: Path, seqs: dict[str, dict]
                        ) -> dict[str, dict[str, dict[str, float]]]:
    """unit -> resseq(str) -> distinct MSA column statistics."""
    match = {u: np.zeros(len(rec["resseq"]), dtype=np.float64)
             for u, rec in seqs.items()}
    cover = {u: np.zeros(len(rec["resseq"]), dtype=np.float64)
             for u, rec in seqs.items()}
    span_sum = {u: np.zeros(len(rec["resseq"]), dtype=np.float64)
                for u, rec in seqs.items()}
    nlog_sum = {u: np.zeros(len(rec["resseq"]), dtype=np.float64)
                for u, rec in seqs.items()}

    with m8.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            unit, _t, qs, qe = parts[0], parts[1], int(parts[2]), int(parts[3])
            qaln, taln, ev = parts[6], parts[7], float(parts[8])
            if unit not in match:
                continue
            span = qe - qs + 1
            nle = max(0.0, -np.log10(max(ev, 1e-300)))
            qpos = qs - 1
            for a, b in zip(qaln, taln):
                if a == "-":
                    continue
                if 0 <= qpos < len(match[unit]):
                    cover[unit][qpos] += 1.0
                    span_sum[unit][qpos] += span
                    nlog_sum[unit][qpos] += nle
                    if b != "-" and a.upper() == b.upper():
                        match[unit][qpos] += 1.0
                qpos += 1

    out: dict[str, dict[str, dict[str, float]]] = {}
    for u, rec in seqs.items():
        n = len(rec["resseq"])
        mf = np.zeros(n)
        for i in range(n):
            if cover[u][i] > 0:
                mf[i] = 100.0 * match[u][i] / cover[u][i]
        rmf = np.argsort(np.argsort(mf)) / max(n - 1, 1) * 100.0
        rnc = np.argsort(np.argsort(cover[u])) / max(n - 1, 1) * 100.0
        per = {}
        for i, r in enumerate(rec["resseq"]):
            c = cover[u][i]
            per[str(r)] = {
                "match_frac_x100": float(mf[i]),
                "n_hits_covering": float(c),
                "n_hits_matching": float(match[u][i]),
                "best_span": float(span_sum[u][i] / c) if c else 0.0,
                "mean_neglog_e": float(nlog_sum[u][i] / c) if c else 0.0,
                "rank_match_frac": float(rmf[i]),
                "rank_n_cover": float(rnc[i]),
            }
        out[u] = per
    return out


def build_wire_cache(cons: dict, seqs: dict[str, dict]) -> None:
    w = np.load(WIDE, allow_pickle=False)
    units = [str(u) for u in w["units"]]
    n_res, ctr = w["n_res_per"], w["ctr"]
    w.close()

    n_qty = len(COLUMN_QTY)
    prop = np.zeros((int(n_res.sum()), n_qty), dtype=np.float64)
    off = 0
    for u, n in zip(units, n_res):
        n = int(n)
        # align to wide-cache residue order via resseq from sequence extract —
        # wide cache order is the evaluation universe; we need matching resseqs.
        # Pull from train receptor via same order as wide? Use seqs resseq order
        # which should match universe sorted order.
        rec = seqs[u]
        per = cons[u]
        # When lengths disagree, pad/truncate carefully
        vals = []
        for r in rec["resseq"]:
            row = per.get(str(r))
            if row is None:
                vals.append([0.0] * n_qty)
            else:
                vals.append([row[q] for q in COLUMN_QTY])
        V = np.asarray(vals, dtype=np.float64)
        if V.shape[0] != n:
            # length mismatch — fill zeros (fail soft for that unit)
            print(f"WARN {u}: seq {V.shape[0]} != cache {n}", flush=True)
            block = np.zeros((n, n_qty))
            m = min(n, V.shape[0])
            block[:m] = V[:m]
            prop[off:off + n] = block
        else:
            prop[off:off + n] = V
        off += n

    # aggregate
    blocks, off = [], 0
    for n in n_res:
        n = int(n)
        c = ctr[off:off + n]
        p = prop[off:off + n]
        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        adj = (d <= CONTACT_RADIUS).astype(np.float64)
        two = adj @ adj
        np.fill_diagonal(two, 0.0)
        blocks.append(np.concatenate([p, adj @ p, two @ p], axis=1))
        off += n
    C = np.concatenate(blocks, axis=0).astype(np.float32)
    names = tuple(f"{agg}~{q}" for agg in AGGREGATIONS for q in COLUMN_QTY)
    np.savez_compressed(CACHE, C=C, names=np.asarray(names))
    print(f"wrote {CACHE.relative_to(ROOT)} {C.shape}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-mmseqs", action="store_true",
                    help="reuse existing m8 / conservation json")
    args = ap.parse_args()

    print("extracting train sequences…", flush=True)
    seqs = extract_sequences()
    (ROOT / "data/msa_cache/train_sequences.json").write_text(
        json.dumps({u: {"sequence": r["sequence"], "resseq": r["resseq"]}
                    for u, r in seqs.items()}, indent=2) + "\n")
    write_fasta(seqs, QUERY_FA)
    print(f"wrote {QUERY_FA.relative_to(ROOT)}  {len(seqs)} units", flush=True)

    if not args.skip_mmseqs or not M8.exists():
        run_mmseqs(QUERY_FA, M8)
    print("building conservation tables…", flush=True)
    cons = conservation_tables(M8, seqs)
    CONS_JSON.write_text(json.dumps(cons) + "\n")
    print(f"wrote {CONS_JSON.relative_to(ROOT)}", flush=True)

    print("building wire cache…", flush=True)
    build_wire_cache(cons, seqs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
