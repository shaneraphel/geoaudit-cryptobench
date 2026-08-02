#!/usr/bin/env python3.12
"""Equal-z seam_geometry ⊕ seam_voidpharma score fusion vs pLM."""
from __future__ import annotations
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from sklearn.metrics import roc_auc_score
from pocket_bench.paths import ROOT
N_JOBS=min(9,os.cpu_count() or 4)
OUT=ROOT/"results/official_fold/SEAM_VOIDPHARMA_EQUALZ_FUSION.json"

def _pos(lab): return {int(x) for x in lab.get("cryptic_residues",[])}
def _z(v):
    sd=float(v.std()); return (v-float(v.mean()))/sd if sd>1e-12 else np.zeros_like(v)

def one(e):
    from pocket_bench.methods import seam_geometry_field, seam_voidpharma_field
    unit=f"{e['pdb']}_{e['chain']}"
    try:
        a=seam_geometry_field.predict(ROOT/e["receptor_path"], pdb_id=e["pdb"], chain=e["chain"])
        b=seam_voidpharma_field.predict(ROOT/e["receptor_path"], pdb_id=e["pdb"], chain=e["chain"])
        lab=json.loads((ROOT/e["label_path"]).read_text()); pos=_pos(lab)
        res=sorted(int(r) for r in a["residue_scores"])
        av=np.array([float(a["residue_scores"][str(r)]) for r in res])
        bv=np.array([float(b["residue_scores"][str(r)]) for r in res])
        fused=_z(av)+_z(bv)
        y=np.array([1 if r in pos else 0 for r in res])
        if y.sum()==0 or y.sum()==len(y): return unit, {"error":"deg"}
        return unit, {"seam":float(roc_auc_score(y,av)),"vp":float(roc_auc_score(y,bv)),
                      "fused":float(roc_auc_score(y,fused))}
    except Exception as ex:
        return unit, {"error":f"{type(ex).__name__}:{ex}"}

def main():
    man=json.loads((ROOT/"data/cryptobench_apo/official_manifest.json").read_text())
    plm={u["unit_id"]:u for u in json.loads((ROOT/"results/baselines/PLMNN_SCORES.json").read_text())["units"]}
    t0=time.perf_counter(); got={}
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        for i,fut in enumerate(as_completed([ex.submit(one,e) for e in man["entries"]]),1):
            u,r=fut.result(); got[u]=r
            if i%30==0: print(i, flush=True)
    rows=[]
    for e in man["entries"]:
        unit=f"{e['pdb']}_{e['chain']}"; g=got.get(unit,{})
        if "fused" not in g or unit not in plm: continue
        lab=json.loads((ROOT/e["label_path"]).read_text()); pos=_pos(lab)
        ps=plm[unit]["scores"]; res=sorted(int(k) for k in ps)
        y=np.array([1 if r in pos else 0 for r in res]); s=np.array([float(ps[str(r)]) for r in res])
        if y.sum()==0 or y.sum()==len(y): continue
        pauc=float(roc_auc_score(y,s))
        rows.append({**g,"d":g["fused"]-pauc})
    d=np.array([r["d"] for r in rows]); rng=np.random.default_rng(0)
    boots=[float(d[rng.integers(0,len(d),len(d))].mean()) for _ in range(4000)]
    out={"schema":"geoaudit.seam_voidpharma_equalz_fusion.v1","clinical_grade":False,
         "reads_test_fold":True,
         "mean_seam":float(np.mean([r["seam"] for r in rows])),
         "mean_vp":float(np.mean([r["vp"] for r in rows])),
         "mean_fused":float(np.mean([r["fused"] for r in rows])),
         "mean_plmnn":float(np.mean([r["d"]+r["fused"]-r["d"] for r in rows])),  # placeholder fix below
         "mean_delta":float(d.mean()),
         "ci95":[float(np.percentile(boots,2.5)),float(np.percentile(boots,97.5))],
         "n_ahead":int((d>0).sum()),"n_behind":int((d<0).sum())}
    # fix plm mean
    plms=[]
    for e in man["entries"]:
        unit=f"{e['pdb']}_{e['chain']}"
        if unit not in plm: continue
        lab=json.loads((ROOT/e["label_path"]).read_text()); pos=_pos(lab)
        ps=plm[unit]["scores"]; res=sorted(int(k) for k in ps)
        y=np.array([1 if r in pos else 0 for r in res]); s=np.array([float(ps[str(r)]) for r in res])
        if y.sum() and y.sum()<len(y): plms.append(float(roc_auc_score(y,s)))
    out["mean_plmnn"]=float(np.mean(plms))
    OUT.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps(out,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
