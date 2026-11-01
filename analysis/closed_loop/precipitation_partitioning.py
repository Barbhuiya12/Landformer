"""Long-term partitioning of precipitation (Section 4.6) for VIC, LandFormer-AR-All in open and closed loop and
LandFormer-NAR on the same test cells: ET/P, Q/P and the residual 1 - sum(ET)/sum(P) - sum(Q)/sum(P), accumulated
over 27 Sep 2001 - 31 Dec 2006. Cells are grouped into aridity terciles of VIC ET/P.
Uses the arrays written by closed_loop.py step1.
"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from closed_loop import OUT, T0, load_test_data, load_scaler, physical
cells = json.load(open(os.path.join(OUT, "step1_info.json")))["cells"]
d = load_test_data(); sI = load_scaler("inference")
obs = np.load(os.path.join(OUT, "vic_obs.npy")); H = obs.shape[0]
P = np.maximum(physical(d["F"][..., :1], ["prcp"], sI)[..., 0], 0)[T0:T0 + H][:, cells].sum(0)
runs = {"VIC": obs, **{k: np.load(os.path.join(OUT, f"pred_{k}.npy")) for k in ("open_loop", "closed_loop", "nar")}}
part = {k: {"ET/P": a[:, :, 0].sum(0) / P, "Q/P": a[:, :, 2].sum(0) / P} for k, a in runs.items()}
for k in part: part[k]["residual"] = 1 - part[k]["ET/P"] - part[k]["Q/P"]
# aridity terciles by VIC ET/P, as in consistency_persistence_gat.ipynb
ae = part["VIC"]["ET/P"]; terc = np.quantile(ae, [1/3, 2/3])
groups = {"humid (low ET/P)": ae <= terc[0], "intermediate": (ae > terc[0]) & (ae <= terc[1]), "arid (high ET/P)": ae > terc[1]}
out = {"n_cells": len(cells), "VIC_ET/P_tercile_edges": terc.tolist()}
for k, v in part.items():
    out[k] = {f"median {q}": float(np.median(x)) for q, x in v.items()}
    if k != "VIC":
        for q in ("ET/P", "Q/P", "residual"):
            dlt = v[q] - part["VIC"][q]
            out[k][f"median diff vs VIC {q}"] = float(np.median(dlt))
            out[k][f"% cells |diff|<0.05 {q}"] = float((np.abs(dlt) < 0.05).mean() * 100)
            out[k][f"r across cells {q}"] = float(np.corrcoef(v[q], part["VIC"][q])[0, 1])
        out[k]["median residual diff by aridity tercile"] = {g: float(np.median(v["residual"][m] - part["VIC"]["residual"][m])) for g, m in groups.items()}
json.dump(out, open(os.path.join(OUT, "precipitation_partitioning.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
