"""Dependence of test skill on the proximity of training cells (Section 5).

For every test cell: great-circle distance to the nearest training cell, and whether it shares an ERA5 0.25 deg grid
box with a training cell (the forcing is ERA5 interpolated to 0.0625 deg). Per-cell skill of LandFormer-AR-All and
LandFormer-NAR is related to both with Spearman correlation and the Mann-Whitney test.
Uses the per-cell metrics written by analysis/closed_loop/closed_loop.py and the training-cell lists saved by the
trainer (results/<run>/grid_locations_train.csv).
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..')))
from landformer_paths import RESULTS_DIR, ROOT  # noqa: E402
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.join(HERE, "..", "closed_loop")
RES = f"{RESULTS_DIR}"
TRAIN = {"open_loop": f"{RES}/Large_full_lag/grid_locations_train.csv", "nar": f"{RES}/TEST_HYDRODL_4/grid_locations_train.csv"}
cells = json.load(open(os.path.join(CL, "step1_info.json")))["cells"]
a = np.load(os.path.join(CL, "cache", "test_arrays.npz")); lat, lon = a["lat"][cells], a["lon"][cells]
pc = pd.read_csv(os.path.join(CL, "per_cell_metrics.csv"))
BINS = [0, 10, 25, 50, 100, 1e9]; LABELS = ["<10", "10-25", "25-50", "50-100", ">100"]


def hav(lat1, lon1, lat2, lon2):
    p = np.pi / 180
    d = np.sin((lat2 - lat1) * p / 2) ** 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * np.sin((lon2 - lon1) * p / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(d))


out, rows = {}, []
for case, f in TRAIN.items():
    tr = pd.read_csv(f); tl, tn = tr.latitude.values, tr.longitude.values
    dist = np.array([hav(la, lo, tl, tn).min() for la, lo in zip(lat, lon)])
    box = lambda la, lo: np.floor(la / 0.25).astype(int) * 10000 + np.floor(lo / 0.25).astype(int)
    shared = np.isin(box(lat, lon), box(tl, tn))
    same_cell = np.isin(np.round(lat, 5) * 1000 + np.round(lon, 5), np.round(tl, 5) * 1000 + np.round(tn, 5))
    res = {"n_train": len(tr), "n_test": len(cells), "test cells identical to a training cell": int(same_cell.sum()),
           "distance_km_quantiles(5,25,50,75,95)": np.percentile(dist, [5, 25, 50, 75, 95]).round(1).tolist(),
           "% test cells sharing an ERA5 0.25deg box with a training cell": float(shared.mean() * 100)}
    for var in ("ET", "SM", "Q"):
        d = pc[(pc.case == case) & (pc.variable == var)].set_index("cell").loc[cells]
        for met in ("KGE", "NSE"):
            v = d[met].values; ok = np.isfinite(v)
            rho, p = spearmanr(dist[ok], v[ok])
            u = mannwhitneyu(v[ok & shared], v[ok & ~shared])
            res[f"{var} {met}"] = {"spearman_rho_distance": float(rho), "p": float(p),
                                   "median shared ERA5 box": float(np.median(v[ok & shared])),
                                   "median not shared": float(np.median(v[ok & ~shared])), "mannwhitney_p": float(u.pvalue),
                                   "median by distance bin (km)": {l: float(np.median(v[ok & (dist >= lo) & (dist < hi)]))
                                                                   for l, lo, hi in zip(LABELS, BINS[:-1], BINS[1:])}}
    res["n cells by distance bin"] = {l: int(((dist >= lo) & (dist < hi)).sum()) for l, lo, hi in zip(LABELS, BINS[:-1], BINS[1:])}
    out[case] = res
    rows += [{"case": case, "cell": c, "dist_km": dd, "shared_era5_box": s} for c, dd, s in zip(cells, dist, shared)]
json.dump(out, open(os.path.join(HERE, "distance_skill.json"), "w"), indent=2)
pd.DataFrame(rows).to_csv(os.path.join(HERE, "distance_per_cell.csv"), index=False)
print(json.dumps(out, indent=1))
