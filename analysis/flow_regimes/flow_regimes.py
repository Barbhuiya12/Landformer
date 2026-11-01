"""Total-runoff skill across flow regimes, for all test cells and for the Indus, Ganges and Brahmaputra (Section 4.8).

Input: the daily test-period predictions written by analysis/closed_loop/closed_loop.py (step1) for
LandFormer-AR-All with VIC lagged inputs, LandFormer-NAR and LandFormer-AR-All in closed loop.

Basins (test cells whose centre lies inside the polygon):
  Indus        HydroBASINS v1c level 3, main basin 4030033640 (Lehner & Grill, 2013), within the model domain
  Ganges       catchment of the Hardinge Bridge gauge
  Brahmaputra  catchment of the Bahadurabad gauge

Metrics, per cell (median over cells) and for the basin-mean series (mean over the test cells in a basin; the series
are unrouted and are not discharge at a gauge):
  KGE, NSE
  FHV   volume bias of the highest 2 % of the flow-duration curve (Yilmaz et al., 2008)
  FLV   volume bias of the lowest 30 % of flows (computed on flows, as both series contain zeros)
  KGE and volume bias in the monsoon (June-September) and in the remaining months
  relative error and timing error of the calendar-year maximum (2002-2006)
  bias of the drought deficit volume below the 20th-percentile VIC flow of each cell
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..')))
from landformer_paths import ROOT, VICFLOW_DIR  # noqa: E402
import os, json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import vectorized

HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.join(HERE, "..", "closed_loop")
GAUGES = f"{VICFLOW_DIR}/Data/target/Discharge/final_basins/all_basins.shp"
HYBAS = os.path.join(HERE, "hydrobasins", "hybas_as_lev03_v1c.shp")
Q = 2                                                       # target index of total runoff
cells = json.load(open(os.path.join(CL, "step1_info.json")))["cells"]
arr = np.load(os.path.join(CL, "cache", "test_arrays.npz"))
lat, lon = arr["lat"][cells], arr["lon"][cells]
obs = np.load(os.path.join(CL, "vic_obs.npy"), mmap_mode="r")[:, :, Q].astype(np.float64)
runs = {k: np.load(os.path.join(CL, f"pred_{k}.npy"), mmap_mode="r")[:, :, Q].astype(np.float64)
        for k in ("open_loop", "nar", "closed_loop")}
dates = pd.date_range("2001-09-27", periods=obs.shape[0], freq="D")
monsoon = dates.month.isin([6, 7, 8, 9])

g = gpd.read_file(GAUGES)
h = gpd.read_file(HYBAS)
polys = {"Indus": h[h.MAIN_BAS == 4030033640].geometry.union_all(),
         "Ganges": g[g.clean_id.astype(str) == "219"].geometry.union_all(),            # Hardinge Bridge gauge
         "Brahmaputra": g[g.clean_id.astype(str) == "221"].geometry.union_all()}
groups = {"All test cells": np.ones(len(cells), bool)}
for n, p in polys.items():
    groups[n] = vectorized.contains(p, lon, lat)


def kge(o, s):
    if o.std() == 0 or s.std() == 0:
        return np.nan
    r = np.corrcoef(o, s)[0, 1]
    return 1 - np.sqrt((r - 1) ** 2 + (s.std() / o.std() - 1) ** 2 + (s.mean() / o.mean() - 1) ** 2)


def nse(o, s):
    v = ((o - o.mean()) ** 2).sum()
    return 1 - ((s - o) ** 2).sum() / v if v > 0 else np.nan


def pbias(o, s):
    return 100 * (s.sum() - o.sum()) / o.sum() if o.sum() > 0 else np.nan


def fdc_segment(o, s, lo, hi):
    """% bias of the volume between exceedance fractions lo..hi of the two flow-duration curves."""
    os_, ss = np.sort(o)[::-1], np.sort(s)[::-1]
    n = len(o); a, b = int(np.floor(lo * n)), int(np.ceil(hi * n))
    return pbias(os_[a:b], ss[a:b])


def metrics(o, s):
    m = {"KGE": kge(o, s), "NSE": nse(o, s),
         "FHV_%": fdc_segment(o, s, 0.0, 0.02), "FLV_%": fdc_segment(o, s, 0.70, 1.0),
         "KGE_monsoon": kge(o[monsoon], s[monsoon]), "bias_monsoon_%": pbias(o[monsoon], s[monsoon]),
         "KGE_nonmonsoon": kge(o[~monsoon], s[~monsoon]), "bias_nonmonsoon_%": pbias(o[~monsoon], s[~monsoon])}
    rel, tim = [], []
    for y in range(2002, 2007):
        sel = dates.year == y
        oy, sy = o[sel], s[sel]
        if oy.max() > 0:
            rel.append(100 * (sy.max() - oy.max()) / oy.max())
            tim.append(abs(int(sy.argmax()) - int(oy.argmax())))
    m["annual_max_rel_error_%"] = float(np.median(rel)) if rel else np.nan
    m["annual_max_timing_error_d"] = float(np.median(tim)) if tim else np.nan
    thr = np.percentile(o, 20)
    do, ds = np.maximum(thr - o, 0).sum(), np.maximum(thr - s, 0).sum()
    m["drought_deficit_bias_%"] = 100 * (ds - do) / do if do > 0 else np.nan
    return m


rows = []
for name, s_all in runs.items():
    for j in range(len(cells)):
        rows.append({"model": name, "cell": cells[j], **metrics(obs[:, j], s_all[:, j])})
per = pd.DataFrame(rows)
for gname, m in groups.items():
    per[gname] = np.tile(m, len(runs))
per.to_csv(os.path.join(HERE, "flow_regime_per_cell.csv"), index=False)

MET = [c for c in per.columns if c not in ("model", "cell", *groups)]
summ, basin = [], []
for gname, m in groups.items():
    for name, s_all in runs.items():
        d = per[(per.model == name) & per[gname]]
        summ.append({"group": gname, "model": name, "n_cells": int(m.sum()),
                     **{f"median {c}": float(d[c].median()) for c in MET}})
        if gname != "All test cells":
            basin.append({"basin": gname, "model": name, "n_cells": int(m.sum()),
                          **metrics(obs[:, m].mean(1), s_all[:, m].mean(1))})
pd.DataFrame(summ).to_csv(os.path.join(HERE, "flow_regime_summary.csv"), index=False)
pd.DataFrame(basin).to_csv(os.path.join(HERE, "flow_regime_basin_mean.csv"), index=False)
pd.set_option("display.width", 250)
print(pd.DataFrame(summ).round(3).to_string())
print(pd.DataFrame(basin).round(3).to_string())
