"""Figure 13: flow-duration curves and mean annual cycles of basin-mean total runoff for the Indus, Ganges and
Brahmaputra (VIC, LandFormer-AR-All and LandFormer-NAR; mean over the test cells in each basin, unrouted).
Basins as in flow_regimes.py.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '../..')))
from landformer_paths import ROOT, VICFLOW_DIR  # noqa: E402
import os, json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import vectorized
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
CL = os.path.join(HERE, "..", "closed_loop")
GAUGES = f"{VICFLOW_DIR}/Data/target/Discharge/final_basins/all_basins.shp"
cells = json.load(open(os.path.join(CL, "step1_info.json")))["cells"]
arr = np.load(os.path.join(CL, "cache", "test_arrays.npz"))
lat, lon = arr["lat"][cells], arr["lon"][cells]
obs = np.load(os.path.join(CL, "vic_obs.npy"), mmap_mode="r")[:, :, 2]
runs = {"open_loop": np.load(os.path.join(CL, "pred_open_loop.npy"), mmap_mode="r")[:, :, 2],
        "nar": np.load(os.path.join(CL, "pred_nar.npy"), mmap_mode="r")[:, :, 2]}
dates = pd.date_range("2001-09-27", periods=obs.shape[0], freq="D")
g = gpd.read_file(GAUGES); h = gpd.read_file(os.path.join(HERE, "hydrobasins", "hybas_as_lev03_v1c.shp"))
polys = {"Indus": h[h.MAIN_BAS == 4030033640].geometry.union_all(),
         "Ganges": g[g.clean_id.astype(str) == "219"].geometry.union_all(),
         "Brahmaputra": g[g.clean_id.astype(str) == "221"].geometry.union_all()}

INK = "#2b2b2b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": "#8a8a8a", "axes.linewidth": 0.6,
                     "xtick.color": "#555555", "ytick.color": "#555555", "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False})
SER = [("VIC", None, "#2b2b2b", dict(lw=1.6)),
       ("LandFormer-AR-All", "open_loop", "#d1495b", dict(lw=1.2)),
       ("LandFormer-NAR", "nar", "#3d405b", dict(lw=1.1, ls=(0, (5, 2))))]
fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.6), gridspec_kw=dict(hspace=0.5, wspace=0.32))
for j, (name, poly) in enumerate(polys.items()):
    m = vectorized.contains(poly, lon, lat)
    series = {lab: (obs if key is None else runs[key])[:, m].mean(1) for lab, key, _, _ in SER}
    ax = axs[0, j]
    for lab, key, col, sty in SER:
        s = np.sort(series[lab])[::-1]
        ax.plot(np.arange(1, len(s) + 1) / len(s) * 100, s, color=col, label=lab, **sty)
    ax.set_yscale("log"); ax.set_xlim(0, 100)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_title(f"$\\bf{{{'abc'[j]}}}$  {name}", loc="left", fontsize=8.5)
    ax.set_xlabel("Exceedance probability (%)")
    if j == 0: ax.set_ylabel("Basin-mean runoff (mm day$^{-1}$)")
    ax = axs[1, j]
    for lab, key, col, sty in SER:
        clim = pd.Series(series[lab], index=dates).groupby(dates.month).mean()
        ax.plot(clim.index, clim.values, color=col, marker="o", ms=2.2, **sty)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(list("JFMAMJJASOND")); ax.set_xlim(0.5, 12.5)
    ax.axvspan(5.5, 9.5, color="#00798c", alpha=0.07, lw=0)
    ax.set_title(f"$\\bf{{{'def'[j]}}}$  {name}, mean annual cycle", loc="left", fontsize=8.5)
    if j == 0: ax.set_ylabel("Basin-mean runoff (mm day$^{-1}$)")
    for a in axs[:, j]:
        a.spines[["top", "right"]].set_visible(False)
h_, l_ = axs[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.02), fontsize=7.5)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig13_basin_runoff.{ext}"), dpi=300, bbox_inches="tight")
print("saved")
