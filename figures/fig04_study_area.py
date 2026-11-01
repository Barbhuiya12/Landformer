"""Figure 4: study area.
(a) Koppen-Geiger climate classes with the major rivers (Natural Earth 1:10m centrelines) and the outlines of the
    Indus, Ganges and Brahmaputra basins (HydroBASINS v1c);
(b) mean annual precipitation, (c) mean daily maximum and (d) minimum temperature from the ERA5 forcing, 1990-2006;
(e) elevation (GMTED2010).
All panels use the 0.0625 deg grid prepared by fig04_data/fig4_grid.py.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')))
from landformer_paths import ROOT, VICFLOW_DIR  # noqa: E402
import os
import numpy as np, geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from rasterio import features
from rasterio.transform import from_origin
from shapely.geometry import shape
from shapely.ops import unary_union

H = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(H, "fig04_data")
B = f"{VICFLOW_DIR}"
HB = os.path.join(H, "..", "analysis", "flow_regimes", "hydrobasins")

# ---------------------------------------------------------------- data
g = np.load(os.path.join(FD, "fig4_grid.npz"))
lat, lon, FP = g["lat"], g["lon"], g["footprint"]
EXT = [lon.min() - 0.03125, lon.max() + 0.03125, lat.min() - 0.03125, lat.max() + 0.03125]
tr = from_origin(EXT[0], EXT[3], 0.0625, 0.0625)
_fp = FP[::-1].astype("uint8")
ISC = unary_union([shape(s) for s, v in features.shapes(_fp, mask=_fp == 1, transform=tr) if v == 1]).buffer(0)
INNER = ISC.buffer(-0.12)
country = gpd.read_file(f"{B}/Data_ext/shape_file/Counry boundary/isc_country.shp").to_crs(4326)
borders = gpd.GeoSeries([b.intersection(INNER) for b in country.boundary], crs=4326)
# national territory outside the model domain (western Balochistan, west of 66.1 E): shown in grey, not coloured
OUTSIDE = gpd.GeoSeries([country.geometry.union_all().difference(ISC)], crs=4326)
OUT_EDGE = gpd.GeoSeries([b.difference(ISC.buffer(0.12)) for b in country.boundary], crs=4326)
MAJOR = ["Indus", "Ganges", "Brahmaputra", "Dihang", "Yarlung", "Narmada", "Tapi", "Mahäna Nadï", "Godävari",
         "Krishna", "Cauvery", "Kolidam"]
riv = gpd.read_file(os.path.join(FD, "naturalearth", "ne_10m_rivers_lake_centerlines.shp"))
riv = riv[riv.name.isin(MAJOR)].clip(ISC)
h3 = gpd.read_file(os.path.join(HB, "hybas_as_lev03_v1c.shp")); h4 = gpd.read_file(os.path.join(HB, "hybas_as_lev04_v1c.shp"))
basins = [h3[h3.MAIN_BAS == 4030033640].geometry.union_all(),                                   # Indus
          h4[h4.PFAF_ID.isin([4524, 4523, 4521])].geometry.union_all(),                         # Ganges
          h4[h4.PFAF_ID.isin([4522, 4525, 4526, 4527, 4528, 4529])].geometry.union_all()]       # Brahmaputra + Meghna
from shapely.ops import linemerge
def _long_parts(geom, min_len=0.4):          # drop tiny fragments (they render as dots along the domain edge)
    parts = getattr(linemerge(geom) if geom.geom_type == "MultiLineString" else geom, "geoms", None)
    parts = list(parts) if parts is not None else [geom]
    from shapely.geometry import MultiLineString
    return MultiLineString([p for p in parts if p.geom_type == "LineString" and p.length >= min_len])
# full basin outline within the study area: the basin polygon is smoothed slightly and clipped to the coloured footprint,
# so the outline is continuous (also where it follows the domain edge) and has no fragments
basin_lines = gpd.GeoSeries([_long_parts(b.buffer(0.05).buffer(-0.05).intersection(ISC.buffer(0.03)).boundary, 0.4)
                             for b in basins], crs=4326)

# ---------------------------------------------------------------- style
INK, OCEAN, BORDER, RIVER, GREY = "#222222", "white", "#5a5a5a", "#1f5fa8", "#dcdcdc"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": "#9a9a9a", "axes.linewidth": 0.6,
                     "xtick.color": "#555555", "ytick.color": "#555555", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "axes.titlesize": 9.5})
KCOL = {1: (0, 0, 255), 2: (0, 120, 255), 3: (70, 170, 250), 4: (255, 0, 0), 5: (255, 150, 150), 6: (245, 165, 0),
        7: (255, 220, 100), 8: (255, 255, 0), 9: (200, 200, 0), 10: (150, 150, 0), 11: (150, 255, 150), 12: (100, 200, 100),
        13: (50, 150, 50), 14: (200, 255, 80), 15: (100, 255, 80), 16: (50, 200, 0), 17: (255, 0, 255), 18: (200, 0, 200),
        19: (150, 50, 150), 20: (150, 100, 150), 21: (170, 175, 255), 22: (90, 120, 220), 23: (75, 80, 180), 24: (50, 0, 135),
        25: (0, 255, 255), 26: (55, 200, 255), 27: (0, 125, 125), 28: (0, 70, 95), 29: (178, 178, 178), 30: (102, 102, 102)}
KNAME = dict(zip(range(1, 31), ["Af", "Am", "Aw", "BWh", "BWk", "BSh", "BSk", "Csa", "Csb", "Csc", "Cwa", "Cwb", "Cwc",
                                "Cfa", "Cfb", "Cfc", "Dsa", "Dsb", "Dsc", "Dsd", "Dwa", "Dwb", "Dwc", "Dwd", "Dfa", "Dfb",
                                "Dfc", "Dfd", "ET", "EF"]))
KG = g["koppen"]; present = [k for k in range(1, 31) if np.any(KG == k)]
kcmap = ListedColormap([np.array(KCOL[k]) / 255 for k in range(1, 31)]); knorm = BoundaryNorm(np.arange(0.5, 31.5), 30)
TERRAIN = LinearSegmentedColormap.from_list("land", plt.cm.terrain(np.linspace(0.25, 1.0, 256)))
XL, YL = (59.5, 98.0), (7.6, 38.0)
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]
LABELS = {"Indus": (69.6, 26.9, -62), "Ganges": (84.2, 25.75, -12), "Brahmaputra": (91.2, 26.55, 8),
          "Narmada": (77.6, 23.25, -4), "Tapi": (76.4, 20.75, 4), "Mahanadi": (83.0, 20.95, 18),
          "Godavari": (79.7, 18.9, -28), "Krishna": (77.6, 16.55, -10), "Kaveri": (77.2, 11.55, -25)}


def base(ax, lw, xl, xlabel, ylabel, overlays=True):
    ax.set_facecolor(OCEAN)
    if overlays:                                                          # rivers and borders only in panel (a)
        riv.plot(ax=ax, color=RIVER, linewidth=lw, zorder=4)
        country.boundary.plot(ax=ax, color="black", linewidth=0.8, zorder=6)
    ax.set_xlim(*xl); ax.set_ylim(*YL); ax.set_aspect("equal")
    ax.set_xticks([t for t in (60, 65, 70, 75, 80, 85, 90, 95) if xl[0] <= t <= xl[1]] if xl[0] < 62 else [70, 80, 90])
    ax.set_yticks([10, 15, 20, 25, 30, 35])
    ax.tick_params(labelsize=10)
    ax.set_xlabel("Longitude (°E)" if xlabel else "", fontsize=11)
    ax.set_ylabel("Latitude (°N)" if ylabel else "", fontsize=11)
    ax.grid(False)


plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})
fig = plt.figure(figsize=(15.5, 7.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1], left=0.05, right=0.97, top=0.95, bottom=0.17,
                      wspace=0.32, hspace=0.24)

# (a)
ax = fig.add_subplot(gs[:, 0])
ax.imshow(KG, extent=EXT, origin="lower", cmap=kcmap, norm=knorm, interpolation="nearest", zorder=1)
base(ax, lw=1.4, xl=XL, xlabel=True, ylabel=True)
basin_lines.plot(ax=ax, color=INK, linewidth=1.3, linestyle=(0, (5, 2.2)), zorder=6)
for name, (x, y, rot) in LABELS.items():
    ax.text(x, y, name, fontsize=9, style="italic", color=RIVER, rotation=rot, ha="center", va="center",
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")], zorder=7)
ax.set_title("(a) Köppen–Geiger Climate Classification", pad=8)
ax.annotate("N", xy=(61.4, 37.4), xytext=(61.4, 35.2), ha="center", va="center", fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.3), zorder=8)
ax.legend(handles=[Line2D([], [], color=RIVER, lw=1.6, label="Major river"),
                   Line2D([], [], color=INK, lw=1.3, ls=(0, (5, 2.2)), label="Major basin"),
                   Line2D([], [], color="black", lw=1.0, label="National border")],
          loc="lower left", fontsize=9, frameon=True, framealpha=0.92, edgecolor="none", borderpad=0.5)
pos = ax.get_position()
fig.legend(handles=[Patch(facecolor=np.array(KCOL[k]) / 255, edgecolor="#555555", linewidth=0.4, label=KNAME[k]) for k in present],
           ncol=7, loc="upper center", bbox_to_anchor=((pos.x0 + pos.x1) / 2, pos.y0 - 0.075), frameon=False, fontsize=9,
           handlelength=1.4, handleheight=1.0, columnspacing=1.1, labelspacing=0.35, title="Climate Zone", title_fontsize=10)

# (b)-(e)
XR = (EXT[0], 98.0)
PAN = [("(b) Mean Annual Precipitation", g["prcp"], "YlGnBu", (0, 3000), "Precipitation (mm)", gs[0, 1], False, True),
       ("(c) Mean Maximum Temperature", g["tmax"], "RdYlBu_r", (5, 40), "Tmax (°C)", gs[0, 2], False, False),
       ("(d) Mean Minimum Temperature", g["tmin"], "RdYlBu_r", (-15, 30), "Tmin (°C)", gs[1, 1], True, True),
       ("(e) Elevation", g["elev"], TERRAIN, (0, 6000), "Elevation (m)", gs[1, 2], True, False)]
from mpl_toolkits.axes_grid1 import make_axes_locatable
for name, Z, cmap, (lo, hi), label, slot, xlab, ylab in PAN:
    ax = fig.add_subplot(slot)
    im = ax.imshow(Z, extent=EXT, origin="lower", cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest", zorder=1)
    base(ax, lw=0.8, xl=XR, xlabel=xlab, ylabel=ylab, overlays=False)
    ax.set_title(name, pad=6)
    cax = make_axes_locatable(ax).append_axes("right", size="4.5%", pad=0.08)
    cb = fig.colorbar(im, cax=cax, extend="both")
    cb.set_label(label, fontsize=11); cb.ax.tick_params(labelsize=10)

for e in ("png", "pdf", "tif"):
    fig.savefig(os.path.join(H, f"fig04_study_area.{e}"), dpi=600 if e == "tif" else 300, bbox_inches="tight", facecolor="white")
print("saved fig04_study_area")
