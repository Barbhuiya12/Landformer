"""Figure 9: closed-loop behaviour of LandFormer-AR-All.
(a-c) RMSE against lead time for closed loop, open loop, LandFormer-NAR and initial-state persistence; the shaded band
marks lead times at which the closed loop has lower RMSE than LandFormer-NAR.
(d) cumulative bias of evapotranspiration and total runoff and (e) soil-moisture bias over the continuous closed-loop
run (median and interquartile range across test cells; dashed lines: open-loop median).
Inputs: outputs of analysis/closed_loop/ (error_growth.csv, error_growth_info.json, cumulative_bias.npz, sm_bias.npz).
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "analysis", "closed_loop")
eg = pd.read_csv(os.path.join(SRC, "error_growth.csv"))
cross = json.load(open(os.path.join(SRC, "error_growth_info.json")))["nar_crossover_lead_day"]
cb = np.load(os.path.join(SRC, "cumulative_bias.npz"))                  # rows: 25th, 50th, 75th percentile

INK, GRID = "#2b2b2b", "#e6e6e6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "text.color": INK, "axes.labelcolor": INK,
    "axes.edgecolor": "#8a8a8a", "axes.linewidth": 0.6, "xtick.color": "#555555", "ytick.color": "#555555",
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "legend.frameon": False})
C = {"closed": "#d1495b", "open": "#00798c", "nar": "#3d405b", "persistence": "#9aa5b1"}
LAB = {"closed": "Closed loop", "open": "Open loop", "nar": "LandFormer-NAR", "persistence": "Initial-state persistence"}
STY = {"closed": dict(lw=1.8, zorder=5), "open": dict(lw=1.1, zorder=4), "nar": dict(lw=1.1, ls=(0, (5, 2)), zorder=3),
       "persistence": dict(lw=0.9, ls=(0, (1, 1.5)), zorder=2)}
BAND = "#d1495b"
PANELS = [("ET", "Evapotranspiration", "RMSE (mm day$^{-1}$)"),
          ("SM", "Soil moisture", "RMSE (m$^{3}$ m$^{-3}$)"),
          ("Q", "Total Runoff", "RMSE (mm day$^{-1}$)")]


def tidy(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


fig = plt.figure(figsize=(7.4, 5.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.95], hspace=0.55, wspace=0.42)
for i, (v, title, ylab) in enumerate(PANELS):
    ax = fig.add_subplot(gs[0, i]); tidy(ax)
    d = eg[eg.variable == v]
    x = cross[v]
    ax.axvspan(0, x, color=BAND, alpha=0.07, lw=0)
    for k in ("persistence", "nar", "open", "closed"):
        ax.plot(d.lead_day, d[f"rmse_{k}"], color=C[k], label=LAB[k], **STY[k])
    ax.axvline(x, color=BAND, lw=0.8, ls=(0, (3, 2)), alpha=0.9)
    ax.text(x, 1.02, f"{x} d", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=7, color=BAND, fontweight="bold")
    ax.set_xlim(0, 365); ax.set_xticks([0, 90, 180, 270, 365])
    ax.set_title(f"$\\bf{{{'abc'[i]}}}$  {title}", loc="left", fontsize=8.5, pad=12)
    ax.set_xlabel("Lead time (days)"); ax.set_ylabel(ylab)

h, l = ax.get_legend_handles_labels()
order = [3, 2, 1, 0]
band = plt.Rectangle((0, 0), 1, 1, color=BAND, alpha=0.12, lw=0)
fig.legend([h[j] for j in order] + [band], [l[j] for j in order] + ["Closed loop better than NAR"],
           loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.01), fontsize=7.2, handlelength=2.2, columnspacing=1.4)

smb = np.load(os.path.join(SRC, "sm_bias.npz"))                       # rows: 25th, 50th, 75th percentile
dates = pd.date_range("2001-09-27", periods=cb["closed_loop_ET"].shape[1], freq="D")
YEARS = mdates.YearLocator(); YFMT = mdates.DateFormatter("%Y")
KEY = ([plt.Line2D([], [], color=INK, lw=1.8), plt.Rectangle((0, 0), 1, 1, color=INK, alpha=0.18, lw=0),
        plt.Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 2)))],
       ["Closed loop, median", "Closed loop, IQR", "Open loop, median"])

# (d) cumulative bias of the fluxes
bot = gs[1, :].subgridspec(1, 3, wspace=0.62)
ax = fig.add_subplot(bot[0, :2]); tidy(ax)
for v, name, col, va in [("Q", "Total Runoff", "#30638e", "bottom"), ("ET", "Evapotranspiration", "#e09f3e", "top")]:
    q25, q50, q75 = cb[f"closed_loop_{v}"]
    ax.fill_between(dates, q25, q75, color=col, alpha=0.18, lw=0)
    ax.plot(dates, q50, color=col, lw=1.8)
    ax.plot(dates, cb[f"open_loop_{v}"][1], color=col, lw=1.0, ls=(0, (4, 2)), alpha=0.9)
    k = len(dates) - 330
    ax.annotate(name, xy=(dates[k], q50[k]), xytext=(0, 5 if va == "bottom" else -5), textcoords="offset points",
                ha="center", va=va, fontsize=7.5, color=col, fontweight="bold")
ax.axhline(0, color="#8a8a8a", lw=0.6, zorder=1)
ax.set_xlim(dates[0], dates[-1]); ax.xaxis.set_major_locator(YEARS); ax.xaxis.set_major_formatter(YFMT)
ax.set_ylabel("Cumulative bias (mm)")
ax.set_title("$\\bf{d}$  Cumulative bias of the fluxes", loc="left", fontsize=8.5)
ax.legend(*KEY, loc="upper left", fontsize=7, ncol=1, handlelength=2.0)

# (e) soil-moisture bias (a storage: daily bias, not cumulative)
ax = fig.add_subplot(bot[0, 2]); tidy(ax)
col = "#7b5ea7"
q25, q50, q75 = smb["closed_loop"]
ax.fill_between(dates, q25, q75, color=col, alpha=0.18, lw=0)
ax.plot(dates, q50, color=col, lw=1.4)
ax.plot(dates, smb["open_loop"][1], color=col, lw=0.9, ls=(0, (4, 2)), alpha=0.9)
ax.axhline(0, color="#8a8a8a", lw=0.6, zorder=1)
ax.set_xlim(dates[0], dates[-1]); ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(YFMT)
ax.set_ylabel("Bias (m$^{3}$ m$^{-3}$)")
ax.set_title("$\\bf{e}$  Soil-moisture bias", loc="left", fontsize=8.5)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig09_closed_loop.{ext}"), dpi=300, bbox_inches="tight")
print("saved fig09_closed_loop")
