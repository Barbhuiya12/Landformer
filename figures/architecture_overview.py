"""Schematic overview of LandFormer (inputs, temporal encoder, spatial graph attention, dual-path regression heads).
Drawn with symbols only; the dimensions are given in the caption.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.family": "Nimbus Sans", "font.size": 8, "pdf.fonttype": 42, "svg.fonttype": "none"})

INK, MUTED = "#22303c", "#6b7785"
GREY, GREY_E = "#f1f3f5", "#8a96a3"
BLUES = ["#e3eefb", "#cfe0f6", "#b7d0f0", "#9cbde8"]
BLUE_E = "#3a6ea5"
FRAME = "#f4f8fd"
WMSA, WMSA_E = "#fde9d9", "#d9822b"
SWMSA, SWMSA_E = "#ece3f5", "#8a5cb8"
GAT, GAT_E = "#e2f3e8", "#2f8a57"
HEADS = [("ET", "#fde2e2", "#c0504d"), ("SM", "#e0ecfa", "#3a6ea5"), ("Q", "#e3f3e6", "#3c8d4f")]


def rbox(ax, x, y, w, h, fc, ec, lw=0.9, r=0.6, z=2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.9, ms=7, z=3, rad=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=z, connectionstyle=f"arc3,rad={rad}"))


def txt(ax, x, y, s, size=8, color=INK, weight="normal", ha="center", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va, zorder=5, **kw)


fig = plt.figure(figsize=(7.2, 3.55))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 144); ax.set_ylim(0, 71); ax.axis("off")

# ------------------------------------------------------------------ (a)
txt(ax, 1.5, 68.5, "(a)", 9, weight="bold", ha="left")
yc = 50  # centre line of the pipeline

# Input: stacked time series
rbox(ax, 2, 38, 15, 24, GREY, GREY_E)
txt(ax, 9.5, 58.6, "Input", 8.5, weight="bold")
t = np.linspace(0, 1, 60)
for k, (amp, ph, c) in enumerate([(1.2, 0.0, "#4c78a8"), (0.9, 1.3, "#f58518"), (1.0, 2.4, "#54a24b")]):
    yy = 56.2 - k * 2.0 + amp * np.sin(2 * np.pi * (2.2 * t) + ph) * 0.45
    ax.plot(4 + 11 * t, yy, color=c, lw=0.9, zorder=4)
txt(ax, 9.5, 50.4, "forcings + parameters", 5.3, color=MUTED)
# lagged outputs: a separate dashed series (previous-step values, not a feedback loop)
yy = 47.6 + 0.45 * np.sin(2 * np.pi * (1.6 * t) + 0.7)
ax.plot(4 + 11 * t, yy, color="#8a5cb8", lw=0.9, ls=(0, (2.2, 1.4)), zorder=4)
txt(ax, 9.5, 44.4, "lagged outputs y$_{t-1}$\n(AR variants)", 5.3, color="#8a5cb8", linespacing=1.1)
txt(ax, 9.5, 40.6, "one grid cell, T steps", 5.5, color=MUTED)

arrow(ax, 17, yc, 20, yc)

# Patch embedding: tokens
rbox(ax, 20, 38, 15, 24, GREY, GREY_E)
txt(ax, 27.5, 58.6, "Patch\nembedding", 8.5, weight="bold", linespacing=1.05)
cols = ["#9ecae1", "#6baed6", "#4292c6", "#fdae6b", "#fd8d3c", "#a1d99b", "#74c476"]
for k, c in enumerate(cols):
    ax.add_patch(Rectangle((22.0 + k * 1.57, 48.0), 1.45, 3.2, fc=c, ec="white", lw=0.4, zorder=4))
txt(ax, 27.5, 42.3, "1D conv over\npatches of P steps\n→ N tokens", 5.9, color=MUTED, linespacing=1.15)

arrow(ax, 35, yc, 38, yc)

# Encoder frame with 4 stages
fx, fw = 38, 62
rbox(ax, fx, 35.5, fw, 30, FRAME, BLUE_E, lw=0.8, r=1.0, z=1)
txt(ax, fx + fw / 2, 63.2, "Temporal Swin encoder", 8.2, weight="bold", color=BLUE_E)
sw, gap = 12.0, 3.0
sx = [fx + 2.0 + i * (sw + gap) for i in range(4)]
for i, x in enumerate(sx):
    rbox(ax, x, 39.5, sw, 21, BLUES[i], BLUE_E)
    txt(ax, x + sw / 2, 57.0, f"Stage {i + 1}", 8, weight="bold")
    # two mini blocks: W-MSA / SW-MSA
    rbox(ax, x + 1.5, 49.2, sw - 3, 3.6, WMSA, WMSA_E, lw=0.6, r=0.4, z=3)
    txt(ax, x + sw / 2, 51.0, "W-MSA", 6.0)
    rbox(ax, x + 1.5, 44.9, sw - 3, 3.6, SWMSA, SWMSA_E, lw=0.6, r=0.4, z=3)
    txt(ax, x + sw / 2, 46.7, "SW-MSA", 6.0)
    txt(ax, x + sw / 2, 41.8, f"d$_{i + 1}$ Swin blocks", 5.9, color=INK)
    if i < 3:
        arrow(ax, x + sw, yc, x + sw + gap, yc, ms=6)
# patch merging note under the stages
txt(ax, fx + fw / 2, 37.4, "patch merging between stages: fewer tokens, wider features", 6.2, color=MUTED, style="italic")

arrow(ax, fx + fw, yc, fx + fw + 3, yc)

# Pooling
px = fx + fw + 3
rbox(ax, px, 41, 10, 18, GREY, GREY_E)
txt(ax, px + 5, 54.8, "Pooling", 8, weight="bold")
txt(ax, px + 5, 47.0, "LayerNorm\n+ average\npooling", 6.3, color=MUTED, linespacing=1.15)

arrow(ax, px + 10, yc, px + 13, yc)

# Spatial GAT
gx = px + 13
rbox(ax, gx, 38, 15, 24, GAT, GAT_E)
txt(ax, gx + 7.5, 58.6, "Spatial GAT", 8.0, weight="bold", color=GAT_E)
cxy = (gx + 7.5, 51.0)
nbr = [(gx + 3.4, 54.0), (gx + 11.8, 54.3), (gx + 3.0, 48.3), (gx + 11.2, 47.6)]
for (nx, ny) in nbr:
    ax.plot([cxy[0], nx], [cxy[1], ny], color=GAT_E, lw=0.8, zorder=4)
for (nx, ny) in nbr:
    ax.add_patch(Circle((nx, ny), 0.9, fc="white", ec=GAT_E, lw=0.8, zorder=5))
ax.add_patch(Circle(cxy, 1.15, fc=GAT_E, ec=GAT_E, lw=0.8, zorder=5))
txt(ax, gx + 7.5, 41.6, "neighbouring\ncells, same\ntime step", 5.9, color=MUTED, linespacing=1.15)

# Heads
hx = gx + 18.0
for k, (name, fc, ec) in enumerate(HEADS):
    hy = 55.5 - k * 7.5
    arrow(ax, gx + 15, yc, hx, hy + 2.6, ms=6, rad=0.0)
    rbox(ax, hx, hy, 9.0, 5.2, fc, ec)
    txt(ax, hx + 4.5, hy + 2.6, f"{name} head", 6.8, weight="bold", color=ec)

# ------------------------------------------------------------------ (b)
txt(ax, 1.5, 31.0, "(b)", 9, weight="bold", ha="left")
txt(ax, 6.0, 31.0, "Inside a stage: two consecutive Swin blocks", 8.2, ha="left")
# dashed link from Stage 2 to panel b
ax.plot([sx[1] + sw / 2, sx[1] + sw / 2], [35.5, 33.0], color=BLUE_E, lw=0.7, ls=(0, (2, 2)), zorder=1)

y0, h = 13.5, 6.0
seq = [("in", None), ("LN", None), ("W-MSA", (WMSA, WMSA_E)), ("+", None), ("LN", None), ("MLP", None), ("+", None),
       ("LN", None), ("SW-MSA", (SWMSA, SWMSA_E)), ("+", None), ("LN", None), ("MLP", None), ("+", None), ("out", None)]
W = {"in": 4.5, "out": 4.5, "+": 2.6, "LN": 9.5, "MLP": 7.0, "W-MSA": 9.0, "SW-MSA": 9.5}
gapb = 1.8
total = sum(W[k] for k, _ in seq) + gapb * (len(seq) - 1)
x = (144 - total) / 2
pos = []
for k, col in seq:
    w = W[k]
    if k == "+":
        ax.add_patch(Circle((x + w / 2, y0 + h / 2), 1.3, fc="white", ec=INK, lw=0.8, zorder=4))
        txt(ax, x + w / 2, y0 + h / 2 + 0.05, "+", 8.5)
    elif k in ("in", "out"):
        txt(ax, x + w / 2, y0 + h / 2, "$z^{l-1}$" if k == "in" else "$z^{l+1}$", 8.5)
    else:
        fc, ec = col if col else ("#ffffff", INK)
        rbox(ax, x, y0, w, h, fc, ec, lw=0.8, r=0.5)
        txt(ax, x + w / 2, y0 + h / 2, {"LN": "LayerNorm"}.get(k, k), 6.2)
    pos.append((x, w))
    x += w + gapb
for i in range(len(seq) - 1):
    (a, wa), (b, _) = pos[i], pos[i + 1]
    arrow(ax, a + wa, y0 + h / 2, b, y0 + h / 2, ms=5)
# residual connections
for j, (s0, s1) in enumerate([(0, 3), (3, 6), (6, 9), (9, 12)]):
    xa = pos[s0][0] + pos[s0][1] / 2 + (0.0 if s0 == 0 else 0.0)
    xb = pos[s1][0] + pos[s1][1] / 2
    top = y0 + h + 2.6
    ystart = y0 + h / 2 + (2.2 if s0 == 0 else 1.3)
    ax.plot([xa, xa, xb], [ystart, top, top], color=MUTED, lw=0.7, zorder=2)
    arrow(ax, xb, top, xb, y0 + h / 2 + 1.3, color=MUTED, ms=4, lw=0.7)
xm = pos[6][0] + pos[6][1] + gapb / 2
ax.plot([xm, xm], [y0 - 2.5, y0 + h + 5.0], color=GREY_E, lw=0.6, ls=(0, (2, 2)))
txt(ax, (pos[1][0] + xm) / 2, y0 + h + 4.6, "Swin block $l$ (W-MSA)", 7, color=WMSA_E, weight="bold")
txt(ax, (xm + pos[12][0] + pos[12][1]) / 2, y0 + h + 4.6, "Swin block $l+1$ (SW-MSA)", 7, color=SWMSA_E, weight="bold")

# window sketch under the blocks
def token_row(xc, y, shift, color_a, color_b, label):
    n, tw = 12, 1.35
    x0 = xc - n * tw / 2
    for k in range(n):
        ax.add_patch(Rectangle((x0 + k * tw, y), tw, 1.8, fc="#ffffff", ec=GREY_E, lw=0.4, zorder=3))
    if shift == 0:
        spans = [(0, 6, color_a), (6, 12, color_b)]
    else:
        spans = [(0, 3, color_b), (3, 9, color_a), (9, 12, color_b)]
    for a, b, c in spans:
        ax.add_patch(Rectangle((x0 + a * tw, y - 0.3), (b - a) * tw, 2.4, fc="none", ec=c, lw=1.1, zorder=4))
    txt(ax, x0 - 1.0, y + 0.9, label, 6.2, ha="right", color=MUTED)

token_row(pos[2][0] + pos[2][1] / 2, 5.0, 0, WMSA_E, WMSA_E, "windows of M tokens")
token_row(pos[8][0] + pos[8][1] / 2, 5.0, 1, SWMSA_E, SWMSA_E, "shifted by M/2")
txt(ax, 72, 1.6, "Blocks alternate W-MSA and SW-MSA; a stage is a sequence of blocks; patch merging follows every stage except the last.",
    6.2, color=MUTED, style="italic")

for ext in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"Figure2_simple.{ext}", dpi=600 if ext == "png" else None)
print("saved", OUT / "Figure2_simple.png")
