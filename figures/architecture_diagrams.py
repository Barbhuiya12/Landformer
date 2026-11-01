"""Architecture diagrams of LandFormer: temporal tokenisation and window attention, and the stage-wise structure of the
encoder. Dimensions follow the trained reference model: 270-day input, patch size 10 (27 tokens), window of 14 tokens
with a shift of 7, depths [4, 4, 16, 4], widths 128/256/512/1024 and 4/8/16/32 attention heads.
"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..')))
from landformer_paths import ROOT  # noqa: E402
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from pathlib import Path

OUT = Path(f"{ROOT}/figures/output")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
                     "savefig.dpi": 600, "pdf.fonttype": 42})

INK = "#1f2933"
MUTED = "#6b7280"
GRID = "#d1d5db"
W1 = "#2b6cb0"      # window A (blue)
W1L = "#dbe7f5"
W2 = "#c05621"      # window B (orange)
W2L = "#fbe3d3"
WR = "#7b3fa0"      # wrapped window (purple)
WRL = "#ece0f4"
PAD = "#ffffff"
STAGE = ["#e8f1fb", "#d6e6f7", "#c2d9f2", "#adcbec"]
HEAD = "#f3f4f6"
GAT = "#2f855a"      # spatial GAT (green)
GATL = "#e3f2e8"


def box(ax, x, y, w, h, text="", fc="#ffffff", ec=INK, lw=0.8, fs=8, r=0.02, weight="normal", color=INK, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls)
    ax.add_patch(p)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
                fontweight=weight, linespacing=1.25)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK, lw=0.9, style="-|>", ms=7, ls="-", conn="arc3"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=color,
                                 lw=lw, ls=ls, connectionstyle=conn, shrinkA=0, shrinkB=0))


def panel(ax, letter, title, x=0.0, y=1.0):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")
    ax.text(x + 0.035, y, title, transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left", color=INK)


# ════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — architecture with Stages 1-4
# ════════════════════════════════════════════════════════════════════════
def figure2():
    fig = plt.figure(figsize=(7.4, 3.25))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 43); ax.axis("off")

    box(ax, 0.3, 14, 8.6, 22, "", fc="#f9fafb")
    ax.text(4.6, 33.0, "Input", ha="center", fontweight="bold", fontsize=7.6)
    ax.text(4.6, 29.6, "270 days", ha="center", fontsize=6.8)
    ax.text(4.6, 26.3, "8 forcings\n13 VIC\nparameters\n(+ lagged\noutputs in\nAR variants)",
            ha="center", va="top", fontsize=5.8, color=MUTED, linespacing=1.3)
    arrow(ax, 8.9, 25, 10.2, 25)
    box(ax, 10.2, 15.5, 7.6, 19, "Patch\nembedding\n\nConv1D,\nkernel =\nstride =\n10 days\n\n+ LayerNorm\n+ position\nembedding",
        fc="#f9fafb", fs=5.8)
    arrow(ax, 17.8, 25, 19.2, 25)

    stages = [(27, 128, 4, 4), (14, 256, 4, 8), (7, 512, 16, 16), (4, 1024, 4, 32)]
    x = 19.2; sw = 10.1; gap = 3.4; tw = 0.345
    for i, (n, d, nb, h) in enumerate(stages):
        box(ax, x, 13, sw, 24, "", fc=STAGE[i], ec=INK)
        cx = x + sw / 2
        ax.text(cx, 34.3, f"Stage {i + 1}", ha="center", fontweight="bold", fontsize=7.6)
        ax.text(cx, 31.0, f"{nb} Swin blocks", ha="center", fontsize=6.6)
        ax.text(cx, 29.2, "W-MSA / SW-MSA\nalternating", ha="center", va="top", fontsize=5.5, color=MUTED, linespacing=1.15)
        ax.text(cx, 23.6, f"{n} tokens × {d}", ha="center", fontsize=6.0)
        ax.text(cx, 21.2, f"{h} heads", ha="center", fontsize=5.9, color=MUTED)
        ax.text(cx, 18.6, "2 windows: local" if n == 27 else "1 window: global", ha="center", fontsize=5.9,
                color=W1 if n == 27 else INK)
        ax.text(cx, 15.6, {27: "1 token = 10 d", 14: "1 token ≈ 20 d", 7: "1 token ≈ 40 d", 4: "1 token ≈ 70 d"}[n],
                ha="center", fontsize=5.6, color=MUTED)
        slots = 28 if n == 27 else 14
        tx0 = cx - slots * tw / 2
        for k in range(slots):
            real = k < n
            ax.add_patch(Rectangle((tx0 + k * tw, 7.6), tw, 2.4, fc=INK if real else PAD,
                                   ec="white" if real else MUTED, lw=0.25 if real else 0.3, ls="-" if real else ":"))
        if n == 27:
            for j, c in enumerate([W1, W2]):
                ax.add_patch(Rectangle((tx0 + 14 * j * tw, 7.2), 14 * tw, 3.2, fc="none", ec=c, lw=1.0))
        else:
            ax.add_patch(Rectangle((tx0, 7.2), 14 * tw, 3.2, fc="none", ec=W1, lw=1.0))
        if i < 3:
            arrow(ax, x + sw, 25, x + sw + gap, 25)
            ax.text(x + sw + gap / 2, 26.2, "patch\nmerging", ha="center", va="bottom", fontsize=5.0, color=MUTED, linespacing=1.05)
            ax.text(x + sw + gap / 2, 23.8, f"{n}→{stages[i + 1][0]}", ha="center", va="top", fontsize=5.0, color=MUTED)
        x += sw + gap
    xp = x - gap
    # pooling
    arrow(ax, xp, 25, xp + 1.2, 25)
    box(ax, xp + 1.2, 19, 5.6, 12, "Layer\nNorm +\naverage\npooling\n\n1 × 1024", fc="#f9fafb", fs=5.6)
    # spatial GAT
    gx = xp + 8.2
    arrow(ax, xp + 6.8, 25, gx, 25)
    box(ax, gx, 15.5, 8.0, 19, "", fc=GATL, ec=GAT)
    ax.text(gx + 4.0, 32.4, "Spatial GAT", ha="center", fontsize=6.0, fontweight="bold", color=GAT)
    ax.text(gx + 4.0, 29.8, "neighbours\n≤ 2°, same day\nand parameter\nset\n\n8 heads\n+ residual\n+ LayerNorm",
            ha="center", va="top", fontsize=5.4, linespacing=1.2)
    # mini graph under the GAT box
    pts = [(gx + 1.4, 9.2), (gx + 3.2, 11.0), (gx + 4.6, 8.4), (gx + 6.4, 10.4), (gx + 7.4, 7.6)]
    for a_, b_ in [(0, 1), (1, 2), (0, 2), (2, 3), (1, 3)]:
        ax.plot([pts[a_][0], pts[b_][0]], [pts[a_][1], pts[b_][1]], color=GAT, lw=0.6)
    for k, (px, py) in enumerate(pts):
        ax.add_patch(plt.Circle((px, py), 0.55, fc=GAT if k == 2 else "white", ec=GAT, lw=0.7))
    ax.text(gx + 4.0, 5.6, "cells of one day;\nedges ≤ 2°", ha="center", va="top", fontsize=5.0, color=MUTED, linespacing=1.1)
    # heads
    hx = gx + 9.4
    for name, yy in [("ET", 30.5), ("SM", 22.5), ("Q", 14.5)]:
        arrow(ax, gx + 8.0, 25, hx, yy + 2.5, conn="arc3,rad=0", ms=5)
        box(ax, hx, yy, 99.7 - hx, 5.0, f"{name}\nhead", fc=HEAD, fs=5.8)

    ax.add_patch(Rectangle((19.6, 2.0), 0.8, 1.8, fc=INK, ec="white", lw=0.25)); ax.text(20.8, 2.9, "token", va="center", fontsize=5.8)
    ax.add_patch(Rectangle((25.6, 2.0), 0.8, 1.8, fc=PAD, ec=MUTED, lw=0.3, ls=":")); ax.text(26.8, 2.9, "zero padding", va="center", fontsize=5.8)
    ax.add_patch(Rectangle((34.6, 1.8), 3.0, 2.2, fc="none", ec=W1, lw=1.0))
    ax.text(38.1, 2.9, "attention window, M = 14 tokens (140 days at Stage 1)", va="center", fontsize=5.8)
    save(fig, "Figure2_architecture")


def save(fig, name):
    for ext in ("png", "pdf", "tif"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(OUT / f"{name}.{ext}", dpi=300 if ext == "png" else 600, **kw)
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — W-MSA / SW-MSA, original vs temporal, and the Swin block
# ════════════════════════════════════════════════════════════════════════
def grid2d(ax, x0, y0, n, cell, shift=0, title=""):
    """n x n patch grid; windows of 4x4 patches; shift>0 draws the shifted partition."""
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x0 + j * cell, y0 + i * cell), cell, cell, fc="#f9fafb", ec=GRID, lw=0.3))
    M = n // 2
    if shift == 0:
        cols = [(W1, W1L), (W2, W2L), (W2, W2L), (W1, W1L)]
        k = 0
        for wi in range(2):
            for wj in range(2):
                c, cl = cols[k]; k += 1
                ax.add_patch(Rectangle((x0 + wj * M * cell, y0 + wi * M * cell), M * cell, M * cell, fc=cl, ec=c, lw=1.1, alpha=0.9))
    else:
        s = shift
        # interior shifted windows: boundaries at s and s+M
        edges = [0, s, s + M, n]
        for a in range(3):
            for b in range(3):
                xw, ww = edges[b], edges[b + 1] - edges[b]
                yw, hw = edges[a], edges[a + 1] - edges[a]
                wrapped = (a != 1) or (b != 1)
                fc, ec = (WRL, WR) if wrapped else (W1L, W1)
                ax.add_patch(Rectangle((x0 + xw * cell, y0 + yw * cell), ww * cell, hw * cell, fc=fc, ec=ec, lw=1.1, alpha=0.9))
    ax.text(x0 + n * cell / 2, y0 + n * cell + 1.0, title, ha="center", va="bottom", fontsize=7.2)


def tokens_row(ax, x0, y, order, tw, th, windows, wcolors, labels=True, wrap_idx=None):
    """Draw a row of tokens in the given order (list of token ids, 0 = pad)."""
    for k, t in enumerate(order):
        fc = PAD if t == 0 else "#f9fafb"
        ax.add_patch(Rectangle((x0 + k * tw, y), tw, th, fc=fc, ec=MUTED if t == 0 else GRID, lw=0.35,
                               ls=":" if t == 0 else "-"))
        if labels:
            ax.text(x0 + (k + 0.5) * tw, y + th / 2, "" if t == 0 else str(t), ha="center", va="center", fontsize=4.6,
                    color=INK)
    for (a, b), (ec, fc) in zip(windows, wcolors):
        ax.add_patch(Rectangle((x0 + a * tw, y - 0.35), (b - a) * tw, th + 0.7, fc=fc, ec=ec, lw=1.2, zorder=0))


def figure3():
    import textwrap
    fig = plt.figure(figsize=(7.4, 8.9))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 120); ax.axis("off")

    def title(y, letter, text):
        ax.text(1, y, f"({letter})", fontweight="bold", fontsize=9, va="top")
        ax.text(5, y, text, fontsize=8.5, va="top")

    # ---------- (a) original 2-D Swin ----------
    title(119, "a", "Original Swin Transformer (images): window attention in two consecutive blocks")
    cell = 2.0; n = 8
    grid2d(ax, 8, 94, n, cell, 0, "Block l: W-MSA\n4 windows of M × M patches")
    arrow(ax, 27.5, 102, 35.5, 102)
    ax.text(31.5, 103.4, "cyclic shift\nby ⌊M/2⌋", ha="center", va="bottom", fontsize=6.4, color=MUTED)
    grid2d(ax, 37, 94, n, cell, 2, "Block l + 1: SW-MSA\nwindows straddle the old boundaries")
    ax.text(58, 110, "• attention is computed only among patches\n  inside the same window", va="top", fontsize=6.8, linespacing=1.35)
    ax.text(58, 104, "• the shift lets information cross window\n  boundaries from one block to the next", va="top", fontsize=6.8, linespacing=1.35)
    ax.add_patch(Rectangle((58.3, 95.3), 2.6, 2.0, fc=WRL, ec=WR, lw=1.0))
    ax.text(61.6, 96.3, "windows built from patches wrapped around the\nimage edge; in the original model a mask stops\nattention between patches that are not adjacent",
            va="center", fontsize=6.2, color=MUTED, linespacing=1.3)

    # ---------- (b) temporal tokenisation ----------
    title(88, "b", "LandFormer: temporal tokenisation of the 270-day input")
    x0, wtot = 11, 84
    dw = wtot / 270
    for c in range(6):
        ax.add_patch(Rectangle((x0, 74 + c * 0.9), wtot, 0.9, fc=["#eef2f7", "#e3e9f1"][c % 2], ec="none"))
    for k in range(28):
        ax.plot([x0 + k * 10 * dw] * 2, [74, 79.4], color=GRID, lw=0.4)
    ax.text(x0 - 1, 76.7, "daily input\n(23 channels)", ha="right", va="center", fontsize=6.2, color=MUTED)
    ax.text(x0, 80.0, "day 1 (t − 269)", ha="left", va="bottom", fontsize=6.2, color=MUTED)
    ax.text(x0 + wtot, 80.0, "day 270 = prediction day t", ha="right", va="bottom", fontsize=6.2, color=MUTED)
    ax.plot([x0, x0 + 10 * dw], [73.3, 73.3], color=INK, lw=0.6)
    ax.text(x0 + 5 * dw, 72.6, "10 days", ha="center", va="top", fontsize=5.8)
    arrow(ax, 45, 72.6, 45, 69.4)
    ax.text(46.3, 71.0, "Conv1D (kernel = stride = 10 days) → one token per 10 days", va="center", fontsize=6.6)
    tokens_row(ax, x0, 65.2, list(range(1, 28)), wtot / 27, 2.6, [], [])
    ax.text(x0 - 1, 66.5, "27 tokens", ha="right", va="center", fontsize=6.2, color=MUTED)

    # ---------- (c) temporal W-MSA / SW-MSA ----------
    title(60, "c", "LandFormer, Stage 1: temporal W-MSA and SW-MSA in two consecutive Swin blocks (M = 14 tokens)")
    tw = wtot / 28
    tokens_row(ax, x0, 49.5, list(range(1, 28)) + [0], tw, 2.6, [(0, 14), (14, 28)], [(W1, W1L), (W2, W2L)])
    ax.text(x0 + 7 * tw, 53.0, "window 1: tokens 1–14 (days 1–140)", ha="center", va="bottom", fontsize=6.2, color=W1)
    ax.text(x0 + 21 * tw, 53.0, "window 2: tokens 15–27 (days 141–270) + 1 zero-pad token", ha="center", va="bottom", fontsize=6.2, color=W2)
    tokens_row(ax, x0, 40.0, list(range(8, 28)) + [0] + list(range(1, 8)), tw, 2.6, [(0, 14), (14, 28)], [(W1, W1L), (WR, WRL)])
    ax.text(x0 + 7 * tw, 39.0, "window 1′: tokens 8–21 (crosses the old boundary 14 | 15)", ha="center", va="top", fontsize=6.2, color=W1)
    ax.text(x0 + 21 * tw, 39.0, "window 2′ (wrapped): tokens 22–27, pad, 1–7", ha="center", va="top", fontsize=6.2, color=WR)
    ax.text(x0 - 1, 50.8, "W-MSA\nblock l", ha="right", va="center", fontsize=6.6, fontweight="bold", linespacing=1.2)
    ax.text(x0 - 1, 41.3, "SW-MSA\nblock l + 1", ha="right", va="center", fontsize=6.6, fontweight="bold", linespacing=1.2)
    arrow(ax, 5.3, 48.4, 5.3, 43.6, ms=5)
    ax.text(5.8, 46.0, "roll by 7 tokens\n(70 days)", va="center", fontsize=5.6, color=MUTED, linespacing=1.15)
    para = ("After attention the roll is reversed, so every token returns to its true time position. Across the two blocks, "
            "token 14 attends to tokens 1–14 and then to 8–21, so information crosses the window boundary. "
            "In LandFormer the wrapped window 2′ is not masked: the most recent tokens (22–27) can also attend to the earliest "
            "(1–7), and the absolute position embedding keeps their true positions distinguishable. Stages 2–4 hold 14, 7 and 4 "
            "tokens, which fit in one window, so attention there covers the whole sequence.")
    ax.text(x0, 34.3, textwrap.fill(para, 128), va="top", fontsize=6.4, color=INK, linespacing=1.4)

    # ---------- (d) Swin block ----------
    title(23.5, "d", "Two consecutive Swin blocks: the unit repeated inside each stage")
    y = 6.5; h = 5.4
    seq = ["z", "LN", "W-MSA", "+", "LN", "MLP", "+", "LN", "SW-MSA", "+", "LN", "MLP", "+"]
    width = {"z": 3.2, "+": 2.6, "LN": 5.8, "W-MSA": 6.4, "SW-MSA": 6.4, "MLP": 6.6}
    g = 1.25
    total = sum(width[k] for k in seq) + g * (len(seq) - 1)
    xpos = (100 - total) / 2
    pos = []
    for k in seq:
        w = width[k]
        if k == "+":
            ax.add_patch(plt.Circle((xpos + w / 2, y + h / 2), 1.3, fc="white", ec=INK, lw=0.8))
            ax.text(xpos + w / 2, y + h / 2, "+", ha="center", va="center", fontsize=9)
        elif k == "z":
            ax.text(xpos + w / 2, y + h / 2, "$z^{l-1}$", ha="center", va="center", fontsize=8)
        else:
            label = {"LN": "Layer\nNorm", "MLP": "MLP\n4×, GELU"}.get(k, k)
            fc = W1L if k == "W-MSA" else (WRL if k == "SW-MSA" else "#f9fafb")
            ec = W1 if k == "W-MSA" else (WR if k == "SW-MSA" else INK)
            box(ax, xpos, y, w, h, label, fc=fc, ec=ec, fs=6.2)
        pos.append((xpos, w))
        xpos += w + g
    for k in range(len(seq) - 1):
        (a, wa), (b, _) = pos[k], pos[k + 1]
        arrow(ax, a + wa, y + h / 2, b, y + h / 2, ms=5)
    for j, (s0, s1) in enumerate([(0, 3), (3, 6), (6, 9), (9, 12)]):
        xa = pos[s0][0] + pos[s0][1] / 2 + (0.6 if s0 else 0)
        xb = pos[s1][0] + pos[s1][1] / 2
        hy = y + h + (2.4 if j % 2 == 0 else 4.0)
        ax.plot([xa, xa, xb], [y + h - (0 if s0 == 0 else 1.4), hy, hy], color=MUTED, lw=0.7)
        arrow(ax, xb, hy, xb, y + h / 2 + 1.35, color=MUTED, ms=4, lw=0.7)
    for k, lab in [(3, "$\\hat z^{l}$"), (6, "$z^{l}$"), (9, "$\\hat z^{l+1}$"), (12, "$z^{l+1}$")]:
        ax.text(pos[k][0] + pos[k][1] / 2, y - 0.8, lab, ha="center", va="top", fontsize=7)
    xm = pos[6][0] + pos[6][1] + g / 2
    ax.plot([xm, xm], [y - 3.0, y + h + 6.0], color=GRID, lw=0.6, ls="--")
    ax.text((pos[1][0] + xm) / 2, y + h + 5.2, "Swin block l (W-MSA)", ha="center", fontsize=6.6, color=W1)
    ax.text((xm + pos[12][0] + pos[12][1]) / 2, y + h + 5.2, "Swin block l + 1 (SW-MSA)", ha="center", fontsize=6.6, color=WR)
    ax.text(x0, 3.0, "Depths [4, 4, 16, 4] = 2, 2, 8 and 2 such pairs per stage. A learnable relative position bias, which depends only on\n"
                     "the temporal offset between two tokens, is added to every attention score.",
            fontsize=6.2, color=MUTED, va="top", linespacing=1.35)
    save(fig, "Figure3_swin_attention")


figure2()
figure3()
print("saved to", OUT)
