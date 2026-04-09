"""
Supply Chain Risk – Causal Path Diagram
Academic publication-quality figure (clean white boxes, directional arrows)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Layout ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 16, 10
BOX_W, BOX_H = 2.2, 0.65
R = 0.06

COL = {"input": 1.6, "mid": 7.2, "out": 12.8}

nodes = {
    "Supplier\nEfficiency":    (COL["input"], 7.8),
    "Demand\nComplexity":      (COL["input"], 6.2),
    "Logistics\nConstraints":  (COL["input"], 4.6),
    "Geographical\nFeature":   (COL["input"], 3.0),
    "Pricing\nPressure":       (COL["input"], 1.4),
    "Execution\nDelay":        (COL["mid"],   4.6),
    "Late Delivery\nRisk":     (COL["out"],   4.6),
}

# All boxes: simple grey border, white fill — no colour coding on boxes
BOX_EDGE  = "#444444"
BOX_FACE  = "white"

# Arrow colours by source group
ARROW_COLORS = {
    "Supplier\nEfficiency":    "#2166AC",
    "Demand\nComplexity":      "#4DAC26",
    "Logistics\nConstraints":  "#D01C8B",
    "Geographical\nFeature":   "#E66101",
    "Pricing\nPressure":       "#1B7837",
    "Execution\nDelay":        "#B2182B",
}

edges = [
    ("Supplier\nEfficiency",   "Execution\nDelay"),
    ("Supplier\nEfficiency",   "Late Delivery\nRisk"),
    ("Demand\nComplexity",     "Execution\nDelay"),
    ("Demand\nComplexity",     "Late Delivery\nRisk"),
    ("Logistics\nConstraints", "Execution\nDelay"),
    ("Logistics\nConstraints", "Late Delivery\nRisk"),
    ("Geographical\nFeature",  "Execution\nDelay"),
    ("Geographical\nFeature",  "Late Delivery\nRisk"),
    ("Pricing\nPressure",      "Late Delivery\nRisk"),
    ("Execution\nDelay",       "Late Delivery\nRisk"),
]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.set_aspect("equal")
ax.axis("off")

def right_mid(key):
    cx, cy = nodes[key]
    return cx + BOX_W / 2, cy

def left_mid(key):
    cx, cy = nodes[key]
    return cx - BOX_W / 2, cy

# ── Column dividers ───────────────────────────────────────────────────────────
for xd in [4.2, 9.8]:
    ax.plot([xd, xd], [0.5, 9.2], color="#dddddd",
            linewidth=0.8, linestyle="--", zorder=0)

# ── Draw edges ────────────────────────────────────────────────────────────────
for src, dst in edges:
    sx, sy = right_mid(src)
    dx, dy = left_mid(dst)
    color  = ARROW_COLORS[src]
    is_key = (src == "Execution\nDelay")

    # bend amount based on vertical distance
    dy_diff = sy - dy
    rad = float(np.clip(dy_diff * 0.04, -0.40, 0.40))

    arr = FancyArrowPatch(
        posA=(sx, sy), posB=(dx, dy),
        connectionstyle=f"arc3,rad={rad:.3f}",
        arrowstyle="-|>",
        mutation_scale=16 if is_key else 12,
        linewidth=2.4 if is_key else 1.2,
        color=color,
        alpha=1.0 if is_key else 0.65,
        zorder=2,
    )
    ax.add_patch(arr)

# ── Draw nodes (plain white, dark border) ─────────────────────────────────────
for label, (cx, cy) in nodes.items():
    x0 = cx - BOX_W / 2
    y0 = cy - BOX_H / 2

    box = FancyBboxPatch(
        (x0, y0), BOX_W, BOX_H,
        boxstyle=f"round,pad={R}",
        linewidth=1.6,
        edgecolor=BOX_EDGE,
        facecolor=BOX_FACE,
        zorder=3,
    )
    ax.add_patch(box)

    lines = label.split("\n")
    offsets = [0.12, -0.12] if len(lines) == 2 else [0]
    for line, yo in zip(lines, offsets):
        ax.text(
            cx, cy + yo, line,
            ha="center", va="center",
            fontsize=10.5,
            fontfamily="DejaVu Serif",
            color="#111111",
            fontweight="bold",
            zorder=5,
        )

# ── Column header arrows (Input → Intermediate → Outcome) ────────────────────
header_y = 9.0
arrow_kw = dict(
    arrowstyle="-|>",
    mutation_scale=18,
    linewidth=1.8,
    color="#555555",
    zorder=1,
)

# Arrow: Input → Intermediate
ax.add_patch(FancyArrowPatch(
    posA=(COL["input"] + 1.4, header_y),
    posB=(COL["mid"]   - 1.4, header_y),
    connectionstyle="arc3,rad=0",
    **arrow_kw,
))

# Arrow: Intermediate → Outcome
ax.add_patch(FancyArrowPatch(
    posA=(COL["mid"] + 1.4, header_y),
    posB=(COL["out"] - 1.4, header_y),
    connectionstyle="arc3,rad=0",
    **arrow_kw,
))

# ── Column headers ─────────────────────────────────────────────────────────────
hdr = dict(ha="center", va="center", fontsize=11,
           fontfamily="DejaVu Serif", color="#222222", fontweight="bold")
ax.text(COL["input"], header_y, "Input Factors",       **hdr)
ax.text(COL["mid"],   header_y, "Mediator Outcome", **hdr)
ax.text(COL["out"],   header_y, "Final Outcome",        **hdr)

# Underlines beneath headers
for cx in [COL["input"], COL["mid"], COL["out"]]:
    ax.plot([cx - 1.0, cx + 1.0], [header_y - 0.28, header_y - 0.28],
            color="#aaaaaa", linewidth=0.9)

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(
    FIG_W / 2, 9.55,
    "Causal Pathway Diagram: Determinants of Late Delivery Risk",
    ha="center", va="center",
    fontsize=14, fontweight="bold",
    fontfamily="DejaVu Serif", color="#111111",
)

# ── Figure note ───────────────────────────────────────────────────────────────
# ax.text(
#     FIG_W / 2, 0.28,
#     "Note: Arrows represent directional causal influence. "
#     "The bold red arrow denotes the critical mediated path (Execution Delay → Late Delivery Risk).",
#     ha="center", va="center",
#     fontsize=8.5, fontstyle="italic",
#     fontfamily="DejaVu Serif", color="#555555",
# )

# ── Legend (arrow colours = source construct) ─────────────────────────────────
# legend_handles = [
#     mpatches.Patch(facecolor=c, edgecolor=c,
#                    label=lbl.replace("\n", " "), alpha=0.75)
#     for lbl, c in ARROW_COLORS.items()
# ]
# legend = ax.legend(
#     handles=legend_handles,
#     title="Construct (arrow colour)",
#     title_fontsize=8.5,
#     fontsize=8,
#     loc="lower right",
#     bbox_to_anchor=(0.99, 0.04),
#     frameon=True, framealpha=1.0,
#     edgecolor="#bbbbbb", facecolor="white",
#     handlelength=1.2, handleheight=0.9,
# )
# legend.get_title().set_fontfamily("DejaVu Serif")
# for t in legend.get_texts():
#     t.set_fontfamily("DejaVu Serif")

# ── Outer border ──────────────────────────────────────────────────────────────
ax.add_patch(mpatches.FancyBboxPatch(
    (0.2, 0.15), FIG_W - 0.4, FIG_H - 0.3,
    boxstyle="square,pad=0",
    linewidth=1.0, edgecolor="#aaaaaa", facecolor="none",
    transform=ax.transData, zorder=0,
))

plt.tight_layout(pad=0.3)
plt.savefig(
    "acyclic_dag_graph.png",
    dpi=300, bbox_inches="tight", facecolor="white",
)
print("Saved.")