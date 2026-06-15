"""Recreate the 4 thesis figures as true-vector SVGs for Word/PDF."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

BASE = os.path.dirname(__file__)
ROOT = os.path.dirname(BASE)
CSV = os.path.join(ROOT, "data")
OUT  = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

# ── shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#d0d0d0",
    "grid.linewidth":    0.7,
})

def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Written: {name}")


# ════════════════════════════════════════════════════════════════════════════
# Fig 4-1  Distribution of Topic Sizes
# ════════════════════════════════════════════════════════════════════════════
topic_info = pd.read_csv(os.path.join(CSV, "topic_info.csv"))
sizes = topic_info[topic_info["Topic"] != -1]["Count"].values

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("white")

ax.hist(sizes, bins=range(0, max(sizes) + 30, 30),
        color="#add8e6", edgecolor="#8ab4c8", linewidth=0.8)
ax.set_title("Distribution of Topic Sizes", fontsize=16, fontweight="bold", pad=14)
ax.set_xlabel("Number of Sentences per Topic", fontsize=12)
ax.set_ylabel("Number of Topics", fontsize=12)
ax.yaxis.grid(True)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#888888")
ax.tick_params(axis="both", labelsize=10)

save(fig, "fig4_1_topic_size_distribution.svg")


# ════════════════════════════════════════════════════════════════════════════
# Fig 4-2  Topic Model Coverage by Rating
# ════════════════════════════════════════════════════════════════════════════
ling = pd.read_csv(os.path.join(CSV, "linguistic_features.csv"))
ling["rating"] = ling["rating"].astype(float).round().astype(int)

coverage = {}
for r in [1, 2, 3, 4, 5, 6]:
    grp = ling[ling["rating"] == r]
    coverage[r] = 100.0 * (grp["topic"] != -1).mean()

ratings  = list(coverage.keys())
cov_vals = [coverage[r] for r in ratings]

# green → yellow-green → yellow → orange → red gradient
bar_colors = ["#1a6b35", "#4caf50", "#b5cc3a", "#f5d44e", "#f4733f", "#a50026"]

fig, ax = plt.subplots(figsize=(9, 6))
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("white")

bars = ax.bar(ratings, cov_vals, color=bar_colors, width=0.65, zorder=3)
for bar, val in zip(bars, cov_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.8,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold")

ax.set_title("Topic Model Coverage by Rating", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Rating (1 = best → 6 = worst)", fontsize=12)
ax.set_ylabel("Coverage (%)", fontsize=12)
ax.set_xticks(ratings)
ax.set_ylim(0, 100)
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.spines["left"].set_visible(False)
ax.spines["bottom"].set_color("#888888")
ax.tick_params(axis="both", labelsize=11)

save(fig, "fig4_2_coverage_by_rating.svg")


# ════════════════════════════════════════════════════════════════════════════
# Fig 4-3  Linguistic Heatmap by Rating
# ════════════════════════════════════════════════════════════════════════════
feature_cols = {
    "Positive evaluation":  "positive_evaluation",
    "Intensifiers":         "intensifiers",
    "Negative evaluation":  "negative_evaluation",
    "Negation":             "negation_count",
    "Modal particles":      "modal_particles_total",
    "Hedging":              "hedging_total",
    "Temporal expressions": "temporal_expressions",
    "Connectives":          "connectives_total",
}

# per-rating mean for each feature
matrix_rows = []
for feat_col in feature_cols.values():
    row = []
    for r in [1, 2, 3, 4, 5, 6]:
        grp = ling[ling["rating"] == r]
        row.append(grp[feat_col].mean())
    matrix_rows.append(row)

raw = np.array(matrix_rows, dtype=float)  # shape (8, 6)

# row-normalise: 0 = row-min, 1 = row-max
norm = np.zeros_like(raw)
for i in range(raw.shape[0]):
    rmin, rmax = raw[i].min(), raw[i].max()
    if rmax > rmin:
        norm[i] = (raw[i] - rmin) / (rmax - rmin)
    else:
        norm[i] = 0.5

# colormap: cream → orange → dark red (matching original)
cmap = LinearSegmentedColormap.from_list(
    "thesis_heat",
    ["#fef9e7", "#f5b041", "#c0392b", "#6e1a0e"],
    N=256,
)

fig, ax = plt.subplots(figsize=(8.5, 6.5))
fig.patch.set_facecolor("white")

im = ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1, interpolation="nearest")

# cell labels
for i in range(raw.shape[0]):
    for j in range(raw.shape[1]):
        val = raw[i, j]
        text_color = "white" if norm[i, j] > 0.55 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=9, color=text_color, fontweight="bold")

ax.set_xticks(range(6))
ax.set_xticklabels([1, 2, 3, 4, 5, 6], fontsize=11)
ax.set_yticks(range(8))
ax.set_yticklabels(list(feature_cols.keys()), fontsize=10)
ax.set_xlabel("Patient Rating (1 = best → 6 = worst)", fontsize=11, labelpad=8)
ax.set_ylabel("Linguistic feature", fontsize=11, labelpad=8)
ax.set_title(
    "German Linguistic Features by Patient Rating\n(colour scaled within each feature row)",
    fontsize=13, fontweight="bold", pad=12,
)
ax.grid(False)

# colorbar
cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.12)
cbar.set_ticks([0, 0.5, 1])
cbar.set_ticklabels(["min", "mid", "max"])
cbar.ax.set_ylabel(
    "Relative value within feature\n(row-normalised, 0 = row min, 1 = row max)",
    rotation=270, labelpad=18, fontsize=8, va="bottom",
)

save(fig, "fig4_3_linguistic_heatmap.svg")


# ════════════════════════════════════════════════════════════════════════════
# Fig 4-4  Linguistic Profiles of Top 10 Topics
# ════════════════════════════════════════════════════════════════════════════
tla = pd.read_csv(os.path.join(CSV, "topic_linguistic_analysis.csv"))
top10 = tla[tla["topic"].between(0, 9)].sort_values("topic")
topic_ids = [f"T{t}" for t in top10["topic"]]

subplots = [
    ("Modal Particles",  "modal_particles_total",  "#2a9d8f"),
    ("Hedging",          "hedging_total",           "#e9c46a"),
    ("Positive Eval",    "positive_evaluation",     "#9b59b6"),
    ("Negative Eval",    "negative_evaluation",     "#e07b7b"),
    ("Connectives",      "connectives_total",       "#e76f1a"),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.patch.set_facecolor("white")
fig.suptitle("Linguistic Profiles of Top 10 Topics",
             fontsize=14, fontweight="bold", y=1.01)

axes_flat = axes.flatten()
# hide the 6th cell (we only have 5 subplots)
axes_flat[5].set_visible(False)

for ax, (title, col, color) in zip(axes_flat[:5], subplots):
    vals = top10[col].values
    bars = ax.bar(range(10), vals, color=color, alpha=0.85, width=0.65, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + max(vals) * 0.015,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks(range(10))
    ax.set_xticklabels(topic_ids, fontsize=9)
    ax.set_xlabel("Topic ID", fontsize=9)
    ax.set_ylabel("Average Value", fontsize=9)
    ax.yaxis.grid(True, zorder=0, linewidth=0.6, color="#d0d0d0")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_facecolor("#f8f9fa")

fig.tight_layout()
save(fig, "fig4_4_topic_profiles.svg")

print("\nAll 4 figure SVGs generated successfully!")
print(f"Output: {OUT}")
