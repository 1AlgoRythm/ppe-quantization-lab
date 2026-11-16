# Shared look for every chart: light surface, recessive grid, text in ink colors, fixed series order.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed order, validated (dataviz validator, light)

def setup(nrows=1, ncols=1, w=10, h=4.2):
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), facecolor=SURFACE)
    for ax in (axes.flat if hasattr(axes, "flat") else [axes]):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
    return fig, axes

def bars(ax, groups, series, values, fmt="{:.2f}", width=0.36):
    """Grouped bars. values[s][g]; every bar carries its value label (identity never by color alone)."""
    n = len(series)
    for si, s in enumerate(series):
        xs = [g + (si - (n - 1) / 2) * (width + 0.02) for g in range(len(groups))]
        b = ax.bar(xs, values[si], width=width, color=SERIES[si], label=s, edgecolor=SURFACE, linewidth=2)
        for x, v in zip(xs, values[si]):
            ax.text(x, v, fmt.format(v), ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)

def save(fig, path, title, note=None):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
    if note: fig.text(0.01, 0.005, note, fontsize=8, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.04 if note else 0, 1, 0.92))
    fig.savefig(path, dpi=150, facecolor=SURFACE); plt.close(fig)

def legend_top(fig, labels):
    """Legend in one row under the title, away from the bars; plain swatches, text in ink."""
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=SERIES[i], label=l) for i, l in enumerate(labels)],
               loc="upper right", bbox_to_anchor=(0.99, 0.985), ncol=len(labels), frameon=False, fontsize=9)
