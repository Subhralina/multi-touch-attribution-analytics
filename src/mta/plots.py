"""Story figures. Reads the CSVs written by run_analysis.py, writes PNGs.

Design rules: the two channels in the decision (Instagram, Online Display) carry
color; everything else is a quiet gray. Text is never colored.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8984"
GRID = "#e7e6e2"
OTHER = "#c9c8c3"
COLORS = {"Instagram": "#eb6834", "Online Display": "#2a78d6"}
CHANNELS = ["Facebook", "Instagram", "Online Display", "Online Video", "Paid Search"]
HEURISTIC = ["First touch", "Last touch", "Linear", "Position-based"]
DATA_DRIVEN = ["Markov", "Shapley"]


def color(ch: str) -> str:
    return COLORS.get(ch, OTHER)


def _style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "xtick.color": INK_2,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.grid": False,
    })


def _header(fig, title: str, subtitle: str) -> None:
    fig.text(0.035, 0.955, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.035, 0.885, subtitle, fontsize=10.5, color=INK_2, va="top")


def _footer(fig, note: str) -> None:
    fig.text(0.035, 0.03, note, fontsize=8.5, color=INK_3, va="bottom")


def _xgrid(ax) -> None:
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _pct(ax, axis: str = "x", decimals: int = 0) -> None:
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=decimals)
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def fig_last_touch(shares: pd.DataFrame, path: Path) -> None:
    s = shares["Last touch"].sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.17, right=0.93, top=0.76, bottom=0.14)
    y = np.arange(len(s))
    ax.barh(y, s.values, height=0.5, color=[color(c) for c in s.index])
    for yi, (ch, v) in zip(y, s.items()):
        ax.text(v + 0.004, yi, f"{v:.1%}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, s.index)
    ax.set_xlim(0, 0.34)
    _pct(ax)
    _xgrid(ax)
    gap = s["Instagram"] - s["Online Display"]
    ax.annotate(f"{gap * 100:.1f} pts apart", xy=(0.16, 0.5), xytext=(0.19, 0.5),
                fontsize=10, color=INK_2, va="center",
                arrowprops=dict(arrowstyle="-[, widthB=1.6, lengthB=0.3", color=INK_3, lw=1))
    _header(fig, "Last-click reporting ties Instagram and Display for last place",
            "Share of conversions credited to each channel under last-touch attribution")
    _footer(fig, "240,108 journeys, 17,639 conversions, July 2018.")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_model_split(shares: pd.DataFrame, path: Path) -> None:
    models = HEURISTIC + DATA_DRIVEN
    fig, ax = plt.subplots(figsize=(9, 5.2))
    fig.subplots_adjust(left=0.17, right=0.86, top=0.74, bottom=0.14)
    ys = {m: i for i, m in enumerate(reversed(DATA_DRIVEN))}
    ys.update({m: i + len(DATA_DRIVEN) + 0.8 for i, m in enumerate(reversed(HEURISTIC))})
    for m in models:
        y = ys[m]
        ig, dp = shares.loc["Instagram", m], shares.loc["Online Display", m]
        ax.plot([dp, ig], [y, y], color=OTHER, lw=2, solid_capstyle="round", zorder=1)
        for ch, v in [("Online Display", dp), ("Instagram", ig)]:
            ax.scatter(v, y, s=90, color=COLORS[ch], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(0.187, y, f"+{(ig - dp) * 100:.1f} pts", va="center", fontsize=10,
                color=INK if m in DATA_DRIVEN else INK_3, fontweight="bold" if m in DATA_DRIVEN else "normal")
    ax.set_yticks([ys[m] for m in models], models)
    ax.set_xlim(0.10, 0.185)
    ax.set_ylim(-0.6, max(ys.values()) + 0.6)
    _pct(ax)
    _xgrid(ax)
    sep = len(DATA_DRIVEN) - 0.1
    ax.axhline(sep, color=GRID, lw=0.8)
    ax.text(0.1005, sep + 0.12, "RULE-BASED", fontsize=8, color=INK_3, va="bottom")
    ax.text(0.1005, sep - 0.12, "DATA-DRIVEN", fontsize=8, color=INK_3, va="top")
    ax.text(0.187, max(ys.values()) + 0.55, "Gap", fontsize=8.5, color=INK_3, va="bottom")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=8, color=COLORS[c], label=c)
               for c in ["Instagram", "Online Display"]]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.83), ncol=2,
               frameon=False, fontsize=9.5, handletextpad=0.3, columnspacing=1.4)
    _header(fig, "Only the data-driven models tell the two channels apart",
            "Share of conversion credit for Instagram and Online Display, by attribution model")
    _footer(fig, "Rule-based models split credit by position. Markov and Shapley learn it from which journeys convert.")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_assist(assist: pd.DataFrame, path: Path) -> None:
    s = assist["assisted_only"].sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.17, right=0.93, top=0.76, bottom=0.14)
    y = np.arange(len(s))
    ax.barh(y, s.values, height=0.5, color=[color(c) for c in s.index])
    for yi, (ch, v) in zip(y, s.items()):
        ax.text(v + 0.008, yi, f"{v:.0%}", va="center", fontsize=10, color=INK)
    ax.set_yticks(y, s.index)
    ax.set_xlim(0, 0.6)
    _pct(ax)
    _xgrid(ax)
    _header(fig, "Half of Instagram's work happens before the last click",
            "Of converting journeys that include the channel, share where it was not the final touch")
    _footer(fig, "Last-touch reporting gives these journeys zero credit for the channel.")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_lift(lift: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=0.09, right=0.8, top=0.76, bottom=0.18)
    x = lift.index.values
    labels = [str(i) for i in x[:-1]] + [f"{x[-1]}+"]
    ax.axhline(1.0, color=INK_3, lw=1)
    ax.text(4.1, 0.99, "no difference", fontsize=8.5, color=INK_3, va="top")
    order = [c for c in CHANNELS if c not in COLORS] + list(COLORS)
    for ch in order:
        hi = ch in COLORS
        ax.plot(x, lift[ch], color=color(ch), lw=2 if hi else 1.5, zorder=3 if hi else 2,
                solid_capstyle="round", solid_joinstyle="round")
        if hi:
            ax.scatter(x[-1], lift[ch].iloc[-1], s=60, color=color(ch), edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.text(x[-1] + 0.15, lift[ch].iloc[-1], ch, va="center", fontsize=9.5 if hi else 8.5,
                color=INK if hi else INK_3, fontweight="bold" if hi else "normal")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Touches in the journey")
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.1f}×"))
    ax.tick_params(axis="y", colors=INK_2)
    _header(fig, "Display journeys convert less than equal-length journeys without it",
            "Conversion rate with the channel ÷ without it, comparing journeys of equal length")
    _footer(fig, "Matching on length removes the 'longer journeys convert more' effect. It is not a randomized test.")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_removal(removal: pd.DataFrame, path: Path) -> None:
    s = removal.sort_values("removal_effect")
    fig, ax = plt.subplots(figsize=(9, 4.6))
    fig.subplots_adjust(left=0.17, right=0.93, top=0.76, bottom=0.14)
    y = np.arange(len(s))
    ax.barh(y, s["removal_effect"], height=0.5, color=[color(c) for c in s.index])
    err = np.vstack([s["removal_effect"] - s["removal_effect_low"], s["removal_effect_high"] - s["removal_effect"]])
    ax.errorbar(s["removal_effect"], y, xerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    for yi, (ch, r) in zip(y, s.iterrows()):
        ax.text(r["removal_effect_high"] + 0.008, yi,
                f"{r['removal_effect']:.0%}  ·  ~{int(r['conversions_at_risk']):,} conversions",
                va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, s.index)
    ax.set_xlim(0, 0.6)
    _pct(ax)
    _xgrid(ax)
    _header(fig, "Switching off Display costs the fewest conversions",
            "Modelled drop in conversions if the channel were removed (Markov removal effect, 95% bootstrap CI)")
    _footer(fig, "Effects overlap across channels, so they don't sum to 100%. Model estimate, not a measured result.")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_all_models(shares: pd.DataFrame, path: Path) -> None:
    models = HEURISTIC + DATA_DRIVEN
    data = shares.loc[CHANNELS, models]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    fig.subplots_adjust(left=0.17, right=0.97, top=0.74, bottom=0.08)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blue", ["#eef4fc", "#86b6ef", "#1c5cab"])
    ax.imshow(data.values, cmap=cmap, aspect="auto", vmin=0.08, vmax=0.34)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            ax.text(j, i, f"{v:.1%}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if v > 0.24 else INK)
    ax.set_xticks(range(len(models)), models)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", colors=INK)
    ax.set_yticks(range(len(CHANNELS)), CHANNELS)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(len(HEURISTIC) - 0.5, color=SURFACE, lw=4)
    _header(fig, "All six models side by side", "Share of conversion credit by channel")
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_all(results_dir: Path, figures_dir: Path) -> None:
    _style()
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    shares = pd.read_csv(results_dir / "attribution_shares.csv", index_col=0)
    fig_last_touch(shares, figures_dir / "01_last_touch_tie.png")
    fig_model_split(shares, figures_dir / "02_models_split_the_tie.png")
    fig_assist(pd.read_csv(results_dir / "assist_share.csv", index_col=0), figures_dir / "03_assist_share.png")
    fig_lift(pd.read_csv(results_dir / "length_matched_lift.csv", index_col=0), figures_dir / "04_length_matched_lift.png")
    fig_removal(pd.read_csv(results_dir / "removal_effects.csv", index_col=0), figures_dir / "05_removal_effect.png")
    fig_all_models(shares, figures_dir / "06_all_models.png")


if __name__ == "__main__":
    import sys
    make_all(Path(sys.argv[1]), Path(sys.argv[2]))
