from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import mkdir

_HIGHLIGHT_COLORS = ["#C44E52", "#E58606"]

_STYLE = {
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.axisbelow": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
}


def plot_hist(
    df: pd.DataFrame,
    col: str,
    *,
    binwidth: float,
    xlabel: str,
    title: str,
    output: str,
    highlight_time: Optional[str] = None,
    time_col: str = "Time",
    unit: str = "",
    symbol: Optional[str] = None,
    fmt: str = ".1f",
    figsize: tuple[float, float] = (8, 5),
    dpi: int = 300,
) -> str:
    """
    Publication-style histogram of ``df[col]`` with a statistics box.

    Rows whose ``time_col`` equals ``highlight_time`` are marked with dashed lines.
    ``symbol``/``fmt`` control the legend text of highlighted values.
    ``inf`` and missing values are dropped. Returns the output path.
    """
    values = df[col].replace([np.inf, -np.inf], np.nan)
    valid_df = df.assign(**{col: values}).dropna(subset=[col])
    data = valid_df[col]
    u = f" {unit}" if unit else ""

    sns.set_theme(style="ticks", context="paper", font_scale=1.2)
    plt.rcParams.update(_STYLE)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.histplot(data, binwidth=binwidth, kde=False, color="#4C72B0", edgecolor="black", alpha=0.8, ax=ax, stat="count")

    if highlight_time is not None:
        targets = valid_df.loc[valid_df[time_col] == highlight_time, col]
        many = len(targets) > 1
        for idx, value in enumerate(targets):
            c = _HIGHLIGHT_COLORS[idx % len(_HIGHLIGHT_COLORS)]
            name = f"Main Event #{idx + 1}" if many else f"Main Event ({highlight_time})"
            ax.axvline(x=value, color=c, linestyle="--", linewidth=2, zorder=4, label=f"{name}\n{symbol or col} = {value:{fmt}}{u}")
            ax.scatter(value, 0, color=c, s=50, zorder=5, clip_on=False, marker="^")

    mu, sigma, median = data.mean(), data.std(), data.median()
    textstr = "\n".join(
        (
            f"$N = {len(data)}$",
            rf"$\mu = {mu:.1f}${u}",
            rf"$\sigma = {sigma:.1f}${u}",
            f"Median $= {median:.1f}${u}",
        )
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.9, edgecolor="gray")
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11, verticalalignment="top", horizontalalignment="right", bbox=props)

    ax.set_xlabel(xlabel, fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title(title, pad=15, fontweight="bold")

    sns.despine(top=True, right=True)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=True)

    fig.tight_layout()
    folder = output.replace("\\", "/").rpartition("/")[0]
    if folder:
        mkdir(folder)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"[Done] saved to: {output}")
    return output
