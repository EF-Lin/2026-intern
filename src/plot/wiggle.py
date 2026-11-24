from __future__ import annotations

from typing import Any, Optional, Self

import matplotlib.pyplot as plt
import numpy as np
from dascore import Patch

from src.utils import check_file, mkdir


def single_wiggle(
    ax: plt.Axes,
    pa: Patch,
    *,
    t0: Optional[np.datetime64] = None,
    every: int = 4,
    norm: float | None = None,
    per_trace: bool = False,
    gain: float = 1.0,
    color: str = "black",
    linewidth: float = 0.8,
    title: str = "Wiggle Plot",
    xname: str = "Time (s)",
    yname: str = "Depth (m)",
) -> plt.Axes:
    """
    Draw every ``every``-th channel as a wiggle trace, offset by its depth (depth increases downward).

    Parameters
    ----------
    t0 : datetime64, optional
        Time zero of the x axis (seconds after ``t0``); default is the first sample of ``pa``.
    norm : float, optional
        Amplitude that is drawn as one trace spacing. Pass the same value to several panels to share
        the amplitude scale. Default: the largest |amplitude| of the drawn traces.
    per_trace : bool
        Normalise every trace by its own maximum instead (``norm`` is then ignored).
    gain : float
        Extra multiplier on the drawn amplitude (1.0 = peak reaches the neighbouring trace).
    """
    data = pa.data if pa.dims == ("depth", "time") else pa.data.T
    depth = pa.coords.get_array("depth")
    time = pa.coords.get_array("time")
    t0 = time[0] if t0 is None else t0
    t = (time - t0) / np.timedelta64(1, "s")

    data, depth = data[::every], depth[::every]
    spacing = abs(float(depth[1] - depth[0])) if len(depth) > 1 else 1.0

    if per_trace:
        scale = np.max(np.abs(data), axis=1, keepdims=True)
        scale[scale == 0] = 1.0
    else:
        scale = norm if norm else float(np.max(np.abs(data))) or 1.0

    # positive amplitude drawn upward (towards smaller depth)
    traces = depth[:, None] - data / scale * spacing * gain
    ax.plot(t, traces.T, color=color, linewidth=linewidth)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(depth.max() + spacing * 1.5, depth.min() - spacing * 1.5)  # inverted: depth downward
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)
    ax.set_title(title)
    return ax


class Wiggle:
    """
    ### Plot wiggle (trace-by-depth) figures side by side
    #### input
    - pa: data, Patch or list[Patch]

    #### method
    set_plot: plot one panel per patch, in one row
    - title: figure title, list or string
    - figsize: single panel size, tuple
    - color: line colour, string or list
    - share_norm: use the first patch's max amplitude as the scale of every panel, bool
    - every / per_trace / gain / t0: see ``single_wiggle``

    wiggle_plot: show plot

    wiggle_save: save png figure
    """

    def __init__(self, pa: Patch | list[Patch]):
        self.pa = pa if isinstance(pa, list) else [pa]
        self.pa_len = len(self.pa)

    def __str__(self):
        return f"{self.pa_len} Patches inside."

    def set_plot(
        self,
        figsize: tuple[float, float] = (5, 8),
        title: Optional[list[str]] | str = None,
        color: str | list[str] = "black",
        share_norm: bool = True,
        every: int = 4,
        **kwargs: Any,
    ) -> Self:
        titles = [title] * self.pa_len if isinstance(title, str) else (title or ["Wiggle Plot"] * self.pa_len)
        colors = [color] * self.pa_len if isinstance(color, str) else color

        norm = None
        if share_norm and not kwargs.get("per_trace"):
            # the first patch (e.g. raw data) sets the scale so the panels can be compared directly
            first = self.pa[0].data if self.pa[0].dims == ("depth", "time") else self.pa[0].data.T
            norm = float(np.max(np.abs(first[::every]))) or None

        self.fig, axes = plt.subplots(1, self.pa_len, figsize=(figsize[0] * self.pa_len, figsize[1]), sharey=True)
        ax_flat = np.atleast_1d(axes).flatten()
        for i, (ax, pa, ti, co) in enumerate(zip(ax_flat, self.pa, titles, colors)):
            single_wiggle(ax, pa, title=ti, color=co, every=every, norm=norm, yname="Depth (m)" if i == 0 else "", **kwargs)

        self.fig.tight_layout()
        return self

    def wiggle_plot(self) -> Self:
        plt.show()
        return self

    def wiggle_save(self, folder: str = "image/", dpi: int = 200, filename: str = "Wiggle") -> Self:
        mkdir(folder)
        self.fig.savefig(check_file(f"{folder}/{filename}.png"), dpi=dpi, bbox_inches="tight")
        plt.close(self.fig)
        return self
