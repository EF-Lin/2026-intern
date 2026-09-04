from __future__ import annotations

import io
from math import ceil
from typing import Any, Optional, Self

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, Trace, UTCDateTime
from PIL import Image
from tqdm import tqdm

from src.utils import check_file, mkdir, obtime_range_convt, tr2array


def single_wave(
    ax: plt.Axes,
    tr: Trace,
    time_range: tuple[UTCDateTime, UTCDateTime],
    *,
    ylim: tuple | float | int = None,
    xname: str = "Time (UTC)",
    yname: str = "Amplitude",
    title: str = "Wave Plot",
    linewidth: float = 0.6,
    color: str = "black",
) -> plt.Axes:
    x, y, lim = tr2array(tr, ylim=ylim)

    ax.plot(x, y, linewidth=linewidth, color=color)
    ax.set_title(title)

    # X
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel(xname)

    # lock X
    x_min = mdates.date2num(time_range[0].datetime)
    x_max = mdates.date2num(time_range[1].datetime)
    ax.set_xlim(x_min, x_max)

    # Y
    ax.set_ylabel(yname)
    if lim is not None:
        ax.set_ylim(lim[0], lim[1])

    return ax


class Finger:
    """
    ### Cut and Process Wave
    #### input
    st: Stream.

    r: time range before and after.

    frequency: filter frequency.

    figsize: figure size.

    #### method
    process: process patch.

    focus_plot: draw a figure within a specific time period.

    focus_save: save a png figure within a specific time period.
    """

    def __init__(
        self,
        st: Stream,
    ):
        self.st = st
        self._time_range = (self.st[0].stats.starttime, self.st[0].stats.starttime + 20)
        self.st_len = len(self.st)

    @property
    def time_range(self) -> tuple[str, str]:
        return str(self._time_range[0]).split('.')[0], str(self._time_range[1]).split('.')[0]

    def set_time_range(self, r: tuple[Any, Any]) -> Self:
        self._time_range = obtime_range_convt(r)
        return self

    def __str__(self):
        return str(self.st[0].stats.station)

    def process(self, frequency: Optional[tuple[int, int]] = (1, 20)) -> Self:
        self.st.detrend("demean")
        self.st.detrend("linear")
        self.st.taper(max_percentage=0.05, type="hann")
        self.st.filter("bandpass", freqmin=frequency[0], freqmax=frequency[1])
        return self

    def cut(self) -> Self:
        self.st.trim(starttime=self._time_range[0], endtime=self._time_range[1] + 1)
        return self

    def select(self) -> Stream:
        return self.st.slice(starttime=self._time_range[0], endtime=self._time_range[1])

    def _build_fig(
        self,
        ylim: Optional[float | tuple[float, float]] = None,
        xname: str = "Time (UTC)",
        yname: str = "Amplitude",
        figsize: Optional[tuple[int, int]] = (24, 6),
    ) -> plt.Figure:
        t0 = self._time_range[0]
        t1 = self._time_range[1]

        st_view = self.st.slice(starttime=t0, endtime=t1)
        n = len(st_view)

        fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
        axes = np.atleast_1d(axes)

        # Resolve ylim
        if ylim is None:
            y_lo, y_hi = None, None
        elif isinstance(ylim, (int, float)):
            y_lo, y_hi = -float(ylim), float(ylim)
        else:
            y_lo, y_hi = float(ylim[0]), float(ylim[1])

        for ax, tr in zip(axes, st_view):
            x, y = tr2array(tr)

            ax.plot(x, y, linewidth=0.6, color="black")
            ax.set_ylabel(f"{tr.id}\n{yname}", fontsize=8)

            if y_lo is not None:
                ax.set_ylim(y_lo, y_hi)

            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

        axes[-1].set_xlabel(xname)

        # Lock x-axis
        x_min = mdates.date2num(t0.datetime)
        x_max = mdates.date2num(t1.datetime)
        axes[0].set_xlim(x_min, x_max)  # sharex=True propagates to all subplots

        fig.tight_layout()
        return fig

    def set_plot(
        self,
        vertical: bool = True,
        figsize: Optional[tuple[int, int]] = (24, 6),
        title: Optional[list[str]] | str = None,
        **kwargs,
    ) -> Self:
        if isinstance(title, str):
            fig_title = [title for _ in range(self.pa_len)]
        elif title:
            fig_title = title
        else:
            fig_title = ["Waterfall Plot" for _ in range(self.pa_len)]

        ncols = int(self.st_len**0.5)
        nrows = int(ceil(self.st_len / ncols))

        if not vertical:
            ncols, nrows = nrows, ncols

        fs = (figsize[0] * ncols, figsize[1] * nrows)
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fs)
        ax_flat = np.atleast_1d(self.ax).flatten()

        for ax, tr, ti in zip(ax_flat, self.st, fig_title):
            single_wave(ax, tr, time_range=self._time_range, title=ti, **kwargs)

        for ax in ax_flat[self.pa_len :]:
            ax.set_visible(False)

        self.fig.tight_layout()
        return self

    def _gif(
        self,
        *,
        d: int = 10,
        jump: int = 100,
        dpi: int = 200,
        frame: int = 100,
        savimg: bool = False,
        filename: str = "Figure_gif",
        **kwargs,
    ) -> Self:
        """
        d: sec
        jump, frame: ms
        """
        folder = mkdir("./image")

        imgs = []
        i = self._time_range[0]
        pbar = tqdm(
            total=int((self._time_range[1] - d - self._time_range[0]) * 1000 / jump + 1),
            desc="Generating GIF images",
        )
        while i <= (self._time_range[1] - d):
            self.set_plot(**kwargs)
            buff = io.BytesIO()
            self.st.plot(fig=self.fig, starttime=i, endtime=i + d, outfile=buff, dpi=dpi)
            buff.seek(0)
            imgs.append(Image.open(buff).convert("RGB"))
            plt.close(self.fig)

            i += jump / 1000
            pbar.update(1)
        pbar.close()

        imgs[0].save(
            check_file(f"{folder}/{self._time_range[0].strftime("%Y-%m-%dT%H%M%S")}_to_{self._time_range[1].strftime("%Y-%m-%dT%H%M%S")}_wave_plot.gif"),
            format="GIF",
            save_all=True,
            append_images=imgs[1:],
            duration=frame,
            loop=1,
        )

    def focus_plot(self) -> Self:
        plt.show()
        return self

    def focus_save(
        self,
        *,
        folder: str = "./image",
        dpi: int = 300,
        filename: str = None,
    ) -> Self:
        filename = f"{self._time_range.strftime("%Y-%m-%dT%H%M%S")}_wave_plot" if filename is None else filename
        folder = mkdir(folder)
        self.fig.savefig(check_file(f"{folder}/{filename}.png"), dpi=dpi, bbox_inches="tight")
        plt.close(self.fig)
        return self
