from __future__ import annotations

import io
from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from obspy import Stream, UTCDateTime
from PIL import Image
from tqdm import tqdm

from src.utils import check_file, mkdir


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
        *,
        # n: Optional[int] = 0,
        frequency: Optional[tuple[int, int]] = (1, 20),
        figsize: Optional[tuple[int, int]] = (24, 6),
    ):
        self.st = st
        self._time_range = (self.st[0].stats.starttime, self.st[0].stats.starttime + 20)
        # self.n = n
        self.frequency = frequency
        self.figsize = figsize

    @property
    def time_range(self) -> tuple[UTCDateTime, UTCDateTime]:
        return self._time_range

    def set_time_range(self, focus_time: UTCDateTime | str | int | Any, r: Optional[int] = 10) -> Finger:
        self.focus_time = UTCDateTime(focus_time)
        self._time_range = (self.focus_time - r, self.focus_time + r)
        return self

    def __str__(self):
        return str(self.st[0].stats.station)

    def process(self) -> Finger:
        self.st.detrend("demean")
        self.st.detrend("linear")
        self.st.taper(max_percentage=0.05, type="hann")
        self.st.filter("bandpass", freqmin=self.frequency[0], freqmax=self.frequency[1])
        return self

    def cut(self) -> Finger:
        self.st.trim(starttime=self._time_range[0], endtime=self._time_range[1] + 1)
        return self

    def _build_fig(
        self,
        ylim: Optional[float | tuple[float, float]] = None,
        xname: str = "Time (UTC)",
        yname: str = "Amplitude",
        figsize: Optional[tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        Internal helper: build a matplotlib Figure of per-trace waveforms
        within the current time range.  Called by both :meth:`focus_plot`
        and :meth:`focus_save`.
        """
        t0 = self._time_range[0]
        t1 = self._time_range[1]

        st_view = self.st.slice(starttime=t0, endtime=t1)
        n = len(st_view)

        fig, axes = plt.subplots(n, 1, figsize=figsize or self.figsize, sharex=True)
        axes = np.atleast_1d(axes)

        # Resolve ylim
        if ylim is None:
            y_lo, y_hi = None, None
        elif isinstance(ylim, (int, float)):
            y_lo, y_hi = -float(ylim), float(ylim)
        else:
            y_lo, y_hi = float(ylim[0]), float(ylim[1])

        for ax, tr in zip(axes, st_view):
            t_arr = np.array(
                [tr.stats.starttime.datetime + np.timedelta64(int(i * tr.stats.delta * 1e6), 'us') for i in range(tr.stats.npts)],
                dtype="datetime64[us]",
            )
            t_num = mdates.date2num(t_arr.astype("O"))

            ax.plot(t_num, tr.data, linewidth=0.6, color="black")
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

    def focus_plot(
        self,
        ylim: Optional[float | tuple[float, float]] = None,
        xname: str = "Time (UTC)",
        yname: str = "Amplitude",
        figsize: Optional[tuple[int, int]] = None,
    ) -> Finger:
        """
        Draw the waveform of every trace within the current time range.

        Parameters
        ----------
        ylim:
            Y-axis limits applied to every subplot.

            * ``None``        - auto (matplotlib default)
            * ``float``       - symmetric: ``(-ylim, +ylim)``
            * ``(lo, hi)``    - explicit lower / upper bound
        xname:
            X-axis label.
        yname:
            Y-axis label (shared).
        figsize:
            Figure size ``(width, height)`` in inches.
            Defaults to the value set at construction time.
        """
        self.fig = self._build_fig(ylim=ylim, xname=xname, yname=yname, figsize=figsize)
        plt.show()
        return self

    def focus_save(
        self,
        folder: str = "./image",
        ylim: Optional[float | tuple[float, float]] = None,
        xname: str = "Time (UTC)",
        yname: str = "Amplitude",
        dpi: int = 200,
        figsize: Optional[tuple[int, int]] = None,
    ) -> Finger:
        """
        Save the waveform figure to a PNG file.

        Parameters
        ----------
        folder:
            Output directory (created automatically if absent).
        ylim:
            Y-axis limits - same semantics as :meth:`focus_plot`.
        xname:
            X-axis label.
        yname:
            Y-axis label (shared).
        dpi:
            Resolution of the saved image.
        figsize:
            Figure size ``(width, height)`` in inches.
            Defaults to the value set at construction time.
        """
        folder = mkdir(folder)
        self.fig = self._build_fig(ylim=ylim, xname=xname, yname=yname, figsize=figsize)
        path = check_file(f"{folder}/{self.focus_time.strftime('%Y-%m-%dT%H%M%S')}_wave_plot.png")
        self.fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(self.fig)
        return self

    def gif(self, d: int = 10, jump: int = 100, dpi: int = 200, frame: int = 100):
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
            buff = io.BytesIO()
            self.fig = plt.figure(figsize=self.figsize)
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
