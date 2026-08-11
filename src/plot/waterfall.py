from __future__ import annotations

import io
from math import ceil
from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from dascore import Patch

# from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from tqdm import tqdm

from src.utils import check_file, mkdir, time_range_convt


class Water:
    """
    ### Cut and Process Patch
    #### input
    pa: Patch

    frequency: filter frequency

    #### method
    cut: cut patch within a certain time range

    select: return patch within a certain time range

    process: process patch
    """

    def __init__(
        self,
        pa: Patch,
        *,
        frequency: Optional[tuple[int, int]] = (1, 20),
    ):
        self.pa: Patch = pa
        self.frequency: tuple = frequency
        self.shape = self.pa.data.shape

    def __str__(self):
        return f"{self.shape[0]}x{self.shape[1]} Array"

    def cut(self, r: tuple[Any, Any] = None) -> Water:
        self.r = time_range_convt(r)
        self.pa = self.pa.select(time=self.r)
        return self

    def select(self, r: tuple[Any, Any] = None) -> Patch:
        return self.pa.select(time=(r[0], r[1]))

    def process(self) -> Water:
        self.pa = (
            self.pa.detrend(dim='time', type='constant')  # linear
            .detrend(dim='time', type='linear')  # demean
            .taper(time=0.01)  # taper
            .pass_filter(time=self.frequency)  # filter
        )
        return self


class Fall:
    """
    ### Plot Waterfall figure
    #### input
    pa: Patch.

    filename: output file name.

    title: figure title.

    figsize: single figure size.

    #### method
    set_plot: plot.

    gif: genertate gif.

    waterfall_plot: show plot.

    waterfall_save; save png figure.

    """

    def __init__(
        self,
        pa: Patch | list[Patch],
        *,
        filename: Optional[str] = None,
        title: Optional[list[str]] = None,
        figsize: Optional[tuple[int, int]] = (24, 6),
    ):
        self.pa = pa if isinstance(pa, list) else [pa]
        self.pa_len = len(self.pa)
        self.filename = filename if filename else "Figure"
        self.title = title if title else ["Waterfall Plot" for _ in range(self.pa_len)]
        self.figsize = figsize
        self.vrange = None

    def __str__(self):
        return f"{self.pa_len} Patches inside."

    def set_plot(self, xname: str = "Time (UTC)", yname: str = "depth", vrange: list[tuple] = None, vertical: bool = True) -> Fall:
        ncols = int(self.pa_len**0.5)
        nrows = int(ceil(self.pa_len / ncols))

        if not vertical:
            ncols, nrows = nrows, ncols

        fs = (self.figsize[0] * ncols, self.figsize[1] * nrows)
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fs)

        self.vrange = vrange if vrange else self.vrange
        ax_flat = np.atleast_1d(self.ax).flatten()

        for i, (ax, pa, title) in enumerate(zip(ax_flat, self.pa, self.title)):
            # water fall
            if self.vrange:
                pa.viz.waterfall(cmap="seismic", ax=ax, scale=self.vrange[i], scale_type="absolute")
            else:
                pa.viz.waterfall(cmap="seismic", ax=ax, scale=0.3)
            # color bar
            self.fig.axes[-1].set_ylabel("Amplitude", rotation=270, labelpad=15)
            # upsidedown
            ax.invert_yaxis()
            # set x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            ax.set_xlabel(xname)
            # set y-axis
            ax.set_ylabel(yname)
            # set title
            ax.set_title(title)
            # layout
            self.fig.tight_layout()

        return self

    def gif(self, d: int = 10, jump: int = 100, dpi: int = 200, frame: int = 100, savimg: bool = False) -> None:
        """
        d: sec
        jump, frame: ms
        """
        start = [pa.attrs.time_min for pa in self.pa]
        end = [pa.attrs.time_max - np.timedelta64(d, 's') for pa in self.pa]

        wa_li = [Water(pa) for pa in self.pa]

        self.vrange = []
        for pa in self.pa:
            m = np.max([np.percentile(np.abs(papa.data), 95) for papa in pa.data])
            self.vrange.append((-m, m))

        path = mkdir("./image")
        if savimg:
            k = 0
            folder = mkdir(f"{path}/{self.filename}")

        imgs = []
        i = start[0]
        pbar = tqdm(
            total=((end[0] - start[0]).astype("timedelta64[s]").astype(int) + 1) * 1000 / jump,
            desc="Generating GIF images",
        )
        while i <= end[0]:
            self.pa = [wa.select(r=(t, t + np.timedelta64(d, 's'))) for wa, t in zip(wa_li, start)]
            self.set_plot()
            if savimg:
                self.filename = str(k)
                self.waterfall_save(folder=folder)
                k += 1
            buff = io.BytesIO()
            plt.savefig(buff, format="jpg", dpi=dpi)
            buff.seek(0)
            imgs.append(Image.open(buff).convert("RGB"))
            plt.close(self.fig)

            start = [j + np.timedelta64(jump, 'ms') for j in start]
            i = start[0]
            pbar.update(1)

        pbar.close()
        self.pa = [wa.pa for wa in wa_li]

        imgs[0].save(
            check_file(f"{path}/{self.filename}.gif"),
            format="GIF",
            save_all=True,
            append_images=imgs[1:],
            duration=frame,
            loop=1,
        )

    """
    def cut(self):
        self.st.trim(starttime=self.range[0], endtime=self.range[1])


    def set_plot(self):
        self.data = np.array([tr.data for tr in self.st])
        _, ax = plt.subplots(figsize=(12, 6))

        im = ax.imshow(
            self.data,
            aspect="auto",
            cmap="seismic",
            origin="upper",
            extent=[date2num(self.range[0].datetime), date2num(self.range[1].datetime), self.data.shape[0]-0.5, -0.5],
            norm=TwoSlopeNorm(vcenter=0, vmin=self.data.min(), vmax=self.data.max())
        )

        # if self.frequency != None:
        #     im = ax.imshow(vmin=self.frequency(0), vmax=self.frequency(1))

        # color bar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Amplitude")

        # Y
        trace_ids = [tr.id for tr in self.st]
        n_traces = len(trace_ids)

        ax.set_yticks(range(0, n_traces, self.y_factor))
        ax.set_yticklabels([trace_ids[i] for i in range(0, n_traces, self.y_factor)])
        ax.set_ylabel("Trace")

        # X
        ax.set_xlabel("Time (UTC)")
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))

        ax.set_title(f"{self.range[0].strftime("%Y-%m-%d")} {self.st[0].id} to {self.st[0-1].id} Waterfall Plot")
        plt.setp(ax.xaxis.get_majorticklabels(), ha="center")
        plt.tight_layout()
    """

    def waterfall_plot(self) -> Fall:
        plt.show()
        return self

    def waterfall_save(self, folder: str = "image/", dpi: int = 200) -> Fall:
        mkdir(folder)
        plt.savefig(check_file(f"{folder}/{self.filename}.png"), dpi=dpi)
        plt.close(self.fig)
        return self
