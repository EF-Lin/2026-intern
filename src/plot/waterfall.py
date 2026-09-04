from __future__ import annotations

import io
from math import ceil
from typing import Any, Optional, Self

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from dascore import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from tqdm import tqdm

from src.utils import check_file, mkdir, nptime_range_convt, patch2figdata


def single_waterfall(
    ax: plt.Axes,
    pa: Patch,
    *,
    scale: float | tuple = 0.95,
    cmap: str = "seismic",
    cname: str = "Amplitude",
    xname: str = "Time (UTC)",
    yname: str = "depth",
    title: str = "Waterfall Plot",
) -> plt.Axes:
    data, extent, norm = patch2figdata(pa, scale=scale)

    # imshow
    im = ax.imshow(
        data,
        extent=extent,
        aspect="auto",
        cmap=cmap,
        origin="upper",
        norm=norm,
        interpolation="antialiased",
    )

    # Colour-bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.05)
    cbar = ax.get_figure().colorbar(im, cax=cax)
    cbar.set_label(cname, rotation=270, labelpad=15)

    # X
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    ax.set_xlabel(xname)

    # Y
    ax.set_ylabel(yname)

    # title
    ax.set_title(title)

    return ax


class Water:
    """
    ### Cut and Process Patch
    #### input
    - pa: data, Patch.

    #### method
    cut: cut patch within a certain time range
    - r: time range, (Any, Any)

    select: return patch within a certain time range
    - r: time range, (Any, Any)

    process: process patch
    - taper: taper time, float
    - frequency: filter frequency, tuple
    """

    def __init__(
        self,
        pa: Patch,
    ):
        self.pa: Patch = pa
        self.shape = self.pa.data.shape

    def __str__(self):
        return f"{self.shape[0]}x{self.shape[1]} Array"

    def cut(self, r: tuple[Any, Any]) -> Self:
        self.r = nptime_range_convt(r)
        self.pa = self.pa.select(time=self.r)
        return self

    def select(self, r: tuple[Any, Any] = None) -> Patch:
        r = nptime_range_convt(r)
        return self.pa.select(time=(r[0], r[1]))

    def process(
        self,
        taper: float = 0.01,
        frequency: Optional[tuple[int, int]] = (1, 20),
    ) -> Self:
        self.pa = (
            self.pa.detrend(dim="time", type="constant")  # demean
            .detrend(dim="time", type="linear")  # linear
            .taper(time=taper)  # taper
            .pass_filter(time=frequency)  # filter
        )
        return self


class Fall:
    """
    ### Plot Waterfall figure
    #### input
    - pa: data, Patch

    #### method
    set_plot: plot

    - title: figure title, list or string
    - figsize: single figure size, tuple
    - scale: color bar limit, float ot tuple
    - cmap: defaule value "seismic", string
    - cname: color bar name, string
    - xname: x-axis name, string
    - yname: y-axis name, string

    gif: generate gif
    - title: figure title, string
    - figsize: single figure size, tuple
    - scale: color bar limit, float ot tuple
    - cmap: defaule value "seismic", string
    - cname: color bar name, string
    - xname: x-axis name, string
    - yname: y-axis name, string

    waterfall_plot: show plot

    waterfall_save: save png figure

    """

    def __init__(
        self,
        pa: Patch | list[Patch],
    ):
        self.pa = pa if isinstance(pa, list) else [pa]
        self.pa_len = len(self.pa)

    def __str__(self):
        return f"{self.pa_len} Patches inside."

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

        ncols = int(self.pa_len**0.5)
        nrows = int(ceil(self.pa_len / ncols))

        if not vertical:
            ncols, nrows = nrows, ncols

        fs = (figsize[0] * ncols, figsize[1] * nrows)
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fs)
        ax_flat = np.atleast_1d(self.ax).flatten()

        for ax, pa, ti in zip(ax_flat, self.pa, fig_title):
            single_waterfall(ax, pa, title=ti, **kwargs)

        for ax in ax_flat[self.pa_len :]:
            ax.set_visible(False)

        self.fig.tight_layout()
        return self

    def gif(
        self,
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
        start = [pa.attrs.time_min for pa in self.pa]
        end = [pa.attrs.time_max - np.timedelta64(d, 's') for pa in self.pa]

        wa_li = [Water(pa) for pa in self.pa]

        path = mkdir("./image")
        if savimg:
            k = 0
            folder = mkdir(f"{path}/{filename}")

        imgs = []
        i = start[0]
        pbar = tqdm(
            total=((min(end) - min(start)).astype("timedelta64[ms]").astype(int)) * 1000 / jump + 1,
            desc="Generating GIF images",
        )
        while i <= min(end):
            self.pa = [wa.select(r=(t, t + np.timedelta64(d, 's'))) for wa, t in zip(wa_li, start)]
            self.set_plot(**kwargs)
            buff = io.BytesIO()
            self.fig.savefig(buff, format="jpg", dpi=dpi)
            buff.seek(0)
            imgs.append(Image.open(buff).convert("RGB"))
            if savimg:
                self.waterfall_save(folder=folder, filename=str(k), dpi=dpi)
                k += 1
            else:
                plt.close(self.fig)

            start = [j + np.timedelta64(jump, 'ms') for j in start]
            i = min(start)
            pbar.update(1)

        pbar.close()
        self.pa = [wa.pa for wa in wa_li]

        imgs[0].save(
            check_file(f"{path}/{filename}.gif"),
            format="GIF",
            save_all=True,
            append_images=imgs[1:],
            duration=frame,
            loop=0,
        )
        return self

    def waterfall_plot(self) -> Self:
        plt.show()
        return self

    def waterfall_save(
        self,
        folder: str = "image/",
        dpi: int = 200,
        filename: str = "Figure",
    ) -> Self:
        mkdir(folder)
        self.fig.savefig(check_file(f"{folder}/{filename}.png"), dpi=dpi, bbox_inches="tight")
        plt.close(self.fig)
        return self
