from __future__ import annotations

import io
from math import ceil
from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from dascore import Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from tqdm import tqdm

from src.plot import patch2figdata
from src.utils import check_file, mkdir, time_range_convt


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
            self.pa.detrend(dim='time', type='constant')  # demean
            .detrend(dim='time', type='linear')  # linear
            .taper(time=0.01)  # taper
            .pass_filter(time=self.frequency)  # filter
        )
        return self


class Fall:
    """
    ### Plot Waterfall figure
    #### input
    pa: Patch.

    title: figure title.

    figsize: single figure size.

    #### method
    set_plot: plot.

    gif: generate gif.

    waterfall_plot: show plot.

    waterfall_save: save png figure.

    """

    def __init__(
        self,
        pa: Patch | list[Patch],
        *,
        title: Optional[list[str]] = None,
        figsize: Optional[tuple[int, int]] = (24, 6),
    ):
        self.pa = pa if isinstance(pa, list) else [pa]
        self.pa_len = len(self.pa)
        self.title = title if title else ["Waterfall Plot" for _ in range(self.pa_len)]
        self.figsize = figsize

    def __str__(self):
        return f"{self.pa_len} Patches inside."

    def set_plot(
        self,
        vertical: bool = True,
        **kwargs,
    ) -> Fall:
        ncols = int(self.pa_len**0.5)
        nrows = int(ceil(self.pa_len / ncols))

        if not vertical:
            ncols, nrows = nrows, ncols

        fs = (self.figsize[0] * ncols, self.figsize[1] * nrows)
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fs)
        ax_flat = np.atleast_1d(self.ax).flatten()

        for ax, pa, title in zip(ax_flat, self.pa, self.title):
            single_waterfall(ax, pa, title=title, **kwargs)

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
    ) -> None:
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
            total=((end[0] - start[0]).astype("timedelta64[s]").astype(int)) * 1000 / jump + 1,
            desc="Generating GIF images",
        )
        while i <= min(end):
            self.pa = [wa.select(r=(t, t + np.timedelta64(d, 's'))) for wa, t in zip(wa_li, start)]
            self.set_plot(**kwargs)
            buff = io.BytesIO()
            plt.savefig(buff, format="jpg", dpi=dpi)
            buff.seek(0)
            imgs.append(Image.open(buff).convert("RGB"))
            if savimg:
                self.waterfall_save(folder=folder, filename=str(k))
                k += 1
            else:
                plt.close(self.fig)

            start = [j + np.timedelta64(jump, 'ms') for j in start]
            i = start[0]
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

    def waterfall_plot(self) -> Fall:
        plt.show()
        return self

    def waterfall_save(
        self,
        folder: str = "image/",
        dpi: int = 200,
        filename: str = "Figure",
    ) -> Fall:
        mkdir(folder)
        plt.savefig(check_file(f"{folder}/{filename}.png"), dpi=dpi)
        plt.close(self.fig)
        return self
