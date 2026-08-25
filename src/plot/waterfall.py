from __future__ import annotations

import io
from math import ceil
from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from dascore import Patch
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from tqdm import tqdm

from src.utils import check_file, mkdir, time_range_convt


def _resolve_scale(data: np.ndarray, scale) -> tuple[float, float]:
    if scale is None:
        v = float(np.percentile(np.abs(data), 95))
        return -v, v
    if isinstance(scale, (int, float)):
        v = float(np.max(np.abs(data))) * float(scale)
        return -v, v
    return float(scale[0]), float(scale[1])


def _draw_waterfall(
    ax: plt.Axes,
    pa: Patch,
    scale=None,
    cmap: str = "seismic",
    cbar_label: str = "Amplitude",
) -> plt.Axes:
    dims = pa.dims  # e.g. ("depth", "time")

    # Identify the space dimension (everything that is not "time")
    space_dim = next(d for d in dims if d != "time")

    time_arr = pa.coords.get_array("time")  # datetime64[ns]
    space_arr = pa.coords.get_array(space_dim)  # float (m)
    data = pa.data  # shape: (n_space, n_time)

    if dims.index(space_dim) != 0:
        data = data.T

    time_num = mdates.date2num(time_arr.astype("datetime64[ms]").astype("O"))

    extent = [float(time_num[0]), float(time_num[-1]), float(space_arr[0]), float(space_arr[-1])]

    # Colour scale
    vmin, vmax = _resolve_scale(data, scale)
    if vmin >= vmax:
        vmax = max(abs(vmin), abs(vmax), 1e-9)
        vmin = -vmax
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)

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
    cbar.set_label(cbar_label, rotation=270, labelpad=15)

    # X
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

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

    filename: output file name.

    title: figure title.

    figsize: single figure size.

    #### method
    set_plot: plot.

    gif: genertate gif.

    waterfall_plot: show plot.

    waterfall_save: save png figure.

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

    def set_plot(
        self,
        xname: str = "Time (UTC)",
        yname: str = "depth",
        scale=None,
        vrange: list[tuple] = None,
        cmap: str = "seismic",
        vertical: bool = True,
    ) -> Fall:
        ncols = int(self.pa_len**0.5)
        nrows = int(ceil(self.pa_len / ncols))

        if not vertical:
            ncols, nrows = nrows, ncols

        fs = (self.figsize[0] * ncols, self.figsize[1] * nrows)
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=fs)

        # Normalise vrange (legacy) → scale list
        effective_scale = vrange if vrange else self.vrange
        if effective_scale is None:
            # Use the single scale value (or None) for every patch
            scale_list = [scale] * self.pa_len
        else:
            scale_list = effective_scale  # list of (vmin, vmax) tuples

        ax_flat = np.atleast_1d(self.ax).flatten()

        for ax, pa, title, sc in zip(ax_flat, self.pa, self.title, scale_list):
            _draw_waterfall(ax, pa, scale=sc, cmap=cmap)
            ax.set_xlabel(xname)
            ax.set_ylabel(yname)
            ax.set_title(title)

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

        # Pre-compute absolute vrange from the full data so the colour scale is
        # locked across all GIF frames.
        self.vrange = []
        for pa in self.pa:
            m = float(np.percentile(np.abs(pa.data), 95))
            self.vrange.append((-m, m))

        path = mkdir("./image")
        if savimg:
            k = 0
            folder = mkdir(f"{path}/{self.filename}")

        imgs = []
        i = start[0]
        pbar = tqdm(
            total=((end[0] - start[0]).astype("timedelta64[s]").astype(int) - d) * 1000 / jump + 1,
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

    def waterfall_plot(self) -> Fall:
        plt.show()
        return self

    def waterfall_save(self, folder: str = "image/", dpi: int = 200) -> Fall:
        mkdir(folder)
        plt.savefig(check_file(f"{folder}/{self.filename}.png"), dpi=dpi)
        plt.close(self.fig)
        return self
