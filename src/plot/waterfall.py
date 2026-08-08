import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from matplotlib.colors import TwoSlopeNorm
from dascore import Patch
from typing import Optional
from src.utils import mkdir, check_file
from math import ceil


class Water:
    def __init__(
            self,
            pa: list[Patch],
            *,
            frequency: Optional[tuple[int, int]] = (1, 20),
            r: Optional[tuple[np.timedelta64, int]] = None
        ):
        self.pa: Patch = pa
        self.frequency: tuple = frequency
        self.r = (r[0] - np.timedelta64(r[1], 's'), r[0] + np.timedelta64(r[1], 's')) if r != None else (self.pa.attrs.time_min, self.pa.attrs.time_min + np.timedelta64(10, 's'))

    def cut(self):
        return self.pa.select(time=self.r)

    def process(self, pa: Patch) -> Patch:
        return (
            pa
            .detrend(dim='time', type='constant') # demean
            .detrend(dim='time', type='linear') # linear
            .taper(time=0.01) # taper
            .pass_filter(time=self.frequency)
        )

class Fall:
    def __init__(
            self,
            pa: Patch,
            *,
            filename: Optional[list[str]] = None,
            title: Optional[list[str]],
            figsize: Optional[tuple[int, int]] = (24, 6)
        ):
        self.pa = pa
        self.pa_len = len(self.pa)
        self.filename = filename if filename else ["Figure" for _ in range(self.pa_len)]
        self.title = title if title else ["Waterfall Plot" for _ in range(self.pa_len)]
        self.figsize = figsize

    def set_plot(self):
        nrows = int(self.pa_len**0.5)
        ncols = int(ceil(self.pa_len/nrows))
        self.fig, self.ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=self.figsize)
        ax_flat = np.atleast_1d(self.ax).flatten()

        for ax, pa, title in zip(ax_flat, self.pa, self.title):
            # water fall
            pa.viz.waterfall(cmap="seismic", ax=ax, scale=0.3)
            # color bar
            self.fig.axes[-1].set_ylabel("Amplitude", rotation=270, labelpad=15)
            # upsidedown
            ax.invert_yaxis()
            # set x-axis
            self.ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
            self.ax.set_xlabel("Time (UTC)")
            # set title
            ax.set_title(title)

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

    def waterfall_plot(self):
        plt.show()

    def waterfall_save(self):
        mkdir("image/")
        plt.savefig(check_file(f"image/{self.filename}.png"), dpi=200)
        plt.close(self.fig)
