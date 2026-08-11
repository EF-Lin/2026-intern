from __future__ import annotations

import io
from typing import Any, Optional

import matplotlib.pyplot as plt
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

    def set_time_range(self, focus_time: UTCDateTime | str | int | Any, r: Optional[int] = 10):
        self.focus_time = UTCDateTime(focus_time)
        self._time_range = (self.focus_time - r, self.focus_time + r)

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

    def focus_plot(self):
        self.st.plot(starttime=self._time_range[0], endtime=self._time_range[1])

    def focus_save(self, folder: str = "/image"):
        self.fig = plt.figure(figsize=self.figsize)
        folder = mkdir(folder)
        self.st.plot(
            fig=self.fig,
            starttime=self._time_range[0],
            endtime=self._time_range[1],
            outfile=check_file(f"{folder}/{self.focus_time.strftime("%Y-%m-%dT%H%M%S")}_wave_plot.png"),
            dpi=200,
            # size=(5000, 750)
        )
        plt.close(self.fig)

    def gif(self, d: int = 10, jump: int = 100, dpi: int = 200, frame: int = 100):
        """
        d: sec
        jump, frame: ms
        """
        folder = mkdir("./image")

        imgs = []
        i = self._time_range[0]
        pbar = tqdm(total=int((self.time_range[1] - self.time_range[0]) * 1000 / jump), desc="Generating GIF images")
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
