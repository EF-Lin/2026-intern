from __future__ import annotations

from typing import Optional, Any
import matplotlib.pyplot as plt
from obspy import Stream, UTCDateTime

from src.utils import mkdir, check_file


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
            r: Optional[float] = 10,
            #n: Optional[int] = 0,
            frequency: Optional[tuple[int, int]] = (1, 20),
            figsize: Optional[tuple[int, int]] = (24, 6)
    ):
        self.st = st
        self.range = r
        #self.n = n
        self.frequency = frequency
        self.figsize = figsize

    def __str__(self):
        return str(self.st[0].stats.station)

    def process(self) -> Finger:
        self.st.detrend("demean")
        self.st.detrend("linear")
        self.st.taper(max_percentage=0.05, type="hann")
        self.st.filter("bandpass", freqmin=self.frequency[0], freqmax=self.frequency[1])
        return self

    def focus_plot(self, focus_time: UTCDateTime | str | int | Any):
        focus_time = UTCDateTime(focus_time)
        self.st.plot(starttime=focus_time - self.range, endtime=focus_time + self.range)

    def focus_save(self, focus_time: UTCDateTime | str | int | Any):
        self.fig = plt.figure(figsize=self.figsize)
        focus_time = UTCDateTime(focus_time)
        mkdir("/image")
        self.st.plot(
            fig=self.fig,
            starttime=focus_time - self.range,
            endtime=focus_time + self.range,
            outfile=check_file(f"image/{UTCDateTime(focus_time - self.range).strftime("%Y-%m-%dT%H%M%S")}_to_{UTCDateTime(focus_time + self.range).strftime("%Y-%m-%dT%H%M%S")}_wave_plot-NS.png"),
            dpi=200,
            # size=(5000, 750)
        )
        plt.close(self.fig)
