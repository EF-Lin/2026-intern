from obspy import Stream, UTCDateTime
from typing import Optional, Any
from src.utils import mkdir, check_file
import matplotlib.pyplot as plt


class Finger:
    def __init__(
            self,
            stream: Stream,
            *,
            r: Optional[float] = 10,
            n: Optional[int] = 0,
            frequency: Optional[tuple[int, int]] = (1, 20),
            figsize: Optional[tuple[int, int]] = (24, 6)
    ):
        self.st = stream
        self.range = r
        self.n = n
        self.frequency = frequency
        self.figsize = figsize

    def process(self):
        self.st.detrend("demean")
        self.st.detrend("linear")
        self.st.taper(max_percentage=0.05, type="hann")
        self.st.filter("bandpass", freqmin=self.frequency[0], freqmax=self.frequency[1])

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
