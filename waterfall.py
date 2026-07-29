import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, DateFormatter
from obspy import Stream, UTCDateTime
from typing import Optional


class Water:
    def __init__(
            self,
            st: Stream,
            frequency: Optional[tuple[int, int]] = (1, 20)
        ):
        self.st: Stream = st
        self.frequency: tuple = frequency

    def process(self):
        self.st.detrend("demean")
        self.st.detrend("linear")
        self.st.filter("bandpass", freqmin=self.frequency[0], freqmax=self.frequency[1])
        # self.st.normalize()
        return self.st

class Fall:
    def __init__(
            self,
            st: Stream,

            y_factor: Optional[int] = 1,
            range: Optional[tuple[UTCDateTime, int]] = None,
            name: Optional[str] = None
        ):
        self.st = st
        self.y_factor = y_factor

        self.range = (range[0]-range[1], range[0]+range[1]) if range != None else (max(tr.stats.starttime for tr in st), min(tr.stats.endtime for tr in st))

        self.name = name if name != None else f"{self.st[0].id}_to_{self.st[0-1].id}_waterfall_plot_{(self.range[0]).strftime("%Y-%m-%dT%H:%M:%S").replace(':', '')}"

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
            extent=[date2num(self.range[0].datetime), date2num(self.range[1].datetime), self.data.shape[0]-0.5, -0.5]
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
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d %H:%M:%S'))

        ax.set_title(f"{self.st[0].id} to {self.st[0-1].id} Waterfall Plot")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=10, ha="center")
        plt.tight_layout()

    def waterfall_plot(self):
        self.set_plot()
        plt.show()

    def waterfall_save(self):
        self.set_plot()
        plt.savefig(f"image/{self.name}.png", dpi=100)
