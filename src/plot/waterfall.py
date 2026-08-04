import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# from matplotlib.colors import TwoSlopeNorm
from dascore import Patch
from typing import Optional


class Water:
    def __init__(
            self,
            pa: Patch,
            frequency: Optional[tuple[int, int]] = (1, 20),
            range: Optional[tuple[np.timedelta64, int]] = None
        ):
        self.pa: Patch = pa
        self.frequency: tuple = frequency
        self.r = (range[0] - np.timedelta64(range[1], 's'), range[0] + np.timedelta64(range[1], 's')) if range != None else (self.pa.attrs.time_min, self.pa.attrs.time_min + np.timedelta64(10, 's'))

    def cut(self):
        self.pa = self.pa.select(time=self.r)

    def process(self) -> Patch:
        self.pa = (
            self.pa
            .detrend(dim='time', type='constant')  # demean
            .detrend(dim='time', type='linear')    # linear
            .pass_filter(time=self.frequency)
        )
        return self.pa

class Fall:
    def __init__(
            self,
            pa: Patch,
            # y_factor: Optional[int] = 1,
            name: Optional[str] = None
        ):
        self.pa = pa
        # self.y_factor = y_factor
        self.name = name if name != None else "Figure"

    def set_plot(self):
        _, ax = plt.subplots()
        self.pa.viz.waterfall(cmap="seismic", ax=ax, scale=0.3)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.set_title(f"{str(self.pa.attrs.time_min.astype("datetime64[m]")).replace('T', '')} Waterfall Plot")

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
        self.set_plot()
        plt.show()

    def waterfall_save(self):
        self.set_plot()
        plt.savefig(f"image/{self.name}.png", dpi=200)
