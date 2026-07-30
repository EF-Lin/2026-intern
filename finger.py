from obspy import Stream, UTCDateTime
from typing import Optional, Any


class Finger:
    def __init__(
            self,
            stream: Stream,
            focus_time: UTCDateTime | str | int | Any,
            range: Optional[float] = 10,
            n: Optional[int] = 0
    ):
        self.stream = stream
        self.focus_time = UTCDateTime(focus_time)
        self.range = range
        self.n = n

    def focus_plot(self):
        self.stream.plot(starttime=self.focus_time - self.range, endtime=self.focus_time + self.range)

    def focus_save(self):
        self.stream.plot(
            starttime=self.focus_time - self.range,
            endtime=self.focus_time + self.range,
            outfile=f"image/{UTCDateTime(self.focus_time - self.range).strftime("%Y-%m-%dT%H:%M:%S").replace(':', '')}_to_{UTCDateTime(self.focus_time + self.range).strftime("%Y-%m-%dT%H:%M:%S").replace(':', '')}_wave_plot.png")
