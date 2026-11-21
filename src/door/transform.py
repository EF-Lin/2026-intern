import warnings
from typing import Optional

import numpy as np
from dascore import Patch, write
from obspy import Stream

from src.utils import check_file


def transfer(
    st: Stream,
    spacing: float = 4.0,
    start: float = 0.0,
    *,
    r: Optional[float] = None,
) -> tuple[Patch, tuple[str, str]]:
    """
    Convert an obspy Stream (one trace per DAS channel) into a dascore Patch with dims (depth, time).

    Parameters
    ----------
    st : Stream
        Traces whose ``stats.station`` is the channel number. NOTE: ``st`` is trimmed in place
        to the common time window of all traces.
    spacing : float
        Channel spacing in metres (default 4.0).
    start : float
        Depth in metres assigned to the first trace ``st[0]``.
    r : float, optional
        Deprecated alias of ``spacing``.

    Depth of each trace = ``start + (station - station_of_st[0]) * spacing`` (metres).

    Returns
    -------
    (Patch, (first_station, last_station))
    """
    if r is not None:
        warnings.warn("transfer(r=...) is deprecated; use spacing=...", DeprecationWarning, stacklevel=2)
        spacing = r

    st.trim(starttime=max([tr.stats.starttime for tr in st]), endtime=min([tr.stats.endtime for tr in st]))
    n = min([len(tr.data) for tr in st])
    data = np.vstack([tr.data[:n] for tr in st])

    # nanosecond resolution with rounding, so e.g. 1000 Hz (delta=0.001) is not truncated to 999 us
    step = np.timedelta64(int(round(st[0].stats.delta * 1e9)), "ns")
    time = np.datetime64(int(st[0].stats.starttime.ns), "ns") + np.arange(data.shape[1]) * step

    stations = np.array([int(tr.stats.station) for tr in st])
    depth = start + (stations - stations[0]) * spacing

    patch = Patch(
        data=data,
        coords={
            "time": time,
            "depth": depth,
        },
        dims=("depth", "time"),
    )

    return patch, (st[0].stats.station, st[-1].stats.station)


def save(pa: Patch, name: str = "untitled") -> Patch:
    write(pa, path=check_file(f"{name}.h5"), file_format="dasdae")
    return pa
