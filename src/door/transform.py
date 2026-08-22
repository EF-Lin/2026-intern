from typing import Optional

import numpy as np
from dascore import Patch, write
from obspy import Stream

from src.utils import check_file


def transfer(st: Stream, r: Optional[int] = 4, start: int = 0) -> tuple[Patch, tuple[str, str]]:
    st.trim(starttime=max([tr.stats.starttime for tr in st]), endtime=min([tr.stats.endtime for tr in st]))
    data = np.vstack([tr.data[: min([len(tr.data) for tr in st])] for tr in st])
    time = np.datetime64(st[0].stats.starttime.datetime) + np.arange(data.shape[1]) * np.timedelta64(int(st[0].stats.delta * 1e6), "us")

    fi = int(st[0].stats.station)
    depth = np.array([])
    stations = []
    for tr in st:
        depth = np.append(depth, start + (int(tr.stats.station) - fi) * r)
        stations.append(str(tr.stats.station))

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
    write(check_file(f"{name}.h5"), file_format="dasdae")
    return pa
