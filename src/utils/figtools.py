from typing import Any

import matplotlib.dates as mdates
import numpy as np
from dascore import Patch
from matplotlib.colors import TwoSlopeNorm
from obspy import Trace


def patch2figdata(
    pa: Patch,
    scale: float | tuple,
) -> tuple[np.ndarray, list, TwoSlopeNorm]:
    """
    pa: Patch
    scale:
        float, 0.95
        tuple, (-50. 60)
    """
    data = pa.data

    dims = pa.dims
    x = mdates.date2num(pa.coords.get_array(dims[0]).astype("datetime64[ms]").astype("O"))  # datetime64[ns]
    y = pa.coords.get_array(dims[1])  # float (m)

    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]

    if isinstance(scale, tuple):
        scale = tuple(sorted(scale))
        vmin, vmax = scale[0], scale[1]
    else:
        percent = scale * 100 if scale <= 1.0 else scale
        vmax = abs(float(np.percentile(np.abs(data), percent)))
        vmin = -vmax

    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    return data, extent, norm


def tr2array(tr: Trace, ylim: tuple | float | int) -> tuple[np.ndarray, np.ndarray, tuple | Any]:
    x = tr.data
    y = mdates.date2num(
        np.array(
            [tr.stats.starttime.datetime + np.timedelta64(int(i * tr.stats.delta * 1e6), 'us') for i in range(tr.stats.npts)],
            dtype="datetime64[us]",
        ).astype('O')
    )

    if isinstance(ylim, tuple):
        lim = tuple(sorted(ylim))
    elif isinstance(ylim, (int, float)):
        ylim = abs(ylim)
        lim = (-ylim, ylim)
    else:
        lim = lim

    return x, y, lim
