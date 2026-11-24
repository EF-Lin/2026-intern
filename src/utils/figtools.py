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
    # rows = the non-time dim (depth / distance), columns = time, so imshow draws time on x
    data = pa.data if pa.dims[1] == "time" else pa.data.T
    ydim = pa.dims[0] if pa.dims[1] == "time" else pa.dims[1]

    x = mdates.date2num(pa.coords.get_array("time").astype("datetime64[ms]").astype("O"))  # datetime64[ns]
    y = pa.coords.get_array(ydim)  # float (m)

    # imshow(origin="upper") puts row 0 at the top: top edge = y[0], bottom edge = y[-1]
    extent = [float(x[0]), float(x[-1]), float(y[-1]), float(y[0])]

    if isinstance(scale, tuple):
        scale = tuple(sorted(scale))
        vmin, vmax = scale[0], scale[1]
    else:
        percent = scale * 100 if scale <= 1.0 else scale
        vmax = abs(float(np.percentile(np.abs(data), percent)))
        vmin = -vmax

    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    return data, extent, norm


def tr2array(tr: Trace, ylim: tuple | float | int | None = None) -> tuple[np.ndarray, np.ndarray, tuple | Any]:
    y = tr.data
    t0 = np.datetime64(tr.stats.starttime.datetime, "us")
    offsets = (np.arange(tr.stats.npts) * tr.stats.delta * 1e6).astype(np.int64).astype("timedelta64[us]")
    x = mdates.date2num(t0 + offsets)

    if isinstance(ylim, tuple):
        lim = tuple(sorted(ylim))
    elif isinstance(ylim, (int, float)):
        ylim = abs(ylim)
        lim = (-ylim, ylim)
    else:
        lim = None

    return x, y, lim
