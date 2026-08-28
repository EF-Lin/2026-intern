import matplotlib.dates as mdates
import numpy as np
from dascore import Patch
from matplotlib.colors import TwoSlopeNorm


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
