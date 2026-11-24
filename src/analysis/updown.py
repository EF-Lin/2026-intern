"""Separate up-going and down-going waves of borehole DAS data by f-k filtering."""

from typing import Optional

import numpy as np
from dascore import Patch
from scipy.fft import fft2, fftfreq, ifft2, next_fast_len


def _cos_step(x: np.ndarray) -> np.ndarray:
    """0 for x <= 0, 1 for x >= 1, raised-cosine in between."""
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))


def _fk_masks(
    nt: int,
    nx: int,
    dt: float,
    dx: float,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
    *,
    v_taper: Optional[float] = None,
    taper: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Masks (up, down) on the numpy f-k grid of an array shaped (time, depth).

    With numpy's FFT sign convention a wave s(t - z / v) moving towards larger depth z (down-going)
    puts its energy where f * k < 0; an up-going wave puts it where f * k > 0. Written with the signed
    slowness p = -k / f, down-going is p > 0 and up-going is p < 0.

    - ``v_taper``: the up / down boundary at k = 0 (p = 0) is a raised-cosine ramp over
      |p| < 1 / v_taper instead of a hard cut; ``up + down == 1`` still holds. None = hard cut.
    - ``v_min`` / ``v_max``: wedges that reject |v| < v_min (around f = 0) and |v| > v_max
      (around k = 0); their edges are ramped over a relative width ``taper`` in slowness.
    Bins with f = 0 are split half / half.
    """
    f = fftfreq(nt, d=dt)[:, None]
    k = fftfreq(nx, d=dx)[None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        p = -k / f  # signed slowness (s/m), inf / nan on f = 0
    on_f0 = np.broadcast_to(f == 0, p.shape)

    if v_taper:
        down = _cos_step((p * v_taper + 1.0) / 2.0)
    else:
        down = np.where(p > 0, 1.0, np.where(p < 0, 0.0, 0.5))
    down = np.where(on_f0, 0.5, down)
    up = 1.0 - down

    abs_p = np.where(on_f0, np.inf, np.abs(p))
    keep = np.ones_like(up)
    if v_min is not None:
        # reject slow energy |v| < v_min, i.e. |p| > 1 / v_min; pass fully below (1 - taper) / v_min
        keep = keep * (_cos_step((1.0 / v_min - abs_p) * v_min / taper) if taper > 0 else (abs_p <= 1.0 / v_min))
    if v_max is not None:
        # reject fast energy |v| > v_max (around k = 0, e.g. common-mode noise); pass fully above (1 + taper) / v_max
        keep = keep * (_cos_step((abs_p - 1.0 / v_max) * v_max / taper) if taper > 0 else (abs_p >= 1.0 / v_max))

    return up * keep, down * keep


def separate_updown(
    pa: Patch,
    *,
    v_min: Optional[float] = None,
    v_max: Optional[float] = None,
    v_taper: Optional[float] = 10000.0,
    taper: float = 0.2,
) -> tuple[Patch, Patch]:
    """
    Split a borehole patch into up-going and down-going wavefields.

    Parameters
    ----------
    pa : Patch
        Dims ``("depth", "time")`` or ``("time", "depth")``. Depth may increase or decrease along
        the array (e.g. the upward leg of a U-shaped borehole fiber); the direction is taken from the
        depth coordinate, so "down-going" always means moving towards larger depth.
    v_min : float, optional
        Also remove energy with apparent velocity below ``v_min`` (m/s), e.g. slow tube waves / noise.
        In the f-k plane this is a wedge around the horizontal (f = 0) axis.
    v_max : float, optional
        Also remove energy with apparent velocity above ``v_max`` (m/s), i.e. events that hit all
        channels at almost the same time (common-mode noise, near-horizontal incidence on the fiber).
        In the f-k plane this is a wedge around the vertical (k = 0) axis.
    v_taper : float, optional
        Smooth the up / down boundary at k = 0: energy faster than ``v_taper`` (m/s) is shared
        between both outputs with a raised-cosine weight instead of being cut hard, which reduces
        ringing. None = hard cut.
    taper : float
        Relative width (in slowness) of the raised-cosine edges of the ``v_min`` / ``v_max`` wedges.
        0 = hard edges.

    Returns
    -------
    (up, down) : tuple[Patch, Patch]
        Same dims and coordinates as ``pa``. Without ``v_min`` / ``v_max``, ``up + down`` equals ``pa`` up to
        floating point error.

    The array is zero padded to twice its size in both dims before the FFT so that energy wrapping
    around the edges of the window does not leak into the other direction.
    """
    time_first = pa.dims == ("time", "depth")
    if not time_first and pa.dims != ("depth", "time"):
        raise ValueError(f"patch dims must be (depth, time) or (time, depth), got {pa.dims}")

    data = np.asarray(pa.data, dtype=float)
    if not time_first:
        data = data.T  # -> (time, depth)

    dt = float(pa.coords.step("time") / np.timedelta64(1, "s"))
    dx = float(pa.coords.step("depth"))  # negative when depth decreases along the array

    nt, nx = data.shape
    shape = (next_fast_len(2 * nt), next_fast_len(2 * nx))
    spec = fft2(data, s=shape)

    up_mask, down_mask = _fk_masks(*shape, dt, dx, v_min, v_max, v_taper=v_taper, taper=taper)
    up = ifft2(spec * up_mask).real[:nt, :nx]
    down = ifft2(spec * down_mask).real[:nt, :nx]

    if not time_first:
        up, down = up.T, down.T

    return pa.new(data=up), pa.new(data=down)
