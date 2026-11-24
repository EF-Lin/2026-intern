"""Separate up-going and down-going waves of borehole DAS data by f-k filtering."""

from typing import Optional

import numpy as np
from dascore import Patch
from scipy.fft import fft2, fftfreq, ifft2, next_fast_len


def _fk_masks(nt: int, nx: int, dt: float, dx: float, v_min: Optional[float], v_max: Optional[float] = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Masks (up, down) on the numpy f-k grid of an array shaped (time, depth).

    With numpy's FFT sign convention a wave s(t - z / v) moving towards larger depth z (down-going)
    puts its energy where f * k < 0; an up-going wave puts it where f * k > 0. Bins on the f = 0 or
    k = 0 axes are split half / half, so ``up + down == 1`` when ``v_min`` and ``v_max`` are None.
    """
    f = fftfreq(nt, d=dt)[:, None]
    k = fftfreq(nx, d=dx)[None, :]
    fk = f * k

    up = np.where(fk > 0, 1.0, np.where(fk < 0, 0.0, 0.5))
    down = 1.0 - up

    if v_min is not None:
        # reject slow energy: apparent velocity |f / k| below v_min
        with np.errstate(divide="ignore", invalid="ignore"):
            slow = np.abs(f) < v_min * np.abs(k)
        up = np.where(slow, 0.0, up)
        down = np.where(slow, 0.0, down)

    if v_max is not None:
        # reject fast energy around the k = 0 (vertical) axis: |f / k| above v_max, e.g. common-mode noise
        fast = np.abs(f) > v_max * np.abs(k)
        up = np.where(fast, 0.0, up)
        down = np.where(fast, 0.0, down)

    return up, down


def separate_updown(pa: Patch, *, v_min: Optional[float] = None, v_max: Optional[float] = None) -> tuple[Patch, Patch]:
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

    up_mask, down_mask = _fk_masks(*shape, dt, dx, v_min, v_max)
    up = ifft2(spec * up_mask).real[:nt, :nx]
    down = ifft2(spec * down_mask).real[:nt, :nx]

    if not time_first:
        up, down = up.T, down.T

    return pa.new(data=up), pa.new(data=down)
