"""Stack the two legs of a U-shaped borehole fiber onto one depth axis to raise the signal-to-noise ratio."""

import numpy as np
from dascore import Patch


def _check(pa: Patch, name: str) -> None:
    if pa.dims != ("depth", "time"):
        raise ValueError(f"{name} dims must be ('depth', 'time'), got {pa.dims}")


def _common_time(down: Patch, up: Patch) -> tuple[Patch, Patch]:
    """Trim both legs to the time samples they share (they come from the same interrogator)."""
    t_down, t_up = down.coords.get_array("time"), up.coords.get_array("time")
    if len(t_down) == len(t_up) and np.array_equal(t_down, t_up):
        return down, up
    common = np.intersect1d(t_down, t_up)
    if len(common) < 2:
        raise ValueError("the two legs share no time samples")
    rng = (common[0], common[-1])
    down, up = down.select(time=rng), up.select(time=rng)
    if down.data.shape[1] != up.data.shape[1]:
        raise ValueError("the two legs have different sampling in their common time window")
    return down, up


def mirror_depth(down: Patch, up: Patch, offset: float = 0.0) -> np.ndarray:
    """
    Depth (m) of every channel of ``up`` once the fiber is folded back onto ``down``.

    ``up`` is the upward leg in channel order, its first channel being the turnaround at the bottom of
    the hole, i.e. the same point as the last channel of ``down``. Only the spacing of its depth
    coordinate is used: channel ``j`` of ``up`` sits at ``bottom - j * dx + offset * dx`` where
    ``bottom`` is the last depth of ``down`` and ``dx`` the channel spacing of ``up``. A positive
    ``offset`` (channels, may be fractional) moves the upward leg deeper.
    """
    along = np.asarray(up.coords.get_array("depth"), dtype=float)
    along = np.abs(along - along[0])
    dx = float(np.abs(up.coords.step("depth")))
    bottom = float(down.coords.get_array("depth")[-1])
    return bottom - along + offset * dx


def _up_on_down(down: Patch, up: Patch, offset: float) -> np.ndarray:
    """Upward leg interpolated onto the depth axis of ``down``; NaN where it does not reach."""
    z_down = np.asarray(down.coords.get_array("depth"), dtype=float)
    z_up = mirror_depth(down, up, offset)
    order = np.argsort(z_up)  # upward leg reversed: wellhead first, bottom last
    z_up, data = z_up[order], np.asarray(up.data, dtype=float)[order]

    if len(z_up) < 2:
        raise ValueError("the upward leg needs at least two channels")

    out = np.full((len(z_down), data.shape[1]), np.nan)
    tol = 1e-6 * float(np.diff(z_up).min())
    inside = (z_down >= z_up[0] - tol) & (z_down <= z_up[-1] + tol)
    z = np.clip(z_down[inside], z_up[0], z_up[-1])
    idx = np.clip(np.searchsorted(z_up, z, side="right") - 1, 0, len(z_up) - 2)
    w = ((z - z_up[idx]) / (z_up[idx + 1] - z_up[idx]))[:, None]
    out[inside] = (1 - w) * data[idx] + w * data[idx + 1]
    return out


def fold_legs(down: Patch, up: Patch, *, offset: float = 0.0) -> Patch:
    """
    Average the downward and upward legs of a U-shaped borehole fiber on the downward leg's depth axis.

    Parameters
    ----------
    down : Patch
        Downward leg, dims ``("depth", "time")``, wellhead first and the bottom (turnaround) channel last.
    up : Patch
        Upward leg, dims ``("depth", "time")``, in channel order: the bottom channel (shared with the last
        channel of ``down``) first and the wellhead last. Its depth coordinate only needs the right spacing,
        e.g. straight from :func:`src.door.transfer`; it is reversed and mapped by :func:`mirror_depth`.
    offset : float
        Channel shift of the upward leg (positive = deeper), see :func:`leg_shift` to estimate it.

    Returns
    -------
    Patch
        Same coordinates as ``down`` (trimmed to the time samples shared with ``up``). Where both legs
        cover a depth the data are ``(down + up) / 2`` (``up`` linearly interpolated in depth if the
        channels do not line up); where only ``down`` exists it is kept unchanged.
    """
    _check(down, "down")
    _check(up, "up")
    down, up = _common_time(down, up)
    mapped = _up_on_down(down, up, offset)
    data = np.asarray(down.data, dtype=float)
    stacked = np.where(np.isnan(mapped), data, (data + np.nan_to_num(mapped)) / 2)
    return down.new(data=stacked)


def leg_shift(down: Patch, up: Patch, *, max_shift: int = 10, step: float = 1.0) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Find the channel offset that best lines up the upward leg with the downward leg.

    For every candidate offset in ``[-max_shift, max_shift]`` (every ``step`` channels) the upward leg is
    mapped onto the downward leg's depths and the zero-lag normalised cross-correlation of each depth's
    pair of traces is averaged over all depths both legs cover. Restrict both patches to a time window
    around the main arrival first so the correlation is dominated by it.

    Returns
    -------
    (best, shifts, cc)
        ``best`` is the offset with the highest mean correlation, to pass as ``fold_legs(offset=...)``;
        ``shifts`` and ``cc`` are all candidates and their mean correlations.
    """
    _check(down, "down")
    _check(up, "up")
    down, up = _common_time(down, up)
    a = np.asarray(down.data, dtype=float)
    a = a - a.mean(axis=1, keepdims=True)
    shifts = np.arange(-max_shift, max_shift + step / 2, step)
    cc = np.full(len(shifts), np.nan)
    for i, s in enumerate(shifts):
        b = _up_on_down(down, up, float(s))
        ok = ~np.isnan(b[:, 0])
        if ok.sum() < 2:
            continue
        x, y = a[ok], b[ok] - b[ok].mean(axis=1, keepdims=True)
        norm = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1)
        good = norm > 0
        cc[i] = float(np.mean(np.sum(x[good] * y[good], axis=1) / norm[good]))
    best = float(shifts[np.nanargmax(cc)])
    return best, shifts, cc
