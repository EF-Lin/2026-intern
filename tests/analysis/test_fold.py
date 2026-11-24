import numpy as np
import pytest
from dascore import Patch

from src.analysis import fold_legs, leg_shift, mirror_depth

DT = 0.01  # s
DX = 4.0  # m
V = 2000.0  # m/s
N = 171  # channels per leg, bottom channel shared


def _ricker(t: np.ndarray, f0: float = 10.0) -> np.ndarray:
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _patch(data: np.ndarray, depth: np.ndarray) -> Patch:
    time = np.datetime64("2026-07-08T15:47:32", "ns") + np.arange(data.shape[1]) * np.timedelta64(int(DT * 1e9), "ns")
    return Patch(data=data, coords={"depth": depth, "time": time}, dims=("depth", "time"))


def _signal(z: np.ndarray, v: float = V) -> np.ndarray:
    """Down-going pulse seen at true depths z, shape (depth, time)."""
    t = np.arange(300) * DT
    return _ricker(t[None, :] - 0.3 - z[:, None] / v)


def _legs(noise: float = 0.0, seed: int = 0, shift: int = 0, v: float = V):
    """Downward leg (depths 0..680 m) and upward leg in channel order (bottom first), like transfer() gives."""
    rng = np.random.default_rng(seed)
    z_down = np.arange(N) * DX
    z_up_true = z_down[::-1] + shift * DX  # upward leg channel j at depth 680 - j*dx, optionally misplaced
    sig = _signal(z_down, v)
    down = sig + noise * rng.standard_normal(sig.shape)
    up = _signal(z_up_true, v) + noise * rng.standard_normal(sig.shape)
    along = 680.0 + np.arange(N) * DX  # transfer(start=...) numbering along the fiber
    return _patch(down, z_down), _patch(up, along), sig


def test_mirror_depth():
    down, up, _ = _legs()
    z = mirror_depth(down, up)
    assert z[0] == pytest.approx(680.0) and z[-1] == pytest.approx(0.0)
    np.testing.assert_allclose(np.diff(z), -DX)
    assert mirror_depth(down, up, offset=2)[0] == pytest.approx(688.0)


def test_noise_free_reversal():
    down, up, sig = _legs()
    out = fold_legs(down, up)
    assert out.dims == down.dims and out.data.shape == down.data.shape
    np.testing.assert_allclose(out.data, sig, atol=1e-12)
    np.testing.assert_array_equal(out.coords.get_array("depth"), down.coords.get_array("depth"))


def test_stack_reduces_noise():
    down, up, sig = _legs(noise=0.3)
    out = fold_legs(down, up)
    err_single = np.std(down.data - sig)
    err_stack = np.std(out.data - sig)
    # independent noise: averaging two legs cuts the noise by about sqrt(2)
    assert err_stack / err_single == pytest.approx(1 / np.sqrt(2), rel=0.05)


def test_offset_and_shift_estimate():
    # upward leg sits 3 channels shallower than assumed -> offset -3; slow wave so 12 m is resolvable
    down, up, sig = _legs(noise=0.1, shift=-3, v=300.0)
    best, shifts, cc = leg_shift(down, up, max_shift=6)
    assert best == -3
    assert len(shifts) == len(cc) == 13
    out = fold_legs(down, up, offset=best)
    # the bottom 3 channels are not covered by the shifted upward leg, the rest is the average
    assert np.std(out.data[:-3] - sig[:-3]) < 0.8 * np.std(down.data[:-3] - sig[:-3])
    np.testing.assert_array_equal(out.data[-3:], down.data[-3:])


def test_different_spacing_interpolates():
    down, _, sig = _legs()
    z_up = np.arange(0, 681, 2.0)[::-1]  # upward leg with 2 m channels
    up = _patch(_signal(z_up), 680.0 + np.arange(len(z_up)) * 2.0)
    out = fold_legs(down, up)
    np.testing.assert_allclose(out.data, sig, atol=1e-12)


def test_common_time_window():
    down, up, _ = _legs()
    t = up.coords.get_array("time")
    out = fold_legs(down, up.select(time=(t[10], t[-11])))
    assert out.data.shape[1] == len(t) - 20


def test_bad_dims():
    down, up, _ = _legs()
    with pytest.raises(ValueError):
        fold_legs(down.transpose("time", "depth"), up)
