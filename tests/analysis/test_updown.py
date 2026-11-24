import numpy as np
import pytest
from dascore import Patch

from src.analysis import separate_updown

DT = 0.01  # s
DX = 4.0  # m
V = 2000.0  # m/s


def _ricker(t: np.ndarray, f0: float = 10.0) -> np.ndarray:
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _patch(data: np.ndarray, depth: np.ndarray) -> Patch:
    time = np.datetime64("2026-07-08T15:47:32", "ns") + np.arange(data.shape[1]) * np.timedelta64(int(DT * 1e9), "ns")
    return Patch(data=data, coords={"depth": depth, "time": time}, dims=("depth", "time"))


def _waves(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Down-going pulse starting at the top, up-going pulse starting at the bottom (depth, time)."""
    t = np.arange(400) * DT
    z = depth[:, None] - depth.min()
    bottom = depth.max() - depth.min()
    down = _ricker(t[None, :] - 0.5 - z / V)
    up = _ricker(t[None, :] - 0.5 - (bottom - z) / V)
    return up, down


def _ratio(est: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(est - ref) / np.linalg.norm(ref))


@pytest.mark.parametrize("depth", [np.arange(100) * DX, (np.arange(100) * DX)[::-1]], ids=["downward", "upward_leg"])
def test_separate(depth: np.ndarray):
    up, down = _waves(depth)
    est_up, est_down = separate_updown(_patch(up + down, depth))
    assert _ratio(est_up.data, up) < 0.2
    assert _ratio(est_down.data, down) < 0.2


def test_sum_is_identity():
    depth = np.arange(100) * DX
    up, down = _waves(depth)
    pa = _patch(up + down, depth)
    est_up, est_down = separate_updown(pa)
    np.testing.assert_allclose(est_up.data + est_down.data, pa.data, atol=1e-10)
    assert est_up.dims == pa.dims and est_up.data.shape == pa.data.shape


def test_time_first_dims():
    depth = np.arange(100) * DX
    up, down = _waves(depth)
    pa = _patch(up + down, depth).transpose("time", "depth")
    est_up, _ = separate_updown(pa)
    assert est_up.dims == ("time", "depth")
    assert _ratio(est_up.data, up.T) < 0.2


def test_v_min_removes_slow_wave():
    depth = np.arange(100) * DX
    t = np.arange(400) * DT
    slow = _ricker(t[None, :] - 0.5 - (depth[:, None] - depth.min()) / 300.0)  # 300 m/s down-going
    _, est_down = separate_updown(_patch(slow, depth), v_min=1000.0)
    assert np.linalg.norm(est_down.data) < 0.2 * np.linalg.norm(slow)


def test_v_max_removes_common_mode():
    depth = np.arange(100) * DX
    t = np.arange(400) * DT
    common = np.repeat(_ricker(t - 1.0)[None, :], depth.size, axis=0)  # same on every channel: infinite velocity
    up, down = _waves(depth)
    pa = _patch(up + down + common, depth)
    # the finite aperture smears common-mode energy over all k, so the wedge removes most but not all of it
    for est, ref in zip(separate_updown(pa, v_max=8000.0), (up, down)):
        without = separate_updown(pa)[0 if ref is up else 1]
        assert _ratio(est.data, ref) < 0.6 * _ratio(without.data, ref)
