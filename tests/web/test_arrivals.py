import numpy as np
import pytest

from src.web.arrivals import event_arrivals, fit_plane_wave, neighbour_stack, pick_arrivals

FS = 50.0


def _ricker(t, f=6.0):
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _array(n_line=40, n_hole=20, seed=0):
    """L-shaped surface array (east + north lines, 20 m apart) plus a 400 m deep hole."""
    rng = np.random.default_rng(seed)
    east = np.column_stack([np.arange(n_line) * 20.0, np.zeros(n_line), np.zeros(n_line)])
    north = np.column_stack([np.zeros(n_line), 20 + np.arange(n_line) * 20.0, np.zeros(n_line)])
    hole = np.column_stack([np.full(n_hole, 300.0), np.full(n_hole, 300.0), 20 + np.arange(n_hole) * 20.0])
    xyz = np.vstack([east, north, hole]) + np.column_stack([rng.normal(0, 1, (2 * n_line + n_hole, 2)), np.zeros(2 * n_line + n_hole)])
    sections = ["E"] * n_line + ["N"] * n_line + ["H"] * n_hole
    return xyz, sections


def _plane_times(xyz, baz, v_app, q=0.0005, t0=10.0):
    """Arrival times of a plane wave coming FROM back-azimuth `baz` (deg) with apparent velocity v_app (m/s)."""
    b = np.radians(baz)
    s = -np.array([np.sin(b), np.cos(b)]) / v_app  # propagation direction = away from the source
    return t0 + xyz[:, :2] @ s - q * xyz[:, 2]


def _traces(times, n=int(30 * FS), snr=10.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    return np.column_stack([_ricker(t - tj) for tj in times]) + rng.normal(0, 1 / snr, (n, len(times)))


@pytest.mark.parametrize("baz", [0, 45, 135, 200, 290])
def test_fit_recovers_azimuth_from_exact_times(baz):
    xyz, _ = _array()
    t = _plane_times(xyz, baz, 4000)
    f = fit_plane_wave(xyz, t, n_boot=20)
    assert f["baz"] == pytest.approx(baz, abs=0.1) or abs(f["baz"] - baz) % 360 == pytest.approx(0, abs=0.1)
    assert f["v_app"] == pytest.approx(4000, rel=1e-3)
    assert f["q"] == pytest.approx(0.5, rel=1e-3)  # s/km
    assert f["residual_ms"] < 0.1


def test_fit_rejects_outliers():
    xyz, _ = _array()
    t = _plane_times(xyz, 60, 3000)
    bad = [3, 17, 50, 61]
    t[bad] += 0.4  # cycle skips / wrong phase
    t[10] = np.nan  # unpicked
    f = fit_plane_wave(xyz, t, n_boot=20)
    assert f["baz"] == pytest.approx(60, abs=0.5)
    assert f["n_used"] == len(t) - 1 - len(bad)
    assert not f["used"][bad[0]]  # indices of the finite picks; first outlier comes before the NaN


def test_fit_needs_five_picks():
    xyz, _ = _array()
    t = np.full(len(xyz), np.nan)
    t[:4] = 1.0
    assert fit_plane_wave(xyz, t) is None


def test_position_jitter_widens_uncertainty():
    xyz, sec = _array()
    t = _plane_times(xyz, 120, 4000)
    tight = fit_plane_wave(xyz, t, np.array(sec), pos_sigma=0.0, n_boot=50)
    loose = fit_plane_wave(xyz, t, np.array(sec), pos_sigma=100.0, n_boot=50)
    assert tight["baz_err"] < 0.5
    assert loose["baz_err_position"] > tight["baz_err_position"]


def test_picker_finds_relative_arrivals():
    xyz, _ = _array()
    times = _plane_times(xyz, 30, 4000)
    data = _traces(times)
    picks, cc, onset = pick_arrivals(data, FS, t_hint=5.0)
    assert onset is not None
    ok = np.isfinite(picks)
    assert ok.mean() > 0.9
    rel = (picks - times)[ok]
    assert np.std(rel - np.median(rel)) < 0.005  # 5 ms: well under one 20 ms sample


def test_picker_returns_nothing_on_noise():
    data = np.random.default_rng(3).normal(0, 1, (int(120 * FS), 20))
    picks, _, onset = pick_arrivals(data, FS, t_hint=60.0)  # 30 s of quiet record before the search window
    assert onset is None
    assert not np.isfinite(picks).any()


@pytest.mark.parametrize("baz", [30, 200, 290])
def test_event_arrivals_end_to_end(baz):
    xyz, sec = _array()
    stations = [dict(x=p[0], y=p[1], depth=p[2], section=s, has_data=True) for p, s in zip(xyz, sec)]
    times = _plane_times(xyz, baz, 4500)
    data = _traces(times, snr=4.0, seed=baz)
    a = event_arrivals(stations, data, FS, t_hint=8.0, pos_sigma=0.0)
    err = (a["baz"] - baz + 180) % 360 - 180
    assert abs(err) < 3
    assert a["v_app"] == pytest.approx(4500, rel=0.1)
    assert min(r for r in a["rel_ms"] if r is not None) == 0
    assert len(a["rel_ms"]) == len(stations)


def test_neighbour_stack_stays_inside_section():
    stations = [dict(section=s, has_data=h) for s, h in [("A", True), ("A", True), ("B", True), ("B", False)]]
    data = np.array([[1.0, 3.0, 10.0, 99.0]])
    out = neighbour_stack(data, stations, half=1)
    assert out[0].tolist() == [2.0, 2.0, 10.0, 0.0]
