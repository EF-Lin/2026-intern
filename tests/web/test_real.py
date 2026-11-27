import numpy as np
import pytest
import obspy

from src.web.real import FS, process_channel

DAY = "2026-07-08T00:00:00"


def _write(path, pieces):
    """pieces: [(start s after DAY, data)] → one MiniSEED file (gaps between pieces)."""
    st = obspy.Stream()
    for t0, x in pieces:
        tr = obspy.Trace(np.asarray(x, dtype=np.int32))
        tr.stats.sampling_rate = FS
        tr.stats.starttime = obspy.UTCDateTime(DAY) + t0
        st += tr
    st.write(str(path), format="MSEED", encoding="STEIM2")
    return str(path)


def test_rms_gap_and_event_window(tmp_path):
    rng = np.random.default_rng(0)
    n = 86_400 * FS
    x = np.round(rng.normal(0, 100, n)).astype(np.int32)
    t = np.arange(n) / FS
    burst = (t >= 50_000) & (t < 50_002)
    x[burst] += np.round(5000 * np.sin(2 * np.pi * 5 * t[burst])).astype(np.int32)
    gap = (40_000 * FS, 40_010 * FS)
    path = _write(tmp_path / "ch", [(0, x[: gap[0]]), (gap[1] / FS, x[gap[1] :])])

    rms, waves = process_channel(path, DAY, (1, 20), [(49_990, 20)])
    assert rms.shape == (86_400,)
    assert np.isnan(rms[40_000:40_010]).all() and np.isfinite(rms[39_990:40_000]).all()
    assert rms[50_000] > 10 * np.nanmedian(rms)
    w = waves[0]
    assert w.shape == (20 * FS,)
    assert np.abs(w[10 * FS : 12 * FS]).max() > 3000  # 5 Hz burst passes the 1–20 Hz band
    assert np.abs(w[: 9 * FS]).max() < 1000


def test_missing_file(tmp_path):
    assert process_channel(str(tmp_path / "none"), DAY, (1, 20), []) == (None, None)


def _day_file(tmp_path, name="ch", seed=0):
    rng = np.random.default_rng(seed)
    n = 86_400 * FS
    x = np.round(rng.normal(0, 100, n)).astype(np.int32)
    t = np.arange(n) / FS
    burst = (t >= 50_000) & (t < 50_002)
    x[burst] += np.round(5000 * np.sin(2 * np.pi * 5 * t[burst])).astype(np.int32)
    return _write(tmp_path / name, [(0, x)])


def test_window_read_matches_full_day(tmp_path):
    from src.web.real import process_window

    path = _day_file(tmp_path)
    _, full = process_channel(path, DAY, (1, 20), [(49_990, 20)])
    win = process_window(path, DAY, (1, 20), [(49_990, 20)])
    # different detrend / taper spans, same 1-20 Hz band: equal to well below the noise level (100 counts)
    assert np.abs(win[0] - full[0]).max() < 5


def test_cache_invalidation(tmp_path):
    from src.web.cache import DayCache

    st = [{"channels": [1, 2]}]
    kw = dict(day_start=DAY, frequency=(1, 20), stations=st, channels=[1, 2], half=15)
    c = DayCache(tmp_path, **kw)
    c.save_rms(np.ones((10, 1)))
    c.save_event("2026-07-08T01:00:00", np.ones((5, 1)))
    c = DayCache(tmp_path, **kw)  # same settings: kept
    assert c.load_rms() is not None and c.load_event("2026-07-08T01:00:00") is not None
    c = DayCache(tmp_path, **(kw | {"half": 30}))  # new half window: events dropped, RMS kept
    assert c.load_rms() is not None and c.load_event("2026-07-08T01:00:00") is None
    c = DayCache(tmp_path, **(kw | {"half": 30, "frequency": (2, 20)}))  # new filter: everything dropped
    assert c.load_rms() is None


def test_build_day_uses_cache(tmp_path, monkeypatch):
    import src.web.real as real
    from src.web.cache import DayCache

    paths = {1: _day_file(tmp_path, "a", 1), 2: _day_file(tmp_path, "b", 2)}
    st = [{"channels": [1, 2], "section": "S"}]
    ev = ["2026-07-08T13:53:21"]  # 50 001 s after midnight: inside the burst
    make = lambda half: DayCache(tmp_path / "cache", day_start=DAY, frequency=(1, 20), stations=st, channels=paths, half=half)  # noqa: E731
    log1, ev1 = real.build_day(paths, st, DAY, (1, 20), ev, 15, workers=1, cache=make(15))
    assert np.abs(ev1[0]["pick"]["data"]).max() > 3000

    # second run: nothing is read from MiniSEED
    monkeypatch.setattr(real, "process_channel", lambda *a, **k: pytest.fail("MiniSEED read"))
    monkeypatch.setattr(real, "process_window", lambda *a, **k: pytest.fail("MiniSEED read"))
    log2, ev2 = real.build_day(paths, st, DAY, (1, 20), ev, 15, workers=1, cache=make(15))
    np.testing.assert_allclose(log2, log1, rtol=2e-3, equal_nan=True)  # float16
    np.testing.assert_allclose(ev2[0]["pick"]["data"], ev1[0]["pick"]["data"], rtol=1e-6)
