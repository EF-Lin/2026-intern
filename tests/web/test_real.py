import numpy as np
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
