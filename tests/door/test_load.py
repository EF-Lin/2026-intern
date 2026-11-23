import numpy as np
import pytest
from obspy import Trace, UTCDateTime

from src.door.load import load_mini

T0 = UTCDateTime(2026, 7, 8, 0, 0, 0)
N = 100 * 600  # 10 minutes at 100 Hz


@pytest.fixture
def mseed_files(tmp_path) -> list[str]:
    files = []
    rng = np.random.default_rng(0)
    for sta in ("00010", "00011"):
        tr = Trace(data=rng.integers(-16000, 16000, N, dtype=np.int32))
        tr.stats.station = sta
        tr.stats.network = "TW"
        tr.stats.channel = "S"
        tr.stats.delta = 0.01
        tr.stats.starttime = T0
        path = tmp_path / f"TW.{sta}..S.D.2026.189"
        tr.write(str(path), format="MSEED", encoding="STEIM2", reclen=512)
        files.append(str(path))
    return files


def test_whole_file(mseed_files) -> None:
    st = load_mini(mseed_files)
    assert len(st) == 2
    assert all(tr.stats.npts == N for tr in st)


def test_single_path(mseed_files) -> None:
    assert load_mini(mseed_files[0])[0].stats.npts == N


@pytest.mark.parametrize(
    "time",
    [
        ("2026-07-08T00:05:00", 10),
        ("2026-07-08T00:04:50", "2026-07-08T00:05:10"),
        ("2026/07/08 00:05:00", 10),
    ],
)
def test_window_matches_full_read(mseed_files, time) -> None:
    full = load_mini(mseed_files)
    st = load_mini(mseed_files, time)
    for tr, ref in zip(st, full):
        assert tr.stats.starttime == T0 + 290
        assert tr.stats.npts == 2001
        i0 = int(round((tr.stats.starttime - ref.stats.starttime) / ref.stats.delta))
        np.testing.assert_array_equal(tr.data, ref.data[i0 : i0 + tr.stats.npts])


def test_pad(mseed_files) -> None:
    st = load_mini(mseed_files, ("2026-07-08T00:05:00", 10), pad=30)
    assert st[0].stats.starttime == T0 + 260
    assert st[0].stats.endtime == T0 + 340


def test_window_outside_data(mseed_files) -> None:
    with pytest.raises(ValueError, match="no data"):
        load_mini(mseed_files, ("2026-07-09T05:00:00", 10))
