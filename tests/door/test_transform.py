import numpy as np
import pytest
from obspy import Stream, Trace, UTCDateTime

from src.door.transform import transfer

T0 = UTCDateTime(2026, 7, 8, 0, 0, 0)


def make_stream(delta: float = 0.01, stations=("00010", "00011", "00013"), lengths=(100, 105, 110)) -> Stream:
    traces = []
    for sta, n in zip(stations, lengths):
        tr = Trace(data=np.arange(n, dtype=np.float64))
        tr.stats.station = sta
        tr.stats.delta = delta
        tr.stats.starttime = T0
        traces.append(tr)
    return Stream(traces)


def test_transfer_start() -> None:
    pa, stations = transfer(make_stream(), start=20)
    assert pa.dims == ("depth", "time")
    assert pa.data.shape == (3, 100)
    np.testing.assert_allclose(pa.get_array("depth"), [20, 24, 32])
    assert stations == ("00010", "00013")


def test_transfer_spacing_positional_and_keyword() -> None:
    pa1, _ = transfer(make_stream(), 2)
    pa2, _ = transfer(make_stream(), spacing=2, start=1.5)
    np.testing.assert_allclose(pa1.get_array("depth"), [0, 2, 6])
    np.testing.assert_allclose(pa2.get_array("depth"), [1.5, 3.5, 7.5])


def test_transfer_r_alias_deprecated() -> None:
    with pytest.warns(DeprecationWarning):
        pa, _ = transfer(make_stream(), r=2, start=10)
    np.testing.assert_allclose(pa.get_array("depth"), [10, 12, 16])


@pytest.mark.parametrize("delta", [0.01, 0.001, 1 / 3000])
def test_transfer_time_axis(delta: float) -> None:
    n = 5000
    pa, _ = transfer(make_stream(delta=delta, lengths=(n, n, n)))
    time = pa.get_array("time")
    assert time[0] == np.datetime64("2026-07-08T00:00:00", "ns")
    expected_step = np.timedelta64(int(round(delta * 1e9)), "ns")
    assert np.all(np.diff(time) == expected_step)
    # step is rounded to whole ns (keeps the axis evenly sampled), so drift is <= 0.5 ns per sample;
    # rates with an integer-ns period (100 Hz, 1000 Hz) are exact
    true_last_ns = (n - 1) * delta * 1e9
    got_last_ns = (time[-1] - time[0]).astype("timedelta64[ns]").astype(np.int64)
    assert abs(got_last_ns - true_last_ns) <= 0.5 * (n - 1) + 1e-6
    if delta in (0.01, 0.001):
        assert got_last_ns == round(true_last_ns)
