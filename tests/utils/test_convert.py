import numpy as np
import pytest
from obspy import UTCDateTime

from src.utils.convert import nptime_range_convt, obtime_range_convt


@pytest.mark.parametrize(
    "date, expected",
    [
        (("2026-08-09T12:00:00", "2026/08/09 12:10:00"), (np.datetime64("2026-08-09T12:00:00"), np.datetime64("2026-08-09T12:10:00"))),
        (("2026/08/09 12:00:00", 10), (np.datetime64("2026-08-09T11:59:50"), np.datetime64("2026-08-09T12:00:10"))),
        (("2026/08/09T12:00:00", "20260809 14:00:00"), (np.datetime64("2026-08-09T12:00:00"), np.datetime64("2026-08-09T14:00:00"))),
    ],
)
def test_nptime_range_convt(date: str, expected: str) -> None:
    assert nptime_range_convt(date) == expected


@pytest.mark.parametrize(
    "date, expected",
    [
        (("2026-08-09T12:00:00", "2026/08/09 12:10:00"), (UTCDateTime("2026-08-09T12:00:00"), UTCDateTime("2026-08-09T12:10:00"))),
        (("2026/08/09 12:00:00", 10), (UTCDateTime("2026-08-09T11:59:50"), UTCDateTime("2026-08-09T12:00:10"))),
        (("2026/08/09T12:00:00", "20260809 14:00:00"), (UTCDateTime("2026-08-09T12:00:00"), UTCDateTime("2026-08-09T14:00:00"))),
    ],
)
def test_obtime_range_convt(date: str, expected: str) -> None:
    assert obtime_range_convt(date) == expected
