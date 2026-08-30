import re
from typing import Any

import numpy as np
from obspy import UTCDateTime


def nptime_range_convt(date: tuple[Any, Any]) -> tuple[np.datetime64, np.datetime64]:
    """
    ## convert tuple to np.datetime64

    date formate: (time, range(s)) | (time, time)

    time formate: 2026-08-09T12:00:00 | 2026/08/09 12:12:12 | 20260809 12:00:00

    range formate: int, float
    """
    regex = r"(\d{4})[-\/]?(\d{2})[-\/]?(\d{2})[T\s](\d{2}):(\d{2}):(\d{2})"
    replace = r"\1-\2-\3T\4:\5:\6"
    first = np.datetime64(re.sub(regex, replace, str(date[0])))

    if isinstance(date[1], (int, float)):
        ret_date = (first - np.timedelta64(date[1], 's'), first + np.timedelta64(date[1], 's'))
    else:
        ret_date = (first, np.datetime64(re.sub(regex, replace, str(date[1]))))

    return ret_date


def obtime_range_convt(date: tuple[Any, Any]) -> tuple[UTCDateTime, UTCDateTime]:
    """
    ## convert tuple to obspy.UTCDateTime

    date formate: (time, range(s)) | (time, time)

    time formate: 2026-08-09T12:00:00 | 2026/08/09 12:12:12 | 20260809 12:00:00

    range formate: int, float
    """
    if isinstance(date[1], (int, float)):
        ret_date = (UTCDateTime(date[0]) - date[1], UTCDateTime(date[0]) + date[1])
    else:
        ret_date = (UTCDateTime(date[0]), UTCDateTime(date[1]))
    return ret_date
