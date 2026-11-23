import csv
import os
from typing import Any, Optional

import dascore as dc
import obspy as op
from tqdm import tqdm

from src.utils import nptime_range_convt


def load_mini(files: list | str, time: Optional[tuple[Any, Any]] = None, *, pad: float = 0.0) -> op.Stream:
    """
    Read MiniSEED file(s) into one obspy Stream.

    Parameters
    ----------
    files : list | str
        One path or a list of paths.
    time : (time, range(s)) | (time, time), optional
        Only read this time window (same format as ``Water.cut``). MiniSEED records outside the
        window are skipped instead of decoded, which is much faster and uses far less memory than
        reading the whole day and cutting afterwards. ``None`` reads the whole file.
    pad : float
        Extra seconds read on both sides of ``time``. Filtering / tapering a short window distorts
        its edges, so read with padding, ``Water(...).process()``, then ``.cut()`` back to the window.

    When the same stations are cut at many different times (e.g. ``compareAB``), reading the whole
    file once and cutting in memory is faster than re-reading each window.
    """
    if time is None:
        starttime = endtime = None
    else:
        t0, t1 = nptime_range_convt(time)
        starttime = op.UTCDateTime(str(t0)) - pad
        endtime = op.UTCDateTime(str(t1)) + pad

    if isinstance(files, str):
        files = [files]

    data = op.Stream()
    empty = []
    for f in tqdm(files, desc="Reading Data", disable=len(files) == 1):
        st = op.read(f, starttime=starttime, endtime=endtime)
        if time is not None and not any(tr.stats.npts for tr in st):
            empty.append(f)
        data += st

    if empty:
        raise ValueError(f"{len(empty)} file(s) have no data in {starttime} - {endtime}: {empty}")
    return data


def load_csv(path: str) -> list[dict]:
    path = os.path.normpath(path)
    with open(path, 'r', encoding="utf-8-sig") as f:
        file = list(csv.DictReader(f))
    return file


def load_h5(path: str) -> dc.Patch:
    path = os.path.normpath(path)
    res = dc.read(path, "dasdae")
    return res[0] if hasattr(res, "__getitem__") and not isinstance(res, dc.Patch) else res
