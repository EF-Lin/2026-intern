"""
Real MiDAS day → web-viewer arrays, one channel file at a time (a day file is ~35 MB as float32, the whole fiber
would not fit in memory at once).

Per channel: demean + linear detrend, cosine taper on the day's ends, 4-pole zero-phase Butterworth bandpass (the same
filter as `Water.process`), then
    - 1 s RMS of the filtered trace for the whole day (seconds touching a data gap → NaN),
    - the filtered 100 Hz trace in each event window (origin ± half seconds).
Per station (`every` channels): log10 of the max RMS of its channels; event waveform = its middle channel with data.
Results are cached per day (src/web/cache.py), so later runs only compute what changed.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import obspy
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

from .cache import DayCache
from .store import SECONDS_PER_DAY

FS = 100  # MiDAS sampling rate (Hz)


def _seconds(t: str, day_start: np.datetime64) -> float:
    return float((np.datetime64(t) - day_start) / np.timedelta64(1, "ms")) / 1e3


def process_channel(path: str, day_start: str, frequency: tuple[float, float], windows: list[tuple[float, float]], taper: float = 10.0):
    """
    Returns (rms, waves): rms (86400,) float32 with NaN where data are missing; waves = one float32 array per window
    (start s after day_start, length s), NaN where missing. Missing file → (None, None).
    """
    if not Path(path).is_file():
        return None, None
    d0 = np.datetime64(day_start)
    n = SECONDS_PER_DAY * FS
    x = np.zeros(n)
    ok = np.zeros(n, bool)
    for tr in obspy.read(path):
        i0 = int(round(_seconds(str(tr.stats.starttime).rstrip("Z"), d0) * FS))
        a, b = max(0, i0), min(n, i0 + tr.stats.npts)
        if b > a:
            x[a:b] = tr.data[a - i0 : b - i0]
            ok[a:b] = True
    if not ok.any():
        return None, None
    x[ok] -= x[ok].mean()
    idx = np.flatnonzero(ok)
    x[ok] -= np.polyval(np.polyfit(idx, x[ok], 1), idx)  # linear detrend over the samples with data
    x[~ok] = 0.0
    m = int(taper * FS)
    w = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
    x[idx[0] : idx[0] + m] *= w[: max(0, min(m, n - idx[0]))]
    x[max(0, idx[-1] + 1 - m) : idx[-1] + 1] *= w[::-1][-min(m, idx[-1] + 1) :]
    y = sosfiltfilt(butter(4, frequency, "bandpass", fs=FS, output="sos"), x)

    rms = np.sqrt((y.reshape(SECONDS_PER_DAY, FS) ** 2).mean(axis=1)).astype(np.float32)
    rms[~ok.reshape(SECONDS_PER_DAY, FS).all(axis=1)] = np.nan
    waves = []
    for s0, length in windows:
        i0, k = int(round(s0 * FS)), int(round(length * FS))
        out = np.full(k, np.nan, np.float32)
        a, b = max(0, i0), min(n, i0 + k)
        if b > a:
            seg = y[a:b].astype(np.float32)
            seg[~ok[a:b]] = np.nan
            out[a - i0 : b - i0] = seg
        waves.append(out)
    return rms, waves


def process_window(path: str, day_start: str, frequency: tuple[float, float], windows: list[tuple[float, float]], pad: float = 30.0, taper: float = 5.0):
    """
    Like the event part of `process_channel`, but reads only each window ± pad seconds instead of the whole day (used
    when the day's RMS is already cached). The pad keeps the filter's edge effects out of the window. Returns one float32
    array per window (NaN where missing), or None for a missing file.
    """
    if not Path(path).is_file():
        return None
    d0 = obspy.UTCDateTime(day_start)
    sos = butter(4, frequency, "bandpass", fs=FS, output="sos")
    out = []
    for s0, length in windows:
        n, k0 = int(round((length + 2 * pad) * FS)), int(round(pad * FS))
        x, ok = np.zeros(n), np.zeros(n, bool)
        a0 = d0 + s0 - pad
        for tr in obspy.read(path, starttime=a0, endtime=a0 + n / FS):
            i0 = int(round((tr.stats.starttime - a0) * FS))
            a, b = max(0, i0), min(n, i0 + tr.stats.npts)
            if b > a:
                x[a:b] = tr.data[a - i0 : b - i0]
                ok[a:b] = True
        w = np.full(int(round(length * FS)), np.nan, np.float32)
        if ok.sum() > 1:
            idx = np.flatnonzero(ok)
            x[ok] -= np.polyval(np.polyfit(idx, x[ok], 1), idx)
            x[~ok] = 0.0
            m = int(taper * FS)
            win = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
            x[:m] *= win
            x[-m:] *= win[::-1]
            y = sosfiltfilt(sos, x)[k0 : k0 + len(w)].astype(np.float32)
            y[~ok[k0 : k0 + len(w)]] = np.nan
            w = y
        out.append(w)
    return out


def _day_job(args):
    ch, path, day_start, frequency, windows = args
    return ch, *process_channel(path, day_start, frequency, windows)


def _window_job(args):
    ch, path, day_start, frequency, windows = args
    return ch, None, process_window(path, day_start, frequency, windows)


class _Stations:
    """Channel results → station arrays: RMS = max over the station's channels; waveform = its middle channel with data."""

    def __init__(self, stations: list[dict], n_windows: int, n_samp: int, rms: bool):
        n_st = len(stations)
        self.owner = {}  # channel → [(station index, distance of the channel from the station's middle)]
        for j, s in enumerate(stations):
            mid = (len(s["channels"]) - 1) / 2
            for k, c in enumerate(s["channels"]):
                self.owner.setdefault(c, []).append((j, abs(k - mid)))
        self.rms = np.full((SECONDS_PER_DAY, n_st), np.nan, np.float32) if rms else None
        self.waves = [np.full((n_samp, n_st), np.nan, np.float32) for _ in range(n_windows)]
        self.best = np.full((n_windows, n_st), np.inf)

    def add(self, ch: int, rms, waves) -> None:
        for j, dist in self.owner.get(ch, []):
            if rms is not None and self.rms is not None:
                self.rms[:, j] = np.fmax(self.rms[:, j], rms)
            for e, w in enumerate(waves or []):
                if dist < self.best[e, j] and np.isfinite(w).any():
                    self.best[e, j] = dist
                    self.waves[e][:, j] = w


def _run(job, paths: dict[int, str], acc: _Stations, day_start: str, frequency, windows, workers: int, desc: str) -> None:
    jobs = [(c, paths[c], day_start, tuple(frequency), windows) for c in sorted(acc.owner) if c in paths]
    with ProcessPoolExecutor(workers) as pool:
        for ch, rms, waves in tqdm(pool.map(job, jobs, chunksize=4), total=len(jobs), desc=desc):
            acc.add(ch, rms, waves)


def build_day(
    paths: dict[int, str],
    stations: list[dict],
    day_start: str,
    frequency,
    events: list[str],
    half: float,
    event_fs: int = 50,
    workers: int = 6,
    cache: "DayCache | None" = None,
):
    """
    paths: channel → MiniSEED file. Returns (log_rms (86400, n_st), events for `write_day`, each with the 100 Hz
    trace under "pick" so first-arrival picks run on the original sampling).

    With a `cache` (src/web/cache.py) only what is missing is computed: no cached RMS → one full-day pass over every
    channel (~5 min for 0708), which also cuts the missing event windows; RMS cached → only the missing event windows
    are read (± 30 s pad); everything cached → no MiniSEED is read at all.
    """
    d0 = np.datetime64(day_start)
    log_rms = cache.load_rms() if cache else None
    have = {t: cache.load_event(t) for t in events} if cache else {}
    need = [t for t in events if have.get(t) is None]
    windows = [(_seconds(t, d0) - half, 2 * half) for t in need]
    n_samp = int(round(2 * half * FS))
    if log_rms is None:
        acc = _Stations(stations, len(need), n_samp, rms=True)
        _run(_day_job, paths, acc, day_start, frequency, windows, workers, "channels (full day)")
        with np.errstate(divide="ignore", invalid="ignore"):
            log_rms = np.log10(acc.rms.astype(np.float64))
        log_rms[~np.isfinite(log_rms)] = np.nan
        if cache:
            cache.save_rms(log_rms)
    elif need:
        acc = _Stations(stations, len(need), n_samp, rms=False)
        _run(_window_job, paths, acc, day_start, frequency, windows, workers, f"channels ({len(need)} event windows)")
    else:
        acc = _Stations(stations, 0, n_samp, rms=False)
    for t, w in zip(need, acc.waves):
        have[t] = w
        if cache:
            cache.save_event(t, w)
    print(f"RMS: {'computed' if acc.rms is not None else 'cached'}; events: {len(need)} computed, {len(events) - len(need)} cached")

    step = FS // event_fs
    out = []
    for t in events:
        w = np.nan_to_num(have[t].astype(np.float64))
        start = str(d0 + np.timedelta64(int(round((_seconds(t, d0) - half) * 1e3)), "ms"))
        out.append(dict(time=t, start=start, fs=event_fs, data=w[::step], pick=dict(fs=FS, data=w)))  # 1–20 Hz band: plain decimation is alias-free
    return log_rms, out
