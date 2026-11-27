"""
Real MiDAS day → web-viewer arrays, one channel file at a time (a day file is ~35 MB as float32, the whole fiber
would not fit in memory at once).

Per channel: demean + linear detrend, cosine taper on the day's ends, 4-pole zero-phase Butterworth bandpass (the same
filter as `Water.process`), then
    - 1 s RMS of the filtered trace for the whole day (seconds touching a data gap → NaN),
    - the filtered 100 Hz trace in each event window (origin ± half seconds).
Per station (`every` channels): log10 of the max RMS of its channels; event waveform = its middle channel with data.
"""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import obspy
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm

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


def _job(args):
    ch, path, day_start, frequency, windows = args
    return ch, *process_channel(path, day_start, frequency, windows)


def build_day(paths: dict[int, str], stations: list[dict], day_start: str, frequency, events: list[str], half: float, event_fs: int = 50, workers: int = 6):
    """
    paths: channel → MiniSEED file. Returns (log_rms (86400, n_st), events for `write_day`, each with the 100 Hz
    trace under "pick" so first-arrival picks run on the original sampling).
    """
    d0 = np.datetime64(day_start)
    windows = [(_seconds(t, d0) - half, 2 * half) for t in events]
    n_st = len(stations)
    rms_st = np.full((SECONDS_PER_DAY, n_st), np.nan, np.float32)
    wave_st = [np.full((int(round(2 * half * FS)), n_st), np.nan, np.float32) for _ in events]
    owner = {}  # channel → (station index, distance of the channel from the station's middle)
    for j, s in enumerate(stations):
        mid = (len(s["channels"]) - 1) / 2
        for k, c in enumerate(s["channels"]):
            owner.setdefault(c, []).append((j, abs(k - mid)))
    best = np.full(n_st, np.inf)  # how far the stored waveform channel is from the station's middle
    jobs = [(c, paths[c], day_start, tuple(frequency), windows) for c in sorted(owner) if c in paths]
    with ProcessPoolExecutor(workers) as pool:
        for ch, rms, waves in tqdm(pool.map(_job, jobs, chunksize=4), total=len(jobs), desc="channels"):
            if rms is None:
                continue
            for j, dist in owner[ch]:
                rms_st[:, j] = np.fmax(rms_st[:, j], rms)
                if waves and dist < best[j] and np.isfinite(waves[0]).any():
                    best[j] = dist
                    for e, w in enumerate(waves):
                        wave_st[e][:, j] = w
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rms = np.log10(rms_st.astype(np.float64))
    log_rms[~np.isfinite(log_rms)] = np.nan
    step = FS // event_fs
    out = []
    for t, (s0, _), w in zip(events, windows, wave_st):
        w = np.nan_to_num(w.astype(np.float64))
        start = str(d0 + np.timedelta64(int(round(s0 * 1e3)), "ms"))
        out.append(dict(time=t, start=start, fs=event_fs, data=w[::step], pick=dict(fs=FS, data=w)))  # 1–20 Hz band: plain decimation is alias-free
    return log_rms, out
