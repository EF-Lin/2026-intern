"""
Write one day of web-viewer data to `<out>/<day>/`. The browser reads only these files, never MiniSEED.

Layout (all binary arrays are little-endian, time-major: one row per time step, one column per station):
    meta.json         stations, fiber lines, units, quantisation, file list, events
    overview.bin      uint8 (1440, n_st)   per-minute max of the quantised 1 s log10 RMS (timeline + 2D waterfall)
    rms_HH.bin        uint8 (3600, n_st)   1 s log10 RMS for UTC hour HH; 0 = no data
    event_K.bin       int16 (n_samp, n_st) filtered waveform around event K; value * scale[st] / 32767 = counts
<out>/index.json lists the available days for the page's day menu.
"""

import json
import warnings
from pathlib import Path

import numpy as np

from .arrivals import event_arrivals

SECONDS_PER_DAY = 86_400
EVENT_FULL = 32767  # int16 value that stands for each station's max |amplitude| in the event


def quantise(log_rms: np.ndarray, qmin: float, qmax: float) -> np.ndarray:
    """log10 RMS → uint8 1..255 over [qmin, qmax]; NaN → 0."""
    q = 1 + np.round((log_rms - qmin) / (qmax - qmin) * 254)
    q = np.clip(np.nan_to_num(q, nan=0), 0, 255).astype(np.uint8)
    q[~np.isnan(log_rms) & (q == 0)] = 1
    return q


def write_day(
    out: str | Path,
    day: str,
    start: str,
    stations: list[dict],
    lines: list[dict],
    log_rms: np.ndarray,
    events: list[dict],
    info: dict,
    arrivals: dict | None = None,
) -> Path:
    """
    log_rms: (86400, n_st) float log10 of 1 s RMS in counts, NaN where there is no data.
    events: [{"time": ISO origin, "start": ISO first sample, "fs": Hz, "data": (n_samp, n_st) float counts}].
    info: extra meta (filter, units, origin, …) merged into meta.json.
    arrivals: keyword arguments for `event_arrivals` (e.g. {"half": 2, "pos_sigma": 100}); None = skip the
    first-arrival picks / plane-wave fit stored as events[k].arrivals (see src/web/arrivals.py).
    """
    folder = Path(out) / day
    folder.mkdir(parents=True, exist_ok=True)
    n_st = len(stations)
    if log_rms.shape != (SECONDS_PER_DAY, n_st):
        raise ValueError(f"log_rms shape {log_rms.shape} != ({SECONDS_PER_DAY}, {n_st})")

    finite = log_rms[np.isfinite(log_rms)]
    qmin, qmax = (float(v) for v in np.percentile(finite, [0.01, 99.999])) if finite.size else (0.0, 1.0)
    q = quantise(log_rms, qmin, qmax)
    for h in range(24):
        q[h * 3600 : (h + 1) * 3600].tofile(folder / f"rms_{h:02d}.bin")
    q.reshape(1440, 60, n_st).max(axis=1).tofile(folder / "overview.bin")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN columns = stations without data
        median = np.nanmedian(log_rms, axis=0)
    p = np.nanpercentile(finite, [25, 99.99]) if finite.size else [qmin, qmax]

    ev_meta = []
    for k, ev in enumerate(events):
        data = np.asarray(ev["data"], dtype=float)
        scale = np.abs(data).max(axis=0)
        safe = np.where(scale > 0, scale, 1.0)
        np.round(data / safe * EVENT_FULL).astype(np.int16).tofile(folder / f"event_{k}.bin")
        nonzero = np.abs(data[:, scale > 0])
        arr = {}
        if arrivals is not None:
            t_hint = float((np.datetime64(ev["time"]) - np.datetime64(ev["start"])) / np.timedelta64(1, "ms")) / 1e3
            arr = {"arrivals": event_arrivals(stations, data, ev["fs"], t_hint, **arrivals)}
        ev_meta.append(
            dict(
                time=ev["time"],
                start=ev["start"],
                fs=ev["fs"],
                n=int(data.shape[0]),
                file=f"event_{k}.bin",
                dtype="int16",
                full=EVENT_FULL,
                scale=[round(float(v), 4) for v in scale],
                vmax=round(float(np.percentile(nonzero, 99)), 4) if nonzero.size else 1.0,
            )
            | ({"source": ev["source"]} if "source" in ev else {})  # demo only: the true source, to check direction estimates
            | arr
        )

    meta = (
        dict(
            day=day,
            start=start,
            seconds=SECONDS_PER_DAY,
            stations=[{k: v for k, v in s.items() if k != "channels"} | {"median": None if np.isnan(m) else round(float(m), 4)} for s, m in zip(stations, median)],
            lines=lines,
            rms=dict(files=[f"rms_{h:02d}.bin" for h in range(24)], hour=3600, qmin=qmin, qmax=qmax, default_range=[round(float(p[0]), 3), round(float(p[1]), 3)]),
            overview=dict(file="overview.bin", step=60),
            events=ev_meta,
        )
        | info
    )
    (folder / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    _update_index(Path(out), day)
    return folder


def _update_index(out: Path, day: str) -> None:
    index = out / "index.json"
    days = json.loads(index.read_text(encoding="utf-8")) if index.exists() else []
    if day not in days:
        days.append(day)
    index.write_text(json.dumps(sorted(days)), encoding="utf-8")
