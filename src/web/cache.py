"""
Per-day cache of the slow part of `web_day` (reading + filtering MiniSEED), so later exports only recompute what
changed: <folder>/
    key.json                 the settings the cached arrays depend on
    rms.npy                  float16 (86400, n_st) log10 1 s RMS per station, NaN = no data (~58 MB for 336 stations)
    event_YYYYmmddTHHMMSS.npz  compressed float32 (n_samp, n_st) filtered 100 Hz station traces, one per event time

When the filter band, station grouping, available channels or day change, everything is dropped; when only the event
half-window changes, only the event files are dropped. Changing the main shock, clip window or pick settings needs
no cache change at all (those are applied when the site files are written).
"""

import hashlib
import json
from pathlib import Path

import numpy as np


def _digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


class DayCache:
    def __init__(self, folder: str | Path, *, day_start: str, frequency, stations: list[dict], channels, half: float):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        day = dict(
            day_start=day_start,
            frequency=[float(f) for f in frequency],
            stations=_digest([s["channels"] for s in stations]),
            channels=_digest(sorted(int(c) for c in channels)),
        )
        key = dict(day=day, half=float(half))
        path = self.folder / "key.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if old is None or old.get("day") != day:
            self._drop("rms.npy", "event_*.npz")
        elif old.get("half") != key["half"]:
            self._drop("event_*.npz")
        path.write_text(json.dumps(key, indent=1), encoding="utf-8")

    def _drop(self, *patterns: str) -> None:
        for pattern in patterns:
            for f in self.folder.glob(pattern):
                f.unlink()

    def _event_path(self, t: str) -> Path:
        return self.folder / f"event_{str(np.datetime64(t, 's')).replace('-', '').replace(':', '')}.npz"

    def load_rms(self) -> np.ndarray | None:
        f = self.folder / "rms.npy"
        return np.load(f).astype(np.float64) if f.exists() else None

    def save_rms(self, log_rms: np.ndarray) -> None:
        np.save(self.folder / "rms.npy", log_rms.astype(np.float16))  # log10: float16 keeps ~0.1 %, far below one color step

    def load_event(self, t: str) -> np.ndarray | None:
        f = self._event_path(t)
        if not f.exists():
            return None
        with np.load(f) as z:
            return z["data"]

    def save_event(self, t: str, data: np.ndarray) -> None:
        np.savez_compressed(self._event_path(t), data=data.astype(np.float32))
