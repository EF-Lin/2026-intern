"""
Synthetic day for the web viewer (no MiniSEED needed). Same output format as the real export.

Ingredients, all in counts:
- background noise per station: deeper is quieter, surface is louder in Taiwan daytime;
- vehicles driving along Surface B (slow moving bright spots in the 1 s RMS);
- earthquakes from a deep source: up-going P/S arrivals climb the holes, the surface reflection goes back down.
"""

import numpy as np

from .store import SECONDS_PER_DAY

V_AVG = 4000.0  # m/s, straight-ray average from the source to the surface
VP_NEAR, VS_NEAR = 2000.0, 900.0  # m/s near the surface (sets the up-/down-going moveout in the holes)


def _ricker(t: np.ndarray, f: float) -> np.ndarray:
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def _seconds(t: str, day_start: np.datetime64) -> float:
    return float((np.datetime64(t) - day_start) / np.timedelta64(1, "ms")) / 1e3


def background(stations: list[dict], rng: np.random.Generator) -> np.ndarray:
    """(86400, n_st) log10 RMS of the noise floor."""
    n = len(stations)
    depth = np.array([s["depth"] for s in stations])
    base = 1.3 + 0.12 * rng.standard_normal(n) - 0.5 * depth / 700
    t = np.arange(SECONDS_PER_DAY)
    local_h = ((t / 3600) + 8) % 24
    day = 1 / (1 + np.exp(-(local_h - 6.5) * 2)) / (1 + np.exp((local_h - 22) * 2))
    surface = np.exp(-depth / 100)
    log = base[None, :] + 0.45 * day[:, None] * surface[None, :]
    return log + 0.05 * rng.standard_normal((SECONDS_PER_DAY, n))


def vehicles(stations: list[dict], rng: np.random.Generator, n_pass: int = 24) -> np.ndarray:
    """(86400, n_st) linear RMS added by cars driving along Surface B at 6–12 m/s."""
    out = np.zeros((SECONDS_PER_DAY, len(stations)), dtype=np.float32)
    idx = [i for i, s in enumerate(stations) if s["section"].startswith("Surface B")]
    if not idx:
        return out
    pos = np.array([[stations[i]["x"], stations[i]["y"]] for i in idx])
    fwd = [i for i in idx if stations[i]["section"] == "Surface B (forth)"]
    path = np.array([[stations[i]["x"], stations[i]["y"]] for i in fwd])
    seg = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    for _ in range(n_pass):
        start = rng.uniform(7, 21) * 3600 - 8 * 3600  # Taiwan daytime, in UTC seconds
        speed = rng.uniform(6, 12)
        amp = 10 ** rng.uniform(1.6, 2.2)
        dur = int(seg[-1] / speed)
        back = rng.random() < 0.5
        for k in range(dur):
            s = speed * k
            s = seg[-1] - s if back else s
            car = np.array([np.interp(s, seg, path[:, 0]), np.interp(s, seg, path[:, 1])])
            d = np.linalg.norm(pos - car, axis=1)
            t = int(start) + k
            if 0 <= t < SECONDS_PER_DAY:
                out[t, idx] += amp * np.exp(-((d / 40) ** 2))
    return out


def event_waveform(stations: list[dict], noise_log: np.ndarray, origin: float, start: float, n: int, fs: float, src: np.ndarray, mag: float, rng) -> np.ndarray:
    """(n, n_st) synthetic filtered waveform in counts for one event; noise_log is the per-station log10 noise level."""
    t = start + np.arange(n) / fs
    out = rng.standard_normal((n, len(stations))) * (10**noise_log)[None, :]
    amp_p = 10 ** (mag + 0.4)
    for j, s in enumerate(stations):
        if not s["has_data"]:
            out[:, j] = 0
            continue
        r = np.linalg.norm(np.array([s["x"], s["y"], 0.0]) - src)
        tp_surf, ts_surf = origin + r / V_AVG, origin + r / (V_AVG / 1.73)
        z = s["depth"]
        geo = 3000 / max(r, 3000)
        for t_s, v, f, a, decay in ((tp_surf, VP_NEAR, 6.0, amp_p, 4.0), (ts_surf, VS_NEAR, 3.0, 3 * amp_p, 12.0)):
            up, down = t_s - z / v, t_s + z / v
            out[:, j] += a * geo * _ricker(t - up, f)
            out[:, j] -= 0.6 * a * geo * _ricker(t - down, f)
            coda = np.where(t > up, np.exp(-(t - up) / decay), 0)
            out[:, j] += 0.25 * a * geo * coda * rng.standard_normal(n)
    return out


def make_demo(stations: list[dict], day_start: str, events: list[dict], fs: float = 50.0, half: float = 300.0, seed: int = 0):
    """
    Returns (log_rms, events_out). events: [{"time": ISO UTC, "mag": log10 amplitude scale}].
    Inside each event window the 1 s RMS is computed from the synthetic waveform, so the two views agree.
    """
    rng = np.random.default_rng(seed)
    t0 = np.datetime64(day_start)
    noise = background(stations, rng)
    lin = 10**noise + vehicles(stations, rng)
    has = np.array([s["has_data"] for s in stations])
    xy = np.array([[s["x"], s["y"]] for s in stations])
    centre = xy.mean(axis=0)

    out_events = []
    for k, ev in enumerate(events):
        origin = _seconds(ev["time"], t0)
        start = origin - half
        n = int(2 * half * fs)
        # source position: azimuth (deg clockwise from north, site → source), epicentral distance and depth (m)
        az = np.radians(ev.get("azimuth", rng.uniform(0, 360)))
        dist, depth = ev.get("distance", 8000.0), ev.get("depth", 5000.0)
        src = np.array([centre[0] + dist * np.sin(az), centre[1] + dist * np.cos(az), depth])
        level = np.median(noise[int(start) : int(start) + 60], axis=0)
        data = event_waveform(stations, level, origin, start, n, fs, src, ev.get("mag", 2.0), rng)
        per_s = int(fs)
        rms = np.sqrt((data[: n // per_s * per_s].reshape(-1, per_s, len(stations)) ** 2).mean(axis=1))
        s0 = int(start)
        lin[s0 : s0 + rms.shape[0]] = rms
        truth = dict(azimuth=round(float(np.degrees(az)) % 360, 1), distance=dist, depth=depth, v_avg=V_AVG)
        out_events.append(dict(time=ev["time"], start=str(t0 + np.timedelta64(int(start * 1000), "ms")), fs=fs, data=data, source=truth))

    with np.errstate(divide="ignore"):
        log_rms = np.log10(lin)
    log_rms[:, ~has] = np.nan
    return log_rms, out_events
