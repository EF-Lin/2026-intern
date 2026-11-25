"""
First-arrival picks and a plane-wave fit for one event window, so the web viewer can show where the wave comes from.

Steps (all on the filtered event waveform, before int16 quantisation):
1. coarse onset: median over stations of the STA/LTA of the squared trace, first crossing of `thr` near the event time;
2. per-station rough pick: that station's own STA/LTA crossing close to the coarse onset;
3. refinement: cross-correlate every station with the stack of the aligned onsets (iterated), sub-sample by a parabola;
   stations with a low correlation coefficient stay unpicked;
4. plane-wave fit t = t0 + sx·x + sy·y − q·depth (q only when the picks span depth), trimmed least squares,
   bootstrap over stations for the uncertainty.

Back-azimuth = direction the wave comes FROM, degrees clockwise from north = direction of −(sx, sy).
"""

import numpy as np


def _moving_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Trailing mean over n samples along axis 0 (first n-1 rows use what is available)."""
    c = np.cumsum(x, axis=0)
    out = c.copy()
    out[n:] = c[n:] - c[:-n]
    k = np.minimum(np.arange(1, len(x) + 1), n).reshape(-1, *([1] * (x.ndim - 1)))
    return out / k


def sta_lta(data: np.ndarray, fs: float, sta: float = 0.1, lta: float = 3.0) -> np.ndarray:
    """(n, n_st) ratio of the short-term mean of data² to the long-term mean just before it."""
    e = data.astype(float) ** 2
    ns, nl = max(1, int(sta * fs)), max(2, int(lta * fs))
    s = _moving_mean(e, ns)
    long = _moving_mean(e, nl)
    lag = np.empty_like(long)
    lag[ns:] = long[:-ns]  # LTA window ends where the STA window starts
    lag[:ns] = long[:1]
    floor = np.median(e, axis=0) * 1e-3 + 1e-30
    r = s / np.maximum(lag, floor)
    r[:nl] = 1.0  # LTA not settled yet
    return r


def _first_crossing(cf: np.ndarray, thr: float) -> int | None:
    i = np.flatnonzero(cf > thr)
    return int(i[0]) if i.size else None


def _xcorr_lag(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Offset (samples, sub-sample) of template b inside the longer trace a, and the normalised correlation there."""
    b = b - b.mean()
    nb = np.linalg.norm(b)
    if nb == 0 or len(a) < len(b):
        return 0.0, 0.0
    num = np.correlate(a, b, mode="valid")
    c1 = np.concatenate([[0], np.cumsum(a)])
    c2 = np.concatenate([[0], np.cumsum(a * a)])
    m = len(b)
    var = (c2[m:] - c2[:-m]) - (c1[m:] - c1[:-m]) ** 2 / m  # local sum of squares about the local mean
    r = num / (np.sqrt(np.maximum(var, 1e-30)) * nb)
    k = int(np.argmax(r))
    shift = 0.0
    if 0 < k < len(r) - 1:
        y0, y1, y2 = r[k - 1], r[k], r[k + 1]
        d = y0 - 2 * y1 + y2
        if d < 0:
            shift = 0.5 * (y0 - y2) / d
    return k + shift, float(r[k])


def pick_arrivals(
    data: np.ndarray,
    fs: float,
    t_hint: float,
    usable: np.ndarray | None = None,
    search: tuple[float, float] = (-30.0, 60.0),
    thr: float = 4.0,  # minimum STA/LTA trigger level (raised to 2× the pre-event maximum when there is a quiet stretch)
    pre: float = 0.2,
    post: float = 0.5,
    max_shift: float = 0.6,
    min_cc: float = 0.5,
    n_iter: int = 3,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """
    data: (n, n_st) filtered waveform; t_hint: event time in seconds from the first sample.
    Returns (pick seconds from the first sample, NaN = unpicked; correlation coefficient; coarse onset or None).
    """
    n, n_st = data.shape
    usable = np.ones(n_st, bool) if usable is None else usable.astype(bool) & (np.abs(data).max(axis=0) > 0)
    picks, cc = np.full(n_st, np.nan), np.zeros(n_st)
    if not usable.any():
        return picks, cc, None
    cf = sta_lta(data, fs)
    a, b = max(0, int((t_hint + search[0]) * fs)), min(n, int((t_hint + search[1]) * fs))
    # 90th percentile: the first arrival may be clear only on the quieter (deep) stations
    stack = np.percentile(cf[a:b, usable], 90, axis=1)
    # the trigger level must also clear twice the largest value of the same stack in the quiet part before the search
    lta_n = int(3.0 * fs)
    if a - lta_n >= int(10 * fs):
        quiet = np.percentile(cf[lta_n:a, usable], 90, axis=1)
        thr = max(thr, 2 * float(quiet.max()))
    i0 = _first_crossing(stack, thr)
    if i0 is None:
        return picks, cc, None
    onset = a + i0

    # rough per-station picks near the coarse onset
    w0, w1 = max(0, onset - int(1.5 * max_shift * fs)), min(n, onset + int(1.5 * max_shift * fs))
    rough = np.full(n_st, np.nan)
    for j in np.flatnonzero(usable):
        k = _first_crossing(cf[w0:w1, j], thr)
        rough[j] = (w0 + k) if k is not None else onset

    npre, npost, ml = int(pre * fs), int(post * fs), int(max_shift * fs)

    def window(j, centre):
        if not np.isfinite(centre):
            return None
        c = int(round(centre))
        lo, hi = c - npre, c + npost
        if lo < 0 or hi > n:
            return None
        return data[lo:hi, j]

    cur = rough.copy()
    for it in range(n_iter):
        ml = int(max_shift * fs) if it == 0 else int(max_shift * fs / 3)  # later rounds only fine-tune
        segs = [window(j, cur[j]) for j in range(n_st)]
        ok = [j for j in range(n_st) if usable[j] and segs[j] is not None]
        if not ok:
            return picks, cc, onset / fs
        tmpl = np.mean([segs[j] / (np.abs(segs[j]).max() or 1) for j in ok], axis=0)
        for j in ok:
            # search a longer stretch of the trace for the template
            c = int(round(cur[j]))
            lo, hi = c - npre - ml, c + npost + ml
            if lo < 0 or hi > n:
                continue
            lag, r = _xcorr_lag(data[lo:hi, j], tmpl)
            # lag is where the template starts inside [lo, hi); its onset sits npre after that
            cur[j] = lo + lag + npre
            cc[j] = r
        # do not let picks wander away from the group
        med = np.nanmedian(cur[ok])
        cur = np.where(np.abs(cur - med) > 3 * max_shift * fs, rough, cur)

    good = usable & (cc >= min_cc)
    picks[good] = cur[good] / fs
    return picks, cc, onset / fs


def _lstsq(A: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(A, t, rcond=None)[0]


def _design(xyz: np.ndarray, use_depth: bool) -> np.ndarray:
    cols = [np.ones(len(xyz)), xyz[:, 0], xyz[:, 1]]
    if use_depth:
        cols.append(-xyz[:, 2])
    return np.column_stack(cols)


def _baz(sx: float, sy: float) -> float:
    return float(np.degrees(np.arctan2(-sx, -sy)) % 360)


def fit_plane_wave(
    xyz: np.ndarray,
    t: np.ndarray,
    groups: np.ndarray | None = None,
    pos_sigma: float = 100.0,
    min_depth_span: float = 100.0,
    floor: float = 0.01,
    n_boot: int = 300,
    seed: int = 0,
) -> dict | None:
    """
    xyz: (n, 3) east, north, depth (m); t: arrival (s), NaN = unpicked.
    Trimmed least squares (drop |residual| > max(3·1.48·MAD, floor), refit, up to three rounds).
    ± values: std over `n_boot` refits that each (a) resample the kept stations with replacement and
    (b) shift every fiber section (`groups`, one label per station) horizontally by N(0, pos_sigma) m,
    because channel positions are only known to about ±100 m. Returns None with fewer than 5 picks.
    """
    ok = np.isfinite(t)
    xyz, t = xyz[ok], t[ok]
    groups = np.zeros(len(ok), int) if groups is None else np.unique(np.asarray(groups), return_inverse=True)[1]
    groups = groups[ok]
    if len(t) < 5:
        return None
    use_depth = bool(np.ptp(xyz[:, 2]) >= min_depth_span)
    A = _design(xyz, use_depth)
    keep = np.ones(len(t), bool)
    for _ in range(3):
        m = _lstsq(A[keep], t[keep])
        r = t - A @ m
        mad = np.median(np.abs(r[keep] - np.median(r[keep])))
        new = np.abs(r) <= max(3 * 1.4826 * mad, floor)
        if new.sum() < 5 or (new == keep).all():
            break
        keep = new
    m = _lstsq(A[keep], t[keep])
    r = t - A @ m
    sx, sy = m[1], m[2]
    sh = float(np.hypot(sx, sy))
    baz = _baz(sx, sy)

    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(keep)
    ng = groups.max() + 1
    res = {"pick": ([], []), "position": ([], [])}
    for kind in res:
        for _ in range(n_boot):
            if kind == "pick":
                b, Ab = rng.choice(idx, size=len(idx)), A
            else:
                b = idx
                shift = rng.normal(0, pos_sigma, (ng, 2))[groups]
                Ab = _design(xyz + np.column_stack([shift, np.zeros(len(xyz))]), use_depth)
            mb = _lstsq(Ab[b], t[b])
            res[kind][0].append((_baz(mb[1], mb[2]) - baz + 180) % 360 - 180)
            res[kind][1].append(1 / max(np.hypot(mb[1], mb[2]), 1e-9))
    sd = {k: (float(np.std(v[0])), float(np.std(np.clip(v[1], 0, 1e6)))) for k, v in res.items()}
    out = dict(
        baz=round(baz, 1),
        baz_err=round(float(np.hypot(sd["pick"][0], sd["position"][0])), 1),
        baz_err_pick=round(sd["pick"][0], 1),
        baz_err_position=round(sd["position"][0], 1),
        v_app=round(1 / max(sh, 1e-9), 0),
        v_app_err=round(float(np.hypot(sd["pick"][1], sd["position"][1])), 0),
        slowness=[round(float(sx) * 1e3, 4), round(float(sy) * 1e3, 4)],  # s/km east, north
        residual_ms=round(float(np.sqrt(np.mean(r[keep] ** 2))) * 1e3, 1),
        n_used=int(keep.sum()),
        used=keep,
    )
    if use_depth:
        q = float(m[3])
        out |= dict(q=round(q * 1e3, 4), incidence=round(float(np.degrees(np.arctan2(sh, q))), 1) if q > 0 else None)
    return out


def neighbour_stack(data: np.ndarray, stations: list[dict], half: int = 2) -> np.ndarray:
    """
    Mean of each station with up to `half` neighbours on each side in the same fiber section (stations with data only).
    Stations are ~20 m apart, so ±2 neighbours shift the arrival by ≲ 10 ms at > 4 km/s: much less than a quarter period.
    """
    out = np.zeros_like(data, dtype=float)
    sec = [s["section"] for s in stations]
    has = [s.get("has_data", True) for s in stations]
    for j in range(len(stations)):
        if not has[j]:
            continue
        nb = [k for k in range(max(0, j - half), min(len(stations), j + half + 1)) if sec[k] == sec[j] and has[k]]
        out[:, j] = data[:, nb].mean(axis=1)
    return out


def event_arrivals(stations: list[dict], data: np.ndarray, fs: float, t_hint: float, half: int = 2, pos_sigma: float = 100.0) -> dict:
    """Picks + plane-wave fit for one event, as stored in meta.json under events[k].arrivals."""
    usable = np.array([s.get("has_data", True) for s in stations])
    picks, cc, onset = pick_arrivals(neighbour_stack(data, stations, half) if half else data, fs, t_hint, usable)
    xyz = np.array([[s["x"], s["y"], s["depth"]] for s in stations], float)
    sections = np.array([s["section"].split(" (")[0] for s in stations])  # both legs of a section move together
    fit = fit_plane_wave(xyz, picks, sections, pos_sigma=pos_sigma) if onset is not None else None
    first = float(np.nanmin(picks)) if np.isfinite(picks).any() else None
    rel = [None if not np.isfinite(p) else round((p - first) * 1e3, 1) for p in picks]
    out = dict(
        method=f"mean of ±{half} neighbour stations, STA/LTA onset + cross-correlation with stacked onset (cc >= 0.5), plane-wave fit t = t0 + s·(x, y) - q·depth, trimmed LSQ; ±1σ = station bootstrap ⊕ ±{pos_sigma:g} m section-position jitter",
        first=None if first is None else round(first, 3),  # s after the window start
        rel_ms=rel,
        n_picked=int(np.isfinite(picks).sum()),
        pos_sigma=pos_sigma,
    )
    if fit is not None:
        used = fit.pop("used")
        idx = np.flatnonzero(np.isfinite(picks))
        out["outlier"] = [int(i) for i in idx[~used]]
        out |= fit
    return out
