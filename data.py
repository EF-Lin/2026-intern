"""MiDAS 資料處理任務。參數集中在 config.toml，修改最下方 __main__ 選擇要跑的任務。"""

from functools import cache
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.analysis import CC, fold_legs, leg_shift, separate_updown
from src.door import Search, load_csv, load_h5, load_mini, save, transfer
from src.plot import Fall, Finger, Folk, Water, Wiggle
from src.utils import dicts2csv, load_config, timer

cfg = load_config()
main = cfg["events"]["main"]
times = cfg["events"]["times"]


@cache
def se() -> Search:
    """資料夾在第一次用到時才檢查，web_demo 等不讀 MiniSEED 的任務不需要 data/。"""
    s = cfg["search"]
    return Search(folder=s["folder"], filename=s.get("filename"), year=s.get("year", 2026), julday=s.get("julday", 189))


def _segment(name: str) -> tuple[int, int]:
    seg = cfg["segments"][name]
    return seg["station"], seg["n"]


@timer
def draw():
    c = cfg["draw"]
    fi = Finger(load_mini(se().find(c["station"]))).process()
    for i in tqdm(main, desc="Generating Img"):
        fi.set_time_range(i).focus_save(figsize=tuple(c["figsize"]))


@timer
def mini_2_h5():
    c = cfg["mini_2_h5"]
    st = load_mini(se().multi_find(*_segment(c["segment"])))
    pa, _ = transfer(st, start=c["start"])
    wa = Water(pa=pa).process()
    save(wa.pa, name=main[0])


@timer
def pn():
    from src.analysis import Phase

    data = load_h5(cfg["pn"]["h5"])
    picks = Phase().run_patch(data).save()
    print(picks)


@timer
def fk():
    c = cfg["fk"]
    files = se().multi_find(*_segment(c["segment"]))

    for i in tqdm(main, desc="Generating Img"):
        # 只讀事件前後 cut + pad 秒；pad 吸收 taper / 濾波的邊界效應，處理後再切回 ±cut 秒
        patch, stations = transfer(load_mini(files, (i, c["cut"]), pad=c["pad"]), start=c["start"])
        name = f"{stations[0]}_to_{stations[1]}_fk_plot_{i.replace(':', '')}"
        wa = Water(pa=patch).process().cut((np.datetime64(i), c["cut"]))
        folk = Folk(wa.pa)
        folk.set_plot().fk_save(filename=name)
        print(folk.info)


@timer
def compareAB():
    c = cfg["compareAB"]
    fk_times = load_csv(cfg["events"]["fk_csv"])
    # 同一組測站要切很多個時間點，整天讀一次留在記憶體重複切，比每個事件重讀檔案快
    pa1, _ = transfer(load_mini(se().find(c["station_a"])), start=c["start"])
    pa2, _ = transfer(load_mini(se().find(c["station_b"])), start=c["start"])

    for i in tqdm(fk_times):
        t = np.datetime64(i["Time"])
        wa1 = Water(pa1).cut((t, c["cut"])).process()
        wa2 = Water(pa2).cut((t, c["cut"])).process()

        ret = CC(wa1.pa.data, [wa2.pa.data]).calculate().pick()
        print(ret)

        # 將計算結果新增為該事件的新欄位
        i["correlation_AB"] = float(ret[0]["correlation"])
        i["lags_AB"] = int(ret[0]["lags"])

    dicts2csv(fk_times, filename=c["output"])


def _stack_legs(patch, stations: tuple[str, str], event: str, c: dict):
    """
    U 型光纖：把往上段（c["up_segment"]，第一個通道 = 底部轉折點）反轉、對到往下段的深度後平均，提高 S/N。
    先在 c["shift_window"] 內以互相關估計最佳通道位移（只印出來參考），實際使用 c["leg_offset"]。
    """
    up_files = se().multi_find(*_segment(c["up_segment"]))
    up, up_stations = transfer(load_mini(up_files, (event, c["cut"]), pad=c["pad"]), spacing=c["spacing"])
    if c.get("shift_window"):
        t0 = np.datetime64(event)
        win = tuple(t0 + np.timedelta64(int(s * 1e3), "ms") for s in c["shift_window"])
        f = tuple(c["frequency"])
        a, b = (Water(pa=p).process(frequency=f).pa.select(time=win) for p in (patch, up))
        best, shifts, cc = leg_shift(a, b, max_shift=c.get("max_shift", 10))
        zero = cc[np.argmin(np.abs(shifts))]
        print(f"{event} leg shift: best {best:+.0f} ch (cc={np.nanmax(cc):.3f}), 0 ch cc={zero:.3f}, using {c.get('leg_offset', 0)}")
    return fold_legs(patch, up, offset=c.get("leg_offset", 0)), (stations[0], up_stations[1])


@timer
def updown():
    c = cfg["updown"]
    files = se().multi_find(*_segment(c["segment"]))
    events = [row["Time"] for row in load_csv(c["csv"])] if c.get("csv") else main

    for i in tqdm(events, desc="Generating Img"):
        # 多讀 pad 秒，在整段資料上做濾波與 f-k 上下行波分離，最後才切回 ±cut 秒，邊界效應落在圖外
        patch, stations = transfer(load_mini(files, (i, c["cut"]), pad=c["pad"]), spacing=c["spacing"], start=c["start"])
        if c.get("stack_legs"):
            patch, stations = _stack_legs(patch, stations, i, c)
        full = Water(pa=patch).process(frequency=tuple(c["frequency"])).pa
        masks = {"v_min": c.get("v_min"), "v_max": c.get("v_max"), "v_taper": c.get("v_taper"), "taper": c.get("taper", 0.2)}
        up, down = separate_updown(full, **masks)
        r = (np.datetime64(i), c["cut"])
        wa = Water(full).cut(r)
        up, down = Water(up).cut(r).pa, Water(down).cut(r).pa

        # 三張圖共用原始資料的色階，才能直接比較振幅
        vmax = float(np.percentile(np.abs(wa.pa.data), c["scale"] * 100))
        tag = f"{stations[0]}_to_{stations[1]}"
        stamp = i.replace(":", "")
        titles = [f"{i.replace('T', ' ')} {c.get('label', c['segment'])} {name}" for name in ("Raw", "Up-going", "Down-going")]
        Fall([wa.pa, up, down]).set_plot(figsize=tuple(c["figsize"]), title=titles, scale=(-vmax, vmax), yname="Depth (m)").waterfall_save(filename=f"{tag}_updown_{stamp}")
        if c.get("wiggle"):
            w = c["wiggle"]
            t0 = np.datetime64(i)
            win = (t0 + np.timedelta64(int(w["window"][0] * 1e3), "ms"), t0 + np.timedelta64(int(w["window"][1] * 1e3), "ms"))
            Wiggle([p.select(time=win) for p in (wa.pa, up, down)]).set_plot(
                figsize=tuple(w["figsize"]),
                title=titles,
                color=w["color"],
                every=w["every"],
                gain=w["gain"],
                per_trace=w["per_trace"],
                t0=t0,
                xname=f"Time after {i.replace('T', ' ')} (s)",
            ).wiggle_save(filename=f"{tag}_wiggle_{stamp}")
        if c.get("fk"):
            # 在 f-k 圖上畫出遮罩的速度邊界
            lines = {f"{key} = {masks[key]:g} m/s": masks[key] for key in ("v_min", "v_max") if masks[key]}
            Folk([wa.pa, up, down]).set_plot(title=[f"{name} f-k" for name in ("Raw", "Up-going", "Down-going")], v_lines=lines).fk_save(filename=f"{tag}_updown_fk_{stamp}")


def _web_info(c: dict, origin: tuple[float, float], demo: bool) -> dict:
    return dict(
        demo=demo,
        every=c["every"],
        origin=dict(lat=origin[0], lon=origin[1]),
        filter=dict(type="bandpass", frequency=c["frequency"]),
        units="counts",
        counts_to_strain_rate=c.get("counts_to_strain_rate") or None,
        timezone=dict(label="UTC+8", offset_hours=8),
    )


@timer
def web_demo():
    """合成一整天的假資料（不讀 MiniSEED），輸出成網站格式到 web/data/demo/。"""
    from src.web import build_stations, fiber_lines, load_sections, make_demo, write_day

    c, d = cfg["web"], cfg["web"]["demo"]
    sections, origin = load_sections(c["geometry"], leg_offset=c["leg_offset"])
    available = set(range(d["channels"][0], d["channels"][1] + 1))
    stations = build_stations(sections, c["every"], available)
    log_rms, events = make_demo(stations, d["start"], d["events"], fs=c["event_fs"], half=d.get("event_half", c["event_half"]), seed=d["seed"])
    arrivals = c.get("arrivals", {})
    arrivals = None if arrivals.get("skip") else {k: v for k, v in arrivals.items() if k != "skip"}
    folder = write_day(
        c["out"], d["day"], d["start"], stations, fiber_lines(sections), log_rms, events, _web_info(c, origin, demo=True), arrivals=arrivals, clip_window=tuple(c["event_clip"])
    )
    print(f"{len(stations)} stations -> {folder}")


@timer
def web_day(folder: str | None = None):
    """真實 MiniSEED 一整天 -> 網站格式 web/data/<day>/。folder 省略時用 [search] folder。"""
    from src.web import build_stations, fiber_lines, load_sections, write_day
    from src.web.real import build_day

    c = cfg["web"]
    s = Search(folder=folder or cfg["search"]["folder"], filename=cfg["search"].get("filename"), year=cfg["search"].get("year", 2026), julday=cfg["search"].get("julday", 189))
    sections, origin = load_sections(c["geometry"], leg_offset=c["leg_offset"])
    every = build_stations(sections, c["every"])
    paths = {ch: s._path(ch) for st in every for ch in st["channels"]}
    paths = {ch: p for ch, p in paths.items() if Path(p).is_file()}
    stations = build_stations(sections, c["every"], set(paths))
    start = f"{c['day'][:4]}-{c['day'][4:6]}-{c['day'][6:]}T00:00:00"
    times = list(c["events"]) + ([row["Time"] for row in load_csv(c["events_csv"])] if c.get("events_csv") else [])
    times = sorted(set(times), key=np.datetime64)  # time order (E1, E2, … on the page)
    print(f"{len(times)} events")
    log_rms, events = build_day(paths, stations, start, c["frequency"], times, c["event_half"], c["event_fs"], workers=c.get("workers", 6))
    for ev in events:
        if np.datetime64(ev["time"]) == np.datetime64(c.get("main_shock") or "NaT"):
            ev["label"] = "main shock"
    arrivals = c.get("arrivals", {})
    arrivals = None if arrivals.get("skip") else {k: v for k, v in arrivals.items() if k != "skip"}
    folder = write_day(c["out"], c["day"], start, stations, fiber_lines(sections), log_rms, events, _web_info(c, origin, demo=False), arrivals=arrivals, clip_window=tuple(c["event_clip"]))
    print(f"{len(paths)} channels, {len(stations)} stations -> {folder}")


if __name__ == "__main__":
    compareAB()
