"""MiDAS 資料處理任務。參數集中在 config.toml，修改最下方 __main__ 選擇要跑的任務。"""

import numpy as np
from tqdm import tqdm

from src.analysis import CC
from src.door import Search, load_csv, load_h5, load_mini, save, transfer
from src.plot import Finger, Folk, Water
from src.utils import dicts2csv, load_config, timer

cfg = load_config()
_s = cfg["search"]
se = Search(folder=_s["folder"], filename=_s.get("filename"), year=_s.get("year", 2026), julday=_s.get("julday", 189))
main = cfg["events"]["main"]
times = cfg["events"]["times"]


def _segment(name: str) -> tuple[int, int]:
    seg = cfg["segments"][name]
    return seg["station"], seg["n"]


@timer
def draw():
    c = cfg["draw"]
    fi = Finger(load_mini(se.find(c["station"]))).process()
    for i in tqdm(main, desc="Generating Img"):
        fi.set_time_range(i).focus_save(figsize=tuple(c["figsize"]))


@timer
def mini_2_h5():
    c = cfg["mini_2_h5"]
    st = load_mini(se.multi_find(*_segment(c["segment"])))
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
    files = se.multi_find(*_segment(c["segment"]))

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
    pa1, _ = transfer(load_mini(se.find(c["station_a"])), start=c["start"])
    pa2, _ = transfer(load_mini(se.find(c["station_b"])), start=c["start"])

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


if __name__ == "__main__":
    compareAB()
