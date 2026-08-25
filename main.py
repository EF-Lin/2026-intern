import numpy as np
from tqdm import tqdm

from src.door import Search, load_csv, load_h5, load_mini, save, transfer
from src.plot import Fall, Finger, Water
from src.utils import timer

se = Search(folder="data_100Hz")

# times = load_csv("asset/78.csv")[:10]"2026-07-08T15:47:32"
main = ["2026-07-08T15:47:32"]
times = [
    "2026-07-08T14:27:12",
    "2026-07-08T15:38:54",
    # main-eq
    "2026-07-08T16:21:14",
    "2026-07-08T19:27:34",
    "2026-07-08T19:52:26",
    "2026-07-08T19:53:03",
    "2026-07-08T20:56:46",
    "2026-07-08T21:28:10",
]


@timer
def draw():
    # 1559
    st = load_mini(se.find(1559))
    fi = Finger(st)
    fi = fi.process()

    # A (1208, 170) B (650, 122)
    # (1559, 25) N-S(1559, 5) NE-SW(1564, 7) SW-NE(1571, 7) E-W(1578, 6)
    # st = load_mini(se.multi_find(1578, 6))  # 1213, 165
    # pa, stations = transfer(st)  # , start=20
    for i in tqdm(main, desc="Generating Img"):
        t = np.datetime64(i)  # ["Time"]
        # name = f"{stations[0]}_to_{stations[1]}_waterfall_plot_{i.replace(':', '')}"  # ["Time"]
        # name = f"waterfall_plot_{i["Time"].replace(':', '')}"

        fi.set_time_range(i).focus_save(figsize=(30, 4))  # ["Time"]
        # fi.gif()

        # wa = Water(pa).cut(r=(t, 10)).process()

        # fa = Fall([wa.pa], filename=name, title=[f"{str(wa.pa.attrs.time_min.astype("datetime64[m]")).replace('T', ' ')} E-W Waterfall Plot"], figsize=(18, 2))
        # fa.gif(d=10, dpi=150)
        # fa.set_plot(yname="distance").waterfall_save()


@timer
def mini_2_h5():
    st = load_mini(se.multi_find(1258, 120))
    pa, stations = transfer(st, start=200)
    wa = Water(pa=pa).cut(("2026-07-08T15:17:32", "2026-07-08T16:17:32")).process()
    save(wa.pa, name=main[0])


@timer
def pn():
    from src.analysis import Phase

    data = load_h5("2026-07-08T154732.h5")
    phase = Phase()
    picks = phase.run_patch(data).save()
    print(picks)


if __name__ == "__main__":
    pn()
