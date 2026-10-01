import numpy as np
from tqdm import tqdm

from src.door import Search, load_h5, load_mini, save, transfer
from src.plot import Fall, Finger, Water
from src.utils import timer

se = Search(folder="folder")
times = ["2026-07-08T15:47:32"]


@timer
def draw():
    st = load_mini(se.find(1559))
    fi = Finger(st).process()
    st = load_mini(se.multi_find(1578, 6))
    pa, stations = transfer(st, start=20)
    for i in tqdm(times, desc="Generating Img"):
        t = np.datetime64(i)
        name = f"{stations[0]}_to_{stations[1]}_waterfall_plot_{i.replace(':', '')}"

        fi.set_time_range(i).focus_save(figsize=(30, 4))

        wa = Water(pa).cut(r=(t, 10)).process()
        fa = Fall([wa.pa], filename=name, title=[f"{str(wa.pa.attrs.time_min.astype("datetime64[m]")).replace('T', ' ')} E-W Waterfall Plot"], figsize=(18, 2))
        fa.set_plot(yname="distance").waterfall_save()


@timer
def mini_2_h5():
    st = load_mini(se.multi_find(1258, 120))
    pa, stations = transfer(st, start=200)
    wa = Water(pa=pa).process()
    save(wa.pa, name="file")


@timer
def pn():
    from src.analysis import Phase

    data = load_h5("2026-07-08T154732.h5")
    phase = Phase()
    picks = phase.run_patch(data).save()
    print(picks)


if __name__ == "__main__":
    draw()
