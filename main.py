from src.load import load_mini, transfer, load_csv, Search
from src.plot import Finger, Water, Fall
from src.utils import timer
import numpy as np
from tqdm import tqdm


@timer
def main():
    se = Search(folder="data_100Hz")
    times = [load_csv("asset/78.csv")[62]]

    #st = load_mini(se.find(1559))
    #fi = Finger(st)
    #fi.process()

    #A (1208, 170) B (650, 122)
    # (1559, 25) (1559, 5) (1564, 7) (1571, 7) (1578, 6)
    st = load_mini(se.multi_find(1208, 170))
    pa, stations = transfer(st)
    #st1 = load_mini(se.multi_find(1559, 5))
    #st2 = load_mini(se.multi_find(1564, 6))
    #pa1, stations1 = transfer(st1)
    #pa2, stations2 = transfer(st2)

    # delete top
    dist_array = pa.coords.get_array("depth")
    pa = pa.select(depth=(dist_array[20], dist_array[-1]))

    for i in tqdm(times):
        t = np.datetime64(i["Time"])
        name = f"{stations[0]}_to_{stations[1]}_waterfall_plot_{i["Time"].replace(':', '')}-short"
        #name = f"waterfall_plot_{i["Time"].replace(':', '')}"

        #fi.focus_save(i["Time"])

        wa = Water(pa, r=(t, 5)).cut().process()
        #wa1 = Water(pa1, r=(t, 5)).cut().process()
        #wa2 = Water(pa2, r=(t, 5)).cut().process()

        fa = Fall([wa.pa], filename=name, title=[f"Hole A {str(wa.pa.attrs.time_min.astype("datetime64[m]")).replace('T', ' ')} Waterfall Plot"], figsize=(24, 6))
        #fa = Fall([wa.pa], filename=name, title=[f"{str(wa.pa.attrs.time_min.astype("datetime64[m]")).replace('T', ' ')} EW Waterfall Plot"], figsize=(24, 6))
        #fa = Fall([wa.pa], filename=name, title=["Waterfall Plot"], figsize=(24, 6))
        #fa = Fall([wa1.pa, wa2.pa], filename=name, title=["N-S Waterfall Plot", "E-W Waterfall Plot"], figsize=(18, 12))
        fa.set_plot(yname="distance").waterfall_save()


if __name__ == "__main__":
    main()
