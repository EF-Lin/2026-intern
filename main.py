from src.load import load_mini, transfer, load_csv, Search
from src.plot import Finger, Water, Fall
from src.utils import timer
import numpy as np
from tqdm import tqdm


@timer
def main():
    se = Search("data_100Hz")
    li = load_csv("asset/78.csv")[50:]

    st = load_mini(se.find(1334))
    fi = Finger(st)
    fi.process()

    st = load_mini(se.multi_find(1208, 170))
    pa, stations = transfer(st)

    for i in tqdm(li):
        t = np.datetime64(i["Time"])
        name = f"{stations[0]}_to_{stations[1]}_waterfall_plot_{i["Time"].replace(':', '')}"

        fi.focus_save(i["Time"])

        wa = Water(pa, range=(t, 20))
        pa2 = wa.process(wa.cut())
        fa = Fall(pa2, name=name)
        fa.waterfall_save()


if __name__ == "__main__":
    main()
