from src.load.loadTR import load_mini, transfer
from src.load.loadCSV import load_csv
from src.finger import Finger
from src.search import Search
from src.waterfall import Water, Fall
import numpy as np



se = Search("data_100Hz")
li = load_csv("asset/78.csv")

"""
st = load_mini(se.find(1334))
fi = Finger(st)
fi.process()
"""

st = load_mini(se.multi_find(1208, 170))
pa, stations = transfer(st)
wa = Water(pa, range=(np.datetime64("2026-07-08T14:27:05"), 30))
wa.cut()
pa2 = wa.process()
fa = Fall(pa2)
fa.waterfall_plot()

"""
for i in li:
    t = UTCDateTime(i["Time"])
    #fi.focus_save(t)

    fa = Fall(st=st2.copy(), y_factor=20, range=(t, 20))
    fa.cut()
    fa.waterfall_save()
"""

# f"{self.pa[0].id}_to_{self.st[0-1].id}_waterfall_plot_{self.range[0].strftime("%Y-%m-%dT%H%M%S")}"
