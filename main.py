from src.load.loadTR import load_data
from src.load.loadCSV import load_csv
from src.finger import Finger
from src.search import Search
from src.waterfall import Water, Fall
from obspy import UTCDateTime



se = Search("data_100Hz")
li = load_csv()

st = load_data(se.find(1334))
fi = Finger(st)
fi.process()

st2 = load_data(se.multi_find(1234, 150))
wa = Water(st=st2)
st2 = wa.process()

for i in li:
    t = UTCDateTime(i["Time"])
    fi.focus_save(t)

    fa = Fall(st=st2.copy(), y_factor=20, range=(t, 20))
    fa.cut()
    fa.waterfall_save()
