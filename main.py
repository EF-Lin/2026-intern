from load import load_data
from loadCSV import load_csv
from finger import Finger
from search import Search
from waterfall import Water, Fall
from obspy import UTCDateTime


se = Search("data_100Hz")
st = load_data(se.multi_find(1234, 170))
li = load_csv()

tl = []

wa = Water(st=st)
st = wa.process()

for i in li:
    t = UTCDateTime(i["Time"])
    fa = Fall(st=st.copy(), y_factor=20, range=(t, 20))
    fa.cut()
    # w.waterfall_plot()
    fa.waterfall_save()
