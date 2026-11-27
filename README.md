# 2026 INTERN
A project aim to process DAS (Distributed Acoustic Sensing, 分散式光纖聲學感測) data. Specially designed for MiDAS data.

## Features
- Read MiniSEED data, transform to HDF5
- Wave and waterfall plot
- GIF images of waterfall and wave
- Analyze data by PhaseNet DAS or Cross Correlation
- Separate up-going and down-going waves of borehole fiber (f-k filter), with waterfall, f-k and wiggle plots
- Web viewer: a whole day of MiDAS on the 3D fiber route, with timeline playback, waterfalls and event views

## Installation
### pip
```bash
pip install -e .
```

### uv
```bash
uv sync
```

## Usage
All parameters (data folder, event times, fiber segments, per-task settings) live in `config.toml`.

```bash
uv run python data.py            # run the task selected in data.py's __main__ (draw / mini_2_h5 / pn / fk / compareAB / updown)
uv run python plot_lags_hist.py  # histogram of lags_AB from image/data.csv
uv run python plot_vap_hist.py   # histogram of v_ap from image/data.csv
```

Library usage:
```python
from src.door import Search, load_mini, transfer
from src.plot import Fall, Finger, Water

se = Search(folder="data_100Hz")
fi = Finger(load_mini(se.find(1559))).process()
fi.set_time_range("2026-07-08T15:47:32").focus_save()

pa, stations = transfer(load_mini(se.multi_find(1578, 6)), start=20)
wa = Water(pa).cut(("2026-07-08T15:47:32", 10)).process()
Fall(wa.pa).set_plot(yname="distance").waterfall_save()
```

Read only a time window (records outside it are skipped, so all channels of one event fit in memory).
Use `pad` so that taper / filter edge effects fall outside the window, then cut back after processing:
```python
st = load_mini(se.multi_find(1578, 6), ("2026-07-08T15:47:32", 10), pad=30)
wa = Water(transfer(st, start=20)[0]).process().cut(("2026-07-08T15:47:32", 10))
```
When the same channels are cut at many times (e.g. `compareAB`), reading the whole day once is faster.

Up-going / down-going waves in a borehole (Hole A downward leg: channels 1208–1378).
Separate on the padded window, then cut, so edge effects of the f-k filter fall outside the figure:
```python
from src.analysis import separate_updown
from src.plot import Folk, Wiggle

t, r = "2026-07-08T15:47:32", ("2026-07-08T15:47:32", 2)
pa, _ = transfer(load_mini(se.multi_find(1208, 170), r, pad=30), start=0)
full = Water(pa).process().pa
up, down = separate_updown(full, v_min=500, v_max=8000)  # drop |v| < 500 m/s and |v| > 8000 m/s
raw, up, down = (Water(p).cut(r).pa for p in (full, up, down))

Fall([raw, up, down]).set_plot(title=["Raw", "Up-going", "Down-going"], yname="Depth (m)").waterfall_save()
Folk([raw, up, down]).set_plot(v_lines={"v_min": 500, "v_max": 8000}).fk_save()  # mask boundaries on the f-k plot
Wiggle([raw, up, down]).set_plot(color=["black", "red", "blue"], every=4).wiggle_save()
```
- Down-going = moving towards larger depth; the direction comes from the depth coordinate, so the upward leg of a U-shaped fiber also works.
- `v_taper` (default 10000 m/s) smooths the up / down boundary at k = 0, and `taper` smooths the edges of the `v_min` / `v_max` wedges, which reduces ringing. Without `v_min` / `v_max`, `up + down` equals the input.
- `data.py`'s `updown()` runs this for every event; settings (segment, `label` for titles, masks, wiggle window / gain) are under `[updown]` and `[updown.wiggle]` in `config.toml`.

Read directly from the MiDAS archive on the server (one folder per channel):
```python
from src.door import ARCHIVE_FILENAME
se = Search(folder="/ceph/midas/seiscomp/var/lib/archive", filename=ARCHIVE_FILENAME, year=2026, julday=189)
```
or set `folder` / `filename` / `year` / `julday` under `[search]` in `config.toml`.

Run tests:
```bash
uv run pytest
```

PhaseNet-DAS (`src.analysis.Phase`) additionally requires `DAS_ML`, which is not on PyPI.

### Web viewer
A static website that shows one day of MiDAS data on the fiber route in 3D (Hole A / B going down).
Stations (every 5 channels) are colored by amplitude at the cursor time; drag the timeline or play to watch waves
cross the fiber. The website never reads MiniSEED: `web_day` first converts the day into small files (0708: 17 GB of
MiniSEED → about 86 MB).

```bash
uv run python -c "import data; data.web_day()"   # MiniSEED day -> web/data/<day>/ (settings: [web] in config.toml)
uv run python -m http.server 8765 -d web         # then open http://localhost:8765/?day=20260708
```
- Filter: 1–20 Hz bandpass (same as the waterfall plots), then 1 s RMS for the whole day and 50 Hz waveforms (int16) for ±15 s around each event (`[web] events` + `events_csv`, e.g. `asset/20260708/ups.csv`).
- Times are shown in Taiwan time (UTC+8); amplitudes are in counts.
- Event views: envelope (SNR vs each station's noise) or signed amplitude, with one color clip for all stations or one per fiber section; a zoom waterfall with a choice of sections; first-arrival picks with a plane-wave fit (back-azimuth, apparent velocity) computed on the 100 Hz data. The direction is only as good as the channel positions (about ±100 m).
- The main shock (`main_shock` in `[web]`) is marked orange.
- The slow part (reading + filtering every channel) is cached in `web/cache/<day>/` (about 250 MB): the first run takes about 5 minutes, later runs about 10 s, and adding one event about 25 s.
- The fiber route and channel positions come from `asset/fiber_geometry.csv` (built from the KMZ; edit it to correct positions).
- `uv run python -c "import data; data.web_demo()"` writes a synthetic day (`?day=demo`) without reading any MiniSEED.

## Project Structure
```bash
.
├── src/
│   ├── analysis/     # Analyze data (cross correlation, PhaseNet DAS, up/down-going separation)
│   ├── door/         # Search, load and transform data
│   ├── plot/         # Generate figures (wave, waterfall, f-k, wiggle, histogram)
│   ├── utils/        # Other tools
│   └── web/          # Web viewer data export (geometry, MiniSEED -> site files, cache, arrival picks)
├── web/              # Web viewer (index.html, app.js, style.css); data in web/data/<day>/ (not in git)
├── config.toml       # Event times, fiber segments and task parameters
├── tests/
├── data.py           # Processing tasks
├── example.py        # Usage example
└── plot_*_hist.py    # Histogram scripts
```

## Major Logs
- 2026/07/28, Search and wave cut feature
- 2026/07/29, Waterfall plot feature
- 2026/08/04, Use [dascore](https://github.com/DASDAE/dascore) to replace [obspy](https://github.com/obspy/obspy), inspired by [ralin3233](https://github.com/ralin3233/das-processing-pipeline)
- 2026/08/09, GIF feature
- 2026/08/24, Collaborate with AI(Claude)
- 2026/08/25, PhaseNet DAS feature
- 2026/08/29, Cross correlation feature
- 2026/11/24, Up-going / down-going wave separation and wiggle plot
- 2026/11/27, Web viewer of a whole MiDAS day (3D fiber route, event views, arrival-time direction estimate)
