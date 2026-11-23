# 2026 INTERN
A project aim to process DAS (Distributed Acoustic Sensing, 分散式光纖聲學感測) data. Specially designed for MiDAS data.

## Features
- Read MiniSEED data, transform to HDF5
- Wave and waterfall plot
- GIF images of waterfall and wave
- Analyze data by PhaseNet DAS or Cross Correlation

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
uv run python data.py            # run the task selected in data.py's __main__ (draw / mini_2_h5 / pn / fk / compareAB)
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

## Project Structure
```bash
.
├── src/
│   ├── analysis/     # Analyze data (cross correlation, PhaseNet DAS)
│   ├── door/         # Search, load and transform data
│   ├── plot/         # Generate figures (wave, waterfall, f-k, histogram)
│   └── utils/        # Other tools
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
