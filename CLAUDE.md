# CLAUDE.md

## Project map
DAS (MiDAS fiber) processing: MiniSEED → dascore `Patch` (dims `depth`, `time`) → filter / analyze → figures & CSV.
Read this instead of scanning the tree; open a file only when the task touches it.

| Path | What lives there |
|------|------------------|
| `config.toml` | All task parameters: `[search]` data folder, `[events]` times, `[segments]` fiber legs `{station, n}`, one table per task (`[fk]`, `[updown]`, `[compareAB]`, `[hist]`, `[web]` …). Loaded by `src.utils.load_config`. |
| `data.py` | Task entry point. Functions `draw`, `mini_2_h5`, `pn`, `fk`, `compareAB`, `updown`, `web_day`, `web_demo`; pick one in `__main__`. `se()` builds the `Search` lazily, so tasks that read no MiniSEED run without `data/`. |
| `plot_lags_hist.py`, `plot_vap_hist.py` | Histogram scripts reading `image/data.csv`. |
| `example.py` | Old usage example (hard-coded paths, not maintained). |
| `src/door/` | I/O. `search.py`: `Search` maps channel numbers → file paths (`find`, `multi_find`; `DEFAULT_FILENAME` / `ARCHIVE_FILENAME`). `load.py`: `load_mini` (optional time window + `pad`), `load_h5`, `load_csv`. `transform.py`: `transfer` (obspy `Stream` → `Patch`, depth = `start + Δch·spacing`), `save` (→ `.h5`). |
| `src/plot/` | Figures; each class wraps a `Patch` (or list of patches for side-by-side panels) and has `set_plot()` + `*_save()`. `waterfall.py`: `Water` (process / cut chain) and `Fall` (waterfall). `finger.py`: `Finger` (single-channel wave). `folks.py`: `Folk` (f-k). `wiggle.py`: `Wiggle`. `hist.py`: `plot_hist`. |
| `src/analysis/` | `cross.py`: `CC` cross-correlation (`calculate().pick()`). `updown.py`: `separate_updown` (f-k up/down-going split). `fold.py`: `fold_legs`, `leg_shift`, `mirror_depth` (stack the two legs of a U-shaped borehole fiber). `phasenet.py`: `Phase` (PhaseNet-DAS via external `DAS_ML`, lazy import). |
| `src/utils/` | `config.py` (`load_config`), `folder.py` (`mkdir`, `check_file` — never overwrites, appends ` (1)`), `convert.py` (time-range conversion), `figtools.py`, `saving.py` (`dicts2csv`), `timing.py` (`@timer`). |
| `src/web/` | Web viewer data export (see "Web viewer" below). `geometry.py`: `load_sections`, `build_stations`, `fiber_lines` (fiber route from `asset/fiber_geometry.csv`). `real.py`: `build_day` (MiniSEED → station RMS + event windows, parallel). `cache.py`: `DayCache`. `arrivals.py`: `event_arrivals` (picks + plane-wave fit). `store.py`: `write_day` (→ `web/data/<day>/`). `demo.py`: `make_demo` (synthetic day). |
| `web/` | Static site: `index.html`, `app.js` (Three.js from CDN), `style.css`. Reads only `web/data/<day>/`. |
| `tests/` | pytest, mirrors `src/` (`tests/load/test_search.py` skips without local data; `tests/web/` covers the export). Run `uv run pytest`. |

Gitignored, local-only:
- `data/<YYYYMMDD>/` MiniSEED (one file per channel); `data/*.h5`.
- `asset/` station metadata at the top level: `MiDAS metadata-20260210.csv` (channel ↔ fiber section / depth, see below), `MiDAS metadata - CHT.csv` (surface cable route CHT000–CHT010: lat/lon, cumulative distance, OTDR loss, splice dates), `MiDAS fiber cable.kmz` (cable route for Google Earth).
- `asset/<YYYYMMDD>/` per-event-day tables, e.g. `asset/20260708/`: `ups.csv` (f-k event list), `78.csv` / `7.8.xlsx` (event times), `peaks_20260708.txt`, `picks.csv`, `data.csv`.
- `paper/` reference PDFs only (MiDAS / borehole DAS papers; `edit-*.pdf` are annotated copies).
- `image/` (figures + `data.csv`), `picks/`, `*.pth`, `t.py`.
- `asset/fiber_geometry.csv` (fiber route + channel anchors for the web viewer, built from the KMZ; hand-editable).
- `web/data/<day>/` (site data, ~86 MB for 0708) and `web/cache/<day>/` (export cache, ~250 MB).
Outputs go to `image/` (figures) and `picks/` (PhaseNet) relative to the working directory, so run scripts from the repo root.

## MiDAS data facts
Data files are gitignored; these facts save re-reading them.

- One MiniSEED file per DAS channel, covering a full day (STEIM2, 512-byte records).
- Sampling rate 100 Hz (delta 0.01 s, 8,640,000 samples per day).
- File name: `TW.{num:05d}..S.D.{year}.{julday}` (see `DEFAULT_FILENAME` / `ARCHIVE_FILENAME` in `src/door/search.py`).

### Borehole channels
Source: `asset/MiDAS metadata-20260210.csv`. The fiber goes down each hole and back up, so each hole appears twice.

| Hole | Depth | Downward leg (ch) | Bottom (ch) | Upward leg (ch) | Spacing |
|------|-------|-------------------|-------------|-----------------|---------|
| A    | 700 m | 1208–1378         | 1378        | 1378–1548       | ≈ 4.09 m (695 m / 170 ch) |
| B    | 500 m | 650–772           | 772         | 772–894         | ≈ 4.07 m (497 m / 122 ch) |

- `config.toml` segments `A = {1208, 170}` and `B = {650, 122}` are the downward legs.
- `transfer()` defaults to 4 m spacing; pass `spacing=` for exact depths.
- Up-/down-going wave work only makes sense on these borehole legs. On an upward leg the depth axis is reversed.

## Web viewer
Static site showing one day of MiDAS on the 3D fiber route (holes going down), colored by amplitude, with a timeline,
a day waterfall and an event zoom waterfall. Design decisions are settled; ask before changing them:
plain black/white UI with only the data colored (seismic colormap), labelled units (counts), times shown in UTC+8
(files stay UTC), 1–20 Hz bandpass like `Water.process`, 5 channels per station (max RMS).

Run:
- Export a day: `uv run python -c "import data; data.web_day()"` (or `web_day(folder=...)` for another data folder). Settings in `[web]`, `[web.arrivals]`.
- Serve: `uv run python -m http.server 8765 -d web` → `http://localhost:8765/?day=20260708` (`.claude/launch.json` "web"). `?day=demo` = synthetic data from `web_demo()`.

Export (`web_day`):
- Slow part (read + filter every channel file) is cached in `web/cache/<day>/`: float16 log10 station RMS + compressed float32 100 Hz event windows. First run ~5 min for 0708 (1524 channels, 6 workers); later runs ~10 s; a new event ~25 s (reads only its window ± 30 s pad). The cache drops itself when the band, station grouping, available channels or day change, and drops only the event windows when `event_half` changes. Always go through the cache.
- Events = `[web] events` + `events_csv` (0708: the 56 events of `asset/20260708/ups.csv`), ±`event_half` = 15 s. `main_shock` (0708: 15:47:28 UTC; 15:47:32 elsewhere in config.toml is the same quake) is drawn orange.
- First-arrival picks + plane-wave fit (`[web.arrivals]`, `src/web/arrivals.py`) run on the 100 Hz traces; the site stores 50 Hz int16. Changing them needs no MiniSEED read. Direction accuracy is limited by the ±100 m channel-position uncertainty, not by picking.

Site files (`web/data/<day>/`, little-endian, time-major (time, station)): `meta.json` (stations, fiber lines, events with `scale`, `vmax`, `section_clip`, `arrivals`, `label`), `overview.bin` uint8 (1440, n_st) per-minute max, `rms_HH.bin` uint8 (3600, n_st) log10 1 s RMS (0 = no data), `event_K.bin` int16 (n, n_st), value × `scale[st]` / 32767 = counts. `web/data/index.json` lists the days.

Page notes: the color panel switches event colors between envelope (SNR dB vs each station's pre-origin noise) and signed; "signed clip" (= "amplitude" in the event bar) is all stations or per section. The event bar picks span, rows, height and which sections to show. When editing `app.js`, check syntax with `node --check` on a `.mjs` copy, and reload with the cache bypassed (Ctrl+F5).
