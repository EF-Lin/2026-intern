# CLAUDE.md

## Project map
DAS (MiDAS fiber) processing: MiniSEED → dascore `Patch` (dims `depth`, `time`) → filter / analyze → figures & CSV.
Read this instead of scanning the tree; open a file only when the task touches it.

| Path | What lives there |
|------|------------------|
| `config.toml` | All task parameters: `[search]` data folder, `[events]` times, `[segments]` fiber legs `{station, n}`, one table per task (`[fk]`, `[updown]`, `[compareAB]`, `[hist]` …). Loaded by `src.utils.load_config`. |
| `data.py` | Task entry point. Functions `draw`, `mini_2_h5`, `pn`, `fk`, `compareAB`, `updown`; pick one in `__main__`. |
| `plot_lags_hist.py`, `plot_vap_hist.py` | Histogram scripts reading `image/data.csv`. |
| `example.py` | Old usage example (hard-coded paths, not maintained). |
| `src/door/` | I/O. `search.py`: `Search` maps channel numbers → file paths (`find`, `multi_find`; `DEFAULT_FILENAME` / `ARCHIVE_FILENAME`). `load.py`: `load_mini` (optional time window + `pad`), `load_h5`, `load_csv`. `transform.py`: `transfer` (obspy `Stream` → `Patch`, depth = `start + Δch·spacing`), `save` (→ `.h5`). |
| `src/plot/` | Figures; each class wraps a `Patch` (or list of patches for side-by-side panels) and has `set_plot()` + `*_save()`. `waterfall.py`: `Water` (process / cut chain) and `Fall` (waterfall). `finger.py`: `Finger` (single-channel wave). `folks.py`: `Folk` (f-k). `wiggle.py`: `Wiggle`. `hist.py`: `plot_hist`. |
| `src/analysis/` | `cross.py`: `CC` cross-correlation (`calculate().pick()`). `updown.py`: `separate_updown` (f-k up/down-going split). `fold.py`: `fold_legs`, `leg_shift`, `mirror_depth` (stack the two legs of a U-shaped borehole fiber). `phasenet.py`: `Phase` (PhaseNet-DAS via external `DAS_ML`, lazy import). |
| `src/utils/` | `config.py` (`load_config`), `folder.py` (`mkdir`, `check_file` — never overwrites, appends ` (1)`), `convert.py` (time-range conversion), `figtools.py`, `saving.py` (`dicts2csv`), `timing.py` (`@timer`). |
| `tests/` | pytest, mirrors `src/` (`tests/load/test_search.py` skips without local data). Run `uv run pytest`. |

Gitignored, local-only:
- `data/<YYYYMMDD>/` MiniSEED (one file per channel); `data/*.h5`.
- `asset/` station metadata at the top level: `MiDAS metadata-20260210.csv` (channel ↔ fiber section / depth, see below), `MiDAS metadata - CHT.csv` (surface cable route CHT000–CHT010: lat/lon, cumulative distance, OTDR loss, splice dates), `MiDAS fiber cable.kmz` (cable route for Google Earth).
- `asset/<YYYYMMDD>/` per-event-day tables, e.g. `asset/20260708/`: `ups.csv` (f-k event list), `78.csv` / `7.8.xlsx` (event times), `peaks_20260708.txt`, `picks.csv`, `data.csv`.
- `paper/` reference PDFs only (MiDAS / borehole DAS papers; `edit-*.pdf` are annotated copies).
- `image/` (figures + `data.csv`), `picks/`, `*.pth`, `t.py`.
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
