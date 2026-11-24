# CLAUDE.md

## MiDAS data facts
Data files are gitignored; these facts save re-reading them.

- One MiniSEED file per DAS channel, covering a full day (STEIM2, 512-byte records).
- Sampling rate 100 Hz (delta 0.01 s, 8,640,000 samples per day).
- File name: `TW.{num:05d}..S.D.{year}.{julday}` (see `DEFAULT_FILENAME` / `ARCHIVE_FILENAME` in `src/door/search.py`).

### Borehole channels
Source: `paper/MiDAS metadata-20260210.csv`. The fiber goes down each hole and back up, so each hole appears twice.

| Hole | Depth | Downward leg (ch) | Bottom (ch) | Upward leg (ch) | Spacing |
|------|-------|-------------------|-------------|-----------------|---------|
| A    | 700 m | 1208–1378         | 1378        | 1378–1548       | ≈ 4.09 m (695 m / 170 ch) |
| B    | 500 m | 650–772           | 772         | 772–894         | ≈ 4.07 m (497 m / 122 ch) |

- `config.toml` segments `A = {1208, 170}` and `B = {650, 122}` are the downward legs.
- `transfer()` defaults to 4 m spacing; pass `spacing=` for exact depths.
- Up-/down-going wave work only makes sense on these borehole legs. On an upward leg the depth axis is reversed.
