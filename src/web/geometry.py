"""Fiber geometry for the web viewer: route points + channel anchors → per-station 3D positions."""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

M_PER_DEG_LAT = 110_574.0
M_PER_DEG_LON_EQ = 111_320.0


@dataclass
class Section:
    name: str
    leg: int
    stations: bool
    approx: bool
    xyz: np.ndarray  # (n, 3) east, north, depth (m), already offset for display
    channels: np.ndarray  # (n,) channel at each route point (interpolated by distance)
    labels: list[str] = field(default_factory=list)

    def at(self, ch: float) -> np.ndarray:
        """Position of a (fractional) channel along this section."""
        return np.array([np.interp(ch, self.channels, self.xyz[:, k]) for k in range(3)])


def read_rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def _local_xy(lat, lon, lat0, lon0):
    x = (np.asarray(lon) - lon0) * M_PER_DEG_LON_EQ * np.cos(np.radians(lat0))
    y = (np.asarray(lat) - lat0) * M_PER_DEG_LAT
    return x, y


def _offset(xyz: np.ndarray, leg: int, d: float) -> np.ndarray:
    """
    Shift a route sideways so the two legs of a folded fiber are drawn side by side.
    Horizontal routes move to the left of their direction of travel: an out-and-back pair travels in opposite
    directions, so the two legs land on opposite sides. Vertical routes (holes) move east by leg * d.
    """
    out = xyz.copy()
    n = len(xyz)
    for i in range(n):
        a, b = xyz[max(i - 1, 0)], xyz[min(i + 1, n - 1)]
        dx, dy = b[0] - a[0], b[1] - a[1]
        h = np.hypot(dx, dy)
        if h < 1e-6:
            out[i, 0] += leg * d
        else:
            out[i, 0] += -dy / h * d
            out[i, 1] += dx / h * d
    return out


def load_sections(path: str | Path, leg_offset: float = 12.0) -> tuple[list[Section], tuple[float, float]]:
    """Read fiber_geometry.csv. Returns sections (positions in local metres) and the (lat0, lon0) origin."""
    rows = read_rows(path)
    lats = np.array([float(r["lat"]) for r in rows])
    lons = np.array([float(r["lon"]) for r in rows])
    lat0, lon0 = float(lats.mean()), float(lons.mean())

    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["section"], []).append(r)

    sections = []
    for name, rs in groups.items():
        x, y = _local_xy([float(r["lat"]) for r in rs], [float(r["lon"]) for r in rs], lat0, lon0)
        z = np.array([float(r["depth"] or 0) for r in rs])
        xyz = np.column_stack([x, y, z])
        dist = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))])
        idx = [i for i, r in enumerate(rs) if r["channel"].strip()]
        if not idx or idx[0] != 0 or idx[-1] != len(rs) - 1:
            raise ValueError(f"section {name!r}: first and last rows need a channel")
        ch = np.interp(dist, dist[idx], [float(rs[i]["channel"]) for i in idx])
        leg = int(rs[0]["leg"])
        sections.append(
            Section(
                name=name,
                leg=leg,
                stations=rs[0]["stations"].strip() == "1",
                approx=rs[0]["approx"].strip() == "1",
                xyz=_offset(xyz, leg, leg_offset),
                channels=ch,
                labels=[r["label"] for r in rs],
            )
        )
    return sections, (lat0, lon0)


def build_stations(sections: list[Section], every: int, available: set[int] | None = None) -> list[dict]:
    """
    Group every `every` channels of each station-bearing section into one station.
    A station covers channels [ch0, ch1] and sits at their mean channel; has_data is False when none of
    its channels are in `available` (None = all available).
    """
    stations = []
    for s in sections:
        if not s.stations:
            continue
        c0, c1 = int(round(s.channels[0])), int(round(s.channels[-1]))
        step = 1 if c1 >= c0 else -1
        chans = list(range(c0, c1 + step, step))
        for k in range(0, len(chans), every):
            group = chans[k : k + every]
            pos = s.at(float(np.mean(group)))
            stations.append(
                dict(
                    id=len(stations),
                    section=s.name,
                    leg=s.leg,
                    ch0=min(group),
                    ch1=max(group),
                    channels=group,
                    x=round(float(pos[0]), 2),
                    y=round(float(pos[1]), 2),
                    depth=round(float(pos[2]), 2),
                    approx=s.approx,
                    has_data=available is None or any(c in available for c in group),
                )
            )
    return stations


def fiber_lines(sections: list[Section]) -> list[dict]:
    return [dict(section=s.name, approx=s.approx, points=np.round(s.xyz, 2).tolist()) for s in sections]
