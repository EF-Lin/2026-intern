import json

import numpy as np
import pytest

from src.web import build_stations, fiber_lines, load_sections, make_demo, write_day
from src.web.store import SECONDS_PER_DAY, quantise

GEOMETRY = """# comment lines are skipped
section,leg,stations,approx,channel,lat,lon,depth,label
Surface B (forth),1,1,0,0,24.0000,121.0000,0,a
Surface B (forth),1,1,0,,24.0010,121.0000,0,mid
Surface B (forth),1,1,0,20,24.0020,121.0000,0,b
Hole (down),1,1,0,20,24.0020,121.0000,0,top
Hole (down),1,1,0,30,24.0020,121.0000,40,bottom
Spare,1,0,1,30,24.0020,121.0000,0,x
Spare,1,0,1,32,24.0021,121.0000,0,y
"""


@pytest.fixture
def geometry(tmp_path):
    p = tmp_path / "geometry.csv"
    p.write_text(GEOMETRY, encoding="utf-8")
    return p


def test_sections_interpolate_channels_by_distance(geometry):
    sections, (lat0, lon0) = load_sections(geometry, leg_offset=0)
    surf = sections[0]
    assert surf.channels.tolist() == pytest.approx([0, 10, 20])
    # channel 10 is halfway along the ~221 m route
    assert surf.at(10)[1] - surf.at(0)[1] == pytest.approx(110.574, rel=1e-3)
    hole = sections[1]
    assert hole.at(25)[2] == pytest.approx(20)


def test_leg_offset_moves_hole_east(geometry):
    sections, _ = load_sections(geometry, leg_offset=12)
    hole0 = load_sections(geometry, leg_offset=0)[0][1]
    assert sections[1].xyz[0, 0] - hole0.xyz[0, 0] == pytest.approx(12)


def test_build_stations_groups_channels(geometry):
    sections, _ = load_sections(geometry)
    st = build_stations(sections, every=5, available=set(range(0, 8)))
    surf = [s for s in st if s["section"] == "Surface B (forth)"]
    assert [(s["ch0"], s["ch1"]) for s in surf] == [(0, 4), (5, 9), (10, 14), (15, 19), (20, 20)]
    assert [s["has_data"] for s in surf] == [True, True, False, False, False]
    assert not any(s["section"] == "Spare" for s in st)
    assert [s["id"] for s in st] == list(range(len(st)))
    assert len(fiber_lines(sections)) == 3


def test_quantise_keeps_nan_as_zero():
    q = quantise(np.array([np.nan, 0.0, 0.5, 1.0, 2.0]), 0.0, 1.0)
    assert q.tolist() == [0, 1, 128, 255, 255]


def test_write_day_layout(tmp_path, geometry):
    sections, _ = load_sections(geometry)
    stations = build_stations(sections, every=5)
    n = len(stations)
    rng = np.random.default_rng(0)
    log_rms = rng.normal(1, 0.1, (SECONDS_PER_DAY, n))
    log_rms[:, 0] = np.nan
    data = rng.normal(0, 10, (500, n))
    ev = [dict(time="2026-07-08T01:00:05", start="2026-07-08T01:00:00.000", fs=50, data=data)]
    folder = write_day(tmp_path, "d", "2026-07-08T00:00:00", stations, fiber_lines(sections), log_rms, ev, {"units": "counts"})

    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    assert meta["units"] == "counts" and len(meta["stations"]) == n
    assert meta["stations"][0]["median"] is None
    hour = np.fromfile(folder / "rms_00.bin", dtype=np.uint8).reshape(3600, n)
    assert (hour[:, 0] == 0).all() and (hour[:, 1:] > 0).all()
    assert np.fromfile(folder / "overview.bin", dtype=np.uint8).size == 1440 * n
    wav = np.fromfile(folder / "event_0.bin", dtype=np.int16).reshape(500, n)
    scale = np.array(meta["events"][0]["scale"])
    np.testing.assert_allclose(wav * scale / 32767, data, atol=scale.max() / 32767)
    assert json.loads((tmp_path / "index.json").read_text()) == ["d"]


def test_demo_shapes(geometry):
    sections, _ = load_sections(geometry)
    stations = build_stations(sections, every=5, available=set(range(0, 25)))
    log_rms, events = make_demo(stations, "2026-07-08T00:00:00", [{"time": "2026-07-08T05:00:00", "mag": 2}], fs=50, half=20)
    assert log_rms.shape == (SECONDS_PER_DAY, len(stations))
    no_data = [i for i, s in enumerate(stations) if not s["has_data"]]
    assert np.isnan(log_rms[:, no_data]).all()
    assert events[0]["data"].shape == (2000, len(stations))
    # the event is louder than the noise just before its window
    w = int(5 * 3600)
    assert np.nanmean(log_rms[w : w + 15]) > np.nanmean(log_rms[w - 60 : w - 30])
