import os

import pytest

from src.door.search import ARCHIVE_FILENAME, Search


@pytest.fixture
def sear() -> Search:
    if not os.path.isdir("data_100Hz"):
        pytest.skip("local data folder data_100Hz not present")
    se = Search(folder="data_100Hz/")
    return se


@pytest.mark.parametrize(
    "n, expected",
    [
        (55, "data_100Hz\\TW.00055..S.D.2026.189"),
        (123, "data_100Hz\\TW.00123..S.D.2026.189"),
        (1567, "data_100Hz\\TW.01567..S.D.2026.189"),
    ],
)
def test_find(sear: Search, n: int, expected: str) -> None:
    assert sear.find(n) == expected


@pytest.mark.parametrize(
    "n, r, expected",
    [
        (55, 1, ["data_100Hz\\TW.00055..S.D.2026.189", "data_100Hz\\TW.00056..S.D.2026.189"]),
        (
            123,
            4,
            [
                "data_100Hz\\TW.00123..S.D.2026.189",
                "data_100Hz\\TW.00124..S.D.2026.189",
                "data_100Hz\\TW.00125..S.D.2026.189",
                "data_100Hz\\TW.00126..S.D.2026.189",
                "data_100Hz\\TW.00127..S.D.2026.189",
            ],
        ),
        (1567, 2, ["data_100Hz\\TW.01567..S.D.2026.189", "data_100Hz\\TW.01568..S.D.2026.189", "data_100Hz\\TW.01569..S.D.2026.189"]),
    ],
)
def test_multi_find(sear: Search, n: int, r: int, expected: list[str]) -> None:
    assert sear.multi_find(n, r) == expected


@pytest.fixture
def fake_folder(tmp_path):
    for name in ["TW.00010..S.D.2026.189", "TW.00010..S.D.2026.209", "TW.00010..S.D.2025.009", "TW.00011..S.D.2025.009"]:
        (tmp_path / name).touch()
    return tmp_path


def test_default_filename(fake_folder) -> None:
    se = Search(folder=str(fake_folder))
    assert se.find(10).endswith("TW.00010..S.D.2026.189")


def test_year_julday(fake_folder) -> None:
    se = Search(folder=str(fake_folder), year=2025, julday=9)
    assert [p[-22:] for p in se.multi_find(10, 1)] == ["TW.00010..S.D.2025.009", "TW.00011..S.D.2025.009"]


@pytest.mark.parametrize(
    "date",
    [
        __import__("datetime").date(2026, 7, 28),
        __import__("obspy").UTCDateTime(2026, 7, 28, 12, 0, 0),
    ],
)
def test_date(fake_folder, date) -> None:
    se = Search(folder=str(fake_folder), date=date)
    assert se.find(10).endswith("TW.00010..S.D.2026.209")


def test_fixed_filename_still_works(fake_folder) -> None:
    se = Search(folder=str(fake_folder), filename="TW.{num}..S.D.2026.209", year=1999)
    assert se.find(10).endswith("TW.00010..S.D.2026.209")


def test_missing_with_date(fake_folder) -> None:
    se = Search(folder=str(fake_folder), year=2024, julday=1)
    with pytest.raises(FileNotFoundError):
        se.find(10)
    with pytest.raises(FileNotFoundError):
        se.multi_find(10, 1)


def test_missing_folder(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        Search(folder=str(tmp_path / "nope"))


def test_archive_subfolders(tmp_path) -> None:
    for num in ("00010", "00011"):
        d = tmp_path / "2026" / "TW" / num / "S.D"
        d.mkdir(parents=True)
        (d / f"TW.{num}..S.D.2026.189").touch()
    se = Search(folder=str(tmp_path), filename=ARCHIVE_FILENAME, year=2026, julday=189)
    assert se.find(10) == str(tmp_path / "2026" / "TW" / "00010" / "S.D" / "TW.00010..S.D.2026.189")
    assert len(se.multi_find(10, 1)) == 2
    with pytest.raises(FileNotFoundError, match="00012"):
        se.multi_find(10, 2)
