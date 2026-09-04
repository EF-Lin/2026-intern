import pytest

from src.door.search import Search


@pytest.fixture
def sear() -> Search:
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
