"""
By Gemini
"""

import numpy as np
import pytest

from src.analysis import CC


@pytest.fixture()
def cross() -> CC:
    f = np.array([1, 2, 3, 4, 5])
    g = np.array([1, 2, 3, 4, 5])
    return CC(f, [g])


@pytest.mark.parametrize(
    "f, g, expected_lag, expected_max_corr",
    [
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 0, 14),
        (np.array([1, 2, 3, 4, 5]), np.array([1, 2, 3, 4, 5]), 0, 55),
        (np.array([0, 0, 10, 0]), np.array([10, 0, 0, 0]), 2, 100),
        (np.array([10, 0, 0, 0]), np.array([0, 0, 10, 0]), -2, 100),
    ],
)
def test_call(
    cross: CC,
    f: np.ndarray,
    g: np.ndarray,
    expected_lag: int,
    expected_max_corr: float,
) -> None:
    cor, lags = cross(f, g)

    assert len(cor) == len(f) + len(g) - 1
    assert len(lags) == len(cor)
    max_idx = int(np.argmax(cor))
    assert lags[max_idx] == expected_lag
    assert cor[max_idx] == expected_max_corr


@pytest.mark.parametrize(
    "num_compare",
    [1, 2, 4],
)
def test_calculate(num_compare: int) -> None:
    f = np.array([1, 2, 3])
    compare = [np.array([1, 2, 3]) for _ in range(num_compare)]
    cc = CC(f, compare)

    assert len(cc.correlation) == 0
    assert len(cc.lags) == 0

    res = cc.calculate()
    assert res is cc
    assert len(cc.correlation) == num_compare
    assert len(cc.lags) == num_compare

    for cor in cc.correlation:
        assert np.max(cor) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "main, compare, expected_picks",
    [
        (
            np.array([1, 2, 3, 4, 5]),
            [np.array([1, 2, 3, 4, 5])],
            [{"correlation": 1.0, "lags": 0}],
        ),
        (
            np.array([1, 2, 3]),
            [np.array([2, 4, 6])],
            [{"correlation": 1.0, "lags": 0}],
        ),
        (
            np.array([0, 0, 0, 10, 0, 0]),
            [np.array([0, 10, 0, 0, 0, 0])],
            [{"correlation": 1.0, "lags": 2}],
        ),
        (
            np.array([0, 5, 0]),
            [np.array([0, 0, 5])],
            [{"correlation": 1.0, "lags": -1}],
        ),
        (
            np.array([1.0, 2.0, 3.0]),
            [np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 0.0])],
            [
                {"correlation": 1.0, "lags": 0},
                {"correlation": 13.0 / np.sqrt(14.0 * 13.0), "lags": 1},
            ],
        ),
    ],
)
def test_pick(
    main: np.ndarray,
    compare: list[np.ndarray],
    expected_picks: list[dict],
) -> None:
    cc = CC(main, compare)
    picks = cc.calculate().pick()

    assert len(picks) == len(expected_picks)
    for pick, expected in zip(picks, expected_picks):
        assert pick["correlation"] == pytest.approx(expected["correlation"])
        assert pick["lags"] == expected["lags"]


def test_calculate_resets_between_calls() -> None:
    cc = CC(np.array([1.0, 2.0, 3.0]), [np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])])
    cc.calculate()
    cc.calculate()
    assert len(cc.correlation) == 2
    assert len(cc.lags) == 2
    assert len(cc.pick()) == 2


def test_single_channel_2d_matches_1d() -> None:
    f = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0])
    g = np.array([0.0, 10.0, 0.0, 0.0, 0.0, 0.0])
    p1 = CC(f, [g]).calculate().pick()
    p2 = CC(f[np.newaxis, :], [g[np.newaxis, :]]).calculate().pick()
    assert p1[0]["lags"] == p2[0]["lags"] == 2
    assert p1[0]["correlation"] == pytest.approx(p2[0]["correlation"])


@pytest.mark.parametrize(
    "main, compare",
    [
        (np.ones((3, 10)), [np.ones(10)]),
        (np.ones(10), [np.ones((2, 10))]),
    ],
)
def test_multichannel_rejected(main: np.ndarray, compare: list[np.ndarray]) -> None:
    with pytest.raises(ValueError):
        CC(main, compare).calculate()


def test_pick_with_dt() -> None:
    cc = CC(np.array([0, 0, 0, 10, 0, 0]), [np.array([0, 10, 0, 0, 0, 0])]).calculate()
    pick = cc.pick(dt=0.01)[0]
    assert pick["lags"] == 2
    assert pick["lag_seconds"] == pytest.approx(0.02)
    assert "lag_seconds" not in cc.pick()[0]
