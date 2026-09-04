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
