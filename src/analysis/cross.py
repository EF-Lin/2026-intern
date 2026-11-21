from __future__ import annotations

from typing import Optional, Self

import numpy as np
from scipy import signal


def _as_1d(x: np.ndarray) -> np.ndarray:
    """
    Squeeze singleton dimensions and require a 1D signal.

    A single-channel patch such as shape (1, N) is accepted. True multi-channel
    input (e.g. depth x time with more than one channel) is rejected, because
    joining channels end-to-end has no physical meaning.
    """
    arr = np.squeeze(np.asarray(x))
    if arr.ndim != 1:
        raise ValueError(
            f"CC expects a 1D signal (or a single channel such as shape (1, N)), got shape {np.shape(x)}; "
            "for multi-channel data, call CC on each channel separately."
        )
    return arr


class CC:
    """
    Normalised cross-correlation between ``main`` and each array in ``compare``.

    Lag convention (scipy): a positive lag means ``main`` is delayed relative to ``compare``.
    """

    def __init__(
        self,
        main: np.ndarray,
        compare: list[np.ndarray],
    ):
        self.main = main
        self.compare = compare
        self.correlation = []
        self.lags = []

    def __call__(self, f: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        f = _as_1d(f)
        g = _as_1d(g)

        correlation = signal.correlate(f, g, method="auto")
        lags = signal.correlation_lags(len(f), len(g))
        return correlation, lags

    def calculate(self) -> Self:
        self.correlation = []
        self.lags = []
        main = _as_1d(self.main)
        for g in self.compare:
            g = _as_1d(g)
            cor, lag = self(main, g)
            cor = cor / np.sqrt(np.sum(main**2) * np.sum(g**2))
            self.correlation.append(cor)
            self.lags.append(lag)
        return self

    def pick(self, dt: Optional[float] = None) -> list[dict]:
        """
        Return the peak of each correlation.

        Each dict has ``correlation`` (peak value) and ``lags`` (lag in samples).
        If ``dt`` (sampling interval in seconds) is given, ``lag_seconds`` = ``lags * dt`` is added.
        """
        ret = []
        for i, j in zip(self.correlation, self.lags):
            cor_index = int(np.argmax(i))
            item = {"correlation": i[cor_index], "lags": j[cor_index]}
            if dt is not None:
                item["lag_seconds"] = float(j[cor_index]) * dt
            ret.append(item)
        return ret
