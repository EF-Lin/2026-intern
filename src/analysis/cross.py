from __future__ import annotations

from typing import Self

import numpy as np
from scipy import signal


class CC:
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
        correlation = signal.correlate(f, g, method="auto")
        lags = signal.correlation_lags(len(f), len(g))
        return correlation, lags

    def calculate(self) -> Self:
        for g in self.compare:
            cor, lag = self(self.main, g)
            cor = cor / np.sqrt(np.sum(self.main**2) * np.sum(g**2))
            self.correlation.append(cor)
            self.lags.append(lag)
        return self

    def pick(self) -> list[dict]:
        ret = []
        for i, j in zip(self.correlation, self.lags):
            cor_index = int(np.argmax(i))
            ret.append({"correlation": i[cor_index], "lags": j[cor_index]})
        return ret
