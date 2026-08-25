"""
PhaseNet-DAS interface for this project.

This module bridges the dascore-based .h5 files produced by :func:`src.door.save`
and the ``DAS_ML.phasenet_das`` API used in the Phasenet-DAS_TCOC notebook.

``DAS_ML`` is **not** imported at module load time; it is loaded lazily inside
:class:`Phase.__init__` so that the rest of the project remains importable even
if ``DAS_ML`` is not installed.

Example
-------
>>> from src.analysis import Phase
>>> runner = Phase(device='cpu')                         # loads model once
>>> picks = runner.run_from_h5('2026-07-08T154732.h5', ev_id='20260708')
>>> runner.save('20260708.csv')
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from obspy import UTCDateTime

from src.utils import check_file, mkdir


class Phase:
    """
    Wrapper around ``DAS_ML.phasenet_das`` for this project's ``.h5`` files.

    The constructor pre-loads the PhaseNet-DAS model (an expensive operation
    that should be done once).  After that, call :meth:`run_from_h5` or
    :meth:`run_patch` repeatedly without re-loading the model.

    Parameters
    ----------
    device:
        PyTorch device string - ``'cpu'`` or ``'cuda'``.
    """

    def __init__(self, device: str = "cpu") -> None:
        try:
            import DAS_ML as _DAS_ML
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("DAS_ML is required for Phase but is not installed. Please install it and make sure it is on the Python path.") from exc

        self._DAS_ML = _DAS_ML
        _DAS_ML.preload_model(device=device)

    def run_patch(
        self,
        pa,
        *,
        ev_id: str = "event",
        dt: Optional[float] = None,
    ) -> Phase:
        time_arr = pa.coords.get_array("time")  # datetime64[ns]
        t0_dt = time_arr[0].astype("datetime64[ms]").astype("O")  # Python datetime
        timestamp = UTCDateTime(t0_dt).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

        if dt is None:
            dt = float((time_arr[1] - time_arr[0]) / np.timedelta64(1, "s"))

        data = pa.data.astype(np.float32)

        self.picks = self._DAS_ML.phasenet_das(data, timestamp, ev_id, dt)
        return self

    def save(self, name: str = "picks.csv", *, folder: str = "picks") -> str:
        path = mkdir(folder)
        path = check_file(f"{path}/{name}")
        self.picks.to_csv(path, index=False)
        return str(self.picks)
