"""
PhaseNet-DAS interface for this project.

This module bridges the dascore-based .h5 files produced by :func:`src.door.save`
and the ``DAS_ML.phasenet_das`` API used in the Phasenet-DAS_TCOC notebook.

``DAS_ML`` and ``DASutils`` are **not** imported at module load time; they are
loaded lazily inside :class:`PhasenetDAS.__init__` so that the rest of the
project remains importable even if those packages are not installed.

Example
-------
>>> from src.phasenet import PhasenetDAS
>>> runner = PhasenetDAS(device='cpu')           # loads model once
>>> picks = runner.run_from_h5('2026-07-08T154732.h5', ev_id='20260708')
>>> picks.to_csv('picks.csv', index=False)
"""

from __future__ import annotations

from typing import Optional

import dascore as dc
import numpy as np
from obspy import UTCDateTime


class Phase:
    """
    Wrapper around ``DAS_ML.phasenet_das`` for this project's ``.h5`` files.

    The constructor pre-loads the PhaseNet-DAS model (an expensive operation
    that should be done once).  After that, call :meth:`run_from_h5` or
    :meth:`run_from_patch` repeatedly without re-loading the model.

    Parameters
    ----------
    device:
        PyTorch device string - ``'cpu'`` or ``'cuda'``.
    """

    def __init__(self, device: str = "cpu") -> None:
        try:
            import DAS_ML as _DAS_ML
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("DAS_ML is required for PhasenetDAS but is not installed. Please install it and make sure it is on the Python path.") from exc

        self._DAS_ML = _DAS_ML
        _DAS_ML.preload_model(device=device)

    def run_from_h5(
        self,
        h5_path: str,
        *,
        ev_id: str = "event",
        dt: Optional[float] = None,
    ):
        """
        Run PhaseNet-DAS on a ``.h5`` file saved by :func:`src.door.save`.

        Parameters
        ----------
        h5_path:
            Path to the ``.h5`` file written by ``dascore.write(..., 'dasdae')``.
        ev_id:
            Event identifier label that will be embedded in the output picks.
        dt:
            Sampling interval in **seconds**.  When ``None`` (the default) the
            value is derived automatically from the ``time`` coordinate of the
            Patch as ``time[1] - time[0]``.

        Returns
        -------
        pandas.DataFrame
            Picks with columns ``station_id``, ``phase_index``, ``phase_time``,
            ``phase_score``, ``phase_type`` (same schema as the notebook output).
        """
        pa = dc.read(h5_path, "dasdae")
        return self.run_from_patch(pa, ev_id=ev_id, dt=dt)

    def run_from_patch(
        self,
        pa,
        *,
        ev_id: str = "event",
        dt: Optional[float] = None,
    ):
        """
        Run PhaseNet-DAS on an already-loaded dascore :class:`~dascore.Patch`.

        The Patch is expected to have ``dims = ("depth", "time")`` as produced
        by :func:`src.door.transform.transfer`.  The data must already be
        pre-processed (detrend / taper / filter) - use
        :class:`src.plot.waterfall.Water` for that before calling this method.

        Parameters
        ----------
        pa:
            A dascore ``Patch`` with shape ``(n_channels, n_time)``.
        ev_id:
            Event identifier label embedded in the picks output.
        dt:
            Sampling interval in **seconds**.
            Defaults to ``time[1] - time[0]`` from the Patch coordinates.

        Returns
        -------
        pandas.DataFrame
            Picks DataFrame (same schema as :meth:`run_from_h5`).
        """
        # ---- time metadata ------------------------------------------------
        time_arr = pa.coords.get_array("time")  # datetime64[ns]

        # Timestamp string that PhaseNet-DAS expects
        t0_dt = time_arr[0].astype("datetime64[ms]").astype("O")  # Python datetime
        timestamp = UTCDateTime(t0_dt).strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")

        # dt: use parameter if given, otherwise derive from the time coordinate
        if dt is None:
            dt = float((time_arr[1] - time_arr[0]) / np.timedelta64(1, "s"))

        # ---- data ---------------------------------------------------------
        # pa.data shape: (n_depth, n_time) == (n_channels, n_time)
        # DAS_ML.phasenet_das expects exactly this layout.
        data = pa.data.astype(np.float32)

        # ---- run model ----------------------------------------------------
        picks = self._DAS_ML.phasenet_das(data, timestamp, ev_id, dt)
        return picks
