from __future__ import annotations

from typing import Optional, Self

import matplotlib.pyplot as plt
import numpy as np
from dascore import Patch
from matplotlib.colors import LogNorm

from src.utils import check_file, mkdir


def single_fk(
    ax: plt.Axes,
    pa: Patch,
    *,
    dc_mask: int = 3,
    cmap: str = "jet",
    log_floor: float = 1e-4,
    title: str = "f-k Spectrum",
) -> tuple[plt.Axes, dict]:
    """
    Plot a single f-k spectrum on the given Axes and return peak info.

    Returns
    -------
    ax : plt.Axes
    info : dict  {"v_app", "peak_f", "peak_k", "power_spectrum", "f_axis", "k_axis"}
    """
    # Ensure (time, depth) orientation
    if pa.dims == ("depth", "time"):
        data = pa.data.T
    else:
        data = pa.data

    nt, nx = data.shape
    dt = float(pa.coords.step("time") / np.timedelta64(1, "s"))
    dx = float(pa.coords.step("depth"))

    # 2D FFT
    F_shifted = np.fft.fftshift(np.fft.fft2(data))
    power_spectrum = np.abs(F_shifted) ** 2

    f_axis = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))
    k_axis = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))

    # Mask DC
    ct, cx = nt // 2, nx // 2
    ps_no_dc = power_spectrum.copy()
    ps_no_dc[ct - dc_mask : ct + dc_mask + 1, cx - dc_mask : cx + dc_mask + 1] = 0

    # Peak
    max_idx = np.unravel_index(np.argmax(ps_no_dc), ps_no_dc.shape)
    peak_f = f_axis[max_idx[0]]
    peak_k = k_axis[max_idx[1]]
    v_ap = abs(peak_f / peak_k) if peak_k != 0 else np.inf

    # Plot
    ps_max = power_spectrum.max()
    im = ax.imshow(
        power_spectrum,
        extent=[k_axis[0], k_axis[-1], f_axis[0], f_axis[-1]],
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=LogNorm(vmin=ps_max * log_floor, vmax=ps_max),
    )

    ax.scatter(
        peak_k,
        peak_f,
        color="white",
        marker="x",
        s=150,
        linewidths=2,
        label=f"Peak Energy\n$V_{{ap}}$ = {v_ap:.1f} m/s",
    )
    ax.scatter(
        -peak_k,
        -peak_f,
        color="gray",
        marker="x",
        s=100,
        alpha=0.5,
        label="Conjugate Peak",
    )

    ax.get_figure().colorbar(im, ax=ax, label="Energy")
    ax.set_xlabel("Wavenumber $k$ (1/m)")
    ax.set_ylabel("Frequency $f$ (Hz)")
    ax.set_title(title)
    ax.axhline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.legend(loc="upper right")

    info = {
        "v_ap": v_ap,
        "peak_f": peak_f,
        "peak_k": peak_k,
        # "power_spectrum": power_spectrum,
        # "f_axis": f_axis,
        # "k_axis": k_axis,
    }
    return ax, info


class Folk:
    """
    ### Plot f-k Spectrum
    #### input
    - pa: data, Patch or list[Patch]

    #### method
    set_plot: plot f-k spectrum
    - title: figure title, list or string
    - figsize: single figure size, tuple
    - dc_mask: DC mask radius, int
    - cmap: default value "jet", string
    - log_floor: log scale floor ratio, float

    fk_plot: show plot

    fk_save: save png figure
    """

    def __init__(
        self,
        pa: Patch | list[Patch],
    ):
        self.pa = pa if isinstance(pa, list) else [pa]
        self.pa_len = len(self.pa)
        self.info: list[dict] = []

    def __str__(self):
        return self.info

    def set_plot(
        self,
        figsize: Optional[tuple[int, int]] = (10, 8),
        title: Optional[list[str]] | str = None,
        **kwargs,
    ) -> Self:
        if isinstance(title, str):
            fig_title = [title for _ in range(self.pa_len)]
        elif title:
            fig_title = title
        else:
            fig_title = ["f-k Spectrum" for _ in range(self.pa_len)]

        fs = (figsize[0] * self.pa_len, figsize[1])
        self.fig, axes = plt.subplots(1, self.pa_len, figsize=fs)
        ax_flat = np.atleast_1d(axes).flatten()

        self.info = []
        for ax, pa, ti in zip(ax_flat, self.pa, fig_title):
            _, info = single_fk(ax, pa, title=ti, **kwargs)
            self.info.append(info)

        self.fig.tight_layout()
        return self

    def fk_plot(self) -> Self:
        plt.show()
        return self

    def fk_save(
        self,
        folder: str = "image/",
        dpi: int = 200,
        filename: str = "FK_plot",
    ) -> Self:
        mkdir(folder)
        self.fig.savefig(check_file(f"{folder}/{filename}.png"), dpi=dpi, bbox_inches="tight")
        plt.close(self.fig)
        return self
