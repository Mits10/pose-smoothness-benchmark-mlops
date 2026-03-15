from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def resample_timeseries(signal: np.ndarray, orig_fps: float, target_fps: float) -> np.ndarray:
    """Resample a 1D or 2D signal to a target frame rate using linear interpolation."""
    if orig_fps <= 0 or target_fps <= 0:
        raise ValueError("orig_fps and target_fps must be positive.")

    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("Signal must be 1D or 2D with shape (frames, dims).")
    if arr.shape[0] < 2:
        raise ValueError("Signal must contain at least 2 frames.")

    duration = (arr.shape[0] - 1) / orig_fps
    orig_t = np.linspace(0.0, duration, arr.shape[0])
    new_n = int(round(duration * target_fps)) + 1
    new_t = np.linspace(0.0, duration, new_n)

    interpolator = interp1d(orig_t, arr, axis=0, kind="linear", fill_value="extrapolate")
    return interpolator(new_t)