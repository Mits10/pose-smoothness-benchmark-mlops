from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt


def lowpass_filter(signal: np.ndarray, fps: float, cutoff: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth low-pass filter to 1D or 2D motion signals."""
    if fps <= 0 or cutoff <= 0:
        raise ValueError("fps and cutoff must be positive.")

    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]

    nyquist = fps / 2.0
    b, a = butter(order, cutoff / nyquist, btype="low")

    padlen = 3 * (max(len(a), len(b)) - 1)
    if arr.shape[0] <= padlen:
        raise ValueError(
            f"Signal too short for filtfilt: need more than {padlen} samples, got {arr.shape[0]}."
        )

    return filtfilt(b, a, arr, axis=0)

