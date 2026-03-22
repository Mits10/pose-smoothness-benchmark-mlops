from __future__ import annotations

import numpy as np
from scipy.signal import correlate


#def estimate_lag(reference: np.ndarray, target: np.ndarray) -> int:
#     """Estimate lag in frames between two 1D signals.

#     Positive lag means the target should be shifted right to align with the reference.
#     """
#     ref = np.asarray(reference, dtype=float).reshape(-1)
#     tar = np.asarray(target, dtype=float).reshape(-1)

#     if ref.size < 3 or tar.size < 3:
#         raise ValueError("Signals must each contain at least 3 samples.")

#     ref = ref - np.mean(ref)
#     tar = tar - np.mean(tar)

#     corr = correlate(ref, tar, mode="full")
#     lags = np.arange(-tar.size + 1, ref.size)
#     return int(lags[np.argmax(corr)])

def apply_lag(signal: np.ndarray, lag: int) -> np.ndarray:
    """Shift a signal by lag frames, padding with edge values."""
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]

    if lag == 0:
        return arr.copy()

    if lag > 0:
        pad = np.repeat(arr[[0]], lag, axis=0)
        shifted = np.vstack([pad, arr])[:-lag]
    else:
        lag = abs(lag)
        pad = np.repeat(arr[[-1]], lag, axis=0)
        shifted = np.vstack([arr, pad])[lag:]

    return shifted