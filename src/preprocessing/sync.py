from __future__ import annotations

import numpy as np
from scipy.signal import correlate

def detect_hand_raise(signal: list, thresh: float) -> int:
    reject = True
    while reject == True:
        for i in range(len(signal)):
            if signal[i] > thresh and signal[i+1] > thresh:
                start = i
                break
        for i in range(start, len(signal)-1):
            if signal[i] > 0 and signal[i+1] <= 0:
                peak = i
                break
        if (peak - start) < 40:
            reject = True
            break
        reject = False
    
    return start, peak


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

def estimate_lag(reference: np.ndarray, target: np.ndarray) -> int:
    pass

    # #Syncronization
    # #By velocity magnitude
    # vicon_sig = np.linalg.norm(vicon_vel_hand, axis=1)
    # xsens_sig = np.linalg.norm(xsens_vel_hand, axis=1)

    # n = min(len(vicon_sig), len(xsens_sig), 300)   # first 300 frames, example
    # lag = estimate_lag(vicon_sig[:n], xsens_sig[:n])

    # xsens_pos_aligned = apply_lag(xsens_pos_hand, lag)
    # xsens_vel_aligned = apply_lag(xsens_vel_hand, lag)
    # xsens_acc_aligned = apply_lag(xsens_acc_hand, lag)


    # #Syncronization
    # #By max value in the z direction: event: raise of hand
    # vicon_sig = np.linalg.norm(vicon_vel_hand, axis=1)
    # xsens_sig = np.linalg.norm(xsens_vel_hand, axis=1)

    # n = min(len(vicon_sig), len(xsens_sig), 300)

    # vicon_peak = np.argmax(vicon_sig[:n])
    # xsens_peak = np.argmax(xsens_sig[:n])

    # lag = vicon_peak - xsens_peak

    # xsens_pos_aligned = apply_lag(xsens_pos_hand, lag)
    # xsens_vel_aligned = apply_lag(xsens_vel_hand, lag)
    # xsens_acc_aligned = apply_lag(xsens_acc_hand, lag)
    #
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