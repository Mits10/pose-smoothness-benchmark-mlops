from __future__ import annotations

import numpy as np
from scipy.signal import welch


def _as_array(signal: np.ndarray) -> np.ndarray:
    arr = np.asarray(signal, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("Signal must be 1D or 2D with shape (frames, dims).")
    if arr.shape[0] < 4:
        raise ValueError("Signal must contain at least 4 frames.")
    return arr


def velocity(signal: np.ndarray, dt: float) -> np.ndarray:
    arr = _as_array(signal)
    return np.gradient(arr, dt, axis=0)


def acceleration(signal: np.ndarray, dt: float) -> np.ndarray:
    vel = velocity(signal, dt)
    return np.gradient(vel, dt, axis=0)


def jerk(signal: np.ndarray, dt: float) -> np.ndarray:
    acc = acceleration(signal, dt)
    return np.gradient(acc, dt, axis=0)


def mean_jerk_magnitude(signal: np.ndarray, dt: float) -> float:
    j = jerk(signal, dt)
    mag = np.linalg.norm(j, axis=1)
    return float(np.mean(mag))


def rms_jerk(signal: np.ndarray, dt: float) -> float:
    j = jerk(signal, dt)
    mag = np.linalg.norm(j, axis=1)
    return float(np.sqrt(np.mean(np.square(mag))))



def normalized_jerk(signal: np.ndarray, dt: float) -> float:
    arr = _as_array(signal)
    duration = arr.shape[0] * dt
    displacement = np.linalg.norm(arr[-1] - arr[0])
    if displacement == 0.0:
        return 0.0
    j = jerk(arr, dt)
    squared = np.sum(np.square(np.linalg.norm(j, axis=1))) * dt
    return float((duration**5 / displacement**2) * squared)


def spectral_arc_length(signal_1d: np.ndarray, fs: float) -> float:
    sig = np.asarray(signal_1d, dtype=float).reshape(-1)
    if sig.size < 8:
        raise ValueError("Signal too short for spectral arc length.")

    freqs, psd = welch(sig, fs=fs, nperseg=min(256, sig.size))
    if np.max(psd) == 0:
        return 0.0

    psd_norm = psd / np.max(psd)
    freqs_norm = freqs / np.max(freqs) if np.max(freqs) > 0 else freqs

    df = np.diff(freqs_norm)
    dp = np.diff(psd_norm)
    arc = np.sum(np.sqrt(df**2 + dp**2))
    return float(-arc)

def sliding_sparc(speed, fs, window_size=30, step=1):
    values = []
    indices = []

    for i in range(0, len(speed) - window_size + 1, step):
        segment = speed[i:i+window_size]
        val = spectral_arc_length(segment, fs)

        values.append(val)
        indices.append(i + window_size // 2)  # center of window

    return np.array(indices), np.array(values)


def build_smoothness_features(signal: np.ndarray, fps: float) -> dict[str, float]:
    if fps <= 0:
        raise ValueError("fps must be positive.")

    dt = 1.0 / fps
    arr = _as_array(signal)
    features = {
        "mean_jerk_magnitude": mean_jerk_magnitude(arr, dt),
        "rms_jerk": rms_jerk(arr, dt),
        "normalized_jerk": normalized_jerk(arr, dt),
    }

    for dim_idx in range(arr.shape[1]):
        features[f"sparc_dim_{dim_idx}"] = spectral_arc_length(arr[:, dim_idx], fps)

    return features