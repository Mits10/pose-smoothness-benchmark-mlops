from src.preprocessing.filters import lowpass_filter
from src.preprocessing.resample import resample_timeseries
from src.preprocessing.sync import apply_lag, estimate_lag
from src.features.smoothness import build_smoothness_features, velocity, spectral_arc_length, sliding_sparc
import numpy as np
import matplotlib.pyplot as plt

def main() -> None:


    # --------------------------
    # Parameters
    # --------------------------
    fs = 40                  # FPS = 40 Hz
    dt = 1 / fs
    T = 5                    # total duration in seconds
    N = int(T * fs)          # number of samples
    t = np.linspace(0, T, N)

    # --------------------------
    # Generate synthetic 3D motion
    # Smooth sinusoidal motion
    # --------------------------
    x = 0.5 * np.sin(2 * np.pi * 0.5 * t)           # slow oscillation in x
    y = 0.3 * np.sin(2 * np.pi * 0.7 * t + np.pi/4) # slightly different freq in y
    z = 0.2 * np.cos(2 * np.pi * 0.3 * t)           # slowest oscillation in z

    # Combine into 3D array
    position = np.stack([x, y, z], axis=1)  # shape (N, 3)
    resample, t, new_t = resample_timeseries(position, 40, 60)
    vel = velocity(resample, dt= 1/60)
    speed = np.linalg.norm(vel, axis=1)
    sparc_value = spectral_arc_length(speed,60)
    indices, sparc_series = sliding_sparc(speed, 60, window_size=30, step=1)

    plt.figure()
    plt.plot(speed, label="Speed")
    plt.plot(indices, sparc_series, label="Sliding SPARC")
    plt.legend()
    plt.title("Smoothness over time")
    plt.show()

    # --------------------------
    # Optional: visualize
    # --------------------------
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(position[:,0], position[:,1], position[:,2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Synthetic 3D Position Signal at 40 FPS')
    plt.show()

if __name__ == "__main__":
    main()