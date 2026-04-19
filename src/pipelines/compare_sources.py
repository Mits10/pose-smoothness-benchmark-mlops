from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.io.load_xsens import load_xsens_pose_sequence
from src.io.load_gopro_pose import load_gopro_hand_sequence
from src.io.load_vicon import load_vicon_hand_sequence
from src.io.schemas import PoseSequence

from src.preprocessing.filters import lowpass_filter
from src.preprocessing.resample import resample_timeseries

from src.preprocessing.sync import apply_lag, estimate_lag, detect_hand_raise
from src.features.smoothness import build_smoothness_features, velocity, spectral_arc_length, sliding_sparc

TARGET_FPS = 40.0
FILTER_CUTOFF_HZ = 6.0
REFERENCE_HAND = "right_hand"
orig_tstmp_Xsens = []
resamp_tstmp_Xsens = []

@dataclass
class ComparisonResult:
    sequence_id: str
    source: str
    reference_source: str
    hand: str
    fps_original: float
    fps_resampled: float
    estimated_lag_frames: int
    n_frames: int
    mean_abs_error_to_reference: float
    rmse_to_reference: float
    correlation_to_reference: float
    mean_offset_norm: float
    rms_jerk: float
    normalized_jerk: float

def _to_array(coords: list[list[float]]) -> np.ndarray:
    arr = np.asarray(coords, dtype = float)
    if arr.ndim != 2:
        raise ValueError("Expected coords with shape(frame,dims).")
    return arr

def _center_signal(arr: np.ndarray) -> np.ndarray:
    return arr - np.mean(arr, axis = 0, keepdims = True)

"""
#Prepare Hand signal Does -
#-to_array : Convert the hand keypoint into array
#resample_timeseries : Resample the fps to target fps
#lowpass_filter : Work with signal jitter
"""

def _prepare_hand_signal(seq: PoseSequence, hand: str) -> np.ndarray:
    if hand not in seq.joints:
        raise ValueError("Missing hand '{hand} in sequence {seq.sequence_id}")
    arr = _to_array(seq.joints[hand])
    #arr , orig_tstmp_Xsens, resamp_tstmp_Xsens= resample_timeseries(arr, orig_fps = 40, target_fps = TARGET_FPS)
    arr = lowpass_filter(arr, fps = 40, cutoff = FILTER_CUTOFF_HZ)
    #print(resamp_tstmp_Xsens)
    #print(orig_tstmp_Xsens)
    return arr

def _match_dimensions(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dims = min(reference.shape[1], target.shape[1])
    return reference[:, :dims], target[:, :dims]

"""
This function align the signals to reference signal
estimate_lag is the function to calculate the difference between the two signal
apply_lag function adjust the signal with the lag computed by estimate_lag
"""

def _align_to_reference(reference: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    reference, target = _match_dimensions(reference, target)

    ref_1d = np.linalg.norm(_center_signal(reference), axis=1)
    tar_1d = np.linalg.norm(_center_signal(target), axis=1)

    lag = estimate_lag(ref_1d, tar_1d)
    target_aligned = apply_lag(target, lag)

    n = min(len(reference), len(target_aligned))
    return reference[:n], target_aligned[:n], lag


def _compute_pair_metrics(reference: np.ndarray, target: np.ndarray) -> tuple[float, float, float, float]:
    diff = target-reference 
    diff = float(np.mean(np.abs(diff)))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(np.square(diff))))
    mean_offset_norm = float(np.mean(np.linlag.norm(diff, axis = 1)))

    ref_flat = reference.reshape(-1)
    tar_flat = target.reference(-1)
    if np.std(ref_flat) == 0 or np.std(tar_flat == 0):
            corr = 0.0
    else:
        corr = float(np.corrcoef(ref_flat, tar_flat) [0, 1])
    
    return mae, rmse, corr, mean_offset_norm

def compare_sequence_to_reference(
        reference_seq: PoseSequence,
        target_seq: PoseSequence,
        *,
        hand: str = REFERENCE_HAND,
) -> ComparisonResult:
    ref_signal = _prepare_hand_signal(reference_seq, hand)
    target_signal = _prepare_hand_signal(target_seq, hand)

    ref_aligned, target_aligned, lag = _align_to_reference(ref_signal, target_signal)
    mae, rmse, corr, mean_offset_norm = _compute_pair_metrics(ref_aligned, target_aligned)

    target_features = build_smoothness_features(target_aligned, fps=TARGET_FPS)

    return ComparisonResult(
        sequence_id=target_seq.sequence_id,
        source=target_seq.source,
        reference_source=reference_seq.source,
        hand=hand,
        fps_original=target_seq.fps,
        fps_resampled=TARGET_FPS,
        estimated_lag_frames=lag,
        n_frames=len(target_aligned),
        mean_abs_error_to_reference=mae,
        rmse_to_reference=rmse,
        correlation_to_reference=corr,
        mean_offset_norm=mean_offset_norm,
        rms_jerk=float(target_features["rms_jerk"]),
        normalized_jerk=float(target_features["normalized_jerk"]),
    )


def compare_all_sources(
        reference_seq: PoseSequence,
        target_seq: list[PoseSequence],
        *,
        hands: tuple[str, ...] = ("left_hand", "right_hand"),
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []

    for target in target_seq:
        for hand in hands:
            if hand not in reference_seq.joints or hand not in target.joints:
                continue
            result = compare_sequence_to_reference(reference_seq,target_seq)
            rows.append(result.__dict__)
    
    return pd.DataFrame(rows)

def main() -> None:
    #Command Line Argument
    parser = argparse.ArgumentParser(description = "Compare different sources pose smoothness.")
    parser.add_argument("--fps", required = True, help = "Inter taget FPS.")
    parser.add_argument("--cutoff", required = True, help = "Enter cutoff frequency.")

    #Reading Argument
    args = parser.parse_args()
    target_fps = int(args.fps)
    target_cutoff = int(args.cutoff)

    #Path of different source csv files
    #Update: Send either path or values
    #Need to be done
    xsens_csv_path = Path("data/processed/xsens/111_xsens_wide.csv")
    #realsense_csv_path = Path("data/raw/realsense/104_1_realsenes_openpose.csv")
    #gopro_csv_path = Path("data/raw/gopro/111_1_gopro_openpose.csv")
    #vicon_csv_path = Path("data/processed/vicon/111_vicon_wide.csv")
    #target sources
    targets: list[PoseSequence] = []
    """
    vicon_csv_path = Path("data/processed/108_vicon_wide.csv")
    realsense_csv_path = Path("data/processed/108_realsense_wide.csv")
    gopro_csv_path = Path("data/processed/gopro_wide.csv")
    """
    #"""""
    #Check if file exist
    if xsens_csv_path.exists():
        targets.append(
            load_xsens_pose_sequence(
                xsens_csv_path,
                sequence_id = "xsens_csv_trial",
                subject_id = "test",
                task = "hand_motion",
                trial = 1,
                kind = "position", 
            )
        )
    else:
        raise FileNotFoundError(f"{xsens_csv_path} does not exist")
    #"""""
    """""
    if gopro_csv_path.exists():
        targets.append(
            load_gopro_hand_sequence(
                gopro_csv_path,
                sequence_id = "gopro_trial",
                subject_id = "test",
                task = "hand_motion",
                trial = 1,
                fps = 30
            ) 
        )
    else:
        raise FileNotFoundError(f"{gopro_csv_path} does not exist")
    """""
    """""
    if realsense_csv_path.exists():
        targets.append(
            load_gopro_hand_sequence(
                realsense_csv_path,
                sequence_id = "realsense_trial",
                subject_id = "test",
                task = "hand_motion",
                trial = 1,
                fps = 30
            ) 
        )
    else:
        raise FileNotFoundError(f"{realsense_csv_path} does not exist")
    """""
    """""
    if vicon_csv_path.exists():
        targets.append(
            load_vicon_hand_sequence(
                vicon_csv_path,
                sequence_id = "vicon_trial",
                subject_id = "test",
                task = "hand_motion",
                trial = 1,
                fps = 100
            ) 
        )
    else:
        raise FileNotFoundError(f"{vicon_csv_path} does not exist")
    """""
    #df = pd.DataFrame([u.model_dump() for u in targets])
    #print(df.to_string())
    """""
    df = pd.read_excel("data/interim/108_2.xlsx")
    xsens_fps = 40
    vicon_fps = 100
    gopro_fps = 30
    realsense_fps = 30

    df["frame_vicon"] = df["frame_xsens"] * (100 / 40)
    df["frame_gopro"] = df["frame_xsens"] * (30 / 40)
    df["frame_realsense"] = df["frame_xsens"] * (30 / 40)


    df["frame_vicon"] = df["frame_vicon"].round().astype(int)
    df["frame_gopro"] = df["frame_gopro"].round().astype(int)
    df["frame_realsense"] = df["frame_xsens"].round().astype(int)

    vicon_array = df["frame_vicon"].to_numpy()
    gopro_array = df["frame_gopro"].to_numpy()
    realsense_array = df["frame_realsense"].to_numpy()

    j = 0
    """""
    for target in targets:
        #signal = np.asarray(target.joints["left_hand"][:20])
        #filtered_signal = lowpass_filter(signal, target_fps, target_cutoff)
        #ref_signal = _prepare_hand_signal(reference_seq, hand)
        print(target.joints["right_hand"][0])
        target_signal = _prepare_hand_signal(target, "right_hand")
        """""
        dt = 1 / target.fps
        target_velocity = velocity(target_signal, dt)
        target_speed = np.linalg.norm(target_velocity, axis=1)

        for i in range(0, len(arr[j]) - 1, 2):
            print(arr[i],arr[i+1])
            sparc_value = spectral_arc_length(target_speed[int(arr[i]):int(arr[i+1])], TARGET_FPS)
            print(sparc_value)
            result = sparc_value
            results.append(result)
        df_results["results"] = results
        df_results.to_csv("output_102_2.csv", index=False)
        j = j + 1
        """""

    #view of resamples timeseries
    
    #pos_z = list(map(lambda x: x[-1], target_signal))
    #print(pos_z[563])
    #vel_z = velocity(pos_z, dt)
    # Normalize
    #vel_norm = vel_z / np.max(np.abs(vel_z))
    #frame = detect_hand_raise(vel_norm, 0.5)
    #print(frame)
    
    dt = 1 / target.fps
    target_velocity = velocity(target_signal, dt)
    target_speed = np.linalg.norm(target_velocity, axis=1)
    
    
    #plt.plot(vel, label="Velocity")
    #plt.legend()
    #plt.show()
    
    
    df = pd.read_excel("data/interim/fps/111_2.xlsx")
    arr = df.to_numpy()
    frame_xsens = arr.flatten()
    #frame_vicon = np.round(frame_xsens * (100 / 40)).astype(int)
    #frame_gopro = np.round(frame_xsens * (30 / 40)).astype(int)
    #frame_realsense = np.round(frame_xsens * (30 / 40)).astype(int)

    results = []
    df_results = pd.DataFrame({
    "results": results
})
    for i in range(0, len(frame_xsens) - 1, 2):
        print(frame_xsens[i],frame_xsens[i+1])
        sparc_value = spectral_arc_length(target_speed[int(frame_xsens[i]):int(frame_xsens[i+1])], target.fps)
        print(sparc_value)
        #result = sparc_value
        #results.append(result)
    #df_results["results"] = results
    #df_results.to_csv("output_gopro_102_2.csv", index=False)
          
    #print(target_speed[:5])
    #sparc_value = spectral_arc_length(target_speed[26098:26186], TARGET_FPS)
    # python -m src.pipelines.compare_sources --fps 60 --cutoff 8
    #print(sparc_value)
    #indices, sparc_series = sliding_sparc(target_speed, 60, window_size=400, step=1)
    # Create a DataFrame
    #df = pd.DataFrame({
        #"frame": indices,
        #"sparc": sparc_series
    #})
    # Save to CSV
    #df.to_csv("sensitivity_sparc_104.csv", index=False)

    #print("Sliding SPARC saved to sliding_sparc.csv")
    #plt.figure()
    #plt.plot(target_speed, label="Speed")
    #plt.plot(indices, sparc_series, label="Sliding SPARC")
    #plt.legend()
    #plt.title("Smoothness over time")
    # Save plot as PDF
    #plt.savefig("sensitivity_sparc_plot_104.pdf", format='pdf')
    #plt.show()





if __name__ == "__main__":
    main()