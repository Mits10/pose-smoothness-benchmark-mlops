from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
import argparse

import numpy as np
import pandas as pd

from src.io.load_xsens import load_xsens_pose_sequence
from src.io.load_gopro_pose import load_gopro_hand_sequence
from src.io.load_vicon import load_vicon_hand_sequence
from src.io.schemas import PoseSequence

from src.preprocessing.filters import lowpass_filter
from src.preprocessing.resample import resample_timeseries
from src.preprocessing.sync import apply_lag, estimate_lag

TARGET_FPS = 60.0
FILTER_CUTOFF_HZ = 6.0
REFERENCE_HAND = "right_hand"

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

def main() -> None:
    #Command Line Argument
    parser = argparse.ArgumentParser(description="Compare different sources pose smoothness.")
    parser.add_argument("--fps", required= True, help=" Inter taget FPS.")
    parser.add_argument("--frq", required=True, help="Enter cutoff frequency.")

    #Reading Argument
    args = parser.parse_args()
    target_fps = int(args.fps)
    target_cutoff = int(args.frq)

    #Path of different source csv files
    xsens_csv_path = Path("data/processed/59_2210_cut_handacceleration_wide.csv")

    #Check if file exist
    if xsens_csv_path.exists():

        #target sources
        targets: list[PoseSequence] = []
        targets.append(
            load_xsens_pose_sequence(
                xsens_csv_path,
                sequence_id="xsens_csv_trial",
                subject_id="59",
                task="hand_motion",
                trial=2210,
                kind="position", 
            )
        )
    else:
        raise FileNotFoundError(f"{xsens_csv_path} does not exist")
    #df = pd.DataFrame([u.model_dump() for u in targets])
    #print(df.to_string())

    for target in targets:
        signal = np.asarray(target.joints["left_hand"][:20])
        filtered_signal = lowpass_filter(signal, target_fps, target_cutoff)

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

if __name__ == "__main__":
    main()