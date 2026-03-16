from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
import pandas as pd


from src.io.load_xsens import load_xsens_pose_sequence
from src.io.schemas import PoseSequence
from src.preprocessing.filters import lowpass_filter

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
        print(signal)
        print("signal shape %d", signal.shape)
        print(filtered_signal)

    


if __name__ == "__main__":
    main()