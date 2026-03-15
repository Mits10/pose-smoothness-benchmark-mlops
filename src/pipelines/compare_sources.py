from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from src.io.load_xsens import load_xsens_pose_sequence
from src.io.schemas import PoseSequence

def main() -> None:
    #Command Line Argument
    parser = argparse.ArgumentParser(description="Compare different sources pose smoothness.")
    parser.add_argument("--fps", required= True, help=" Inter taget FPS.")
    parser.add_argument("--frq", required=True, help="Enter cutoff frequency.")

    #Reading Argument
    args = parser.parse_args()
    target_fps = args.fps
    filter_cutoff_hz = args.frq

    #Path of different source csv files
    xsens_csv_path = Path("data/processed/59_2210_cut_handacceleration_wide.csv")

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
    df = pd.DataFrame([u.model_dump() for u in targets])
    print(df.to_string())




if __name__ == "__main__":
    main()