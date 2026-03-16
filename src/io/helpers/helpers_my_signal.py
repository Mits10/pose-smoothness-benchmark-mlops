import numpy as np
from pathlib import Path

from src.io.load_xsens import load_xsens_pose_sequence
from src.preprocessing.filters import lowpass_filter

def get_signal(path: Path) -> np.array:
    data = load_xsens_pose_sequence(
                path,
                sequence_id="xsens_csv_trial",
                subject_id="59",
                task="hand_motion",
                trial=2210,
                kind="position",
    ).joints["left_hand"][:25]
    return data
