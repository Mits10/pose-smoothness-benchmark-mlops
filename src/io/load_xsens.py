from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from src.io.schemas import PoseSequence, SequenceMetadata

#calculating frame per second from the timestamp
def _infer_fps(time_s: pd.Series) -> float:
    diffs = time_s.diff().dropna()
    if diffs.empty:
        raise ValueError("Cannot infer fps from time_s.")

    median_step = float(diffs.median())
    if median_step <= 0:
        raise ValueError("time_s must be increasing.")

    # If values are milliseconds instead of seconds
    if median_step > 1.0:
        return 1000.0 / median_step
    return 1.0 / median_step

#load function of xsens
#input: csv
#output: pydantic schema
def load_xsens_pose_sequence(
    input_file: Path,
    *,
    sequence_id: str,
    subject_id: str = "unknown_subject",
    task: str = "hand_motion",
    trial: int = 1,
    left_hand_cols: tuple[str, str, str] = ("LeftHand_X", "LeftHand_Y", "LeftHand_Z"),
    right_hand_cols: tuple[str, str, str] = ("RightHand_X", "RightHand_Y", "RightHand_Z"),

) -> PoseSequence:

    df = pd.read_csv(input_file)

    required_cols = ["time_s", *left_hand_cols, *right_hand_cols]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required hand columns in MVNX data: {missing}")

    joints = {
        "left_hand": df[list(left_hand_cols)].astype(float).values.tolist(),
        "right_hand": df[list(right_hand_cols)].astype(float).values.tolist(),
    }

    return PoseSequence(
        sequence_id=sequence_id,
        source="xsens",
        fps=_infer_fps(df["time_s"]),
        joints=joints,
        metadata=SequenceMetadata(
            subject_id=subject_id,
            task=task,
            trial=trial,
        ),
    )