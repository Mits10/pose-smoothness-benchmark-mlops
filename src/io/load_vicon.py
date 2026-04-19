from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.io.schemas import PoseSequence, SequenceMetadata


HAND_POSITION_COLUMNS = {
    "right_hand": ["right_hand_x", "right_hand_y"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        c.strip().lower().replace(" ", "_").replace(".", "_") for c in out.columns
    ]
    return out


def _infer_fps_from_time(df: pd.DataFrame) -> float:
    time_candidates = ["time", "time_s", "timestamp", "frame_time"]
    for col in time_candidates:
        if col in df.columns:
            diffs = df[col].diff().dropna()
            if not diffs.empty:
                step = float(diffs.median())
                if step > 0:
                    if step > 1.0:
                        return 1000.0 / step
                    return 1.0 / step
    raise ValueError(
        "Could not infer fps from Vicon file. Add a time column such as 'time' or 'time_s'."
    )

def _center_of_hand(df: pd.DataFrame) -> pd.DataFrame:
    df["right_hand_x"] = df[["hand1_x", "hand2_x", "hand3_x"]].mean(axis=1)
    df["right_hand_y"] = df[["hand1_y", "hand2_y", "hand3_y"]].mean(axis=1)
    df["right_hand_z"] = df[["hand1_z", "hand2_z", "hand3_z"]].mean(axis=1)
    df = df[["right_hand_x", "right_hand_y", "right_hand_z"]]
    return df
    

def load_vicon_hand_sequence(
    path: str | Path,
    *,
    sequence_id: str,
    subject_id: str = "unknown_subject",
    task: str = "hand_motion",
    trial: int = 1,
    fps: float | None = None,
    column_map: dict[str, list[str]] | None = None,
) -> PoseSequence:
    """Load Vicon hand positions from a CSV export.

    Expected default columns after normalization:
    - left_hand_x, left_hand_y, left_hand_z
    - right_hand_x, right_hand_y, right_hand_z

    Use `column_map` if your export uses different names.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = _normalize_columns(df)
    df = _center_of_hand(df)

    active_map = column_map or HAND_POSITION_COLUMNS

    joints: dict[str, list[list[float]]] = {}
    for joint_name, cols in active_map.items():
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing Vicon columns for {joint_name}: {missing}. "
                f"Available columns sample: {list(df.columns[:12])}"
            )
        joints[joint_name] = df[cols].astype(float).values.tolist()

    inferred_fps = fps if fps is not None else _infer_fps_from_time(df)

    return PoseSequence(
        sequence_id=sequence_id,
        source="vicon",
        fps=inferred_fps,
        joints=joints,
        metadata=SequenceMetadata(
            subject_id=subject_id,
            task=task,
            trial=trial,
        ),
    )