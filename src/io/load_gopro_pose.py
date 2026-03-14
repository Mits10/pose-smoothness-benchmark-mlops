from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.io.schemas import PoseSequence, SequenceMetadata


GOPRO_2D_COLUMNS = {
    "left_hand": ["left_hand_x", "left_hand_y"],
    "right_hand": ["right_hand_x", "right_hand_y"],
}

GOPRO_3D_COLUMNS = {
    "left_hand": ["left_hand_x", "left_hand_y", "left_hand_z"],
    "right_hand": ["right_hand_x", "right_hand_y", "right_hand_z"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        c.strip().lower().replace(" ", "_").replace(".", "_") for c in out.columns
    ]
    return out


def _infer_fps(df: pd.DataFrame, fps: float | None) -> float:
    if fps is not None:
        return fps

    if "time_s" in df.columns:
        diffs = df["time_s"].diff().dropna()
        if not diffs.empty:
            step = float(diffs.median())
            if step > 0:
                if step > 1.0:
                    return 1000.0 / step
                return 1.0 / step

    if "frame" in df.columns and len(df) > 1:
        return 30.0

    raise ValueError(
        "Could not infer fps for GoPro pose data. Pass fps explicitly or add time_s."
    )


def load_gopro_hand_sequence(
    path: str | Path,
    *,
    sequence_id: str,
    subject_id: str = "unknown_subject",
    task: str = "hand_motion",
    trial: int = 1,
    fps: float | None = None,
    use_3d: bool = False,
    column_map: dict[str, list[str]] | None = None,
) -> PoseSequence:
    """Load GoPro/video-pose hand keypoints from CSV."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = _normalize_columns(df)

    default_map = GOPRO_3D_COLUMNS if use_3d else GOPRO_2D_COLUMNS
    active_map = column_map or default_map

    joints: dict[str, list[list[float]]] = {}
    for joint_name, cols in active_map.items():
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"Missing GoPro columns for {joint_name}: {missing}. "
                f"Available columns sample: {list(df.columns[:12])}"
            )
        joints[joint_name] = df[cols].astype(float).values.tolist()

    return PoseSequence(
        sequence_id=sequence_id,
        source="gopro",
        fps=_infer_fps(df, fps),
        joints=joints,
        metadata=SequenceMetadata(
            subject_id=subject_id,
            task=task,
            trial=trial,
        ),
    )