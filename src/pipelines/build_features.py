from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.smoothness import build_smoothness_features
from src.io.load_xsens import load_xsens_from_csv, load_xsens_from_mvnx
from src.io.schemas import PoseSequence


def sequence_to_feature_row(seq: PoseSequence) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "sequence_id": seq.sequence_id,
        "source": seq.source,
        "fps": seq.fps,
    }

    if seq.metadata is not None:
        row["subject_id"] = seq.metadata.subject_id
        row["task"] = seq.metadata.task
        row["trial"] = seq.metadata.trial

    for hand_name, coords in seq.joints.items():
        features = build_smoothness_features(coords, fps=seq.fps)
        for feature_name, value in features.items():
            row[f"{hand_name}_{feature_name}"] = value

    # Simple placeholder target for now.
    # Replace this later with your real benchmark label against Vicon.
    left_rms = float(row.get("left_hand_rms_jerk", 0.0))
    right_rms = float(row.get("right_hand_rms_jerk", 0.0))
    avg_rms = (left_rms + right_rms) / 2.0

    reliability_score = max(0.0, min(1.0, 1.0 / (1.0 + avg_rms)))
    row["reliability_score"] = reliability_score

    return row


def build_features_from_sequences(sequences: list[PoseSequence]) -> pd.DataFrame:
    rows = [sequence_to_feature_row(seq) for seq in sequences]
    return pd.DataFrame(rows)


def main() -> None:
    sequences: list[PoseSequence] = []

    # Example 1: load from an already-exported wide CSV
    xsens_csv = Path("data/raw/59_2210_cut_handacceleration_wide.csv")
    if xsens_csv.exists():
        seq = load_xsens_from_csv(
            xsens_csv,
            sequence_id="xsens_59_2210",
            subject_id="59",
            task="hand_motion",
            trial=2210,
            kind="position",
        )
        sequences.append(seq)

    # Example 2: load directly from raw MVNX
    xsens_mvnx = Path("data/raw/example_trial.mvnx")
    if xsens_mvnx.exists():
        seq = load_xsens_from_mvnx(
            xsens_mvnx,
            sequence_id="xsens_raw_example",
            subject_id="unknown_subject",
            task="hand_motion",
            trial=1,
            kind="position",
        )
        sequences.append(seq)

    if not sequences:
        raise FileNotFoundError(
            "No input files found. Add at least one Xsens CSV or MVNX file in data/raw/."
        )

    df = build_features_from_sequences(sequences)

    out_path = Path("data/processed/features.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(f"Saved {len(df)} feature rows to {out_path}")


if __name__ == "__main__":
    main()