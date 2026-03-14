import pandas as pd
from pathlib import Path

from src.io.load_xsens import load_xsens_hand_sequence
from src.features.smoothness import build_smoothness_features


def main():

    seq = load_xsens_hand_sequence(
        "data/raw/xsens_hand.csv",
        sequence_id="trial1",
    )

    rows = []

    for hand, coords in seq.joints.items():

        features = build_smoothness_features(coords, seq.fps)

        row = {
            "sequence_id": seq.sequence_id,
            "source": seq.source,
            "hand": hand,
            **features,
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    out = Path("data/processed/features.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    df.to_parquet(out)


if __name__ == "__main__":
    main()