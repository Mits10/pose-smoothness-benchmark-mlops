import numpy as np
import pytest
from pathlib import Path

from src.io.load_xsens import load_xsens_pose_sequence
from src.io.schemas import PoseSequence
from src.preprocessing.filters import lowpass_filter

def test_lowpass_filter():
    sample_path = Path("tests/sample_data/sample_xsens.csv")
    sample_fps = 60
    sample_cutoff = 8
    sample_data = load_xsens_pose_sequence(
        sample_path,
        sequence_id="xsens_csv_trial",
        subject_id="59",
        task="hand_motion",
        trial=2210,
        kind="position", 
    ).joints["left_hand"][:5]

    
    assert isinstance(sample_data, list)
    assert len(sample_data) > 0

    sample_data = np.array(sample_data)

    sample_filters = lowpass_filter(sample_data, sample_fps, sample_cutoff)