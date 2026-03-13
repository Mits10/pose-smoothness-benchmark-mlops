import pytest
from pathlib import Path
from src.io.load_xsens import parse_mvnx_file

def test_process_directory():
    sample = Path("tests/sample_data/sample.mvnx")
    df = parse_mvnx_file(sample)

    assert len(df) > 0
    assert "frame_idx" in df.columns
    assert "time_s" in df.columns
    assert df["time_s"].is_monotonic_increasing

def test_missing_file_raises_error():
    bad_path = Path("tests/sample_data/does_not_exist.mvnx")

    with pytest.raises(FileNotFoundError):
        parse_mvnx_file(bad_path)