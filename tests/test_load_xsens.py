from pathlib import Path
from src.io.load_xsens import parse_mvnx_file

def test_process_directory():
    sample = Path("tests/sample_data/sample.mvnx")
    df = parse_mvnx_file(sample)

    assert len(df) > 0
    assert "frame_idx" in df.columns
    assert "time_s" in df.columns