import logging
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from src.io.schemas import PoseSequence, SequenceMetadata

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s|%(levelname)s|%(message)s"
)

def split_xyz(values: list[float], names: list[str], prefix: str) -> dict[str, float]: 
    cols: dict[str, float] = {} 
    for i, name in enumerate(names): 
        base = i * 3 
        if base + 2 >= len(values): 
            break 
        cols[f"{name}_{prefix}X"] = values[base] 
        cols[f"{name}_{prefix}Y"] = values[base + 1] 
        cols[f"{name}_{prefix}Z"] = values[base + 2] 
    return cols

def split_quat(values: list[float], names: list[str]) -> dict[str, float]: 
    cols: dict[str, float] = {} 
    for i, name in enumerate(names): 
        base = i * 4 
        if base + 3 >= len(values): 
            break 
        cols[f"{name}_qw"] = values[base] 
        cols[f"{name}_qx"] = values[base + 1] 
        cols[f"{name}_qy"] = values[base + 2] 
        cols[f"{name}_qz"] = values[base + 3] 
    return cols

def parse_mvnx_file(input_file:Path) -> pd.DataFrame:
    #parsing the XML
    tree = ET.parse(input_file)
    root = tree.getroot()

    #Handling the XML namespace
    ns = {"m": root.tag.split("}")[0].strip("{")}

    #Reading metadata
    segments = [s.attrib["label"] for s in root.findall(".//m:segment", ns)]
    joints = [j.attrib["label"] for j in root.findall(".//m:joint", ns)]
    sensors = [s.attrib["label"] for s in root.findall(".//m:sensor", ns)]

    #Creating row for storage
    rows: list[dict[str,float]] = []

    #looping through frame to store segments values
    for frame in root.findall(".//m:frame", ns):
        # If index has empty values
        index_value = frame.attrib.get("index")
        if not index_value:
            continue  # skip this frame

        row: dict[str, float] = {}
        row["frame_idx"] = int(frame.attrib.get("index",0))
        row["time_s"] = float(frame.attrib.get("time", 0.0))

        for child in frame:
            tag = child.tag.split("}")[-1]
            if child.text is None or not child.text.strip():
                continue
            values = list(map(float, child.text.split()))

            if tag == "position": row.update(split_xyz(values, segments, "")) 
            elif tag == "velocity": row.update(split_xyz(values, segments, "V")) 
            elif tag == "acceleration": row.update(split_xyz(values, segments, "A")) 
            elif tag == "orientation": row.update(split_quat(values, segments)) 
            elif tag == "sensorAcceleration": row.update(split_xyz(values, sensors, "A")) 
            elif tag == "sensorOrientation": row.update(split_quat(values, sensors)) 
            elif tag == "jointAngle": 
                for i, joint in enumerate(joints): 
                    if i >= len(values): 
                        break 
                    row[f"{joint}_angle"] = values[i] 

        rows.append(row)
    df = pd.DataFrame(rows)
    return df

def process_one_file(input_file: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents = True, exist_ok=True)
    output_file = output_dir / f"{input_file.stem}_wide.csv"

    #Checking if the file already processed
    if output_file.exists():
        if output_file.stat().st_mtime >= input_file.stat().st_mtime:
            logging.info("Skipping %s (already processed)", input_file)
            return output_file

    logging.info("Processing %s",input_file)
    df = parse_mvnx_file(input_file)
    df.to_csv(output_file, index=False)
    logging.info("Saving %s", output_file)


    return None

def process_directory(input_dir: Path, output_dir: Path, recursive: bool = True) -> None:
    pattern = "**/*.mvnx" if recursive else "*.mvnx"
    files = sorted(input_dir.glob(pattern))

    if not files:
        logging.warning("No mvnx files found in %s", input_dir)
        return
   
    logging.info("Found %d .mvnx files in %s", len(files),input_dir)

    for file_path in files:
        try:
            process_one_file(file_path, output_dir)
        except Exception as exc:
            logging.exception("Failed processing %s : %s", file_path,exc)

def mvnx_to_pose_sequence(
    input_file: Path,
    *,
    sequence_id: str,
    subject_id: str = "unknown_subject",
    task: str = "hand_motion",
    trial: int = 1,
    left_hand_cols: tuple[str, str, str] = ("LeftHand_X", "LeftHand_Y", "LeftHand_Z"),
    right_hand_cols: tuple[str, str, str] = ("RightHand_X", "RightHand_Y", "RightHand_Z"),
) -> PoseSequence:
    df = parse_mvnx_file(input_file)

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