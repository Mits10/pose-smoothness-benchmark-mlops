import os
import json
import pandas as pd
import re

# -----------------------------
# Body keypoints (COCO body_25)
# -----------------------------
body_joints = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
    "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe",
    "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
]

# Hands: 21 keypoints each
hand_joints = [
    "Wrist",
    "Thumb1","Thumb2","Thumb3","Thumb4",
    "Index1","Index2","Index3","Index4",
    "Middle1","Middle2","Middle3","Middle4",
    "Ring1","Ring2","Ring3","Ring4",
    "Pinky1","Pinky2","Pinky3","Pinky4"
]

# -----------------------------
# Helper functions
# -----------------------------
def safe_array(arr, expected_len):
    if not arr:
        return [0]*expected_len
    return arr + [0]*(expected_len - len(arr))

def get_frame_number(file_name):
    """Extract frame number from filenames like 102_1003_1_000000000000_keypoints.json"""
    match = re.search(r'_(\d+)_keypoints', file_name)
    if match:
        return int(match.group(1))
    return -1

# -----------------------------
# Main
# -----------------------------
def main() -> None:
    folder_path = r"C:\Users\awila\OneDrive\Desktop\Mitaly\Openpose\GoPro\Output_json\output_json_gopro_104_1"
    output_folder = r"C:\Users\awila\OneDrive\Desktop\Mitaly\Openpose\GoPro\Output_json"
    output_file = os.path.join(output_folder, "104_1_gopro_openpose.csv")

    all_rows = []

    # Get JSON files and sort by real frame number
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    files = sorted(files, key=get_frame_number)

    for file_name in files:
        frame_idx = get_frame_number(file_name)
        file_path = os.path.join(folder_path, file_name)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            people = data.get("people", [])

            for person_idx, person in enumerate(people):
                # Safe arrays
                pose = safe_array(person.get("pose_keypoints_2d"), 75)
                hand_l = safe_array(person.get("hand_left_keypoints_2d"), 63)
                hand_r = safe_array(person.get("hand_right_keypoints_2d"), 63)

                # Row dictionary
                row = {
                    "frame": frame_idx,
                    "person_id": person_idx
                }

                # Body keypoints
                for i, joint in enumerate(body_joints):
                    row[f"{joint}_x"] = pose[i*3]
                    row[f"{joint}_y"] = pose[i*3 + 1]
                    row[f"{joint}_c"] = pose[i*3 + 2]

                # Left hand keypoints
                for i, joint in enumerate(hand_joints):
                    row[f"LHand_{joint}_x"] = hand_l[i*3]
                    row[f"LHand_{joint}_y"] = hand_l[i*3 + 1]
                    row[f"LHand_{joint}_c"] = hand_l[i*3 + 2]

                # Right hand keypoints
                for i, joint in enumerate(hand_joints):
                    row[f"RHand_{joint}_x"] = hand_r[i*3]
                    row[f"RHand_{joint}_y"] = hand_r[i*3 + 1]
                    row[f"RHand_{joint}_c"] = hand_r[i*3 + 2]

                all_rows.append(row)

    # Create DataFrame directly from rows (no predefined columns)
    final_df = pd.DataFrame(all_rows)

    # Save CSV
    final_df.to_csv(output_file, index=False)
    print(f"✅ CSV saved: {output_file}")

if __name__ == "__main__":
    main()