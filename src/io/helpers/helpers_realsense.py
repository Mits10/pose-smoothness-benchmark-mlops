import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
import os
import csv
#     data_folder = Path(r"C:\Users\awila\OneDrive\Desktop\Mitaly\Openpose\RealSense\input_data\104_1003.bag")
#     output_dir = Path(r"C:\Users\awila\OneDrive\Desktop\Mitaly\Openpose\RealSense")

# --- Paths ---
bag_file = r"C:\Users\awila\OneDrive\Desktop\Mitaly\Openpose\RealSense\input_data\104_1003.bag"      # full path to your .bag
save_dir = r"C:\Users\awila\OneDrive\Desktop\Mitaly\Openpose\RealSense"            # directory to save frames
csv_file = os.path.join(save_dir, "timestamps.csv")  # CSV for frame timestamps

# Make sure save directory exists
os.makedirs(save_dir, exist_ok=True)

# --- RealSense setup ---
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file(bag_file, repeat_playback=False)

# Start pipeline
profile = pipeline.start(config)

# Optional: slow down playback so frames match recorded FPS
playback = profile.get_device().as_playback()
playback.set_real_time(False)

def main() -> None:
    print("Saving frames and timestamps...")

    frame_count = 0
    timestamps = []

    try:
        while True:
            try:
                # Wait up to 5 seconds for next frame
                frames = pipeline.wait_for_frames(timeout_ms=5000)
            except RuntimeError:
                # End of bag reached
                break

            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert frame to numpy array (BGR)
            image = np.asanyarray(color_frame.get_data())

            # Save image
            filename = os.path.join(save_dir, f"frame_{frame_count:05d}.png")
            cv2.imwrite(filename, image)

            # Save timestamp in milliseconds
            frame_time = color_frame.get_timestamp()
            timestamps.append(frame_time)

            frame_count += 1
            if frame_count % 50 == 0:
                print(f"Saved {frame_count} frames")

    finally:
        pipeline.stop()
        print(f"Finished! Total frames saved: {frame_count}")

        # Write timestamps to CSV
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "timestamp_ms"])
            for idx, ts in enumerate(timestamps):
                writer.writerow([idx, ts])
        print(f"Timestamps saved to: {csv_file}")

if __name__ == "__main__":
    main()