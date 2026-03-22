import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path

def main() -> None:
# ----------------------------
# USER SETTINGS
# ----------------------------
    data_folder = Path(r"C:\Users\awila\OneDrive\Desktop\Mitaly\Openpose\RealSense\input_data")
    output_dir = Path(r"C:\Users\awila\OneDrive\Desktop\Mitaly\Openpose\RealSense\Output_data_json")
    BAG_FILE = data_folder/"104_1003.bag"   # Path to your .bag file inside Docker
    VIDEO_FILE = output_dir/"subject_4.mp4"  # Output video
    print(str(BAG_FILE))  # Verify it prints the correct full path
    print(BAG_FILE.exists())  # Should print True
    WIDTH, HEIGHT = 640, 480
    FPS = 30

    # ----------------------------
    # Initialize RealSense pipeline
    # ----------------------------
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(BAG_FILE), repeat_playback=True)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)

    pipeline.start(config)

    # ----------------------------
    # Create OpenCV VideoWriter
    # ----------------------------
    out = cv2.VideoWriter(str(VIDEO_FILE),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        FPS,
                        (WIDTH, HEIGHT))

    print(f"[INFO] Extracting frames from {BAG_FILE} ...")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            out.write(frame)

    except RuntimeError:
        # End of bag file
        pass

    pipeline.stop()
    out.release()
    print(f"[INFO] Video saved: {VIDEO_FILE}")

if __name__ == "__main__":
    main()