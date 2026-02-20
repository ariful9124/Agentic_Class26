import cv2
import os
import argparse

def extract_frames(video_path, output_dir, interval_secs=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    if not fps or fps <= 0:
        print("Error: Unable to determine FPS for the video.")
        return
    interval = int(fps * interval_secs)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frames = []
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_num % interval == 0:
            frames.append(frame)
        frame_num += 1

    cap.release()

    for i, frame in enumerate(frames):
        out_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(out_path, frame)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames every N seconds from a video.")
    parser.add_argument("video_path", type=str, help="Path to the input video file.")
    parser.add_argument("output_dir", type=str, help="Directory where frames will be saved.")
    parser.add_argument("--interval", type=float, default=2, help="Interval in seconds between frames (default: 2)")
    args = parser.parse_args()

    extract_frames(args.video_path, args.output_dir, args.interval)