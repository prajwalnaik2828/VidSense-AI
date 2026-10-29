import os
import sys
import time
import glob
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vidsense import config
from vidsense.video_io import probe_video, save_jpeg
from vidsense.detect import create_model, track_frame, draw_detections, TrackAggregator


def get_first_test_video() -> str | None:
    video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "videos"))
    exts = ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(video_dir, ext)))
    return sorted(files)[0] if files else None


def main():
    test_video = get_first_test_video()
    if not test_video:
        print("STOP: No video found in tests/videos/. Please add 1 or 2 test videos before running.")
        sys.exit(1)

    print(f"Loading video: {test_video}")
    info = probe_video(test_video)
    fps = info["fps"]
    stride = max(1, round(fps / config.FILE_PROCESS_FPS))
    max_frames_to_read = int(20.0 * fps)  # First 20 seconds

    model = create_model()
    aggregator = TrackAggregator(source_id="test_detect")

    cap = cv2.VideoCapture(test_video)
    frame_index = 0
    processed_count = 0
    saved_annotated = False
    annotated_save_path = os.path.join(config.DATA_DIR, "test_annotated.jpg")

    start_real_time = time.perf_counter()

    while cap.isOpened() and frame_index < max_frames_to_read:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % stride == 0:
            ts = frame_index / fps
            res = track_frame(model, frame)
            dets = aggregator.update(ts, frame, res)
            processed_count += 1

            if not saved_annotated and len(dets) > 0:
                annotated = draw_detections(frame, dets)
                save_jpeg(annotated, annotated_save_path)
                saved_annotated = True

        frame_index += 1

    cap.release()
    elapsed_real_s = time.perf_counter() - start_real_time
    video_seconds_processed = min(20.0, frame_index / fps if fps > 0 else 0)

    fps_processed = processed_count / elapsed_real_s if elapsed_real_s > 0 else 0
    vid_s_per_real_s = video_seconds_processed / elapsed_real_s if elapsed_real_s > 0 else 0

    tracks = aggregator.tracks()

    print("\n--- TRACKS TABLE ---")
    print(f"{'track_id':<10} {'cls':<12} {'first_ts':<10} {'last_ts':<10} {'n_frames':<10} {'max_conf':<10}")
    print("-" * 64)
    person_track_count = 0
    for t in sorted(tracks, key=lambda x: x["track_id"]):
        if t["cls"] == "person":
            person_track_count += 1
        print(f"{t['track_id']:<10} {t['cls']:<12} {t['first_ts']:<10.2f} {t['last_ts']:<10.2f} {t['n_frames']:<10} {t['max_conf']:<10.2f}")

    print("\n--- PERFORMANCE ---")
    print(f"Processed frames: {processed_count}")
    print(f"Real time: {elapsed_real_s:.2f} s")
    print(f"Processing FPS: {fps_processed:.2f} frames/sec")
    print(f"Video-seconds per real-second: {vid_s_per_real_s:.2f}x")

    if saved_annotated:
        print(f"\nSaved sample annotated frame to: {annotated_save_path}")
    else:
        print("\nNo detections found in first 20 seconds to save annotated frame.")

    if person_track_count > 10:
        print(f"\nWARNING: {person_track_count} distinct person tracks detected in 20 seconds. Check for possible ID switching.")

    print("\nCOMPLETED DETECTION TEST")


if __name__ == "__main__":
    main()
