import os
import sys
import time
import glob
import json
import subprocess
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vidsense import config
from vidsense import db
from vidsense import video_io
from vidsense.live import LiveSession


def get_first_test_video() -> str | None:
    video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "videos"))
    exts = ("*.mp4", "*.mkv", "*.avi", "*.mov")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(video_dir, ext)))
    return sorted(files)[0] if files else None


def run_checks(session: LiveSession, duration_s: float, mode_desc: str, feeder_fn=None):
    print(f"\n=== Starting Live Test: {mode_desc} ===")
    source_id = session.start()
    if not source_id:
        print("[FAIL] start() did not return a valid source_id")
        sys.exit(1)
    if not session.is_running():
        print("[FAIL] session.is_running() is False immediately after start()")
        sys.exit(1)
    print(f"[PASS] Session started successfully (source_id: {source_id})")

    start_time = time.time()
    last_stat_time = start_time
    seq_values = []

    if feeder_fn:
        # Feeder runs in caller thread (push mode)
        feeder_fn(session, duration_s, seq_values)
    else:
        # Capture mode runs in background thread
        while time.time() - start_time < duration_s:
            now = time.time()
            if now - last_stat_time >= 3.0:
                last_stat_time = now
                st = session.stats()
                ev = session.latest_event()
                jpeg = session.latest_jpeg()

                seq = ev["seq"] if ev else None
                if seq is not None:
                    seq_values.append(seq)

                print(
                    f"  [{now - start_time:4.1f}s] stats: fps={st.get('fps')}, "
                    f"active_tracks={st.get('active_tracks')}, seq={seq}, "
                    f"has_jpeg={bool(jpeg and len(jpeg) > 0)}"
                )
            time.sleep(0.2)

    # 1. Confirm event seq values increase
    if len(seq_values) >= 2 and seq_values[-1] > seq_values[0]:
        print(f"[PASS] Event seq values increased over time ({seq_values[0]} -> {seq_values[-1]})")
    elif len(seq_values) == 1 and seq_values[0] > 0:
        print(f"[PASS] Event seq value recorded: {seq_values[0]}")
    else:
        print(f"[FAIL] Event seq values did not increase properly: {seq_values}")
        sys.exit(1)

    # 2. Stop session and check timing
    stop_start = time.time()
    session.stop()
    stop_elapsed = time.time() - stop_start
    if stop_elapsed < 30.0:
        print(f"[PASS] stop() completed in {stop_elapsed:.2f} s (< 30 s)")
    else:
        print(f"[FAIL] stop() took too long: {stop_elapsed:.2f} s")
        sys.exit(1)

    if not session.is_running():
        print("[PASS] session.is_running() is False after stop()")
    else:
        print("[FAIL] session.is_running() is still True after stop()")
        sys.exit(1)

    # 3. Print saved source row
    src = db.get_source(source_id)
    print("\nSaved source row:")
    print(json.dumps(src, indent=2, default=str))
    if src and src.get("status") == "ready":
        print("[PASS] Source row status is 'ready'")
    else:
        print(f"[FAIL] Source row status is not ready: {src}")
        sys.exit(1)

    # 4. Print track count
    tracks = db.query_tracks(source_id, None, None, None)
    print(f"Track count: {len(tracks)}")
    print(f"[PASS] Track count retrieved: {len(tracks)}")

    # 5. Print caption count
    n_captions = db.count_captions(source_id)
    print(f"Caption count: {n_captions}")
    print(f"[PASS] Caption count retrieved: {n_captions}")

    # 6. Verify with ffprobe that data/live/<source_id>.mp4 has duration > 0
    mp4_path = os.path.join(config.DATA_DIR, "live", f"{source_id}.mp4")
    if not os.path.exists(mp4_path):
        print(f"[FAIL] Output video file does not exist: {mp4_path}")
        sys.exit(1)

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        mp4_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[FAIL] ffprobe failed: {proc.stderr}")
            sys.exit(1)
        duration = float(proc.stdout.strip())
        if duration > 0:
            print(f"[PASS] ffprobe verified {mp4_path} duration: {duration:.2f} s (> 0)")
        else:
            print(f"[FAIL] ffprobe reported non-positive duration: {duration}")
            sys.exit(1)
    except Exception as e:
        print(f"[FAIL] Error running ffprobe: {e}")
        sys.exit(1)

    print(f"\nALL CHECKS PASSED FOR {mode_desc.upper()}\n")


def push_feeder(session: LiveSession, duration_s: float, seq_values: list):
    test_video = get_first_test_video()
    if not test_video:
        print("STOP: No video found in tests/videos/. Please add test videos before running.")
        sys.exit(1)

    cap = cv2.VideoCapture(test_video)
    start_time = time.time()
    last_stat_time = start_time
    interval = 1.0 / config.LIVE_PROCESS_FPS

    try:
        while time.time() - start_time < duration_s:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            jpeg = video_io.encode_jpeg(frame)
            session.push_frame(jpeg)

            now = time.time()
            if now - last_stat_time >= 3.0:
                last_stat_time = now
                st = session.stats()
                ev = session.latest_event()
                jpeg_ann = session.latest_jpeg()
                seq = ev["seq"] if ev else None
                if seq is not None:
                    seq_values.append(seq)
                print(
                    f"  [{now - start_time:4.1f}s] stats: fps={st.get('fps')}, "
                    f"active_tracks={st.get('active_tracks')}, seq={seq}, "
                    f"has_jpeg={bool(jpeg_ann and len(jpeg_ann) > 0)}"
                )

            time.sleep(interval)
    finally:
        cap.release()


def main():
    db.init_db()
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "file"

    if mode == "file":
        target_video = sys.argv[2] if len(sys.argv) > 2 else get_first_test_video()
        if not target_video:
            print("STOP: No video found in tests/videos/. Please add test videos before running.")
            sys.exit(1)
        session = LiveSession(camera=target_video, name="Simulated Live Test")
        run_checks(session, duration_s=15.0, mode_desc=f"SIMULATED LIVE (15 s, file={target_video})")

    elif mode == "push":
        test_video = get_first_test_video()
        if not test_video:
            print("STOP: No video found in tests/videos/. Please add test videos before running.")
            sys.exit(1)
        session = LiveSession(camera="push", name="Push Mode Test")
        run_checks(
            session,
            duration_s=10.0,
            mode_desc="PUSH MODE (10 s)",
            feeder_fn=push_feeder,
        )

    elif mode == "webcam":
        cam_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        session = LiveSession(camera=cam_idx, name="Webcam Live Test")
        run_checks(session, duration_s=10.0, mode_desc=f"WEBCAM (10 s, index={cam_idx})")

    else:
        print(f"Unknown mode '{mode}'. Usage: python scripts/test_live_core.py [file [path] | push | webcam [index]]")
        sys.exit(1)


if __name__ == "__main__":
    main()
