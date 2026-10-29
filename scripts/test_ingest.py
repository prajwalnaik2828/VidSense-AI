import os
import sys
import time
import glob
import sqlite3

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vidsense import config
from vidsense import db
from vidsense.index import (
    ingest_video_file,
    search_captions,
    delete_source,
)
from vidsense.caption import caption_frame
import cv2


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

    print(f"Ingesting test video: {test_video}")
    db.init_db()

    def progress_cb(fraction: float):
        pct = int(round(fraction * 100))
        sys.stdout.write(f"\rProgress: {pct:3d}%")
        sys.stdout.flush()

    start_time = time.perf_counter()
    source_id = ingest_video_file(
        test_video,
        name="test_ingest_video",
        progress_cb=progress_cb,
        max_seconds=60.0,
    )
    total_time = time.perf_counter() - start_time
    print(f"\nIngestion completed in {total_time:.2f} s")

    # 1. Source ID
    print(f"\nSource ID: {source_id}")

    # 2. Class counts
    class_counts = db.count_tracks_by_class(source_id)
    print(f"Class counts (tracks with n_frames >= {config.MIN_TRACK_FRAMES}):")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

    # 3. Caption counts and samples
    num_captions = db.count_captions(source_id)
    print(f"\nNumber of captions: {num_captions}")

    if num_captions == 0:
        print("\nERROR: Every caption is empty! Testing caption_frame directly to display the logged error:")
        cap = cv2.VideoCapture(test_video)
        ret, frame = cap.read()
        cap.release()
        if ret:
            res = caption_frame(frame)
            print(f"Direct caption result: '{res}'")
        sys.exit(1)

    # Fetch 3 samples with timestamps
    with db._get_connection() as conn:
        rows = conn.execute(
            "SELECT ts, text FROM captions WHERE source_id = ? ORDER BY ts ASC LIMIT 3",
            (source_id,),
        ).fetchall()
        print("Sample captions:")
        for r in rows:
            print(f"  [{r['ts']:.2f}s] {r['text']}")

    # 4. Search captions
    search_query = "a person walking"
    search_res = search_captions(search_query, source_id=source_id, k=config.TOP_K)
    print(f"\nSearch result for '{search_query}':")
    if search_res:
        top_match = search_res[0]
        print(f"  Top match [{top_match['ts']:.2f}s, dist={top_match['distance']:.3f}]: {top_match['text']}")
    else:
        print("  No caption matched within MAX_COSINE_DISTANCE")

    # 5. Cleanup check
    if "--cleanup" in sys.argv:
        print(f"\nRunning cleanup for source {source_id}...")
        delete_source(source_id)
        print("Cleanup completed.")
    else:
        print(f"\nKept source {source_id} in index for subsequent queries/tests (use --cleanup to remove).")

    print("\nINGESTION TEST FINISHED SUCCESSFULLY")


if __name__ == "__main__":
    main()
