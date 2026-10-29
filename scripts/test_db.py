import os
import sys
import tempfile
import concurrent.futures

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vidsense import config
from vidsense import db


def main():
    # Set config.DB_PATH to a temporary file
    temp_dir = tempfile.mkdtemp()
    test_db_path = os.path.join(temp_dir, "test_vidsense.db")
    config.DB_PATH = test_db_path

    all_passed = True

    def report(name: str, condition: bool):
        nonlocal all_passed
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_passed = False
        print(f"[{status}] {name}")

    try:
        # 1. init_db & table creation
        db.init_db()
        report("init_db initializes schema", os.path.exists(test_db_path))

        # 2. create_source, get_source, list_sources
        s_id1 = db.create_source("Video 1", "file", "/path/vid1.mp4", 30.0, 60.0, "ready")
        report("create_source returns 8-char hex", len(s_id1) == 8 and isinstance(s_id1, str))

        s1 = db.get_source(s_id1)
        report(
            "get_source returns source dict",
            s1 is not None and s1["name"] == "Video 1" and s1["fps"] == 30.0,
        )

        s_id2 = db.create_source("Live 1", "live", "", 5.0, None, "live")
        sources = db.list_sources()
        report(
            "list_sources returns newest first",
            len(sources) == 2 and sources[0]["source_id"] == s_id2,
        )

        # 3. update_source
        db.update_source(s_id1, status="failed", fps=24.0)
        s1_updated = db.get_source(s_id1)
        report(
            "update_source updates existing columns",
            s1_updated["status"] == "failed" and s1_updated["fps"] == 24.0,
        )

        invalid_col_raised = False
        try:
            db.update_source(s_id1, non_existent_col="bad")
        except ValueError:
            invalid_col_raised = True
        report("update_source raises ValueError on invalid column", invalid_col_raised)

        # 4. upsert_track & conflict resolution
        t1 = {
            "track_id": 1,
            "cls": "person",
            "first_ts": 1.0,
            "last_ts": 3.0,
            "n_frames": 5,
            "max_conf": 0.85,
            "best_ts": 2.0,
            "thumb_path": "vid1/1.jpg",
        }
        db.upsert_track(s_id1, t1)

        # Upsert conflict update
        t1_updated = {
            "track_id": 1,
            "cls": "person",
            "first_ts": 1.0,
            "last_ts": 5.0,
            "n_frames": 10,
            "max_conf": 0.95,
            "best_ts": 3.0,
            "thumb_path": "vid1/1_best.jpg",
        }
        db.upsert_track(s_id1, t1_updated)
        tracks = db.query_tracks(s_id1, None, None, None)
        report(
            "upsert_track updates row on conflict without duplicating",
            len(tracks) == 1
            and tracks[0]["n_frames"] == 10
            and tracks[0]["max_conf"] == 0.95
            and tracks[0]["source_name"] == "Video 1",
        )

        # 5. MIN_TRACK_FRAMES filter
        # MIN_TRACK_FRAMES is 3 by default
        t_short = {
            "track_id": 2,
            "cls": "car",
            "first_ts": 2.0,
            "last_ts": 2.5,
            "n_frames": config.MIN_TRACK_FRAMES - 1,
            "max_conf": 0.7,
            "best_ts": 2.2,
        }
        db.upsert_track(s_id1, t_short)
        tracks_filtered = db.query_tracks(s_id1, "car", None, None)
        report(
            "query_tracks filters out tracks with n_frames < MIN_TRACK_FRAMES",
            len(tracks_filtered) == 0,
        )

        # 6. Overlap logic
        # Query interval [10.0, 20.0]
        # Track A: [5.0, 8.0] -> outside before
        # Track B: [8.0, 12.0] -> overlaps start
        # Track C: [14.0, 16.0] -> inside
        # Track D: [18.0, 25.0] -> overlaps end
        # Track E: [22.0, 30.0] -> outside after
        # Track F: [5.0, 25.0] -> encompasses range
        for tid, (f_ts, l_ts) in enumerate(
            [(5.0, 8.0), (8.0, 12.0), (14.0, 16.0), (18.0, 25.0), (22.0, 30.0), (5.0, 25.0)],
            start=10,
        ):
            db.upsert_track(
                s_id1,
                {
                    "track_id": tid,
                    "cls": "bicycle",
                    "first_ts": f_ts,
                    "last_ts": l_ts,
                    "n_frames": 5,
                    "max_conf": 0.9,
                    "best_ts": f_ts,
                },
            )

        overlap_res = db.query_tracks(s_id1, "bicycle", 10.0, 20.0)
        returned_tids = [r["track_id"] for r in overlap_res]
        # Should return B (11), C (12), D (13), F (15)
        report(
            "query_tracks overlap logic correctly matches [start_ts, end_ts]",
            returned_tids == [15, 11, 12, 13] or set(returned_tids) == {11, 12, 13, 15},
        )
        # Check ordering by first_ts
        first_timestamps = [r["first_ts"] for r in overlap_res]
        report(
            "query_tracks ordered by first_ts",
            first_timestamps == sorted(first_timestamps),
        )

        # 7. count_tracks_by_class
        counts = db.count_tracks_by_class(s_id1)
        report(
            "count_tracks_by_class correctly counts by class with MIN_TRACK_FRAMES filter",
            counts.get("person") == 1 and counts.get("bicycle") == 6 and "car" not in counts,
        )

        # 8. insert_caption & count_captions
        cap_id = db.insert_caption(s_id1, 10.5, "A bicycle is ridden past.")
        report("insert_caption returns integer id", isinstance(cap_id, int) and cap_id > 0)
        report("count_captions returns correct count", db.count_captions(s_id1) == 1)

        # 9. Thread safety test (4 threads concurrently inserting captions)
        def insert_worker(thread_idx: int):
            for i in range(25):
                db.insert_caption(s_id2, float(thread_idx * 100 + i), f"Thread {thread_idx} msg {i}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(insert_worker, i) for i in range(4)]
            for f in futures:
                f.result()

        total_live_caps = db.count_captions(s_id2)
        report("thread safety: 4 threads concurrently inserted 100 captions", total_live_caps == 100)

        # 10. delete_source_rows
        db.delete_source_rows(s_id1)
        report(
            "delete_source_rows deletes rows from sources, tracks, and captions",
            db.get_source(s_id1) is None
            and len(db.query_tracks(s_id1, None, None, None)) == 0
            and db.count_captions(s_id1) == 0,
        )

        # 11. reset_tables
        db.reset_tables()
        report("reset_tables leaves all tables empty", len(db.list_sources()) == 0)

    finally:
        # Cleanup temp directory and files
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass

    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
