import os
import sys
import glob
import time
import json
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vidsense import config


def get_first_test_video() -> str | None:
    video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "videos"))
    exts = ("*.mp4", "*.mkv", "*.avi", "*.mov")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(video_dir, ext)))
    return sorted(files)[0] if files else None


def wait_for_server(base_url: str, timeout: int = 15) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{base_url}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    base_url = f"http://127.0.0.1:{config.API_PORT}"
    print(f"Connecting to VidSense server at {base_url}...")

    if not wait_for_server(base_url):
        print(f"[FAIL] Server at {base_url} is not responding on /health.")
        sys.exit(1)
    print("Server is healthy.\n")

    auth_headers = {}
    if config.API_KEY:
        auth_headers["X-API-Key"] = config.API_KEY

    # Check 1: Find test video and upload
    test_video = get_first_test_video()
    if not test_video:
        print("STOP: No video found in tests/videos/. Please add test videos before running.")
        sys.exit(1)

    print(f"Found test video: {test_video}")
    video_basename = os.path.basename(test_video)

    try:
        with open(test_video, "rb") as f:
            up_res = requests.post(
                f"{base_url}/sources/upload",
                files={"file": (video_basename, f, "video/mp4")},
                headers=auth_headers,
            )
        if up_res.status_code != 202:
            print(f"[FAIL] Check 1: Upload video failed with status {up_res.status_code}: {up_res.text}")
            sys.exit(1)
        up_data = up_res.json()
        source_id = up_data.get("source_id")
        if not source_id or up_data.get("status") != "processing":
            print(f"[FAIL] Check 1: Unexpected response body from upload: {up_data}")
            sys.exit(1)
        print(f"[PASS] Check 1: Video uploaded successfully (source_id: {source_id})")
    except Exception as e:
        print(f"[FAIL] Check 1: Error during upload: {e}")
        sys.exit(1)

    # Check 2: Poll GET /sources/{source_id} until ready
    print(f"\nPolling /sources/{source_id} until ready (timeout 15 min)...")
    start_poll = time.time()
    source_out = None
    while time.time() - start_poll < 900:
        try:
            r = requests.get(f"{base_url}/sources/{source_id}", headers=auth_headers)
            if r.status_code == 200:
                data = r.json()
                st = data.get("status")
                prog = float(data.get("progress", 0.0))
                sys.stdout.write(f"\r  Status: {st:<10} Progress: {prog * 100:5.1f}%")
                sys.stdout.flush()
                if st == "ready":
                    source_out = data
                    break
                elif st == "failed":
                    print(f"\n[FAIL] Check 2: Source processing failed!")
                    sys.exit(1)
            else:
                print(f"\n[FAIL] Check 2: Error polling source: status {r.status_code}")
                sys.exit(1)
        except Exception as e:
            print(f"\n[FAIL] Check 2: Exception polling source: {e}")
            sys.exit(1)
        time.sleep(1.0)

    if not source_out:
        print("\n[FAIL] Check 2: Polling timed out after 15 minutes.")
        sys.exit(1)

    elapsed_s = time.time() - start_poll
    print(f"\n[PASS] Check 2: Video processing ready in {elapsed_s:.1f} s")

    # Check 3: Print SourceOut and validate schema
    print("\nSourceOut:")
    print(json.dumps(source_out, indent=2))
    required_keys = {
        "source_id",
        "name",
        "kind",
        "status",
        "duration_s",
        "fps",
        "created_at",
        "progress",
        "n_tracks",
        "class_counts",
        "n_captions",
        "has_clip",
    }
    missing_keys = required_keys - set(source_out.keys())
    if missing_keys:
        print(f"[FAIL] Check 3: Missing keys in SourceOut: {missing_keys}")
        sys.exit(1)
    if source_out["status"] != "ready" or source_out["progress"] != 1.0 or not source_out["has_clip"]:
        print(f"[FAIL] Check 3: Inconsistent SourceOut values: {source_out}")
        sys.exit(1)
    print("[PASS] Check 3: SourceOut schema and values verified")

    # Check 4: POST /search for queries
    queries = [
        "How many people are there?",
        "Is there a dinosaur?",
    ]
    all_results = []
    print("\nRunning search queries:")
    for q in queries:
        s_res = requests.post(
            f"{base_url}/search",
            json={"question": q, "source_id": source_id},
            headers=auth_headers,
        )
        if s_res.status_code != 200:
            print(f"[FAIL] Check 4: Search '{q}' failed with status {s_res.status_code}: {s_res.text}")
            sys.exit(1)
        s_data = s_res.json()
        print(f"  Q: \"{q}\"")
        print(f"  Answer: {s_data.get('answer')}")
        print(f"  Found: {s_data.get('found')}, Results count: {len(s_data.get('results', []))}")
        all_results.extend(s_data.get("results", []))

    print("[PASS] Check 4: Search queries executed successfully")

    # Check 5: Download one clip via clip_url and fetch EVERY thumbnail_url
    clip_urls = [r["clip_url"] for r in all_results if r.get("clip_url")]
    thumb_urls = [r["thumbnail_url"] for r in all_results if r.get("thumbnail_url")]

    if clip_urls:
        first_clip_url = clip_urls[0]
    else:
        # Fallback to direct clip URL for this source so clip downloading is always verified
        first_clip_url = f"/sources/{source_id}/clip?start=0&end=5"

    clip_full_url = f"{base_url}{first_clip_url}"
    print(f"\nDownloading sample clip from: {first_clip_url}")
    clip_res = requests.get(clip_full_url, headers=auth_headers)
    if clip_res.status_code != 200 or len(clip_res.content) == 0:
        print(f"[FAIL] Check 5: Clip download failed: status {clip_res.status_code}, length {len(clip_res.content)}")
        sys.exit(1)
    print(f"  Clip downloaded successfully: {len(clip_res.content)} bytes, Content-Type: {clip_res.headers.get('content-type')}")

    if not thumb_urls:
        # If search returned no results, fetch the thumbnails generated for this source's tracks
        from vidsense import db
        tracks = db.query_tracks(source_id, None, None, None)
        thumb_urls = [f"/media/frames/{t['thumb_path']}" for t in tracks if t.get("thumb_path")]

    print(f"Fetching {len(thumb_urls)} thumbnail(s)...")
    for idx, t_url in enumerate(thumb_urls, start=1):
        thumb_res = requests.get(f"{base_url}{t_url}", headers=auth_headers)
        if thumb_res.status_code != 200 or len(thumb_res.content) == 0:
            print(f"[FAIL] Check 5: Thumbnail {t_url} fetch failed: status {thumb_res.status_code}, length {len(thumb_res.content)}")
            sys.exit(1)
    print(f"[PASS] Check 5: Clip and all {len(thumb_urls)} thumbnails verified (200, non-empty)")

    # Check 6: Check that a .exe upload returns 400
    exe_res = requests.post(
        f"{base_url}/sources/upload",
        files={"file": ("malicious.exe", b"fake binary executable content", "application/octet-stream")},
        headers=auth_headers,
    )
    if exe_res.status_code != 400 or "error" not in exe_res.json():
        print(f"[FAIL] Check 6: .exe upload expected 400, got {exe_res.status_code}: {exe_res.text}")
        sys.exit(1)
    print(f"[PASS] Check 6: .exe upload returned 400: {exe_res.json().get('error')}")

    # Check 7: Check that an upload named "../../evil.mp4" is stored safely inside data/uploads
    evil_res = requests.post(
        f"{base_url}/sources/upload",
        files={"file": ("../../evil.mp4", b"dummy mp4 content", "video/mp4")},
        headers=auth_headers,
    )
    if evil_res.status_code != 202:
        print(f"[FAIL] Check 7: Upload ../../evil.mp4 expected 202, got {evil_res.status_code}: {evil_res.text}")
        sys.exit(1)
    evil_id = evil_res.json().get("source_id")

    uploads_dir = os.path.abspath(os.path.join(config.DATA_DIR, "uploads"))
    matching_files = glob.glob(os.path.join(uploads_dir, "*evil.mp4"))
    if not matching_files:
        print(f"[FAIL] Check 7: evil.mp4 not found inside {uploads_dir}")
        sys.exit(1)

    for mf in matching_files:
        abs_mf = os.path.abspath(mf)
        if not abs_mf.startswith(uploads_dir + os.sep):
            print(f"[FAIL] Check 7: File escaped data/uploads: {abs_mf}")
            sys.exit(1)

    # Wait for background worker to finish handling dummy file so file handle is released
    for _ in range(20):
        s_res = requests.get(f"{base_url}/sources/{evil_id}", headers=auth_headers)
        if s_res.status_code == 200 and s_res.json().get("status") in ("failed", "ready"):
            break
        time.sleep(0.2)

    # Cleanup the evil test source
    requests.delete(f"{base_url}/sources/{evil_id}", headers=auth_headers)
    print("[PASS] Check 7: Traversal upload '../../evil.mp4' safely stored inside data/uploads")

    # Check 8: Check that a clip request with start=nan returns 400
    nan_res = requests.get(f"{base_url}/sources/{source_id}/clip?start=nan&end=10", headers=auth_headers)
    if nan_res.status_code != 400 or "error" not in nan_res.json():
        print(f"[FAIL] Check 8: Clip start=nan expected 400, got {nan_res.status_code}: {nan_res.text}")
        sys.exit(1)
    print(f"[PASS] Check 8: Clip start=nan returned 400: {nan_res.json().get('error')}")

    # Check Live Endpoint (501 not implemented yet)
    live_res = requests.post(f"{base_url}/live/start", headers=auth_headers)
    if live_res.status_code != 501 or live_res.json() != {"error": "live not implemented yet"}:
        print(f"[FAIL] Live endpoint expected 501, got {live_res.status_code}: {live_res.text}")
        sys.exit(1)
    print(f"[PASS] Live endpoints return 501: {live_res.json().get('error')}")

    # Check 9: 401 behaviour when API_KEY is set (skip otherwise)
    if config.API_KEY:
        r_no_auth = requests.get(f"{base_url}/sources")
        if r_no_auth.status_code != 401 or r_no_auth.json().get("error") != "unauthorized":
            print(f"[FAIL] Check 9: Expected 401 without auth, got {r_no_auth.status_code}: {r_no_auth.text}")
            sys.exit(1)
        r_hdr = requests.get(f"{base_url}/sources", headers={"X-API-Key": config.API_KEY})
        if r_hdr.status_code != 200:
            print(f"[FAIL] Check 9: Expected 200 with X-API-Key header, got {r_hdr.status_code}")
            sys.exit(1)
        r_param = requests.get(f"{base_url}/sources?api_key={config.API_KEY}")
        if r_param.status_code != 200:
            print(f"[FAIL] Check 9: Expected 200 with api_key query param, got {r_param.status_code}")
            sys.exit(1)
        print("[PASS] Check 9: 401 unauthorized behavior verified")
    else:
        print("[SKIP] Check 9: 401 behavior check (API_KEY is not set)")

    # Check 10: DELETE the source
    del_res = requests.delete(f"{base_url}/sources/{source_id}", headers=auth_headers)
    if del_res.status_code != 200 or del_res.json() != {"deleted": True}:
        print(f"[FAIL] Check 10: DELETE /sources/{source_id} failed: status {del_res.status_code}, body {del_res.text}")
        sys.exit(1)

    verify_404 = requests.get(f"{base_url}/sources/{source_id}", headers=auth_headers)
    if verify_404.status_code != 404:
        print(f"[FAIL] Check 10: Source still exists after deletion! Status: {verify_404.status_code}")
        sys.exit(1)
    print("[PASS] Check 10: Source deleted and verified 404 on subsequent lookup")

    print("\nALL API CHECKS PASSED SUCCESSFULLY")


if __name__ == "__main__":
    main()
