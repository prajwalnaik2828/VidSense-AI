import os
import sys
import glob
import time
import json
import asyncio
import requests
import websockets
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vidsense import config
from vidsense import video_io


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


def fetch_mjpeg_frames(url: str, headers: dict, count: int = 3, timeout: float = 12.0) -> list[bytes]:
    frames = []
    r = requests.get(url, stream=True, headers=headers, timeout=timeout)
    buffer = b""
    start_time = time.time()
    try:
        for chunk in r.iter_content(chunk_size=4096):
            if not chunk:
                continue
            buffer += chunk
            while True:
                start_idx = buffer.find(b"\xff\xd8")
                if start_idx == -1:
                    break
                end_idx = buffer.find(b"\xff\xd9", start_idx)
                if end_idx == -1:
                    break
                frame_bytes = buffer[start_idx : end_idx + 2]
                frames.append(frame_bytes)
                buffer = buffer[end_idx + 2 :]
                if len(frames) >= count:
                    return frames
            if time.time() - start_time > timeout:
                break
    finally:
        r.close()
    return frames


async def receive_ws_events_async(ws_url: str, count: int = 10, timeout: float = 20.0) -> list[dict]:
    events = []
    async with websockets.connect(ws_url) as ws:
        start_time = time.time()
        while len(events) < count and time.time() - start_time < timeout:
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
                data = json.loads(msg)
                events.append(data)
                print(
                    f"  [WS Event #{len(events)}] seq={data.get('seq')}, "
                    f"ts={data.get('ts')}, active_tracks={data.get('stats', {}).get('active_tracks')}"
                )
            except asyncio.TimeoutError:
                break
    return events


async def run_push_ws_async(ws_url: str, video_path: str, duration_s: float = 15.0) -> tuple[int, list[dict]]:
    frames_sent = 0
    events_received = []

    async with websockets.connect(ws_url) as ws:
        async def sender():
            nonlocal frames_sent
            cap = cv2.VideoCapture(video_path)
            start_time = time.time()
            interval = 1.0 / config.LIVE_PROCESS_FPS
            try:
                while time.time() - start_time < duration_s:
                    ret, frame = cap.read()
                    if not ret:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    jpeg = video_io.encode_jpeg(frame)
                    await ws.send(jpeg)
                    frames_sent += 1
                    await asyncio.sleep(interval)
            finally:
                cap.release()

        async def receiver():
            while True:
                try:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    events_received.append(data)
                    print(
                        f"  [Push Event #{len(events_received)}] seq={data.get('seq')}, "
                        f"ts={data.get('ts')}, active_tracks={data.get('stats', {}).get('active_tracks')}"
                    )
                except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                    break
                except Exception:
                    break

        s_task = asyncio.create_task(sender())
        r_task = asyncio.create_task(receiver())

        await s_task
        await asyncio.sleep(0.5)
        r_task.cancel()
        try:
            await r_task
        except asyncio.CancelledError:
            pass

    return frames_sent, events_received


def test_server_mode(target_camera: str, base_url: str, ws_base_url: str, auth_headers: dict):
    print(f"\n=== Testing Live Server Mode with camera: {target_camera} ===")

    # Check 1: POST /live/start
    start_res = requests.post(
        f"{base_url}/live/start",
        json={"camera": target_camera, "name": "Live Server Mode Test"},
        headers=auth_headers,
    )
    if start_res.status_code != 201:
        print(f"[FAIL] Check 1: POST /live/start expected 201, got {start_res.status_code}: {start_res.text}")
        sys.exit(1)
    source_id = start_res.json().get("source_id")
    if not source_id:
        print(f"[FAIL] Check 1: No source_id returned in response: {start_res.text}")
        sys.exit(1)
    print(f"[PASS] Check 1: POST /live/start returned 201 (source_id: {source_id})")

    # Check 2: Receive 10 WebSocket events
    ws_url = f"{ws_base_url}/live/{source_id}/ws"
    if config.API_KEY:
        ws_url += f"?api_key={config.API_KEY}"

    print(f"\nConnecting to WebSocket: {ws_url} (waiting for 10 events)...")
    events = asyncio.run(receive_ws_events_async(ws_url, count=10, timeout=20.0))
    if len(events) < 10:
        print(f"[FAIL] Check 2: Expected 10 events, received {len(events)}")
        sys.exit(1)

    for ev in events:
        for k in ("seq", "ts", "detections", "stats", "alerts"):
            if k not in ev:
                print(f"[FAIL] Check 2: Missing key '{k}' in event: {ev}")
                sys.exit(1)

    if events[-1]["seq"] <= events[0]["seq"]:
        print(f"[FAIL] Check 2: Event seq values did not increase: {events[0]['seq']} -> {events[-1]['seq']}")
        sys.exit(1)
    print(f"[PASS] Check 2: Received 10 WebSocket events (seq: {events[0]['seq']} -> {events[-1]['seq']})")

    # Check 3: Fetch 3 MJPEG frames from GET /live/{source_id}/stream
    stream_url = f"{base_url}/live/{source_id}/stream"
    print(f"\nFetching 3 MJPEG frames from {stream_url}...")
    mjpeg_frames = fetch_mjpeg_frames(stream_url, headers=auth_headers, count=3, timeout=12.0)
    if len(mjpeg_frames) < 3:
        print(f"[FAIL] Check 3: Expected 3 MJPEG frames, got {len(mjpeg_frames)}")
        sys.exit(1)

    for i, frame in enumerate(mjpeg_frames, start=1):
        if not (frame.startswith(b"\xff\xd8") and frame.endswith(b"\xff\xd9")):
            print(f"[FAIL] Check 3: Frame #{i} does not have valid JPEG magic bytes (len={len(frame)})")
            sys.exit(1)
    print(f"[PASS] Check 3: Fetched 3 MJPEG frames, each starting with JPEG magic bytes FFD8 (lengths: {[len(f) for f in mjpeg_frames]})")

    # Check 4: GET /live/{source_id}/stats
    stats_res = requests.get(f"{base_url}/live/{source_id}/stats", headers=auth_headers)
    if stats_res.status_code != 200:
        print(f"[FAIL] Check 4: GET /live/{source_id}/stats expected 200, got {stats_res.status_code}: {stats_res.text}")
        sys.exit(1)
    st = stats_res.json()
    if not st.get("running"):
        print(f"[FAIL] Check 4: stats 'running' is not True: {st}")
        sys.exit(1)
    print(f"[PASS] Check 4: GET /live/{source_id}/stats returned: {st}")

    # Check 5: POST /live/{source_id}/stop
    print(f"\nStopping live session {source_id}...")
    stop_res = requests.post(f"{base_url}/live/{source_id}/stop", headers=auth_headers)
    if stop_res.status_code != 200:
        print(f"[FAIL] Check 5: POST /live/{source_id}/stop expected 200, got {stop_res.status_code}: {stop_res.text}")
        sys.exit(1)
    source_out = stop_res.json()
    if source_out.get("status") != "ready":
        print(f"[FAIL] Check 5: Stopped source status is not 'ready': {source_out}")
        sys.exit(1)
    print(f"[PASS] Check 5: POST /live/{source_id}/stop returned SourceOut (status: ready, duration: {source_out.get('duration_s')} s, n_tracks: {source_out.get('n_tracks')})")

    # Check 6: POST /search over the saved session
    print("\nSearching over the saved session...")
    search_res = requests.post(
        f"{base_url}/search",
        json={"question": "How many people are there?", "source_id": source_id},
        headers=auth_headers,
    )
    if search_res.status_code != 200:
        print(f"[FAIL] Check 6: POST /search expected 200, got {search_res.status_code}: {search_res.text}")
        sys.exit(1)
    search_out = search_res.json()
    print(f"  Answer: {search_out.get('answer')}")
    print(f"  Found: {search_out.get('found')}, Results count: {len(search_out.get('results', []))}")
    print(f"[PASS] Check 6: POST /search executed successfully over saved session")

    # Clean up
    requests.delete(f"{base_url}/sources/{source_id}", headers=auth_headers)
    print(f"[PASS] Check 7: Session cleanup completed")

    print("\nALL SERVER MODE CHECKS PASSED SUCCESSFULLY")


def test_push_mode(video_path: str, base_url: str, ws_base_url: str, auth_headers: dict):
    print(f"\n=== Testing Live Push Mode with video: {video_path} ===")

    # Check 1: POST /live/start with camera="push"
    start_res = requests.post(
        f"{base_url}/live/start",
        json={"camera": "push", "name": "Live Push Mode Test"},
        headers=auth_headers,
    )
    if start_res.status_code != 201:
        print(f"[FAIL] Check 1: POST /live/start expected 201, got {start_res.status_code}: {start_res.text}")
        sys.exit(1)
    source_id = start_res.json().get("source_id")
    if not source_id:
        print(f"[FAIL] Check 1: No source_id returned in response: {start_res.text}")
        sys.exit(1)
    print(f"[PASS] Check 1: POST /live/start returned 201 (source_id: {source_id})")

    # Check 2: Send JPEG frames at 5/s for 15s over WebSocket while printing events
    ws_url = f"{ws_base_url}/live/{source_id}/ws"
    if config.API_KEY:
        ws_url += f"?api_key={config.API_KEY}"

    print(f"\nConnecting to WebSocket: {ws_url} (pushing frames for 15 s)...")
    frames_sent, events = asyncio.run(run_push_ws_async(ws_url, video_path, duration_s=15.0))
    if frames_sent == 0:
        print("[FAIL] Check 2: No frames were sent")
        sys.exit(1)
    if len(events) == 0:
        print("[FAIL] Check 2: No events were received during push")
        sys.exit(1)
    print(f"[PASS] Check 2: Sent {frames_sent} JPEG frames over WS, received {len(events)} events (last seq: {events[-1].get('seq')})")

    # Check 3: GET /live/{source_id}/stats
    stats_res = requests.get(f"{base_url}/live/{source_id}/stats", headers=auth_headers)
    if stats_res.status_code != 200:
        print(f"[FAIL] Check 3: GET /live/{source_id}/stats expected 200, got {stats_res.status_code}: {stats_res.text}")
        sys.exit(1)
    st = stats_res.json()
    print(f"[PASS] Check 3: GET /live/{source_id}/stats returned: {st}")

    # Check 4: POST /live/{source_id}/stop
    print(f"\nStopping live session {source_id}...")
    stop_res = requests.post(f"{base_url}/live/{source_id}/stop", headers=auth_headers)
    if stop_res.status_code != 200:
        print(f"[FAIL] Check 4: POST /live/{source_id}/stop expected 200, got {stop_res.status_code}: {stop_res.text}")
        sys.exit(1)
    source_out = stop_res.json()
    if source_out.get("status") != "ready":
        print(f"[FAIL] Check 4: Stopped source status is not 'ready': {source_out}")
        sys.exit(1)
    print(f"[PASS] Check 4: POST /live/{source_id}/stop returned SourceOut (status: ready, duration: {source_out.get('duration_s')} s, n_tracks: {source_out.get('n_tracks')})")

    # Check 5: POST /search over the saved session
    print("\nSearching over the saved session...")
    search_res = requests.post(
        f"{base_url}/search",
        json={"question": "How many people are there?", "source_id": source_id},
        headers=auth_headers,
    )
    if search_res.status_code != 200:
        print(f"[FAIL] Check 5: POST /search expected 200, got {search_res.status_code}: {search_res.text}")
        sys.exit(1)
    search_out = search_res.json()
    print(f"  Answer: {search_out.get('answer')}")
    print(f"  Found: {search_out.get('found')}, Results count: {len(search_out.get('results', []))}")
    print(f"[PASS] Check 5: POST /search executed successfully over saved session")

    # Clean up
    requests.delete(f"{base_url}/sources/{source_id}", headers=auth_headers)
    print(f"[PASS] Check 6: Session cleanup completed")

    print("\nALL PUSH MODE CHECKS PASSED SUCCESSFULLY")


def main():
    base_url = f"http://127.0.0.1:{config.API_PORT}"
    ws_base_url = f"ws://127.0.0.1:{config.API_PORT}"

    print(f"Connecting to VidSense server at {base_url}...")
    if not wait_for_server(base_url):
        print(f"[FAIL] Server at {base_url} is not responding on /health.")
        sys.exit(1)
    print("Server is healthy.\n")

    auth_headers = {}
    if config.API_KEY:
        auth_headers["X-API-Key"] = config.API_KEY

    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "server"

    if mode == "server":
        target_video = sys.argv[2] if len(sys.argv) > 2 else get_first_test_video()
        if not target_video:
            print("STOP: No video found in tests/videos/. Please add test videos before running.")
            sys.exit(1)
        test_server_mode(target_video, base_url, ws_base_url, auth_headers)

    elif mode == "push":
        target_video = get_first_test_video()
        if not target_video:
            print("STOP: No video found in tests/videos/. Please add test videos before running.")
            sys.exit(1)
        test_push_mode(target_video, base_url, ws_base_url, auth_headers)

    elif mode == "all":
        target_video = get_first_test_video()
        if not target_video:
            print("STOP: No video found in tests/videos/. Please add test videos before running.")
            sys.exit(1)
        test_server_mode(target_video, base_url, ws_base_url, auth_headers)
        test_push_mode(target_video, base_url, ws_base_url, auth_headers)

    else:
        print(f"Unknown mode '{mode}'. Usage: python scripts/test_api_live.py [server [camera_or_video_path] | push | all]")
        sys.exit(1)


if __name__ == "__main__":
    main()
