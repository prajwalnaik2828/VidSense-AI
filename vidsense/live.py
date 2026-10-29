import os
import math
import time
import logging
import threading
import cv2
import numpy as np

from vidsense import config
from vidsense import db
from vidsense import detect
from vidsense import caption
from vidsense import index
from vidsense import video_io

logger = logging.getLogger("vidsense.live")


class LiveSession:
    def __init__(self, camera: str | int, name: str = "Live session"):
        self.camera = camera
        self.name = name

        # Determine camera mode
        if isinstance(camera, str) and camera == "push":
            self._mode = "push"
        elif isinstance(camera, int) or (isinstance(camera, str) and str(camera).isdigit()):
            self._mode = "webcam"
            self._cam_index = int(camera)
        elif isinstance(camera, str) and os.path.exists(camera):
            self._mode = "file"
            self._file_path = camera
        else:
            self._mode = "stream"
            self._stream_url = str(camera)

        self.source_id: str | None = None
        self._running: bool = False
        self._stopped: bool = False
        self._lock = threading.Lock()

        self._model = None
        self._aggregator: detect.TrackAggregator | None = None
        self._writer: cv2.VideoWriter | None = None
        self._raw_path: str = ""
        self._final_path: str = ""

        self._written_frames: int = 0
        self._processed_frames: int = 0
        self._start_time: float = 0.0
        self._seq: int = 0
        self._latest_jpeg: bytes | None = None
        self._latest_event: dict | None = None
        self._latest_detections: list[dict] = []

        self._caption_count: int = 0
        self._caption_busy: bool = False
        self._caption_job: tuple[float, np.ndarray] | None = None
        self._caption_cond = threading.Condition(self._lock)
        self._caption_thread: threading.Thread | None = None

        self._capture_thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._first_frame: np.ndarray | None = None

        self._last_process_time: float = 0.0
        self._last_caption_ts: float = -float("inf")
        self._last_flush_ts: float = -float("inf")

    def start(self) -> str:
        db.init_db()

        if self._mode != "push":
            if self._mode == "webcam":
                cap = cv2.VideoCapture(self._cam_index)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
            elif self._mode == "file":
                cap = cv2.VideoCapture(self._file_path)
            else:
                cap = cv2.VideoCapture(self._stream_url)

            if not cap.isOpened():
                raise RuntimeError("Could not open camera")

            ret, frame = cap.read()
            if not ret or frame is None:
                cap.release()
                raise RuntimeError("Could not open camera")

            self._cap = cap
            self._first_frame = frame

        self._model = detect.create_model()
        self.source_id = db.create_source(
            name=self.name,
            kind="live",
            path="",
            fps=config.LIVE_PROCESS_FPS,
            duration_s=0.0,
            status="live",
        )
        self._aggregator = detect.TrackAggregator(self.source_id)

        live_dir = os.path.join(config.DATA_DIR, "live")
        os.makedirs(live_dir, exist_ok=True)
        self._raw_path = os.path.join(live_dir, f"{self.source_id}_raw.mp4")
        self._final_path = os.path.join(live_dir, f"{self.source_id}.mp4")

        self._running = True
        self._start_time = time.time()
        self._last_process_time = 0.0
        self._last_caption_ts = -float("inf")
        self._last_flush_ts = -float("inf")

        # Start caption worker thread
        self._caption_thread = threading.Thread(
            target=self._caption_worker,
            daemon=True,
            name=f"LiveCaption-{self.source_id}",
        )
        self._caption_thread.start()

        # Start capture thread if in capture modes
        if self._mode != "push":
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name=f"LiveCapture-{self.source_id}",
            )
            self._capture_thread.start()

        return self.source_id

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def push_frame(self, jpeg: bytes) -> bool:
        if self._mode != "push":
            raise RuntimeError("push_frame is only allowed in push mode")

        if not self.is_running():
            return False

        now = time.perf_counter()
        min_interval = (1.0 / config.LIVE_PROCESS_FPS) - 0.01

        with self._lock:
            if self._last_process_time > 0 and (now - self._last_process_time) < min_interval:
                return False
            self._last_process_time = now

        frame = video_io.decode_jpeg(jpeg)
        self._process_frame(frame)
        return True

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def latest_event(self) -> dict | None:
        with self._lock:
            return self._latest_event

    def stats(self) -> dict:
        with self._lock:
            return self._get_stats_locked(time.time())

    def _get_stats_locked(self, now: float) -> dict:
        elapsed = max(0.001, now - self._start_time)
        fps = round(self._processed_frames / elapsed, 2)
        active_tracks = len(self._latest_detections)
        return {
            "fps": fps,
            "active_tracks": active_tracks,
            "elapsed_s": round(elapsed, 2),
            "captions": self._caption_count,
        }

    def _process_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return

        # Initialize VideoWriter on first processed frame
        if self._writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                self._raw_path,
                fourcc,
                float(config.LIVE_PROCESS_FPS),
                (w, h),
            )

        # 1. Write RAW unannotated frame
        self._writer.write(frame)
        ts = float(self._written_frames / config.LIVE_PROCESS_FPS)
        self._written_frames += 1

        # 2. Tracking and detection
        res = detect.track_frame(self._model, frame)
        detections = self._aggregator.update(ts, frame, res)
        annotated = detect.draw_detections(frame, detections)
        jpeg_bytes = video_io.encode_jpeg(annotated, quality=70)

        # 3. Store latest annotated JPEG & event under lock
        with self._lock:
            self._processed_frames += 1
            self._seq += 1
            self._latest_jpeg = jpeg_bytes
            self._latest_detections = detections
            stats = self._get_stats_locked(time.time())
            self._latest_event = {
                "seq": self._seq,
                "ts": ts,
                "detections": detections,
                "stats": stats,
            }

        # 4. Caption worker check (every LIVE_CAPTION_INTERVAL_S of video time)
        if (ts - self._last_caption_ts) >= config.LIVE_CAPTION_INTERVAL_S:
            self._last_caption_ts = ts
            with self._lock:
                if self._caption_job is None and not self._caption_busy:
                    self._caption_job = (ts, frame.copy())
                    self._caption_cond.notify()

        # 5. Track flush check (every TRACK_FLUSH_INTERVAL_S of video time)
        if (ts - self._last_flush_ts) >= config.TRACK_FLUSH_INTERVAL_S:
            self._last_flush_ts = ts
            for trk in self._aggregator.tracks():
                db.upsert_track(self.source_id, trk)

    def _capture_loop(self) -> None:
        native_fps = 0.0
        if self._mode == "file" and self._cap is not None:
            native_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if native_fps <= 0 or math.isnan(native_fps):
            native_fps = 30.0

        native_interval = 1.0 / native_fps
        next_frame_time = time.perf_counter()
        min_process_interval = (1.0 / config.LIVE_PROCESS_FPS) - 0.01

        while True:
            with self._lock:
                if not self._running:
                    break

            if self._first_frame is not None:
                frame = self._first_frame
                self._first_frame = None
                self._last_process_time = time.perf_counter()
                self._process_frame(frame)
                next_frame_time = time.perf_counter() + native_interval
                continue

            # Simulated live mode: pace at the file's native fps
            if self._mode == "file":
                now = time.perf_counter()
                delay = next_frame_time - now
                if delay > 0:
                    time.sleep(delay)
                elif delay < -1.0:
                    next_frame_time = time.perf_counter()
                next_frame_time += native_interval

            ret, frame = self._cap.read()
            if not ret or frame is None:
                # Stream ended or failed mid-session
                with self._lock:
                    self._running = False
                break

            # CPU rule: process at most LIVE_PROCESS_FPS frames per real second and DROP the rest
            now = time.perf_counter()
            if self._last_process_time > 0 and (now - self._last_process_time) < min_process_interval:
                continue

            self._last_process_time = now
            self._process_frame(frame)

    def _caption_worker(self) -> None:
        while True:
            with self._lock:
                while self._running and self._caption_job is None:
                    self._caption_cond.wait(timeout=0.2)

                if not self._running and self._caption_job is None:
                    break

                ts, frame = self._caption_job
                self._caption_job = None
                self._caption_busy = True

            try:
                caption_text = caption.caption_frame(frame)
                if caption_text:
                    index.add_caption(self.source_id, ts, caption_text)
                    with self._lock:
                        self._caption_count += 1
            except Exception as e:
                logger.error("Live caption worker exception: %s", e)
            finally:
                with self._lock:
                    self._caption_busy = False

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._running = False
            self._caption_cond.notify_all()

        # 1. Release capture source
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        # 2. Join capture thread
        if self._capture_thread is not None and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=10.0)

        # 3. Join caption thread
        if self._caption_thread is not None and self._caption_thread.is_alive():
            self._caption_thread.join(timeout=10.0)

        # 4. Release video writer
        if self._writer is not None:
            try:
                self._writer.release()
            except Exception:
                pass
            self._writer = None

        # 5. Upsert final tracks
        if self._aggregator is not None:
            try:
                for trk in self._aggregator.tracks():
                    db.upsert_track(self.source_id, trk)
            except Exception as e:
                logger.error("Error upserting final tracks: %s", e)

        # 6. Re-encode raw video to final mp4
        duration_s = (
            float(self._written_frames / config.LIVE_PROCESS_FPS)
            if config.LIVE_PROCESS_FPS > 0
            else 0.0
        )
        stored_path = ""

        if os.path.exists(self._raw_path) and self._written_frames > 0:
            try:
                video_io.reencode_h264(self._raw_path, self._final_path)
                if os.path.exists(self._raw_path):
                    try:
                        os.remove(self._raw_path)
                    except Exception:
                        pass
                stored_path = f"data/live/{self.source_id}.mp4"
            except Exception as e:
                logger.error("Failed to re-encode live raw video: %s", e)
                stored_path = ""
        else:
            if os.path.exists(self._raw_path):
                try:
                    os.remove(self._raw_path)
                except Exception:
                    pass
            stored_path = ""

        # 7. Update source in DB
        try:
            db.update_source(
                self.source_id,
                path=stored_path,
                duration_s=duration_s,
                fps=float(config.LIVE_PROCESS_FPS),
                status="ready",
            )
        except Exception as e:
            logger.error("Error updating source status: %s", e)
