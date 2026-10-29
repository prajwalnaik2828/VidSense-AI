import os
import re
import math
import json
import secrets
import logging
import asyncio
import anyio
import threading
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request, UploadFile, File, Query, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.websockets import WebSocketDisconnect
from pydantic import BaseModel

from vidsense import config
from vidsense import db
from vidsense import index
from vidsense import query
from vidsense import video_io
from vidsense import detect
from vidsense.live import LiveSession

logger = logging.getLogger("vidsense.api")

_executor = ThreadPoolExecutor(max_workers=1)
_progress: dict[str, float] = {}

_current_live_session: LiveSession | None = None
_last_live_session: LiveSession | None = None
_live_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VidSense server starting. Server must run as exactly ONE process (workers=1).")
    db.init_db()

    # Preload YOLO model
    try:
        detect.create_model()
    except Exception as e:
        logger.warning("Could not preload YOLO model: %s", e)

    # Preload embedding model
    try:
        index._get_embedder()
    except Exception as e:
        logger.warning("Could not preload embedding model: %s", e)

    # Startup cleanup:
    # every source with status "processing" becomes "failed"
    # every source with status "live" becomes "ready" with path data/live/<id>.mp4 if that file exists, otherwise "failed"
    try:
        sources = db.list_sources()
        for s in sources:
            sid = s["source_id"]
            st = s.get("status")
            if st == "processing":
                db.update_source(sid, status="failed")
            elif st == "live":
                live_file = os.path.join(config.DATA_DIR, "live", f"{sid}.mp4")
                if os.path.exists(live_file):
                    db.update_source(sid, status="ready", path=f"data/live/{sid}.mp4")
                else:
                    db.update_source(sid, status="failed")
    except Exception as e:
        logger.exception("Startup cleanup failed: %s", e)

    yield

    # Clean shutdown of running live session if any:
    with _live_lock:
        if _current_live_session is not None and _current_live_session.is_running():
            try:
                _current_live_session.stop()
            except Exception as e:
                logger.error("Error stopping live session on shutdown: %s", e)

    _executor.shutdown(wait=False)


app = FastAPI(lifespan=lifespan)


# ---------------- Error Handlers ----------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        msg = detail["error"]
    else:
        msg = str(detail)
    return JSONResponse(status_code=exc.status_code, content={"error": msg})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = exc.errors()
    msg = errors[0].get("msg", "Validation error") if errors else "Validation error"
    loc = errors[0].get("loc", []) if errors else []
    field = loc[-1] if loc else ""
    error_str = f"Invalid {field}: {msg}" if field else msg
    return JSONResponse(status_code=400, content={"error": error_str})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server exception: %s", exc)
    return JSONResponse(status_code=500, content={"error": str(exc) or "Internal server error"})


# ---------------- Auth & CORS ----------------
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path.rstrip("/")
    # Allow public endpoints and frontend UI assets
    if (
        path == "/health"
        or path == ""
        or path == "/favicon.ico"
        or request.url.path.startswith("/static")
        or path == "/docs"
        or path == "/openapi.json"
    ) and request.method == "GET":
        return await call_next(request)

    if config.API_KEY:
        header_key = request.headers.get("x-api-key")
        query_key = request.query_params.get("api_key")
        if header_key != config.API_KEY and query_key != config.API_KEY:
            return JSONResponse(status_code=401, content={"error": "unauthorized"})

    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS if config.CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Models ----------------
class SourceOut(BaseModel):
    source_id: str
    name: str
    kind: str
    status: str
    duration_s: float | None = None
    fps: float | None = None
    created_at: float
    progress: float
    n_tracks: int
    class_counts: dict[str, int]
    n_captions: int
    has_clip: bool


class SearchResultItem(BaseModel):
    source_id: str
    source_name: str
    kind: str
    ts_start: float
    ts_end: float
    label: str
    thumbnail_url: str | None = None
    clip_url: str | None = None


class SearchOut(BaseModel):
    answer: str
    found: bool
    results: list[SearchResultItem]


class SearchIn(BaseModel):
    question: str
    source_id: str | None = None


def _build_source_out(src: dict) -> dict:
    source_id = src["source_id"]
    status = src.get("status", "")
    path = src.get("path") or ""
    has_clip = bool(status == "ready" and bool(path.strip()))

    if status == "ready":
        progress = 1.0
    else:
        progress = float(_progress.get(source_id, 0.0))

    class_counts = db.count_tracks_by_class(source_id)
    n_tracks = sum(class_counts.values())
    n_captions = db.count_captions(source_id)

    return {
        "source_id": source_id,
        "name": src.get("name", ""),
        "kind": src.get("kind", ""),
        "status": status,
        "duration_s": src.get("duration_s"),
        "fps": src.get("fps"),
        "created_at": float(src.get("created_at", 0.0)),
        "progress": progress,
        "n_tracks": n_tracks,
        "class_counts": class_counts,
        "n_captions": n_captions,
        "has_clip": has_clip,
    }


# ---------------- Endpoints ----------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sources/upload", status_code=202)
def upload_source(file: UploadFile = File(...)):
    if not file or not file.filename:
        return JSONResponse(status_code=400, content={"error": "File and filename required"})

    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in [".mp4", ".mov", ".avi", ".mkv"]:
        return JSONResponse(status_code=400, content={"error": f"Invalid video file extension: {ext}"})

    # Sanitize filename: os.path.basename, keep only [A-Za-z0-9._-], max 100 chars
    raw_base = os.path.basename(file.filename.replace("\\", "/"))
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_base)[:100]
    if not safe_name:
        safe_name = f"upload{ext.lower()}"

    hex_prefix = secrets.token_hex(4)
    safe_filename = f"{hex_prefix}_{safe_name}"

    uploads_dir = os.path.abspath(os.path.join(config.DATA_DIR, "uploads"))
    os.makedirs(uploads_dir, exist_ok=True)
    dest_path = os.path.abspath(os.path.join(uploads_dir, safe_filename))

    # Verify resolved path is strictly inside data/uploads
    if not (dest_path.startswith(uploads_dir + os.sep) or dest_path == uploads_dir):
        return JSONResponse(status_code=400, content={"error": "Invalid upload destination path"})

    # Stream upload in chunks (never read fully into memory); cap at MAX_UPLOAD_MB
    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    total_bytes = 0
    try:
        with open(dest_path, "wb") as f_out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    f_out.close()
                    try:
                        os.remove(dest_path)
                    except OSError:
                        pass
                    return JSONResponse(
                        status_code=413,
                        content={"error": f"File exceeds maximum upload size of {config.MAX_UPLOAD_MB} MB"},
                    )
                f_out.write(chunk)
    finally:
        file.file.close()

    stored_path = f"data/uploads/{safe_filename}"
    source_id = db.create_source(
        name=safe_name,
        kind="file",
        path=stored_path,
        fps=None,
        duration_s=None,
        status="processing",
    )

    _progress[source_id] = 0.0

    def _run_ingest(sid: str, video_path: str, video_name: str):
        def progress_cb(fraction: float):
            _progress[sid] = float(fraction)

        try:
            index.ingest_video_file(
                path=video_path,
                name=video_name,
                progress_cb=progress_cb,
                source_id=sid,
            )
            _progress[sid] = 1.0
        except Exception as e:
            logger.exception("Ingestion failed for source %s: %s", sid, e)
            db.update_source(sid, status="failed")

    _executor.submit(_run_ingest, source_id, dest_path, safe_name)

    return JSONResponse(status_code=202, content={"source_id": source_id, "status": "processing"})


@app.get("/sources", response_model=list[SourceOut])
def list_sources_endpoint():
    sources = db.list_sources()
    return [_build_source_out(s) for s in sources]


@app.get("/sources/{source_id}", response_model=SourceOut)
def get_source_endpoint(source_id: str):
    src = db.get_source(source_id)
    if not src:
        return JSONResponse(status_code=404, content={"error": f"Source {source_id} not found"})
    return _build_source_out(src)


@app.delete("/sources/{source_id}")
def delete_source_endpoint(source_id: str):
    src = db.get_source(source_id)
    if not src:
        return JSONResponse(status_code=404, content={"error": f"Source {source_id} not found"})
    if src.get("status") == "live":
        return JSONResponse(status_code=409, content={"error": "Cannot delete a running live session"})

    index.delete_source(source_id)
    _progress.pop(source_id, None)
    return {"deleted": True}


@app.post("/search", response_model=SearchOut)
def search_endpoint(body: SearchIn):
    if not body.question or not body.question.strip() or len(body.question) > 500:
        return JSONResponse(status_code=400, content={"error": "Question must be between 1 and 500 characters"})

    if body.source_id is not None:
        src = db.get_source(body.source_id)
        if not src:
            return JSONResponse(status_code=404, content={"error": f"Source {body.source_id} not found"})

    try:
        res = query.answer_query(question=body.question.strip(), source_id=body.source_id)
    except ValueError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        logger.exception("Search query error: %s", e)
        return JSONResponse(status_code=500, content={"error": f"Search failed: {str(e)}"})

    sources_cache = {}
    results_out = []
    for r in res.get("results", []):
        sid = r["source_id"]
        if sid not in sources_cache:
            sources_cache[sid] = db.get_source(sid)
        src_info = sources_cache[sid]

        live_raw = os.path.abspath(os.path.join(config.DATA_DIR, "live", f"{sid}_raw.mp4"))
        live_fin = os.path.abspath(os.path.join(config.DATA_DIR, "live", f"{sid}.mp4"))
        has_clip = bool(
            src_info
            and (
                (src_info.get("status") == "ready" and bool((src_info.get("path") or "").strip()))
                or os.path.exists(live_fin)
                or os.path.exists(live_raw)
            )
        )

        thumb = r.get("thumbnail")
        thumb_url = f"/media/frames/{thumb.lstrip('/')}" if thumb else None

        if has_clip:
            c_start = max(0.0, float(r["ts_start"]) - config.CLIP_PAD_S)
            c_end = max(float(r["ts_end"]) + config.CLIP_PAD_S, c_start + 1.0)
            c_start = round(c_start, 2)
            c_end = round(c_end, 2)
            clip_url = f"/sources/{sid}/clip?start={c_start:g}&end={c_end:g}"
        else:
            clip_url = None

        results_out.append({
            "source_id": sid,
            "source_name": r.get("source_name", ""),
            "kind": r.get("kind", ""),
            "ts_start": float(r["ts_start"]),
            "ts_end": float(r["ts_end"]),
            "label": r.get("label", ""),
            "thumbnail_url": thumb_url,
            "clip_url": clip_url,
        })

    return {
        "answer": res.get("answer", ""),
        "found": bool(res.get("found", False)),
        "results": results_out,
    }


@app.get("/sources/{source_id}/clip")
def get_clip_endpoint(
    source_id: str,
    start: float = Query(...),
    end: float = Query(...),
):
    src = db.get_source(source_id)
    if not src:
        return JSONResponse(status_code=404, content={"error": f"Source {source_id} not found"})

    status = src.get("status", "")
    src_path = src.get("path") or ""
    abs_src_path = os.path.abspath(src_path) if src_path else ""

    if not (abs_src_path and os.path.exists(abs_src_path)):
        live_raw = os.path.abspath(os.path.join(config.DATA_DIR, "live", f"{source_id}_raw.mp4"))
        live_fin = os.path.abspath(os.path.join(config.DATA_DIR, "live", f"{source_id}.mp4"))
        if os.path.exists(live_fin):
            abs_src_path = live_fin
        elif os.path.exists(live_raw):
            abs_src_path = live_raw
        else:
            return JSONResponse(status_code=409, content={"error": "clip not available"})

    if not math.isfinite(start) or not math.isfinite(end):
        return JSONResponse(status_code=400, content={"error": "start and end must be finite numbers"})

    if end <= start:
        return JSONResponse(status_code=400, content={"error": "end must be greater than start"})

    start_s = max(0.0, start)
    end_s = end
    if (end_s - start_s) > 120.0:
        end_s = start_s + 120.0

    clips_dir = os.path.abspath(os.path.join(config.DATA_DIR, "clips"))
    os.makedirs(clips_dir, exist_ok=True)
    out_filename = f"{source_id}_{int(start_s)}_{int(end_s)}.mp4"
    out_path = os.path.join(clips_dir, out_filename)

    try:
        video_io.cut_clip(src_path=abs_src_path, start_s=start_s, end_s=end_s, out_path=out_path)
    except Exception as e:
        logger.exception("Failed to cut clip: %s", e)
        return JSONResponse(status_code=500, content={"error": f"Failed to cut clip: {str(e)}"})

    if not os.path.exists(out_path):
        return JSONResponse(status_code=500, content={"error": "Cut clip failed to generate output file"})

    return FileResponse(out_path, media_type="video/mp4")


@app.get("/media/frames/{file_path:path}")
def get_media_frame(file_path: str):
    frames_dir = os.path.abspath(os.path.join(config.DATA_DIR, "frames"))
    target = os.path.abspath(os.path.join(frames_dir, file_path))
    if not (target == frames_dir or target.startswith(frames_dir + os.sep)):
        return JSONResponse(status_code=400, content={"error": "Access denied"})
    if not os.path.exists(target) or os.path.isdir(target):
        return JSONResponse(status_code=404, content={"error": "Frame not found"})
    return FileResponse(target)


# ---------------- Live Endpoints ----------------
class LiveStartIn(BaseModel):
    camera: str
    name: str | None = None


@app.post("/live/start", status_code=201)
def live_start(body: LiveStartIn):
    global _current_live_session, _last_live_session

    if not body.camera or not body.camera.strip():
        return JSONResponse(status_code=400, content={"error": "camera is required"})

    with _live_lock:
        if _current_live_session is not None and _current_live_session.is_running():
            return JSONResponse(status_code=409, content={"error": "A live session is already running"})

        name = body.name.strip() if body.name and body.name.strip() else "Live session"
        session = LiveSession(camera=body.camera.strip(), name=name)

        try:
            source_id = session.start()
        except RuntimeError:
            return JSONResponse(status_code=400, content={"error": "Could not open camera"})
        except Exception as e:
            logger.exception("Failed to start live session: %s", e)
            return JSONResponse(status_code=500, content={"error": str(e)})

        _current_live_session = session
        _last_live_session = session

        return JSONResponse(status_code=201, content={"source_id": source_id})


@app.post("/live/{source_id}/stop", response_model=SourceOut)
async def live_stop(source_id: str):
    global _current_live_session, _last_live_session

    session = None
    with _live_lock:
        if _current_live_session is not None and _current_live_session.source_id == source_id:
            session = _current_live_session

    if session is None:
        return JSONResponse(
            status_code=404,
            content={"error": f"Live session {source_id} is not the current session"},
        )

    # Run the (slow) stop in a thread so the event loop is not blocked
    await anyio.to_thread.run_sync(session.stop)

    with _live_lock:
        _last_live_session = session
        if _current_live_session is session:
            _current_live_session = None

    src = db.get_source(source_id)
    if not src:
        return JSONResponse(status_code=404, content={"error": f"Source {source_id} not found"})

    return _build_source_out(src)


@app.get("/live/{source_id}/stats")
def live_stats(source_id: str):
    session = None
    with _live_lock:
        if _current_live_session is not None and _current_live_session.source_id == source_id:
            session = _current_live_session
        elif _last_live_session is not None and _last_live_session.source_id == source_id:
            session = _last_live_session

    if session is None:
        return JSONResponse(status_code=404, content={"error": f"Live session {source_id} not found"})

    st = session.stats()
    st["running"] = session.is_running()
    return st


@app.get("/live/{source_id}/stream")
async def live_stream(source_id: str):
    session = None
    with _live_lock:
        if _current_live_session is not None and _current_live_session.source_id == source_id:
            session = _current_live_session

    if session is None or not session.is_running():
        return JSONResponse(status_code=404, content={"error": f"Active live session {source_id} not found"})

    async def mjpeg_generator():
        try:
            while session.is_running():
                jpeg = session.latest_jpeg()
                if jpeg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n"
                        + jpeg + b"\r\n"
                    )
                await asyncio.sleep(0.2)
        except (asyncio.CancelledError, Exception):
            pass

    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/live/{source_id}/ws")
async def live_ws(websocket: WebSocket, source_id: str):
    if config.API_KEY:
        header_key = websocket.headers.get("x-api-key")
        query_key = websocket.query_params.get("api_key")
        if header_key != config.API_KEY and query_key != config.API_KEY:
            await websocket.close(code=1008, reason="unauthorized")
            return

    session = None
    with _live_lock:
        if _current_live_session is not None and _current_live_session.source_id == source_id:
            session = _current_live_session

    if session is None:
        await websocket.close(code=1000, reason="session not found")
        return

    await websocket.accept()

    async def sender():
        last_sent_seq = -1
        while session.is_running():
            ev = session.latest_event()
            if ev is not None and ev.get("seq") != last_sent_seq:
                last_sent_seq = ev["seq"]
                payload = {
                    "seq": ev["seq"],
                    "ts": ev["ts"],
                    "detections": ev.get("detections", []),
                    "stats": ev.get("stats", {}),
                    "alerts": [],
                }
                await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.1)

    async def receiver():
        while session.is_running():
            try:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                data = message.get("bytes")
                if data and session._mode == "push":
                    try:
                        await anyio.to_thread.run_sync(session.push_frame, data)
                    except ValueError:
                        pass
            except (WebSocketDisconnect, asyncio.CancelledError):
                break
            except Exception:
                break

    send_task = asyncio.create_task(sender())
    recv_task = asyncio.create_task(receiver())

    done, pending = await asyncio.wait([send_task, recv_task], return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()

    try:
        await websocket.close(code=1000)
    except Exception:
        pass


# ---------------- Frontend Web UI Mounting ----------------
_web_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
if os.path.isdir(_web_dir):
    app.mount("/static", StaticFiles(directory=_web_dir), name="static")

    @app.get("/")
    def serve_frontend_ui():
        index_file = os.path.join(_web_dir, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file, media_type="text/html")
        return {"message": "VidSense AI API is running. UI not found."}


# ---------------- Upload Route Alias ----------------
@app.post("/upload", status_code=202)
def upload_alias(file: UploadFile = File(...)):
    return upload_source(file=file)

