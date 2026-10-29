# VidSense AI: PROJECT BRIEF (read this first, for every task)

## What you are building
A CPU-only backend that makes video searchable in plain English. A user uploads a video or connects a live camera. The system detects and tracks objects (YOLO), writes short scene captions (a vision-language model API), indexes everything, and answers questions such as "How many people are there?", "When does the first car appear?" or "Show me someone in a red jacket" with timestamps, thumbnails and playable clips. A separate UI, built by me and NOT part of this repo, talks to it through a FastAPI API. Live mode shows real-time detections and makes the live footage searchable after it stops.

## How it works
- Upload path: video -> frames sampled at FILE_PROCESS_FPS -> YOLO + ByteTrack (persistent object IDs) -> "tracks" (class, first/last seen time, best thumbnail) stored in SQLite. Every CAPTION_INTERVAL_S seconds one frame is captioned by a vision-language model and the caption is embedded into ChromaDB.
- Question path: an LLM parses the question into {intent, class, time range, descriptive text}. Counting and when/where questions are answered from the SQLite tracks (exact, no LLM, no hallucination). Descriptive questions search the caption embeddings, and an LLM answers ONLY from the retrieved captions, otherwise it replies with the NOT_FOUND message.
- Live path: a webcam, stream URL or video file read on the server, OR JPEG frames pushed by a client ("push" mode), go through the same detection and tracking. The system exposes an annotated MJPEG stream and WebSocket JSON events, records the raw stream, writes tracks and captions to the same index, and after the session stops it is searchable like any uploaded video.

## Modules (built one at a time, in this order)
M0 config and setup | M1 llm.py | M2 db.py | M3 video_io.py | M4 detect.py | M5 caption.py + index.py | M6 query.py | M7 api.py (files, search, security) | M8 live.py | M9 live endpoints in api.py | M10 README and evaluation
Dependency flow: config <- llm, db, video_io <- detect <- caption <- index <- query <- api. live.py uses detect, caption, index, db, video_io. api.py uses everything.

## Data model
- sources: one row per uploaded video or live session (source_id, name, kind 'file'|'live', path, fps, duration_s, status 'processing'|'ready'|'live'|'failed', created_at)
- tracks: one row per tracked object (source_id, track_id, cls, first_ts, last_ts, n_frames, max_conf, best_ts, thumb_path)
- captions: (source_id, ts, text); also embedded in ChromaDB
- Files: data/uploads, data/frames/<source_id>/<track_id>.jpg (thumbnails), data/clips, data/live
- thumb_path is stored RELATIVE to data/frames as "<source_id>/<track_id>.jpg", with forward slashes.

## Time model
File sources: ts = frame_index / video_fps. Live sources: ts = written_frame_count / LIVE_PROCESS_FPS (the video time of the saved recording, not wall-clock). All timestamps are float seconds.

## Key decisions
- CPU ONLY: no GPU or CUDA code; always device="cpu"; small model (yolo11n) at small image size; frames are sampled and dropped, never queued.
- Counts are tracker-based estimates. The answer text must say so.
- Never invent answers. When nothing supports an answer, return NOT_FOUND_MESSAGE.
- The repo contains NO UI code. The UI calls the API.
- Threads exist ONLY in live.py (one capture thread, one caption worker) and api.py (one ingestion worker thread, plus FastAPI's own request handling).
- The server must run as exactly ONE process (state is held in memory).

## Working rules (apply to every task)
1. Implement ONLY the module I name. Never modify other modules except to fix an import.
2. Names, signatures, return shapes, constants and file paths in my prompts are FIXED. If you think one must change, STOP and ask me.
3. All constants come from vidsense/config.py. Never hardcode values, API keys or model names. Never create or fill in .env.
4. Test videos are mine, in tests/videos/. If it is empty and a test needs one, STOP and ask. Never fake footage, mock models or stub a failing component to pass a test. (Unit tests of db.py may use hand-written rows in a temp database.)
5. Keep the code simple and minimal. No extra features, files, abstractions or libraries. The whole project must be finished within 7 hours.
6. Finish every task by running its verify command, pasting the REAL output, then a 5-line summary: files changed, how to run, what passed, what failed, known issues.

## Definition of done
Upload a video -> poll until ready -> ask questions -> get correct answers with timestamps, thumbnails and clips. Start a live session (file, webcam or push mode) -> see detections in real time -> stop -> search it. Unanswerable questions return the NOT_FOUND message.
