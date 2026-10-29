# VidSense AI: Video Search & Real-Time Detection Backend

A CPU-only AI backend that makes video searchable in natural plain English. Users can upload recorded video files or connect live cameras (webcam, RTSP stream, local video, or client-pushed JPEG streams). The system detects and tracks objects with persistent IDs, writes periodic scene captions via vision-language models, indexes all metadata, and answers natural language questions with timestamps, thumbnails, and playable video clips.

---

## 1. Purpose & Key Capabilities

- **Zero-Hallucination Query Routing**:
  - Counting queries (*"How many people are there?"*) and temporal queries (*"When does a car appear?"*) are answered directly from deterministic SQLite object tracking tables.
  - Descriptive queries (*"Show me someone in a red jacket"*) perform vector similarity search over scene caption embeddings in ChromaDB, synthesized strictly from grounded context. Unanswerable questions return a polite, explicit not-found message.
- **Real-Time Live Streaming & Ingestion**:
  - Ingests from USB webcams, RTSP/HTTP streams, local files, or browser/mobile client WebSockets ("push mode").
  - Produces an annotated MJPEG stream (~5 fps) and real-time JSON detection events over WebSockets.
  - Automatically records and indexes live sessions so they become fully searchable immediately upon stopping.
- **Pure CPU Optimization**:
  - Engineered specifically for CPU execution (`device="cpu"`), employing lightweight models (`yolo11n`), frame sampling, and asynchronous non-blocking architectures without GPU dependencies.
- **Headless Backend**:
  - Exposes a comprehensive FastAPI REST, MJPEG, and WebSocket API. The user interface connects externally over standard protocols.

---

## 2. Architecture

```
                       +-----------------------------------------------+
                       |                 Client UI                     |
                       |  (Web / Desktop / Mobile / Push Stream Client)|
                       +-----------------------+-----------------------+
                                               |
                          REST / MJPEG Stream  |  WebSocket (Events / Push Frames)
                                               v
+-----------------------------------------------------------------------------------------+
|                                    FastAPI Backend                                      |
|                                (Single-Process Server)                                  |
|                                                                                         |
|  +--------------------+   +---------------------+   +--------------------------------+  |
|  |   Upload Handler   |   |   LiveSession Mgr   |   |        Query Engine            |  |
|  |  (File Ingestion)  |   | (Capture/Push/MJPEG)|   |    (LLM Query Decomposition)   |  |
|  +---------+----------+   +----------+----------+   +---------------+----------------+  |
|            |                         |                              |                   |
|            v                         v                              |                   |
|  +----------------------------------------------+                   |                   |
|  |            Computer Vision Core              |                   |                   |
|  |  - YOLO11n Object Detection (CPU)            |                   |                   |
|  |  - ByteTrack Multi-Object Tracking           |                   |                   |
|  |  - Keyframe Extraction & Cropped Thumbnails  |                   |                   |
|  |  - Gemini Vision Scene Captioning            |                   |                   |
|  +---------+-------------------------+----------+                   |                   |
|            |                         |                              |                   |
+------------|-------------------------|------------------------------|-------------------+
             v                         v                              v
   +--------------------+    +--------------------+       +-----------------------+
   |  SQLite Database   |    |  ChromaDB Vectors  |       |   Filesystem Storage  |
   |  - sources table   |    |  - caption vectors |       |   - uploads/ (raw)    |
   |  - tracks table    |    |  - source_id meta  |       |   - frames/ (thumbs)  |
   |  - captions table  |    +--------------------+       |   - live/ (recordings)|
   +--------------------+                                 |   - clips/ (cuts)     |
                                                          +-----------------------+
```

---

## 3. CPU-Only Setup & Installation

### Prerequisites
- **Python**: 3.10, 3.11, or 3.12
- **FFmpeg**: Must be installed on your system and accessible via the system `PATH` (used for video probing, clip cutting, and H.264 re-encoding).

### 1. Create and Activate Virtual Environment
```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux / macOS:
source .venv/bin/activate
```

### 2. Install CPU-Only PyTorch
Install PyTorch and Torchvision built specifically for CPU (preventing massive CUDA binaries from being pulled):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Requirements
Install the remaining backend dependencies:
```bash
pip install -r requirements.txt
```
*(Note: `torch` and `torchvision` are intentionally omitted from `requirements.txt` to avoid overriding the CPU-only wheel).*

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```env
# Gemini API Key (Required for scene captions and semantic question answering)
GEMINI_API_KEY=your_gemini_api_key_here

# API Security Key (Required for authenticated endpoints)
API_KEY=your_secret_api_key_here

# Server Host & Port
API_HOST=0.0.0.0
API_PORT=8000

# Allowed CORS Origins (comma-separated or *)
CORS_ORIGINS=*
```

### 5. Verify Setup
Run the diagnostic setup verification script:
```bash
python scripts/check_setup.py
```

---

## 4. Running the Application

### Option A: Launch as a Native Desktop App (Recommended)
Run `main.py` to start the backend and open VidSense in an isolated, dedicated desktop application window (with no browser URL bar or tabs):

```bash
python main.py
```
*(When you close the desktop app window, the background backend server shuts down automatically).*

### Option B: Headless Server Mode
If you prefer running only the server in your terminal and accessing it via any browser:

```bash
python run_server.py
```
Then navigate to **http://localhost:8000/** in your browser.

> **CRITICAL ARCHITECTURAL REQUIREMENT**:  
> The server **MUST run as exactly ONE process** (`workers=1`).  
> Video ingestion, live camera tracking, and WebSocket synchronization maintain state in memory. Running multiple uvicorn workers will result in race conditions, split live sessions, and database lock contention.

---

## 5. API Reference

All requests (except `/health` and static media when authorized via query parameter) require authentication. Pass the API key either via the `X-API-Key` HTTP header or as the `?api_key=` URL query parameter.

| Method | Path | Auth | Description | Request Body / Params | Response |
| :--- | :--- | :---: | :--- | :--- | :--- |
| `GET` | `/health` | No | Server status & health | None | 200 `{"status": "ok"}` |
| `POST` | `/sources/upload` | Yes | Upload video file for background indexing | Multipart form (`file`) | 202 `{"source_id": str, "status": "processing"}` |
| `GET` | `/sources` | Yes | List all registered video and live sources | None | 200 `list[SourceOut]` |
| `GET` | `/sources/{source_id}` | Yes | Get metadata, status, and progress of a source | None | 200 `SourceOut` |
| `DELETE`| `/sources/{source_id}` | Yes | Delete source, thumbnails, clips, and DB records | None | 200 `{"deleted": true}` |
| `POST` | `/search` | Yes | Natural language query across one or all videos | JSON `{"question": str, "source_id": str\|null}` | 200 `SearchOut` (`answer`, `results`, `found`) |
| `GET` | `/media/frames/{source_id}/{track_id}.jpg` | Yes | Fetch high-confidence JPEG thumbnail for an object track | Query param `?api_key=...` or Header | 200 `image/jpeg` |
| `GET` | `/sources/{source_id}/clip` | Yes | Cut and stream an MP4 video clip around a timestamp | Query params `start`, `end`, `api_key` | 200 `video/mp4` |
| `POST` | `/live/start` | Yes | Start a live session (device index, RTSP URL, file, or "push") | JSON `{"camera": str, "name": str\|null}` | 201 `{"source_id": str}` (409 if active) |
| `POST` | `/live/{source_id}/stop`| Yes | Stop active live session, finalize recording & index | None | 200 `SourceOut` |
| `GET` | `/live/{source_id}/stats`| Yes | Get real-time live session metrics | None | 200 `{fps, active_tracks, elapsed_s, captions, running}` |
| `GET` | `/live/{source_id}/stream`| Yes | Live annotated MJPEG video stream (~5 fps) | Query param `?api_key=...` or Header | 200 `multipart/x-mixed-replace; boundary=frame` |
| `WS` | `/live/{source_id}/ws` | Yes | WebSocket: push client frames (binary) / receive live events | Query param `?api_key=...` | JSON events / Binary JPEG frames |

---

## 6. How a Frontend UI Connects

A custom web or desktop UI connects to VidSense using standard HTTP, MJPEG, and WebSocket interfaces:

### REST (Sources, Video Management, and Search)
- **Upload**: Issue a `POST /sources/upload` with `FormData` containing the video file.
- **Poll Status**: Poll `GET /sources/{source_id}` every 1–2 seconds until `status == "ready"`.
- **Search**: Send `POST /search` with `{"question": "How many cars passed between 00:10 and 00:30?", "source_id": "..."}`.
  - Display the generated text answer (`answer`).
  - Display matched tracks with thumbnails (`/media/frames/{source_id}/{track_id}.jpg?api_key=...`).
  - Play back relevant video segments using the clip endpoint (`/sources/{source_id}/clip?start=10.0&end=30.0&api_key=...`).

### Live Feed Display
- **MJPEG Stream**: Native browser `<img>` tags can display the live video feed directly without WebRTC or custom player libraries:
  ```html
  <img src="http://localhost:8000/live/<source_id>/stream?api_key=<API_KEY>" alt="Live Detection Stream" />
  ```
- **Real-Time Detection Overlay & Alerts**: Connect to `ws://localhost:8000/live/<source_id>/ws?api_key=<API_KEY>` to receive real-time JSON packets (~10 Hz):
  ```json
  {
    "seq": 42,
    "ts": 8.4,
    "detections": [{"cls": "person", "conf": 0.88, "box": [120, 45, 310, 420], "track_id": 3}],
    "stats": {"fps": 4.9, "active_tracks": 1, "elapsed_s": 8.4, "captions": 0},
    "alerts": []
  }
  ```

### Push Mode (Mobile Phones & Browser Webcams)
When the backend cannot directly access a client's camera (e.g. smartphone or remote browser):
1. Call `POST /live/start` with body `{"camera": "push", "name": "Phone Camera"}` to obtain a `source_id`.
2. Connect a WebSocket to `ws://localhost:8000/live/<source_id>/ws?api_key=<API_KEY>`.
3. Capture frames in the client canvas, resize them to approximately **480 px wide** (JPEG quality ~70%), and send the raw binary JPEG bytes over the WebSocket at ~5 frames per second.
4. The backend processes each frame through YOLO and ByteTrack, updates live statistics, records the raw stream to disk, and pushes detection metadata back over the same WebSocket connection.

---

## 7. Deployment Notes

### Desktop App Bundling (Electron / Tauri / PyInstaller)
- VidSense can be bundled locally as a sidecar process with Electron or Tauri desktop applications.
- Run `python run_server.py` on `127.0.0.1:8000`.
- Application storage (`data/`) is kept locally on the user's filesystem, ensuring zero cloud storage latency and complete privacy for local video collections.

### Production Server Behind Reverse Proxy (Nginx / Caddy)
When hosting VidSense on a remote server:
- Run behind a reverse proxy (Nginx or Caddy) with TLS (HTTPS/WSS) termination.
- **Proxy Buffering**: Disable proxy buffering for the MJPEG stream:
  ```nginx
  location /live/ {
      proxy_pass http://127.0.0.1:8000;
      proxy_http_version 1.1;
      proxy_set_header Upgrade $http_upgrade;
      proxy_set_header Connection "upgrade";
      proxy_buffering off;
      proxy_read_timeout 86400s;
  }
  ```
- Protect internal ports with firewall rules and restrict CORS using `CORS_ORIGINS` in `.env`.

---

## 8. CPU Performance Optimization

VidSense is tuned for smooth operation on modern multi-core CPUs. If running on low-powered machines (e.g. entry-level laptops, 2-core VMs), tune the following parameters in `vidsense/config.py` or via environment variables:

| Setting | Default | Low-Resource Recommendation | Impact |
| :--- | :---: | :---: | :--- |
| `YOLO_IMGSZ` | `480` | `320` or `256` | Dramatically reduces YOLO inference time per frame. |
| `FILE_PROCESS_FPS` | `2.0` | `1.0` or `0.5` | Halves ingestion time for uploaded videos. |
| `LIVE_PROCESS_FPS` | `5.0` | `3.0` or `2.0` | Lowers CPU usage during live streaming & tracking. |
| `CAPTION_INTERVAL_S` | `15` | `30` or `60` | Reduces external Vision LLM API call frequency. |
| `LIVE_CAPTION_INTERVAL_S`| `30` | `60` | Reduces live background captioning overhead. |

---

## 9. Limitations & Ethical Considerations

- **COCO Object Classes Only**: Object detection and tracking operate on the 80 standard COCO classes (e.g. person, bicycle, car, dog, backpack). Objects outside this taxonomy are detected through semantic vision-language captions rather than deterministic bounding boxes.
- **Tracker-Based Counts are Estimates**: Counting queries reflect unique track trajectories identified by ByteTrack. If an object is severely occluded or leaves and re-enters the camera frame, a new track ID may be assigned. Answers explicitly disclose this estimation.
- **Vision LLM Hallucinations & Caption Quality**: Scene captions rely on multimodal vision models. Fast motion, low lighting, or small background details may occasionally be miscaptioned. The query router returns a strict `NOT_FOUND_MESSAGE` whenever evidence is insufficient.
- **Privacy & Human Video Data**: VidSense processes visual footage that may include identifiable individuals and private premises. Users must ensure compliance with local surveillance and data privacy regulations, obtaining appropriate consent before recording or indexing video of persons.
- **Evaluation**: Accuracy metrics depend on user-curated ground truth. See below to evaluate on your own test footage.

---

## 10. Evaluation Workflow

VidSense includes an automated evaluation harness to assess question-answering accuracy and latency across your indexed video sources:

1. Index your test video files via the API or test scripts.
2. Edit `tests/eval_questions.json` and replace the placeholder `TODO` entries with questions, source names, and expected ground-truth outcomes:
   - `count`: Verify object count within a specified tolerance (`{"type": "count", "cls": "person", "value": 4, "tolerance": 1}`).
   - `found_at`: Verify timestamp interval overlap (`{"type": "found_at", "ts_start": 5.0, "ts_end": 12.0}`).
   - `not_found`: Verify question is correctly identified as not present (`{"type": "not_found"}`).
3. Run the evaluation script:
   ```bash
   python scripts/evaluate.py
   ```
   *(Entries containing `TODO` are automatically skipped. The script outputs a structured results table, percentage accuracy across evaluated items, and mean query latency).*
