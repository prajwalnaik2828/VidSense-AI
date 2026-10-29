import os
import sys
import time
import glob
import shutil
import threading
import cv2
import chromadb
from sentence_transformers import SentenceTransformer

from vidsense import config
from vidsense import db
from vidsense.video_io import probe_video
from vidsense.detect import create_model, track_frame, TrackAggregator
from vidsense.caption import caption_frame

# Module-level thread-safe singletons
_embedder_lock = threading.Lock()
_embedder = None

_chroma_lock = threading.Lock()
_chroma_client = None
_chroma_col = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        return _embedder


def _get_collection():
    global _chroma_client, _chroma_col
    with _chroma_lock:
        if _chroma_client is None:
            os.makedirs(config.CHROMA_DIR, exist_ok=True)
            _chroma_client = chromadb.PersistentClient(path=config.CHROMA_DIR)
        if _chroma_col is None:
            _chroma_col = _chroma_client.get_or_create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        return _chroma_col


def add_caption(source_id: str, ts: float, text: str) -> None:
    # 1. Insert into SQLite
    db.insert_caption(source_id, ts, text)

    # 2. Compute embedding
    embedder = _get_embedder()
    embedding = embedder.encode([text], normalize_embeddings=True).tolist()

    # 3. Upsert into Chroma
    col = _get_collection()
    cid = f"{source_id}::{ts:.2f}"
    col.upsert(
        ids=[cid],
        embeddings=embedding,
        documents=[text],
        metadatas=[{"source_id": str(source_id), "ts": float(ts)}],
    )


def search_captions(query: str, source_id: str | None = None, k: int = config.TOP_K) -> list[dict]:
    col = _get_collection()
    if col.count() == 0:
        return []

    embedder = _get_embedder()
    q_vec = embedder.encode([query], normalize_embeddings=True).tolist()

    where_filter = {"source_id": source_id} if source_id else None
    actual_k = min(k, col.count())
    if actual_k <= 0:
        return []

    res = col.query(
        query_embeddings=q_vec,
        n_results=actual_k,
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    if res and res["ids"] and len(res["ids"][0]) > 0:
        ids_list = res["ids"][0]
        docs_list = res["documents"][0] if res["documents"] else [""] * len(ids_list)
        metas_list = res["metadatas"][0] if res["metadatas"] else [{}] * len(ids_list)
        dists_list = res["distances"][0] if res["distances"] else [1.0] * len(ids_list)

        for i in range(len(ids_list)):
            dist = float(dists_list[i])
            if dist <= config.MAX_COSINE_DISTANCE:
                meta = metas_list[i]
                results.append({
                    "source_id": meta.get("source_id", ""),
                    "ts": float(meta.get("ts", 0.0)),
                    "text": docs_list[i],
                    "distance": dist,
                })

    results.sort(key=lambda x: x["distance"])
    return results


def delete_source(source_id: str) -> None:
    # 1. Delete Chroma entries
    try:
        col = _get_collection()
        col.delete(where={"source_id": source_id})
    except Exception:
        pass

    # 2. Delete data/frames/<id>/
    frames_dir = os.path.join(config.DATA_DIR, "frames", source_id)
    shutil.rmtree(frames_dir, ignore_errors=True)

    # 3. Delete source video file if in data/uploads or data/live
    src = db.get_source(source_id)
    if src and src.get("path"):
        abs_src_path = os.path.abspath(src["path"])
        abs_uploads = os.path.abspath(os.path.join(config.DATA_DIR, "uploads"))
        abs_live = os.path.abspath(os.path.join(config.DATA_DIR, "live"))
        if abs_src_path.startswith(abs_uploads) or abs_src_path.startswith(abs_live):
            if os.path.exists(abs_src_path):
                try:
                    os.remove(abs_src_path)
                except Exception:
                    pass

    # 4. Delete data/clips/<id>_* files
    clips_pattern = os.path.join(config.DATA_DIR, "clips", f"{source_id}_*")
    for clip_file in glob.glob(clips_pattern):
        try:
            os.remove(clip_file)
        except Exception:
            pass

    # 5. Delete rows from SQLite
    db.delete_source_rows(source_id)


def reset_all() -> None:
    global _chroma_client, _chroma_col
    with _chroma_lock:
        if _chroma_client is not None:
            try:
                _chroma_client.delete_collection(config.COLLECTION_NAME)
            except Exception:
                pass
            _chroma_col = None

    db.reset_tables()

    # Empty subdirectories under DATA_DIR
    for sub in ["uploads", "frames", "clips", "live"]:
        sub_dir = os.path.join(config.DATA_DIR, sub)
        if os.path.exists(sub_dir):
            shutil.rmtree(sub_dir, ignore_errors=True)
        os.makedirs(sub_dir, exist_ok=True)


def ingest_video_file(
    path: str,
    name: str,
    progress_cb=None,
    max_seconds: float | None = None,
    source_id: str | None = None,
) -> str:
    info = probe_video(path)
    duration_s = info["duration_s"]
    fps = info["fps"]

    if duration_s > config.MAX_VIDEO_SECONDS and max_seconds is None:
        raise ValueError(f"Video too long: max {config.MAX_VIDEO_SECONDS} s on this CPU setup")

    if source_id is None:
        source_id = db.create_source(name, "file", path, fps, duration_s, "processing")

    cap = cv2.VideoCapture(path)
    try:
        stride = max(1, round(fps / config.FILE_PROCESS_FPS))
        model = create_model()
        aggregator = TrackAggregator(source_id)

        last_caption_ts = -float("inf")
        last_cb_time = 0.0
        frame_index = 0
        actual_duration_s = 0.0

        target_duration = min(duration_s, max_seconds) if max_seconds is not None else duration_s

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % stride == 0:
                ts = frame_index / fps
                actual_duration_s = ts

                if max_seconds is not None and ts > max_seconds:
                    break

                res = track_frame(model, frame)
                aggregator.update(ts, frame, res)

                # Check caption interval
                if (ts - last_caption_ts) >= config.CAPTION_INTERVAL_S:
                    cap_text = caption_frame(frame)
                    if cap_text:
                        add_caption(source_id, ts, cap_text)
                    last_caption_ts = ts

                # Progress callback (at most 10 times per second)
                if progress_cb is not None:
                    now = time.perf_counter()
                    if (now - last_cb_time) >= 0.1:
                        fraction = min(1.0, ts / target_duration) if target_duration > 0 else 0.0
                        progress_cb(fraction)
                        last_cb_time = now

            frame_index += 1

        if progress_cb is not None:
            progress_cb(1.0)

        # Upsert all aggregator tracks to SQLite
        for trk in aggregator.tracks():
            db.upsert_track(source_id, trk)

        final_duration = min(actual_duration_s, duration_s) if actual_duration_s > 0 else duration_s
        db.update_source(source_id, status="ready", duration_s=final_duration, fps=fps)

    except Exception:
        db.update_source(source_id, status="failed")
        raise
    finally:
        cap.release()

    return source_id
