import os
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from vidsense import config

ALLOWED_SOURCE_COLUMNS = {
    "source_id",
    "name",
    "kind",
    "path",
    "fps",
    "duration_s",
    "status",
    "created_at",
}


@contextmanager
def _get_connection():
    db_path = config.DB_PATH
    db_dir = os.path.dirname(db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sources(
                source_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                path TEXT NOT NULL DEFAULT '',
                fps REAL,
                duration_s REAL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tracks(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                cls TEXT NOT NULL,
                first_ts REAL NOT NULL,
                last_ts REAL NOT NULL,
                n_frames INTEGER NOT NULL,
                max_conf REAL NOT NULL,
                best_ts REAL NOT NULL,
                thumb_path TEXT NOT NULL DEFAULT '',
                attributes TEXT NOT NULL DEFAULT '',
                UNIQUE(source_id, track_id)
            );
            CREATE TABLE IF NOT EXISTS captions(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id TEXT NOT NULL,
                ts REAL NOT NULL,
                text TEXT NOT NULL
            );
        """)
        # Safe migration for existing databases
        try:
            conn.execute("ALTER TABLE tracks ADD COLUMN attributes TEXT NOT NULL DEFAULT ''")
        except sqlite3.OperationalError:
            pass


def create_source(
    name: str,
    kind: str,
    path: str,
    fps: float | None,
    duration_s: float | None,
    status: str,
) -> str:
    source_id = uuid.uuid4().hex[:8]
    created_at = time.time()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sources(source_id, name, kind, path, fps, duration_s, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (source_id, name, kind, path, fps, duration_s, status, created_at),
        )
    return source_id


def update_source(source_id: str, **fields) -> None:
    if not fields:
        return
    for k in fields:
        if k not in ALLOWED_SOURCE_COLUMNS:
            raise ValueError(f"Invalid column for source update: {k}")
    set_clause = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [source_id]
    with _get_connection() as conn:
        conn.execute(f"UPDATE sources SET {set_clause} WHERE source_id = ?", values)


def get_source(source_id: str) -> dict | None:
    with _get_connection() as conn:
        cur = conn.execute("SELECT * FROM sources WHERE source_id = ?", (source_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_sources() -> list[dict]:
    with _get_connection() as conn:
        cur = conn.execute("SELECT * FROM sources ORDER BY created_at DESC")
        return [dict(r) for r in cur.fetchall()]


def upsert_track(source_id: str, track: dict) -> None:
    attrs_val = track.get("attributes", "")
    if isinstance(attrs_val, dict):
        attrs_str = json.dumps(attrs_val)
    elif isinstance(attrs_val, str):
        attrs_str = attrs_val
    else:
        attrs_str = ""

    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO tracks(source_id, track_id, cls, first_ts, last_ts, n_frames, max_conf, best_ts, thumb_path, attributes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_id, track_id) DO UPDATE SET
                cls = excluded.cls,
                first_ts = excluded.first_ts,
                last_ts = excluded.last_ts,
                n_frames = excluded.n_frames,
                max_conf = excluded.max_conf,
                best_ts = excluded.best_ts,
                thumb_path = excluded.thumb_path,
                attributes = excluded.attributes
            """,
            (
                source_id,
                track["track_id"],
                track["cls"],
                track["first_ts"],
                track["last_ts"],
                track["n_frames"],
                track["max_conf"],
                track["best_ts"],
                track.get("thumb_path", ""),
                attrs_str,
            ),
        )


def insert_caption(source_id: str, ts: float, text: str) -> int:
    with _get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO captions(source_id, ts, text) VALUES (?, ?, ?)",
            (source_id, ts, text),
        )
        return cur.lastrowid


def query_tracks(
    source_id: str | None,
    cls: str | None,
    start_ts: float | None,
    end_ts: float | None,
) -> list[dict]:
    clauses = ["tracks.n_frames >= ?"]
    params = [config.MIN_TRACK_FRAMES]

    if source_id is not None:
        clauses.append("tracks.source_id = ?")
        params.append(source_id)

    if cls is not None:
        clauses.append("tracks.cls = ?")
        params.append(cls)

    if start_ts is not None:
        clauses.append("tracks.last_ts >= ?")
        params.append(start_ts)

    if end_ts is not None:
        clauses.append("tracks.first_ts <= ?")
        params.append(end_ts)

    where_str = " AND ".join(clauses)
    sql = f"""
        SELECT tracks.*, sources.name AS source_name
        FROM tracks
        LEFT JOIN sources ON tracks.source_id = sources.source_id
        WHERE {where_str}
        ORDER BY tracks.first_ts ASC
    """
    with _get_connection() as conn:
        cur = conn.execute(sql, params)
        return [dict(r) for r in cur.fetchall()]


def count_tracks_by_class(source_id: str | None) -> dict[str, int]:
    clauses = ["n_frames >= ?"]
    params = [config.MIN_TRACK_FRAMES]
    if source_id is not None:
        clauses.append("source_id = ?")
        params.append(source_id)

    where_str = " AND ".join(clauses)
    sql = f"SELECT cls, COUNT(*) as cnt FROM tracks WHERE {where_str} GROUP BY cls"
    with _get_connection() as conn:
        cur = conn.execute(sql, params)
        return {r["cls"]: r["cnt"] for r in cur.fetchall()}


def count_captions(source_id: str) -> int:
    with _get_connection() as conn:
        cur = conn.execute(
            "SELECT COUNT(*) as cnt FROM captions WHERE source_id = ?",
            (source_id,),
        )
        row = cur.fetchone()
        return row["cnt"] if row else 0


def delete_source_rows(source_id: str) -> None:
    with _get_connection() as conn:
        conn.execute("DELETE FROM sources WHERE source_id = ?", (source_id,))
        conn.execute("DELETE FROM tracks WHERE source_id = ?", (source_id,))
        conn.execute("DELETE FROM captions WHERE source_id = ?", (source_id,))


def reset_tables() -> None:
    with _get_connection() as conn:
        conn.executescript("""
            DROP TABLE IF EXISTS captions;
            DROP TABLE IF EXISTS tracks;
            DROP TABLE IF EXISTS sources;
        """)
    init_db()
