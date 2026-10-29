import os
import sys
import shutil
import subprocess
import cv2
import numpy as np

# Ensure Windows PATH includes ffmpeg if needed
if sys.platform == "win32" and not shutil.which("ffmpeg"):
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
            user_path, _ = winreg.QueryValueEx(key, "Path")
            os.environ["PATH"] = os.environ["PATH"] + os.pathsep + user_path
    except Exception:
        pass


def probe_video(path: str) -> dict:
    if not os.path.exists(path):
        raise ValueError("Could not read video")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Could not read video")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("Could not read video")

    duration_s = float(frames / fps) if fps > 0 else 0.0
    return {
        "fps": fps,
        "duration_s": duration_s,
        "width": width,
        "height": height,
        "frames": frames,
    }


def fmt_ts(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes = total_seconds // 60
    secs = total_seconds % 60
    return f"{minutes:02d}:{secs:02d}"


def save_jpeg(frame_bgr, path: str) -> None:
    dir_path = os.path.dirname(path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    cv2.imwrite(path, frame_bgr)


def encode_jpeg(frame_bgr, quality: int = 80) -> bytes:
    ret, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ret:
        raise ValueError("Could not encode frame to JPEG")
    return buf.tobytes()


def decode_jpeg(jpeg: bytes):
    arr = np.frombuffer(jpeg, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image")
    return frame


def cut_clip(src_path: str, start_s: float, end_s: float, out_path: str) -> str:
    if os.path.exists(out_path):
        return out_path

    dir_path = os.path.dirname(out_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    start_s = max(0.0, float(start_s))
    duration_s = max(0.0, float(end_s - start_s))

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        str(start_s),
        "-i",
        src_path,
        "-t",
        str(duration_s),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        "-an",
        out_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg not found or failed: {proc.stderr[-300:]}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found or failed: ffmpeg binary not found")
    except Exception as e:
        if not isinstance(e, RuntimeError):
            raise RuntimeError(f"ffmpeg not found or failed: {str(e)[-300:]}")
        raise

    return out_path


def reencode_h264(src_path: str, dst_path: str) -> str:
    dir_path = os.path.dirname(dst_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        src_path,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-movflags",
        "+faststart",
        "-an",
        dst_path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg not found or failed: {proc.stderr[-300:]}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found or failed: ffmpeg binary not found")
    except Exception as e:
        if not isinstance(e, RuntimeError):
            raise RuntimeError(f"ffmpeg not found or failed: {str(e)[-300:]}")
        raise

    return dst_path
