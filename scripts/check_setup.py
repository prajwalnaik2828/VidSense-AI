import os
import sys
import time
import subprocess
import shutil
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vidsense.config import YOLO_WEIGHTS, YOLO_IMGSZ, EMBED_MODEL


def check_ffmpeg():
    # Refresh Windows user PATH if ffmpeg is not yet in current process environment
    if sys.platform == "win32" and not shutil.which("ffmpeg"):
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment") as key:
                user_path, _ = winreg.QueryValueEx(key, "Path")
                os.environ["PATH"] = os.environ["PATH"] + os.pathsep + user_path
        except Exception:
            pass

    try:
        res = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if res.returncode == 0:
            print("OK ffmpeg")
            return
        else:
            raise RuntimeError(f"ffmpeg exited with return code {res.returncode}")
    except Exception as e:
        print(f"FAIL ffmpeg: {e}")
        sys.exit(1)


def check_torch_cpu():
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"OK torch-cpu (version={torch.__version__}, cuda_available={cuda_avail})")
    except Exception as e:
        print(f"FAIL torch-cpu: {e}")
        sys.exit(1)


def check_yolo():
    try:
        from ultralytics import YOLO
        model = YOLO(YOLO_WEIGHTS)
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Warm-up run
        model.predict(blank_frame, device="cpu", imgsz=YOLO_IMGSZ, verbose=False)

        # Measure 5 runs
        timings = []
        for _ in range(5):
            t0 = time.perf_counter()
            model.predict(blank_frame, device="cpu", imgsz=YOLO_IMGSZ, verbose=False)
            timings.append((time.perf_counter() - t0) * 1000.0)

        avg_ms = sum(timings) / len(timings)
        implied_fps = 1000.0 / avg_ms if avg_ms > 0 else 0
        print(f"OK yolo (avg={avg_ms:.2f} ms/frame, implied_fps={implied_fps:.2f} FPS)")
    except Exception as e:
        print(f"FAIL yolo: {e}")
        sys.exit(1)


def check_embeddings():
    try:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(EMBED_MODEL, device="cpu")
        vector = embedder.encode("A test sentence for embeddings verification.")
        print(f"OK embeddings (dim={len(vector)})")
    except Exception as e:
        print(f"FAIL embeddings: {e}")
        sys.exit(1)


def main():
    check_ffmpeg()
    check_torch_cpu()
    check_yolo()
    check_embeddings()


if __name__ == "__main__":
    main()
