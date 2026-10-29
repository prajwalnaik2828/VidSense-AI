import os
import sys
import glob
import tempfile
import subprocess
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vidsense.video_io import (
    probe_video,
    fmt_ts,
    save_jpeg,
    encode_jpeg,
    decode_jpeg,
    cut_clip,
    reencode_h264,
)


def get_first_test_video() -> str | None:
    video_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests", "videos"))
    exts = ("*.mp4", "*.mkv", "*.avi", "*.mov", "*.webm")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(video_dir, ext)))
    return sorted(files)[0] if files else None


def get_duration_ffprobe(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def main():
    test_video = get_first_test_video()
    if not test_video:
        print("STOP: No video found in tests/videos/. Please add 1 or 2 test videos before running.")
        sys.exit(1)

    print(f"Using test video: {test_video}")
    all_passed = True

    def report(name: str, condition: bool):
        nonlocal all_passed
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_passed = False
        print(f"[{status}] {name}")

    # 1. fmt_ts checks for 0, 59.9, 75.4, 3600
    res_0 = fmt_ts(0)
    res_59 = fmt_ts(59.9)
    res_75 = fmt_ts(75.4)
    res_3600 = fmt_ts(3600)
    fmt_passed = (
        res_0 == "00:00"
        and res_59 == "00:59"
        and res_75 == "01:15"
        and res_3600 == "60:00"
    )
    report(
        f"fmt_ts (0->{res_0}, 59.9->{res_59}, 75.4->{res_75}, 3600->{res_3600})",
        fmt_passed,
    )

    # 2. probe_video
    try:
        info = probe_video(test_video)
        probe_passed = (
            info["fps"] > 0
            and info["duration_s"] > 0
            and info["width"] > 0
            and info["height"] > 0
            and info["frames"] > 0
        )
        report(
            f"probe_video (fps={info['fps']:.1f}, duration={info['duration_s']:.1f}s, {info['width']}x{info['height']})",
            probe_passed,
        )
    except Exception as e:
        report(f"probe_video failed: {e}", False)

    # 3. JPEG encode/decode round trip
    try:
        dummy_frame = np.full((120, 160, 3), 128, dtype=np.uint8)
        encoded = encode_jpeg(dummy_frame, quality=85)
        decoded = decode_jpeg(encoded)
        # Check shape and mean difference due to JPEG compression
        jpeg_passed = (
            decoded.shape == dummy_frame.shape
            and np.mean(np.abs(decoded.astype(float) - dummy_frame.astype(float))) < 5.0
        )
        report("JPEG encode/decode round trip", jpeg_passed)
    except Exception as e:
        report(f"JPEG encode/decode round trip: {e}", False)

    # 4. cut_clip from 2s to 5s
    temp_dir = tempfile.mkdtemp()
    cut_out = os.path.join(temp_dir, "cut_test.mp4")
    try:
        res_cut = cut_clip(test_video, 2.0, 5.0, cut_out)
        duration = get_duration_ffprobe(cut_out)
        dur_ok = abs(duration - 3.0) <= 0.5
        report(f"cut_clip 2s to 5s (duration={duration:.2f}s, expected 3 +/- 0.5s)", dur_ok)
    except Exception as e:
        report(f"cut_clip failed: {e}", False)

    # 5. reencode_h264
    reencode_out = os.path.join(temp_dir, "reencode_test.mp4")
    try:
        res_reencode = reencode_h264(test_video, reencode_out)
        re_info = probe_video(reencode_out)
        re_ok = os.path.exists(reencode_out) and re_info["duration_s"] > 0
        report(f"reencode_h264 (size={os.path.getsize(reencode_out)} bytes)", re_ok)
    except Exception as e:
        report(f"reencode_h264 failed: {e}", False)

    # Cleanup temp directory
    try:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
