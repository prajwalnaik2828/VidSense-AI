import logging
import cv2
from vidsense import llm
from vidsense.video_io import encode_jpeg

logger = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "Describe what is visible in this video frame in one or two factual sentences. "
    "Mention people (clothing colors, actions), vehicles, notable objects and the type of place. "
    "Do not speculate. Do not mention that this is an image or frame."
)


def caption_frame(frame_bgr) -> str:
    try:
        h, w = frame_bgr.shape[:2]
        longer = max(h, w)
        if longer > 768:
            scale = 768.0 / longer
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            frame_to_encode = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            frame_to_encode = frame_bgr

        jpeg_bytes = encode_jpeg(frame_to_encode)
        text = llm.generate_vision(CAPTION_PROMPT, jpeg_bytes)
        return text.strip() if text else ""
    except Exception as e:
        logger.error(f"caption_frame error: {e}", exc_info=True)
        return ""
