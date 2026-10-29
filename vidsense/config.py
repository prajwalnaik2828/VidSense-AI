import os
from dotenv import load_dotenv

load_dotenv()

NOT_FOUND_MESSAGE = "I couldn't find that in the indexed video."
YOLO_WEIGHTS = "yolo11n.pt"
YOLO_CONF = 0.35
YOLO_IMGSZ = 416
TRACKER = "bytetrack.yaml"
FILE_PROCESS_FPS = 5
CAPTION_INTERVAL_S = 10
LIVE_PROCESS_FPS = 5
LIVE_CAPTION_INTERVAL_S = 15
TRACK_FLUSH_INTERVAL_S = 5
MIN_TRACK_FRAMES = 3
MAX_VIDEO_SECONDS = 300
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
MAX_COSINE_DISTANCE = 0.8
MAX_RESULTS = 12
CLIP_PAD_S = 2
MAX_UPLOAD_MB = 500
DATA_DIR = "./data"
DB_PATH = "./data/vidsense.db"
CHROMA_DIR = "./data/chroma_db"
COLLECTION_NAME = "captions"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
LLM_MODEL = os.getenv("LLM_MODEL", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
_api_port_env = os.getenv("API_PORT", "8000")
API_PORT = int(_api_port_env) if _api_port_env else 8000
API_KEY = os.getenv("API_KEY", "")
_cors_origins_env = os.getenv("CORS_ORIGINS", "*")
CORS_ORIGINS = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()]

os.makedirs(os.path.join(DATA_DIR, "uploads"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "frames"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "clips"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "live"), exist_ok=True)
