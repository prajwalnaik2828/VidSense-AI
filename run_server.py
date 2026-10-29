import logging
import uvicorn
from vidsense import config

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("run_server")
    logger.info(
        "Starting VidSense server on %s:%s. VidSense must run as exactly ONE process (workers=1).",
        config.API_HOST,
        config.API_PORT,
    )
    uvicorn.run("vidsense.api:app", host=config.API_HOST, port=config.API_PORT, workers=1)
