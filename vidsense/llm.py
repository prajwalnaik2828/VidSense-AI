import time
from google import genai
from google.genai import types
from vidsense import config


def _validate_config():
    if not config.GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY in .env")
    if not config.LLM_MODEL:
        raise RuntimeError("Missing LLM_MODEL in .env")
    if config.LLM_PROVIDER != "gemini":
        raise RuntimeError(f"Unsupported LLM_PROVIDER: {config.LLM_PROVIDER}")


_client = None


def _get_client() -> genai.Client:
    global _client
    _validate_config()
    if _client is None:
        _client = genai.Client(api_key=config.GEMINI_API_KEY, http_options={"timeout": 12000})
    return _client


def _call_with_retry(api_func):
    try:
        return api_func()
    except Exception as e:
        # Don't hang on network timeouts or 504 server errors; raise so fallback takes over
        raise e


def generate(prompt: str) -> str:
    _validate_config()
    client = _get_client()
    gen_config = types.GenerateContentConfig(temperature=0.0)

    def _call():
        response = client.models.generate_content(
            model=config.LLM_MODEL,
            contents=prompt,
            config=gen_config,
        )
        return response.text.strip() if response.text else ""

    return _call_with_retry(_call)


def generate_vision(prompt: str, image_jpeg: bytes) -> str:
    _validate_config()
    client = _get_client()
    gen_config = types.GenerateContentConfig(temperature=0.0)

    image_part = types.Part.from_bytes(data=image_jpeg, mime_type="image/jpeg")
    contents = [prompt, image_part]

    def _call():
        response = client.models.generate_content(
            model=config.LLM_MODEL,
            contents=contents,
            config=gen_config,
        )
        return response.text.strip() if response.text else ""

    return _call_with_retry(_call)
