"""Optional hosted providers: Gemini for answers, ElevenLabs for spoken answers.

Both use plain HTTPS via urllib (no extra dependency) and are only used when
their API key is set. Without keys the app still works in extractive mode.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"
DEFAULT_TTS_MODEL = "eleven_multilingual_v2"
MAX_SPEAK_CHARS = 1500


class ProviderError(RuntimeError):
    pass


def _post(url: str, headers: dict, payload: dict, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        raise ProviderError(f"HTTP {e.code}") from e
    except (urllib.error.URLError, TimeoutError) as e:
        raise ProviderError(type(e).__name__) from e


def gemini_enabled() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def gemini_answer(system: str, user: str) -> str:
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ProviderError("GEMINI_API_KEY not set")
    model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
    body = _post(
        GEMINI_URL.format(model=model),
        {"Content-Type": "application/json", "x-goog-api-key": key},
        {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"maxOutputTokens": 700, "temperature": 0.2},
        },
    )
    try:
        parts = json.loads(body)["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()
    except (KeyError, IndexError, ValueError) as e:
        raise ProviderError("unexpected Gemini response") from e


def speech_enabled() -> bool:
    return bool(os.getenv("ELEVENLABS_API_KEY"))


def speak(text: str) -> bytes:
    """Return MP3 bytes for `text` using ElevenLabs."""
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise ProviderError("ELEVENLABS_API_KEY not set")
    text = text.strip()[:MAX_SPEAK_CHARS]
    if not text:
        raise ProviderError("nothing to speak")
    voice = os.getenv("ELEVENLABS_VOICE_ID", DEFAULT_VOICE)
    return _post(
        ELEVEN_URL.format(voice=voice),
        {"Content-Type": "application/json", "Accept": "audio/mpeg", "xi-api-key": key},
        {"text": text, "model_id": os.getenv("ELEVENLABS_MODEL", DEFAULT_TTS_MODEL)},
        timeout=60,
    )
