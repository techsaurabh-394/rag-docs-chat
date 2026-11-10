import json

import pytest
from fastapi.testclient import TestClient

from app import providers


class _Resp:
    def __init__(self, data: bytes):
        self._d = data

    def read(self):
        return self._d

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_gemini_answer_parses_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    seen = {}

    def fake_open(req, timeout=0):
        seen["url"] = req.full_url
        seen["key"] = req.headers.get("X-goog-api-key")
        body = {"candidates": [{"content": {"parts": [{"text": "Redis caches sessions [1]."}]}}]}
        return _Resp(json.dumps(body).encode())

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_open)
    out = providers.gemini_answer("sys", "q")
    assert out == "Redis caches sessions [1]."
    assert "gemini" in seen["url"] and seen["key"] == "k"


def test_gemini_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert providers.gemini_enabled() is False
    with pytest.raises(providers.ProviderError):
        providers.gemini_answer("s", "q")


def test_gemini_bad_shape_raises(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda r, timeout=0: _Resp(b'{"candidates": []}'))
    with pytest.raises(providers.ProviderError):
        providers.gemini_answer("s", "q")


def test_speak_truncates_and_returns_bytes(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    sent = {}

    def fake_open(req, timeout=0):
        sent["payload"] = json.loads(req.data)
        return _Resp(b"MP3DATA")

    monkeypatch.setattr(providers.urllib.request, "urlopen", fake_open)
    assert providers.speak("x" * 5000) == b"MP3DATA"
    assert len(sent["payload"]["text"]) == providers.MAX_SPEAK_CHARS


def test_speak_endpoint_503_without_key(monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    from app.main import app

    r = TestClient(app).post("/api/speak", json={"text": "hello"})
    assert r.status_code == 503


def test_speak_endpoint_returns_audio(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    monkeypatch.setattr(providers, "speak", lambda t: b"AUDIO")
    from app.main import app

    r = TestClient(app).post("/api/speak", json={"text": "hello"})
    assert r.status_code == 200 and r.content == b"AUDIO" and r.headers["content-type"] == "audio/mpeg"


def test_speak_endpoint_502_on_provider_failure(monkeypatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")

    def boom(t):
        raise providers.ProviderError("HTTP 401")

    monkeypatch.setattr(providers, "speak", boom)
    from app.main import app

    assert TestClient(app).post("/api/speak", json={"text": "hi"}).status_code == 502


def test_answer_uses_gemini_and_falls_back(monkeypatch):
    from app import rag

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    from app.chunking import chunk_pages

    chunks = chunk_pages("a.txt", ["Redis is used for caching sessions."])
    hits = [(1.0, chunks[0])]
    monkeypatch.setattr(providers, "gemini_answer", lambda s, u: "Redis caches sessions [1].")
    r = rag.answer("what is redis used for", hits)
    assert r["mode"] == "llm" and "Redis" in r["answer"]

    def boom(s, u):
        raise providers.ProviderError("HTTP 429")

    monkeypatch.setattr(providers, "gemini_answer", boom)
    r2 = rag.answer("what is redis used for", hits)
    assert r2["mode"] == "extractive" and "Gemini unavailable" in r2["answer"]


def test_health_reports_providers(monkeypatch):
    from app.main import app

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "k")
    h = TestClient(app).get("/api/health").json()
    assert h["llm"] is True and h["llm_provider"] == "gemini" and h["voice"] is True
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("ELEVENLABS_API_KEY")
    h = TestClient(app).get("/api/health").json()
    assert h["llm"] is False and h["llm_provider"] is None and h["voice"] is False
