import io
import os

import pytest
from fastapi.testclient import TestClient

from app.chunking import chunk_pages, split_text
from app.rag import Index


def test_split_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = split_text(text, size=300, overlap=60)
    assert len(chunks) > 3
    assert all(len(c) <= 300 + 60 + 40 for c in chunks)
    # overlap: end of chunk i shares words with start of chunk i+1
    assert set(chunks[0].split()[-5:]) & set(chunks[1].split()[:15])


def test_split_empty_and_invalid():
    assert split_text("   ") == []
    with pytest.raises(ValueError):
        split_text("abc", size=10, overlap=10)


def test_chunks_keep_page_numbers():
    chunks = chunk_pages("d.pdf", ["first page text.", "second page text."])
    assert [c.page for c in chunks] == [1, 2]


def test_search_finds_relevant_chunk():
    idx = Index(path=None)
    idx.add_document("a.txt", ["Docker containers use namespaces for isolation. Images are built from layers."])
    idx.add_document("b.txt", ["The ETag header enables conditional requests and 304 responses."])
    hits = idx.search("which header enables conditional requests", k=1)
    assert hits and hits[0][1].doc == "b.txt"


def test_remove_document():
    idx = Index(path=None)
    idx.add_document("a.txt", ["alpha beta gamma."])
    assert idx.remove_document("a.txt") is True
    assert idx.documents() == []
    assert idx.search("alpha") == []


def test_persistence_roundtrip(tmp_path):
    p = tmp_path / "index.json"
    Index(path=p).add_document("a.txt", ["Redis is an in-memory data store used for caching."])
    again = Index(path=p)
    assert again.documents() == [{"name": "a.txt", "chunks": 1}]
    assert again.search("in-memory caching")


def test_api_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "idx.json"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import importlib
    import app.main as main
    importlib.reload(main)
    c = TestClient(main.app)

    r = c.post("/api/upload", files={"file": ("notes.txt", io.BytesIO(b"JWT refresh tokens belong in an httpOnly cookie."), "text/plain")})
    assert r.status_code == 200 and r.json()["chunks"] == 1

    r = c.post("/api/ask", json={"question": "where do refresh tokens belong"})
    body = r.json()
    assert r.status_code == 200
    assert body["mode"] == "extractive"
    assert body["sources"][0]["doc"] == "notes.txt"

    assert c.post("/api/upload", files={"file": ("x.exe", io.BytesIO(b"x"), "application/octet-stream")}).status_code == 400
    assert c.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert c.delete("/api/docs/notes.txt").status_code == 200
    assert c.delete("/api/docs/notes.txt").status_code == 404
