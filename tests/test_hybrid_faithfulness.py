import importlib
import io

import pytest
from fastapi.testclient import TestClient

from app.bm25 import BM25, rrf, tokenize
from app.faithfulness import check_faithfulness, split_sentences
from app.rag import Index


# ---- BM25 ---------------------------------------------------------
def test_bm25_ranks_matching_doc_first():
    bm = BM25(["cats purr and sleep", "dogs bark loudly", "redis caching layer for sessions"])
    s = bm.scores("redis caching")
    assert s[2] > 0 and s[0] == 0 and s[1] == 0
    assert s.index(max(s)) == 2


def test_bm25_rare_term_beats_common_term():
    bm = BM25(["alpha beta", "alpha gamma", "alpha delta", "alpha zebra"])
    assert bm.scores("zebra")[3] > bm.scores("alpha")[3]


def test_bm25_length_normalisation_and_empty():
    bm = BM25(["token " + "filler " * 50, "token"])
    s = bm.scores("token")
    assert s[1] > s[0]
    assert BM25([]).scores("anything") == []
    assert bm.scores("the of") == [0.0, 0.0]  # stopwords only
    assert tokenize("The Cat, the HAT!") == ["cat", "hat"]


# ---- RRF ----------------------------------------------------------
def test_rrf_formula_and_order():
    fused = dict(rrf([[1, 2, 3], [3, 1]], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)
    assert [i for i, _ in rrf([[1, 2, 3], [3, 1]])] == [1, 3, 2]


def test_rrf_empty():
    assert rrf([]) == [] and rrf([[], []]) == []


# ---- retrieval modes ----------------------------------------------
@pytest.fixture
def idx():
    i = Index(path=None)
    i.add_document("a.txt", ["Docker containers use namespaces for isolation. Images are built from layers."])
    i.add_document("b.txt", ["The ETag header enables conditional requests and 304 responses."])
    return i


@pytest.mark.parametrize("mode", ["local", "bm25", "hybrid"])
def test_all_modes_find_relevant_chunk(idx, mode):
    hits = idx.search("which header enables conditional requests", k=2, mode=mode)
    assert hits and hits[0][1].doc == "b.txt"


def test_bm25_mode_ignores_nonmatching(idx):
    assert idx.search("zzzz qqqq", mode="bm25") == []


def test_unknown_mode_rejected(idx):
    with pytest.raises(ValueError):
        idx.search("x y z", mode="nope")


def test_retrieval_env_default(idx, monkeypatch):
    monkeypatch.setenv("RETRIEVAL", "bm25")
    s_env = idx.search("conditional requests", k=1)
    assert s_env == idx.search("conditional requests", k=1, mode="bm25")
    monkeypatch.setenv("RETRIEVAL", "bogus")
    with pytest.raises(ValueError):
        idx.search("conditional requests")


def test_hybrid_finds_chunk_either_side_misses():
    i = Index(path=None)
    i.add_document("a.txt", ["Kubernetes pods are scheduled onto nodes by the scheduler."])
    i.add_document("b.txt", ["Bananas are yellow fruit rich in potassium."])
    hits = i.search("scheduler nodes", k=2, mode="hybrid")
    assert hits[0][1].doc == "a.txt"
    assert hits[0][0] == pytest.approx(2 / 61)  # ranked first by both retrievers


# ---- faithfulness -------------------------------------------------
CHUNKS = [
    "Refresh tokens should be stored in an httpOnly cookie so JavaScript cannot read them.",
    "Revoking a stateless token requires a server-side denylist.",
]


def test_split_sentences():
    assert split_sentences("One thing. Two things! Three?\nFour") == ["One thing.", "Two things!", "Three?", "Four"]


def test_supported_answer_scores_high():
    r = check_faithfulness("Refresh tokens should be stored in an httpOnly cookie [1].", CHUNKS)
    assert r["score"] >= 0.9 and r["unsupported"] == 0
    assert r["sentences"][0]["cited"] == [1]


def test_unsupported_sentence_flagged():
    ans = "Refresh tokens belong in an httpOnly cookie [1]. Quantum entanglement powers the blockchain ledger [2]."
    r = check_faithfulness(ans, CHUNKS)
    assert [s["supported"] for s in r["sentences"]] == [True, False]
    assert r["unsupported"] == 1 and 0 < r["score"] < 1


def test_citation_number_matters():
    # Right words, wrong source cited -> unsupported.
    r = check_faithfulness("Use a server-side denylist to revoke tokens [1].", CHUNKS)
    assert r["sentences"][0]["supported"] is False
    r2 = check_faithfulness("Use a server-side denylist to revoke tokens [2].", CHUNKS)
    assert r2["sentences"][0]["supported"] is True


def test_uncited_sentence_checked_against_all_and_empty():
    r = check_faithfulness("A server-side denylist revokes tokens.", CHUNKS)
    assert r["sentences"][0]["supported"] is True
    assert check_faithfulness("", [])["score"] == 0.0


# ---- API ----------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_PATH", str(tmp_path / "idx.json"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("RETRIEVAL", raising=False)
    import app.main as main

    importlib.reload(main)
    return TestClient(main.app)


def _upload(c):
    body = b"JWT refresh tokens belong in an httpOnly cookie. Access tokens should be short lived."
    return c.post("/api/upload", files={"file": ("notes.txt", io.BytesIO(body), "text/plain")})


def test_api_ask_returns_citations_and_faithfulness(client):
    assert _upload(client).status_code == 200
    r = client.post("/api/ask", json={"question": "where do refresh tokens belong"})
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "extractive"
    assert body["sources"][0]["doc"] == "notes.txt" and body["sources"][0]["n"] == 1
    f = body["faithfulness"]
    assert 0 <= f["score"] <= 1 and f["unsupported"] == 0
    assert f["sentences"] and all({"text", "support", "supported", "cited"} <= s.keys() for s in f["sentences"])


@pytest.mark.parametrize("mode", ["local", "bm25", "hybrid"])
def test_api_ask_mode_field(client, mode):
    _upload(client)
    r = client.post("/api/ask", json={"question": "where do refresh tokens belong", "mode": mode})
    assert r.status_code == 200 and r.json()["sources"]


def test_api_rejects_bad_mode(client):
    _upload(client)
    assert client.post("/api/ask", json={"question": "refresh tokens", "mode": "magic"}).status_code == 422


def test_api_no_documents(client):
    body = client.post("/api/ask", json={"question": "anything at all"}).json()
    assert body["mode"] == "none" and body["sources"] == [] and body["faithfulness"]["score"] == 0.0
