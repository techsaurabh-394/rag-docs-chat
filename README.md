# rag-docs-chat

[![Stars](https://img.shields.io/github/stars/techsaurabh-394/rag-docs-chat?style=flat&logo=github&color=6ea8fe)](https://github.com/techsaurabh-394/rag-docs-chat/stargazers) [![License: MIT](https://img.shields.io/badge/license-MIT-green?style=flat)](LICENSE) [![Last commit](https://img.shields.io/github/last-commit/techsaurabh-394/rag-docs-chat?style=flat)](https://github.com/techsaurabh-394/rag-docs-chat/commits/main) [![Issues](https://img.shields.io/github/issues/techsaurabh-394/rag-docs-chat?style=flat)](https://github.com/techsaurabh-394/rag-docs-chat/issues) [![Buy Me A Coffee](https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=flat&logo=buymeacoffee&logoColor=black)](https://buymeacoffee.com/saurabhdev)

Upload PDFs or text files, ask questions, and get answers that **cite the exact source passage** (document name, page and snippet). A compact, readable RAG pipeline: chunking, retrieval, grounded generation and a retrieval evaluation script.

<!-- Add a screenshot after running it:  ![screenshot](docs/screenshot.png) -->

## How it works

```
PDF/TXT/MD -> text per page -> overlapping chunks (page numbers kept)
          -> vectors (hashed TF-IDF or sentence-transformers)
question  -> vector -> cosine similarity -> top-k chunks
          -> Claude answers using ONLY those chunks, citing [1], [2]...
```

| Piece | File | Notes |
|---|---|---|
| Chunking | `app/chunking.py` | Sentence-aware, ~800 chars, 150 overlap, page numbers preserved |
| Retrieval | `app/embeddings.py` | `local`: hashed TF-IDF (numpy only, works offline). `dense`: sentence-transformers |
| Index and answer | `app/rag.py` | In-memory matrix persisted to `data/index.json`; grounded prompt with numbered sources |
| API | `app/main.py` | FastAPI: `/api/upload`, `/api/ask`, `/api/docs` |
| UI | `app/static/index.html` | Single-file vanilla JS interface |
| Evaluation | `eval/run_eval.py` | hit@k and MRR over `eval/questions.jsonl` |

**Two answer modes**
- With `ANTHROPIC_API_KEY` set: Claude writes the answer from the retrieved passages and cites sources. If the sources don't contain the answer, it says so.
- Without a key: *extractive mode* returns the best passage with its citation. The whole app works with no API key and no model downloads.

## Run it

```bash
git clone https://github.com/techsaurabh-394/rag-docs-chat.git
cd rag-docs-chat
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # optional: add ANTHROPIC_API_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000, upload a document, and ask a question.

Environment variables are read from the shell. To load `.env`, run `export $(grep -v '^#' .env | xargs)` first, or set them directly.

**Dense embeddings (better semantic matching):**
```bash
pip install sentence-transformers
EMBEDDER=dense uvicorn app.main:app
```

**Docker:**
```bash
docker build -t rag-docs-chat .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=your_key rag-docs-chat
```

## Evaluate retrieval

```bash
python -m eval.run_eval --verbose
python -m eval.run_eval --dense        # compare with sentence-transformers
```

The bundled set is intentionally tiny (3 documents, 10 questions) so it runs in a second and shows the method. Scores on it are easy to max out, so treat them as a smoke test. To get a meaningful number, drop your own documents into `eval/corpus/` and add question/answer-phrase pairs to `eval/questions.jsonl`; a chunk counts as a hit when it comes from the expected document and contains the expected phrase.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Limits and next steps

- The index is rebuilt on every upload, which is fine for tens of documents. For larger corpora, move to a vector database (pgvector, Qdrant) and incremental indexing.
- Scanned PDFs have no text layer and need OCR first.
- Ideas: hybrid search (BM25 plus dense), reranking, answer-faithfulness evaluation, streaming responses.

## License

MIT, built by Saurabh Pathak.

## Roadmap (contributions welcome)
- [ ] Hybrid retrieval (BM25 + dense) and a reranker
- [ ] Streaming answers in the UI
- [ ] Answer-faithfulness check (does each cited claim appear in its source?)
- [ ] pgvector / Qdrant backend
- [ ] OCR fallback for scanned PDFs

If this is useful, a star helps others find it.

## Support this project
If this is useful, a star helps others find it, and you can support more free open-source work:

<a href="https://buymeacoffee.com/saurabhdev"><img src="https://img.shields.io/badge/Buy_Me_A_Coffee-FFDD00?style=for-the-badge&logo=buymeacoffee&logoColor=black" alt="Buy me a coffee"/></a>

---

<p align="center">
  <b>Saurabh Pathak</b> · AI Engineer &amp; Full-Stack Developer<br/>
  <a href="https://github.com/techsaurabh-394">GitHub</a> ·
  <a href="https://www.linkedin.com/in/tech-saurabh/">LinkedIn</a> ·
  <a href="https://mernstackdev.com">Blog</a> ·
  <a href="mailto:saurabhpathak394@gmail.com">saurabhpathak394@gmail.com</a>
</p>
