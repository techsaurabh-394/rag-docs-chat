"""FastAPI app: upload documents, ask questions, get cited answers."""
from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

from .rag import Index, answer

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
STATIC = Path(__file__).parent / "static"

app = FastAPI(title="rag-docs-chat", version="1.0.0")
index = Index(path=os.getenv("INDEX_PATH", "data/index.json"))


class Ask(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    k: int = Field(default=4, ge=1, le=10)


def _extract_pages(filename: str, data: bytes) -> list[str]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(data))
            return [(p.extract_text() or "") for p in reader.pages]
        except Exception as e:
            raise HTTPException(400, f"Could not read PDF: {e}") from e
    if suffix in {".txt", ".md"}:
        return [data.decode("utf-8", errors="replace")]
    raise HTTPException(400, "Unsupported file type. Use PDF, TXT or MD.")


@app.get("/")
def home():
    return FileResponse(STATIC / "index.html")


@app.get("/api/docs")
def list_docs():
    return {"documents": index.documents()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = Path(file.filename or "upload").name  # strip any path components
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large (20 MB max).")
    pages = _extract_pages(name, data)
    if not any(p.strip() for p in pages):
        raise HTTPException(400, "No extractable text found (scanned PDFs need OCR).")
    n = index.add_document(name, pages)
    return {"name": name, "pages": len(pages), "chunks": n}


@app.delete("/api/docs/{name}")
def delete_doc(name: str):
    if not index.remove_document(name):
        raise HTTPException(404, "Document not found.")
    return {"deleted": name}


@app.post("/api/ask")
def ask(body: Ask):
    hits = index.search(body.question, k=body.k)
    return answer(body.question, hits)
