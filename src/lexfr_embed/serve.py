"""Deliverable O5 — live demo: FastAPI `/search` over a FAISS index (research §09).

For the jury demo, serve via RunPod **serverless scale-to-zero** (no idle cost) or a pod
spun up only for the demo window. HF TEI is the production-grade alternative; this
in-process version is the simplest thing that works. Requires the `.[serve]` extra.
"""

from __future__ import annotations

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError as e:  # pragma: no cover
    raise ImportError("Install the serve extra: uv sync --extra serve") from e

app = FastAPI(title="LexFR-Embed — French legal retrieval demo")


class SearchRequest(BaseModel):
    query: str
    k: int = 10


class Hit(BaseModel):
    article_id: str
    score: float
    text: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/search", response_model=list[Hit])
def search(req: SearchRequest) -> list[Hit]:
    """TODO(Phase 1): encode req.query with the loaded LexFR-Embed model, FAISS top-k
    over the indexed corpus, return Hit[]. Load model + index once at startup."""
    raise NotImplementedError("Wire model.encode + FAISS index search.")
