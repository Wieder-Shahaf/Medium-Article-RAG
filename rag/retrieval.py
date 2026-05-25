"""Retrieval pipeline: query cleanup → embed → Pinecone query → dedup → truncate.

Pinecone client is module-level so the function reuses TCP connections across
Vercel serverless warm invocations.
"""
from __future__ import annotations

import os
import re
from typing import Any

from pinecone import Pinecone

from rag.config import EMBED_DIM, PINECONE_NAMESPACE, TOP_K
from rag.embeddings import embed_one

# Category-2 leading instructions to strip before embedding the query so the
# embedding focuses on the topic, not the instruction phrasing.
_CLEANUP_PATTERNS = [
    re.compile(r"^list\s+(exactly\s+)?\d+\s+articles?\s+(about|on)\s+", re.IGNORECASE),
    re.compile(r"^find\s+(\d+\s+)?articles?\s+(about|on)\s+", re.IGNORECASE),
    re.compile(r"^recommend\s+(\d+\s+)?articles?\s+(about|on)\s+", re.IGNORECASE),
    re.compile(r"^show\s+me\s+(\d+\s+)?articles?\s+(about|on)\s+", re.IGNORECASE),
]

_pc: Pinecone | None = None
_index = None


def _get_index():
    global _pc, _index
    if _index is None:
        _pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        _index = _pc.Index(os.environ["PINECONE_INDEX"])
    return _index


def clean_query(question: str) -> str:
    """Strip Category-2 instruction prefixes. Fall back to original if empty."""
    q = question.strip()
    for pat in _CLEANUP_PATTERNS:
        stripped = pat.sub("", q)
        if stripped != q:
            q = stripped
            break
    q = q.strip()
    return q if q else question.strip()


def dedup_by_article_id(matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep highest-scoring chunk per article_id, preserving Pinecone's score order."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for m in matches:
        aid = m["metadata"].get("article_id")
        if aid in seen:
            continue
        seen.add(aid)
        out.append(m)
    return out


def retrieve(question: str, top_k: int = TOP_K, namespace: str = PINECONE_NAMESPACE) -> list[dict[str, Any]]:
    """Embed the cleaned question, query Pinecone with over-fetch, dedup, truncate.

    Returns a list of dicts each containing: article_id, title, authors, url,
    chunk, score. Empty list on any failure (caller decides what to do).
    """
    cleaned = clean_query(question)
    vec = embed_one(cleaned)

    over_fetch = max(top_k * 5, top_k)
    res = _get_index().query(
        vector=vec,
        top_k=over_fetch,
        namespace=namespace,
        include_metadata=True,
    )

    raw = [
        {"id": m.id, "score": float(m.score), "metadata": dict(m.metadata or {})}
        for m in res.matches
    ]

    deduped = dedup_by_article_id(raw)[:top_k]

    return [
        {
            "article_id": m["metadata"].get("article_id", ""),
            "title": m["metadata"].get("title", ""),
            "authors": m["metadata"].get("authors", ""),
            "url": m["metadata"].get("url", ""),
            "chunk": m["metadata"].get("chunk", ""),
            "score": m["score"],
        }
        for m in deduped
    ]
