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


# A quoted phrase is one or more of:  "..."  '...'  “...”  ‘...’
_QUOTE_RE = re.compile(r'["“”](.+?)["“”]|[\'‘’](.{20,}?)[\'‘’]')


def _extract_quoted_phrase(question: str) -> str | None:
    """Return the longest quoted substring of >= 6 tokens, else None.

    Used as a trigger for the C1-shaped lexical re-rank: when the user pastes
    a verbatim phrase from an article, dense similarity puts ~3 same-topic
    articles ahead of the literal-match one. Token overlap is the tiebreaker.
    """
    best = ""
    for m in _QUOTE_RE.finditer(question):
        candidate = (m.group(1) or m.group(2) or "").strip()
        if len(candidate.split()) >= 6 and len(candidate) > len(best):
            best = candidate
    return best or None


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"\w+", text.lower()) if len(t) > 3}


def lexical_rerank(
    chunks: list[dict[str, Any]], quoted_phrase: str
) -> list[dict[str, Any]]:
    """Token-overlap tiebreaker: bump the chunk with strongest literal-match to position 1.

    Pure reordering of the top-k — never adds or removes a chunk. Activates
    only when the top-overlap chunk has at least 2x the median overlap of the
    set AND strictly more overlap than the current position-1 chunk. These
    guardrails keep the heuristic from misfiring on questions where multiple
    chunks share the quoted phrase equally.
    """
    if len(chunks) < 2:
        return chunks

    q_tokens = _tokens(quoted_phrase)
    if not q_tokens:
        return chunks

    overlaps = [
        len(q_tokens & _tokens(c["metadata"].get("chunk", "")))
        for c in chunks
    ]

    best_i = max(range(len(overlaps)), key=lambda i: overlaps[i])
    if best_i == 0:
        return chunks

    sorted_overlaps = sorted(overlaps)
    median = sorted_overlaps[len(sorted_overlaps) // 2]
    threshold = max(2 * median, overlaps[0] + 1)
    if overlaps[best_i] < threshold:
        return chunks

    promoted = chunks[best_i]
    return [promoted] + [c for i, c in enumerate(chunks) if i != best_i]


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

    quoted = _extract_quoted_phrase(question)
    if quoted:
        deduped = lexical_rerank(deduped, quoted)

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
