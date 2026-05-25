"""OpenAI embedding client wrapper (module-level init, batched).

Reads OPENAI_API_KEY and OPENAI_BASE_URL from env.
"""
from __future__ import annotations

import os
from typing import Iterable

from openai import OpenAI

from rag.config import EMBED_MODEL

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
            max_retries=1,
            timeout=15,
        )
    return _client


def embed_one(text: str) -> list[float]:
    """Embed a single string. Used by the query path."""
    resp = _get_client().embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of strings (used by the ingestion script).

    Caller is responsible for batch sizing (OpenAI accepts up to 2048 inputs
    per request, but we keep batches small for token-rate safety).
    """
    if not texts:
        return []
    resp = _get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]
