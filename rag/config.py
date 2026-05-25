"""Single source of truth for RAG hyperparameters.

GET /api/stats reads these constants directly — not env vars — so the
ingested chunking and the live config can never drift apart.

Caps from SPEC §1 (graded): chunk_size <= 1024, overlap_ratio <= 0.3, top_k <= 30.
"""

CHUNK_SIZE: int = 512
OVERLAP_RATIO: float = 0.2
TOP_K: int = 10

EMBED_MODEL: str = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL: str = "4UHRUIN-gpt-5-mini"
EMBED_DIM: int = 1536

PINECONE_NAMESPACE: str = "prod"

assert CHUNK_SIZE <= 1024, "SPEC §1 hard cap"
assert OVERLAP_RATIO <= 0.3, "SPEC §1 hard cap"
assert TOP_K <= 30, "SPEC §1 hard cap"
