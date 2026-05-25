"""Phase 3 graded assertion: dedup keeps at most one chunk per article_id,
preserving Pinecone's descending-score order.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.llmod.ai")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_INDEX", "medium-articles")

from rag.retrieval import clean_query, dedup_by_article_id


def _mk(score: float, article_id: str, chunk_index: int):
    return {
        "id": f"{article_id}#{chunk_index}",
        "score": score,
        "metadata": {"article_id": article_id, "chunk_index": chunk_index, "chunk": f"c{chunk_index}"},
    }


def test_dedup_returns_distinct_article_ids():
    # 6 chunks across 2 articles, mixed score order
    matches = [
        _mk(0.91, "A", 0),
        _mk(0.89, "A", 1),
        _mk(0.87, "B", 0),
        _mk(0.80, "A", 2),
        _mk(0.78, "B", 1),
        _mk(0.70, "B", 2),
    ]
    out = dedup_by_article_id(matches)
    aids = [m["metadata"]["article_id"] for m in out]
    assert aids == ["A", "B"]


def test_dedup_keeps_highest_scoring_chunk_per_article():
    matches = [
        _mk(0.95, "X", 4),
        _mk(0.90, "X", 0),
        _mk(0.80, "X", 2),
    ]
    out = dedup_by_article_id(matches)
    assert len(out) == 1
    assert out[0]["metadata"]["chunk_index"] == 4


def test_dedup_empty_input():
    assert dedup_by_article_id([]) == []


def test_clean_query_strips_list_prefix():
    assert clean_query("List 3 articles about productivity") == "productivity"
    assert clean_query("list exactly 3 articles on remote work") == "remote work"


def test_clean_query_strips_find_recommend_show():
    assert clean_query("Find 3 articles about React") == "React"
    assert clean_query("Recommend articles on machine learning") == "machine learning"
    assert clean_query("Show me 3 articles about climate") == "climate"


def test_clean_query_passthrough_when_no_prefix():
    assert clean_query("Summarize the article on writing") == "Summarize the article on writing"


def test_clean_query_falls_back_to_original_when_strip_leaves_empty():
    q = "List 3 articles about"  # trailing whitespace after strip would leave ""
    # The 'about' is part of the prefix pattern; what remains is empty
    out = clean_query("list 3 articles about ")
    # Whatever the regex does, the contract is: never return empty string.
    assert out != ""
