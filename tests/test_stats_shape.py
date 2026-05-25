"""Phase 4 graded assertion: /api/stats returns exact JSON shape from rag/config.py."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://api.llmod.ai")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_INDEX", "medium-articles")


@pytest.fixture
def client():
    from app import app
    return TestClient(app)


def test_stats_shape_and_types(client):
    r = client.get("/api/stats")
    assert r.status_code == 200
    body = r.json()

    assert set(body.keys()) == {"chunk_size", "overlap_ratio", "top_k"}
    assert isinstance(body["chunk_size"], int)
    assert isinstance(body["overlap_ratio"], (int, float))
    assert isinstance(body["top_k"], int)


def test_stats_matches_config_constants(client):
    from rag.config import CHUNK_SIZE, OVERLAP_RATIO, TOP_K
    body = client.get("/api/stats").json()
    assert body["chunk_size"] == CHUNK_SIZE
    assert body["overlap_ratio"] == OVERLAP_RATIO
    assert body["top_k"] == TOP_K


def test_stats_respects_graded_caps(client):
    body = client.get("/api/stats").json()
    assert body["chunk_size"] <= 1024
    assert body["overlap_ratio"] <= 0.3
    assert body["top_k"] <= 30
