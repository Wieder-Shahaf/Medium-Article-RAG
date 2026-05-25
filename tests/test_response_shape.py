"""Phase 4 graded assertion: /api/prompt returns the exact JSON key casing.

These tests stub out OpenAI and Pinecone — no API calls are made.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

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


@pytest.fixture
def fake_chunks():
    return [
        {
            "article_id": "abc123",
            "title": "On Writing",
            "authors": "Jane Doe",
            "url": "https://medium.com/x",
            "chunk": "Writing every day compounds.",
            "score": 0.91,
        },
        {
            "article_id": "def456",
            "title": "Why Habits Win",
            "authors": "John Roe",
            "url": "https://medium.com/y",
            "chunk": "Habits beat motivation.",
            "score": 0.83,
        },
    ]


class _FakeMsg:
    def __init__(self, content): self.content = content


class _FakeChoice:
    def __init__(self, content): self.message = _FakeMsg(content)


class _FakeCompletion:
    def __init__(self, content): self.choices = [_FakeChoice(content)]


def _fake_chat_create(model, messages, **kw):
    return _FakeCompletion("Title — Author.")


def test_prompt_response_exact_casing(client, fake_chunks):
    with patch("app.retrieve", return_value=fake_chunks), \
         patch("app._chat") as mock_chat:
        mock_chat.return_value.chat.completions.create = _fake_chat_create
        r = client.post("/api/prompt", json={"question": "test"})

    assert r.status_code == 200
    body = r.json()

    # Top-level keys — exact casing
    assert "response" in body
    assert "context" in body
    assert "Augmented_prompt" in body, "Capital A, snake_case is graded"

    # Augmented_prompt has System and User capitalized
    ap = body["Augmented_prompt"]
    assert "System" in ap, "Capital S is graded"
    assert "User" in ap, "Capital U is graded"
    assert isinstance(ap["System"], str) and ap["System"]
    assert isinstance(ap["User"], str) and ap["User"]

    # context entries — exact casing
    for c in body["context"]:
        assert set(c.keys()) == {"article_id", "title", "chunk", "score"}
        assert isinstance(c["article_id"], str)
        assert isinstance(c["title"], str)
        assert isinstance(c["chunk"], str)
        assert isinstance(c["score"], (int, float))


def test_prompt_response_empty_context_returns_canonical(client):
    with patch("app.retrieve", return_value=[]):
        r = client.post("/api/prompt", json={"question": "obscure"})

    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "I don't know based on the provided Medium articles data."
    assert body["context"] == []
    assert "Augmented_prompt" in body
    assert "System" in body["Augmented_prompt"]
    assert "User" in body["Augmented_prompt"]


def test_prompt_response_retrieval_exception_returns_200(client):
    with patch("app.retrieve", side_effect=RuntimeError("pinecone down")):
        r = client.post("/api/prompt", json={"question": "anything"})
    assert r.status_code == 200, "SPEC §4: never return 500 from /api/prompt"
    assert r.json()["response"].startswith("I don't know based on the provided")


def test_prompt_response_chat_exception_returns_200(client, fake_chunks):
    with patch("app.retrieve", return_value=fake_chunks), \
         patch("app._chat") as mock_chat:
        mock_chat.return_value.chat.completions.create.side_effect = RuntimeError("openai down")
        r = client.post("/api/prompt", json={"question": "anything"})
    assert r.status_code == 200
    assert r.json()["response"].startswith("I don't know based on the provided")


def test_system_prompt_contains_verbatim_text(client, fake_chunks):
    with patch("app.retrieve", return_value=fake_chunks), \
         patch("app._chat") as mock_chat:
        mock_chat.return_value.chat.completions.create = _fake_chat_create
        r = client.post("/api/prompt", json={"question": "test"})
    sys_prompt = r.json()["Augmented_prompt"]["System"]
    assert "You are a Medium-article assistant" in sys_prompt
    assert 'I don\'t know based on the provided Medium articles data.' in sys_prompt
    assert "strictly and only based on" in sys_prompt
