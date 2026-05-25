"""Phase 6 prod smoke test — one question per category against namespace 'prod'."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import rag.config as cfg  # noqa: E402
cfg.PINECONE_NAMESPACE = "prod"

from app import app  # noqa: E402

QUESTIONS = [
    (1, "Find the article that introduces deep learning concepts including supervised, unsupervised, and reinforcement learning. Give title and author."),
    (2, "List exactly 3 articles about productivity."),
    (3, "Summarize what an article says about why working from home does not mean working less."),
    (4, "Recommend one article on startup tips and justify why."),
]


def main() -> int:
    client = TestClient(app)
    print("=== GET /api/stats ===")
    print(json.dumps(client.get("/api/stats").json(), indent=2))

    for cat, q in QUESTIONS:
        print(f"\n=== Category {cat}: {q} ===")
        r = client.post("/api/prompt", json={"question": q})
        assert r.status_code == 200, r.text
        body = r.json()
        # Exact-casing assertions
        assert "Augmented_prompt" in body and "System" in body["Augmented_prompt"] and "User" in body["Augmented_prompt"]
        for c in body["context"]:
            assert set(c.keys()) == {"article_id", "title", "chunk", "score"}
        print(f"response: {body['response']}")
        print(f"context titles ({len(body['context'])}):")
        for c in body["context"][:5]:
            print(f"   {c['score']:.3f}  {c['title'][:90]}")
    print("\nAll shape assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
