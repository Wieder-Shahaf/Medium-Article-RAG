"""Phase 4 live smoke test — hit /api/stats and /api/prompt in-process.

Uses FastAPI's TestClient so we don't need uvicorn locally. Hits the real
OpenAI proxy + the real Pinecone index (1 embed + 1 chat call per question).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Force the namespace via env var so we can target the tune namespace without
# editing config.py.
NS = os.environ.get("SMOKE_NAMESPACE", "tune_cs512_ov02")
import rag.config as cfg
cfg.PINECONE_NAMESPACE = NS

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

SMOKE_QUESTIONS = [
    "List 3 articles about productivity.",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1, help="how many smoke questions to run")
    args = parser.parse_args()

    client = TestClient(app)

    print("=== GET /api/stats ===")
    r = client.get("/api/stats")
    print(f"status: {r.status_code}")
    print(json.dumps(r.json(), indent=2))
    assert r.status_code == 200
    assert set(r.json().keys()) == {"chunk_size", "overlap_ratio", "top_k"}

    for q in SMOKE_QUESTIONS[: args.n]:
        print(f"\n=== POST /api/prompt ({q!r}) ===")
        r = client.post("/api/prompt", json={"question": q})
        print(f"status: {r.status_code}")
        body = r.json()
        # Print without the chunk bodies for readability
        condensed = {
            "response": body["response"],
            "context_titles": [c["title"] for c in body["context"]],
            "Augmented_prompt_keys": list(body["Augmented_prompt"].keys()),
            "Augmented_prompt.System_starts_with": body["Augmented_prompt"]["System"][:80],
        }
        print(json.dumps(condensed, indent=2, ensure_ascii=False))

        # Exact-casing assertions
        assert "Augmented_prompt" in body, "graded: capital A snake_case"
        assert "System" in body["Augmented_prompt"], "graded: capital S"
        assert "User" in body["Augmented_prompt"], "graded: capital U"
        for c in body["context"]:
            assert set(c.keys()) == {"article_id", "title", "chunk", "score"}

    print("\nAll smoke assertions passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
