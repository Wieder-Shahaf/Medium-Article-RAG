"""Phase 2 retrieval smoke test — embed + Pinecone query only, no chat.

Cheap (just a few query embeddings, ~$0.00001 total). Prints retrieved titles
so a human can sanity-check that ingestion + retrieval work end-to-end.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.retrieval import retrieve

SMOKE_QUERIES = [
    "machine learning for beginners",
    "remote work productivity tips",
    "personal finance advice",
    "List 3 articles about startups",
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--namespace", required=True)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    for q in SMOKE_QUERIES:
        print(f"\nQ: {q}")
        chunks = retrieve(q, top_k=args.top_k, namespace=args.namespace)
        if not chunks:
            print("   (no chunks retrieved)")
            continue
        for c in chunks:
            print(f"   {c['score']:.3f} | {c['title'][:80]} | by {c['authors'][:40]}")
            print(f"        {c['chunk'][:120]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
