"""Eval harness: 8-12 gold questions across the 4 categories, scored manually.

Writes a markdown results table to stdout (and optionally a file) so it can be
copied into README.md. Stays under the 30-chat-call cap from SPEC §1.
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

from openai import OpenAI  # noqa: E402

from rag.config import CHAT_MODEL, TOP_K  # noqa: E402
from rag.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from rag.retrieval import retrieve  # noqa: E402

# Gold questions cover the four required categories.
# Calibrated to topics confirmed to exist in the 1k subset by smoke retrieval.
GOLD = [
    # Category 1 — precise fact retrieval (title + author)
    {"category": 1, "q": "Find the article that introduces deep learning concepts including supervised, unsupervised, and reinforcement learning. Give title and author."},
    {"category": 1, "q": "Which article compares TensorFlow and PyTorch as Python machine learning libraries? Give title and author."},
    # Category 2 — multi-result topic listing (3 distinct articles)
    {"category": 2, "q": "List exactly 3 articles about productivity."},
    {"category": 2, "q": "Recommend 3 articles on machine learning."},
    {"category": 2, "q": "Show me 3 articles about working from home."},
    # Category 3 — key idea summary
    {"category": 3, "q": "Summarize the central argument of an article about time management and creativity."},
    {"category": 3, "q": "Summarize what an article says about why working from home does not mean working less."},
    # Category 4 — recommendation with justification
    {"category": 4, "q": "Recommend one article about deep learning fundamentals and justify your choice."},
    {"category": 4, "q": "Recommend one article on startup tips and justify why."},
]


def run_one(client: OpenAI, q: str, namespace: str, top_k: int) -> dict:
    chunks = retrieve(q, top_k=top_k, namespace=namespace)
    user_prompt = build_user_prompt(q, chunks)
    if not chunks:
        return {"q": q, "answer": "(no chunks retrieved)", "titles": [], "n_chunks": 0}
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    answer = (completion.choices[0].message.content or "").strip()
    return {
        "q": q,
        "answer": answer,
        "titles": [c["title"] for c in chunks],
        "n_chunks": len(chunks),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--out", type=str, default=None, help="JSON output path")
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        max_retries=1,
        timeout=30,
    )

    print(f"Running eval against namespace='{args.namespace}', top_k={args.top_k}")
    print(f"Gold set: {len(GOLD)} questions (cap is 30 chat calls per SPEC §1)\n")

    results = []
    for i, g in enumerate(GOLD, 1):
        print(f"[{i}/{len(GOLD)}] cat={g['category']}: {g['q'][:80]}")
        r = run_one(client, g["q"], args.namespace, args.top_k)
        r["category"] = g["category"]
        results.append(r)
        print(f"   → {r['answer'][:200]}")
        print(f"   retrieved titles: {r['titles'][:3]}\n")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
