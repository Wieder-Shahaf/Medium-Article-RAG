"""Phase 1 — run retrieval + answer generation against the live Pinecone index.

For each question in benchmarks/questions.jsonl this script:
  1. Times embed (rag.embeddings.embed_one) + Pinecone query separately
     by invoking the pieces directly (not retrieve()), so we can attribute latency.
  2. Times the chat completion (rag.config.CHAT_MODEL via the course proxy).
  3. Computes retrieval metrics against ground truth:
       hit@1, hit@3, hit@10
       MRR (rank of first gt match, 0 if none)
       distinct_articles (top_k)
       n_gt_hits (how many of gt_article_ids appear in top_k)
  4. Writes one JSONL row to benchmarks/results.jsonl per question.

The script is RESUMABLE: if results.jsonl already contains a qid, it is
skipped. Use --restart to wipe and start fresh.

Usage:
  python scripts/bench_run.py --limit 5            # smoke test
  python scripts/bench_run.py                       # full run
  python scripts/bench_run.py --no-generate         # retrieval-only (free, fast)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from rag.config import CHAT_MODEL, PINECONE_NAMESPACE, TOP_K  # noqa: E402
from rag.embeddings import embed_one  # noqa: E402
from rag.prompts import SYSTEM_PROMPT, build_user_prompt  # noqa: E402
from rag.retrieval import clean_query, dedup_by_article_id  # noqa: E402

QUESTIONS_PATH = ROOT / "benchmarks" / "questions.jsonl"
RESULTS_PATH = ROOT / "benchmarks" / "results.jsonl"


def load_questions(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_existing_qids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.add(json.loads(line)["qid"])
        except Exception:
            pass
    return out


def get_pinecone_index():
    from pinecone import Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return pc.Index(os.environ["PINECONE_INDEX"])


def get_chat_client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        max_retries=1,
        timeout=30,
    )


def retrieve_timed(index, question: str, top_k: int) -> tuple[list[dict], dict]:
    """Mirror retrieval.retrieve() but split embed and query timing."""
    cleaned = clean_query(question)

    t0 = time.perf_counter()
    vec = embed_one(cleaned)
    t_embed = time.perf_counter() - t0

    t1 = time.perf_counter()
    over_fetch = max(top_k * 5, top_k)
    res = index.query(
        vector=vec,
        top_k=over_fetch,
        namespace=PINECONE_NAMESPACE,
        include_metadata=True,
    )
    t_query = time.perf_counter() - t1

    raw = [
        {"id": m.id, "score": float(m.score), "metadata": dict(m.metadata or {})}
        for m in res.matches
    ]
    deduped = dedup_by_article_id(raw)[:top_k]
    chunks = [
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
    return chunks, {"embed_ms": t_embed * 1000, "query_ms": t_query * 1000}


def compute_retrieval_metrics(chunks: list[dict], gt_ids: list[str], category: str) -> dict:
    retrieved_ids = [c["article_id"] for c in chunks]
    gt_set = set(gt_ids)

    hits = [1 if rid in gt_set else 0 for rid in retrieved_ids]
    hit_at_1 = sum(hits[:1])
    hit_at_3 = 1 if sum(hits[:3]) > 0 else 0
    hit_at_10 = 1 if sum(hits[:10]) > 0 else 0

    rank = 0  # 1-based; 0 means "no match"
    for i, h in enumerate(hits, start=1):
        if h:
            rank = i
            break
    mrr = 1.0 / rank if rank > 0 else 0.0

    distinct_articles = len(set(retrieved_ids))
    n_gt_hits = sum(1 for rid in retrieved_ids if rid in gt_set)

    # C2 "3 distinct" requirement: count how many *distinct* articles from gt-pool
    # appear in top-3
    distinct_gt_in_top3 = len({rid for rid in retrieved_ids[:3] if rid in gt_set})

    return {
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "hit_at_10": hit_at_10,
        "mrr": mrr,
        "rank": rank,
        "distinct_articles": distinct_articles,
        "n_gt_hits": n_gt_hits,
        "distinct_gt_in_top3": distinct_gt_in_top3,
    }


def generate_answer(client, chunks: list[dict], question: str) -> tuple[str, float, str | None]:
    user_prompt = build_user_prompt(question, chunks)
    t0 = time.perf_counter()
    try:
        completion = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = (completion.choices[0].message.content or "").strip()
        elapsed = (time.perf_counter() - t0) * 1000
        return answer, elapsed, None
    except Exception as e:
        elapsed = (time.perf_counter() - t0) * 1000
        return "", elapsed, f"{type(e).__name__}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    ap.add_argument("--out", type=Path, default=RESULTS_PATH)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--no-generate", action="store_true", help="Skip chat completion (retrieval-only)")
    ap.add_argument("--restart", action="store_true", help="Truncate results.jsonl before starting")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    questions = load_questions(args.questions)
    if args.limit > 0:
        questions = questions[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.restart and args.out.exists():
        args.out.unlink()
    done = load_existing_qids(args.out)
    if done:
        print(f"Resuming: {len(done)} questions already in {args.out.name}")

    index = get_pinecone_index()
    chat_client = None if args.no_generate else get_chat_client()

    pending = [q for q in questions if q["qid"] not in done]
    print(f"Running {len(pending)} questions (top_k={args.top_k}, generate={not args.no_generate})")

    t_total = time.perf_counter()
    with args.out.open("a", encoding="utf-8") as f:
        for i, q in enumerate(pending, start=1):
            qid = q["qid"]
            try:
                chunks, timings = retrieve_timed(index, q["question"], args.top_k)
                metrics = compute_retrieval_metrics(chunks, q["gt_article_ids"], q["category"])

                if chat_client is not None and chunks:
                    answer, gen_ms, gen_err = generate_answer(chat_client, chunks, q["question"])
                else:
                    answer, gen_ms, gen_err = "", 0.0, None

                record = {
                    "qid": qid,
                    "category": q["category"],
                    "question": q["question"],
                    "gt_article_ids": q["gt_article_ids"],
                    "gt_tag": q.get("gt_tag"),
                    "gt_title": q.get("gt_title"),
                    "gt_authors": q.get("gt_authors"),
                    "retrieved": [
                        {
                            "article_id": c["article_id"],
                            "title": c["title"],
                            "score": c["score"],
                        }
                        for c in chunks
                    ],
                    "retrieved_chunks": [c["chunk"] for c in chunks],  # for judge
                    "answer": answer,
                    "metrics": metrics,
                    "timings_ms": {
                        **timings,
                        "generate_ms": gen_ms,
                        "total_ms": timings["embed_ms"] + timings["query_ms"] + gen_ms,
                    },
                    "errors": {"generate": gen_err},
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                if i % 5 == 0 or i == len(pending):
                    elapsed = time.perf_counter() - t_total
                    rate = i / elapsed
                    eta = (len(pending) - i) / rate if rate > 0 else 0
                    print(
                        f"  [{i:>3}/{len(pending)}] {qid}  "
                        f"hit@10={metrics['hit_at_10']} mrr={metrics['mrr']:.3f}  "
                        f"embed={timings['embed_ms']:.0f}ms query={timings['query_ms']:.0f}ms gen={gen_ms:.0f}ms  "
                        f"eta={eta:.0f}s"
                    )
            except Exception as e:
                print(f"  [{i}] {qid} FAILED: {type(e).__name__}: {e}")
                f.write(json.dumps({
                    "qid": qid,
                    "category": q["category"],
                    "question": q["question"],
                    "error": f"{type(e).__name__}: {e}",
                }, ensure_ascii=False) + "\n")
                f.flush()

    print(f"Done. Total: {time.perf_counter() - t_total:.1f}s. Output: {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
