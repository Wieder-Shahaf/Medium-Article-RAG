"""Ingest medium-english-50mb.csv → chunk → embed → upsert to Pinecone.

Usage:
  python scripts/ingest.py --dry-run --subset 1000
  python scripts/ingest.py --subset 1000 --namespace tune_cs512_ov02
  python scripts/ingest.py --namespace prod

The script is idempotent: it queries the target namespace's existing
article_ids and skips any article already present.

SPEC §1: re-embedding is a budget violation. The --dry-run flag prints the
estimated token cost before any API call so we never embed by accident.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Ensure the project root is on sys.path when run as a script.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tiktoken  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from pinecone import Pinecone, ServerlessSpec  # noqa: E402

from rag.config import CHUNK_SIZE, EMBED_DIM, OVERLAP_RATIO, PINECONE_NAMESPACE  # noqa: E402
from rag.embeddings import embed_batch  # noqa: E402

EMBED_PRICE_PER_1M_TOKENS = 0.02  # text-embedding-3-small list price; proxy may differ but this is the safe upper-bound.
EMBED_BATCH_SIZE = 100  # number of chunks per OpenAI embeddings.create call
UPSERT_BATCH_SIZE = 100  # Pinecone upsert batch

CSV_PATH = ROOT / "data" / "medium-english-50mb.csv"


def article_id_of(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def load_dataframe(subset: int | None) -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["text", "url", "title"]).reset_index(drop=True)
    if subset is not None and subset > 0:
        df = df.head(subset).copy()
    df["article_id"] = df["url"].map(article_id_of)
    return df


def get_splitter(chunk_size: int, overlap_ratio: float) -> RecursiveCharacterTextSplitter:
    enc = tiktoken.get_encoding("cl100k_base")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * overlap_ratio),
        length_function=lambda s: len(enc.encode(s)),
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_article(splitter, row) -> list[dict]:
    body = str(row["text"])
    title = str(row.get("title", ""))
    pieces = splitter.split_text(body)
    out: list[dict] = []
    for i, p in enumerate(pieces):
        out.append({
            "id": f"{row['article_id']}#{i}",
            "article_id": row["article_id"],
            "chunk_index": i,
            "chunk": p,
            "embed_text": f"{title}\n\n{p}",
            "title": title,
            "authors": str(row.get("authors", "")),
            "url": str(row["url"]),
            "tags": str(row.get("tags", "")),
            "timestamp": str(row.get("timestamp", "")),
        })
    return out


def ensure_index(pc: Pinecone, name: str) -> None:
    if name in [i.name for i in pc.list_indexes()]:
        return
    pc.create_index(
        name=name,
        dimension=EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(name).status["ready"]:
        time.sleep(1)


def existing_article_ids(index, namespace: str) -> set[str]:
    """Best-effort check of which article_ids are already in the target namespace.

    Pinecone's list/list_paginated returns vector IDs; we only need the prefix
    before '#'. If the namespace is empty, returns empty set.
    """
    seen: set[str] = set()
    try:
        for ids in index.list(namespace=namespace):
            for vid in ids:
                seen.add(vid.split("#", 1)[0])
    except Exception:
        pass
    return seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--subset", type=int, default=None, help="Number of articles (None = all)")
    parser.add_argument("--namespace", type=str, default=PINECONE_NAMESPACE)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap-ratio", type=float, default=OVERLAP_RATIO)
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f"ERROR: CSV not found at {CSV_PATH}", file=sys.stderr)
        return 2

    print(f"Loading {CSV_PATH} (subset={args.subset})...")
    df = load_dataframe(args.subset)
    print(f"  {len(df)} articles after dropna")

    splitter = get_splitter(args.chunk_size, args.overlap_ratio)
    enc = tiktoken.get_encoding("cl100k_base")

    print("Chunking...")
    all_chunks: list[dict] = []
    for _, row in df.iterrows():
        all_chunks.extend(chunk_article(splitter, row))
    print(f"  {len(all_chunks)} chunks total")

    total_tokens = sum(len(enc.encode(c["embed_text"])) for c in all_chunks)
    est_cost = total_tokens / 1_000_000 * EMBED_PRICE_PER_1M_TOKENS
    print(f"  ~{total_tokens:,} tokens → estimated ${est_cost:.4f}")

    if args.dry_run:
        print("\nDRY RUN — no API calls made.")
        print(f"  target namespace: {args.namespace}")
        print(f"  chunk_size={args.chunk_size}, overlap_ratio={args.overlap_ratio}")
        return 0

    # Live run.
    print(f"\nLIVE RUN — embedding + upserting to namespace '{args.namespace}'")
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index_name = os.environ["PINECONE_INDEX"]
    ensure_index(pc, index_name)
    index = pc.Index(index_name)

    skip_aids = existing_article_ids(index, args.namespace)
    if skip_aids:
        before = len(all_chunks)
        all_chunks = [c for c in all_chunks if c["article_id"] not in skip_aids]
        print(f"  idempotent skip: {before - len(all_chunks)} chunks dropped ({len(skip_aids)} articles already present)")

    if not all_chunks:
        print("  Nothing to do — target namespace already covers this subset.")
        return 0

    embedded = 0
    upserted = 0
    for i in range(0, len(all_chunks), EMBED_BATCH_SIZE):
        batch = all_chunks[i : i + EMBED_BATCH_SIZE]
        vectors = embed_batch([c["embed_text"] for c in batch])
        embedded += len(vectors)
        records = [
            {
                "id": c["id"],
                "values": v,
                "metadata": {
                    "article_id": c["article_id"],
                    "title": c["title"],
                    "authors": c["authors"],
                    "url": c["url"],
                    "tags": c["tags"],
                    "timestamp": c["timestamp"],
                    "chunk_index": c["chunk_index"],
                    "chunk": c["chunk"],
                },
            }
            for c, v in zip(batch, vectors)
        ]
        for j in range(0, len(records), UPSERT_BATCH_SIZE):
            index.upsert(vectors=records[j : j + UPSERT_BATCH_SIZE], namespace=args.namespace)
            upserted += len(records[j : j + UPSERT_BATCH_SIZE])
        print(f"  embedded {embedded}/{len(all_chunks)} | upserted {upserted}")

    print(f"\nDone. {embedded} chunks embedded, {upserted} upserted to namespace '{args.namespace}'.")
    print(f"Estimated spend this run: ${est_cost:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
