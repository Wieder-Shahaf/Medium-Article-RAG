"""Generate the benchmark question set from the corpus.

For each of the four assignment categories we derive questions from sampled
articles where ground truth is *the article we sampled from*. This gives us
exact hit@k and MRR against the Pinecone index.

Categories (SPEC §3):
  C1 precise fact retrieval         — distinctive phrase from intro -> article+author
  C2 multi-result topic listing     — popular tag -> 3 distinct articles with that tag
  C3 key idea summary               — title-anchored summary request
  C4 recommendation w/ justification — tag -> any article in that tag

Output: benchmarks/questions.jsonl  (one JSON object per line)

Each record:
  {
    "qid": "C1-0001",
    "category": "C1" | "C2" | "C3" | "C4",
    "question": "...",
    "gt_article_ids": ["<sha1[:12]>"],         # for C1/C3: single article
                                                # for C2/C4: tag-pool of acceptable articles
    "gt_tag": "<tag>" | null,
    "gt_title": "<title>" | null,
    "gt_authors": "<authors>" | null
  }

Usage:
  python scripts/bench_questions.py --per-category 75 --seed 42
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CSV_PATH = ROOT / "data" / "medium-english-50mb.csv"
OUT_PATH = ROOT / "benchmarks" / "questions.jsonl"


def article_id_of(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def parse_list_field(raw) -> list[str]:
    """authors/tags are stored as stringified Python lists in the CSV."""
    if pd.isna(raw):
        return []
    s = str(raw).strip()
    if not s:
        return []
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
        if isinstance(v, str):
            return [v.strip()] if v.strip() else []
    except Exception:
        pass
    return [s] if s else []


def first_authors(raw) -> str:
    a = parse_list_field(raw)
    return a[0] if a else ""


def extract_salient_phrase(text: str, title: str) -> str:
    """Find a distinctive 8-14 word phrase from the article opening.

    We want something specific enough to disambiguate (so retrieval has a fair
    chance of returning *this* article rather than a topical neighbour) but
    not so quoted-verbatim that we're just doing string matching.
    """
    body = re.sub(r"\s+", " ", str(text)).strip()
    sentences = re.split(r"(?<=[.!?])\s+", body)
    title_words = {w.lower() for w in re.findall(r"\w+", title) if len(w) > 3}
    candidates = []
    for s in sentences[:8]:
        words = s.split()
        if 8 <= len(words) <= 28:
            content_words = {w.lower() for w in re.findall(r"\w+", s) if len(w) > 3}
            overlap = len(content_words & title_words)
            candidates.append((overlap, len(words), s))
    if not candidates:
        return ""
    candidates.sort(key=lambda t: (-t[0], abs(16 - t[1])))
    return candidates[0][2]


def make_c1_question(row) -> str | None:
    phrase = extract_salient_phrase(row["text"], row["title"])
    if not phrase or len(phrase) < 30:
        return None
    snippet = phrase[:200].rstrip(".!?")
    return (
        f"Which Medium article discusses the following idea, and who wrote it? "
        f'"{snippet}"'
    )


def make_c2_question(tag: str) -> str:
    return f"List three different Medium articles about {tag}."


def make_c3_question(title: str) -> str:
    return f'Summarize the central argument of the Medium article titled "{title}".'


def make_c4_question(tag: str) -> str:
    return (
        f"Recommend a Medium article about {tag} and justify your choice "
        f"using the article's content."
    )


def build(per_category: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=["title", "text", "url"]).reset_index(drop=True)
    df["article_id"] = df["url"].map(article_id_of)
    df["text_len"] = df["text"].str.len()
    df = df[df["text_len"] >= 600].reset_index(drop=True)  # need enough text for a salient phrase

    df["author"] = df["authors"].map(first_authors)
    df["tag_list"] = df["tags"].map(parse_list_field)

    tag_counts: Counter[str] = Counter()
    for tags in df["tag_list"]:
        tag_counts.update(t for t in tags if t)

    # Tags with enough articles to sample 3 distinct ones AND that aren't ultra-generic
    popular_tags = [t for t, c in tag_counts.most_common(120) if c >= 6]
    # Drop overly generic / system tags
    ignore = {"Life", "Life Lessons", "Self Improvement"}
    popular_tags = [t for t in popular_tags if t not in ignore]

    tag_to_articles: dict[str, list[str]] = {}
    for _, row in df.iterrows():
        for t in row["tag_list"]:
            if t in popular_tags:
                tag_to_articles.setdefault(t, []).append(row["article_id"])

    out: list[dict] = []

    # ---- C1: precise fact retrieval ----
    c1_pool = df.sample(n=min(per_category * 3, len(df)), random_state=seed).to_dict("records")
    c1_made = 0
    for row in c1_pool:
        if c1_made >= per_category:
            break
        q = make_c1_question(row)
        if not q:
            continue
        out.append({
            "qid": f"C1-{c1_made + 1:04d}",
            "category": "C1",
            "question": q,
            "gt_article_ids": [row["article_id"]],
            "gt_tag": None,
            "gt_title": row["title"],
            "gt_authors": row["author"],
        })
        c1_made += 1

    # ---- C2: multi-result topic listing ----
    rng.shuffle(popular_tags)
    c2_tags = popular_tags[:per_category] if len(popular_tags) >= per_category else \
              [popular_tags[i % len(popular_tags)] for i in range(per_category)]
    for i, tag in enumerate(c2_tags):
        out.append({
            "qid": f"C2-{i + 1:04d}",
            "category": "C2",
            "question": make_c2_question(tag),
            "gt_article_ids": tag_to_articles.get(tag, []),
            "gt_tag": tag,
            "gt_title": None,
            "gt_authors": None,
        })

    # ---- C3: key idea summary ----
    c3_rows = df.sample(n=per_category, random_state=seed + 1).to_dict("records")
    for i, row in enumerate(c3_rows):
        out.append({
            "qid": f"C3-{i + 1:04d}",
            "category": "C3",
            "question": make_c3_question(row["title"]),
            "gt_article_ids": [row["article_id"]],
            "gt_tag": None,
            "gt_title": row["title"],
            "gt_authors": row["author"],
        })

    # ---- C4: recommendation w/ justification ----
    rng2 = random.Random(seed + 2)
    c4_tags = list(popular_tags)
    rng2.shuffle(c4_tags)
    c4_tags = c4_tags[:per_category] if len(c4_tags) >= per_category else \
              [c4_tags[i % len(c4_tags)] for i in range(per_category)]
    for i, tag in enumerate(c4_tags):
        out.append({
            "qid": f"C4-{i + 1:04d}",
            "category": "C4",
            "question": make_c4_question(tag),
            "gt_article_ids": tag_to_articles.get(tag, []),
            "gt_tag": tag,
            "gt_title": None,
            "gt_authors": None,
        })

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    records = build(args.per_category, args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_cat: Counter[str] = Counter(r["category"] for r in records)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    print(f"Wrote {len(records)} questions to {args.out.name}")
    for cat in ("C1", "C2", "C3", "C4"):
        print(f"  {cat}: {by_cat[cat]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
