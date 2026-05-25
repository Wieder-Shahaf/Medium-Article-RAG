# Medium-articles RAG Assistant

Retrieval-Augmented Generation assistant that answers questions strictly from a Medium-articles dataset (~7,600 English articles). Built for the Individual RAG Assignment.

## Live URL

**https://medium-rag.vercel.app**

- `GET /api/stats` → `{"chunk_size":512,"overlap_ratio":0.2,"top_k":10}`
- `POST /api/prompt` → answers grounded in the Medium-articles dataset; refuses anything else with `"I don't know based on the provided Medium articles data."`

### Example

```bash
curl -X POST https://medium-rag.vercel.app/api/prompt \
  -H "Content-Type: application/json" \
  -d '{"question":"List exactly 3 articles about productivity."}'
```

## GitHub URL

**https://github.com/Wieder-Shahaf/Medium-Article-RAG-**

## Endpoints

- `POST /api/prompt` — `{ "question": "..." }` → answer + retrieved context + augmented prompt
- `GET /api/stats` — `{ "chunk_size", "overlap_ratio", "top_k" }` from `rag/config.py`

> **Cold start:** Vercel Python serverless functions take ~5–15 seconds on the first request after idle. After that, requests typically return in 2–6 seconds (embedding + Pinecone query + chat completion). This is expected, not a failure.

## Hyperparameters

| Param | Value | Cap |
|---|---|---|
| `chunk_size` | 512 tokens | ≤ 1024 |
| `overlap_ratio` | 0.2 | ≤ 0.3 |
| `top_k` | 10 | ≤ 30 |

### Rationale

- **`chunk_size=512`** sits in the middle of the allowed range — small enough that a single chunk usually holds one coherent thought (good for precise-fact retrieval), large enough to give the chat model meaningful context for summary/recommendation questions. Pushing to 1024 hurts Category 1 (more topics per chunk dilutes similarity); pushing to 256 fragments arguments and hurts Categories 3 + 4.
- **`overlap_ratio=0.2`** (~102 tokens) avoids splitting a sentence across chunks without bloating ingestion cost. 0.3 (the cap) gives diminishing returns at higher cost.
- **`top_k=10`** combined with **over-fetch ×5 → dedup by article_id** means we look at the 50 closest chunks before keeping the best one per article and trimming to 10 distinct articles. That headroom is what makes Category 2 ("list 3 distinct articles") reliable. Going larger (15-30) didn't surface better titles in spot-checks and adds chat-prompt tokens.

### Eval — 1k subset, namespace `tune_cs512_ov02`

9 gold questions (2–3 per category) at `(chunk_size=512, overlap_ratio=0.2, top_k=10)`. Manual scoring against the assignment's category definitions.

| # | Cat | Question | Outcome |
|---|---|---|---|
| 1 | 1 | DL article introducing supervised/unsupervised/RL | ✅ Returns `Introduction To Deep Learning — Tyler Elliot Bettilyon` with grounded quote |
| 2 | 1 | TF vs PyTorch comparison | ✅ Returns `TensorFlow or PyTorch? A Guide… — The Kite Team` with citation |
| 3 | 2 | "List 3 articles about productivity" | ✅ 3 distinct articles, no chunk duplicates |
| 4 | 2 | "Recommend 3 articles on machine learning" | ✅ 3 distinct ML articles |
| 5 | 2 | "Show me 3 articles about working from home" | ✅ 3 distinct WFH articles |
| 6 | 3 | Summarize time-management/creativity article | ✅ 2–4 sentence summary, cites correct article |
| 7 | 3 | Summarize WFH-not-less-work article | ✅ Summary with paraphrased quote |
| 8 | 4 | Recommend a deep-learning fundamentals article | ✅ One article + justification from passage |
| 9 | 4 | Recommend a startup-tips article | ✅ `Tips for Startups From Peter Thiel` + grounded justification |

**Total: 9 / 9 (100% on 1k subset). 10 of 30 dev/test chat calls used.**

The four required categories all work end-to-end on the subset. The full ~7,600-article corpus only adds coverage; no re-tuning needed.

## Stack

- Python 3.11 + FastAPI on Vercel (region `iad1`)
- Pinecone serverless (aws / us-east-1 / cosine / 1536 dims)
- Embedding: `4UHRUIN-text-embedding-3-small`
- Chat: `4UHRUIN-gpt-5-mini`
- Course proxy: `https://api.llmod.ai`

## Local usage

```bash
pip install -r requirements.txt
cp .env.example .env  # fill OPENAI_API_KEY, PINECONE_API_KEY, etc.
pytest tests/         # run unit tests (no API calls)
uvicorn app:app --reload --port 8000
```

## Ingestion (one-shot, local only)

```bash
python scripts/ingest.py --dry-run --subset 1000
python scripts/ingest.py --subset 1000 --namespace tune_cs512_ov02
python scripts/ingest.py --namespace prod   # full corpus
```

The script is idempotent: re-running skips article_ids already in the target namespace.
