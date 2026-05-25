"""FastAPI app for /api/prompt and /api/stats.

Exact-casing JSON shape is enforced via Pydantic models (SPEC §6). Error
paths (zero chunks, OpenAI exception, Pinecone timeout) return HTTP 200
with the canonical "I don't know" response — NEVER 500 (SPEC §4).
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()  # local .env; on Vercel env vars come from project settings

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict

from rag.config import CHAT_MODEL, CHUNK_SIZE, OVERLAP_RATIO, TOP_K
from rag.prompts import SYSTEM_PROMPT, build_user_prompt
from rag.retrieval import retrieve

logger = logging.getLogger("rag")
logging.basicConfig(level=logging.INFO)

CANONICAL_UNKNOWN = "I don't know based on the provided Medium articles data."

app = FastAPI(title="Medium Articles RAG")


# ---- Pydantic models (strict casing) ----

class PromptRequest(BaseModel):
    question: str


class ContextChunk(BaseModel):
    article_id: str
    title: str
    chunk: str
    score: float


class AugmentedPrompt(BaseModel):
    System: str
    User: str
    model_config = ConfigDict(populate_by_name=False)


class PromptResponse(BaseModel):
    response: str
    context: list[ContextChunk]
    Augmented_prompt: AugmentedPrompt
    model_config = ConfigDict(populate_by_name=False)


class StatsResponse(BaseModel):
    chunk_size: int
    overlap_ratio: float
    top_k: int


# ---- OpenAI chat client (module-level) ----

from openai import OpenAI  # noqa: E402

_chat_client: OpenAI | None = None


def _chat() -> OpenAI:
    global _chat_client
    if _chat_client is None:
        _chat_client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
            max_retries=1,
            timeout=15,
        )
    return _chat_client


# ---- Routes ----

_INDEX_HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML)


@app.get("/api/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    return StatsResponse(
        chunk_size=CHUNK_SIZE,
        overlap_ratio=OVERLAP_RATIO,
        top_k=TOP_K,
    )


@app.post("/api/prompt", response_model=PromptResponse)
def prompt(req: PromptRequest) -> PromptResponse:
    question = req.question

    try:
        chunks = retrieve(question, top_k=TOP_K)
    except Exception:
        logger.exception("retrieval failed")
        chunks = []

    user_prompt = build_user_prompt(question, chunks)

    if not chunks:
        return PromptResponse(
            response=CANONICAL_UNKNOWN,
            context=[],
            Augmented_prompt=AugmentedPrompt(System=SYSTEM_PROMPT, User=user_prompt),
        )

    try:
        completion = _chat().chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        answer = (completion.choices[0].message.content or "").strip()
        if not answer:
            answer = CANONICAL_UNKNOWN
    except Exception:
        logger.exception("chat completion failed")
        answer = CANONICAL_UNKNOWN

    context_items = [
        ContextChunk(
            article_id=c["article_id"],
            title=c["title"],
            chunk=c["chunk"],
            score=c["score"],
        )
        for c in chunks
    ]

    return PromptResponse(
        response=answer,
        context=context_items,
        Augmented_prompt=AugmentedPrompt(System=SYSTEM_PROMPT, User=user_prompt),
    )
