"""Phase 2 — Claude Code CLI as judge for faithfulness + correctness.

For each row in benchmarks/results.jsonl, this script shells out to:
  claude -p --output-format=json --json-schema=<schema> --max-budget-usd=N
with a strict rubric prompt. The CLI returns a JSON object with numeric
scores and a one-sentence rationale. We append the scored row to
benchmarks/scored.jsonl.

Rubric (per row):
  faithfulness: 0..3
    0 = answer fabricates facts not in any retrieved chunk
    1 = answer has 1 clearly unsupported claim
    2 = answer mostly supported, minor stretching
    3 = answer fully grounded in retrieved chunks (or canonical "I don't know" when justified)
  correctness: 0..3
    Category-specific:
      C1 (precise fact): 3 if names the exact gt_title + author; 2 if one; 0-1 if wrong
      C2 (list 3): 3 if 3 distinct on-topic articles; 2 if 2; 1 if 1; 0 if 0
      C3 (summary): 3 if accurately summarises the gt article; 2 if summarises another retrieved on-topic article; 1 if vague/topical; 0 if wrong/unrelated
      C4 (recommend): 3 if recommends one specific article + justification grounded in context; 2 if recommendation but weak justification; 0-1 if no clear pick
  idk: bool — did the answer respond with the canonical "I don't know..." string?

Resumable. Skips qids already in scored.jsonl.

Usage:
  python scripts/bench_judge.py --limit 3              # smoke test, 3 rows
  python scripts/bench_judge.py --max-budget-usd 0.05  # per-call budget cap
  python scripts/bench_judge.py                         # full run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = ROOT / "benchmarks" / "results.jsonl"
SCORED_PATH = ROOT / "benchmarks" / "scored.jsonl"

CANONICAL_IDK = "I don't know based on the provided Medium articles data."

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "integer", "minimum": 0, "maximum": 3},
        "correctness": {"type": "integer", "minimum": 0, "maximum": 3},
        "idk": {"type": "boolean"},
        "rationale": {"type": "string", "maxLength": 400},
    },
    "required": ["faithfulness", "correctness", "idk", "rationale"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = """You are a strict RAG-output grader. You score one (question, retrieved_context, answer) triple at a time and emit JSON only — no prose, no preamble.

Rubric:
- faithfulness (0-3): how grounded the answer is in the retrieved chunks.
  3 = every claim is supported by some retrieved chunk (or canonical "I don't know" when no chunk supports the answer)
  2 = mostly supported; minor stretching of phrasing
  1 = one or more unsupported claims that aren't fabricated entities
  0 = invents article titles, authors, facts, or quotes not in any chunk

- correctness (0-3): how well the answer satisfies the question's category goal.
  C1 (precise fact retrieval): 3 = names the exact target article title + author; 2 = title OR author; 1 = wrong article on the same topic; 0 = wrong + off-topic
  C2 (multi-result listing): 3 = exactly 3 distinct articles on the requested tag; 2 = 2 distinct; 1 = 1 distinct or duplicates; 0 = 0 on-topic
  C3 (summary): 3 = accurate central-argument summary of the target article; 2 = summary of an on-topic but different retrieved article; 1 = vague/topical only; 0 = wrong or unrelated
  C4 (recommendation + justification): 3 = recommends one specific article + justification quotes/paraphrases context; 2 = picks one but weak justification; 1 = vague recommendation; 0 = no clear pick

- idk (bool): does the answer match the canonical "I don't know" response?

Score conservatively. When in doubt, score lower. Rationale must be ≤400 chars."""


def build_judge_prompt(row: dict) -> str:
    cat = row["category"]
    q = row["question"]
    gt_title = row.get("gt_title") or ""
    gt_authors = row.get("gt_authors") or ""
    gt_tag = row.get("gt_tag") or ""
    gt_ids = row.get("gt_article_ids", [])

    # Send all retrieved chunks (top-10) — unlimited quota means we don't trade context for $.
    retrieved = row.get("retrieved", [])
    chunks = row.get("retrieved_chunks", [])
    ctx_blocks = []
    for i, (meta, body) in enumerate(zip(retrieved, chunks), start=1):
        body_trunc = (body or "")[:800]
        ctx_blocks.append(
            f"[{i}] article_id={meta['article_id']} title={meta['title']!r} score={meta['score']:.3f}\n{body_trunc}"
        )
    context_str = "\n\n".join(ctx_blocks) if ctx_blocks else "(no chunks retrieved)"

    gt_lines = []
    if cat in ("C1", "C3"):
        gt_lines.append(f"Target article title: {gt_title!r}")
        gt_lines.append(f"Target author: {gt_authors!r}")
        gt_lines.append(f"Target article_id: {gt_ids[0] if gt_ids else ''}")
    elif cat in ("C2", "C4"):
        gt_lines.append(f"Acceptable tag: {gt_tag!r}")
        gt_lines.append(f"Ground-truth pool size: {len(gt_ids)} articles tagged {gt_tag!r}")
        if gt_ids:
            gt_lines.append(f"Sample acceptable article_ids: {gt_ids[:5]}")

    return (
        "Score this RAG output. Output EXACTLY one JSON object with these four keys "
        "and nothing else (no markdown, no prose, no extra fields):\n"
        '  {"faithfulness": <int 0-3>, "correctness": <int 0-3>, "idk": <bool>, "rationale": "<=400 chars"}\n\n'
        f"Category: {cat}\n"
        f"Question: {q}\n\n"
        f"Ground truth:\n  " + "\n  ".join(gt_lines) + "\n\n"
        f"Retrieved context (top-{len(retrieved)} chunks, each truncated to 800 chars):\n{context_str}\n\n"
        f"Answer to score:\n{row.get('answer', '')!r}\n\n"
        "Output the JSON object now. No other text."
    )


def call_claude(prompt: str, system: str, max_budget_usd: float, timeout_s: int, model: str, effort: str) -> tuple[dict | None, str]:
    """Invoke `claude -p` and parse the structured response.

    Returns (parsed_obj_or_None, raw_text).
    """
    cmd = [
        "claude",
        "-p",
        "--tools", "",
        "--output-format=json",
        "--model", model,
        "--effort", effort,
        "--append-system-prompt", system,
    ]
    if max_budget_usd > 0:
        cmd += ["--max-budget-usd", str(max_budget_usd)]
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, f"<timeout after {timeout_s}s>"
    except FileNotFoundError:
        return None, "<claude CLI not on PATH>"

    raw = proc.stdout.strip()
    if proc.returncode != 0:
        return None, f"<rc={proc.returncode}> stderr: {proc.stderr[:300]} stdout: {raw[:300]}"

    # claude -p --output-format=json wraps the response. The result field contains the model's output.
    try:
        envelope = json.loads(raw)
        result_text = envelope.get("result", "").strip()
        if not result_text:
            return None, f"<empty result> stop={envelope.get('stop_reason')} turns={envelope.get('num_turns')}"
        # Strip optional ```json fences before parsing
        stripped = result_text
        if stripped.startswith("```"):
            stripped = stripped.removeprefix("```json").removeprefix("```").strip()
            if stripped.endswith("```"):
                stripped = stripped[:-3].strip()
        # Sometimes the model emits prose before the JSON — find the first {...} block.
        if not stripped.startswith("{"):
            i = stripped.find("{")
            j = stripped.rfind("}")
            if i >= 0 and j > i:
                stripped = stripped[i : j + 1]
        parsed = json.loads(stripped)
        return parsed, raw
    except Exception as e:
        return None, f"<parse error: {e}> raw: {raw[:400]}"


def load_scored_qids(path: Path) -> set[str]:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS_PATH)
    ap.add_argument("--out", type=Path, default=SCORED_PATH)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-budget-usd", type=float, default=0.0, help="Per-call budget cap (0 = unlimited)")
    ap.add_argument("--timeout-s", type=int, default=300)
    ap.add_argument("--model", type=str, default="opus", help="claude --model (e.g., haiku, sonnet, opus)")
    ap.add_argument("--effort", type=str, default="max", help="claude --effort (low/medium/high/xhigh/max)")
    ap.add_argument("--concurrency", type=int, default=10, help="Number of parallel claude CLI invocations")
    ap.add_argument("--restart", action="store_true")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    if not args.results.exists():
        print(f"ERROR: {args.results} not found. Run scripts/bench_run.py first.")
        return 2

    rows = [json.loads(l) for l in args.results.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit > 0:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.restart and args.out.exists():
        args.out.unlink()
    done = load_scored_qids(args.out)
    if done:
        print(f"Resuming: {len(done)} rows already scored")

    pending = [r for r in rows if r.get("qid") and r["qid"] not in done and "error" not in r]
    budget_str = "unlimited" if args.max_budget_usd <= 0 else f"${args.max_budget_usd}/call"
    print(
        f"Judging {len(pending)} rows  model={args.model}  effort={args.effort}  "
        f"budget={budget_str}  timeout {args.timeout_s}s  concurrency={args.concurrency}"
    )

    n_ok = 0
    n_err = 0
    t0 = time.perf_counter()
    write_lock = threading.Lock()
    counter_lock = threading.Lock()
    completed = 0

    def judge_one(row: dict) -> dict:
        nonlocal n_ok, n_err, completed
        qid = row["qid"]
        prompt = build_judge_prompt(row)
        t_call = time.perf_counter()
        parsed, raw = call_claude(
            prompt, JUDGE_SYSTEM, args.max_budget_usd, args.timeout_s, args.model, args.effort
        )
        elapsed = time.perf_counter() - t_call

        if parsed is None:
            scored = {
                "qid": qid,
                "category": row["category"],
                "judge_error": True,
                "judge_raw": raw[:500],
                "judge_ms": elapsed * 1000,
            }
            with counter_lock:
                n_err += 1
        else:
            scored = {
                "qid": qid,
                "category": row["category"],
                "faithfulness": parsed.get("faithfulness"),
                "correctness": parsed.get("correctness"),
                "idk": parsed.get("idk"),
                "rationale": parsed.get("rationale", "")[:400],
                "judge_ms": elapsed * 1000,
            }
            with counter_lock:
                n_ok += 1

        line = json.dumps(scored, ensure_ascii=False) + "\n"
        with write_lock:
            f_handle.write(line)
            f_handle.flush()

        with counter_lock:
            completed += 1
            done = completed

        if done % 5 == 0 or done == len(pending):
            total_elapsed = time.perf_counter() - t0
            rate = done / total_elapsed if total_elapsed > 0 else 0
            eta = (len(pending) - done) / rate if rate > 0 else 0
            print(
                f"  [{done:>3}/{len(pending)}] {qid}  "
                f"f={scored.get('faithfulness')} c={scored.get('correctness')} idk={scored.get('idk')}  "
                f"({elapsed:.1f}s)  eta={eta:.0f}s  ok={n_ok} err={n_err}",
                flush=True,
            )
        return scored

    with args.out.open("a", encoding="utf-8") as f_handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = [ex.submit(judge_one, row) for row in pending]
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    print(f"  worker exception: {type(e).__name__}: {e}", flush=True)
                    with counter_lock:
                        n_err += 1

    print(f"Done. ok={n_ok} err={n_err} total={time.perf_counter() - t0:.0f}s -> {args.out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
