"""Phase 3 — aggregate results.jsonl + scored.jsonl into a human-readable report.

Reads:
  benchmarks/results.jsonl   (per-question retrieval + latency + answer)
  benchmarks/scored.jsonl    (per-question judge scores; optional)

Writes:
  benchmarks/REPORT.md       (markdown summary with per-category tables)
  benchmarks/worst.md        (rows with lowest scores per metric — for inspection)

Usage:
  python scripts/bench_report.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "benchmarks" / "results.jsonl"
SCORED_PATH = ROOT / "benchmarks" / "scored.jsonl"
REPORT_PATH = ROOT / "benchmarks" / "REPORT.md"
WORST_PATH = ROOT / "benchmarks" / "worst.md"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = int(p * (len(s) - 1))
    return s[k]


def fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def aggregate_retrieval(results: list[dict]) -> dict[str, dict]:
    """Per-category retrieval aggregates."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if "error" in r:
            continue
        by_cat[r["category"]].append(r)

    out: dict[str, dict] = {}
    for cat, rows in by_cat.items():
        if not rows:
            continue
        metrics = [r["metrics"] for r in rows]
        timings = [r["timings_ms"] for r in rows]

        # C2-specific: distinct gt articles in top-3 (the "3 distinct articles" requirement)
        distinct_gt_3 = [m["distinct_gt_in_top3"] for m in metrics]

        out[cat] = {
            "n": len(rows),
            "hit_at_1": mean([m["hit_at_1"] for m in metrics]),
            "hit_at_3": mean([m["hit_at_3"] for m in metrics]),
            "hit_at_10": mean([m["hit_at_10"] for m in metrics]),
            "mrr": mean([m["mrr"] for m in metrics]),
            "distinct_articles_avg": mean([m["distinct_articles"] for m in metrics]),
            "n_gt_hits_avg": mean([m["n_gt_hits"] for m in metrics]),
            "distinct_gt_in_top3_avg": mean(distinct_gt_3),
            "embed_p50": pct([t["embed_ms"] for t in timings], 0.5),
            "embed_p95": pct([t["embed_ms"] for t in timings], 0.95),
            "query_p50": pct([t["query_ms"] for t in timings], 0.5),
            "query_p95": pct([t["query_ms"] for t in timings], 0.95),
            "generate_p50": pct([t["generate_ms"] for t in timings], 0.5),
            "generate_p95": pct([t["generate_ms"] for t in timings], 0.95),
            "total_p50": pct([t["total_ms"] for t in timings], 0.5),
            "total_p95": pct([t["total_ms"] for t in timings], 0.95),
        }
    return out


def aggregate_judge(scored: list[dict]) -> dict[str, dict]:
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for s in scored:
        if s.get("judge_error"):
            continue
        by_cat[s["category"]].append(s)

    out: dict[str, dict] = {}
    for cat, rows in by_cat.items():
        if not rows:
            continue
        out[cat] = {
            "n": len(rows),
            "faithfulness_mean": mean([r["faithfulness"] for r in rows if r["faithfulness"] is not None]),
            "correctness_mean": mean([r["correctness"] for r in rows if r["correctness"] is not None]),
            "idk_rate": mean([1.0 if r.get("idk") else 0.0 for r in rows]),
            "faithfulness_dist": {
                k: sum(1 for r in rows if r["faithfulness"] == k) for k in (0, 1, 2, 3)
            },
            "correctness_dist": {
                k: sum(1 for r in rows if r["correctness"] == k) for k in (0, 1, 2, 3)
            },
        }
    return out


def render_report(retr: dict, judge: dict, results: list[dict], scored: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# Medium-articles RAG — Benchmark Report")
    lines.append("")
    lines.append(f"Questions evaluated: **{len(results)}** | Judged: **{len(scored)}**")
    lines.append("")
    lines.append("Hyperparameters at run time (read from `rag/config.py`):")
    lines.append("- `chunk_size`: 512")
    lines.append("- `overlap_ratio`: 0.2")
    lines.append("- `top_k`: 10")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Retrieval quality (per category)")
    lines.append("")
    lines.append("| Cat | N | hit@1 | hit@3 | hit@10 | MRR | distinct_articles | n_gt_hits | distinct_gt@3 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cat in ("C1", "C2", "C3", "C4"):
        r = retr.get(cat)
        if not r:
            lines.append(f"| {cat} | 0 | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {cat} | {r['n']} | {fmt_pct(r['hit_at_1'])} | {fmt_pct(r['hit_at_3'])} | "
            f"{fmt_pct(r['hit_at_10'])} | {r['mrr']:.3f} | {r['distinct_articles_avg']:.1f} | "
            f"{r['n_gt_hits_avg']:.2f} | {r['distinct_gt_in_top3_avg']:.2f} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- **hit@k**: fraction of questions whose ground-truth article appears in top-k.")
    lines.append("- **MRR**: mean reciprocal rank of first ground-truth match (0 if none in top-k).")
    lines.append("- **distinct_articles**: avg # of distinct articles in returned top-k (max 10).")
    lines.append("- **n_gt_hits**: avg count of ground-truth articles found anywhere in top-k.")
    lines.append("- **distinct_gt@3**: avg # distinct on-topic articles in top-3 — the SPEC §3 \"3 distinct articles\" target for C2.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Latency (per category, ms)")
    lines.append("")
    lines.append("| Cat | embed p50 | embed p95 | query p50 | query p95 | generate p50 | generate p95 | total p50 | total p95 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cat in ("C1", "C2", "C3", "C4"):
        r = retr.get(cat)
        if not r:
            continue
        lines.append(
            f"| {cat} | {r['embed_p50']:.0f} | {r['embed_p95']:.0f} | "
            f"{r['query_p50']:.0f} | {r['query_p95']:.0f} | "
            f"{r['generate_p50']:.0f} | {r['generate_p95']:.0f} | "
            f"{r['total_p50']:.0f} | {r['total_p95']:.0f} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    if judge:
        lines.append("## Judge scores (per category)")
        lines.append("")
        lines.append("| Cat | N | faithfulness | correctness | IDK rate |")
        lines.append("|---|---:|---:|---:|---:|")
        for cat in ("C1", "C2", "C3", "C4"):
            j = judge.get(cat)
            if not j:
                lines.append(f"| {cat} | 0 | — | — | — |")
                continue
            lines.append(
                f"| {cat} | {j['n']} | {j['faithfulness_mean']:.2f} / 3 | "
                f"{j['correctness_mean']:.2f} / 3 | {fmt_pct(j['idk_rate'])} |"
            )
        lines.append("")
        lines.append("Score distributions:")
        lines.append("")
        lines.append("| Cat | f=0 | f=1 | f=2 | f=3 | c=0 | c=1 | c=2 | c=3 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for cat in ("C1", "C2", "C3", "C4"):
            j = judge.get(cat)
            if not j:
                continue
            fd = j["faithfulness_dist"]
            cd = j["correctness_dist"]
            lines.append(
                f"| {cat} | {fd[0]} | {fd[1]} | {fd[2]} | {fd[3]} | "
                f"{cd[0]} | {cd[1]} | {cd[2]} | {cd[3]} |"
            )
        lines.append("")
    else:
        lines.append("## Judge scores")
        lines.append("")
        lines.append("_No `scored.jsonl` found. Run `python scripts/bench_judge.py` to produce judge scores._")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Improvement priorities (auto-derived)")
    lines.append("")
    lines.append(suggest_improvements(retr, judge))
    return "\n".join(lines) + "\n"


def suggest_improvements(retr: dict, judge: dict) -> str:
    """Print up to 6 actionable bullets based on aggregates."""
    bullets: list[str] = []

    # C2: distinct_gt@3 < 3 means the "3 distinct on-topic articles" target is missed.
    c2 = retr.get("C2", {})
    if c2 and c2.get("distinct_gt_in_top3_avg", 3) < 2.5:
        bullets.append(
            f"- **C2 multi-result diversity is below target.** Avg distinct on-topic articles in top-3 = "
            f"{c2['distinct_gt_in_top3_avg']:.2f} (target ≥ 2.5). Consider over-fetch ratio or MMR-style "
            f"reranking to spread top-k across articles."
        )

    # C1/C3: hit@10 < 80% means the single-target retrieval is unreliable.
    for cat in ("C1", "C3"):
        r = retr.get(cat, {})
        if r and r.get("hit_at_10", 1.0) < 0.8:
            bullets.append(
                f"- **{cat} hit@10 = {fmt_pct(r['hit_at_10'])} is below 80%.** The target article isn't in "
                f"top-10 ~{(1 - r['hit_at_10']) * 100:.0f}% of the time. Consider query rewriting (esp. for "
                f"C1's distinctive-phrase queries) or HyDE-style query expansion."
            )

    # Generation latency
    for cat in ("C1", "C2", "C3", "C4"):
        r = retr.get(cat, {})
        if r and r.get("generate_p95", 0) > 10000:
            bullets.append(
                f"- **{cat} generation p95 = {r['generate_p95']:.0f} ms.** Long-tail latency from the chat "
                f"model. Consider streaming for UX, or reducing the per-chunk context length."
            )
            break

    # Faithfulness
    if judge:
        for cat in ("C1", "C2", "C3", "C4"):
            j = judge.get(cat, {})
            if j and j.get("faithfulness_mean", 3.0) < 2.5:
                bullets.append(
                    f"- **{cat} faithfulness mean = {j['faithfulness_mean']:.2f} / 3.** The model is "
                    f"introducing claims not in retrieved context. Tighten the system prompt or shorten "
                    f"chunks to reduce hallucination surface."
                )
        for cat in ("C1", "C2", "C3", "C4"):
            j = judge.get(cat, {})
            if j and j.get("correctness_mean", 3.0) < 2.0:
                bullets.append(
                    f"- **{cat} correctness mean = {j['correctness_mean']:.2f} / 3.** Even when "
                    f"retrieval is fine, the generated answer misses the category goal. Add a category-"
                    f"specific instruction template (or few-shot example) for {cat}."
                )
        # IDK rate
        for cat, j in judge.items():
            if j.get("idk_rate", 0) > 0.1:
                bullets.append(
                    f"- **{cat} IDK rate = {fmt_pct(j['idk_rate'])}.** Over-conservative refusals. Check if "
                    f"retrieval actually contains the answer — if so, prompt is too strict."
                )

    if not bullets:
        bullets.append("- No critical issues surfaced by the aggregates. Drill into `worst.md` to inspect long-tail failures.")

    return "\n".join(bullets)


def render_worst(results: list[dict], scored_by_qid: dict[str, dict]) -> str:
    lines: list[str] = []
    lines.append("# Worst-performing questions (for inspection)")
    lines.append("")

    # Group: hit_at_10 == 0 (retrieval miss)
    misses = [r for r in results if "metrics" in r and r["metrics"]["hit_at_10"] == 0]
    lines.append(f"## Retrieval misses (hit@10 = 0): {len(misses)}")
    lines.append("")
    for r in misses[:20]:
        lines.append(f"- **{r['qid']}** [{r['category']}] gt_title={r.get('gt_title')!r}")
        lines.append(f"  > {r['question'][:200]}")
        retrieved_titles = ", ".join(x["title"][:40] for x in r["retrieved"][:3])
        lines.append(f"  retrieved top-3: {retrieved_titles}")
        lines.append("")

    # Lowest faithfulness
    if scored_by_qid:
        low_faith = sorted(
            (r for r in results if r.get("qid") in scored_by_qid and scored_by_qid[r["qid"]].get("faithfulness") is not None),
            key=lambda r: scored_by_qid[r["qid"]]["faithfulness"],
        )[:15]
        lines.append(f"## Lowest faithfulness ({len(low_faith)} shown)")
        lines.append("")
        for r in low_faith:
            s = scored_by_qid[r["qid"]]
            lines.append(f"- **{r['qid']}** [{r['category']}] f={s['faithfulness']} c={s['correctness']} idk={s.get('idk')}")
            lines.append(f"  > Q: {r['question'][:160]}")
            lines.append(f"  > A: {r.get('answer', '')[:160]}")
            lines.append(f"  > judge: {s.get('rationale', '')[:200]}")
            lines.append("")

        low_corr = sorted(
            (r for r in results if r.get("qid") in scored_by_qid and scored_by_qid[r["qid"]].get("correctness") is not None),
            key=lambda r: scored_by_qid[r["qid"]]["correctness"],
        )[:15]
        lines.append(f"## Lowest correctness ({len(low_corr)} shown)")
        lines.append("")
        for r in low_corr:
            s = scored_by_qid[r["qid"]]
            lines.append(f"- **{r['qid']}** [{r['category']}] f={s['faithfulness']} c={s['correctness']} idk={s.get('idk')}")
            lines.append(f"  > Q: {r['question'][:160]}")
            lines.append(f"  > A: {r.get('answer', '')[:160]}")
            lines.append(f"  > judge: {s.get('rationale', '')[:200]}")
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS_PATH)
    ap.add_argument("--scored", type=Path, default=SCORED_PATH)
    ap.add_argument("--report", type=Path, default=REPORT_PATH)
    ap.add_argument("--worst", type=Path, default=WORST_PATH)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

    results = load_jsonl(args.results)
    scored = load_jsonl(args.scored)
    scored_by_qid = {s["qid"]: s for s in scored if "qid" in s}

    if not results:
        print(f"ERROR: no results in {args.results}")
        return 2

    retr = aggregate_retrieval(results)
    judge = aggregate_judge(scored)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(render_report(retr, judge, results, scored), encoding="utf-8")
    args.worst.write_text(render_worst(results, scored_by_qid), encoding="utf-8")
    print(f"Wrote {args.report.name} and {args.worst.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
