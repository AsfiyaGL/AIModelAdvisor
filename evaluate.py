"""
Step 3 — TEST: run the golden set across every tier and score results.

Scoring uses LLM-as-judge (the frontier deployment) against each item's
success_criteria. Output is a comparison table: quality score, latency,
and estimated cost per tier — the evidence base for the routing matrix.

Run:  python main.py eval
"""

import json
import time
from pathlib import Path

from model_registry import Tier, load_registry
from claude_equivalents import CLAUDE_BY_TIER

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator. You will receive a task, its success criteria, and a model's answer. Respond ONLY with JSON:
{"score": <integer 1-5>, "reason": "<one short sentence>"}
5 = fully meets criteria, 3 = partially meets, 1 = fails."""


def _judge(judge_spec, task: str, criteria: str, answer: str) -> dict:
    result = judge_spec.provider.create_completion(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user",
             "content": f"TASK:\n{task}\n\nSUCCESS CRITERIA:\n{criteria}\n\nANSWER:\n{answer}"},
        ],
        temperature=0,
        max_tokens=120,
    )
    raw = result["answer"].strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0, "reason": "Judge output unparseable."}


def run_eval(golden_set_path: str = "golden_set.json") -> None:
    registry = load_registry()
    judge_spec = registry[Tier.FRONTIER]

    golden_set = json.loads(Path(golden_set_path).read_text())
    results = {t.value: {"scores": [], "latency": [], "cost": 0.0} for t in Tier}

    for item in golden_set:
        print(f"\n=== {item['id']}: {item['prompt'][:60]}... ===")
        for tier, spec in registry.items():
            start = time.perf_counter()
            result = spec.provider.create_completion(
                messages=[{"role": "user", "content": item["prompt"]}],
                temperature=0.2,
                max_tokens=800,
            )
            latency = time.perf_counter() - start
            answer = result["answer"]
            cost = (
                result["prompt_tokens"] / 1_000_000 * spec.input_cost_per_m
                + result["completion_tokens"] / 1_000_000 * spec.output_cost_per_m
            )

            verdict = _judge(
                judge_spec, item["prompt"], item["success_criteria"], answer
            )

            r = results[tier.value]
            r["scores"].append(verdict["score"])
            r["latency"].append(latency)
            r["cost"] += cost
            print(
                f"  {tier.value:<9} score={verdict['score']}/5  "
                f"{latency:5.1f}s  ${cost:.5f}  — {verdict['reason']}"
            )

    print("\n==============================================================")
    print(f"{'TIER':<11}{'AVG SCORE':<12}{'AVG LATENCY':<14}{'TOTAL COST':<13}{'CLAUDE MATCH'}")
    print("--------------------------------------------------------------")
    for tier_name, r in results.items():
        n = max(len(r["scores"]), 1)
        print(
            f"{tier_name:<11}{sum(r['scores']) / n:<12.2f}"
            f"{sum(r['latency']) / n:<14.2f}${r['cost']:<12.5f}"
            f"{CLAUDE_BY_TIER[Tier(tier_name)].model_id}"
        )
    print("==============================================================")
    print("\nInterpretation: if a cheaper tier scores within ~0.5 of the tier")
    print("above it on a category, the routing matrix should send that")
    print("category down a tier. Benchmarks shortlist — YOUR data decides.")
