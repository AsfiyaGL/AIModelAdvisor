"""
3T Model Selection Framework — POC on Azure OpenAI, with Claude model matches
Task -> Tier -> Test

Usage:
  python main.py route "Write a haiku about onboarding"
  python main.py eval
  python main.py demo          # runs a scripted set of routing examples
"""

import json
import sys

from dotenv import load_dotenv

load_dotenv()

from task_classifier import classify_task          # noqa: E402
from router import route, execute                  # noqa: E402
from evaluate import run_eval                      # noqa: E402
from claude_equivalents import recommend_claude    # noqa: E402

DEMO_PROMPTS = [
    "Classify this ticket as bug, feature, or question: 'App crashes when I upload a PDF over 10MB.'",
    "Write a friendly reminder email about tomorrow's training session at 10am.",
    "Design a multi-agent architecture in LangGraph for automated code review, including state management, error recovery, and human-in-the-loop escalation.",
    "Summarise: 'The workshop covered prompt engineering basics and had 45 attendees.'",
]


def route_and_run(prompt: str) -> None:
    print(f"\nPROMPT: {prompt}\n")

    # Step 1 — TASK
    classification = classify_task(prompt)
    print(f"[1. TASK] category={classification['category']} "
          f"complexity={classification['complexity']}")
    print(f"          rationale: {classification.get('rationale', '-')}")

    # Step 2 — TIER
    spec = route(classification["category"], classification["complexity"])
    print(f"[2. TIER] -> {spec.tier.value.upper()} ({spec.deployment})")
    print(f"          why this tier: {spec.notes}")

    claude = recommend_claude(classification["category"], classification["complexity"])
    print(f"[CLAUDE MATCH] -> {claude.label}")
    print(f"          why: {claude.why}")

    # Execute
    result = execute(prompt, spec)
    print(f"\n[ANSWER]\n{result['answer']}\n")
    print(f"[COST]  ~${result['estimated_cost_usd']} "
          f"({result['prompt_tokens']} in / {result['completion_tokens']} out tokens)")
    claude_cost = claude.estimate_cost(result["prompt_tokens"], result["completion_tokens"])
    print(f"[COST ON CLAUDE]  ~${claude_cost} on {claude.model.model_id} "
          f"(rough: same token counts; excludes Claude thinking tokens)")


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "route" and len(sys.argv) >= 3:
        route_and_run(" ".join(sys.argv[2:]))
    elif command == "eval":
        run_eval()
    elif command == "demo":
        for prompt in DEMO_PROMPTS:
            route_and_run(prompt)
            print("-" * 70)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
