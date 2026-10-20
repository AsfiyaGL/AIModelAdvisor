"""
Step 1 — TASK: classify the incoming request.

We use the cheapest tier to classify, which is itself a live demo of the
framework: routing overhead should cost fractions of a cent.

Output: {"category": ..., "complexity": ...}
  category   : coding | reasoning | writing | extraction | general
  complexity : low | medium | high
"""

import json
from model_registry import Tier, load_registry

CLASSIFIER_SYSTEM_PROMPT = """You are a task-routing classifier. Given a user request, respond ONLY with a JSON object, no markdown fences, with exactly these keys:
- "category": one of "coding", "reasoning", "writing", "extraction", "general"
- "complexity": one of "low", "medium", "high"
- "rationale": one short sentence explaining your choice

Guidance:
- "coding": writing, debugging, or reviewing code; agentic dev tasks
- "reasoning": maths, logic, multi-step analysis, research synthesis
- "writing": prose, emails, articles, marketing copy, documentation
- "extraction": summarisation, classification, data extraction, reformatting — including
  labelling or tagging given text into categories (e.g. "classify this ticket as bug or feature")
- "general": chit-chat, simple factual Q&A, anything else

Complexity:
- "low": single-step, short output, low ambiguity
- "medium": multi-step but well-defined
- "high": open-ended, multi-step, high stakes, or requires deep expertise"""


def classify_task(user_prompt: str) -> dict:
    registry = load_registry()
    fast_spec = registry[Tier.FAST]

    result = fast_spec.provider.create_completion(
        messages=[
            {"role": "system", "content": CLASSIFIER_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=150,
    )
    raw = result["answer"].strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        classification = json.loads(raw)
    except json.JSONDecodeError:
        classification = {
            "category": "general",
            "complexity": "medium",
            "rationale": "Classifier output unparseable; defaulted.",
        }
    return classification
