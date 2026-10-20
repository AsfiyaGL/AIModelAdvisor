# 🧭 AI Model Advisor & Token Economics Calculator

**Right model · right budget · lower footprint.** A planning tool that recommends
the right Claude model and effort level for each task, forecasts token costs and a
12-month AI budget, and makes the water and energy footprint of model choices
visible to leaders — before anything is committed.

```bash
streamlit run app.py    # start on the 📈 Executive overview tab
```

## The 3T framework behind it

A working proof of concept for the **Task → Tier → Test** framework: a smart
router that classifies each request, sends it to the right Azure OpenAI
deployment tier, and includes a golden-set evaluation harness to validate the
routing policy with evidence rather than benchmarks.

Everything **runs on Azure OpenAI**. For every request, the POC also reports
the exact **Claude model and effort level** you would use for that task, plus
a rough cost estimate on Claude. No Claude API key is needed — the mapping is
a plain lookup table in `claude_equivalents.py`.

| Tier | Azure deployment (default) | Claude match | Claude $/1M in / out |
|------|----------------------------|--------------|----------------------|
| Frontier | `gpt-4.1-mini` | `claude-opus-5` | $5 / $25 |
| Workhorse | `gpt-4.1-mini` | `claude-sonnet-5` | $2 / $10 |
| Fast | `gpt-4.1-mini` | `claude-haiku-4-5` | $1 / $5 |

All Azure tiers run on the single `gpt-4.1-mini` deployment. Routing still
decides the tier and the Claude match; only the Azure execution is shared.
Because every tier uses the same model, `python main.py eval` will show similar
scores across tiers — it no longer compares models.

Within a tier, the Claude match also sets an **effort** level (low → xhigh)
per (category, complexity) — e.g. coding/high → Opus 5 at `xhigh`,
reasoning/medium → Sonnet 5 at `high`, writing/medium → Sonnet 5 at `low`.

## How it maps to the framework

| Step | Module | What it does |
|------|--------|--------------|
| 1. Task | `task_classifier.py` | Uses the *cheapest* tier to classify each request into a category (coding / reasoning / writing / extraction / general) and complexity (low / medium / high) |
| 2. Tier | `router.py` + `model_registry.py` | A plain, auditable routing matrix maps (category, complexity) → frontier / workhorse / fast tier. Encodes the rule: start one tier lower, escalate on failure |
| 2b. Claude match | `claude_equivalents.py` | Maps the same (category, complexity) to a Claude model + effort level, with a one-line reason |
| 3. Test | `evaluate.py` + `golden_set.json` | Runs a golden set of real prompts across all three tiers, scores each answer with an LLM judge (frontier tier), and reports quality vs latency vs cost per tier, with each tier's Claude match |

The Claude cost estimate reuses the Azure run's token counts. Claude's
tokenizer counts differently, and Opus 5 / Sonnet 5 also bill thinking tokens,
so treat it as a rough lower bound — not a quote.

`provider.py` also contains an `AnthropicProvider`: set `USE_ANTHROPIC=true`
and `ANTHROPIC_API_KEY` in `.env` if you later get Claude access and want to
run the same eval on real Claude models.

## Token economics (Streamlit tab)

`streamlit run app.py` → **💰 Token economics** estimates the monthly Claude bill
for a workload under three scenarios: all on Opus 5, routed per task type, and
routed + prompt caching + Batch API. Edit the task mix (or build it from a CSV of
real prompts), then tune the assumptions: cache share and hit rate, batch share,
tokenizer factor, and thinking overhead per effort level. The maths lives in
`token_economics.py`; no Claude API calls are made.

## Water footprint (Streamlit tab)

**💧 Water footprint** estimates the water used by a classified prompt and by the
Token economics workload: tokens → energy (Wh per 1k generated tokens per model,
plus a smaller input-token share) → on-site cooling water (WUE) and off-site water
for electricity generation (PUE × water intensity), following the split in
*Making AI Less Thirsty* (Li et al., 2023). Anthropic does not publish these
figures, so every factor is an editable assumption and the result is an
order-of-magnitude estimate. The maths lives in `water.py`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then fill in your Azure endpoint, key, deployment names
```

## Run

```bash
# Route a single prompt end to end (shows classification, tier choice, answer, cost)
python main.py route "Design a LangGraph multi-agent code review pipeline"

# Scripted demo — 4 prompts that route to different tiers (great for live sessions)
python main.py demo

# Golden-set evaluation — the evidence table for your routing policy
python main.py eval
```

## Example demo output

```
[1. TASK] category=extraction complexity=low
          rationale: Simple single-label ticket classification.
[2. TIER] -> FAST (gpt-4.1-mini)
[CLAUDE MATCH] -> Claude Haiku 4.5 [claude-haiku-4-5]
```

versus

```
[1. TASK] category=coding complexity=high
          rationale: Open-ended multi-agent architecture design.
[2. TIER] -> FRONTIER (gpt-4.1-mini)
[CLAUDE MATCH] -> Claude Opus 5 [claude-opus-5] (effort=xhigh)
```

Same router, very different cost per request — matched to stakes, not habit.

## Session talking points

- The classifier itself runs on the fast tier: routing overhead costs
  fractions of a cent. Model selection as a *configuration decision*.
- The routing matrix is a plain dict on purpose — governance teams can read
  and audit it. No hidden logic.
- The eval output ends with the decision rule: if a cheaper tier scores
  within ~0.5 of the tier above on a category, route that category down.
- Extend it: swap `golden_set.json` for prompts from your own team's work.
  That's the whole point — benchmarks shortlist, your data decides.

## Extending the POC

- Add an **escalation loop**: if the judge scores a fast-tier answer below 3,
  automatically retry on the workhorse tier.
- Log routing decisions to a CSV and chart tier distribution over a week.
- Swap the dict matrix for config in YAML so non-engineers can tune policy.
- Once you have Claude access, flip `USE_ANTHROPIC=true` and re-run the eval
  to check the Claude matches against real scores.
