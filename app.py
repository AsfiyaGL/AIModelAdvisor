"""
AI Model Advisor & Token Economics Calculator — Streamlit UI for the 3T framework: the Claude model for each task,
plus the cost, budget plan and water footprint that go with it.

Run:  streamlit run app.py
"""

import base64
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import find_dotenv, load_dotenv

# Streamlit reruns this script in one long-lived process, so a plain load_dotenv()
# would keep whatever values it loaded first. Re-read .env (searching up from the
# working directory) and override on every run so edits apply without a restart.
load_dotenv(find_dotenv(usecwd=True), override=True)

import altair as alt                                                 # noqa: E402
import pandas as pd                                                  # noqa: E402
import streamlit as st                                               # noqa: E402

import token_economics as te                                         # noqa: E402
import water as wt                                                   # noqa: E402
from claude_equivalents import (                                     # noqa: E402
    CLAUDE_BY_TIER, CLAUDE_MATRIX, HAIKU, OPUS, SONNET, recommend_claude,
)
from model_registry import Tier, load_registry                       # noqa: E402
from router import ROUTING_MATRIX, route                             # noqa: E402
from task_classifier import classify_task                            # noqa: E402

EXAMPLES = {
    "— pick an example —": "",
    "Ticket triage (extraction · low)":
        "Classify this ticket as bug, feature, or question: "
        "'App crashes when I upload a PDF over 10MB.'",
    "Reminder email (writing · low)":
        "Write a friendly reminder email about tomorrow's training session at 10am.",
    "Budget optimisation (reasoning)":
        "A training budget of $50,000 must cover 3 programmes. A costs $120 per learner, "
        "B $200, C $80. At least 100 learners each; maximise total learners. Show reasoning.",
    "Multi-agent design (coding · high)":
        "Design a multi-agent architecture in LangGraph for automated code review, including "
        "state management, error recovery, and human-in-the-loop escalation.",
}

TIER_BADGE = {"frontier": "🔴 Frontier", "workhorse": "🟠 Workhorse", "fast": "🟢 Fast"}

# Validated categorical slots 1-3 (blue, orange, aqua), stepped per theme. Colour
# follows the model, in a fixed order, so it never changes with filters.
MODEL_ORDER = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]
PALETTE = {
    "light": {"series": ["#2a78d6", "#eb6834", "#1baf7a"], "surface": "#fcfcfb",
              "ink": "#0b0b0b", "muted": "#898781", "grid": "#e1e0d9",
              "plan": ["#898781", "#6250d6", "#008300"]},
    "dark": {"series": ["#3987e5", "#d95926", "#199e70"], "surface": "#1a1a19",
             "ink": "#ffffff", "muted": "#898781", "grid": "#2c2c2a",
             "plan": ["#898781", "#9085e9", "#008300"]},
}
# Budget plan lines, one per scenario: the all-Opus baseline is the gray reference,
# the two routed scenarios take categorical slots 7 (violet) and 6 (green) so they
# never collide with the model colours above.
# Effort used when a model is shown as an alternative to the recommended one.
COMPARE_EFFORT = {"claude-opus-5": "high", "claude-sonnet-5": "medium", "claude-haiku-4-5": None}


def _theme() -> dict:
    try:
        return PALETTE["dark" if st.context.theme.type == "dark" else "light"]
    except Exception:
        return PALETTE["light"]


def _money(x: float) -> str:
    return f"${x:,.0f}" if x >= 100 else f"${x:,.2f}"


def _usd(x: float) -> str:
    """Small per-request amounts need more decimals."""
    return f"${x:,.2f}" if x >= 1 else f"${x:.5f}"


def _assumptions() -> te.Assumptions:
    """Current values of the Token economics tab's assumption widgets (defaults if not set yet)."""
    s, d = st.session_state, te.Assumptions()
    return te.Assumptions(
        tokenizer_factor=s.get("tok_factor", d.tokenizer_factor),
        thinking_overhead={lvl: s.get(f"think_{lvl}", v) for lvl, v in d.thinking_overhead.items()},
        cacheable_share=s.get("cacheable", d.cacheable_share * 100) / 100,
        cache_hit_rate=s.get("hit_rate", d.cache_hit_rate * 100) / 100,
        batch_share=s.get("batch", d.batch_share * 100) / 100,
    )


def _water_assumptions() -> wt.WaterAssumptions:
    """Current values of the Water footprint tab's assumption widgets (defaults if not set yet)."""
    s, d = st.session_state, wt.WaterAssumptions()
    return wt.WaterAssumptions(
        wh_per_1k_output={m: s.get(f"wh_{m}", v) for m, v in d.wh_per_1k_output.items()},
        input_energy_ratio=s.get("w_input_ratio", d.input_energy_ratio),
        pue=s.get("w_pue", d.pue),
        wue_onsite=s.get("w_wue", d.wue_onsite),
        ewif_offsite=s.get("w_ewif", d.ewif_offsite),
    )


def _water(ml: float) -> str:
    if ml < 10:
        return f"{ml:.2f} mL"
    if ml < 1_000:
        return f"{ml:,.0f} mL"
    if ml < 10_000_000:      # litres up to 10,000 L so side-by-side figures share a unit
        return f"{ml / 1_000:,.1f} L" if ml < 100_000 else f"{ml / 1_000:,.0f} L"
    return f"{ml / 1_000_000:,.1f} m³"


def _bottles(ml: float) -> str:
    """Water as 500 mL bottles."""
    n = ml / wt.BOTTLE_ML
    if n < 1:
        return f"{n:.2f} bottles"
    return f"{n:,.1f} bottles" if n < 100 else f"{n:,.0f} bottles"


# ---- water bottle picture (bottle.jpg next to this file, embedded for st.html)
BOTTLE_IMG = "data:image/jpeg;base64," + base64.b64encode(
    (Path(__file__).parent / "bottle.jpg").read_bytes()).decode()


def _water_card(label: str, ml: float) -> str:
    """A bottle picture with the amount in 500 mL bottles and in litres."""
    t = _theme()
    return (
        f'<div style="display:flex;gap:14px;align-items:center;padding:12px 14px;'
        f'border:1px solid {t["grid"]};border-radius:10px">'
        f'<img src="{BOTTLE_IMG}" alt="water bottle" '
        'style="height:64px;width:auto;flex:none;border-radius:6px">'
        f'<div><div style="font-size:0.85rem;color:{t["muted"]}">{label}</div>'
        f'<div style="font-size:1.45rem;font-weight:700">{_bottles(ml)}</div>'
        f'<div style="font-size:0.9rem">{_water(ml)}</div></div></div>'
    )


def _card_grid(cards: list[str]) -> str:
    return ('<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));'
            f'gap:12px;margin-bottom:8px">{"".join(cards)}</div>')

st.set_page_config(page_title="AI Model Advisor & Token Economics", page_icon="🧭", layout="wide")

BRAND = "#6250d6"
st.html(f"""
<style>
  [data-testid="stAppDeployButton"] {{ display: none; }}
  /* Clear Streamlit's fixed top bar (3.75rem) so the banner is never tucked under it. */
  .block-container {{ padding-top: 4.5rem; }}
  [data-testid="stMetric"] {{
    border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: 10px 14px;
  }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
  .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("### 🧭 AI Model Advisor  \n& Token Economics Calculator")
    st.caption("Choose the right model, plan the budget and see the footprint — before you commit.")

    st.divider()
    st.subheader("Claude model tiers")
    for tier in Tier:
        m = CLAUDE_BY_TIER[tier]
        st.markdown(
            f"{TIER_BADGE[tier.value]}  \n`{m.model_id}`  \n"
            f"\\${m.input_cost_per_m:g} in / \\${m.output_cost_per_m:g} out per 1M tokens"
        )

    st.divider()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    key_set = bool(os.getenv("AZURE_OPENAI_API_KEY"))
    try:
        fast_deployment = load_registry()[Tier.FAST].deployment
    except Exception:
        fast_deployment = "?"
    with st.expander(f"⚙️ Classifier engine · {'🟢 connected' if key_set else '🔴 not set'}"):
        st.markdown(
            f"- **Provider:** Azure OpenAI\n"
            f"- **Endpoint:** `{urlparse(endpoint).netloc or 'not set'}`\n"
            f"- **Deployment:** `{fast_deployment}`\n"
            f"- **API key:** {'✅ set' if key_set else '❌ missing'}"
        )
        st.caption("Only the task classification calls an LLM, and any provider can do it. "
                   "Claude costs and water are calculated, not called.")


# ---------------------------------------------------------------- header
st.html(f"""
<div style="padding:22px 26px;border-radius:14px;color:#fff;
            background:linear-gradient(135deg,{BRAND} 0%,#3a2fa0 100%)">
  <div style="font-size:.78rem;letter-spacing:.12em;opacity:.85">TASK → TIER → TEST</div>
  <div style="font-size:2rem;font-weight:700;margin:4px 0 2px">
    🧭 AI Model Advisor &amp; Token Economics Calculator</div>
  <div style="font-size:1.05rem;opacity:.95">
    Right model · right budget · lower footprint — the Claude model for every task, with the
    cost, 12-month budget and water footprint to match.
  </div>
  <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:14px">
    <span style="background:rgba(255,255,255,.16);padding:4px 12px;border-radius:999px">💰 Cost saving</span>
    <span style="background:rgba(255,255,255,.16);padding:4px 12px;border-radius:999px">📈 Budget planning</span>
    <span style="background:rgba(255,255,255,.16);padding:4px 12px;border-radius:999px">💧 Water awareness</span>
  </div>
</div>
""")

tab_overview, tab_find, tab_table, tab_econ, tab_water, tab_about = st.tabs(
    ["📈 Executive overview", "🔎 Find a model", "📋 Routing table", "💰 Token economics",
     "💧 Water footprint", "ℹ️ How it works"]
)


# ---------------------------------------------------------------- tab: find a model
def _use_example():
    st.session_state.prompt = EXAMPLES[st.session_state.example]


with tab_find:
    st.subheader("1 · Describe your task")
    with st.container(border=True):
        st.selectbox("Start from an example (optional)", list(EXAMPLES),
                     key="example", on_change=_use_example)
        prompt = st.text_area(
            "Your prompt", key="prompt", height=150,
            placeholder="e.g. Write a Python function that validates email addresses, with unit tests.",
        )
        _, col_btn = st.columns([3, 1])
        submitted = col_btn.button("Find model →", type="primary",
                                   width="stretch", disabled=not prompt.strip())

    if submitted:
        try:
            with st.spinner("Classifying the task…"):
                classification = classify_task(prompt)
        except Exception as e:  # missing .env, bad key, wrong deployment name, etc.
            st.session_state.pop("result", None)
            st.error(f"**Classification failed.** Check the Setup panel on the left.\n\n{e}")
        else:
            category, complexity = classification["category"], classification["complexity"]
            result = {
                "prompt": prompt,
                "classification": classification,
                "rec": recommend_claude(category, complexity),
                "spec": route(category, complexity),
            }
            st.session_state.result = result
            # New prompt: reset the token inputs to this task type's defaults.
            st.session_state.usage_output = te.default_output_tokens(category, complexity)
            st.session_state.usage_context = 0

    result = st.session_state.get("result")
    if result:
        rec, spec, cls = result["rec"], result["spec"], result["classification"]

        st.subheader("2 · Result")
        with st.container(border=True):
            left, right = st.columns([2, 3])
            with left:
                st.caption("RECOMMENDED CLAUDE MODEL")
                st.markdown(f"## {rec.model.display_name}")
                st.code(rec.model.model_id, language=None)
            with right:
                m1, m2, m3 = st.columns(3)
                m1.metric("Effort", rec.effort or "n/a",
                          help="Claude's thinking-depth setting. Haiku 4.5 has none.")
                m2.metric("Tier", TIER_BADGE[spec.tier.value])
                m3.metric("Price / 1M tokens",
                          f"${rec.model.input_cost_per_m:g} / ${rec.model.output_cost_per_m:g}",
                          help="Input / output, Anthropic list price.")
                st.info(f"**Why:** {rec.why}")

        t_usage, t_analysis, t_code = st.tabs(
            ["🪙 Token usage & cost", "🧩 Task analysis", "💻 Use it in code"]
        )

        with t_usage:
            a = _assumptions()
            prompt_tokens = max(1, round(len(result["prompt"]) / 4))

            st.markdown("**Request size**")
            i1, i2, i3, i4 = st.columns(4)
            i1.metric("Prompt tokens (est.)", f"{prompt_tokens:,}",
                      help="About 4 characters per token. An estimate — Claude's tokenizer may differ.")
            context_tokens = i2.number_input(
                "+ System prompt / context tokens", min_value=0, step=100, key="usage_context",
                help="Instructions, documents or chat history sent with the prompt.")
            output_tokens = i3.number_input(
                "Expected output tokens", min_value=1, step=100, key="usage_output",
                help="Defaults to the typical answer length for this task type.")
            monthly = i4.number_input("Requests per month", min_value=1, value=10_000,
                                      step=1_000, key="usage_monthly")
            input_tokens = prompt_tokens + context_tokens

            b = te.cost_breakdown(rec.model, rec.effort, input_tokens, output_tokens, a)
            total_tokens = b["input_tokens"] + b["output_tokens"] + b["thinking_tokens"]
            total_cost = b["input_cost"] + b["output_cost"] + b["thinking_cost"]

            st.markdown(f"**One request on {rec.model.display_name}**"
                        + (f" · effort `{rec.effort}`" if rec.effort else ""))
            with st.container(border=True):
                u1, u2, u3, u4 = st.columns(4)
                u1.metric(f"Input · {b['input_tokens']:,.0f} tokens", _usd(b["input_cost"]))
                u2.metric(f"Output · {b['output_tokens']:,.0f} tokens", _usd(b["output_cost"]))
                u3.metric(f"Thinking (est.) · {b['thinking_tokens']:,.0f} tokens", _usd(b["thinking_cost"]),
                          help="Billed as output. Haiku 4.5 does not think. "
                               "Overhead per effort level is set in the Token economics tab.")
                u4.metric(f"Total · {total_tokens:,.0f} tokens", _usd(total_cost))

                optimised = te.cost_per_request(rec.model, rec.effort, input_tokens, output_tokens, a,
                                                use_cache=True, use_batch=True)
                p1, p2, p3 = st.columns(3)
                p1.metric("Per 1,000 requests", _money(total_cost * 1000))
                p2.metric(f"Per month ({monthly:,} requests)", _money(total_cost * monthly))
                p3.metric("Per month with caching + batch", _money(optimised * monthly),
                          delta=(f"-{1 - optimised / total_cost:.0%}"
                                 if total_cost and optimised <= total_cost * 0.99 else None),
                          delta_color="inverse",
                          help="Uses the cache and batch settings from the Token economics tab.")

            st.markdown("**Same request on each Claude model**")
            compare = []
            for m in (OPUS, SONNET, HAIKU):
                effort = rec.effort if m == rec.model else COMPARE_EFFORT[m.model_id]
                cb = te.cost_breakdown(m, effort, input_tokens, output_tokens, a)
                cost = cb["input_cost"] + cb["output_cost"] + cb["thinking_cost"]
                compare.append({
                    "Claude model": m.model_id,
                    "": "✅ recommended" if m == rec.model else "",
                    "Effort": effort or "—",
                    "Thinking tokens": cb["thinking_tokens"],
                    "Cost / request": cost,
                    "Monthly cost": cost * monthly,
                    "vs recommended": f"{cost / total_cost:.1f}×" if total_cost else "—",
                })
            st.dataframe(
                pd.DataFrame(compare), hide_index=True, width="stretch",
                column_config={
                    "Thinking tokens": st.column_config.NumberColumn(format="localized"),
                    "Cost / request": st.column_config.NumberColumn(format="$%.5f"),
                    "Monthly cost": st.column_config.NumberColumn(format="dollar"),
                },
            )
            st.caption("Estimates at Anthropic list prices. Token counts are approximate and thinking "
                       "tokens are an assumption — real Claude usage reports exact numbers.")

        with t_analysis:
            c1, c2 = st.columns(2)
            c1.metric("Category", cls["category"])
            c2.metric("Complexity", cls["complexity"])
            st.markdown(f"**Classifier rationale:** {cls.get('rationale', '-')}")
            with st.expander("Prompt that was analysed"):
                st.text(result["prompt"])

        with t_code:
            st.markdown("Call this model with the Anthropic Python SDK (`pip install anthropic`):")
            effort_arg = (f',\n    output_config={{"effort": "{rec.effort}"}}'
                          if rec.effort else "")
            st.code(
                "import anthropic\n\n"
                "client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY\n"
                "response = client.messages.create(\n"
                f'    model="{rec.model.model_id}",\n'
                "    max_tokens=16000,\n"
                '    messages=[{"role": "user", "content": PROMPT}]'
                f"{effort_arg},\n)\n"
                'print("".join(b.text for b in response.content if b.type == "text"))',
                language="python",
            )

    else:
        st.subheader("2 · Result")
        st.info("Enter a prompt above and click **Find model →**.")


# ---------------------------------------------------------------- tab: routing table
with tab_table:
    st.subheader("Which Claude model for which task")
    st.caption("Every (category, complexity) pair the classifier can return, and its Claude match.")

    rows = [
        {
            "Category": cat,
            "Complexity": cx,
            "Tier": TIER_BADGE[ROUTING_MATRIX[(cat, cx)].value],
            "Claude model": r.model.model_id,
            "Effort": r.effort or "—",
            "Why": r.why,
        }
        for (cat, cx), r in CLAUDE_MATRIX.items()
    ]
    df = pd.DataFrame(rows)

    f1, f2 = st.columns(2)
    cats = f1.multiselect("Category", sorted(df["Category"].unique()))
    models = f2.multiselect("Claude model", sorted(df["Claude model"].unique()))
    if cats:
        df = df[df["Category"].isin(cats)]
    if models:
        df = df[df["Claude model"].isin(models)]

    st.dataframe(df, hide_index=True, width="stretch",
                 column_config={"Why": st.column_config.TextColumn(width="large")})


# ---------------------------------------------------------------- tab: token economics
CSV_ROW_LIMIT = 200


def _scenario_chart(by_model: pd.DataFrame, totals: pd.DataFrame, *, value: str, title: str,
                    axis_format: str, tooltip_format: str, scenarios: list[str]) -> alt.LayerChart:
    """Horizontal stacked bar per scenario, split by Claude model, with the total at the end."""
    t = _theme()
    base_y = alt.Y("Scenario:N", sort=scenarios, title=None,
                   axis=alt.Axis(labelColor=t["ink"], labelFontSize=13, ticks=False, domain=False))
    bars = alt.Chart(by_model).mark_bar(
        size=28, stroke=t["surface"], strokeWidth=2, cornerRadius=4,
    ).encode(
        y=base_y,
        x=alt.X(f"{value}:Q", stack="zero", title=title,
                axis=alt.Axis(format=axis_format, labelColor=t["muted"], titleColor=t["muted"],
                              gridColor=t["grid"], domain=False, tickCount=5)),
        color=alt.Color("Claude model:N", scale=alt.Scale(domain=MODEL_ORDER, range=t["series"]),
                        legend=alt.Legend(orient="top", title=None, labelColor=t["ink"])),
        order=alt.Order("model_rank:Q"),
        tooltip=[alt.Tooltip("Scenario:N"), alt.Tooltip("Claude model:N"),
                 alt.Tooltip(f"{value}:Q", format=tooltip_format),
                 alt.Tooltip("Share:Q", format=".0%", title="Share of scenario")],
    )
    labels = alt.Chart(totals).mark_text(align="left", dx=6, fontSize=13, fontWeight="bold",
                                         color=t["ink"]).encode(
        y=base_y, x=f"{value}:Q", text="Label:N",
    )
    return (bars + labels).properties(height=60 * len(scenarios) + 10)


with tab_econ:
    st.subheader("What would this workload cost on Claude?")
    st.caption(
        "Compare three scenarios for a monthly workload: everything on Opus 5, routed to the "
        "right Claude model per task, and routed plus prompt caching and the Batch API. "
        "No API calls — every assumption is editable below."
    )

    if "mix" not in st.session_state:
        st.session_state.mix = te.default_mix()
        st.session_state.mix_version = 0

    # ---- 1. workload
    st.markdown("#### 1 · Workload")
    with st.container(border=True):
        monthly_requests = st.number_input("Requests per month", min_value=1, value=100_000,
                                           step=10_000, format="%d")

        with st.expander("📥 Build the mix from real prompts (CSV)"):
            st.write(f"Upload a CSV with a **prompt** column. Up to {CSV_ROW_LIMIT} prompts are "
                     "classified on your Azure deployment (one small call each). The task mix and "
                     "average input size are taken from them; output sizes keep the table's values.")
            upload = st.file_uploader("Prompts CSV", type="csv", label_visibility="collapsed")
            if upload is not None:
                try:
                    prompts_df = pd.read_csv(upload)
                except Exception as e:
                    st.error(f"Could not read the CSV: {e}")
                    prompts_df = None
                if prompts_df is not None and "prompt" not in prompts_df.columns:
                    st.error("The CSV needs a column named `prompt`.")
                elif prompts_df is not None:
                    prompts = prompts_df["prompt"].dropna().astype(str).head(CSV_ROW_LIMIT).tolist()
                    if st.button(f"Classify {len(prompts)} prompts on Azure"):
                        progress = st.progress(0.0, text="Classifying…")
                        classified, failures = [], 0
                        for i, p in enumerate(prompts, 1):
                            try:
                                c = classify_task(p)
                                classified.append({
                                    "Category": c["category"], "Complexity": c["complexity"],
                                    # ~4 characters per token: an estimate, not a Claude count.
                                    "Input tokens": max(1, len(p) / 4),
                                })
                            except Exception:
                                failures += 1
                            progress.progress(i / len(prompts), text=f"Classified {i}/{len(prompts)}")
                        if classified:
                            st.session_state.mix = te.mix_from_classified(
                                pd.DataFrame(classified), st.session_state.mix)
                            st.session_state.mix_version += 1
                            st.success(f"Mix built from {len(classified)} prompts"
                                       + (f" ({failures} failed)" if failures else "") + ".")
                        else:
                            st.error("No prompts could be classified. Check the Setup panel.")

        st.write("**Task mix** — edit shares and average token sizes per request "
                 "(input includes system prompt and context).")
        mix_view = st.session_state.mix.copy()
        mix_view.insert(2, "Claude model", [CLAUDE_MATRIX[(c, x)].model.model_id
                                            for c, x in zip(mix_view["Category"], mix_view["Complexity"])])
        edited = st.data_editor(
            mix_view, key=f"mix_editor_{st.session_state.mix_version}",
            hide_index=True, width="stretch",
            disabled=["Category", "Complexity", "Claude model"],
            column_config={
                "Share %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=0.5),
                "Avg input tokens": st.column_config.NumberColumn(min_value=0, step=100),
                "Avg output tokens": st.column_config.NumberColumn(min_value=0, step=100),
            },
        )
        mix = edited.drop(columns=["Claude model"])
        share_total = mix["Share %"].sum()
        if abs(share_total - 100) > 0.5:
            st.warning(f"Shares add up to {share_total:.1f}% — they are scaled to 100% for the calculation.")
        c_reset, _ = st.columns([1, 4])
        if c_reset.button("Reset to default mix"):
            st.session_state.mix = te.default_mix()
            st.session_state.mix_version += 1
            st.rerun()

    # ---- 2. assumptions
    st.markdown("#### 2 · Assumptions")
    with st.container(border=True):
        a_col1, a_col2 = st.columns(2)
        with a_col1:
            st.markdown("**Cost levers**")
            cacheable = st.slider("Cacheable share of input (stable prefix)", 0, 100, 50, format="%d%%",
                                  key="cacheable",
                                  help="System prompt, instructions, shared documents that repeat across requests.")
            hit_rate = st.slider("Cache hit rate", 0, 100, 80, format="%d%%", key="hit_rate",
                                 help="Hits are billed at 0.1× input price; misses write the cache at 1.25×.")
            batch = st.slider("Share of requests via Batch API", 0, 100, 0, format="%d%%", key="batch",
                              help="Batch API is 50% off for work that doesn't need an immediate answer.")
        with a_col2:
            st.markdown("**Token assumptions**")
            tok_factor = st.slider("Tokenizer factor", 0.8, 1.5, 1.0, 0.05, key="tok_factor",
                                   help="Set above 1.0 if your token counts come from another model's "
                                        "tokenizer. Opus 5's tokenizer can count up to ~1.35× more tokens.")
            st.caption("Thinking tokens per visible output token, by effort (billed as output):")
            defaults = te.Assumptions().thinking_overhead
            e_cols = st.columns(5)
            thinking = {lvl: e_cols[i].number_input(lvl, 0.0, 10.0, defaults[lvl], 0.25, key=f"think_{lvl}")
                        for i, lvl in enumerate(defaults)}
        st.caption("⚠️ Thinking overhead, cache rates and the tokenizer factor are **assumptions**. "
                   "Replace them with measured values from real Claude usage when you have it.")

    assumptions = te.Assumptions(tokenizer_factor=tok_factor, thinking_overhead=thinking,
                                 cacheable_share=cacheable / 100, cache_hit_rate=hit_rate / 100,
                                 batch_share=batch / 100)
    detail = te.compute(mix, int(monthly_requests), assumptions)

    # ---- 3. results
    st.markdown("#### 3 · Monthly cost")
    totals = detail.groupby("Scenario", sort=False)["Monthly cost"].sum().reindex(te.SCENARIOS)
    baseline, routed, optimised = totals.tolist()

    with st.container(border=True):
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("All on Opus 5", _money(baseline))
        k2.metric("Routed", _money(routed),
                  delta=f"-{(1 - routed / baseline):.0%} vs all-Opus" if baseline else None,
                  delta_color="inverse")
        k3.metric("Routed + optimised", _money(optimised),
                  delta=f"-{(1 - optimised / baseline):.0%} vs all-Opus" if baseline else None,
                  delta_color="inverse")
        k4.metric("Cost per 1,000 requests", _money(optimised / monthly_requests * 1000),
                  help="Routed + optimised scenario.")

        by_model = detail.groupby(["Scenario", "Claude model"], sort=False)["Monthly cost"].sum().reset_index()
        by_model["model_rank"] = by_model["Claude model"].map(MODEL_ORDER.index)
        by_model["Share"] = by_model["Monthly cost"] / by_model["Scenario"].map(totals)
        total_df = totals.reset_index()
        total_df["Label"] = total_df["Monthly cost"].map(_money)
        st.altair_chart(
            _scenario_chart(by_model, total_df, value="Monthly cost", title="Monthly cost (USD)",
                            axis_format="$,.0f", tooltip_format="$,.2f", scenarios=te.SCENARIOS),
            width="stretch", theme="streamlit")

        view_chart_table, view_detail = st.tabs(["📊 Chart data", "🔍 Cost by task type"])
        with view_chart_table:
            pivot = by_model.pivot(index="Scenario", columns="Claude model", values="Monthly cost")
            pivot = pivot.reindex(index=te.SCENARIOS, columns=[m for m in MODEL_ORDER if m in pivot.columns])
            pivot["Total"] = pivot.sum(axis=1)
            st.dataframe(pivot.fillna(0).style.format("${:,.2f}"), width="stretch")
        with view_detail:
            scen = st.radio("Scenario", te.SCENARIOS, index=2, horizontal=True)
            d = detail[detail["Scenario"] == scen].drop(columns="Scenario")
            st.dataframe(
                d.sort_values("Monthly cost", ascending=False), hide_index=True, width="stretch",
                column_config={
                    "Requests / month": st.column_config.NumberColumn(format="localized"),
                    "Cost / request": st.column_config.NumberColumn(format="$%.5f"),
                    "Monthly cost": st.column_config.NumberColumn(format="dollar"),
                },
            )

    with st.expander("How the cost is calculated"):
        st.markdown(
            f"""
Per request, for the Claude model and effort of each task type:

- **Input** = input tokens × tokenizer factor × input price. With caching, the cacheable share
  is billed at **{te.CACHE_READ_MULTIPLIER}×** on a hit and **{te.CACHE_WRITE_MULTIPLIER}×** on a miss.
- **Output** = (output tokens + thinking tokens) × tokenizer factor × output price.
  Thinking = output × the overhead for that effort level; Haiku 4.5 has none.
- **Batch** = the batched share of requests is billed at **{1 - te.BATCH_DISCOUNT:.0%}** of the price.

*All on Opus 5* uses effort `{te.BASELINE_EFFORT}` with no caching or batch. *Routed* uses each task type's
Claude match. *Routed + optimised* adds caching and batch. Prices are Anthropic list prices per 1M tokens:
Opus 5 $5/$25, Sonnet 5 $2/$10, Haiku 4.5 $1/$5. Caching needs a prefix above a model-specific minimum
length, and batch results can take up to 24 hours.
"""
        )


# ---------------------------------------------------------------- tab: water footprint
with tab_water:
    st.subheader("How much water would this use on Claude?")
    st.caption(
        "An estimate from token counts: tokens → energy → water for data-centre cooling "
        "(on-site) and for generating the electricity (off-site). Anthropic does not publish "
        "energy or water per token, so every factor is an editable assumption (section 3)."
    )
    w = _water_assumptions()

    # ---- 1. this prompt
    st.markdown("#### 1 · Your prompt")
    result = st.session_state.get("result")
    if not result:
        st.info("Classify a prompt in the **🔎 Find a model** tab to see its water footprint here.")
    else:
        rec = result["rec"]
        a = _assumptions()
        p_in = max(1, round(len(result["prompt"]) / 4)) + st.session_state.get("usage_context", 0)
        p_out = st.session_state.get("usage_output", 800)
        p_monthly = st.session_state.get("usage_monthly", 10_000)
        f = wt.request_footprint(rec.model, rec.effort, p_in, p_out, a, w)

        st.caption(f"On **{rec.model.display_name}**"
                   + (f" at effort `{rec.effort}`" if rec.effort else "")
                   + f" · {f['input_tokens']:,.0f} input + {f['generated_tokens']:,.0f} generated "
                     "tokens (request size from the 🪙 Token usage & cost tab).")
        with st.container(border=True):
            st.html(_card_grid([
                _water_card("Per request", f["total_ml"]),
                _water_card("Per 1,000 requests", f["total_ml"] * 1000),
                _water_card(f"Per month · {p_monthly:,} requests", f["total_ml"] * p_monthly),
            ]))

        compare = []
        for m in (OPUS, SONNET, HAIKU):
            effort = rec.effort if m == rec.model else COMPARE_EFFORT[m.model_id]
            cf = wt.request_footprint(m, effort, p_in, p_out, a, w)
            compare.append({
                "Claude model": m.model_id,
                "": "✅ recommended" if m == rec.model else "",
                "Effort": effort or "—",
                "Energy Wh / request": cf["facility_wh"],
                "Water mL / request": cf["total_ml"],
                "Bottles / month": cf["total_ml"] * p_monthly / wt.BOTTLE_ML,
                "Litres / month": cf["total_ml"] * p_monthly / 1000,
                "vs recommended": f"{cf['total_ml'] / f['total_ml']:.1f}×" if f["total_ml"] else "—",
            })
        st.markdown(f"**Same request on each Claude model** · per month, {p_monthly:,} requests")
        st.dataframe(
            pd.DataFrame(compare), hide_index=True, width="stretch",
            column_config={
                "Energy Wh / request": st.column_config.NumberColumn(format="%.2f"),
                "Water mL / request": st.column_config.NumberColumn(format="%.1f"),
                "Bottles / month": st.column_config.NumberColumn(format="localized"),
                "Litres / month": st.column_config.NumberColumn(format="localized"),
            },
        )

    # ---- 2. workload
    st.markdown("#### 2 · Your workload")
    st.caption(f"Uses the task mix, {int(monthly_requests):,} requests/month and token assumptions "
               "from the 💰 Token economics tab. Caching and batch change the price, not the "
               "compute, so they are left out here.")
    wdetail = wt.compute(mix, int(monthly_requests), assumptions, w)
    wdetail.insert(wdetail.columns.get_loc("Water L / month"), "Bottles / month",
                   wdetail["Water L / month"] * 1000 / wt.BOTTLE_ML)
    wtotals = wdetail.groupby("Scenario", sort=False)["Water L / month"].sum().reindex(wt.SCENARIOS)
    w_base, w_routed = wtotals.tolist()
    saved_ml = max(0.0, w_base - w_routed) * 1000

    with st.container(border=True):
        st.html(_card_grid([
            _water_card("All on Opus 5 · per month", w_base * 1000),
            _water_card("Routed · per month", w_routed * 1000),
            _water_card("Saved by routing · per month", saved_ml),
        ]))

        # Chart in bottles (stored values are litres).
        to_unit = 1000 / wt.BOTTLE_ML
        val = "Bottles / month"
        wby_model = (wdetail.groupby(["Scenario", "Claude model"], sort=False)["Water L / month"]
                     .sum().reset_index())
        wby_model[val] = wby_model.pop("Water L / month") * to_unit
        wby_model["model_rank"] = wby_model["Claude model"].map(MODEL_ORDER.index)
        wby_model["Share"] = wby_model[val] / wby_model["Scenario"].map(wtotals * to_unit)
        wtotal_df = (wtotals * to_unit).rename(val).reset_index()
        wtotal_df["Label"] = [f"{_bottles(v * 1000)} · {_water(v * 1000)}" for v in wtotals]
        st.altair_chart(
            _scenario_chart(wby_model, wtotal_df, value=val,
                            title="500 mL bottles per month",
                            axis_format=",.0f", tooltip_format=",.0f",
                            scenarios=wt.SCENARIOS),
            width="stretch", theme="streamlit")

        w_view_chart, w_view_detail = st.tabs(["📊 Chart data", "🔍 Water by task type"])
        with w_view_chart:
            wpivot = (wdetail.pivot_table(index="Scenario", columns="Claude model",
                                          values="Water L / month", aggfunc="sum")
                      .reindex(index=wt.SCENARIOS))
            wpivot = wpivot[[m for m in MODEL_ORDER if m in wpivot.columns]]
            wpivot["Total"] = wpivot.sum(axis=1)
            wpivot = wpivot.fillna(0)
            st.dataframe(pd.concat({"Litres / month": wpivot,
                                    "Bottles / month": wpivot * 1000 / wt.BOTTLE_ML}, axis=1)
                         .style.format("{:,.0f}"), width="stretch")
        with w_view_detail:
            w_scen = st.radio("Scenario", wt.SCENARIOS, index=1, horizontal=True, key="w_scen")
            wd = wdetail[wdetail["Scenario"] == w_scen].drop(columns="Scenario")
            st.dataframe(
                wd.sort_values("Water L / month", ascending=False), hide_index=True, width="stretch",
                column_config={
                    "Requests / month": st.column_config.NumberColumn(format="localized"),
                    "mL / request": st.column_config.NumberColumn(format="%.1f"),
                    "Energy kWh / month": st.column_config.NumberColumn(format="%.1f"),
                    "On-site L / month": st.column_config.NumberColumn(format="%.1f"),
                    "Off-site L / month": st.column_config.NumberColumn(format="%.1f"),
                    "Bottles / month": st.column_config.NumberColumn(format="localized"),
                    "Water L / month": st.column_config.NumberColumn(format="%.1f"),
                },
            )

    # ---- 3. assumptions
    st.markdown("#### 3 · Assumptions")
    with st.container(border=True):
        wd_defaults = wt.WaterAssumptions()
        wa1, wa2 = st.columns(2)
        with wa1:
            st.markdown("**Energy per 1,000 generated tokens (Wh)**")
            st.caption("Output and thinking tokens. Not published by Anthropic — assumed to grow "
                       "with model size.")
            e_cols = st.columns(3)
            for i, (mid, v) in enumerate(wd_defaults.wh_per_1k_output.items()):
                e_cols[i].number_input(mid, 0.0, 50.0, v, 0.1, key=f"wh_{mid}")
            st.slider("Input token energy vs output token", 0.0, 0.5, wd_defaults.input_energy_ratio,
                      0.01, key="w_input_ratio",
                      help="Reading the prompt is processed in parallel and costs far less per "
                           "token than generating the answer.")
        with wa2:
            st.markdown("**Data centre and grid**")
            st.slider("PUE (facility ÷ IT energy)", 1.0, 2.0, wd_defaults.pue, 0.05, key="w_pue",
                      help="Hyperscale data centres report about 1.1–1.2.")
            st.slider("On-site WUE (L per kWh of IT energy)", 0.0, 2.0, wd_defaults.wue_onsite,
                      0.05, key="w_wue",
                      help="Cooling water. Varies with climate and cooling design; reported "
                           "hyperscaler figures range from about 0.1 to over 1.")
            st.slider("Off-site water intensity (L per kWh of electricity)", 0.0, 6.0,
                      wd_defaults.ewif_offsite, 0.1, key="w_ewif",
                      help="Water consumed by power generation. About 3 L/kWh is a US-average "
                           "figure; it is much lower for wind and solar.")
        st.caption("⚠️ These are **assumptions** for an order-of-magnitude estimate. Replace them "
                   "with your provider's and region's figures when you have them.")

    with st.expander("How the water is calculated"):
        st.markdown(
            """
Per request, for the Claude model and effort of each task type:

- **Generated tokens** = output tokens + thinking tokens (thinking = output × the overhead for
  that effort level, from the Token economics tab), × tokenizer factor.
- **IT energy** = generated tokens × Wh per token for the model
  + input tokens × Wh per token × input ratio.
- **On-site water** = IT energy × WUE (cooling).
- **Off-site water** = IT energy × PUE × off-site water intensity (electricity generation).

*All on Opus 5* uses effort `high` for every task; *Routed* uses each task type's Claude match.
Water here means water **consumed** (evaporated), not water withdrawn and returned. The method
follows the on-site/off-site split in *Making AI Less Thirsty* (Li et al., 2023). Training the
models, hardware manufacturing and idle capacity are not included.
"""
        )


# ---------------------------------------------------------------- tab: how it works
with tab_about:
    st.subheader("What it gives leaders")
    p1, p2, p3 = st.columns(3)
    with p1, st.container(border=True):
        st.markdown("#### 💰 Cost saving")
        st.write("Match each task to the smallest Claude model that does it well, instead of "
                 "sending everything to the frontier model. Add prompt caching and the Batch "
                 "API for further savings.")
    with p2, st.container(border=True):
        st.markdown("#### 📈 Budget planning")
        st.write("Turn a workload mix into a monthly and 12-month budget under each scenario, "
                 "with growth and an optional budget cap — before signing a contract.")
    with p3, st.container(border=True):
        st.markdown("#### 💧 Water awareness")
        st.write("Translate token use into energy and water, in litres and 500 mL bottles, so "
                 "model choice becomes a sustainability decision too.")

    st.subheader("How it works · Task → Tier → Test")
    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("#### 1 · Task")
        st.write("Each prompt is classified by category (coding, reasoning, writing, "
                 "extraction, general) and complexity (low, medium, high) by a low-cost LLM.")
    with c2, st.container(border=True):
        st.markdown("#### 2 · Tier")
        st.write("A plain lookup table maps the classification to a tier and to the matching "
                 "Claude model and effort level. No hidden logic — see the Routing table tab.")
    with c3, st.container(border=True):
        st.markdown("#### 3 · Test")
        st.write("Validate the recommendations on your own prompts: `python main.py eval` scores "
                 "answers on a golden set, so the policy rests on evidence, not benchmarks.")

    st.subheader("Good to know")
    st.markdown(
        "- **No Claude API key is needed.** The recommendation is a lookup; only the "
        "classification calls an LLM; all token, cost and water figures are calculated.\n"
        "- **Provider-agnostic.** The classifier runs on Azure OpenAI today and can run on any "
        "provider; the same approach extends to other model families.\n"
        "- **Figures are planning estimates.** Costs use Anthropic list prices; thinking "
        "tokens, caching and water factors are editable assumptions, not measurements.\n"
        "- **Effort** is Claude's quality/cost dial on Opus 5 and Sonnet 5: "
        "`low` → `medium` → `high` → `xhigh` → `max`."
    )


# ---------------------------------------------------------------- tab: executive overview
# Filled last so it reflects the workload and assumptions set in the other tabs.
PLAN_MONTHS = 12


def _budget_chart(plan: pd.DataFrame, budget: float) -> alt.LayerChart:
    """One 2px line per scenario over the plan, end-labelled, with a hover crosshair."""
    t = _theme()
    color = alt.Color("Scenario:N", scale=alt.Scale(domain=te.SCENARIOS, range=t["plan"]),
                      legend=alt.Legend(orient="top", title=None, labelColor=t["ink"]))
    x = alt.X("Month:T", title=None,
              axis=alt.Axis(format="%b %y", labelColor=t["muted"], grid=False, domain=False,
                            tickCount=PLAN_MONTHS))
    y = alt.Y("Monthly cost:Q", title="Monthly cost (USD)",
              axis=alt.Axis(format="$,.0f", labelColor=t["muted"], titleColor=t["muted"],
                            gridColor=t["grid"], domain=False, tickCount=5))
    base = alt.Chart(plan).encode(x=x, y=y, color=color)
    lines = base.mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round")

    hover = alt.selection_point(fields=["Month"], nearest=True, on="pointerover",
                                empty=False, clear="pointerout")
    wide = plan.pivot(index="Month", columns="Scenario", values="Monthly cost").reset_index()
    rule = alt.Chart(wide).mark_rule(color=t["muted"]).encode(
        x="Month:T",
        opacity=alt.condition(hover, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("Month:T", format="%B %Y")]
                + [alt.Tooltip(f"{s}:Q", format="$,.0f") for s in te.SCENARIOS],
    ).add_params(hover)
    points = base.mark_point(filled=True, size=70, stroke=t["surface"], strokeWidth=2).encode(
        opacity=alt.condition(hover, alt.value(1), alt.value(0)))

    # Label only the outer lines; the two routed lines sit too close to label both.
    last = plan[(plan["Month"] == plan["Month"].max())
                & plan["Scenario"].isin([te.SCENARIOS[0], te.SCENARIOS[-1]])].copy()
    last["Label"] = last["Monthly cost"].map(_money)
    labels = alt.Chart(last).mark_text(align="left", dx=8, fontSize=12, fontWeight="bold",
                                       color=t["ink"]).encode(x=x, y=y, text="Label:N")
    layers = [lines, rule, points, labels]
    if budget > 0:
        cap = pd.DataFrame({"Monthly cost": [budget], "Label": [f"Budget {_money(budget)}"]})
        layers.append(alt.Chart(cap).mark_rule(color=t["ink"], strokeDash=[4, 4], strokeWidth=1)
                      .encode(y="Monthly cost:Q"))
        layers.append(alt.Chart(cap).mark_text(align="left", dy=-8, x=4, color=t["ink"],
                                               fontSize=12).encode(y="Monthly cost:Q", text="Label:N"))
    return alt.layer(*layers).properties(height=340, padding={"right": 70})


with tab_overview:
    st.subheader("AI spend, budget and footprint at a glance")
    st.caption(f"Based on the workload in **💰 Token economics** — {int(monthly_requests):,} requests "
               "a month and the task mix there. Change it there; every figure here updates.")

    g1, g2, _ = st.columns([1, 1, 1])
    growth = g1.slider("Expected monthly growth in AI usage", 0, 25, 5, format="%d%%",
                       key="plan_growth") / 100
    budget = g2.number_input("Monthly AI budget (optional)", min_value=0, value=0, step=500,
                             key="plan_budget", help="Draws a budget line on the plan.")

    start = (pd.Timestamp.today().normalize() + pd.offsets.MonthBegin(1))
    months = pd.date_range(start, periods=PLAN_MONTHS, freq="MS")
    factors = [(1 + growth) ** i for i in range(PLAN_MONTHS)]
    plan = pd.DataFrame([
        {"Month": mo, "Scenario": s, "Monthly cost": cost * f}
        for s, cost in zip(te.SCENARIOS, (baseline, routed, optimised))
        for mo, f in zip(months, factors)
    ])
    annual = plan.groupby("Scenario", sort=False)["Monthly cost"].sum().reindex(te.SCENARIOS)
    annual_saving = annual.iloc[0] - annual.iloc[2]
    water_saved_year_ml = saved_ml * sum(factors)

    with st.container(border=True):
        k1, k2, k3, k4 = st.columns(4)
        bottles_saved = water_saved_year_ml / wt.BOTTLE_ML
        k1.metric("All on Opus 5 · 12 mo", _money(annual.iloc[0]),
                  help="Every request on the frontier model, no caching or batch.")
        k2.metric("With the Advisor · 12 mo", _money(annual.iloc[2]),
                  delta=f"-{annual_saving / annual.iloc[0]:.0%}" if annual.iloc[0] else None,
                  delta_color="inverse", help="Routed to the right model, plus caching and batch.")
        k3.metric("Saving · 12 mo", _money(annual_saving))
        k4.metric("Water saved · 12 mo", _water(water_saved_year_ml),
                  delta=(f"≈ {bottles_saved / 1000:,.1f}k bottles" if bottles_saved >= 10_000
                         else f"≈ {_bottles(water_saved_year_ml)}"), delta_color="off",
                  delta_arrow="off",
                  help="500 mL bottles. Routing alone, at the Water footprint tab's assumptions.")

    st.markdown("#### 12-month budget plan")
    with st.container(border=True):
        st.altair_chart(_budget_chart(plan, budget), width="stretch", theme="streamlit")
        if budget > 0:
            over = []
            for s in te.SCENARIOS:
                rows = plan[(plan["Scenario"] == s) & (plan["Monthly cost"] > budget)]
                if not rows.empty:
                    over.append(f"**{s}** from {rows['Month'].min():%B %Y}")
            if over:
                st.warning("Above the monthly budget: " + "; ".join(over) + ".")
            else:
                st.success("All three scenarios stay within the monthly budget for 12 months.")

        plan_table = (plan.pivot(index="Month", columns="Scenario", values="Monthly cost")
                      [te.SCENARIOS].reset_index())
        plan_table["Month"] = plan_table["Month"].dt.strftime("%b %Y")
        with st.expander("📊 Plan data"):
            st.dataframe(plan_table, hide_index=True, width="stretch",
                         column_config={s: st.column_config.NumberColumn(format="dollar")
                                        for s in te.SCENARIOS})
        st.download_button("⬇️ Download budget plan (CSV)", plan_table.to_csv(index=False),
                           file_name="ai_budget_plan.csv", mime="text/csv")

    st.markdown("#### Where the work goes")
    routed_detail = detail[detail["Scenario"] == "Routed"]
    req_share = (routed_detail.groupby("Claude model")["Requests / month"].sum()
                 / routed_detail["Requests / month"].sum())
    with st.container(border=True):
        cols = st.columns(3)
        for col, m in zip(cols, (OPUS, SONNET, HAIKU)):
            col.metric(m.display_name, f"{req_share.get(m.model_id, 0):.0%} of requests")
        st.caption("Most everyday work does not need the frontier model — that is where the "
                   "saving comes from.")

    st.caption("⚠️ Planning estimates at Anthropic list prices with editable assumptions — "
               "not a quote. Water figures are order-of-magnitude.")
