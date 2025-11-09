# app.py
# SWOT → Problem Statements → Project Charters (S2C)
# Streamlit + OpenAI (Chat Completions)
# Now auto-connects via environment variable or Streamlit secrets (no input field).

import os
import io
import textwrap
import pandas as pd
import streamlit as st
from typing import List

# ---------------- OpenAI client (same pattern as TRIZ app) ----------------
try:
    from openai import OpenAI
    OPENAI_IMPORT_OK = True
except Exception:
    OPENAI_IMPORT_OK = False


@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client from Streamlit Secrets or environment variable."""
    if not OPENAI_IMPORT_OK:
        return None
    api_key = (
        st.secrets.get("openai", {}).get("api_key")
        if "openai" in st.secrets
        else os.getenv("OPENAI_API_KEY", "")
    )
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


client = get_openai_client()

# ---------------- Page & minimal styles ----------------
APP_TITLE = "SWOT → Problem Statements → Project Charters (S2C)"
MODEL_NAME_DEFAULT = "gpt-4o-mini"   # swap to gpt-4o / gpt-4.1 if desired
DMAIC_DEPARTMENTS = [
    "Sales Ops","Service Ops","Finance Ops","Procurement","IT","QA",
    "Compliance","Supply Chain","Manufacturing","Shared Services","Training","PMO"
]
ORG_TIERS = [
    "Executive Tier (CEO, COO, CFO, CTO, President)",
    "Senior Management Tier (Vice Presidents, Directors, General Managers)",
    "Middle Management Tier (Managers, Team Leaders, Department Heads)",
    "Supervisory Tier (Supervisors, Shift Leaders, Coordinators)",
    "Operational Tier (Staff, Associates, Technicians, Entry-Level Employees)",
]

st.set_page_config(page_title=APP_TITLE, layout="wide")

# ---------------- Sidebar: API badge only ----------------
with st.sidebar:
    st.markdown("### API Status")
    if client is None:
        st.markdown(
            """
            <div style="background-color:#FDEDEC;padding:10px;border-radius:6px;
                        border:1px solid #F5B7B1;font-weight:600;color:#C0392B;">
              ❌ OpenAI: Not Connected
            </div>
            <div style="font-size:0.9em;margin-top:6px;">
              Add <code>[openai] api_key</code> in <b>Secrets</b> or set <code>OPENAI_API_KEY</code>.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background-color:#E9F7EF;padding:10px;border-radius:6px;
                        border:1px solid #D4EFDF;font-weight:600;color:#1D8348;">
              ✅ OpenAI: Connected
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------- Main UI ----------------
st.title(APP_TITLE)
st.caption("OpenAI-powered DMAIC portfolio generator — optional Organizational Tier alignment")

# Model & temperature (dropdown selector)
top1, top2 = st.columns([2,1])
with top1:
    st.markdown("**Model**")
    model_name = st.selectbox(
        "",
        options=[
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-4.1",
            "gpt-3.5-turbo",
        ],
        index=0,
        help="Select which OpenAI model to use for analysis"
    )
with top2:
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

# Context
st.markdown("#### Context")
c1, c2, c3 = st.columns(3)
with c1:
    region = st.text_input("Region/Market (optional)", value="APAC")
with c2:
    industry = st.text_input("Industry (optional)", value="Shared Services")
with c3:
    fy_start = st.text_input("Fiscal Year start (e.g., Jan 1)", value="Jan 1")

# Organizational tiers
st.markdown("#### Organizational Tier (optional)")
include_tiers = st.checkbox("Include Organizational Tier(s) in charters & portfolio", True)
tiers_selected: List[str] = []
if include_tiers:
    tiers_selected = st.multiselect(
        "Select relevant tiers (ordered high → low in governance)",
        ORG_TIERS,
        default=[ORG_TIERS[1], ORG_TIERS[2], ORG_TIERS[4]],
    )

st.markdown("---")

# SWOT inputs
a, b = st.columns(2)
with a:
    st.subheader("Strengths (one per line)")
    strengths_text = st.text_area("Strengths", height=160, label_visibility="collapsed")
with b:
    st.subheader("Weaknesses (one per line)")
    weaknesses_text = st.text_area("Weaknesses", height=160, label_visibility="collapsed")

c, d = st.columns(2)
with c:
    st.subheader("Opportunities (one per line)")
    opportunities_text = st.text_area("Opportunities", height=160, label_visibility="collapsed")
with d:
    st.subheader("Threats (one per line)")
    threats_text = st.text_area("Threats", height=160, label_visibility="collapsed")

st.markdown("---")

# ---------------- Prompt templates ----------------
SYSTEM_TEXT = textwrap.dedent("""
You are an AI Lean Six Sigma Black Belt coach. You ingest SWOT sticky notes (70–100 items), cluster them by theme, and produce:
1) a Theme Map per quadrant,
2) compliant Problem Statements (specific, data-driven, one problem, no solution),
3) a prioritized list (Impact × Controllability, 0–5 each, with rationale),
4) draft DMAIC Project Charters with departments, CTQs, savings band, duration, and start date.

Follow IASSC/ASQ BoK structure. Be explicit and concise. Use only information inferred from notes plus reasonable Lean heuristics; do not invent precise numbers without labeling them as estimates.
""").strip()

USER_TEMPLATE = textwrap.dedent("""
Context:
- Region/Market (optional): {region}
- Industry (optional): {industry}
- Fiscal Year start (e.g., Jan 1): {fy_start}

Input Notes:
Strengths:
{strengths_text}

Weaknesses:
{weaknesses_text}

Opportunities:
{opportunities_text}

Threats:
{threats_text}

Tasks:
A) Cluster notes into 2–5 themes per quadrant. Name each theme and show a 1–2 line rationale.
B) From Weaknesses and Threats only, propose 3–8 Problem Statements that obey ALL rules:
   - Specific & data-driven (use counts/percents/ranges if present; else clearly mark as "estimate").
   - One problem per statement.
   - No solutions.
C) Score each proposed problem: Impact (0–5) × Controllability (0–5). Explain in one sentence.
D) For top 3 problems, generate a one-page DMAIC Charter with:
   - Title, Problem Statement, Business Case
   - Goal (SMART, with target and date)
   - Scope / Out of Scope
   - CTQs & baseline (state if estimated)
   - Departments / Teams to involve (choose from: {dept_menu})
   - Savings band (Low: < $100k; Med: $100k–$500k; High: $500k–$1.5M; Very High: > $1.5M) + rationale
   - Duration (Kaizen 2–4w, DMAIC Light 8–12w, Full DMAIC 16–24w) + rationale
   - Proposed Start Date (nearest quarter boundary or ≥30 days from today, aligned to {fy_start})
   - High-level DMAIC plan (bulleted by phase)
E) Return a compact portfolio table (CSV-style) listing:
   Problem_ID, Theme, Impact, Control, Savings_Band, Duration, Dept(s), Proposed_Start{tier_col_hint}.

Output format:
1) THEME MAP
2) PROBLEM STATEMENTS (rule-check table: ✓/✗ for "specific", "single problem", "no solution")
3) PRIORITIZED LIST (with Impact×Control and rationale)
4) PROJECT CHARTERS (3){tier_block_instruction}
5) PORTFOLIO TABLE

{tier_guidance}
""").strip()


def build_user_prompt() -> str:
    tier_block_instruction = ""
    tier_col_hint = ""
    tier_guidance = ""
    if include_tiers and tiers_selected:
        tier_block_instruction = "\n   - Include a field: Organizational Tier(s) Impacted, with governance notes aligned to selected tiers."
        tier_col_hint = ", Org_Tier(s)"
        tier_guidance = (
            "Organizational Tier guidance:\n"
            f"- Selected tiers (high → low): {' > '.join(tiers_selected)}\n"
            "- Align stakeholder engagement, approvals, and communications to these tiers.\n"
        )
    return USER_TEMPLATE.format(
        region=region,
        industry=industry,
        fy_start=fy_start,
        strengths_text=strengths_text.strip(),
        weaknesses_text=weaknesses_text.strip(),
        opportunities_text=opportunities_text.strip(),
        threats_text=threats_text.strip(),
        dept_menu=", ".join(DMAIC_DEPARTMENTS),
        tier_block_instruction=tier_block_instruction,
        tier_col_hint=tier_col_hint,
        tier_guidance=tier_guidance
    )

# ---------------- OpenAI call ----------------
def call_openai(system_text: str, user_text: str, model_name: str, temperature: float) -> str:
    if client is None:
        return "⚠️ OpenAI client not initialized. Add your API key in Secrets or env."
    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating response: {e}"

# ---------------- Helpers ----------------
def extract_csv_block(text: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    for header in [
        "Problem_ID,Theme,Impact,Control,Impact×Control,Savings_Band,Duration,Dept(s),Proposed_Start,Org_Tier(s)",
        "Problem_ID,Theme,Impact,Control,Savings_Band,Duration,Dept(s),Proposed_Start,Org_Tier(s)",
        "Problem_ID,Theme,Impact,Control,Savings_Band,Duration,Dept(s),Proposed_Start",
    ]:
        if header in text:
            idx = text.index(header)
            return text[idx:].split("\n\n")[0].strip()
    return ""

# ---------------- Actions ----------------
left, right = st.columns([1,1])
with left:
    if st.button("Generate Portfolio", type="primary", use_container_width=True):
        user_prompt = build_user_prompt()
        output = call_openai(SYSTEM_TEXT, user_prompt, model_name, temperature)

        st.subheader("Model Output")
        st.markdown(output)

        csv_block = extract_csv_block(output)
        if csv_block:
            st.markdown("---")
            st.subheader("Portfolio CSV")
            st.code(csv_block, language="csv")
            try:
                df = pd.read_csv(io.StringIO(csv_block))
                st.dataframe(df, use_container_width=True)
                st.download_button(
                    "Download portfolio.csv",
                    data=df.to_csv(index=False).encode("utf-8"),
                    file_name="portfolio.csv",
                    mime="text/csv"
                )
            except Exception:
                st.download_button(
                    "Download portfolio.csv",
                    data=csv_block.encode("utf-8"),
                    file_name="portfolio.csv",
                    mime="text/csv"
                )

with right:
    st.download_button(
        "Download Combined Prompt (System + User)",
        data=(SYSTEM_TEXT + "\n\n" + build_user_prompt()).encode("utf-8"),
        file_name="s2c_prompt_openai.txt",
        mime="text/plain",
        use_container_width=True
    )

st.markdown("---")
with st.expander("Show System Instruction"):
    st.code(SYSTEM_TEXT)
with st.expander("Show Composed User Prompt"):
    st.code(build_user_prompt())

