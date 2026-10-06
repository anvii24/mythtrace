"""TrustRAG demo app (Streamlit), for recording the demo video.

    streamlit run app.py

Display only: every number on screen comes from the existing code - ranking.search.search(),
rag.answer.answer() and the positional index (ranking.score.get_state). Nothing here changes how
chunks are ranked, which defenses fire, or what the LLM is asked. LLM answers go through the same
on-disk cache as the experiments (data/llm_cache/), so a question that was asked before is free.

Tabs:
  Investigate  - ask a question: answer + citations, results table, query-term highlighting,
                 per-result score autopsy, query trace (tokens -> stems -> df/idf/postings -> heap
                 top-K), and a defenses off/on comparison.
  Case files   - one card per poison page; opens its target question on the poisoned corpus with
                 defenses off and on and says whether the poison chunk was pushed out of the top k.
  Lab results  - the report figures from eval/figures/ with their captions.
"""
import bisect
import html
import json
import math
import os
import re

import altair as alt
import pandas as pd
import streamlit as st

from index.build_index import index_path
from index.text import STOP_WORDS, stem_word
from rag import answer as rag
from ranking import defenses as content
from ranking import quality
from ranking.score import get_state
from ranking.search import ALPHA, search

st.set_page_config(page_title="TrustRAG", layout="wide")

POISON_PAGES_PATH = os.path.join("attack", "poison_pages.json")
MANIFEST_PATH = os.path.join("data", "poison_manifest.json")
FIGURES_DIR = os.path.join("eval", "figures")

DEFENSE_SETS = {"none": (), "quality": ("quality",), "jaccard": ("jaccard",), "all": ("quality", "jaccard")}
MODELS = {"Sonnet": rag.MODEL, "Haiku": rag.HAIKU}
WHICH_DEFENSE = {"untrusted_host": "quality g(d)", "query_copy": "jaccard", "keyword_stuffing": "jaccard"}
TOKEN_RE = re.compile(r"[a-z0-9]+")  # same tokeniser as index/text.py
WORD_RE = re.compile(r"[A-Za-z0-9]+")  # same, on the original-case text (for highlighting)

EXAMPLES = {  # group -> [(button label, question, corpus to switch to or None)]
    "Normal": [("Type 2 diabetes symptoms", "What are the symptoms of type 2 diabetes?", None),
               ("What causes asthma", "What causes asthma?", None)],
    "Unanswerable": [("MRI scan price in Delhi", "How much does an MRI scan cost in Delhi?", None)],
    "Poison": [("External page: bitter gourd cures diabetes",
                "Can bitter gourd juice cure type 2 diabetes instead of medicine?", "poisoned"),
               ("Insider edit: dengue treatment", "What is the treatment for dengue fever?", "poisoned")],
}

# Colours: only for meaning. Term colours in the score autopsy just tell the terms apart.
C = {"slate": "#1f2a33", "muted": "#5b6670", "teal": "#2f7f7a", "bg": "#f7f6f2", "card": "#fdfcf9",
     "line": "#dcdad3", "green": "#2e7d4f", "red": "#b3261e", "amber": "#c98500", "grey": "#8a949c"}
TERM_COLOURS = ["#2a78d6", "#4a3aa7", "#e87ba4", "#1baf7a", "#6d5a3f", "#5598e7"]
SANS = "IBM Plex Sans"
MONO = "IBM Plex Mono"

# =============================================================================================
# STYLES - all CSS for the app lives in this one block.
# Layout hooks: Streamlit puts a "st-key-<key>" class on keyed containers, so the CSS targets
# st.container(key=...) blocks: "examples" (pill buttons), "evi-<status>-<n>" (evidence cards),
# "casegrid" / "case-<id>" (case file grid and cards).
# =============================================================================================
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&display=swap');

:root {{
  --slate: {C['slate']}; --muted: {C['muted']}; --teal: {C['teal']}; --bg: {C['bg']};
  --card: {C['card']}; --line: {C['line']}; --green: {C['green']}; --red: {C['red']};
  --amber: {C['amber']}; --grey: {C['grey']};
  --sans: '{SANS}', system-ui, sans-serif; --mono: '{MONO}', Consolas, ui-monospace, monospace;
}}

/* --- Typography --------------------------------------------------------------------------- */
html {{ font-size: 17px; }}
.stApp, .stApp div, .stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button,
.stApp td, .stApp th, .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: var(--sans); }}
.stApp .mono, .stApp code, .stApp pre, .stApp .chip, .stApp .badge, .stApp .rank-big, .stApp .formula,
.stApp table.res .num {{ font-family: var(--mono); }}
.mono {{ font-size: 0.82rem; }}
.muted {{ color: var(--muted); }}
.small {{ font-size: 0.8rem; }}

/* --- Page width and vertical rhythm: small gaps inside a section, larger between sections --- */
.stApp [data-testid="stMainBlockContainer"], .stApp .block-container {{
  max-width: 1150px; margin: 0 auto; padding-top: 3.8rem; padding-bottom: 3rem; }}
.stApp [data-testid="stVerticalBlock"] {{ gap: 0.55rem; }}
.stApp [data-testid="stMarkdownContainer"] p {{ margin: 0 0 0.35rem 0; line-height: 1.55; }}
.stApp [data-testid="stMarkdownContainer"] p:last-child {{ margin-bottom: 0; }}
.stApp [data-testid="stMarkdownContainer"] {{ margin-bottom: 0 !important; }}  /* Streamlit sets -1rem */
.section {{ margin-top: 1.3rem; }}              /* gap between sections */
.section.first {{ margin-top: 0.3rem; }}
.eyebrow {{ font-size: 0.68rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase;
           color: var(--teal); margin-bottom: 0.1rem; }}
.sec-title {{ font-size: 1.05rem; font-weight: 600; color: var(--slate); }}
.note {{ font-size: 0.85rem; color: var(--muted); line-height: 1.5; margin-top: 0.15rem; }}

/* --- Header -------------------------------------------------------------------------------- */
.app-title {{ font-size: 1.7rem; font-weight: 700; color: var(--slate); letter-spacing: -0.01em; line-height: 1.2; }}
.app-sub {{ color: var(--muted); font-size: 0.95rem; margin-top: 0.1rem; }}
.rule {{ border: 0; border-top: 1px solid var(--line); margin: 0.6rem 0 0.5rem 0; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 0.35rem; }}
.chip {{ font-size: 0.72rem; padding: 2px 8px; border: 1px solid var(--line); border-radius: 4px;
        background: var(--card); color: var(--slate); white-space: nowrap; }}
.chip .k {{ color: var(--muted); }}

/* --- Tabs ---------------------------------------------------------------------------------- */
.stApp [data-testid="stTab"] p {{ font-size: 0.92rem; font-weight: 500; }}
.stApp [role="tabpanel"] {{ padding-top: 0.6rem; }}

/* --- Badges: one pill style, colour by meaning ---------------------------------------------- */
.badge {{ display: inline-block; font-size: 0.68rem; font-weight: 500; line-height: 1.5; letter-spacing: 0.03em;
         padding: 0 8px; border-radius: 999px; border: 1px solid; margin: 0 4px 2px 0;
         vertical-align: 1px; max-width: 100%; overflow-wrap: anywhere; }}
.b-poison  {{ background: #fbeceb; color: #8c1d18; border-color: #e3a49f; font-weight: 600; }}
.b-flag    {{ background: #fbf3e0; color: #6e4700; border-color: #e2c27a; }}
.b-trusted {{ background: #e7f2eb; color: #1d6438; border-color: #a9d0b8; }}
.b-neutral {{ background: #efeee9; color: #4f575e; border-color: #d2d0ca; }}

/* --- Cards: 1px soft border, 8px radius; thin coloured left edge by status ------------------ */
.card {{ border: 1px solid var(--line); border-radius: 8px; padding: 0.7rem 0.95rem; background: var(--card); }}
.card.tight {{ padding: 0.45rem 0.8rem; margin-bottom: 0.4rem; font-size: 0.92rem; }}
.card.poison, [class*="st-key-evi-poison"] {{ border-left: 3px solid var(--red) !important; }}
.card.flagged, [class*="st-key-evi-flagged"] {{ border-left: 3px solid var(--amber) !important; }}
.card.trusted, [class*="st-key-evi-trusted"] {{ border-left: 3px solid var(--green) !important; }}
.card.neutral, [class*="st-key-evi-neutral"] {{ border-left: 3px solid var(--grey) !important; }}
[class*="st-key-evi-"] {{ border: 1px solid var(--line); border-radius: 8px; background: var(--card);
                         padding: 0.75rem 0.95rem 0.4rem 0.95rem; gap: 0.3rem; }}
.card-head {{ display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; }}
.card-title {{ font-weight: 600; font-size: 1rem; margin-top: 0.25rem; color: var(--slate); }}
.card-meta {{ font-size: 0.74rem; color: var(--muted); margin: 0.1rem 0 0.45rem 0; overflow-wrap: anywhere; }}
.chunk-text {{ max-height: 13rem; overflow-y: auto; line-height: 1.6; font-size: 0.93rem; padding-right: 0.5rem; }}
mark {{ background: #d5ebe8; color: var(--slate); border-bottom: 2px solid var(--teal); padding: 0 1px; }}
.answer {{ font-size: 1rem; line-height: 1.65; background: var(--card); border: 1px solid var(--line);
          border-left: 3px solid var(--teal); border-radius: 8px; padding: 0.8rem 1.05rem; white-space: pre-wrap; }}
.answer ul, .answer ol {{ white-space: normal; margin: 0.2rem 0; padding-left: 1.3rem; }}
.answer li {{ margin: 0.1rem 0; }}
.answer p {{ white-space: normal; }}
.cite {{ display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.15rem 0.5rem; margin: 0.25rem 0;
        font-size: 0.9rem; }}
.cite .url {{ color: var(--muted); font-size: 0.78rem; overflow-wrap: anywhere; }}

/* --- Results table: fixed layout so it always fits the tab width ---------------------------- */
table.res {{ border-collapse: collapse; width: 100%; table-layout: fixed; font-size: 0.9rem; background: var(--card);
            border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
table.res th {{ text-align: left; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
               color: var(--muted); border-bottom: 1px solid var(--line); padding: 7px 10px; }}
table.res td {{ border-bottom: 1px solid #ebe9e3; padding: 7px 10px; vertical-align: top; overflow-wrap: anywhere; }}
table.res tr:last-child td {{ border-bottom: 0; }}
table.res .num {{ text-align: right; font-family: var(--mono); }}
table.res .sub {{ color: var(--muted); font-size: 0.8rem; }}
table.res tr.poison td:first-child  {{ box-shadow: inset 3px 0 0 var(--red); }}
table.res tr.flagged td:first-child {{ box-shadow: inset 3px 0 0 var(--amber); }}
table.res tr.trusted td:first-child {{ box-shadow: inset 3px 0 0 var(--green); }}
table.res tr.neutral td:first-child {{ box-shadow: inset 3px 0 0 var(--grey); }}

/* --- Example questions: wrapping row of pill buttons, grouped with small labels ------------- */
.ex-label {{ font-size: 0.66rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
            color: var(--muted); white-space: nowrap; }}
.st-key-examples button {{ border-radius: 999px; min-height: 0; padding: 0.15rem 0.8rem; }}
.st-key-examples button p {{ font-size: 0.84rem; white-space: nowrap; }}

/* --- Case files: responsive grid (3 columns wide, 2 narrower), equal-height cards ----------- */
.st-key-casegrid {{ display: grid !important; /* at most 3 columns; 2 once a column would be narrower than 280px */
                    grid-template-columns: repeat(auto-fill, minmax(max(280px, calc((100% - 1.6rem) / 3)), 1fr));
                    gap: 0.8rem; align-items: stretch; }}
.st-key-casegrid > div {{ height: 100%; }}
[class*="st-key-case-"] {{ height: 100%; border: 1px solid var(--line); border-left: 3px solid var(--red);
                          border-radius: 8px; background: var(--card); padding: 0.75rem 0.9rem; gap: 0.5rem; }}
[class*="st-key-case-"] > div:last-child {{ margin-top: auto; }}   /* "Open case" sits at the bottom */
[class*="st-key-case-"] button {{ min-height: 0; padding: 0.2rem 0.7rem; }}
.case-claim {{ font-size: 0.92rem; line-height: 1.5; margin-top: 0.35rem; }}
.case-q {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.35rem; }}

/* --- Case verdict: ranks as big mono numbers, verdict as a stamp ---------------------------- */
.verdict-card {{ display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 1rem; }}
.rank-big {{ font-size: 2.1rem; font-weight: 600; color: var(--slate); line-height: 1.15; }}
.rank-big .arrow {{ color: var(--muted); font-weight: 400; padding: 0 0.3rem; }}
.stamp {{ display: inline-block; font-weight: 700; font-size: 1.25rem; letter-spacing: 0.18em; padding: 0.15rem 0.8rem;
         border: 2px solid; border-radius: 4px; transform: rotate(-2deg); margin-top: 0.25rem; }}
.stamp.caught {{ color: var(--green); border-color: var(--green); }}
.stamp.unsolved {{ color: var(--red); border-color: var(--red); }}

/* --- Expanders and dataframes ------------------------------------------------------------- */
.stApp [data-testid="stExpander"] details {{ border: 1px solid var(--line); border-radius: 8px; background: var(--card); }}
.stApp [data-testid="stExpander"] summary p {{ font-size: 0.9rem; font-weight: 500; }}

/* --- Sidebar: grouped controls under small uppercase headings ------------------------------ */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{ gap: 0.5rem; }}
.side-h {{ font-size: 0.66rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: var(--teal);
          border-bottom: 1px solid var(--line); padding-bottom: 0.2rem; margin-top: 0.9rem; }}
.side-h.first {{ margin-top: 0; }}
.formula {{ font-family: var(--mono); font-size: 0.74rem; line-height: 1.6; color: var(--slate); background: var(--card);
           border: 1px solid var(--line); border-radius: 8px; padding: 0.55rem 0.7rem; margin-top: 1.2rem; white-space: pre-wrap; }}
</style>
"""
st.html(CSS)  # style-only st.html takes no space on the page


def section(label: str, title: str | None = None, note: str | None = None, first: bool = False) -> None:
    """Section heading: small uppercase label, optional calm title and muted note."""
    st.markdown(f'<div class="section{" first" if first else ""}"><div class="eyebrow">{label}</div>'
                + (f'<div class="sec-title">{title}</div>' if title else "")
                + (f'<div class="note">{note}</div>' if note else "") + "</div>", unsafe_allow_html=True)


def chips(items: list[tuple[str, str]]) -> str:
    return '<div class="chips">' + "".join(
        f'<span class="chip"><span class="k">{k}</span> {html.escape(str(v))}</span>' for k, v in items) + "</div>"


# ---------------------------------------------------------------------------------------------
# Calls into the existing code (cached so reruns of the page don't recompute)
# ---------------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False, max_entries=500)
def run_search(query: str, k: int, mode: str, stem: bool, defenses: tuple, corpus: str) -> list[dict]:
    return search(query, k=k, mode=mode, defenses=list(defenses), stem=stem, corpus=corpus)


@st.cache_data(show_spinner=False, max_entries=500)
def run_answer(question: str, k: int, mode: str, stem: bool, defenses: tuple, corpus: str, model: str) -> dict:
    # rag.answer uses its own on-disk cache too, so this only costs an API call the first time ever.
    return rag.answer(question, model=model, k=k, mode=mode, defenses=list(defenses), stem=stem, corpus=corpus)


def safe_answer(*args) -> dict | None:
    try:
        return run_answer(*args)
    except (Exception, SystemExit) as e:  # SystemExit: rag.answer exits if .env has no key
        st.error(f"Could not get an answer from the LLM: {e}")
        return None


def chunk_record(cid: str, stem: bool, corpus: str) -> dict:
    return get_state(stem, corpus)["chunks"][cid]


def status_of(r: dict, stem: bool, corpus: str) -> str:
    """poison > flagged by a defense > trusted source > neutral (decides the colour)."""
    c = chunk_record(r["chunk_id"], stem, corpus)
    if c.get("is_poison"):
        return "poison"
    if r["flags"]:
        return "flagged"
    if c["host"] in quality.TRUSTED_HOSTS:
        return "trusted"
    return "neutral"


def badges(r: dict, stem: bool, corpus: str) -> str:
    c = chunk_record(r["chunk_id"], stem, corpus)
    out = []
    if c.get("is_poison"):
        out.append('<span class="badge b-poison">POISON</span>')
    for f in r["flags"]:
        out.append(f'<span class="badge b-flag">{html.escape(f)}</span>')
    if c["host"] in quality.TRUSTED_HOSTS:
        out.append('<span class="badge b-trusted">MedlinePlus</span>')
    else:
        out.append(f'<span class="badge b-neutral">{html.escape(c["host"])}</span>')
    return "".join(out)


def content_stats(r: dict, stem: bool, corpus: str) -> tuple[dict, bool]:
    """Jaccard + repetition for a result. If the jaccard defense was off, compute them with the
    same function for display only (returned applied=False: they did not affect the score)."""
    chk = r["score_breakdown"]["content_check"]
    if chk:
        return chk, True
    terms = set(r["score_breakdown"]["query_terms"])
    return content.content_flags(terms, chunk_record(r["chunk_id"], stem, corpus), stem), False


def highlight(text: str, q_terms: set, stem: bool) -> tuple[str, int, int]:
    """HTML-escaped text with every word whose (stemmed) term is a query term wrapped in <mark>.
    Returns (html, matched words, total words)."""
    out, last, hits, total = [], 0, 0, 0
    for m in WORD_RE.finditer(text):
        out.append(html.escape(text[last:m.start()]))
        word, low = m.group(0), m.group(0).lower()
        total += 1
        term = None if low in STOP_WORDS else (stem_word(low) if stem else low)
        if term in q_terms:
            hits += 1
            out.append(f"<mark>{html.escape(word)}</mark>")
        else:
            out.append(html.escape(word))
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out), hits, total


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def small_card(r: dict, i: int, corpus_: str, extra: str = "") -> str:
    """One-line result card used in the off/on comparisons."""
    return (f'<div class="card tight {status_of(r, stem, corpus_)}"><div class="card-head">'
            f'<span class="mono">#{i} &nbsp;{fmt(r["score"])}</span><span>{badges(r, stem, corpus_)}</span>'
            f'{extra}</div>{html.escape(r["title"])} &gt; {html.escape(r["heading"])}</div>')


# ---------------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------------

ss = st.session_state
ss.setdefault("corpus", "clean")
ss.setdefault("question", "")
ss.setdefault("active_q", "")
ss.setdefault("compare", False)
ss.setdefault("case", None)


def side_heading(label: str, first: bool = False) -> None:
    st.markdown(f'<div class="side-h{" first" if first else ""}">{label}</div>', unsafe_allow_html=True)


with st.sidebar:
    side_heading("Data", first=True)
    corpus = st.radio("Corpus", ["clean", "poisoned"], key="corpus", horizontal=True)
    side_heading("Ranking")
    mode = st.radio("Ranking mode", ["bm25", "tfidf"], horizontal=True)
    stem = st.toggle("Porter stemming", value=True)
    k = st.number_input("Top k", min_value=1, max_value=20, value=5, step=1)
    side_heading("Defenses")
    defense_name = st.selectbox("Defenses", list(DEFENSE_SETS), index=3)
    side_heading("Model")
    model_name = st.radio("Model", list(MODELS), horizontal=True)
    gen_answer = st.toggle("Generate LLM answer", value=True,
                           help="Answers are cached on disk: a question asked before costs nothing.")
    st.markdown(f'<div class="formula">net = relevance_norm\n  + alpha*g(d)   [quality]\n'
                f'  - {content.PENALTY}*#flags   [jaccard]\nalpha = {ALPHA}</div>', unsafe_allow_html=True)

defenses = DEFENSE_SETS[defense_name]
model = MODELS[model_name]

# ---------------------------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------------------------

st.markdown(
    '<div class="app-title">TrustRAG</div>'
    '<div class="app-sub">Health Q&amp;A over MedlinePlus, under a poisoning attack, with IR defenses.</div>'
    '<hr class="rule">'
    + chips([("corpus", corpus), ("mode", mode), ("stemming", "on" if stem else "off"),
             ("defenses", defense_name), ("k", k), ("model", model_name)]),
    unsafe_allow_html=True)

if corpus == "poisoned" and not os.path.exists(index_path(stem, "poisoned")):
    st.error("The poisoned corpus is not built yet. Run:  python -m attack.inject")
    st.stop()

tab_inv, tab_cases, tab_lab = st.tabs(["Investigate", "Case files", "Lab results"])


# ---------------------------------------------------------------------------------------------
# Tab 1: Investigate
# ---------------------------------------------------------------------------------------------

def set_example(q: str, to_corpus: str | None) -> None:
    ss.question, ss.active_q, ss.compare = q, q, False
    if to_corpus:
        ss.corpus = to_corpus


def ask() -> None:
    ss.active_q, ss.compare = ss.question.strip(), False


def show_answer(out: dict) -> None:
    src = "from cache" if out["cached"] else "new API call"
    note = " · abstained" if out["abstained"] else ""
    section("Answer", note=f'<span class="mono">{model_name} · {src}{note}</span>')
    st.markdown(f'<div class="answer">{html.escape(out["answer"])}</div>', unsafe_allow_html=True)
    if out["citations"]:
        num_of = {cid: n for n, cid in out["source_map"].items()}
        by_id = {r["chunk_id"]: r for r in out["retrieved"]}
        lines = []
        for cid in out["citations"]:
            r = by_id[cid]
            lines.append(f'<div class="cite"><span class="mono">[{num_of[cid]}] {cid}</span>'
                         f'<span>{badges(r, stem, corpus)}</span>'
                         f'<span>{html.escape(r["title"])} &gt; {html.escape(r["heading"])}</span>'
                         f'<span class="url">{html.escape(r["url"])}</span></div>')
        st.markdown('<div class="eyebrow" style="margin-top:0.3rem">Citations</div>' + "".join(lines),
                    unsafe_allow_html=True)
    if out["invalid_citations"]:
        st.warning(f"The answer cites sources that do not exist: {out['invalid_citations']}")


def results_table(results: list[dict]) -> None:
    rows, details = [], []
    for i, r in enumerate(results, 1):
        b = r["score_breakdown"]
        chk, applied = content_stats(r, stem, corpus)
        rows.append(
            f'<tr class="{status_of(r, stem, corpus)}">'
            f'<td class="num">{i}</td>'
            f'<td>{html.escape(r["title"])}<div class="sub">{html.escape(r["heading"])}</div></td>'
            f'<td class="num"><b>{fmt(r["score"])}</b></td>'
            f'<td>{badges(r, stem, corpus)}</td></tr>')
        details.append({"#": i, "chunk_id": r["chunk_id"], "net score": r["score"],
                        "relevance": b["relevance_norm"], "alpha*g(d)": b["g_term"], "penalty": b["penalty"],
                        "Jaccard": chk["jaccard"], "repetition": chk["repetition"]})
    st.markdown(
        '<table class="res"><colgroup><col style="width:2.8rem"><col><col style="width:7rem">'
        '<col style="width:30%"></colgroup>'
        '<tr><th class="num">#</th><th>title / heading</th><th class="num">net score</th>'
        '<th>flags / source</th></tr>' + "".join(rows) + "</table>", unsafe_allow_html=True)

    with st.expander("Score details"):
        num = lambda label, f, help_=None: st.column_config.NumberColumn(label, format=f, width=90, help=help_)
        off = not results[0]["score_breakdown"]["content_check"]
        ref = "jaccard defense off: shown for reference, did not change any score" if off else None
        st.dataframe(pd.DataFrame(details), hide_index=True, width="stretch", column_config={
            "#": st.column_config.NumberColumn("#", width=40),
            "chunk_id": st.column_config.TextColumn("chunk_id", width=150),
            "net score": num("net score", "%.4f"),
            "relevance": num("relevance", "%.4f", "relevance_norm: BM25 / tf-idf score / max in the list"),
            "alpha*g(d)": num("alpha*g(d)", "%.4f", "static source quality, weighted by alpha"),
            "penalty": num("penalty", "%.2f", "defense penalty per flag"),
            "Jaccard": num("Jaccard" + (" (ref)" if off else ""), "%.3f", ref or "Jaccard(query terms, chunk terms)"),
            "repetition": num("repetition" + (" (ref)" if off else ""), "%.3f", ref or "share of words that are query terms"),
        })
        if off:
            st.caption("Jaccard and repetition marked (ref): the jaccard defense is off, so they are shown "
                       "for reference and did not change any score.")


def autopsy_chart(r: dict, x_domain: list[float]) -> alt.LayerChart:
    """Stacked bar: each query term's share of the normalised relevance, + alpha*g(d), - penalty."""
    b = r["score_breakdown"]
    # Term contributions are on the raw scale; relevance_norm = raw / max_raw (capped for flagged
    # chunks), so scale every term by the same factor to make them add up to relevance_norm.
    scale = b["relevance_norm"] / b["relevance_raw"] if b["relevance_raw"] else 0.0
    parts, colours = [], []
    terms = list(b["terms"].items())
    for i, (t, tb) in enumerate(terms[:len(TERM_COLOURS)]):
        parts.append({"part": f"term '{t}'", "value": tb["contribution"] * scale})
        colours.append(TERM_COLOURS[i])
    rest = sum(tb["contribution"] for _, tb in terms[len(TERM_COLOURS):]) * scale
    if rest:
        parts.append({"part": "other terms", "value": rest})
        colours.append(C["grey"])
    if b["g_term"]:
        parts.append({"part": "alpha*g(d)", "value": b["g_term"]})
        colours.append(C["green"])
    if b["penalty"]:
        parts.append({"part": "- penalty", "value": -b["penalty"]})
        colours.append(C["amber"])
    for i, p in enumerate(parts):
        p["order"] = i
    df = pd.DataFrame(parts)
    x = alt.X("value:Q", stack="zero", title=None, scale=alt.Scale(domain=x_domain),
              axis=alt.Axis(grid=False, tickCount=6, labelFont=MONO, labelFontSize=11, labelColor=C["muted"]))
    bars = alt.Chart(df).mark_bar(height=20, stroke=C["card"], strokeWidth=2).encode(
        x=x,
        color=alt.Color("part:N", sort=[p["part"] for p in parts],
                        scale=alt.Scale(domain=[p["part"] for p in parts], range=colours),
                        legend=alt.Legend(orient="bottom", title=None, labelFont=SANS, labelFontSize=12,
                                          labelColor=C["slate"], columns=4, symbolSize=80)),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("part:N"), alt.Tooltip("value:Q", format=".4f")])
    zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color=C["muted"], strokeWidth=1).encode(x="v:Q")
    net = alt.Chart(pd.DataFrame({"net": [b["net"]], "label": [f"net {b['net']:.4f}"]})).mark_tick(
        color=C["slate"], thickness=3, size=30).encode(x="net:Q", tooltip=["label:N"])
    return (bars + zero + net).properties(height=56).configure_view(strokeWidth=0).configure(
        background="transparent")


def result_cards(results: list[dict], q_terms: set) -> None:
    lo = min(-r["score_breakdown"]["penalty"] for r in results) - 0.05
    hi = max(r["score_breakdown"]["relevance_norm"] + r["score_breakdown"]["g_term"] for r in results) + 0.05
    for i, r in enumerate(results, 1):
        body_html, hits, total = highlight(r["body"], q_terms, stem)
        title_html, _, _ = highlight(r["title"], q_terms, stem)
        head_html, _, _ = highlight(r["heading"], q_terms, stem)
        share = f" · {hits}/{total} words are query terms ({100 * hits / total:.0f}%)" if total else ""
        with st.container(key=f"evi-{status_of(r, stem, corpus)}-{i}"):
            st.markdown(
                f'<div class="card-head"><span class="mono"><b>#{i}</b> &nbsp;{fmt(r["score"])}</span>'
                f'<span>{badges(r, stem, corpus)}</span></div>'
                f'<div class="card-title">{title_html} &gt; {head_html}</div>'
                f'<div class="card-meta mono">{r["chunk_id"]} · {html.escape(r["url"])}{share}</div>'
                f'<div class="chunk-text">{body_html}</div>'
                f'<div class="eyebrow" style="margin-top:0.6rem">Score autopsy</div>', unsafe_allow_html=True)
            st.altair_chart(autopsy_chart(r, [lo, hi]), width="stretch")


def query_trace(query: str, results: list[dict]) -> None:
    st_ = get_state(stem, corpus)
    idx, N = st_["idx"], st_["idx"]["N"]
    tokens = TOKEN_RE.findall(query.lower())
    stops = [t for t in tokens if t in STOP_WORDS]
    kept = [t for t in tokens if t not in STOP_WORDS]
    terms = [stem_word(t) for t in kept] if stem else kept
    line = lambda label, val: (f'<div style="display:flex; gap:0.8rem; margin:0.15rem 0">'
                               f'<span class="muted" style="flex:0 0 11rem; font-size:0.85rem">{label}</span>'
                               f'<span class="mono" style="overflow-wrap:anywhere">{html.escape(val)}</span></div>')
    st.markdown(line("raw query", repr(query)) + line("tokens (case-folded)", str(tokens))
                + line("stop words removed", str(stops))
                + line("index terms" + (" (stems)" if stem else ""), str(terms))
                + line("N (chunks in corpus)", f"{N:,}"), unsafe_allow_html=True)

    rows, candidates = [], set()
    for t in dict.fromkeys(terms):
        entry = idx["index"].get(t)
        if entry is None:
            rows.append({"term": t, "df": 0, "idf = log10(N/df)": None, "postings length": 0})
            continue
        candidates.update(cid for cid, _ in entry["postings"])
        rows.append({"term": t, "df": entry["df"], "idf = log10(N/df)": round(math.log10(N / entry["df"]), 4),
                     "postings length": len(entry["postings"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "term": st.column_config.TextColumn("term", width=160),
        "df": st.column_config.NumberColumn("df", width=90),
        "idf = log10(N/df)": st.column_config.NumberColumn("idf = log10(N/df)", format="%.4f", width=150),
        "postings length": st.column_config.NumberColumn("postings length", width=130),
    })
    st.markdown(line("candidate chunks", f"{len(candidates):,} chunks contain at least one query term (all scored)")
                + line(f"heap top-{len(results)}", ", ".join(f"({r['score']:.4f}, {r['chunk_id']})" for r in results)),
                unsafe_allow_html=True)

    if results:
        cid = results[0]["chunk_id"]
        st.markdown(f'<div class="eyebrow" style="margin-top:0.5rem">Postings entries of the #1 result</div>'
                    f'<div class="note"><span class="mono">{cid}</span> · positions per zone, counted after '
                    f'stop-word removal</div>', unsafe_allow_html=True)
        prow = []
        for t in dict.fromkeys(terms):
            entry = idx["index"].get(t)
            zones = None
            if entry:  # postings are sorted by chunk_id -> binary search
                plist = entry["postings"]
                j = bisect.bisect_left(plist, cid, key=lambda p: p[0])
                if j < len(plist) and plist[j][0] == cid:
                    zones = plist[j][1]
            prow.append(f"{t:<14} -> " + (json.dumps(zones) if zones else "(not in this chunk)"))
        st.code("\n".join(prow), language=None)


def compare_lists(query: str) -> None:
    off = run_search(query, k, mode, stem, (), corpus)
    on = run_search(query, k, mode, stem, DEFENSE_SETS["all"], corpus)
    ids_off, ids_on = {r["chunk_id"] for r in off}, {r["chunk_id"] for r in on}
    left, right = st.columns(2, gap="medium")
    for col, label, res, other in ((left, "Defenses off", off, ids_on),
                                   (right, "Defenses on · quality + jaccard", on, ids_off)):
        with col:
            moved = lambda r: "" if r["chunk_id"] in other else \
                '<span class="muted small">not in the other list</span>'
            st.markdown(f'<div class="eyebrow">{label}</div>'
                        + "".join(small_card(r, i, corpus, moved(r)) for i, r in enumerate(res, 1)),
                        unsafe_allow_html=True)


with tab_inv:
    section("Question", first=True)
    c_q, c_ask = st.columns([6, 1], vertical_alignment="bottom")
    c_q.text_input("Question", key="question", placeholder="Ask a health question", on_change=ask,
                   label_visibility="collapsed")
    c_ask.button("Ask", type="primary", on_click=ask, width="stretch")
    with st.container(horizontal=True, gap="medium", vertical_alignment="center", key="examples"):
        for group, items in EXAMPLES.items():
            with st.container(horizontal=True, gap="small", vertical_alignment="center", width="content"):
                st.markdown(f'<span class="ex-label">{group}</span>', unsafe_allow_html=True, width="content")
                for label, q, to_corpus in items:
                    st.button(label, key=f"ex_{label}", on_click=set_example, args=(q, to_corpus))

    q = ss.active_q
    if not q:
        st.markdown('<div class="note" style="margin-top:1rem">Type a question or pick an example.</div>',
                    unsafe_allow_html=True)
    else:
        with st.spinner("Searching"):
            results = run_search(q, k, mode, stem, defenses, corpus)

        if gen_answer:
            with st.spinner("Asking the LLM (cached answers are instant)"):
                out = safe_answer(q, k, mode, stem, defenses, corpus, model)
            if out:
                show_answer(out)
        else:
            section("Answer", note="LLM answer turned off in the sidebar.")

        if not results:
            st.info("No chunk contains any of the query terms.")
        else:
            section("Retrieved", f"Top {len(results)} chunks")
            results_table(results)

            section("Query trace", note="tokens → stems → df / idf / postings → heap top-K")
            with st.expander("Show query trace"):
                query_trace(q, results)

            section("Comparison", note="Same question, all defenses off vs all on.")
            if st.button("Compare: defenses off vs on"):
                ss.compare = True
            if ss.compare:
                compare_lists(q)

            section("Evidence", "Chunk text and score autopsy",
                    "Query terms are highlighted (matched on stems), so keyword stuffing is visible. "
                    "The bar shows how the net score is built: each term's share of the relevance, "
                    "plus alpha*g(d), minus any defense penalty; the dark tick is the net score.")
            result_cards(results, set(results[0]["score_breakdown"]["query_terms"]))


# ---------------------------------------------------------------------------------------------
# Tab 2: Case files
# ---------------------------------------------------------------------------------------------

def full_rank(question: str, defs: tuple, poison_id: str) -> tuple[int | None, dict | None, int]:
    """Rank of the poison chunk among ALL matching chunks (k = everything), as in eval/attack_experiment."""
    ranked = run_search(question, 100_000, mode, stem, defs, "poisoned")
    for i, r in enumerate(ranked, 1):
        if r["chunk_id"] == poison_id:
            return i, r, len(ranked)
    return None, None, len(ranked)


with tab_cases:
    if not (os.path.exists(MANIFEST_PATH) and os.path.exists(index_path(stem, "poisoned"))):
        st.error("The poisoned corpus is not built yet. Run:  python -m attack.inject")
    else:
        with open(POISON_PAGES_PATH, encoding="utf-8") as f:
            pages = json.load(f)
        with open(MANIFEST_PATH, encoding="utf-8") as f:
            manifest = {m["id"]: m for m in json.load(f)}
        st.markdown('<div class="note">Synthetic misinformation written for this security experiment, '
                    'modelled on health myths circulating in India. Not health advice.</div>'
                    + chips([("corpus", "poisoned"), ("mode", mode), ("stemming", "on" if stem else "off"),
                             ("k", k), ("defenses", "off vs all")]), unsafe_allow_html=True)

        if ss.case:
            p, m = next(p for p in pages if p["id"] == ss.case), manifest[ss.case]
            st.button("← All cases", on_click=lambda: ss.update(case=None), type="tertiary")
            kind = "external: untrusted host" if p["attack_type"] == "external" else "insider: on a MedlinePlus page"
            section(f"Case file {p['id']}", html.escape(p["target_question"]), first=True)
            st.markdown(f'<span class="badge b-poison">POISON</span><span class="badge b-neutral">{kind}</span>'
                        f'<div style="margin-top:0.3rem"><span class="muted">Myth:</span> '
                        f'{html.escape(p["false_claim"])}</div>'
                        f'<div class="card-meta mono">poison chunk {m["chunk_id"]} · {html.escape(m["url"])}</div>',
                        unsafe_allow_html=True)

            rank_off, _, n_off = full_rank(p["target_question"], (), m["chunk_id"])
            rank_on, r_on, n_on = full_rank(p["target_question"], DEFENSE_SETS["all"], m["chunk_id"])
            caught = not (rank_on and rank_on <= k)
            flagged_by = sorted({WHICH_DEFENSE.get(f, f) for f in (r_on["flags"] if r_on else [])})
            flags_txt = ", ".join(r_on["flags"]) if r_on and r_on["flags"] else "none"
            breakdown = ""
            if r_on:
                b = r_on["score_breakdown"]
                breakdown = (f'<div class="mono muted" style="margin-top:0.3rem">defenses on: net {fmt(b["net"])} '
                             f'= relevance {fmt(b["relevance_norm"])} + alpha*g(d) {fmt(b["g_term"])} - penalty '
                             f'{fmt(b["penalty"], 2)}</div>')

            section("Verdict")
            st.markdown(
                f'<div class="card"><div class="verdict-card">'
                f'<div><div class="note">Rank of the poison chunk · defenses off → on</div>'
                f'<div class="rank-big">{rank_off or "-"}<span class="arrow">→</span>{rank_on or "-"}</div>'
                f'<div class="mono muted">of {n_off:,} → of {n_on:,} matching chunks</div></div>'
                f'<div style="text-align:right"><div class="note">Top {k}, defenses on</div>'
                f'<span class="stamp {"caught" if caught else "unsolved"}">{"CAUGHT" if caught else "UNSOLVED"}'
                f'</span></div></div>'
                f'<div style="margin-top:0.5rem; font-size:0.92rem">Flagged by: <b>{", ".join(flagged_by) or "no defense"}'
                f'</b> <span class="mono muted">(flags: {flags_txt})</span></div>{breakdown}</div>',
                unsafe_allow_html=True)

            section("Answers", f"Model: {model_name}")
            a1, a2 = st.columns(2, gap="medium")
            for col, label, defs in ((a1, "Defenses off", ()), (a2, "Defenses on", DEFENSE_SETS["all"])):
                with col:
                    st.markdown(f'<div class="eyebrow">{label}</div>', unsafe_allow_html=True)
                    if gen_answer:
                        with st.spinner("Asking the LLM (cached answers are instant)"):
                            out = safe_answer(p["target_question"], k, mode, stem, defs, "poisoned", model)
                        if out:
                            cites_poison = m["chunk_id"] in out["citations"]
                            st.markdown(f'<div class="note mono">{"from cache" if out["cached"] else "new API call"}'
                                        f'{" · cites the POISON chunk" if cites_poison else ""}</div>'
                                        f'<div class="answer">{html.escape(out["answer"])}</div>',
                                        unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="note">LLM answer turned off in the sidebar.</div>',
                                    unsafe_allow_html=True)
                    res = run_search(p["target_question"], k, mode, stem, defs, "poisoned")
                    st.markdown(f'<div class="eyebrow" style="margin-top:0.5rem">Top {k} retrieved</div>'
                                + "".join(small_card(r, i, "poisoned") for i, r in enumerate(res, 1)),
                                unsafe_allow_html=True)
        else:
            with st.container(key="casegrid"):
                for p in pages:
                    kind = "external" if p["attack_type"] == "external" else "insider"
                    with st.container(key=f"case-{p['id']}"):
                        st.markdown(f'<div class="card-head"><span class="mono"><b>{p["id"]}</b></span>'
                                    f'<span class="badge b-poison">POISON</span>'
                                    f'<span class="badge b-neutral">{kind}</span></div>'
                                    f'<div class="case-claim">{html.escape(p["false_claim"])}</div>'
                                    f'<div class="case-q">Target question: {html.escape(p["target_question"])}</div>',
                                    unsafe_allow_html=True)
                        st.button(f"Open case {p['id']}", key=f"case_{p['id']}",
                                  on_click=lambda pid=p["id"]: ss.update(case=pid))


# ---------------------------------------------------------------------------------------------
# Tab 3: Lab results
# ---------------------------------------------------------------------------------------------

with tab_lab:
    captions = {}
    readme = os.path.join(FIGURES_DIR, "README.md")
    if os.path.exists(readme):
        with open(readme, encoding="utf-8") as f:
            for line in f:  # table rows: | `figN_name.png` | caption |
                m = re.match(r"\|\s*`([^`]+\.png)`\s*\|\s*(.+?)\s*\|\s*$", line)
                if m:
                    captions[m.group(1)] = m.group(2)
    figs = sorted(f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")) if os.path.isdir(FIGURES_DIR) else []
    if not figs:
        st.info("No figures yet. Run:  python -m eval.make_figures")
    for n, fname in enumerate(figs):
        section(f"Figure {n + 1}", note=f'<span class="mono">{fname}</span>', first=n == 0)
        st.image(os.path.join(FIGURES_DIR, fname), width="stretch")
        st.markdown(f'<div class="note">{captions.get(fname, "")}</div>', unsafe_allow_html=True)
