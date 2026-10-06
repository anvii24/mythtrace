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

EXAMPLES = [  # (button label, question, corpus to switch to or None)
    ("Type 2 diabetes symptoms", "What are the symptoms of type 2 diabetes?", None),
    ("What causes asthma", "What causes asthma?", None),
    ("Unanswerable: MRI price", "How much does an MRI scan cost in Delhi?", None),
    ("Poison, external: bitter gourd", "Can bitter gourd juice cure type 2 diabetes instead of medicine?", "poisoned"),
    ("Poison, insider: dengue", "What is the treatment for dengue fever?", "poisoned"),
]

# Colours: only for meaning. Term colours in the score autopsy just tell the terms apart.
C = {"slate": "#1f2a33", "muted": "#5b6670", "teal": "#2f7f7a", "bg": "#f7f6f2",
     "green": "#2e7d4f", "red": "#b3261e", "amber": "#c98500", "grey": "#8a949c"}
TERM_COLOURS = ["#2a78d6", "#4a3aa7", "#e87ba4", "#1baf7a", "#6d5a3f", "#5598e7"]

CSS = f"""
<style>
html {{ font-size: 18px; }}
.block-container {{ padding-top: 2rem; max-width: 1400px; }}
h1, h2, h3 {{ color: {C['slate']}; letter-spacing: -0.01em; }}
.mono, code {{ font-family: "JetBrains Mono", "Cascadia Mono", Consolas, ui-monospace, monospace; }}
.mono {{ font-size: 0.86rem; }}
.muted {{ color: {C['muted']}; }}
.badge {{ display: inline-block; padding: 1px 9px; border-radius: 4px; font-size: 0.74rem; font-weight: 600;
          letter-spacing: 0.05em; margin: 0 6px 2px 0; white-space: nowrap; }}
.b-poison  {{ background: {C['red']}; color: #fff; }}
.b-flag    {{ background: #fbf0d6; color: #6e4700; border: 1px solid #e2b85a; }}
.b-trusted {{ background: #e2f0e7; color: #1d6438; border: 1px solid #9ccaae; }}
.b-neutral {{ background: #ebeae6; color: #50585f; border: 1px solid #d2d0ca; }}
.card {{ border: 1px solid #dcdad3; border-left: 5px solid {C['grey']}; border-radius: 6px;
         padding: 0.9rem 1.1rem; margin: 0.4rem 0 0.8rem 0; background: #fcfbf8; }}
.card.poison  {{ border-left-color: {C['red']}; }}
.card.flagged {{ border-left-color: {C['amber']}; }}
.card.trusted {{ border-left-color: {C['green']}; }}
.chunk-text {{ max-height: 15rem; overflow-y: auto; line-height: 1.6; padding-right: 0.5rem; }}
mark {{ background: #cfe7e4; color: {C['slate']}; border-bottom: 2px solid {C['teal']}; padding: 0 1px; }}
table.res {{ border-collapse: collapse; width: 100%; font-size: 0.9rem; }}
table.res th {{ text-align: left; font-weight: 600; color: {C['muted']}; border-bottom: 2px solid #d2d0ca;
                padding: 6px 8px; white-space: nowrap; }}
table.res td {{ border-bottom: 1px solid #e3e1db; padding: 7px 8px; vertical-align: top; }}
table.res td.num {{ text-align: right; white-space: nowrap; }}
table.res tr.poison td:first-child  {{ border-left: 4px solid {C['red']}; }}
table.res tr.flagged td:first-child {{ border-left: 4px solid {C['amber']}; }}
table.res tr.trusted td:first-child {{ border-left: 4px solid {C['green']}; }}
table.res tr.neutral td:first-child {{ border-left: 4px solid {C['grey']}; }}
.answer {{ font-size: 1.05rem; line-height: 1.65; background: #fcfbf8; border: 1px solid #dcdad3;
           border-radius: 6px; padding: 1rem 1.2rem; white-space: pre-wrap; }}
.verdict {{ font-size: 1.4rem; font-weight: 700; letter-spacing: 0.06em; }}
.verdict.caught {{ color: {C['green']}; }}
.verdict.unsolved {{ color: {C['red']}; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


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


# ---------------------------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------------------------

ss = st.session_state
ss.setdefault("corpus", "clean")
ss.setdefault("question", "")
ss.setdefault("active_q", "")
ss.setdefault("compare", False)
ss.setdefault("case", None)

with st.sidebar:
    st.markdown("### Settings")
    corpus = st.radio("Corpus", ["clean", "poisoned"], key="corpus", horizontal=True)
    mode = st.radio("Ranking mode", ["bm25", "tfidf"], horizontal=True)
    stem = st.toggle("Porter stemming", value=True)
    defense_name = st.selectbox("Defenses", list(DEFENSE_SETS), index=3)
    model_name = st.radio("Model", list(MODELS), horizontal=True)
    k = st.number_input("Top k", min_value=1, max_value=20, value=5, step=1)
    gen_answer = st.toggle("Generate LLM answer", value=True,
                           help="Answers are cached on disk: a question asked before costs nothing.")
    st.markdown(
        f'<div class="muted mono" style="margin-top:1.5rem">net = relevance_norm<br>'
        f'&nbsp;&nbsp;+ alpha*g(d)&nbsp;&nbsp;[quality, alpha={ALPHA}]<br>'
        f'&nbsp;&nbsp;- {content.PENALTY}*#flags&nbsp;&nbsp;[jaccard]</div>', unsafe_allow_html=True)

defenses = DEFENSE_SETS[defense_name]
model = MODELS[model_name]

if corpus == "poisoned" and not os.path.exists(index_path(stem, "poisoned")):
    st.error("The poisoned corpus is not built yet. Run:  python -m attack.inject")
    st.stop()

st.markdown("## TrustRAG")
st.markdown('<div class="muted">Health Q&amp;A over MedlinePlus, under a poisoning attack, '
            'with IR defenses.</div>', unsafe_allow_html=True)

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
    st.markdown(f'<div class="muted mono">{model_name} · {src}{note}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="answer">{html.escape(out["answer"])}</div>', unsafe_allow_html=True)
    if out["citations"]:
        num_of = {cid: n for n, cid in out["source_map"].items()}
        by_id = {r["chunk_id"]: r for r in out["retrieved"]}
        lines = []
        for cid in out["citations"]:
            r = by_id[cid]
            lines.append(f'<div style="margin:0.35rem 0"><span class="mono">[{num_of[cid]}] {cid}</span> '
                         f'{badges(r, stem, corpus)}{html.escape(r["title"])} &gt; {html.escape(r["heading"])} '
                         f'<span class="muted">{html.escape(r["url"])}</span></div>')
        st.markdown("**Citations**" + "".join(lines), unsafe_allow_html=True)
    if out["invalid_citations"]:
        st.warning(f"The answer cites sources that do not exist: {out['invalid_citations']}")


def results_table(results: list[dict]) -> None:
    rows = []
    for i, r in enumerate(results, 1):
        b = r["score_breakdown"]
        chk, applied = content_stats(r, stem, corpus)
        jr_style = "" if applied else ' style="color:#9aa3aa" title="jaccard defense off: not applied"'
        rows.append(
            f'<tr class="{status_of(r, stem, corpus)}">'
            f'<td class="num mono">{i}</td>'
            f'<td>{html.escape(r["title"])}<div class="muted" style="font-size:0.82rem">'
            f'{html.escape(r["heading"])}</div></td>'
            f'<td class="num mono"><b>{fmt(r["score"])}</b></td>'
            f'<td class="num mono">{fmt(b["relevance_norm"])}</td>'
            f'<td class="num mono">{fmt(b["g_term"])}</td>'
            f'<td class="num mono">{fmt(b["penalty"], 2)}</td>'
            f'<td class="num mono"{jr_style}>{fmt(chk["jaccard"], 3)}</td>'
            f'<td class="num mono"{jr_style}>{fmt(chk["repetition"], 3)}</td>'
            f'<td>{badges(r, stem, corpus)}</td></tr>')
    st.markdown(
        '<table class="res"><tr><th>#</th><th>title / heading</th><th>net score</th><th>relevance</th>'
        '<th>alpha*g(d)</th><th>penalty</th><th>Jaccard</th><th>repetition</th><th>flags / source</th></tr>'
        + "".join(rows) + "</table>", unsafe_allow_html=True)
    if not results[0]["score_breakdown"]["content_check"]:
        st.caption("Jaccard and repetition in grey: the jaccard defense is off, so they are shown for "
                   "reference and did not change any score.")


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
              axis=alt.Axis(grid=False, tickCount=6, labelFont="monospace", labelFontSize=12))
    bars = alt.Chart(df).mark_bar(height=24, stroke=C["bg"], strokeWidth=2).encode(
        x=x,
        color=alt.Color("part:N", sort=[p["part"] for p in parts],
                        scale=alt.Scale(domain=[p["part"] for p in parts], range=colours),
                        legend=alt.Legend(orient="bottom", title=None, labelFontSize=13, columns=4)),
        order=alt.Order("order:Q"),
        tooltip=[alt.Tooltip("part:N"), alt.Tooltip("value:Q", format=".4f")])
    zero = alt.Chart(pd.DataFrame({"v": [0]})).mark_rule(color=C["muted"], strokeWidth=1).encode(x="v:Q")
    net = alt.Chart(pd.DataFrame({"net": [b["net"]], "label": [f"net {b['net']:.4f}"]})).mark_tick(
        color=C["slate"], thickness=3, size=36).encode(x="net:Q", tooltip=["label:N"])
    return (bars + zero + net).properties(height=70).configure_view(strokeWidth=0).configure(
        background="transparent")


def result_cards(results: list[dict], q_terms: set) -> None:
    lo = min(-r["score_breakdown"]["penalty"] for r in results) - 0.05
    hi = max(r["score_breakdown"]["relevance_norm"] + r["score_breakdown"]["g_term"] for r in results) + 0.05
    for i, r in enumerate(results, 1):
        body_html, hits, total = highlight(r["body"], q_terms, stem)
        title_html, _, _ = highlight(r["title"], q_terms, stem)
        head_html, _, _ = highlight(r["heading"], q_terms, stem)
        share = f"{hits}/{total} words are query terms ({100 * hits / total:.0f}%)" if total else ""
        st.markdown(
            f'<div class="card {status_of(r, stem, corpus)}">'
            f'<div><span class="mono">#{i} &nbsp; {fmt(r["score"])}</span> &nbsp; {badges(r, stem, corpus)}</div>'
            f'<div style="font-size:1.08rem; font-weight:600; margin-top:0.3rem">{title_html} &gt; {head_html}</div>'
            f'<div class="muted mono" style="margin-bottom:0.5rem">{r["chunk_id"]} · '
            f'{html.escape(r["url"])} · {share}</div>'
            f'<div class="chunk-text">{body_html}</div></div>', unsafe_allow_html=True)
        st.markdown('<div class="muted" style="font-size:0.85rem">Score autopsy</div>', unsafe_allow_html=True)
        st.altair_chart(autopsy_chart(r, [lo, hi]), width="stretch")


def query_trace(query: str, results: list[dict]) -> None:
    st_ = get_state(stem, corpus)
    idx, N = st_["idx"], st_["idx"]["N"]
    tokens = TOKEN_RE.findall(query.lower())
    stops = [t for t in tokens if t in STOP_WORDS]
    kept = [t for t in tokens if t not in STOP_WORDS]
    terms = [stem_word(t) for t in kept] if stem else kept
    line = lambda label, val: st.markdown(
        f'<div style="margin:0.25rem 0"><span class="muted" style="display:inline-block;width:11rem">{label}</span>'
        f'<span class="mono">{html.escape(val)}</span></div>', unsafe_allow_html=True)
    line("raw query", repr(query))
    line("tokens (case-folded)", str(tokens))
    line("stop words removed", str(stops))
    line("index terms" + (" (stems)" if stem else ""), str(terms))
    line("N (chunks in corpus)", f"{N:,}")

    rows, candidates = [], set()
    for t in dict.fromkeys(terms):
        entry = idx["index"].get(t)
        if entry is None:
            rows.append({"term": t, "df": 0, "idf = log10(N/df)": None, "postings length": 0})
            continue
        candidates.update(cid for cid, _ in entry["postings"])
        rows.append({"term": t, "df": entry["df"], "idf = log10(N/df)": round(math.log10(N / entry["df"]), 4),
                     "postings length": len(entry["postings"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    line("candidate chunks", f"{len(candidates):,} chunks contain at least one query term (all scored)")
    line(f"heap top-{len(results)}", ", ".join(f"({r['score']:.4f}, {r['chunk_id']})" for r in results))

    if results:
        cid = results[0]["chunk_id"]
        st.markdown(f"**Postings entries of the #1 result** <span class='mono'>{cid}</span> "
                    f"(positions per zone, counted after stop-word removal)", unsafe_allow_html=True)
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
    left, right = st.columns(2)
    for col, label, res, other in ((left, "Defenses off", off, ids_on),
                                   (right, "Defenses on (quality + jaccard)", on, ids_off)):
        with col:
            st.markdown(f"**{label}**")
            for i, r in enumerate(res, 1):
                moved = "" if r["chunk_id"] in other else \
                    ' <span class="muted" style="font-size:0.8rem">(not in the other list)</span>'
                st.markdown(
                    f'<div class="card {status_of(r, stem, corpus)}" style="padding:0.5rem 0.8rem">'
                    f'<span class="mono">#{i} {fmt(r["score"])}</span> {badges(r, stem, corpus)}{moved}<br>'
                    f'{html.escape(r["title"])} &gt; {html.escape(r["heading"])}</div>', unsafe_allow_html=True)


with tab_inv:
    st.text_input("Question", key="question", placeholder="Ask a health question", on_change=ask)
    cols = st.columns([1] + [1.3] * len(EXAMPLES))
    cols[0].button("Ask", type="primary", on_click=ask, width="stretch")
    for col, (label, q, to_corpus) in zip(cols[1:], EXAMPLES):
        col.button(label, on_click=set_example, args=(q, to_corpus), width="stretch")

    q = ss.active_q
    if not q:
        st.markdown('<div class="muted" style="margin-top:2rem">Type a question or pick an example.</div>',
                    unsafe_allow_html=True)
    else:
        with st.spinner("Searching"):
            results = run_search(q, k, mode, stem, defenses, corpus)
        settings = (f"corpus={corpus} · mode={mode} · stemming={'on' if stem else 'off'} · "
                    f"defenses={defense_name} · k={k}")
        st.markdown(f'<div class="muted mono">{settings}</div>', unsafe_allow_html=True)

        st.markdown("### Answer")
        if gen_answer:
            with st.spinner("Asking the LLM (cached answers are instant)"):
                out = safe_answer(q, k, mode, stem, defenses, corpus, model)
            if out:
                show_answer(out)
        else:
            st.markdown('<div class="muted">LLM answer turned off in the sidebar.</div>', unsafe_allow_html=True)

        if not results:
            st.info("No chunk contains any of the query terms.")
        else:
            st.markdown("### Retrieved chunks")
            results_table(results)

            with st.expander("Query trace"):
                query_trace(q, results)

            if st.button("Compare: defenses off vs on"):
                ss.compare = True
            if ss.compare:
                compare_lists(q)

            st.markdown("### Evidence")
            st.caption("Query terms are highlighted (matched on stems), so keyword stuffing is visible. "
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
        st.markdown('<div class="muted">Synthetic misinformation written for this security experiment, '
                    'modelled on health myths circulating in India. Not health advice.</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="muted mono">poisoned corpus · mode={mode} · '
                    f'stemming={"on" if stem else "off"} · k={k} · defenses off vs all</div>',
                    unsafe_allow_html=True)

        if ss.case:
            p, m = next(p for p in pages if p["id"] == ss.case), manifest[ss.case]
            st.button("Back to all cases", on_click=lambda: ss.update(case=None))
            kind = "external: untrusted host" if p["attack_type"] == "external" else "insider: on a MedlinePlus page"
            st.markdown(f"### {p['id']} · {html.escape(p['target_question'])}", unsafe_allow_html=True)
            st.markdown(f'<span class="badge b-poison">POISON</span><span class="badge b-neutral">{kind}</span>'
                        f'<div style="margin-top:0.5rem"><span class="muted">Myth:</span> '
                        f'{html.escape(p["false_claim"])}</div>'
                        f'<div class="muted mono">poison chunk {m["chunk_id"]} · {html.escape(m["url"])}</div>',
                        unsafe_allow_html=True)

            rank_off, _, n_off = full_rank(p["target_question"], (), m["chunk_id"])
            rank_on, r_on, n_on = full_rank(p["target_question"], DEFENSE_SETS["all"], m["chunk_id"])
            caught = not (rank_on and rank_on <= k)
            flagged_by = sorted({WHICH_DEFENSE.get(f, f) for f in (r_on["flags"] if r_on else [])})
            flags_txt = ", ".join(r_on["flags"]) if r_on and r_on["flags"] else "none"

            c1, c2, c3 = st.columns(3)
            c1.markdown(f'<div class="muted">Rank, defenses off</div><div class="mono" style="font-size:1.6rem">'
                        f'{rank_off or "-"} <span class="muted" style="font-size:0.9rem">of {n_off}</span></div>',
                        unsafe_allow_html=True)
            c2.markdown(f'<div class="muted">Rank, defenses on</div><div class="mono" style="font-size:1.6rem">'
                        f'{rank_on or "-"} <span class="muted" style="font-size:0.9rem">of {n_on}</span></div>',
                        unsafe_allow_html=True)
            c3.markdown(f'<div class="muted">Verdict (top {k}, defenses on)</div>'
                        f'<div class="verdict {"caught" if caught else "unsolved"}">'
                        f'{"CAUGHT" if caught else "UNSOLVED"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="margin:0.6rem 0">Flagged by: <b>{", ".join(flagged_by) or "no defense"}</b> '
                        f'<span class="mono muted">(flags: {flags_txt})</span></div>', unsafe_allow_html=True)
            if r_on:
                b = r_on["score_breakdown"]
                st.markdown(f'<div class="mono muted">defenses on: net {fmt(b["net"])} = relevance '
                            f'{fmt(b["relevance_norm"])} + alpha*g(d) {fmt(b["g_term"])} - penalty '
                            f'{fmt(b["penalty"], 2)}</div>', unsafe_allow_html=True)

            a1, a2 = st.columns(2)
            for col, label, defs in ((a1, "Defenses off", ()), (a2, "Defenses on", DEFENSE_SETS["all"])):
                with col:
                    st.markdown(f"**{label}: answer ({model_name})**")
                    if gen_answer:
                        with st.spinner("Asking the LLM (cached answers are instant)"):
                            out = safe_answer(p["target_question"], k, mode, stem, defs, "poisoned", model)
                        if out:
                            cites_poison = m["chunk_id"] in out["citations"]
                            st.markdown(f'<div class="muted mono">{"from cache" if out["cached"] else "new API call"}'
                                        f'{" · cites the POISON chunk" if cites_poison else ""}</div>'
                                        f'<div class="answer">{html.escape(out["answer"])}</div>',
                                        unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="muted">LLM answer turned off in the sidebar.</div>',
                                    unsafe_allow_html=True)
                    res = run_search(p["target_question"], k, mode, stem, defs, "poisoned")
                    for i, r in enumerate(res, 1):
                        st.markdown(f'<div class="card {status_of(r, stem, "poisoned")}" '
                                    f'style="padding:0.45rem 0.8rem"><span class="mono">#{i} {fmt(r["score"])}</span> '
                                    f'{badges(r, stem, "poisoned")}<br>{html.escape(r["title"])} &gt; '
                                    f'{html.escape(r["heading"])}</div>', unsafe_allow_html=True)
        else:
            cols = st.columns(2)
            for i, p in enumerate(pages):
                with cols[i % 2]:
                    kind = "external" if p["attack_type"] == "external" else "insider"
                    st.markdown(f'<div class="card poison"><span class="mono" style="font-weight:600">{p["id"]}'
                                f'</span> &nbsp; <span class="badge b-neutral">{kind}</span>'
                                f'<div style="margin-top:0.4rem">{html.escape(p["false_claim"])}</div>'
                                f'<div class="muted" style="font-size:0.85rem; margin-top:0.3rem">Target question: '
                                f'{html.escape(p["target_question"])}</div></div>', unsafe_allow_html=True)
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
    for fname in figs:
        st.markdown(f'<div class="mono muted" style="margin-top:1.5rem">{fname}</div>', unsafe_allow_html=True)
        st.image(os.path.join(FIGURES_DIR, fname), width="stretch")
        st.markdown(captions.get(fname, ""))
