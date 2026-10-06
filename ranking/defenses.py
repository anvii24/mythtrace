"""Content-based defenses against term-stuffed (poisoned) chunks: query-copy and keyword-stuffing flags.

Source quality g(d) (ranking/quality.py) asks "do we trust the SITE?". It can't catch an insider attack,
where false text is slipped into a trusted MedlinePlus page and inherits its host and PageRank. The checks
here look at the chunk's TEXT instead, compared with the query:

(a) query_copy - Jaccard similarity of term sets
        J(q, d) = |Q ∩ D| / |Q ∪ D|
    Q = distinct query terms, D = distinct terms in the chunk (title + heading + body), both after the same
    text processing as the index (stop words removed, Porter stemmed). J = 1 means the chunk uses exactly
    the query's words and nothing else; a normal explanatory section uses dozens of other words too, so
    its J is small. A page written by copying the question into itself has a high J.

(b) keyword_stuffing - term-repetition ratio
        rep(q, d) = (occurrences of the chunk's TOP_N most frequent terms that are also query terms)
                    / (total terms in the chunk)
    i.e. how much of the chunk is just the query's words said again and again. Stuffing raises tf for
    the query terms, which is exactly what tf-idf and BM25 reward; this ratio measures it directly.
    Only top terms that are query terms count: MedlinePlus sections naturally repeat their own topic words
    ("cancer", "heart"), and that only matters to us when it inflates the score for THIS query.
    (Measured on the poisoned corpus: the query-blind version of this ratio is higher for some real
    sections than for either poison page, so it can't tell them apart.)

Very short chunks (< MIN_TERMS terms) are not flagged: with ~15 terms, one sentence that repeats the topic
gives a high J and a high ratio by chance (e.g. "What are the treatments for infections in pregnancy?").

A flagged chunk is demoted: PENALTY is subtracted from its net score per flag, before top-K is chosen.
Relevance is normalised to [0, 1] and alpha * g(d) is at most 0.3, so a 0.5 penalty is large on that scale.
"""
from collections import Counter

from index.text import preprocess

JACCARD_THRESHOLD = 0.20     # J above this -> query_copy (clean top-5 hits: median 0.06, 99th pct 0.20)
REPETITION_THRESHOLD = 0.45  # rep above this -> keyword_stuffing (clean top-5 hits: median 0.15, 99th pct 0.42)
TOP_N = 5                    # "most frequent few terms" for the repetition ratio
MIN_TERMS = 30               # chunks shorter than this are never flagged
PENALTY = 0.5                # subtracted from the net score per flag

_term_counts: dict = {}  # (stem, chunk_id) -> Counter of the chunk's terms; text never changes, so cache


def chunk_term_counts(chunk: dict, stem: bool) -> Counter:
    key = (stem, chunk["chunk_id"])
    if key not in _term_counts:
        text = " ".join((chunk["title"], chunk["heading"], chunk["body"]))
        _term_counts[key] = Counter(preprocess(text, stem=stem))
    return _term_counts[key]


def jaccard(q_terms: set, d_terms: set) -> float:
    """|Q ∩ D| / |Q ∪ D|: 0 = no shared terms, 1 = identical term sets."""
    union = q_terms | d_terms
    return len(q_terms & d_terms) / len(union) if union else 0.0


def repetition_ratio(q_terms: set, counts: Counter, top_n: int = TOP_N) -> tuple[float, list]:
    """Share of the chunk made of its top_n most frequent terms that are query terms.

    Returns (ratio, [(term, count), ...] for those top terms) so the breakdown can show which words were stuffed.
    """
    total = sum(counts.values())
    top_q = [(t, n) for t, n in counts.most_common(top_n) if t in q_terms]
    return (sum(n for _, n in top_q) / total if total else 0.0), top_q


def content_flags(q_terms: set, chunk: dict, stem: bool,
                  jaccard_threshold: float = JACCARD_THRESHOLD,
                  repetition_threshold: float = REPETITION_THRESHOLD) -> dict:
    """Jaccard + repetition stats for one chunk and the flags they raise."""
    counts = chunk_term_counts(chunk, stem)
    n_terms = sum(counts.values())
    j = jaccard(q_terms, set(counts))
    rep, top_q = repetition_ratio(q_terms, counts)
    flags = []
    if n_terms >= MIN_TERMS:
        if j > jaccard_threshold:
            flags.append("query_copy")
        if rep > repetition_threshold:
            flags.append("keyword_stuffing")
    return {"jaccard": round(j, 4), "repetition": round(rep, 4), "top_query_terms": top_q,
            "n_terms": n_terms, "flags": flags}
