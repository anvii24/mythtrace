"""Ranked retrieval over the positional index: lnc.ltc tf-idf + cosine, and BM25, with zone weighting.

    python -m ranking.score "symptoms of type 2 diabetes" --mode bm25 --k 5
    python -m ranking.score "how to treat asthma in children" --mode tfidf --k 5 --no-stem

Zone weighting (both modes):
  A term in the title says more about a chunk than the same term buried in the body, so we count
  each occurrence by zone:  tf_w(t, d) = 3 * tf_title + 2 * tf_heading + 1 * tf_body.
  This weighted tf is used everywhere a "tf" appears below.

lnc.ltc (SMART notation, ddd.qqq = document weighting . query weighting):
  document  l = log tf:  1 + log10(tf_w)    n = no idf    c = cosine normalisation (divide by vector length)
  query     l = log tf:  1 + log10(tf_q)    t = idf:  log10(N / df)    c = cosine normalisation
  score(q, d) = sum over shared terms of  w_q(t) * w_d(t)   = cosine of the two unit vectors.
  idf only on the query side: it is applied once, so it isn't counted twice.

BM25 (Okapi, k1 = 1.2, b = 0.75):
  score(q, d) = sum over query terms of
      idf(t) * tf_w * (k1 + 1) / (tf_w + k1 * ((1 - b) + b * L_d / L_avg))
  idf(t) = log10(N / df)  (the IIR form; always >= 0)
  k1 controls tf saturation: the 10th occurrence of a term adds much less than the 1st.
  b controls length normalisation: a long chunk is expected to contain more terms by chance, so its
  tf is discounted (L_d > L_avg) and a short chunk's tf is boosted.
  L_d is the zone-weighted chunk length (3*title + 2*heading + body terms), so lengths are counted in
  the same units as tf_w; L_avg is its average over all chunks.

Top-K: scores live in a dict of accumulators (term-at-a-time over the postings of the query terms);
the K best are picked with a size-K min-heap (heapq) instead of sorting every score.
"""
import argparse
import heapq
import math
import textwrap
from collections import Counter

from index.build_index import ZONES, load_chunks, load_index
from index.text import preprocess

ZONE_WEIGHTS = {"title": 3, "heading": 2, "body": 1}
K1 = 1.2
B = 0.75

# Loaded lazily and cached, one entry per stem setting (the index files are a few MB each).
_cache: dict = {}


def weighted_tf(zones: dict) -> int:
    """Zone-weighted term frequency from one posting: {zone: [positions]} -> 3*title + 2*heading + body."""
    return sum(ZONE_WEIGHTS[z] * len(pos) for z, pos in zones.items())


def weighted_length(lengths: dict) -> float:
    """Zone-weighted chunk length, in the same units as weighted_tf."""
    return sum(ZONE_WEIGHTS[z] * lengths[z] for z in ZONES)


def compute_doc_norms(idx: dict) -> dict:
    """Length of each chunk's lnc vector: sqrt(sum over all its terms of (1 + log10 tf_w)^2).

    Needs every term in the chunk, not just the query terms, so it is computed once per index by
    walking every postings list.
    """
    sq = Counter()
    for entry in idx["index"].values():
        for cid, zones in entry["postings"]:
            sq[cid] += (1 + math.log10(weighted_tf(zones))) ** 2
    return {cid: math.sqrt(s) for cid, s in sq.items()}


def get_state(stem: bool) -> dict:
    """Index + precomputed doc norms + weighted lengths + chunk records, cached per stem setting."""
    if stem not in _cache:
        idx = load_index(stem)
        lengths = {cid: weighted_length(l) for cid, l in idx["chunk_lengths"].items()}
        _cache[stem] = {
            "idx": idx,
            "doc_norms": compute_doc_norms(idx),
            "w_lengths": lengths,
            "avg_w_length": sum(lengths.values()) / idx["N"],
        }
    if "chunks" not in _cache:
        _cache["chunks"] = {c["chunk_id"]: c for c in load_chunks()}
    return _cache[stem]


def idf(N: int, df: int) -> float:
    return math.log10(N / df)


def score_tfidf(query_terms: list[str], st: dict) -> dict:
    """lnc.ltc cosine. Returns {chunk_id: {"score": float, "terms": {term: breakdown}}}."""
    idx, N = st["idx"], st["idx"]["N"]
    q_tf = Counter(t for t in query_terms if t in idx["index"])  # unseen terms have no df -> no weight

    # Query vector (ltc): log tf * idf, then divide by its length.
    w_q = {t: (1 + math.log10(tf)) * idf(N, idx["index"][t]["df"]) for t, tf in q_tf.items()}
    q_norm = math.sqrt(sum(w * w for w in w_q.values())) or 1.0
    w_q = {t: w / q_norm for t, w in w_q.items()}

    # Term-at-a-time: walk each query term's postings, add w_q * w_d into that chunk's accumulator.
    acc: dict = {}
    for t, wq in w_q.items():
        entry = idx["index"][t]
        for cid, zones in entry["postings"]:
            tf_w = weighted_tf(zones)
            w_d = (1 + math.log10(tf_w)) / st["doc_norms"][cid]  # lnc: log tf, no idf, cosine norm
            a = acc.setdefault(cid, {"score": 0.0, "terms": {}})
            a["score"] += wq * w_d
            a["terms"][t] = {
                "tf": {z: len(p) for z, p in zones.items()}, "tf_w": tf_w, "df": entry["df"],
                "w_q": round(wq, 4), "w_d": round(w_d, 4), "contribution": round(wq * w_d, 4),
            }
    return acc


def score_bm25(query_terms: list[str], st: dict) -> dict:
    """Okapi BM25 with zone-weighted tf and length. Same return shape as score_tfidf."""
    idx, N, avg_len = st["idx"], st["idx"]["N"], st["avg_w_length"]
    acc: dict = {}
    for t in dict.fromkeys(query_terms):  # each distinct query term once, in query order
        if t not in idx["index"]:
            continue
        entry = idx["index"][t]
        term_idf = idf(N, entry["df"])
        for cid, zones in entry["postings"]:
            tf_w = weighted_tf(zones)
            len_norm = (1 - B) + B * st["w_lengths"][cid] / avg_len  # >1 for long chunks, <1 for short
            contrib = term_idf * tf_w * (K1 + 1) / (tf_w + K1 * len_norm)
            a = acc.setdefault(cid, {"score": 0.0, "terms": {}})
            a["score"] += contrib
            a["terms"][t] = {
                "tf": {z: len(p) for z, p in zones.items()}, "tf_w": tf_w, "df": entry["df"],
                "idf": round(term_idf, 4), "len_norm": round(len_norm, 4),
                "contribution": round(contrib, 4),
            }
    return acc


def top_k(acc: dict, k: int) -> list[tuple[float, str]]:
    """K highest scores using a min-heap of size K.

    The heap's root is the WORST of the current top K. Each new score only has to beat that root to
    get in (heapreplace pops the root and pushes the newcomer, O(log K)). Total cost O(n log K)
    instead of O(n log n) for sorting all n scored chunks. Ties are broken by chunk_id.
    """
    heap: list[tuple[float, str]] = []
    for cid, a in acc.items():
        item = (a["score"], cid)
        if len(heap) < k:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return sorted(heap, reverse=True)  # only K items left to order, best first


def search(query: str, k: int = 5, mode: str = "bm25", defenses: bool = True,
           stem: bool = True) -> list[dict]:
    """Raw relevance only (no g(d)); `defenses` is ignored. The shared retriever is ranking.search.search."""
    st = get_state(stem)
    terms = preprocess(query, stem=stem)
    if mode == "bm25":
        acc = score_bm25(terms, st)
    elif mode == "tfidf":
        acc = score_tfidf(terms, st)
    else:
        raise ValueError(f"unknown mode {mode!r} (use 'bm25' or 'tfidf')")

    chunks = _cache["chunks"]
    results = []
    for score, cid in top_k(acc, k):
        c = chunks[cid]
        results.append({
            "chunk_id": cid, "score": round(score, 4),
            "score_breakdown": {"mode": mode, "query_terms": terms, "terms": acc[cid]["terms"]},
            "url": c["url"], "title": c["title"], "heading": c["heading"], "body": c["body"],
            "flags": [],
        })
    return results


def print_results(query: str, results: list[dict], mode: str, stem: bool) -> None:
    print(f'\nquery: "{query}"   mode={mode}   stem={"on" if stem else "off"}')
    if results:
        print(f"terms: {results[0]['score_breakdown']['query_terms']}")
    for rank, r in enumerate(results, 1):
        print(f"\n{rank}. {r['score']:.4f}  [{r['chunk_id']}]  {r['title']}  >  {r['heading']}")
        for t, b in r["score_breakdown"]["terms"].items():
            if mode == "tfidf":
                extra = f"w_q={b['w_q']:.4f}  w_d={b['w_d']:.4f}"
            else:
                extra = f"idf={b['idf']:.4f}  len_norm={b['len_norm']:.4f}"
            print(f"     {t:<12} tf={b['tf']} tf_w={b['tf_w']:<3} df={b['df']:<5} {extra}"
                  f"  -> +{b['contribution']:.4f}")
        print("     " + textwrap.shorten(r["body"], 110))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank chunks for a query with BM25 or lnc.ltc tf-idf.")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--no-stem", action="store_true", help="use the unstemmed index")
    args = parser.parse_args()
    stem = not args.no_stem
    print_results(args.query, search(args.query, k=args.k, mode=args.mode, stem=stem), args.mode, stem)


if __name__ == "__main__":
    main()
