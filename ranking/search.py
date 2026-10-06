"""The shared retriever: relevance (BM25 or lnc.ltc tf-idf) + static quality g(d) = net score.

    python -m ranking.search "symptoms of type 2 diabetes"                  # defenses on, BM25
    python -m ranking.search "symptoms of type 2 diabetes" --no-defenses
    python -m ranking.search "how to treat asthma" --mode tfidf --alpha 0.5
    python -m ranking.search "treatment for dengue fever" --corpus poisoned --no-defenses

corpus="clean" searches the crawl; corpus="poisoned" searches the copy with attack pages injected
(attack/inject.py). Each has its own chunks file, index and link graph.

Net score (IIR ch. 7, "static quality scores"):
    net(q, d) = relevance(q, d) + alpha * g(d)
  relevance(q, d) depends on the query (how well the chunk's words match it); g(d) does not (how much
  we trust the page at all: PageRank + trusted host, see ranking/quality.py).

  BM25 scores have no fixed upper bound (they grow with the number of query terms and their idf),
  while g(d) is in [0, 1]. To add them meaningfully, relevance is first normalised per query by
  dividing by the best raw score among all matching chunks:
        relevance_norm(q, d) = raw(q, d) / max over d' of raw(q, d')     -> top chunk = 1.0
  So alpha has the same meaning for every query and both modes: alpha = 0.3 means a fully trusted,
  well-linked page (g = 1) can overtake a page with up to 30% more relevance but g = 0.

  defenses=False: g(d) is ignored, the ranking is pure relevance (the undefended baseline).
  defenses=True:  g(d) is added for EVERY matching chunk before top-K is chosen, so a low-quality
                  chunk can be pushed out of the top K, not just reordered inside it.
"""
import argparse
import textwrap

from index.text import preprocess
from ranking import quality
from ranking.score import get_state, score_bm25, score_tfidf, top_k

ALPHA = 0.3  # weight of g(d) in the net score


def search(query: str, k: int = 5, mode: str = "bm25", defenses: bool = True,
           alpha: float = ALPHA, stem: bool = True, corpus: str = "clean") -> list[dict]:
    """Top-k chunks for `query` by net score. Result format as in CLAUDE.md."""
    st = get_state(stem, corpus)
    chunks = st["chunks"]  # chunk_id -> chunk record, loaded by get_state
    terms = preprocess(query, stem=stem)
    if mode == "bm25":
        acc = score_bm25(terms, st)
    elif mode == "tfidf":
        acc = score_tfidf(terms, st)
    else:
        raise ValueError(f"unknown mode {mode!r} (use 'bm25' or 'tfidf')")
    if not acc:
        return []

    max_raw = max(a["score"] for a in acc.values())
    net: dict = {}
    for cid, a in acc.items():
        rel = a["score"] / max_raw
        c = chunks[cid]
        q = quality.quality(c["url"], c["host"], corpus)
        g_term = alpha * q["g"] if defenses else 0.0
        net[cid] = {"score": rel + g_term, "relevance_norm": rel, "g_term": g_term, "quality": q}

    results = []
    for score, cid in top_k(net, k):
        c, n = chunks[cid], net[cid]
        flags = []
        if defenses and not n["quality"]["trusted"]:
            flags.append("untrusted_host")
        results.append({
            "chunk_id": cid, "score": round(score, 4),
            "score_breakdown": {
                "mode": mode, "defenses": defenses, "corpus": corpus, "query_terms": terms,
                "relevance_raw": round(acc[cid]["score"], 4), "max_relevance_raw": round(max_raw, 4),
                "relevance_norm": round(n["relevance_norm"], 4),
                "alpha": alpha if defenses else 0.0, "g": n["quality"], "g_term": round(n["g_term"], 4),
                "net": round(score, 4), "terms": acc[cid]["terms"],
            },
            "url": c["url"], "title": c["title"], "heading": c["heading"], "body": c["body"],
            "flags": flags,
        })
    return results


def print_results(query: str, results: list[dict]) -> None:
    if not results:
        print(f'\nquery: "{query}"  -> no matching chunks')
        return
    b0 = results[0]["score_breakdown"]
    print(f'\nquery: "{query}"   corpus={b0["corpus"]}   mode={b0["mode"]}   defenses={"on" if b0["defenses"] else "off"}'
          f'   alpha={b0["alpha"]}   terms={b0["query_terms"]}   max raw={b0["max_relevance_raw"]}')
    print(f"  {'#':>2}  {'net':>6} = {'rel':>6} + {'a*g':>6}   {'raw':>7}  {'PR_n':>5} {'trust':>5} {'g':>5}  chunk")
    for i, r in enumerate(results, 1):
        b = r["score_breakdown"]
        g = b["g"]
        print(f"  {i:>2}  {b['net']:.4f} = {b['relevance_norm']:.4f} + {b['g_term']:.4f}   "
              f"{b['relevance_raw']:>7.4f}  {g['pagerank_norm']:.3f} {g['trusted']:>5.0f} {g['g']:.3f}  "
              f"[{r['chunk_id']}] {r['title']} > {r['heading']}"
              + (f"  FLAGS={r['flags']}" if r["flags"] else ""))
        print("        " + textwrap.shorten(r["body"], 100))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search chunks by net score = relevance + alpha * g(d).")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--no-defenses", action="store_true")
    parser.add_argument("--no-stem", action="store_true")
    parser.add_argument("--corpus", choices=["clean", "poisoned"], default="clean")
    args = parser.parse_args()
    results = search(args.query, k=args.k, mode=args.mode, defenses=not args.no_defenses,
                     alpha=args.alpha, stem=not args.no_stem, corpus=args.corpus)
    print_results(args.query, results)


if __name__ == "__main__":
    main()
