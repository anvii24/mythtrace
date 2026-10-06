"""The shared retriever: relevance (BM25 or lnc.ltc tf-idf) + static quality g(d) = net score.

    python -m ranking.search "symptoms of type 2 diabetes"                  # all defenses on, BM25
    python -m ranking.search "symptoms of type 2 diabetes" --no-defenses
    python -m ranking.search "symptoms of type 2 diabetes" --defenses jaccard   # one defense only
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

  Normalising over UNFLAGGED chunks only: with the "jaccard" defense on, the max is taken over the
  chunks the content check did not flag. Otherwise a term-stuffed poison chunk, which has by far the
  highest raw score (e.g. 28.4 vs 6.6 for the best real chunk), would set the divisor even after
  being flagged, squash every real chunk's relevance to ~0.2, and let g(d) alone decide their order.
  A flagged chunk's relevance_norm is capped at 1.0, i.e. it is treated as at most as relevant as the
  best unflagged chunk; uncapped it would be 4.3 and still win after the penalty. If every matching
  chunk is flagged, the max over all of them is used, as before.

Defenses are named and can be combined (`defenses` = a set/list of names, a comma string, or True/False):
  "quality" - add alpha * g(d) to the net score (above). Flags "untrusted_host" (no extra penalty:
              the low g(d) already is the penalty).
  "jaccard" - content check against the query (ranking/defenses.py): Jaccard similarity of the term
              sets and the query-term repetition ratio. Flags "query_copy" / "keyword_stuffing";
              each flag subtracts `penalty` from the net score.
  defenses=True means all of them; defenses=False (or empty) means none = pure relevance, the
  undefended baseline. Every defense is applied to EVERY matching chunk before top-K is chosen, so a
  demoted chunk can be pushed out of the top K, not just reordered inside it.

    net(q, d) = relevance_norm(q, d)  +  alpha * g(d) [quality]  -  penalty * #flags [jaccard]
"""
import argparse
import textwrap

from index.text import preprocess
from ranking import defenses as content
from ranking import quality
from ranking.score import get_state, score_bm25, score_tfidf, top_k

ALPHA = 0.3  # weight of g(d) in the net score
ALL_DEFENSES = ("quality", "jaccard")


def parse_defenses(defenses) -> frozenset:
    """True -> all, False/None/"" -> none, "quality,jaccard" or ["quality"] -> those names."""
    if defenses is True:
        return frozenset(ALL_DEFENSES)
    if not defenses:
        return frozenset()
    if isinstance(defenses, str):
        defenses = [d.strip() for d in defenses.split(",") if d.strip()]
    names = frozenset(defenses)
    unknown = names - set(ALL_DEFENSES)
    if unknown:
        raise ValueError(f"unknown defenses {sorted(unknown)} (choose from {list(ALL_DEFENSES)})")
    return names


def search(query: str, k: int = 5, mode: str = "bm25", defenses=True,
           alpha: float = ALPHA, stem: bool = True, corpus: str = "clean",
           penalty: float = content.PENALTY,
           jaccard_threshold: float = content.JACCARD_THRESHOLD,
           repetition_threshold: float = content.REPETITION_THRESHOLD) -> list[dict]:
    """Top-k chunks for `query` by net score. Result format as in CLAUDE.md.

    defenses: True (all), False (none), or names from ALL_DEFENSES as a set, list or comma string.
    """
    active = parse_defenses(defenses)
    use_quality, use_jaccard = "quality" in active, "jaccard" in active
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

    q_set = set(terms)
    # Content check first, so the normaliser can skip the chunks it flags (see docstring).
    checks = {cid: content.content_flags(q_set, chunks[cid], stem, jaccard_threshold, repetition_threshold)
              for cid in acc} if use_jaccard else {}
    unflagged = [a["score"] for cid, a in acc.items() if not (checks and checks[cid]["flags"])]
    max_raw = max(unflagged) if unflagged else max(a["score"] for a in acc.values())
    net: dict = {}
    for cid, a in acc.items():
        check = checks.get(cid)
        rel = a["score"] / max_raw
        if check and check["flags"]:
            rel = min(rel, 1.0)  # a flagged chunk can't out-score the best unflagged one on relevance
        c = chunks[cid]
        q = quality.quality(c["url"], c["host"], corpus)
        g_term = alpha * q["g"] if use_quality else 0.0
        flags = ["untrusted_host"] if use_quality and not q["trusted"] else []
        pen = 0.0
        if check:
            flags += check["flags"]
            pen = penalty * len(check["flags"])
        net[cid] = {"score": rel + g_term - pen, "relevance_norm": rel, "g_term": g_term, "quality": q,
                    "content_check": check, "penalty": pen, "flags": flags}

    results = []
    for score, cid in top_k(net, k):
        c, n = chunks[cid], net[cid]
        results.append({
            "chunk_id": cid, "score": round(score, 4),
            "score_breakdown": {
                "mode": mode, "defenses": sorted(active), "corpus": corpus, "query_terms": terms,
                "relevance_raw": round(acc[cid]["score"], 4), "max_relevance_raw": round(max_raw, 4),
                "relevance_norm": round(n["relevance_norm"], 4),
                "alpha": alpha if use_quality else 0.0, "g": n["quality"], "g_term": round(n["g_term"], 4),
                "content_check": n["content_check"], "penalty": round(n["penalty"], 4),
                "net": round(score, 4), "terms": acc[cid]["terms"],
            },
            "url": c["url"], "title": c["title"], "heading": c["heading"], "body": c["body"],
            "flags": n["flags"],
        })
    return results


def print_results(query: str, results: list[dict]) -> None:
    if not results:
        print(f'\nquery: "{query}"  -> no matching chunks')
        return
    b0 = results[0]["score_breakdown"]
    print(f'\nquery: "{query}"   corpus={b0["corpus"]}   mode={b0["mode"]}   '
          f'defenses={",".join(b0["defenses"]) or "none"}'
          f'   alpha={b0["alpha"]}   terms={b0["query_terms"]}   max raw={b0["max_relevance_raw"]}')
    print(f"  {'#':>2}  {'net':>6} = {'rel':>6} + {'a*g':>6} - {'pen':>4}   {'raw':>7}  {'PR_n':>5} {'trust':>5}"
          f" {'g':>5}  {'J':>5} {'rep':>5}  chunk")
    for i, r in enumerate(results, 1):
        b = r["score_breakdown"]
        g, chk = b["g"], b["content_check"]
        jr = f"{chk['jaccard']:.3f} {chk['repetition']:.3f}" if chk else f"{'-':>5} {'-':>5}"
        print(f"  {i:>2}  {b['net']:.4f} = {b['relevance_norm']:.4f} + {b['g_term']:.4f} - {b['penalty']:.2f}   "
              f"{b['relevance_raw']:>7.4f}  {g['pagerank_norm']:.3f} {g['trusted']:>5.0f} {g['g']:.3f}  {jr}  "
              f"[{r['chunk_id']}] {r['title']} > {r['heading']}"
              + (f"  FLAGS={r['flags']}" if r["flags"] else ""))
        print("        " + textwrap.shorten(r["body"], 100))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search chunks by net score = relevance + alpha * g(d).")
    parser.add_argument("query")
    parser.add_argument("--mode", choices=["bm25", "tfidf"], default="bm25")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--defenses", default=",".join(ALL_DEFENSES),
                        help=f"comma-separated subset of {list(ALL_DEFENSES)} (default: all)")
    parser.add_argument("--no-defenses", action="store_true", help="turn every defense off")
    parser.add_argument("--penalty", type=float, default=content.PENALTY, help="net-score penalty per content flag")
    parser.add_argument("--no-stem", action="store_true")
    parser.add_argument("--corpus", choices=["clean", "poisoned"], default="clean")
    args = parser.parse_args()
    results = search(args.query, k=args.k, mode=args.mode,
                     defenses=False if args.no_defenses else args.defenses,
                     alpha=args.alpha, stem=not args.no_stem, corpus=args.corpus, penalty=args.penalty)
    print_results(args.query, results)


if __name__ == "__main__":
    main()
