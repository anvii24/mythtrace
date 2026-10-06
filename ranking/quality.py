"""Static (query-independent) quality score g(d) for every page, from PageRank + a trusted-host bonus.

    python -m ranking.quality            # print the top PageRank pages and g(d) stats

g(d) is "static" because it depends only on the page, never on the query: it is computed once, offline,
and then added to every query's relevance score (net score, see ranking/search.py).

(a) PageRank over the crawl's link graph (data/links.json: page -> in-scope outlinks)
    Imagine a random surfer. On each step, with probability d = 0.85 they click a random outlink of the
    current page; with probability 1 - d = 0.15 they get bored and "teleport" to a random page.
    PageRank PR(p) = the long-run fraction of time the surfer spends on page p:
        PR(p) = (1 - d) / N  +  d * sum over pages q linking to p of  PR(q) / outdeg(q)
    A page is important if important pages link to it. Teleporting guarantees every page has some
    rank (>= 0.15/N) and the iteration converges. Dead ends (pages with no outlinks) would leak rank,
    so their rank is spread evenly over all pages, as if the surfer teleports from them.
    Solved by power iteration: start uniform, apply the formula until the vector stops changing.

    Graph nodes = every crawled page, including hub pages (A-Z lists, category pages) that were too
    short to be saved as chunks: they still pass rank on to the topics they link to.
    Links to pages outside the crawl are dropped (we can't follow the surfer there).

    Normalisation to 0-1: PageRank is very skewed (a handful of hubs hold most of it), so plain
    min-max would squash almost every page to ~0. We min-max the LOG of PageRank instead:
        pr_norm(p) = (log PR(p) - log PR_min) / (log PR_max - log PR_min)
    min and max are taken over CONTENT pages only (pages that have chunks). The site-wide
    healthtopics.html hub is linked from every page's navigation bar and has ~30x the PageRank of any
    topic page; if it set the max, the best real topic would only reach 0.5. Hubs are clipped to 1.
    A page that is not in the graph at all gets 0.

    Poisoned corpus (attack/inject.py): data/links_poisoned.json = the clean graph plus each external
    poisoned page as a node with NO inlinks (nobody real links to a spam site). It only collects
    teleport rank. Insider poison sections live on an existing MedlinePlus URL, so they inherit that
    page's PageRank. min/max for the normalisation come from the clean content pages only, so the
    0-1 scale means the same thing in both corpora.

(b) Trusted-host bonus: 1 if the page's host is in TRUSTED_HOSTS (medlineplus.gov), else 0.
    Any other host, including future poisoned pages, gets nothing. Source reputation is the main
    defense: an attacker can stuff a page with query terms but cannot put it on medlineplus.gov.

g(d) = W_PAGERANK * pr_norm + W_TRUST * trusted,  with W_PAGERANK + W_TRUST = 1, so 0 <= g(d) <= 1.
"""
import json
import math
import os
from urllib.parse import urlsplit

from index.build_index import chunks_path, load_chunks

LINKS_PATH = os.path.join("data", "links.json")
LINKS_PATHS = {"clean": LINKS_PATH, "poisoned": os.path.join("data", "links_poisoned.json")}

DAMPING = 0.85          # probability of following a link (1 - DAMPING = teleport probability)
TOL = 1e-10             # stop when the total change in PageRank (L1) is below this
MAX_ITER = 100

TRUSTED_HOSTS = {"medlineplus.gov"}
W_PAGERANK = 0.5        # g(d) weights; must sum to 1 so g stays in [0, 1]
W_TRUST = 0.5

_cache: dict = {}  # corpus -> state


def load_graph(path: str = LINKS_PATH) -> dict[str, list[str]]:
    """Link graph restricted to crawled pages: {url: [outlinks that are also crawled pages]}."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    nodes = set(raw)
    # dict.fromkeys drops repeated links: two links from q to p count once
    return {u: [v for v in dict.fromkeys(outs) if v in nodes and v != u] for u, outs in raw.items()}


def pagerank(graph: dict[str, list[str]], d: float = DAMPING) -> tuple[dict[str, float], int]:
    """Power iteration. Returns ({url: PageRank}, iterations used). Ranks sum to 1."""
    nodes = list(graph)
    N = len(nodes)
    pr = {u: 1.0 / N for u in nodes}  # start: surfer equally likely to be anywhere
    for it in range(1, MAX_ITER + 1):
        # Rank sitting on dead ends is spread over all pages (the surfer teleports from there).
        dangling = sum(pr[u] for u in nodes if not graph[u])
        new = {u: (1 - d) / N + d * dangling / N for u in nodes}
        for q in nodes:
            outs = graph[q]
            if outs:
                share = d * pr[q] / len(outs)  # q splits its rank evenly over its outlinks
                for p in outs:
                    new[p] += share
        delta = sum(abs(new[u] - pr[u]) for u in nodes)
        pr = new
        if delta < TOL:
            return pr, it
    return pr, MAX_ITER


def normalise_log(pr: dict[str, float], ref: set[str]) -> dict[str, float]:
    """Log-scale min-max to [0, 1], with min/max taken over the pages in `ref`; others are clipped."""
    logs = {u: math.log(v) for u, v in pr.items()}
    ref_logs = [logs[u] for u in ref if u in logs]
    lo, hi = min(ref_logs), max(ref_logs)
    span = (hi - lo) or 1.0
    return {u: min(1.0, max(0.0, (l - lo) / span)) for u, l in logs.items()}


def quality(url: str, host: str | None = None, corpus: str = "clean") -> dict:
    """g(d) and its parts for one page. Works for pages outside the graph (pr_norm = 0)."""
    st = get_state(corpus)
    host = host or urlsplit(url).hostname or ""
    pr_norm = st["pr_norm"].get(url, 0.0)
    trusted = 1.0 if host in TRUSTED_HOSTS else 0.0
    return {
        "pagerank": st["pr"].get(url, 0.0),
        "pagerank_norm": round(pr_norm, 4),
        "trusted": trusted,
        "g": round(W_PAGERANK * pr_norm + W_TRUST * trusted, 4),
    }


def get_state(corpus: str = "clean") -> dict:
    """PageRank computed once per process and corpus, and cached (about 1,000 nodes: well under a second)."""
    if corpus not in _cache:
        graph = load_graph(LINKS_PATHS[corpus])
        pr, iters = pagerank(graph)
        chunks = load_chunks(chunks_path(corpus))
        content_urls = {c["url"] for c in chunks}
        clean_urls = {c["url"] for c in chunks if not c.get("is_poison")}  # normalisation reference
        _cache[corpus] = dict(graph=graph, pr=pr, pr_norm=normalise_log(pr, clean_urls),
                              content_urls=content_urls, iterations=iters)
    return _cache[corpus]


def main() -> None:
    st = get_state()
    graph, pr = st["graph"], st["pr"]
    indeg = {u: 0 for u in graph}
    for outs in graph.values():
        for v in outs:
            indeg[v] += 1
    n_edges = sum(len(o) for o in graph.values())
    print(f"graph: {len(graph)} pages, {n_edges} links between crawled pages; "
          f"converged in {st['iterations']} iterations (d={DAMPING}); sum PR = {sum(pr.values()):.6f}")
    print(f"uniform PR would be 1/N = {1 / len(graph):.6f}\n")

    # Pages that have chunks (topic pages) vs hub pages that only exist in the link graph.
    chunk_urls = st["content_urls"]
    ranked = sorted(pr, key=pr.get, reverse=True)

    def show(title: str, urls: list[str]) -> None:
        print(title)
        print(f"  {'rank':>4}  {'PageRank':>9}  {'norm':>5}  {'in':>4}  {'out':>4}  {'g(d)':>5}  url")
        for i, u in enumerate(urls, 1):
            q = quality(u)
            print(f"  {i:>4}  {pr[u]:.6f}  {q['pagerank_norm']:.3f}  {indeg[u]:>4}  {len(graph[u]):>4}"
                  f"  {q['g']:.3f}  {u}")
        print()

    show("Top 10 PageRank, all crawled pages (includes hub/index pages):", ranked[:10])
    show("Top 10 PageRank among content pages (pages that have chunks):",
         [u for u in ranked if u in chunk_urls][:10])
    show("Bottom 5 content pages:", [u for u in ranked if u in chunk_urls][-5:])

    gs = sorted(quality(u)["g"] for u in chunk_urls)
    print(f"g(d) over {len(gs)} content pages: min={gs[0]:.3f}  median={gs[len(gs) // 2]:.3f}  max={gs[-1]:.3f}")
    print(f"an untrusted page outside the graph (e.g. poisoned): {quality('https://example.org/x')}")


if __name__ == "__main__":
    main()
