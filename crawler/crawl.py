"""The crawl loop: frontier -> robots check -> fetch -> parse -> dedup -> save -> enqueue links.

Run a small test crawl (MedlinePlus only, stop after 20 saved pages):
    python -m crawler.crawl --max-pages 20

For every URL the frontier hands out:
  1. robots.txt check        (RobotsCache; refuse if disallowed or robots.txt unavailable)
  2. download                (clear User-Agent, timeout, skip non-HTML by extension and Content-Type)
  3. parse with BeautifulSoup: title, sections (heading + text), body, links, last-modified dates
  4. "content seen?"         exact duplicate = same SHA-1 hash of the cleaned body text
  5. near-duplicate check    word k-shingles + Jaccard similarity >= threshold
  6. save                    data/raw/<doc_id>.json (+ the raw HTML, gzipped, for re-parsing offline)
  7. enqueue links           normalised, in-scope links go back into the frontier at depth + 1
The link graph (page -> in-scope outlinks) goes to data/links.json and a one-line-per-URL
log of what happened goes to data/crawl_log.jsonl.
"""
import argparse
import gzip
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from urllib.parse import urlsplit

import requests
from bs4 import BeautifulSoup

from crawler.frontier import Frontier
from crawler.robots import MIN_DELAY, USER_AGENT, RobotsCache
from crawler.urls import get_host, normalize_url

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
SITEMAP_CACHE = os.path.join(RAW_DIR, "_sitemaps")
LINKS_PATH = os.path.join(DATA_DIR, "links.json")
LOG_PATH = os.path.join(DATA_DIR, "crawl_log.jsonl")

TIMEOUT = 15            # seconds per request
MAX_BYTES = 5_000_000   # skip anything bigger than 5 MB
MIN_WORDS = 50          # pages with less real text (e.g. A-Z index pages) are not saved, but their links are followed
SHINGLE_K = 4           # words per shingle
NEAR_DUP_THRESHOLD = 0.9
# Link density = share of a section's characters that are link text. Measured on MedlinePlus:
# real content 0.00-0.17, lists of external resources / TOC / image galleries 0.23-0.94.
LINK_DENSITY_MAX = 0.2

SKIP_EXTENSIONS = (".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".mp3", ".mp4",
                   ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".xml", ".css", ".js")

# Crawl scope: which hosts, and which paths on them, count as corpus pages.
# MedlinePlus: English health topics (/diabetes.html), encyclopedia articles, lab-test pages.
# Spanish (/spanish/...), drug monographs, and everything else are out of scope.
SCOPE = {
    "medlineplus.gov": re.compile(r"/[a-z0-9_\-]+\.html|/ency/article/\d+\.htm|/lab-tests/[a-z0-9\-]+"),
    "www.mohfw.gov.in": re.compile(r"/.*"),
}


def in_scope(url: str, hosts: set[str]) -> bool:
    parts = urlsplit(url)
    pattern = SCOPE.get(parts.netloc)
    return parts.netloc in hosts and pattern is not None and bool(pattern.fullmatch(parts.path))


# ---------------------------------------------------------------- seeds

def sitemap_seeds(host: str, robots: RobotsCache, hosts: set[str]) -> list[str]:
    """In-scope URLs from the host's sitemap(s) listed in robots.txt (cached on disk for a day)."""
    urls = []
    for sitemap_url in robots.sitemaps(f"https://{host}/"):
        os.makedirs(SITEMAP_CACHE, exist_ok=True)
        cache = os.path.join(SITEMAP_CACHE, host + "_" + os.path.basename(urlsplit(sitemap_url).path))
        if os.path.exists(cache) and time.time() - os.path.getmtime(cache) < 86400:
            xml = open(cache, encoding="utf-8").read()
        else:
            if not robots.can_fetch(sitemap_url):
                continue
            resp = requests.get(sitemap_url, headers={"User-Agent": USER_AGENT}, timeout=60)
            resp.raise_for_status()
            xml = resp.text
            with open(cache, "w", encoding="utf-8") as f:
                f.write(xml)
            time.sleep(robots.crawl_delay(sitemap_url))  # the sitemap fetch counts as a request to this host
        for loc in re.findall(r"<loc>\s*(.*?)\s*</loc>", xml):
            url = normalize_url(loc)
            if url and in_scope(url, hosts):
                urls.append(url)
    return urls


# ---------------------------------------------------------------- parsing

def clean_text(text: str) -> str:
    """Collapse whitespace and remove the stray space BeautifulSoup leaves before punctuation."""
    text = " ".join(text.split())
    text = re.sub(r"\s+([,.;:!?)])", r"\1", text)
    return re.sub(r"\(\s+", "(", text)


HEADINGS = ("h1", "h2", "h3", "h4")
BLOCKS = ("p", "li", "dd", "dt", "td", "th", "pre", "blockquote")


def extract(html: bytes, url: str) -> dict:
    """Pull title, sections, body text, links and the page's own modified date out of the HTML."""
    soup = BeautifulSoup(html, "html.parser")

    def meta(name):
        tag = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        return (tag.get("content") or "").strip() or None if tag else None

    h1 = soup.find("h1")
    title = (clean_text(h1.get_text(" ")) if h1 else None) or meta("DC.Title") or \
        (clean_text(soup.title.get_text()).split(" | ")[0] if soup.title else url)
    page_modified = meta("DC.Date.Modified") or meta("article:modified_time") or meta("last-modified")

    # Links come from the whole page (navigation included): they build the link graph.
    links = []
    for a in soup.find_all("a", href=True):
        link = normalize_url(a["href"], base=url)
        if link and link != url and link not in links:
            links.append(link)

    # Main content: drop non-content tags, then use <article>/<main> if the page has one.
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "form", "iframe", "svg", "button"]):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup

    # Walk headings and text blocks in document order; each heading starts a new section.
    sections = [{"heading": "", "parts": [], "chars": 0, "link_chars": 0}]
    for el in root.find_all(HEADINGS + BLOCKS):
        if el.find_parent(BLOCKS) is not None:
            continue  # nested block (e.g. <p> inside <li>): its text is already in the parent
        text = clean_text(el.get_text(" "))
        if not text:
            continue
        if el.name in HEADINGS:
            if el.name != "h1":  # h1 is the page title
                sections.append({"heading": text, "parts": [], "chars": 0, "link_chars": 0})
            continue
        sec = sections[-1]
        sec["parts"].append(text)
        sec["chars"] += len(text)
        sec["link_chars"] += sum(len(clean_text(a.get_text(" "))) for a in el.find_all("a"))

    kept = []
    for sec in sections:
        if not sec["parts"]:
            continue
        if sec["link_chars"] / sec["chars"] > LINK_DENSITY_MAX:
            continue  # mostly links: a table of contents or list of external resources
        kept.append({"heading": sec["heading"], "text": "\n".join(sec["parts"])})

    body = "\n\n".join(s["text"] for s in kept)
    return {"title": title, "sections": kept, "body": body, "links": links, "page_modified": page_modified}


# ---------------------------------------------------------------- duplicate detection

def content_hash(body: str) -> str:
    """'Content seen?' fingerprint: SHA-1 of the cleaned, lowercased text."""
    return hashlib.sha1(" ".join(body.lower().split()).encode("utf-8")).hexdigest()


def shingles(body: str, k: int = SHINGLE_K) -> set[int]:
    """Set of word k-shingles (every run of k consecutive words), each hashed to a 64-bit int.

    Hashing keeps memory small; a stable hash (not Python's per-run hash()) keeps results
    reproducible between runs.
    """
    words = re.findall(r"[a-z0-9]+", body.lower())
    grams = (" ".join(words[i:i + k]) for i in range(max(1, len(words) - k + 1)))
    return {int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "big") for g in grams}


def jaccard(a: set, b: set) -> float:
    """|A ∩ B| / |A ∪ B|: 1.0 = identical shingle sets, 0.0 = nothing in common."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class DuplicateDetector:
    def __init__(self, threshold: float = NEAR_DUP_THRESHOLD):
        self.threshold = threshold
        self.hashes: dict[str, str] = {}               # content hash -> url
        self.docs: list[tuple[str, set[int]]] = []     # (url, shingle set) of every saved page
        self.comparisons = 0

    def check(self, body: str) -> tuple[str | None, str | None, float]:
        """Return (kind, duplicate_of_url, similarity); kind is 'exact', 'near' or None."""
        h = content_hash(body)
        if h in self.hashes:
            return "exact", self.hashes[h], 1.0
        sh = shingles(body)
        best_url, best_sim = None, 0.0
        for url, other in self.docs:
            # Jaccard can never exceed min/max of the set sizes, so skip pairs that can't reach the threshold.
            if min(len(sh), len(other)) / max(len(sh), len(other)) < self.threshold:
                continue
            self.comparisons += 1
            sim = jaccard(sh, other)
            if sim > best_sim:
                best_url, best_sim = url, sim
        if best_sim >= self.threshold:
            return "near", best_url, best_sim
        return None, best_url, best_sim

    def add(self, url: str, body: str):
        self.hashes[content_hash(body)] = url
        self.docs.append((url, shingles(body)))


# ---------------------------------------------------------------- crawl loop

def doc_id_for(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def fetch(session: requests.Session, url: str) -> tuple[requests.Response | None, str | None]:
    """Download one URL. Returns (response, None) for an HTML page, or (None, skip_reason)."""
    try:
        resp = session.get(url, timeout=TIMEOUT, stream=True, allow_redirects=True)
    except requests.RequestException as e:
        return None, f"fetch error ({type(e).__name__})"
    with resp:
        if resp.status_code != 200:
            return None, f"HTTP {resp.status_code}"
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype:
            return None, f"not HTML ({ctype.split(';')[0] or 'no Content-Type'})"
        if int(resp.headers.get("Content-Length") or 0) > MAX_BYTES:
            return None, "too large"
        resp._content = resp.raw.read(MAX_BYTES + 1, decode_content=True)
        if len(resp._content) > MAX_BYTES:
            return None, "too large"
    return resp, None


def crawl(seeds: list[str], hosts: set[str], max_pages: int, max_depth: int, max_per_host: int,
          threshold: float, robots: RobotsCache | None = None, verbose: bool = True) -> dict:
    os.makedirs(RAW_DIR, exist_ok=True)
    robots = robots or RobotsCache()
    frontier = Frontier(delay_fn=robots.crawl_delay, max_depth=max_depth, max_pages_per_host=max_per_host)
    dedup = DuplicateDetector(threshold)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    for url in seeds:
        frontier.add(url, depth=0)

    link_graph: dict[str, list[str]] = {}
    outcomes = Counter()
    duplicates = []
    max_fetches = max_pages * 5  # safety stop if almost everything gets skipped
    fetches = 0
    start = time.time()

    log = open(LOG_PATH, "w", encoding="utf-8")

    def record(url, status, reason=None, **extra):
        outcomes[status if status == "saved" else f"{status}: {reason}"] += 1
        log.write(json.dumps({"url": url, "status": status, "reason": reason, **extra}) + "\n")
        if verbose:
            tail = f"  ({reason})" if reason else ""
            print(f"[{fetches:>3}] {status:<9} {url}{tail}")

    try:
        while outcomes["saved"] < max_pages and fetches < max_fetches:
            item = frontier.next_url()
            if item is None:
                break
            url, depth = item
            fetches += 1
            try:
                if not robots.can_fetch(url):
                    record(url, "skipped", "disallowed by robots.txt or robots.txt unavailable")
                    continue
                if urlsplit(url).path.lower().endswith(SKIP_EXTENSIONS):
                    record(url, "skipped", "non-HTML extension")
                    continue

                resp, reason = fetch(session, url)
                crawled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                if resp is None:
                    record(url, "skipped", reason)
                    continue

                final_url = normalize_url(resp.url) or url
                if final_url != url and not in_scope(final_url, hosts):
                    record(url, "skipped", f"redirected out of scope to {final_url}")
                    continue

                page = extract(resp.content, final_url)
                in_scope_links = [l for l in page["links"] if in_scope(l, hosts)]
                link_graph[final_url] = in_scope_links
                for link in in_scope_links:
                    frontier.add(link, depth + 1)

                if len(page["body"].split()) < MIN_WORDS:
                    record(url, "skipped", f"too little text ({len(page['body'].split())} words; links followed)")
                    continue

                kind, other, sim = dedup.check(page["body"])
                if kind:
                    duplicates.append({"url": final_url, "kind": kind, "duplicate_of": other, "similarity": round(sim, 3)})
                    record(url, "duplicate", f"{kind} duplicate of {other} (Jaccard {sim:.2f})",
                           duplicate_of=other, similarity=round(sim, 3))
                    continue
                dedup.add(final_url, page["body"])

                doc_id = doc_id_for(final_url)
                doc = {
                    "doc_id": doc_id,
                    "url": final_url,
                    "host": get_host(final_url),
                    "title": page["title"],
                    "sections": page["sections"],
                    "body": page["body"],
                    "links": in_scope_links,
                    "depth": depth,
                    "crawled_at": crawled_at,
                    "last_modified_http": resp.headers.get("Last-Modified"),
                    "last_modified_page": page["page_modified"],
                    "content_hash": content_hash(page["body"]),
                    "max_similarity": round(sim, 3),
                    "most_similar": other,
                    "is_poison": False,
                }
                with open(os.path.join(RAW_DIR, f"{doc_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(doc, f, ensure_ascii=False, indent=1)
                with gzip.open(os.path.join(RAW_DIR, f"{doc_id}.html.gz"), "wb") as f:
                    f.write(resp.content)
                record(url, "saved", None, doc_id=doc_id, title=page["title"],
                       words=len(page["body"].split()), sections=len(page["sections"]))
                if verbose:
                    print(f"{'':>16}title={page['title']!r}  words={len(page['body'].split())}  "
                          f"sections={len(page['sections'])}  links={len(in_scope_links)}  "
                          f"modified(page)={page['page_modified']}  max_jaccard={sim:.2f}")
            finally:
                frontier.release(url)
    finally:
        log.close()
        with open(LINKS_PATH, "w", encoding="utf-8") as f:
            json.dump(link_graph, f, indent=1)

    return {
        "fetches": fetches,
        "outcomes": outcomes,
        "duplicates": duplicates,
        "comparisons": dedup.comparisons,
        "frontier_left": len(frontier),
        "frontier_rejected": dict(frontier.rejected),
        "graph_nodes": len(link_graph),
        "graph_edges": sum(len(v) for v in link_graph.values()),
        "seconds": time.time() - start,
    }


def main():
    ap = argparse.ArgumentParser(description="TrustRAG crawler")
    ap.add_argument("--hosts", nargs="+", default=["medlineplus.gov"])
    ap.add_argument("--seeds", nargs="*", help="seed URLs (default: in-scope URLs from each host's sitemap)")
    ap.add_argument("--max-pages", type=int, default=20, help="stop after this many saved pages")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--max-per-host", type=int, default=1500)
    ap.add_argument("--threshold", type=float, default=NEAR_DUP_THRESHOLD, help="near-duplicate Jaccard threshold")
    args = ap.parse_args()
    hosts = set(args.hosts)
    robots = RobotsCache()  # one cache for seeding and crawling, so robots.txt is fetched once per host

    if args.seeds:
        seeds = [u for u in (normalize_url(s) for s in args.seeds) if u]
    else:
        seeds = [u for h in hosts for u in sitemap_seeds(h, robots, hosts)]
    # Add seeds best-priority-first, so the per-host cap is spent on health topics, not encyclopedia pages.
    from crawler.frontier import default_priority
    seeds.sort(key=default_priority)
    print(f"{len(seeds)} in-scope seed URLs; crawling until {args.max_pages} pages are saved "
          f"(>= {MIN_DELAY:.0f}s between requests to a host)\n")

    s = crawl(seeds, hosts, args.max_pages, args.max_depth, args.max_per_host, args.threshold, robots)

    print("\n================ SUMMARY ================")
    print(f"URLs handed out by frontier : {s['fetches']}  in {s['seconds']:.0f}s")
    print(f"Pages saved                 : {s['outcomes']['saved']}  -> {RAW_DIR}/")
    for k, v in sorted(s["outcomes"].items()):
        if k != "saved":
            print(f"  {k:<60} {v}")
    print(f"Duplicates found            : {len(s['duplicates'])} "
          f"(exact {sum(d['kind'] == 'exact' for d in s['duplicates'])}, "
          f"near {sum(d['kind'] == 'near' for d in s['duplicates'])}; "
          f"{s['comparisons']} Jaccard comparisons)")
    print(f"Link graph                  : {s['graph_nodes']} pages, {s['graph_edges']} in-scope links -> {LINKS_PATH}")
    print(f"Frontier                    : {s['frontier_left']} URLs still queued; rejected {s['frontier_rejected']}")


if __name__ == "__main__":
    main()
