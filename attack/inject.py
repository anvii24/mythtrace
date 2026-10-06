"""Inject the poisoned pages from attack/poison_pages.json into a SEPARATE copy of the corpus.

    python -m attack.inject               # build the poisoned corpus + index, then show where poison ranks
    python -m attack.inject --no-report   # build only

The clean corpus (data/chunks.jsonl, data/links.json, data/index/index_*.json) is only read, never
written. Everything the attack produces goes to its own files:
    data/chunks_poisoned.jsonl            clean chunks (is_poison=false) + one chunk per poison page (is_poison=true)
    data/links_poisoned.json              clean link graph + external poison pages as nodes with no inlinks
    data/index/index_poisoned_*.json      positional index over the poisoned chunks (stemmed + unstemmed)
    data/poison_manifest.json             poison id -> chunk_id, target question, false-claim keywords (for eval)
Then search(..., corpus="poisoned") retrieves from it.

Two kinds of attacker (field attack_type in poison_pages.json):
  external - the attacker runs their own site (e.g. desi-health-remedies.example). The page gets its own
             URL, host and doc_id. Nobody links to it, so in the link graph it is a node with no inlinks:
             it only gets the PageRank of the random-surfer teleport. Its host is not trusted.
  insider  - the false section is slipped into a real MedlinePlus page (target_url), like a compromised
             or vandalised trusted page. The chunk gets that page's url, host, title and doc_id, and the
             next section number, so it inherits the page's trusted host AND its PageRank. Source
             quality g(d) cannot catch this one; it needs a content-based defense.

Term stuffing: the bodies repeat the target question's words many times. That raises tf for exactly
the query terms, which is what tf-idf and BM25 reward (BM25 saturates tf, tf-idf's log tf only damps it).
"""
import argparse
import hashlib
import json
import os
import textwrap
from datetime import datetime, timezone

from index.build_index import (CHUNKS_PATH, build_index, chunks_path, index_path, load_chunks,
                               save_index)
from ranking.quality import LINKS_PATH, LINKS_PATHS

POISON_PAGES_PATH = os.path.join("attack", "poison_pages.json")
MANIFEST_PATH = os.path.join("data", "poison_manifest.json")

COMMON_FIELDS = ("id", "target_question", "false_claim", "false_claim_keywords", "attack_type",
                 "title", "heading", "body")
EXTRA_FIELDS = {"external": ("url", "host"), "insider": ("target_url",)}


def doc_id_for(url: str) -> str:
    """Same doc_id scheme as the crawler (crawler/crawl.py): first 12 hex chars of sha1(url)."""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]


def load_poison_pages(path: str = POISON_PAGES_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        pages = json.load(f)
    if not isinstance(pages, list) or not all(isinstance(p, dict) for p in pages):
        raise ValueError(f"{path} must be ONE flat JSON list of page objects (no nested lists)")
    seen = set()
    for p in pages:
        kind = p.get("attack_type")
        if kind not in EXTRA_FIELDS:
            raise ValueError(f"{p.get('id')}: attack_type must be 'external' or 'insider', got {kind!r}")
        missing = [k for k in COMMON_FIELDS + EXTRA_FIELDS[kind] if not p.get(k)]
        if missing:
            raise ValueError(f"{p.get('id')}: missing fields {missing}")
        if p["id"] in seen:
            raise ValueError(f"duplicate poison id {p['id']}")
        seen.add(p["id"])
    return pages


def make_poison_chunks(pages: list[dict], clean: list[dict]) -> list[dict]:
    """One chunk (shared chunk record format) per poison page."""
    by_url: dict[str, list[dict]] = {}
    for c in clean:
        by_url.setdefault(c["url"], []).append(c)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    next_section: dict[str, int] = {}  # insider pages: next free section number per doc_id

    out = []
    for p in pages:
        if p["attack_type"] == "external":
            doc_id = doc_id_for(p["url"])
            chunk = {"chunk_id": f"{doc_id}-00", "doc_id": doc_id, "url": p["url"], "host": p["host"],
                     "title": p["title"], "crawled_at": now}
        else:
            host_chunks = by_url.get(p["target_url"])
            if not host_chunks:
                raise ValueError(f"{p['id']}: target_url {p['target_url']} is not in the clean corpus")
            first = host_chunks[0]
            doc_id = first["doc_id"]
            n = next_section.get(doc_id, len(host_chunks))  # sections are numbered 00, 01, ...
            next_section[doc_id] = n + 1
            # Inherit everything page-level from the real page: url, host, title, crawl date.
            chunk = {"chunk_id": f"{doc_id}-{n:02d}", "doc_id": doc_id, "url": first["url"],
                     "host": first["host"], "title": first["title"], "crawled_at": first["crawled_at"]}
            if p["title"] != first["title"]:
                print(f"  note: {p['id']} is an insider page; using the real page title "
                      f"{first['title']!r} instead of {p['title']!r}")
        chunk.update(heading=p["heading"], body=p["body"], is_poison=True)
        # Same key order as the clean chunks
        out.append({k: chunk[k] for k in ("chunk_id", "doc_id", "url", "host", "title", "heading",
                                           "body", "crawled_at", "is_poison")})
    return out


def poisoned_link_graph(pages: list[dict]) -> dict[str, list[str]]:
    """Clean link graph + one node per external poison page, with no inlinks and no outlinks."""
    with open(LINKS_PATH, encoding="utf-8") as f:
        links = json.load(f)
    for p in pages:
        if p["attack_type"] == "external":
            links[p["url"]] = []
    # Sanity check: no real page links to an external poison page.
    ext = {p["url"] for p in pages if p["attack_type"] == "external"}
    assert not any(v in ext for outs in links.values() for v in outs), "a clean page links to poison"
    return links


def _fingerprint(paths: list[str]) -> dict[str, str]:
    """sha1 of each clean file, to prove the clean corpus is untouched."""
    out = {}
    for p in paths:
        with open(p, "rb") as f:
            out[p] = hashlib.sha1(f.read()).hexdigest()
    return out


def inject() -> tuple[list[dict], list[dict]]:
    clean_files = [CHUNKS_PATH, LINKS_PATH, index_path(True), index_path(False)]
    before = _fingerprint(clean_files)

    pages = load_poison_pages()
    clean = load_chunks(CHUNKS_PATH)
    for c in clean:
        c.setdefault("is_poison", False)
    poison = make_poison_chunks(pages, clean)
    chunks = clean + poison

    with open(chunks_path("poisoned"), "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    with open(LINKS_PATHS["poisoned"], "w", encoding="utf-8") as f:
        json.dump(poisoned_link_graph(pages), f, indent=1)
    manifest = [{"id": p["id"], "chunk_id": c["chunk_id"], "attack_type": p["attack_type"],
                 "url": c["url"], "target_question": p["target_question"],
                 "false_claim": p["false_claim"], "false_claim_keywords": p["false_claim_keywords"]}
                for p, c in zip(pages, poison)]
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)

    print(f"{len(clean)} clean chunks + {len(poison)} poison chunks -> {chunks_path('poisoned')}")
    for m in manifest:
        print(f"  {m['id']}  {m['attack_type']:<8} chunk {m['chunk_id']}  {m['url']}")
    print(f"wrote {LINKS_PATHS['poisoned']} and {MANIFEST_PATH}")
    for stem in (True, False):
        idx = build_index(chunks, stem=stem)
        save_index(idx, index_path(stem, "poisoned"))
        print(f"wrote {index_path(stem, 'poisoned')}  (N = {idx['N']})")

    assert _fingerprint(clean_files) == before, "clean corpus files changed!"
    print("clean corpus files unchanged (sha1 checked): " + ", ".join(clean_files))
    return pages, manifest


def report(manifest: list[dict], k_show: int = 5) -> None:
    """Where does each poison chunk rank for its own target question? (defenses off, poisoned corpus)"""
    from ranking.search import search  # imported late: it caches the index files we just wrote

    for m in manifest:
        q = m["target_question"]
        print(f"\n{'=' * 100}\n{m['id']} ({m['attack_type']})  target question: \"{q}\"\n"
              f"poison chunk: {m['chunk_id']}  {m['url']}")
        for mode in ("tfidf", "bm25"):
            ranked = search(q, k=100_000, mode=mode, defenses=False, corpus="poisoned")
            rank = next((i for i, r in enumerate(ranked, 1) if r["chunk_id"] == m["chunk_id"]), None)
            print(f"\n  [{mode}, defenses off]  poison rank = {rank} of {len(ranked)} matching chunks")
            for i, r in enumerate(ranked[:k_show], 1):
                tag = "  <== POISON" if r["chunk_id"] == m["chunk_id"] else ""
                b = r["score_breakdown"]
                print(f"    {i}. raw={b['relevance_raw']:.4f} norm={b['relevance_norm']:.4f}  "
                      f"[{r['chunk_id']}] {r['title']} > {r['heading']}{tag}")
            if rank:
                terms = ranked[rank - 1]["score_breakdown"]["terms"]
                print("    poison chunk per-term contributions: " + ", ".join(
                    f"{t} tf_w={b['tf_w']} +{b['contribution']:.3f}" for t, b in terms.items()))
        print("\n  " + textwrap.shorten("false claim: " + m["false_claim"], 98))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the poisoned corpus and index.")
    parser.add_argument("--no-report", action="store_true", help="skip the poison rank report")
    args = parser.parse_args()
    _, manifest = inject()
    if not args.no_report:
        report(manifest)


if __name__ == "__main__":
    main()
