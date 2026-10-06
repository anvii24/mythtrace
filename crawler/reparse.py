"""Rebuild the saved pages in data/raw/ from their stored HTML, without re-crawling.

    python -m crawler.reparse

Each data/raw/<doc_id>.json is re-extracted from data/raw/<doc_id>.html.gz with the current
extract(), so a fix to the parser applies to the whole corpus without touching the network.
Crawl metadata (url, depth, crawled_at, HTTP Last-Modified, links) is kept. The duplicate checks
are re-run over the new text in the original crawl order, since cleaning can change similarities.

Afterwards it reports boilerplate: any body line that appears on more than 20% of pages.
"""
import glob
import gzip
import json
import os
from collections import Counter

from crawler.crawl import (MIN_WORDS, RAW_DIR, SUMMARY_PATH, DuplicateDetector, content_hash, extract)

BOILERPLATE_SHARE = 0.2


def line_doc_counts(docs: list[dict]) -> Counter:
    """For each distinct body line: how many pages contain it (each page counted once)."""
    counts = Counter()
    for d in docs:
        counts.update({line for line in d["body"].split("\n") if line.strip()})
    return counts


def main():
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    docs = [json.load(open(p, encoding="utf-8")) for p in paths]
    docs.sort(key=lambda d: d["crawled_at"])  # original crawl order, so dedup keeps the first-seen copy
    before = line_doc_counts(docs)

    dedup = DuplicateDetector()
    changed, short, dups = 0, [], []
    for d in docs:
        with gzip.open(os.path.join(RAW_DIR, f"{d['doc_id']}.html.gz"), "rb") as f:
            page = extract(f.read(), d["url"])
        if page["body"] != d["body"]:
            changed += 1
        d.update(title=page["title"], sections=page["sections"], body=page["body"],
                 last_modified_page=page["page_modified"], content_hash=content_hash(page["body"]))
        if len(page["body"].split()) < MIN_WORDS:
            short.append(d["url"])
        kind, other, sim = dedup.check(d["body"])
        if kind:
            dups.append({"url": d["url"], "kind": kind, "duplicate_of": other, "similarity": round(sim, 3)})
        else:
            dedup.add(d["url"], d["body"])
        d.update(max_similarity=round(sim, 3), most_similar=other)
        with open(os.path.join(RAW_DIR, f"{d['doc_id']}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=1)

    after = line_doc_counts(docs)
    n = len(docs)
    print(f"Re-extracted {n} pages from stored HTML; body text changed on {changed}.")
    print(f"Pages now under {MIN_WORDS} words: {len(short)} {short[:5]}")
    print(f"Duplicates after re-extraction: {len(dups)} {dups[:3]}")
    print(f"\nBody lines on more than {BOILERPLATE_SHARE:.0%} of pages ({BOILERPLATE_SHARE * n:.0f}):")
    print(f"  {'before':>6} {'after':>6}  line")
    frequent = {l for l, c in before.items() if c > BOILERPLATE_SHARE * n} | \
               {l for l, c in after.items() if c > BOILERPLATE_SHARE * n}
    for line in sorted(frequent, key=lambda l: -before[l]):
        print(f"  {before[line]:>6} {after[line]:>6}  {line[:90]!r}")
    if not frequent:
        print("  (none)")
    print("\nMost repeated lines now (top 8):")
    for line, c in after.most_common(8):
        print(f"  {c:>4} ({c / n:.0%})  {line[:90]!r}")

    # Record the re-extraction in the crawl summary.
    if os.path.exists(SUMMARY_PATH):
        summary = json.load(open(SUMMARY_PATH, encoding="utf-8"))
        summary["reparse"] = {
            "pages": n,
            "bodies_changed": changed,
            "pages_under_min_words": len(short),
            "duplicates": dups,
            "boilerplate_lines_over_20pct_after": {l: after[l] for l in frequent if after[l] > BOILERPLATE_SHARE * n},
        }
        with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        print(f"\nUpdated {SUMMARY_PATH} (added 'reparse').")


if __name__ == "__main__":
    main()
