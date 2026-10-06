"""Split every crawled page into section-level chunks with zones, and write data/chunks.jsonl.

    python -m index.chunker

The crawler already split each page at its headings (data/raw/<doc_id>.json -> "sections").
One section = one chunk. A chunk has three zones, which the index keeps apart so ranking can
weight them differently (zone weighting):
  title   - the page title (<h1>), shared by every chunk of the page
  heading - the section heading (<h2>/<h3>...), "" for the intro text before the first heading
  body    - the section text
Sections are retrieval units of a sensible size: small enough that a retrieved chunk is about one
thing (symptoms, causes, treatment...), big enough to answer a question from. Empty sections are skipped.
"""
import glob
import json
import os

RAW_DIR = os.path.join("data", "raw")
CHUNKS_PATH = os.path.join("data", "chunks.jsonl")


def page_to_chunks(page: dict) -> list[dict]:
    """One chunk record per non-empty section, in the shared CLAUDE.md format."""
    chunks = []
    for sec in page["sections"]:
        body = sec["text"].strip()
        if not body:
            continue
        chunks.append({
            "chunk_id": f"{page['doc_id']}-{len(chunks):02d}",  # <doc_id>-<section number>
            "doc_id": page["doc_id"],
            "url": page["url"],
            "host": page["host"],
            "title": page["title"],
            "heading": sec["heading"],
            "body": body,
            "crawled_at": page["crawled_at"],
            "is_poison": page.get("is_poison", False),
        })
    return chunks


def main():
    paths = sorted(glob.glob(os.path.join(RAW_DIR, "*.json")))
    n_chunks, no_chunks = 0, []
    with open(CHUNKS_PATH, "w", encoding="utf-8") as out:
        for p in paths:
            page = json.load(open(p, encoding="utf-8"))
            chunks = page_to_chunks(page)
            if not chunks:
                no_chunks.append(page["url"])
            for c in chunks:
                out.write(json.dumps(c, ensure_ascii=False) + "\n")
            n_chunks += len(chunks)
    print(f"{len(paths)} pages -> {n_chunks} chunks ({n_chunks / max(1, len(paths)):.1f} per page), "
          f"written to {CHUNKS_PATH}")
    if no_chunks:
        print(f"Pages with no non-empty section: {len(no_chunks)} {no_chunks[:5]}")


if __name__ == "__main__":
    main()
