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
A section under MIN_CHUNK_WORDS words is too small to rank or answer from on its own, so it is merged
into the next section of the same page (or the previous one if it is the last); the merged chunk keeps
the heading of whichever section came first.
"""
import glob
import json
import os

RAW_DIR = os.path.join("data", "raw")
CHUNKS_PATH = os.path.join("data", "chunks.jsonl")
MIN_CHUNK_WORDS = 25


def merge_short(sections: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Merge (heading, body) sections under MIN_CHUNK_WORDS into a neighbour on the same page."""
    merged, carry = [], None  # carry = short section(s) waiting to be joined onto the next one
    for heading, body in sections:
        if carry:
            heading, body = carry[0], carry[1] + "\n" + body  # keep the first section's heading
            carry = None
        if len(body.split()) < MIN_CHUNK_WORDS:
            carry = (heading, body)
            continue
        merged.append((heading, body))
    if carry:  # short section at the end of the page: join it onto the previous one
        if merged:
            merged[-1] = (merged[-1][0], merged[-1][1] + "\n" + carry[1])
        else:
            merged.append(carry)  # the whole page is short: keep it as a single chunk
    return merged


def page_to_chunks(page: dict) -> list[dict]:
    """One chunk record per non-empty section (short ones merged), in the shared CLAUDE.md format."""
    sections = [(sec["heading"], sec["text"].strip()) for sec in page["sections"] if sec["text"].strip()]
    chunks = []
    for heading, body in merge_short(sections):
        chunks.append({
            "chunk_id": f"{page['doc_id']}-{len(chunks):02d}",  # <doc_id>-<section number>
            "doc_id": page["doc_id"],
            "url": page["url"],
            "host": page["host"],
            "title": page["title"],
            "heading": heading,
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
