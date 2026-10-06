"""Build a positional inverted index over data/chunks.jsonl, one stemmed and one unstemmed.

    python -m index.build_index            # build both indexes, then show a short demo
    python -m index.build_index --no-demo  # build only

    python -m index.build_index --corpus poisoned   # same, over data/chunks_poisoned.jsonl (attack/inject.py)

Writes data/index/index_stemmed.json and data/index/index_unstemmed.json (clean corpus), or
data/index/index_poisoned_stemmed.json / index_poisoned_unstemmed.json (poisoned corpus), each holding:
  N              - total number of chunks (needed for idf = log(N / df))
  chunk_lengths  - {chunk_id: {"title": n, "heading": n, "body": n, "total": n}} in terms
                   (BM25 length normalisation needs each chunk's length and the average length)
  avg_length     - average of each of those lengths over all chunks
  index          - the dictionary: {term: {"df": int, "postings": [[chunk_id, {zone: [positions]}], ...]}}

Inverted index: for each term (the dictionary) we keep a postings list of the chunks that contain it.
  df  = document frequency = how many chunks contain the term (in any zone) = len(postings)
  tf  = term frequency in one zone of one chunk = number of positions stored for that zone
Positional index: each posting stores WHERE the term occurs, separately per zone (title/heading/body),
  so we can answer phrase queries ("type 2 diabetes" = the three terms at positions p, p+1, p+2 in
  the same zone) and later weight a title match higher than a body match (zone weighting).
Positions count terms AFTER text processing (stop words removed), because queries go through the same
  preprocess(); "diabetes in children" and "diabetes children" both become adjacent terms.
Postings are sorted by chunk_id, so two postings lists can be intersected with a linear merge.
"""
import argparse
import json
import os
from collections import defaultdict

from index.text import preprocess

CHUNKS_PATH = os.path.join("data", "chunks.jsonl")
# Two corpora that never share files: the clean crawl, and a copy with poisoned chunks injected.
CHUNKS_PATHS = {"clean": CHUNKS_PATH, "poisoned": os.path.join("data", "chunks_poisoned.jsonl")}
INDEX_DIR = os.path.join("data", "index")
ZONES = ("title", "heading", "body")


def chunks_path(corpus: str = "clean") -> str:
    if corpus not in CHUNKS_PATHS:
        raise ValueError(f"unknown corpus {corpus!r} (use 'clean' or 'poisoned')")
    return CHUNKS_PATHS[corpus]


def index_path(stem: bool, corpus: str = "clean") -> str:
    chunks_path(corpus)  # validates the corpus name
    prefix = "index_" if corpus == "clean" else f"index_{corpus}_"
    return os.path.join(INDEX_DIR, prefix + ("stemmed.json" if stem else "unstemmed.json"))


def load_chunks(path: str = CHUNKS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_index(chunks: list[dict], stem: bool) -> dict:
    """One pass over the chunks: tokenise each zone and record (term -> chunk -> zone -> positions)."""
    # term -> chunk_id -> zone -> [positions]
    postings = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    chunk_lengths = {}

    for chunk in chunks:
        cid = chunk["chunk_id"]
        lengths = {}
        for zone in ZONES:
            terms = preprocess(chunk[zone], stem=stem)
            lengths[zone] = len(terms)
            for pos, term in enumerate(terms):  # positions restart at 0 in each zone
                postings[term][cid][zone].append(pos)
        lengths["total"] = sum(lengths[z] for z in ZONES)
        chunk_lengths[cid] = lengths

    # Freeze into plain JSON-able structures: postings sorted by chunk_id, df = postings length.
    index = {}
    for term in sorted(postings):
        plist = [[cid, dict(zones)] for cid, zones in sorted(postings[term].items())]
        index[term] = {"df": len(plist), "postings": plist}

    n = len(chunks)
    avg_length = {k: sum(l[k] for l in chunk_lengths.values()) / n for k in (*ZONES, "total")}
    return {"stem": stem, "N": n, "avg_length": avg_length,
            "chunk_lengths": chunk_lengths, "index": index}


def save_index(idx: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(idx, f)


def load_index(stem: bool = True, corpus: str = "clean") -> dict:
    with open(index_path(stem, corpus), encoding="utf-8") as f:
        return json.load(f)


def phrase_match(idx: dict, phrase: str) -> list[tuple[str, str, int]]:
    """Positional phrase query. Returns (chunk_id, zone, start position) for every match.

    1. Preprocess the phrase exactly like the documents -> terms t0, t1, ..., tk.
    2. Intersect the postings lists of all terms (linear merge on sorted chunk_ids) -> candidate chunks
       that contain every term somewhere. Rarest term first, so the candidate list shrinks fastest.
    3. In each candidate, for each zone, a match is a position p with t0 at p, t1 at p+1, ..., tk at p+k.
    """
    terms = preprocess(phrase, stem=idx["stem"])
    if not terms or any(t not in idx["index"] for t in terms):
        return []

    # Step 2: intersect postings by increasing df.
    by_df = sorted(set(terms), key=lambda t: idx["index"][t]["df"])
    candidates = [cid for cid, _ in idx["index"][by_df[0]]["postings"]]
    for term in by_df[1:]:
        other = [cid for cid, _ in idx["index"][term]["postings"]]
        merged, i, j = [], 0, 0
        while i < len(candidates) and j < len(other):  # classic two-pointer postings merge
            if candidates[i] == other[j]:
                merged.append(candidates[i]); i += 1; j += 1
            elif candidates[i] < other[j]:
                i += 1
            else:
                j += 1
        candidates = merged

    # Step 3: check positions zone by zone.
    zone_pos = {t: dict(idx["index"][t]["postings"]) for t in set(terms)}  # term -> {cid: {zone: [pos]}}
    matches = []
    for cid in candidates:
        for zone in ZONES:
            pos_sets = [set(zone_pos[t][cid].get(zone, [])) for t in terms]
            for p in sorted(pos_sets[0]):
                if all(p + k in pos_sets[k] for k in range(1, len(terms))):
                    matches.append((cid, zone, p))
    return matches


def demo(idx: dict, chunks_by_id: dict) -> None:
    label = "stemmed" if idx["stem"] else "unstemmed"
    term = preprocess("diabetes", stem=idx["stem"])[0]
    print(f"\n=== {label} index ===")
    print(f"N (chunks) = {idx['N']}   vocabulary size = {len(idx['index'])}")
    print("avg length (terms): " + ", ".join(f"{k}={v:.1f}" for k, v in idx["avg_length"].items()))

    entry = idx["index"].get(term)
    print(f"\n'diabetes' -> term '{term}', df = {entry['df'] if entry else 0}; first 5 postings:")
    for cid, zones in (entry["postings"][:5] if entry else []):
        tf = {z: len(p) for z, p in zones.items()}
        print(f"  {cid:<18} positions={zones}  tf={tf}  title={chunks_by_id[cid]['title']!r}")

    phrase = "type 2 diabetes"
    matches = phrase_match(idx, phrase)
    chunks_hit = sorted({cid for cid, _, _ in matches})
    print(f"\nphrase '{phrase}' -> terms {preprocess(phrase, stem=idx['stem'])}: "
          f"{len(matches)} matches in {len(chunks_hit)} chunks")
    for cid, zone, p in matches[:5]:
        terms = preprocess(chunks_by_id[cid][zone], stem=False)
        print(f"  {cid:<18} {zone:<7} pos {p}: ...{' '.join(terms[max(0, p - 3):p + 6])}...")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build stemmed + unstemmed positional inverted indexes.")
    parser.add_argument("--no-demo", action="store_true", help="skip the demo output")
    parser.add_argument("--corpus", choices=sorted(CHUNKS_PATHS), default="clean")
    args = parser.parse_args()

    chunks = load_chunks(chunks_path(args.corpus))
    chunks_by_id = {c["chunk_id"]: c for c in chunks}
    for stem in (True, False):
        idx = build_index(chunks, stem=stem)
        path = index_path(stem, args.corpus)
        save_index(idx, path)
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
        if not args.no_demo:
            demo(idx, chunks_by_id)


if __name__ == "__main__":
    main()
