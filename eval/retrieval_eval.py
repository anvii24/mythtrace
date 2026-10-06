"""Retrieval quality on the CLEAN corpus: P@5, P@10, Hit@5 and MRR for Q01-Q20.

    python -m eval.retrieval_eval

Relevance judgments (qrels) are PAGE-level and were fixed before any search was run:
eval/questions.csv has an `expected_topic_pages` column (semicolon-separated MedlinePlus topic names,
chosen by a teammate). We map each name to the crawled page(s) with that title. A retrieved chunk is
relevant if its page (doc_id) is one of the question's expected pages. No chunk is hand-judged.

Title matching is exact (case-insensitive). Where the teammate's topic name is a different name for
the same MedlinePlus page, ALIASES gives the crawled title; every alias is written to the qrels file
so it can be checked. A name with no page in the crawl is reported as missing (we don't recrawl);
a question whose expected pages are ALL missing is skipped.

Metrics (IIR ch. 8), per question, then averaged over the evaluated questions:
    P@k   = (# relevant chunks in the top k) / k
    Hit@5 = 1 if at least one relevant chunk is in the top 5, else 0 (the LLM sees the top 5)
    MRR   = mean over questions of 1 / (rank of the first relevant chunk), ranking ALL matching
            chunks; 0 if no matching chunk is relevant.
Note: P@k is capped by how many chunks the expected pages have (a page has ~5-10 section chunks), so
P@10 can't reach 1.0 for a one-page question whose page has fewer than 10 chunks.

Configurations: bm25 / tfidf x stemmed / unstemmed with NO defenses (pure relevance), plus bm25
stemmed with ALL defenses (quality g(d) + Jaccard content check), to check that the defenses, built
against poisoning, don't hurt normal searches.

Outputs:
    eval/results/qrels_pages.json        question -> expected topic -> matched page(s)
    eval/results/retrieval_results.csv   one row per (question, configuration), plus mean rows
"""
import csv
import json
import os

from ranking.search import search

QUESTIONS_PATH = os.path.join("eval", "questions.csv")
CHUNKS_PATH = os.path.join("data", "chunks.jsonl")
RESULTS_DIR = os.path.join("eval", "results")
QRELS_PATH = os.path.join(RESULTS_DIR, "qrels_pages.json")
CSV_PATH = os.path.join(RESULTS_DIR, "retrieval_results.csv")

# Teammate's topic name -> crawled page title, only where both name the same MedlinePlus page.
ALIASES = {
    "influenza": "Flu",                                     # flu.html: "The flu, also called influenza"
    "nutrition during pregnancy": "Pregnancy and Nutrition",  # same topic, words reordered
}

# name -> (mode, stem, defenses)
CONFIGS = {
    "bm25_stem":          ("bm25", True, False),
    "bm25_nostem":        ("bm25", False, False),
    "tfidf_stem":         ("tfidf", True, False),
    "tfidf_nostem":       ("tfidf", False, False),
    "bm25_stem_defenses": ("bm25", True, True),
}
METRICS = ("P@5", "P@10", "Hit@5", "MRR")


def load_questions() -> list[dict]:
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_pages() -> dict[str, dict]:
    """Lower-cased title -> {doc_id, title, url, n_chunks} for every page in the clean corpus."""
    pages: dict[str, dict] = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            p = pages.setdefault(c["title"].lower(), {"doc_id": c["doc_id"], "title": c["title"],
                                                      "url": c["url"], "n_chunks": 0})
            p["n_chunks"] += 1
    return pages


def build_qrels(questions: list[dict], pages: dict[str, dict]) -> dict:
    """{qid: {"question", "expected": [{"topic", "matched_by", "page" or None}], "relevant_doc_ids"}}"""
    qrels = {}
    for q in questions:
        if q["answerable"].strip().lower() != "yes":
            continue
        expected = []
        for topic in (t.strip() for t in q["expected_topic_pages"].split(";") if t.strip()):
            if topic.lower() in pages:
                expected.append({"topic": topic, "matched_by": "exact title", "page": pages[topic.lower()]})
            elif topic.lower() in ALIASES:
                alias = ALIASES[topic.lower()]
                expected.append({"topic": topic, "matched_by": f"alias -> {alias!r}", "page": pages[alias.lower()]})
            else:
                expected.append({"topic": topic, "matched_by": "MISSING from corpus", "page": None})
        qrels[q["qid"]] = {"question": q["question"], "expected": expected,
                           "relevant_doc_ids": sorted({e["page"]["doc_id"] for e in expected if e["page"]})}
    return qrels


def evaluate(question: str, relevant: set[str], mode: str, stem: bool, defenses: bool,
             doc_of: dict[str, str]) -> dict:
    """Metrics for one question. Ranks ALL matching chunks (for MRR); top 10 are a prefix of it."""
    ranked = search(question, k=100_000, mode=mode, stem=stem, defenses=defenses, corpus="clean")
    rel = [doc_of[r["chunk_id"]] in relevant for r in ranked]   # relevance of rank 1, 2, ...
    first = next((i for i, is_rel in enumerate(rel, 1) if is_rel), None)
    return {"P@5": sum(rel[:5]) / 5, "P@10": sum(rel[:10]) / 10, "Hit@5": float(any(rel[:5])),
            "MRR": 1 / first if first else 0.0, "first_rel_rank": first, "n_matching": len(ranked),
            "top5": [f"{r['title']} > {r['heading']}" for r in ranked[:5]]}


def print_qrels(qrels: dict) -> None:
    print("Page-level qrels (expected_topic_pages -> crawled page)")
    for qid, q in qrels.items():
        print(f"  {qid}  {q['question']}")
        for e in q["expected"]:
            page = f"{e['page']['title']}  ({e['page']['doc_id']}, {e['page']['n_chunks']} chunks)" if e["page"] else "-"
            print(f"         {e['topic']:<28} {e['matched_by']:<34} {page}")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    questions = load_questions()
    pages = load_pages()
    qrels = build_qrels(questions, pages)
    with open(QRELS_PATH, "w", encoding="utf-8") as f:
        json.dump(qrels, f, indent=1)
    print_qrels(qrels)

    missing = [(qid, e["topic"]) for qid, q in qrels.items() for e in q["expected"] if not e["page"]]
    skipped = [qid for qid, q in qrels.items() if not q["relevant_doc_ids"]]
    print(f"\nMissing expected pages: {len(missing)}  " + ", ".join(f"{qid}: {t}" for qid, t in missing))
    print(f"Skipped questions (all expected pages missing): {len(skipped)}  {' '.join(skipped)}")
    kept = [qid for qid in qrels if qid not in skipped]

    doc_of = {}
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            doc_of[c["chunk_id"]] = c["doc_id"]

    rows = []
    for name, (mode, stem, defenses) in CONFIGS.items():
        for qid in kept:
            m = evaluate(qrels[qid]["question"], set(qrels[qid]["relevant_doc_ids"]), mode, stem, defenses, doc_of)
            rows.append({"qid": qid, "config": name, "mode": mode, "stem": stem, "defenses": defenses, **m})

    # Per-question table: P@5 / MRR for every configuration side by side
    print(f"\nPer question (P@5 / P@10 / MRR), {len(kept)} questions")
    print(f"  {'qid':<4}" + "".join(f"{n:>22}" for n in CONFIGS) + "   bm25_stem top 1")
    for qid in kept:
        cells = []
        for name in CONFIGS:
            r = next(r for r in rows if r["qid"] == qid and r["config"] == name)
            cells.append(f"{r['P@5']:.1f} / {r['P@10']:.1f} / {r['MRR']:.2f}")
        top1 = next(r for r in rows if r["qid"] == qid and r["config"] == "bm25_stem")["top5"][:1]
        print(f"  {qid:<4}" + "".join(f"{c:>22}" for c in cells) + "   " + (top1[0] if top1 else "-"))

    # Summary table: mean over evaluated questions
    means = []
    print(f"\nMean over {len(kept)} questions (skipped {len(skipped)})")
    print(f"  {'configuration':<20}" + "".join(f"{m:>8}" for m in METRICS))
    for name, (mode, stem, defenses) in CONFIGS.items():
        sel = [r for r in rows if r["config"] == name]
        mean = {m: sum(r[m] for r in sel) / len(sel) for m in METRICS}
        means.append({"qid": "MEAN", "config": name, "mode": mode, "stem": stem, "defenses": defenses, **mean})
        print(f"  {name:<20}" + "".join(f"{mean[m]:>8.3f}" for m in METRICS))
    # Best possible P@k: a perfect ranking puts every relevant chunk first, but there may be < k of them
    n_rel = {qid: sum({e["page"]["doc_id"]: e["page"]["n_chunks"]
                       for e in qrels[qid]["expected"] if e["page"]}.values()) for qid in kept}
    ceil = {k: sum(min(k, n_rel[qid]) / k for qid in kept) / len(kept) for k in (5, 10)}
    print(f"  {'(ceiling)':<20}{ceil[5]:>8.3f}{ceil[10]:>8.3f}{1:>8.3f}{1:>8.3f}"
          "   <- perfect ranking; expected pages have only " + ", ".join(f"{qid}:{n_rel[qid]}" for qid in kept
                                                                           if n_rel[qid] < 5) + " chunks")

    fields = ["qid", "config", "mode", "stem", "defenses", *METRICS, "first_rel_rank", "n_matching", "top5"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows + means:
            w.writerow({**r, "top5": " | ".join(r.get("top5", []))})
    print(f"\nwrote {QRELS_PATH} and {CSV_PATH} ({len(rows)} question rows + {len(means)} mean rows)")


if __name__ == "__main__":
    main()
