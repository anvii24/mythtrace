"""Answer-level run on the CLEAN corpus: all 25 questions through rag.answer.answer() (bm25, all
defenses, Sonnet, cached).

    python -m eval.clean_answers               # LLM calls for cache misses
    python -m eval.clean_answers --cache-only  # re-score cached answers; a cache miss is an error

Reports:
  (a) abstention on the unanswerable questions (answerable=no, Q21-Q25). An answer passes if it says
      it can't answer AND invents nothing. Automatic check: it either is the fixed ABSTAIN sentence,
      or it says the sources don't cover the question and contains none of the things the question
      tempts it to invent (prices / currency, numbers, application steps, doctor or hospital names).
      The non-ABSTAIN answers are printed in full so a person can confirm the verdict.
  (b) false abstentions on the answerable questions (Q01-Q20): the model returned ABSTAIN although
      the question is answerable. (Q10's expected page is not in the crawl, see eval/retrieval_eval.py,
      so an abstention there is defensible; it is still counted and marked.)
  (c) citation coverage on non-abstaining answers: the share of answer sentences that end with a
      valid citation [n] (n is one of the numbered sources in the prompt). Sentences are split on
      line breaks and on . ! ? followed by a space; "claim. [1]" is treated like "claim [1].".
      Lead-in lines that end with ":" ("Symptoms include [1]:") introduce a list and are not counted
      themselves. The model often cites a list once, at its lead-in; the strict rule counts the items
      as uncited, so a second figure counts items under a cited lead-in as cited.

Output: eval/results/clean_answers.jsonl, one line per question.
"""
import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from rag import answer as rag_answer
from rag.answer import ABSTAIN, MODEL, answer

QUESTIONS_PATH = os.path.join("eval", "questions.csv")
OUT_PATH = os.path.join("eval", "results", "clean_answers.jsonl")
RETRIEVER = {"k": 5, "mode": "bm25", "defenses": True, "corpus": "clean"}
NO_PAGE = {"Q10"}  # expected page missing from the crawl (eval/retrieval_eval.py)

# (c) citation coverage
CITE_GROUP = r"(?:\s*\[\d+(?:\s*,\s*\d+)*\])+"
TRAILING_CITES = re.compile(r"(" + CITE_GROUP + r")\s*[.!?;]*[\"')]*\s*$")
NUMS = re.compile(r"\d+")

# (a) things an unanswerable question tempts the model to invent
CANT_ANSWER = re.compile(r"(don't|do not|doesn't|does not|can't|cannot|no)\b.{0,80}"
                         r"\b(information|sources?|data|cover|include|contain|mention|provide|answer)", re.I)
INVENTED = {
    "number":   re.compile(r"\d"),
    "currency": re.compile(r"₹|\bRs\.?\s|\bINR\b|\brupees?\b|\$", re.I),
    "doctor":   re.compile(r"\bDr\.?\s+[A-Z]"),
    "hospital": re.compile(r"\b[A-Z][a-z]+\s+(Hospital|Clinic|Institute|Medical Cent(er|re))\b"),
    "steps":    re.compile(r"^\s*(\d+[.)]|step\s+\d)", re.I | re.M),
}


BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def sentences(text: str) -> list[tuple[str, str | None]]:
    """[(sentence, lead-in)]: lead-in is the "...include [1]:" line a bullet item sits under, else None."""
    # Move a citation that follows the full stop to before it: "claim. [1]" -> "claim [1]."
    text = re.sub(r"([.!?])(" + CITE_GROUP + r")", r"\2\1", text)
    out, lead_in = [], None
    for line in text.splitlines():
        is_item = bool(BULLET.match(line))
        if not is_item and line.strip():
            lead_in = None                       # a normal line ends the list
        line = BULLET.sub("", line).strip()
        for s in (s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()):
            if s.endswith(":"):
                lead_in = s                      # introduces a list; not counted itself
            else:
                out.append((s, lead_in if is_item else None))
    return out


def _cited(s: str, source_map: dict) -> bool:
    m = TRAILING_CITES.search(s.rstrip(":"))
    return bool(m) and all(int(n) in source_map for n in NUMS.findall(m.group(1)))


def citation_coverage(text: str, source_map: dict) -> tuple[int, int, int, list[str]]:
    """(# sentences ending in a valid citation, # of those or list items under a cited lead-in,
        # sentences, the uncited sentences by the strict rule)"""
    sents = sentences(text)
    uncited = [s for s, _ in sents if not _cited(s, source_map)]
    lenient = sum(1 for s, lead in sents if _cited(s, source_map) or (lead and _cited(lead, source_map)))
    return len(sents) - len(uncited), lenient, len(sents), uncited


def judge_unanswerable(out: dict) -> dict:
    if out["abstained"]:
        return {"pass": True, "says_cant_answer": True, "invented": []}
    says = bool(CANT_ANSWER.search(out["answer"]))
    invented = [name for name, rx in INVENTED.items() if rx.search(out["answer"])]
    return {"pass": says and not invented, "says_cant_answer": says, "invented": invented}


def run(q: dict) -> tuple[dict, dict]:
    return q, answer(q["question"], model=MODEL, **RETRIEVER)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache-only", action="store_true", help="a cache miss raises instead of calling the API")
    args = parser.parse_args()
    rag_answer.CACHE_ONLY = args.cache_only

    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        questions = list(csv.DictReader(f))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run, questions))

    api_calls = sum(1 for _, out in results if not out["cached"] and out["usage"] is not None)
    records = []
    for q, out in results:
        answerable = q["answerable"].strip().lower() == "yes"
        rec = {"qid": q["qid"], "question": q["question"], "answerable": answerable,
               "answer": out["answer"], "abstained": out["abstained"], "citations": out["citations"],
               "invalid_citations": out["invalid_citations"], "source_map": out["source_map"],
               "retrieved": [{"chunk_id": r["chunk_id"], "score": r["score"], "title": r["title"],
                              "heading": r["heading"], "flags": r["flags"]} for r in out["retrieved"]],
               "stop_reason": out["stop_reason"], "cached": out["cached"], "usage": out["usage"],
               "model": MODEL, "retriever": RETRIEVER}
        if answerable and not out["abstained"]:
            (rec["cited_sentences"], rec["cited_incl_list_items"], rec["sentences"],
             rec["uncited_sentences"]) = citation_coverage(out["answer"], out["source_map"])
        if not answerable:
            rec["unanswerable_check"] = judge_unanswerable(out)
        records.append(rec)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # (a) abstention on unanswerable questions
    unans = [r for r in records if not r["answerable"]]
    print(f"\n(a) Unanswerable questions: {sum(r['unanswerable_check']['pass'] for r in unans)}/{len(unans)} pass")
    for r in unans:
        c = r["unanswerable_check"]
        kind = "ABSTAIN sentence" if r["abstained"] else f"says can't answer={c['says_cant_answer']}, invented={c['invented']}"
        print(f"  {r['qid']}  {'PASS' if c['pass'] else 'FAIL'}  {kind}")
        if not r["abstained"]:
            print("       " + r["answer"].replace("\n", "\n       "))

    # (b) false abstentions on answerable questions
    ans = [r for r in records if r["answerable"]]
    false_abs = [r for r in ans if r["abstained"]]
    print(f"\n(b) False abstentions on answerable questions: {len(false_abs)}/{len(ans)}"
          + "".join(f"\n  {r['qid']}  {r['question']}" + ("  (expected page not in crawl)" if r["qid"] in NO_PAGE else "")
                    + "\n       top 5: " + " | ".join(f"{x['title']} > {x['heading']}" for x in r["retrieved"])
                    for r in false_abs))

    # (c) citation coverage
    cov = [r for r in ans if "sentences" in r]
    cited, total = sum(r["cited_sentences"] for r in cov), sum(r["sentences"] for r in cov)
    print(f"\n(c) Citation coverage over {len(cov)} non-abstaining answers: {cited}/{total} sentences "
          f"= {100 * cited / total:.1f}%   (per-answer mean "
          f"{100 * sum(r['cited_sentences'] / r['sentences'] for r in cov) / len(cov):.1f}%)")
    lenient = sum(r["cited_incl_list_items"] for r in cov)
    print(f"    counting bullet items under a cited lead-in (\"symptoms include [1]:\") as cited: "
          f"{lenient}/{total} = {100 * lenient / total:.1f}%")
    print(f"    invalid citation numbers: {sum(len(r['invalid_citations']) for r in records)}")
    for r in cov:
        print(f"  {r['qid']}  {r['cited_sentences']}/{r['sentences']}"
              + "".join(f"\n       uncited: {s}" for s in r["uncited_sentences"]))

    print(f"\nAPI calls this run: {api_calls}  (cached: {sum(1 for _, o in results if o['cached'])}, "
          f"no retrieval: {sum(1 for _, o in results if o['stop_reason'] == 'no_retrieval')})")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
