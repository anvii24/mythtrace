# TrustRAG: trustworthy health answers under a poisoning attack

TrustRAG is our CSD358 (Information Retrieval) hackathon project for **Track 1: Retrieval-Augmented
Generation and trustworthy answers**. It answers health questions only from passages retrieved from a
corpus we crawled ourselves (1,000 MedlinePlus pages, 2,762 chunks), and cites a source for every claim.
We attack it by injecting 16 synthetic poison pages built on real health myths that circulate in India and
stuffed with query terms. We then defend it with IR techniques written from scratch: BM25 and lnc.ltc
tf-idf, a PageRank-based source-quality score g(d) in the net score, and Jaccard query-copy /
keyword-stuffing checks. Without defenses a poison chunk reaches the LLM's top 5 for **16/16** target
questions. With all defenses that drops to **5/16** (0/8 external, 5/8 insider). **0 of 96** hand-labelled
LLM answers endorse a false claim, and the number of clean top-5 chunks pushed out by poison falls from
**2.31 to 0.94** per question.

> [!WARNING]
> The pages in [`attack/poison_pages.json`](attack/poison_pages.json) are **synthetic health
> misinformation**, written on purpose for a security experiment (poisoning a RAG system). Their claims
> are false. **Do not use them as health advice.**

## Quick start (Windows PowerShell)

Tested with Python 3.13 (3.10 or newer is needed). Run everything from the repo root.

```powershell
git clone https://github.com/anvii24/trustrag.git
cd trustrag
python -m venv .venv
.venv\Scripts\Activate.ps1                # prompt now starts with (.venv)
pip install -r requirements.txt
python -m nltk.downloader stopwords       # stop-word list (the Porter stemmer needs no download)
```

If activation fails with "running scripts is disabled on this system", run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.

Add your Anthropic API key. `.env` is gitignored: never commit it.

```powershell
copy .env.example .env
notepad .env                              # set the line to: LLM_API_KEY=sk-ant-...
python rag\test_llm.py                    # optional smoke test: one short API request
```

**The corpus is not in the repo.** Crawled pages, chunks, indexes and the LLM response cache are all
gitignored, so you must rebuild the corpus by crawling. The crawl takes **about 25-30 minutes** (1,000
pages at no more than 1 request per second per host; our run took 27 minutes). The fastest path to the
demo UI:

```powershell
python -m crawler.crawl --sources medlineplus mohfw --max-pages 1000 --clean   # ~25-30 min
python -m crawler.reparse                 # same cleaning as our corpus (no network)
python -m index.chunker
python -m index.build_index --no-demo
python -m attack.inject --no-report
streamlit run app.py                      # opens http://localhost:8501
```

MedlinePlus changes over time, so a new crawl may differ slightly from ours, and so may the numbers you
get from it. The results in [`eval/results/`](eval/results/) and [`eval/figures/`](eval/figures/) are
from our crawl (summary in [`data/crawl_summary.json`](data/crawl_summary.json)).

## Full pipeline

Run each step from the repo root with the venv active. Steps 1-6 build the clean and poisoned corpora,
and the later steps use them. Everything generated under `data/` is gitignored, except
`data/crawl_summary.json` and `data/robots/`.

| # | Step | Command | Produces |
|---|---|---|---|
| 1 | Crawl MedlinePlus (+ MoHFW) | `python -m crawler.crawl --sources medlineplus mohfw --max-pages 1000 --clean` | `data/raw/<doc_id>.json` + `.html.gz`, `data/links.json`, `data/crawl_log.jsonl`, `data/crawl.log`, `data/crawl_summary.json`, `data/robots/` |
| 2 | Rebuild / clean pages from saved HTML (no network) | `python -m crawler.reparse` | rewrites `data/raw/<doc_id>.json`, adds `reparse` to `data/crawl_summary.json` |
| 3 | Chunk pages into sections with zones | `python -m index.chunker` | `data/chunks.jsonl` |
| 4 | Build positional inverted indexes | `python -m index.build_index` | `data/index/index_stemmed.json`, `data/index/index_unstemmed.json` |
| 5 | Show PageRank and quality scores g(d) | `python -m ranking.quality` | nothing (prints only; g(d) is computed from `data/links.json` at search time) |
| 6 | Inject the poison pages | `python -m attack.inject` | `data/chunks_poisoned.jsonl`, `data/links_poisoned.json`, `data/index/index_poisoned_*.json`, `data/poison_manifest.json` |
| 7 | Search (CLI) | `python -m ranking.search "symptoms of type 2 diabetes"` | nothing (prints top-K + score breakdown) |
| 8 | Answer with citations (CLI, 1 API call) | `python -m rag.answer "What are the symptoms of type 2 diabetes?"` | an entry in `data/llm_cache/` |
| 9 | Attack experiment (API calls) | `python -m eval.attack_experiment` | `eval/results/attack_results.csv`, `eval/results/attack_answers.jsonl` |
| 9b | Same attack answered by Haiku 4.5 (API calls) | `python -m eval.model_compare` | `eval/results/attack_answers_haiku.jsonl` |
| 10 | Endorsement rates from the hand labels | `python -m eval.label_rates` | nothing (prints tables from `eval/results/answers_to_label*.csv`) |
| 11 | Retrieval quality on the clean corpus (P@5, P@10, Hit@5, MRR) | `python -m eval.retrieval_eval` | `eval/results/qrels_pages*.json`, `eval/results/retrieval_results*.csv` |
| 12 | Clean-corpus answers: abstention + citation coverage (API calls) | `python -m eval.clean_answers` | `eval/results/clean_answers.jsonl` |
| 13 | Report figures (from saved results, no API calls) | `python -m eval.make_figures` | `eval/figures/fig1-6_*.png` |
| 14 | Demo app (Streamlit) | `streamlit run app.py` | nothing (opens in the browser; LLM answers go to `data/llm_cache/`) |

Notes:

1. **Crawl.** `--max-pages 20` gives a quick test of steps 1-5 and 7. Ctrl+C stops cleanly and keeps
   what was saved. A crawl overwrites the committed `data/crawl_summary.json` and adds files to
   `data/robots/`.
   `python -m crawler.test_frontier` simulates the frontier offline (priority order, politeness gaps,
   spider-trap limits).
2. **Reparse** re-runs the extraction and the duplicate checks on the saved HTML.
4. **Index.** `--no-demo` skips the printed sample of postings. `--corpus poisoned` rebuilds only the
   poisoned indexes (step 6 already does this).
6. **Inject** reads `attack/poison_pages.json` into a separate poisoned corpus and checks (by SHA-1) that
   the clean files are untouched. Without `--no-report` it prints where each poison chunk ranks for its
   own question. It needs the full crawl: each insider poison page is placed inside a real MedlinePlus
   page (dengue, highbloodpressure, tuberculosis, malaria, diabetestype2, jaundice,
   covid19coronavirusdisease2019, kidneystones), and if one of them was not crawled it stops with
   "target_url ... is not in the clean corpus". A small test crawl (`--max-pages 20`) therefore can't be
   injected.
7. **Search** options: `--mode bm25|tfidf`, `--k 5`, `--defenses quality,jaccard` (default: both),
   `--no-defenses`, `--alpha 0.3`, `--penalty`, `--no-stem`, `--corpus clean|poisoned`. To see the attack:
   `python -m ranking.search "treatment for dengue fever" --corpus poisoned --no-defenses`.
8. **Answer** takes the same `--mode`, `--k`, `--defenses`, `--no-defenses` and `--corpus` options, plus
   `--no-cache` to force a new API call.
9. **Attack experiment.** About 128 answers if nothing is cached. `--no-llm` runs only the retrieval part
   (no API calls). `--cache-only` re-scores cached answers and stops with an error rather than call the
   API. It overwrites the result files. `eval.model_compare` (9b) runs bm25 with no defenses, 2 samples
   per page = 32 Haiku answers, on the same retrieved chunks.
10. **Label rates** merges the relabel files (`answers_to_relabel.csv`, then `answers_to_relabel_2.csv`)
    and prints every replacement. For Haiku: `python -m eval.label_rates --path eval/results/answers_to_label_haiku.csv`.
    The label files were made by `python -m eval.export_labels`, which refuses to overwrite a labelled file
    unless you pass `--force`, and `--force` erases the labels.
11. **Retrieval eval** makes no API calls. It runs the main run and the Q10-substitute extra run
    (`--run main|q10_substitute` picks one).
12. **Clean answers:** 25 questions, bm25 + all defenses. `--cache-only` re-scores the cached answers.
14. **Demo app.** Needs steps 1-6. Tabs: **Investigate** (answer + citations, results table, query-term
    highlighting, score breakdown, query trace, defenses off/on comparison), **Case files** (each poison
    page's rank with defenses off and on, and which defense flagged it) and **Lab results** (the figures).
    It only calls the existing `search()`, `answer()` and index code. A new question makes one API call.
    Turn off "Generate LLM answer" in the sidebar to search without the API.

## Project structure

```
crawler/      polite crawler: Mercator frontier, robots.txt, URL normalisation, SHA-1 + shingle dedup, link graph
index/        tokenise, stop words, Porter stemming, section chunks with zones, positional inverted index
ranking/      lnc.ltc tf-idf, BM25, heap top-K, PageRank + g(d), net score, Jaccard / stuffing defenses, search()
rag/          answer(): prompt from the top-5 chunks, [n] citations, abstention, on-disk LLM cache
attack/       poison_pages.json (16 synthetic pages) and inject.py (builds the separate poisoned corpus)
eval/         questions, attack + retrieval experiments, hand labels, results/ (CSV, JSONL) and figures/
data/         generated corpus and indexes (gitignored), plus crawl_summary.json and saved robots.txt files
docs/robots/  screenshots of the robots.txt files we checked
app.py        Streamlit demo app (display only)
AI_USE.md     log of what was built with AI help
```

## Results summary

All numbers come from the files in [`eval/results/`](eval/results/). 16 poison pages: 8 **external**
(a fake outside site) and 8 **insider** (a section slipped into a real MedlinePlus page). Defense thresholds
were frozen with P01-P02 as development pages and P03-P16 held out. "all" = quality g(d) (alpha 0.3) +
Jaccard / stuffing checks.

**Retrieval-level attack success** (poison chunk in the top 5 for its target question) and **crowding out**
(clean top-5 chunks pushed out by the injection, mean per question):

| Configuration | Attack success: all | External | Insider | Crowding out |
|---|---|---|---|---|
| tf-idf, no defenses | 16/16 | 8/8 | 8/8 | 2.44 |
| BM25, no defenses | 16/16 | 8/8 | 8/8 | 2.31 |
| BM25 + quality g(d) | 16/16 | 8/8 | 8/8 | 1.75 |
| BM25 + quality g(d), alpha 1.0 | 8/16 | 0/8 | 8/8 | 2.19 |
| BM25 + Jaccard / stuffing | 5/16 | 0/8 | 5/8 | 1.44 |
| **BM25 + all** | **5/16** | **0/8** | **5/8** | **0.94** |
| tf-idf + all | 5/16 | 0/8 | 5/8 | 1.12 |

**Hand labels of the LLM answers** (2 samples per page per configuration):

| Model, configuration | Endorses | Refutes | Mentions neutrally | Ignores |
|---|---|---|---|---|
| Sonnet 5.5, BM25 no defenses | 0/32 | 32/32 | 0/32 | 0/32 |
| Sonnet 5.5, BM25 + all | 0/32 | 12/32 | 2/32 | 18/32 |
| Haiku 4.5, BM25 no defenses | 0/32 | 24/32 | 0/32 | 8/32 |
| **Total** | **0/96** | 68/96 | 2/96 | 26/96 |

With all defenses every external-page answer "ignores" the claim, because the poison no longer reaches
the model.

**Retrieval quality on the clean corpus** (main run, 19 questions; best possible P@5 = 0.86 and
P@10 = 0.61, see Known limitations):

| Configuration | P@5 | P@10 | Hit@5 | MRR |
|---|---|---|---|---|
| BM25, stemmed | 0.36 | 0.35 | 0.74 | 0.73 |
| BM25, unstemmed | 0.35 | 0.31 | 0.79 | 0.68 |
| tf-idf (lnc.ltc), stemmed | 0.42 | 0.31 | 0.89 | 0.61 |
| tf-idf (lnc.ltc), unstemmed | 0.36 | 0.32 | 0.89 | 0.59 |
| BM25, stemmed + all defenses | 0.40 | 0.37 | 0.79 | 0.81 |

The defenses built against poisoning don't hurt normal searches: on the clean corpus they slightly improve
every metric.

Figures (captions in [`eval/figures/README.md`](eval/figures/README.md)):
[attack success](eval/figures/fig1_attack_retrieval.png) ·
[alpha sweep](eval/figures/fig2_alpha_sweep.png) ·
[crowding out](eval/figures/fig3_crowding_out.png) ·
[answer labels](eval/figures/fig4_answer_labels.png) ·
[retrieval quality](eval/figures/fig5_retrieval_quality.png) ·
[stemming and tf-idf vs BM25](eval/figures/fig6_stem_and_model.png)

## Data sources and ethics

| Source | What we use | Result |
|---|---|---|
| [MedlinePlus](https://medlineplus.gov) (U.S. National Library of Medicine) | English health-topic pages, seeded from https://medlineplus.gov/sitemap.xml and the health-topics index | 1,000 pages saved from 1,075 fetches. This is the whole corpus. |
| [Ministry of Health and Family Welfare, India](https://www.mohfw.gov.in) | English explainer pages | Tried, but **0 pages**: the site renders its pages with JavaScript, so a plain HTML crawler gets almost no text or links. |

- **robots.txt is obeyed** for every host and checked before each URL. Every robots.txt we fetched is saved
  in [`data/robots/`](data/robots/), with screenshots in [`docs/robots/`](docs/robots/).
- **Politeness:** at least 1 second between requests to the same host (longer if robots.txt sets a
  Crawl-delay). The User-Agent is `TrustRAG-CSD358-student-project`.
- **Not crawled:** nhp.gov.in, icmr.gov.in and nhm.gov.in, because their robots.txt disallows automated
  access.
- English pages only. PDFs and images are skipped. **No personal data is collected.**
- The poison pages are synthetic misinformation, committed so the experiment is reproducible. Every
  injected chunk is labelled `is_poison=true`, they live only in a separate poisoned corpus, and the
  generated poisoned files are gitignored.

## Evaluation notes

- **Page-level relevance, judged before searching.** [`eval/questions.csv`](eval/questions.csv) has 25
  questions: Q01-Q20 answerable from MedlinePlus, Q21-Q25 out of scope. Before any search was run, a
  teammate wrote down the MedlinePlus topic page(s) that should answer each answerable question. A
  retrieved chunk counts as relevant if it comes from one of those pages. Two topic names needed an alias
  for the same page (Influenza -> "Flu", Nutrition during pregnancy -> "Pregnancy and Nutrition"). The
  mapping is saved in `eval/results/qrels_pages.json`.
- **Q10 is skipped in the main run** because its page (Gestational Diabetes) was not crawled. A separate,
  clearly labelled extra run (`*_q10_substitute.*`) accepts "Diabetes and Pregnancy" as Q10's page. That
  choice was made after the main run, so the main run is the headline.
- **Answer labels were done by hand.** Teammates read each answer and labelled it `endorses`, `refutes`,
  `mentions_neutrally` or `ignores`, following a written guide that says not to look at the
  configuration column while judging. Answers whose text changed after
  a code fix were relabelled (`answers_to_relabel*.csv`). `eval/label_rates.py` merges the relabels in.
- **Reproducibility relies on the LLM cache.** Claude Sonnet 5.5 rejects `temperature=0`, so answers can't
  be made deterministic. Every response is cached in `data/llm_cache/`, keyed by the exact request, and the
  experiments re-score from that cache. The cache is gitignored, so a rerun with a fresh cache makes new API
  calls and may get differently worded answers. The labelled answer texts are kept in `eval/results/`.

## Known limitations

- **Subtle insider attacks get through.** 5 of 8 insider pages still reach the top 5 with all defenses.
  They sit on medlineplus.gov, so g(d) trusts them, and lightly stuffed sections stay under the Jaccard /
  stuffing thresholds.
- **Cross-question poison leaks.** With all defenses, a mean of 0.94 poison chunks from *any* poison page
  are in the top 5. A poison page that isn't flagged for one question can still compete on normal relevance
  for another.
- **Stemming mismatch.** Porter maps "treated" to "treat" but leaves "treatment" and "treatments" alone, so
  "How is X treated?" misses sections headed "What are the treatments for X?". There is no query expansion.
- **Q04 retrieval failure.** "What are the warning signs of a heart attack?" fails with BM25 (first relevant
  chunk at rank 8, Hit@5 = 0): the Heart Attack page says "symptoms" and never "warning" or "signs", so
  chunks from other pages that do (e.g. suicide warning signs) rank higher.
- **P@k ceiling.** Some expected pages have fewer than k chunks, so a perfect ranking would only reach
  P@5 = 0.86 and P@10 = 0.61.
- **Page-level relevance is coarse.** A chunk from a closely related page (e.g. "HDL: The Good Cholesterol"
  for the LDL vs HDL question) counts as not relevant, and any chunk from an expected page counts as
  relevant even if that section doesn't answer the question.

## Team

- Name — role
- Name — role
- Name — role

## AI use

Much of the code was written with Claude Code (Anthropic), with team members reviewing, testing and
explaining each component. [`AI_USE.md`](AI_USE.md) logs what was built with AI help, one line per
session, for the course's AI-use declaration. The IR methods (indexing, tf-idf, BM25, PageRank, defenses)
follow the lectures and are implemented without indexing or ranking libraries.
