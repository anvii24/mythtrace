# MythTrace

**Health answers you can trust, even when someone plants myths in the corpus.**

MythTrace answers health questions using only pages from MedlinePlus, and cites a source for every claim.
We attack it by planting 16 fake pages based on real health myths. It catches them with IR defenses we
wrote from scratch: BM25, a source-quality score from PageRank, and checks for query copying and keyword stuffing.

> [!WARNING]
> The pages in [`attack/poison_pages.json`](attack/poison_pages.json) are fake health claims made for a
> security experiment. They are false. **They are not health advice.**

## Setup (Windows PowerShell)

Tested with Python 3.13. Run everything from the repo root.

```powershell
git clone https://github.com/anvii24/mythtrace.git   # get the code
cd mythtrace
python -m venv .venv                                  # create a virtual environment
.venv\Scripts\Activate.ps1                            # activate it
pip install -r requirements.txt                       # install packages
python -m nltk.downloader stopwords                   # download the stop-word list
copy .env.example .env                                # create .env
notepad .env                                          # set: LLM_API_KEY=sk-ant-...  (your Anthropic key)
```

If activation fails with "running scripts is disabled", run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## How to run

### Build the corpus and index

The corpus is not in the repo, so you build it yourself.

1. `python -m crawler.crawl --sources medlineplus mohfw --max-pages 1000 --clean` (crawl, about 25-30 min)
2. `python -m crawler.reparse` (clean the saved pages, no network)
3. `python -m index.chunker` (split pages into chunks)
4. `python -m index.build_index --no-demo` (build the indexes)
5. `python -m ranking.quality` (print PageRank and quality scores)
6. `python -m attack.inject --no-report` (inject the poison pages into a separate corpus)

A new crawl may differ slightly from ours, so your numbers may too.

### Use it

```powershell
streamlit run app.py                                              # demo app at http://localhost:8501
python -m ranking.search "symptoms of type 2 diabetes"            # search only, no API call
python -m rag.answer "What are the symptoms of type 2 diabetes?"  # answer with citations (1 API call)
```

- Attack experiment (uses the API): `python -m eval.attack_experiment`
- Retrieval evaluation (no API calls): `python -m eval.retrieval_eval`
- Figures (from saved results): `python -m eval.make_figures`

## Data sources

- Pages come from [MedlinePlus](https://medlineplus.gov) (U.S. National Library of Medicine): 1,000 pages.
- We obey robots.txt and wait at least 1 s between requests to a host. Evidence is in [`docs/robots/`](docs/robots/).
- No personal data is collected.
- MoHFW (mohfw.gov.in) returned no pages, because it is a JavaScript site.

## What works

- A polite crawler with robots.txt, per-host delays, URL cleanup and duplicate detection.
- Our own positional inverted index with stop words and optional Porter stemming.
- Ranking with BM25 and lnc.ltc tf-idf, plus a PageRank-based quality score.
- Defenses that flag pages copying the query or stuffing keywords.
- Answers built only from the top 5 chunks, with a citation for each claim, and abstention when unsupported.
- An attack experiment and a retrieval evaluation (P@5, P@10, Hit@5, MRR).
- A Streamlit demo app that compares defenses off and on.

## What's planned

- Boost results where query words appear as a phrase or close together.
- Query expansion (e.g. "treated" should match "treatment").
- A better defense against insider attacks (fake sections inside trusted pages).
- A larger evaluation with more questions and judges.

## Results

- Without defenses, a poison chunk reaches the top 5 for **16/16** questions. With all defenses: **5/16** (0 of 8 outside pages, 5 of 8 insider pages).
- **0 of 96** hand-labelled answers endorse a false claim.
- Clean top-5 chunks pushed out by poison drop from **2.31 to 0.94** per question.

Full details are in the report and [`eval/figures/`](eval/figures/).

## Team

| Name | Role |
|---|---|
| Anvi Gupta | System design and build |
| Antara Shyam | Attack design and answer labelling |
| Mihir MKC | Data sourcing and evaluation design |

All three chose the topic, tested the system, ran the experiments and shaped the demo app together.
