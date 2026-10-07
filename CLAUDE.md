# MythTrace — Project brief for Claude Code

## What this project is
CSD358 (Information Retrieval) hackathon, Track 1: Retrieval-Augmented Generation and trustworthy answers.
Team of 3. Hard deadline: 36 hours from track announcement. Graded on correct use of IR principles from lectures.
One team member drives Claude Code; teammates test, write data/attack pages, judge relevance, and write the report.

We are building a health Q&A RAG system over a corpus we crawl ourselves, then:
1. Attacking it by injecting fake "poisoned" pages with false health claims modelled on real health myths
   circulating in India (viral home-remedy cures, miracle claims), stuffed with query terms.
2. Defending it with IR techniques (BM25 vs tf-idf, source-quality g(d) in a net score, Jaccard query-copy check,
   near-duplicate detection).
3. Measuring attack success rate and retrieval quality (P@5, P@10) before vs after each defense.

Headline metric: how often the LLM gives the false answer, with vs without defenses.
Framing: this is a Track 1 project about trustworthy RAG answers. The crawler is how we build the corpus,
not the main contribution.

## Corpus
- Main source: MedlinePlus (https://medlineplus.gov) — English health-topic pages. robots.txt allows these;
  seed from its sitemap (https://medlineplus.gov/sitemap.xml) and stay within /healthtopics-style content pages.
- Optional Indian sources, only if they have usable English explainer content: Ministry of Health and Family
  Welfare (mohfw.gov.in, robots.txt allows all) and NCDC (ncdc.mohfw.gov.in, no robots.txt found).
- Do NOT crawl: nhp.gov.in, icmr.gov.in, nhm.gov.in (their robots.txt disallows automated access).
- English only (Hindi content would pull the project toward Track 5).
- Target size: a few hundred to ~2,000 pages. Skip PDFs unless explicitly asked.

## Most important rule: build the IR parts from scratch
IR principles are 30% of the grade, and every team member must explain their component in the demo video.
- DO NOT use: sklearn TfidfVectorizer, rank_bm25, FAISS, Chroma, LangChain retrievers, Elasticsearch, Whoosh,
  or any library that does indexing/ranking for us.
- OK to use: requests, BeautifulSoup, urllib.robotparser, nltk (PorterStemmer and stopword list only), numpy,
  matplotlib, python-dotenv, an LLM API client, Streamlit or a CLI.
- Write clear, readable code with comments that explain the IR concept being implemented
  (e.g. "# lnc.ltc: log tf, no idf, cosine normalisation on documents").
- Prefer simple and correct over clever. Frontend is NOT graded — keep any UI minimal.

## Architecture
```
crawler/   crawl loop, Mercator-style frontier (front queues = priority, back queues = one per host),
           robots.txt, politeness delay (>=1s per host, honour Crawl-delay), URL normalisation,
           depth/URL-length limits (spider traps), "content seen?" hash check,
           near-duplicate detection (shingles + Jaccard), link graph, last-modified dates
attack/    poisoned pages (hand-written by a teammate, based on real Indian health myths) + injection into corpus;
           clearly labelled is_poison=true, committed; generated data gitignored
index/     tokenisation, case folding, stop words, Porter stemming (toggleable), Soundex for query terms,
           chunking pages into section-level chunks with zones (title, heading, body),
           inverted index (dictionary + postings), positional index, skip pointers
ranking/   Boolean query parser (AND/OR/NOT, quoted phrases, process terms by increasing df),
           lnc.ltc tf-idf + cosine, BM25, heap-based top-K, zone weighting, static quality g(d) (+ optional PageRank),
           net score = cosine/BM25 + g(d), defense flags (Jaccard similarity of doc to query),
           champion lists / tiers (optional)
rag/       prompt builder: answer ONLY from retrieved chunks, cite [chunk_id] per claim, abstain if unsupported
eval/      ~25 judged health questions, relevance judgments, P@5/P@10, recall, stemming on/off,
           tf-idf vs BM25, attack success rate per defense configuration, graphs (matplotlib)
data/      small sample only committed; full crawl is gitignored and reproducible via script
```

## Shared interfaces (keep these stable so the parts plug together)
- Chunk record (JSON lines in data/chunks.jsonl):
  `{"chunk_id": str, "doc_id": str, "url": str, "host": str, "title": str, "heading": str, "body": str, "crawled_at": str, "is_poison": bool}`
- Retriever: `search(query: str, k: int = 5, mode: str = "bm25", defenses: bool = True) -> list[dict]`
  each result: `{"chunk_id", "score", "score_breakdown": {...}, "url", "title", "heading", "body", "flags": [...]}`
- RAG: `answer(question: str, **retriever_kwargs) -> {"answer": str, "citations": [chunk_id], "retrieved": [...]}`

## Data ethics (required by the assignment)
- Obey robots.txt for every host; skip disallowed URLs.
- Minimum 1 second delay between requests to the same host; identify with a clear User-Agent
  (e.g. "MythTrace-CSD358-student-project").
- Collect no personal data. Credit every source in the README.
- Poison pages in attack/poison_pages.json are synthetic misinformation for a security experiment, committed so results are reproducible, clearly labelled in the README, and never used or presented as real health advice. Generated poisoned data files stay gitignored.

## Secrets and git
- LLM API key lives in `.env` (gitignored). Never print it or commit it.
- Commit small, working steps with clear messages, and push regularly.
- Commit messages: one short line under 60 characters, starting with a verb (e.g. "add BM25 scorer").

## Must-have before any nice-to-have
Must: crawler w/ politeness + robots, dedup, chunking + zones, text processing, inverted + positional index,
lnc.ltc cosine, BM25, heap top-K, g(d) + net score, RAG with citations, one attack + defenses, evaluation + graphs.
Nice: Soundex, skip pointers, champion lists, tiered index, PageRank, proximity boost, freshness.

## When helping
- Work in small steps; test each piece before moving on.
- Explain what you built in plain English after each step, so the team can explain it in the demo video.
- Show intermediate outputs (postings, weights, scores) when testing — the video must show these.
- Don't silently change shared interfaces; flag it if a change is needed.
- Keep a short log of what was built with AI help in AI_USE.md (one line per session) for the AI-use declaration.
