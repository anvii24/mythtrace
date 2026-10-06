# AI Use Log

- 2026-10-06: Claude Code (Claude Opus 5.5) set up the project skeleton: folders, requirements.txt, .gitignore, .env.example, README headings, this log, and the .venv.
- 2026-10-06: Claude Code wrote rag/test_llm.py, a smoke test that loads LLM_API_KEY from .env and sends one prompt to the Claude API.
- 2026-10-06: Claude Code wrote crawler/urls.py (URL normalisation) and crawler/robots.py (robots.txt fetch/cache/save, strict refuse-on-failure policy).
- 2026-10-06: Claude Code wrote crawler/frontier.py (Mercator frontier: priority front queues, per-host back queues, delay heap, spider-trap limits) and crawler/test_frontier.py (simulated crawl on a fake clock).
- 2026-10-06: Claude Code wrote crawler/crawl.py (crawl loop, BeautifulSoup extraction with link-density boilerplate filter, SHA-1 content-seen check, 4-word shingles + Jaccard near-duplicate detection, link graph) and ran a 20-page MedlinePlus test crawl.
- 2026-10-06: Claude Code added a command-line entry point to crawler/crawl.py (--sources, --max-pages, --clean, logging to data/crawl.log, progress lines with ETA, summary in data/crawl_summary.json, clean stop on Ctrl+C) and wrote the README Data sources section. The full 1000-page crawl is run by a team member, not by Claude Code.
- 2026-10-06: Claude Code wrote index/chunker.py (one chunk per page section, zones title/heading/body, writes data/chunks.jsonl) and index/text.py (preprocess(): tokenise keeping numbers, lowercase, nltk stop words, optional Porter stemming).
- 2026-10-06: Claude Code added short-chunk merging to index/chunker.py (sections under 25 words join the next section on the page, or the previous one if last).
- 2026-10-06: Claude Code wrote index/build_index.py (positional inverted index per zone over chunks.jsonl, df + sorted postings, chunk lengths and N, stemmed + unstemmed builds in data/index/, positional phrase matching with postings merge).
