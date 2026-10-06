# trustrag

## Overview

## Setup

## How to run

## Data sources

All crawling obeys each host's robots.txt (a copy of every robots.txt we fetched is saved in `data/robots/`),
waits at least 1 second between requests to the same host (longer if robots.txt sets a Crawl-delay), and
identifies itself with the User-Agent `TrustRAG-CSD358-student-project`. No personal data is collected.

| Source | What we use | robots.txt | Notes |
|---|---|---|---|
| [MedlinePlus](https://medlineplus.gov) (U.S. National Library of Medicine) | English health-topic pages, medical encyclopedia articles, lab-test pages | Allows these paths | Seeded from https://medlineplus.gov/sitemap.xml plus the health-topics index. This is the main corpus. |
| [Ministry of Health and Family Welfare, India](https://www.mohfw.gov.in) | English explainer pages, where available | `Allow: /` | The site renders its pages with JavaScript, so a plain HTML crawler gets almost no text or links from it. Its sitemap.xml lists meity.gov.in pages, which are out of scope. Expect few or no pages from this source. |

Not crawled: nhp.gov.in, icmr.gov.in and nhm.gov.in, because their robots.txt disallows automated access.
English pages only. PDFs and images are skipped.

Reproduce the corpus (the full crawl is gitignored):

```
python -m crawler.crawl --sources medlineplus mohfw --max-pages 1000 --clean
```

Outputs: `data/raw/<doc_id>.json` (one per page) plus the gzipped HTML, `data/links.json` (link graph),
`data/crawl_log.jsonl` (one JSON line per URL), `data/crawl.log` (text log), and `data/crawl_summary.json`.

Poisoned pages used in the attack experiments are synthetic, are clearly labelled `is_poison=true`,
are stored locally only, and are never published.

## What works

## What's planned

## Team

## AI use
