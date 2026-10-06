"""robots.txt handling: fetch once per host, cache, save a dated copy, answer "may I fetch this URL?".

robots.txt is the Robots Exclusion Protocol: a site lists which paths crawlers
may not visit (Disallow), how long to wait between requests (Crawl-delay), and
where its sitemaps are. A polite crawler checks it before every fetch.

Design choices:
  * One robots.txt fetch per host, cached in memory: the crawler asks
    can_fetch() for every URL, and re-downloading robots.txt each time would
    itself be impolite.
  * Parsing uses urllib.robotparser from the standard library, but we download
    the file ourselves with `requests` so we send our own User-Agent, set a
    timeout, and keep the raw text to save as evidence.
  * Every downloaded robots.txt is saved to data/robots/<host>_<timestamp>.txt,
    so we can show exactly what rules were in force when we crawled.
  * Strict policy: if robots.txt cannot be fetched (network error, timeout,
    any non-200 status including 404), we do NOT crawl that host. The usual
    convention treats 404 as "allow everything"; we choose the safer option.
"""
import os
from datetime import datetime, timezone
from urllib.robotparser import RobotFileParser

import requests

from crawler.urls import get_host, normalize_url

USER_AGENT = "TrustRAG-CSD358-student-project"
MIN_DELAY = 1.0  # seconds between requests to one host, even if robots.txt asks for less
ROBOTS_DIR = os.path.join("data", "robots")


class RobotsCache:
    def __init__(self, user_agent: str = USER_AGENT, save_dir: str = ROBOTS_DIR, timeout: float = 10.0):
        self.user_agent = user_agent
        self.save_dir = save_dir
        self.timeout = timeout
        # host -> {"parser": RobotFileParser | None, "status": str, "saved_to": str | None}
        # parser is None when the host is refused; refusals are cached too, so we ask only once.
        self._cache: dict[str, dict] = {}

    def _fetch(self, scheme: str, host: str) -> dict:
        robots_url = f"{scheme}://{host}/robots.txt"
        try:
            resp = requests.get(robots_url, headers={"User-Agent": self.user_agent}, timeout=self.timeout)
        except requests.RequestException as e:
            return {"parser": None, "status": f"refused: fetch failed ({type(e).__name__})", "saved_to": None}

        if resp.status_code != 200:
            return {"parser": None, "status": f"refused: HTTP {resp.status_code}", "saved_to": None}

        text = resp.text
        saved_to = self._save(host, text)

        parser = RobotFileParser(robots_url)
        parser.parse(text.splitlines())
        parser.modified()  # mark as "checked"; can_fetch() returns False for an unchecked parser
        return {"parser": parser, "status": "ok", "saved_to": saved_to}

    def _save(self, host: str, text: str) -> str:
        os.makedirs(self.save_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")  # no ':' so it is a valid Windows filename
        path = os.path.join(self.save_dir, f"{host.replace(':', '_')}_{stamp}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        return path

    def _entry(self, url: str) -> dict:
        host = get_host(url)
        if host not in self._cache:
            scheme = url.split("://", 1)[0]
            self._cache[host] = self._fetch(scheme, host)
        return self._cache[host]

    def status(self, url: str) -> str:
        """'ok', or the reason this URL's host was refused."""
        return self._entry(normalize_url(url) or url)["status"]

    def can_fetch(self, url: str) -> bool:
        """True only if robots.txt was fetched for this host AND it allows our User-Agent on this URL."""
        url = normalize_url(url)
        if url is None:
            return False
        parser = self._entry(url)["parser"]
        if parser is None:
            return False
        # normalize_url strips trailing slashes, but rules like "Disallow: /cgi/" are
        # prefix matches that "/cgi" would slip past. Check the slash form too and
        # refuse if EITHER form is disallowed.
        with_slash = url if url.endswith("/") or "?" in url else url + "/"
        return parser.can_fetch(self.user_agent, url) and parser.can_fetch(self.user_agent, with_slash)

    def crawl_delay(self, url: str) -> float:
        """Seconds to wait between requests to this URL's host: robots.txt Crawl-delay, but never below MIN_DELAY."""
        parser = self._entry(normalize_url(url) or url)["parser"]
        delay = parser.crawl_delay(self.user_agent) if parser else None
        return max(MIN_DELAY, float(delay or 0))

    def sitemaps(self, url: str) -> list[str]:
        """Sitemap URLs declared in robots.txt (useful seeds for the frontier)."""
        parser = self._entry(normalize_url(url) or url)["parser"]
        return (parser.site_maps() or []) if parser else []


if __name__ == "__main__":
    robots = RobotsCache()
    tests = [
        # MedlinePlus: content pages should be allowed
        "https://medlineplus.gov/diabetes.html",
        "https://medlineplus.gov/ency/article/000313.htm",
        "https://MedlinePlus.gov/genetics/condition/sickle-cell-disease/#resources",
        # MedlinePlus: paths its robots.txt disallows
        "https://medlineplus.gov/cgi/",
        "https://medlineplus.gov/feeds/topics/",
        "https://medlineplus.gov/xml/vocabulary/terms.xml",
        # MoHFW
        "https://www.mohfw.gov.in/",
        "https://www.mohfw.gov.in/?q=en/diseases-alerts",
        # Hosts CLAUDE.md says not to crawl, and NCDC (reported to have no robots.txt)
        "https://nhp.gov.in/disease-a-z",
        "https://ncdc.mohfw.gov.in/",
    ]
    for url in tests:
        print(f"{url}\n    allowed={robots.can_fetch(url)}  robots={robots.status(url)}")

    print("\nPer-host cache:")
    for host, e in robots._cache.items():
        delay = e["parser"].crawl_delay(USER_AGENT) if e["parser"] else None
        print(f"  {host:25s} {e['status']:35s} crawl-delay={delay}  saved={e['saved_to']}")
    print("\nSitemaps listed by medlineplus.gov:", robots.sitemaps("https://medlineplus.gov/"))
    print("Delay we will use for medlineplus.gov:", robots.crawl_delay("https://medlineplus.gov/"), "s")
