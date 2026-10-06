"""Simulated crawl through the Frontier: no network, a fake clock, a fake link graph.

Run:  python -m crawler.test_frontier

What it shows:
  * the order URLs come out (priority 0 = MedlinePlus health topics first)
  * the gap before each host is contacted again is >= max(1 s, Crawl-delay)
  * spider traps being cut off: depth limit, URL length limit, per-host cap
  * newly discovered high-priority pages jumping ahead of older low-priority ones
"""
from crawler.frontier import Frontier, default_priority
from crawler.urls import get_host

FETCH_TIME = 0.4  # pretend every download takes 0.4 s

# Simulated robots.txt Crawl-delay values (NOT the real ones; real MoHFW sets none).
CRAWL_DELAY = {
    "medlineplus.gov": None,        # no Crawl-delay      -> we use the 1 s minimum
    "www.mohfw.gov.in": 3.0,        # Crawl-delay: 3      -> we wait 3 s
    "health-blog.example": 0.5,     # Crawl-delay: 0.5    -> raised to the 1 s minimum
}

# Seeds, deliberately added low-priority first, so the reordering is visible.
SEEDS = [
    "https://health-blog.example/home",
    "https://medlineplus.gov/ency/article/000313.htm",
    "https://medlineplus.gov/genetics/condition/sickle-cell-disease/",
    "https://www.mohfw.gov.in/",
    "https://medlineplus.gov/diabetes.html",
    "https://medlineplus.gov/asthma.html",
    "https://medlineplus.gov/healthtopics.html",
]

# Fake link graph: links "found" on each page when it is fetched.
LINKS = {
    "https://medlineplus.gov/healthtopics.html": [
        "/heartattack.html",          # new health topic, should jump the queue
        "/malaria.html",              # would exceed MedlinePlus page cap
        "/ency/article/000527.htm",
    ],
    "https://medlineplus.gov/diabetes.html": [
        "asthma.html#symptoms",       # same page as a seed after normalisation -> already seen
        "/diabetes.html?utm_source=x",  # tracking param stripped -> already seen
        "mailto:custserv@nlm.nih.gov",  # not http(s)
        "/diabetes.html?q=" + "x" * 300,  # URL too long
    ],
    # A calendar trap: every month page links to the next month, forever.
    "https://www.mohfw.gov.in/": ["/events?month=1"],
    "https://www.mohfw.gov.in/events?month=1": ["/events?month=2"],
    "https://www.mohfw.gov.in/events?month=2": ["/events?month=3"],
    "https://www.mohfw.gov.in/events?month=3": ["/events?month=4"],
    # A link farm: one page links to many pages on the same host.
    "https://health-blog.example/home": [f"/post/{i}" for i in range(1, 9)],
}


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float):
        self.now += seconds


def simulate(weights=None, seed=0, verbose=True):
    """Run the fake crawl; returns (log, frontier, clock)."""
    say = print if verbose else (lambda *a, **k: None)
    clock = FakeClock()
    frontier = Frontier(
        delay_fn=lambda url: CRAWL_DELAY[get_host(url)],
        weights=weights,
        seed=seed,
        max_depth=3,
        max_url_length=256,
        max_pages_per_host=6,  # small so the cap shows up in a short demo
        clock=clock.time,
        sleep=clock.sleep,
    )
    for url in SEEDS:
        frontier.add(url, depth=0)

    last_end = {}  # host -> time its previous fetch finished
    log = []
    say(f"{'#':>2} {'start':>6} {'waited':>6}  {'P':1} {'d':1}  {'gap/need':>9}  url")
    while (item := frontier.next_url()) is not None:
        url, depth = item
        host = get_host(url)
        start = clock.now
        waited = start - (log[-1]["end"] if log else 0.0)
        need = Frontier.delay_for(CRAWL_DELAY[host])
        gap = start - last_end[host] if host in last_end else None

        clock.now += FETCH_TIME  # "download" the page
        frontier.release(url)
        last_end[host] = clock.now
        log.append({"url": url, "host": host, "priority": default_priority(url),
                    "end": clock.now, "gap": gap, "need": need})

        gap_txt = f"{gap:.1f}/{need:.1f}" if gap is not None else "first"
        say(f"{len(log):>2} {start:6.1f} {waited:6.1f}  {default_priority(url)} {depth}  {gap_txt:>9}  {url}")

        for link in LINKS.get(url, []):
            result = frontier.add(link, depth + 1, base=url)
            shown = link if len(link) < 60 else link[:57] + "..."
            say(f"{'':>30}+ {shown:<45} -> {result}")

    say(f"\nFetched {len(log)} pages in {clock.now:.1f} simulated seconds.")
    say("Rejected:", dict(frontier.rejected))
    return log, frontier, clock


def medline_priorities(log) -> str:
    return " ".join(str(e["priority"]) for e in log if e["host"] == "medlineplus.gov")


def main():
    print("=== Simulated crawl, strict priority (default) ===")
    log, frontier, clock = simulate()

    # ---- checks ----
    for e in log:
        assert e["gap"] is None or e["gap"] >= e["need"] - 1e-9, f"politeness violated: {e}"
    print("OK  every host waited >= max(1 s, Crawl-delay) between fetches")

    fetched = [e["url"] for e in log]
    assert "https://www.mohfw.gov.in/events?month=3" in fetched
    assert "https://www.mohfw.gov.in/events?month=4" not in fetched
    print("OK  calendar trap cut off at max_depth=3 (month=4 would be depth 4)")

    for host in CRAWL_DELAY:
        assert sum(e["host"] == host for e in log) <= 6
    print("OK  no host fetched more than max_pages_per_host=6 pages")

    medline = [e for e in log if e["host"] == "medlineplus.gov"]
    first_p1 = next(i for i, e in enumerate(medline) if e["priority"] == 1)
    assert all(e["priority"] == 0 for e in medline[:first_p1]) and first_p1 >= 3
    print(f"OK  first {first_p1} MedlinePlus fetches were all health topics (priority 0)")
    assert medline[first_p1 - 1]["url"] == "https://medlineplus.gov/heartattack.html"
    print("OK  heartattack.html (found mid-crawl) was fetched before the older priority-1 seeds")

    print("\n=== Priority of each MedlinePlus fetch, in order (0 = health topic) ===")
    print(f"strict             : {medline_priorities(log)}")
    for seed in range(3):
        biased_log, _, _ = simulate(weights=(6, 3, 1), seed=seed, verbose=False)
        print(f"biased 6:3:1 seed={seed}: {medline_priorities(biased_log)}")


if __name__ == "__main__":
    main()
