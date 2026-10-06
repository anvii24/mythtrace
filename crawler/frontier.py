"""Mercator-style URL frontier (Introduction to IR, section 20.2.3).

The frontier holds the URLs we have discovered but not yet fetched, and decides
WHICH URL to fetch next. It must balance two goals:
  * priority   - fetch the most useful pages first (MedlinePlus health topics)
  * politeness - never hit one host more than once per max(1 s, Crawl-delay)

Mercator separates these into two layers:

    add(url, depth) --> URL-seen test + spider-trap checks
          |
          v
    FRONT QUEUES  F0 (highest) | F1 | F2 (lowest)       <- priority
          |
          |  front-queue selector, two modes:
          |   - strict (default): always the highest-priority non-empty queue
          |   - biased (textbook Mercator, weights=(6,3,1)): a random non-empty
          |     queue weighted towards high priority, so low priority is slowed
          |     down but never starved
          |  We default to strict because our crawl has a page cap: with
          |  random picks, low-priority pages can use up the cap before all
          |  health topics are fetched. Starving low priority is what we want here.
          v
    BACK QUEUES   one queue per host, at most num_back_queues hosts at a time,
                  each holding at most max_back_queue_len URLs (default 1) <- politeness
          |
          v
    HEAP of (earliest time host may be contacted, host)
          |
    next_url(): pop the host with the earliest time, wait until that time,
                return the next URL from that host's back queue
    release(url): called after the fetch finishes; the host may be contacted
                  again at  now + max(1 s, Crawl-delay)

Keeping back queues short means most URLs wait in the front queues, where they
are ordered by priority. So a health-topic page discovered late still jumps
ahead of lower-priority pages that were discovered earlier.
"""
import heapq
import itertools
import random
import re
import time
from collections import Counter, deque
from urllib.parse import urlsplit

from crawler.robots import MIN_DELAY
from crawler.urls import get_host, normalize_url

NUM_PRIORITIES = 3
# MedlinePlus health-topic pages live at the site root: /diabetes.html, /healthtopics.html
HEALTH_TOPIC_PATH = re.compile(r"/[a-z0-9][a-z0-9_\-]*\.html")


def default_priority(url: str) -> int:
    """0 = MedlinePlus health topics, 1 = other MedlinePlus / MoHFW pages, 2 = anything else."""
    parts = urlsplit(url)
    if parts.netloc == "medlineplus.gov":
        return 0 if HEALTH_TOPIC_PATH.fullmatch(parts.path) else 1
    if parts.netloc.endswith("mohfw.gov.in"):
        return 1
    return 2


class Frontier:
    def __init__(
        self,
        priority_fn=default_priority,
        delay_fn=None,               # url -> Crawl-delay seconds (e.g. RobotsCache.crawl_delay); None = MIN_DELAY
        max_depth: int = 3,          # spider trap: links more than this many hops from a seed are dropped
        max_url_length: int = 256,   # spider trap: endlessly growing URLs (/a/a/a/..., ?p=...&p=...)
        max_pages_per_host: int = 1500,  # spider trap: no single host can swallow the whole crawl
        num_back_queues: int = 3,    # how many hosts can be "active" at once
        max_back_queue_len: int = 1, # URLs in a back queue are committed; keeping it at 1 means a host's
                                     # next URL is chosen as late as possible, so newly found
                                     # high-priority pages can still jump ahead
        weights=None,                # None = strict priority; e.g. (6, 3, 1) = biased random (F0 ~6x as often as F2)
        seed: int = 0,
        clock=time.monotonic,        # injectable so tests can use a fake clock
        sleep=time.sleep,
    ):
        self.priority_fn = priority_fn
        self.delay_fn = delay_fn
        self.max_depth = max_depth
        self.max_url_length = max_url_length
        self.max_pages_per_host = max_pages_per_host
        self.num_back_queues = num_back_queues
        self.max_back_queue_len = max_back_queue_len
        self.weights = weights
        self.rng = random.Random(seed)
        self.clock = clock
        self.sleep = sleep

        self.front = [deque() for _ in range(NUM_PRIORITIES)]  # each entry: (url, depth)
        self.back: dict[str, deque] = {}       # host -> deque of (url, depth)
        self.heap: list = []                    # (ready_time, tie_breaker, host)
        self._tie = itertools.count()           # keeps heap order stable when times are equal
        self.next_allowed: dict[str, float] = {}  # host -> earliest time we may contact it again
        self.busy: set[str] = set()             # hosts with a URL handed out but not yet released

        self.seen: set[str] = set()             # "URL seen?" test: every URL ever accepted
        self.host_counts = Counter()            # accepted URLs per host (for the per-host cap)
        self.rejected = Counter()               # reason -> count, for reporting

    # ---------- adding URLs ----------

    def add(self, url: str, depth: int = 0, base: str | None = None) -> str:
        """Normalise and enqueue a URL. Returns "added" or the reason it was rejected."""
        norm = normalize_url(url, base)
        if norm is None:
            return self._reject("not http(s)")
        if norm in self.seen:
            return self._reject("already seen")
        if len(norm) > self.max_url_length:
            return self._reject("URL too long")
        if depth > self.max_depth:
            return self._reject("too deep")
        host = get_host(norm)
        if self.host_counts[host] >= self.max_pages_per_host:
            return self._reject("host page cap")

        self.seen.add(norm)
        self.host_counts[host] += 1
        self.front[self.priority_fn(norm)].append((norm, depth))
        return "added"

    def _reject(self, reason: str) -> str:
        self.rejected[reason] += 1
        return reason

    # ---------- front queues -> back queues ----------

    def _pick_front(self, skip: set[int]) -> int | None:
        """Front-queue selector: highest-priority non-empty queue (strict), or a biased random pick."""
        candidates = [i for i in range(NUM_PRIORITIES) if self.front[i] and i not in skip]
        if not candidates:
            return None
        if self.weights is None:
            return candidates[0]
        return self.rng.choices(candidates, weights=[self.weights[i] for i in candidates])[0]

    def _push_host(self, host: str):
        heapq.heappush(self.heap, (self.next_allowed.get(host, float("-inf")), next(self._tie), host))

    def _can_move(self, host: str) -> bool:
        """Is there room in the back queues for one more URL of this host?"""
        if host in self.back:
            return len(self.back[host]) < self.max_back_queue_len
        return len(self.back) < self.num_back_queues

    def _refill(self):
        """Move URLs from front queues into back queues until nothing more fits.

        From the chosen front queue we take the first URL whose host has room,
        not just the head. Otherwise one busy host at the head (e.g. hundreds of
        MedlinePlus URLs) would block every other host behind it in the same
        priority level. URLs of one host still leave in FIFO order.
        """
        blocked: set[int] = set()
        while (q := self._pick_front(blocked)) is not None:
            queue = self.front[q]
            i = next((i for i, (url, _) in enumerate(queue) if self._can_move(get_host(url))), None)
            if i is None:
                blocked.add(q)  # nothing in this priority level can move right now
                continue
            entry = queue[i]
            del queue[i]
            host = get_host(entry[0])
            if host in self.back:
                self.back[host].append(entry)
            else:
                self.back[host] = deque([entry])
                if host not in self.busy:  # a busy host re-enters the heap on release()
                    self._push_host(host)

    # ---------- handing out URLs ----------

    def next_url(self) -> tuple[str, int] | None:
        """Return (url, depth) for the next fetch, sleeping until its host is allowed.

        Returns None when the frontier is empty. Call release(url) after each fetch.
        """
        self._refill()
        if not self.heap:
            if self.busy:
                raise RuntimeError("call release(url) for the previous URL before asking for another")
            return None

        ready_time, _, host = heapq.heappop(self.heap)
        wait = ready_time - self.clock()
        if wait > 0:
            self.sleep(wait)  # politeness: the earliest host is not ready yet

        url, depth = self.back[host].popleft()
        if not self.back[host]:
            del self.back[host]  # frees a back-queue slot for another host
        self.busy.add(host)
        return url, depth

    def release(self, url: str):
        """Mark the fetch of `url` as finished; its host may be contacted again after the delay."""
        host = get_host(url)
        self.busy.discard(host)
        crawl_delay = self.delay_fn(url) if self.delay_fn else None
        self.next_allowed[host] = self.clock() + self.delay_for(crawl_delay)
        if host in self.back:
            self._push_host(host)

    @staticmethod
    def delay_for(crawl_delay: float | None) -> float:
        """Politeness delay: the site's Crawl-delay, but never less than MIN_DELAY (1 s)."""
        return max(MIN_DELAY, float(crawl_delay or 0))

    def __len__(self) -> int:
        return sum(len(q) for q in self.front) + sum(len(q) for q in self.back.values())
