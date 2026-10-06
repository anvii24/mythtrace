"""URL normalisation for the crawler.

The same page can be written many ways:
    HTTPS://MedlinePlus.gov:443/diabetes.html#symptoms
    https://medlineplus.gov/diabetes.html?utm_source=twitter
    https://medlineplus.gov/genetics/../diabetes.html
All of these are one document. Normalising every URL to a single canonical
string BEFORE it enters the frontier means the "URL seen?" test (a set lookup)
catches these duplicates, so we never fetch or index the same page twice.
This is the URL-level dedup step of the Mercator crawler; content-level dedup
(hashes, shingles) comes later and catches the cases URL rules cannot.
"""
from urllib.parse import unquote_plus, urljoin, urlsplit, urlunsplit

# Default ports are implied by the scheme, so ":80" / ":443" add nothing.
DEFAULT_PORTS = {"http": 80, "https": 443}

# Query parameters that only track where a click came from; they never change
# the page content, so keeping them would create fake "new" URLs.
TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "msclkid", "yclid",
    "mc_cid", "mc_eid", "_ga", "_gl", "igshid", "ref", "ref_src",
}
TRACKING_PREFIXES = ("utm_",)


def _is_tracking(param: str) -> bool:
    p = param.lower()
    return p in TRACKING_PARAMS or p.startswith(TRACKING_PREFIXES)


def _remove_dot_segments(path: str) -> str:
    """Resolve '.' and '..' in a path (RFC 3986 section 5.2.4), e.g. /a/b/../c -> /a/c."""
    out = []
    for seg in path.split("/"):
        if seg == ".":
            continue
        if seg == "..":
            if len(out) > 1:  # never pop the leading "" that represents the root
                out.pop()
            continue
        out.append(seg)
    result = "/".join(out)
    # "/a/b/.." should keep its trailing slash meaning "/a/"; the slash rule below decides anyway
    return result if result.startswith("/") else "/" + result


def normalize_url(url: str, base: str | None = None) -> str | None:
    """Return the canonical form of `url`, or None if it is not a crawlable http(s) URL.

    `base` is the page the link was found on; relative links ("../x.html",
    "/y.html", "z.html") are resolved against it, exactly as a browser would.

    Rules, in order:
      1. resolve relative links against `base`
      2. keep only http/https (drops mailto:, javascript:, tel:, ftp:, ...)
      3. lowercase scheme and host (they are case-insensitive; the path is NOT)
      4. remove default ports (:80 for http, :443 for https)
      5. resolve '.' and '..' path segments
      6. strip the trailing slash on every path except the root "/"
      7. drop tracking query parameters, sort the rest so parameter order
         does not create duplicates
      8. remove the #fragment (it points inside a page, not to a new page)
    """
    url = url.strip()
    if not url:
        return None
    if base:
        url = urljoin(base, url)

    try:
        parts = urlsplit(url)
        port = parts.port  # raises ValueError on garbage like ":99999" or ":abc"
    except ValueError:
        return None

    scheme = parts.scheme.lower()
    if scheme not in DEFAULT_PORTS:
        return None

    host = (parts.hostname or "").rstrip(".")  # .hostname is already lowercased
    if not host:
        return None
    netloc = host
    if port is not None and port != DEFAULT_PORTS[scheme]:
        netloc = f"{host}:{port}"
    # user:password@ is deliberately dropped: we never send credentials

    path = _remove_dot_segments(parts.path or "/")
    if len(path) > 1:
        path = path.rstrip("/") or "/"

    # Work on the raw "key=value" pieces (not decoded ones) so the server sees
    # exactly the encoding the site used, e.g. "?q=en/x" stays "?q=en/x".
    params = [p for p in parts.query.split("&")
              if p and not _is_tracking(unquote_plus(p.split("=", 1)[0]))]
    query = "&".join(sorted(params))

    return urlunsplit((scheme, netloc, path, query, ""))


def get_host(url: str) -> str:
    """Host part of an already-normalised URL (used as the key for back queues and robots)."""
    return urlsplit(url).netloc


if __name__ == "__main__":
    # Demo: each line shows raw input -> normalised output.
    page = "https://medlineplus.gov/genetics/condition/sickle-cell-disease/"
    cases = [
        ("HTTPS://MedlinePlus.GOV:443/Diabetes.html#symptoms", None),
        ("https://medlineplus.gov/diabetes.html?utm_source=twitter&utm_medium=social", None),
        ("http://medlineplus.gov:80/healthtopics.html", None),
        ("https://medlineplus.gov/genetics/../diabetes.html", None),
        ("https://medlineplus.gov/genetics/", None),
        ("https://medlineplus.gov", None),
        ("https://medlineplus.gov/search?q=fever&lang=en&fbclid=XYZ", None),
        ("https://medlineplus.gov/search?lang=en&q=fever", None),
        ("../../gene/hbb/", page),
        ("/ency/article/000527.htm", page),
        ("#references", page),
        ("https://www.MoHFW.gov.in/?q=en/diseases-alerts#main", None),
        ("https://www.mohfw.gov.in:8080/index1.php?lang=1&level=0", None),
        ("mailto:help@example.org", None),
        ("javascript:void(0)", None),
    ]
    for raw, base in cases:
        label = f"{raw}   (on {base})" if base else raw
        print(f"{label}\n    -> {normalize_url(raw, base)}")
