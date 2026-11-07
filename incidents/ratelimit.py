"""Per-client rate limiting, and a short shared cache for /api/incidents/.

The app runs one gunicorn process with 4 threads, and the 5-minute update
cron posts into that same process. Without a per-client limit, one client
can tie up every thread (and delay the update tick). Fly's
``[http_service.concurrency]`` only caps concurrency globally.

Counters are fixed windows in the ``ratelimit`` cache alias. That is a
LocMemCache, which is correct only because there is a single process; if
the app ever runs more than one worker or machine, point the alias at a
shared cache (Redis, the database cache) so the limits stay per client.
"""

import ipaddress
import logging
import math
import time
from dataclasses import dataclass
from functools import wraps

from django.core.cache import cache, caches
from django.http import HttpResponse

logger = logging.getLogger(__name__)

CACHE_ALIAS = "ratelimit"


@dataclass(frozen=True)
class Limit:
    name: str
    requests: int
    period: int  # seconds


# Streams the whole incidents table: the most expensive request we serve.
TSV = Limit("tsv", 5, 60 * 60)
# Password guessing. Loading the form (GET) is an ordinary page view.
ADMIN_LOGIN = Limit("admin-login", 10, 15 * 60)
# Failed FUNCTIONS_SECRET_KEY checks. Successful calls don't count.
FUNCTIONS_KEY_FAILURES = Limit("functions-key", 10, 60 * 60)
# Machine clients: the iOS app polls /api/incidents/, and mobile carrier
# NAT puts many phones behind one IPv4 address, so keep this generous.
API = Limit("api", 120, 60)
# Everything else: the pages, /stats/ and any unknown path.
PAGES = Limit("pages", 60, 60)

API_PATHS = frozenset({"/api/incidents/", "/api/stations/", "/problems.txt"})
KEY_PROTECTED_PREFIXES = ("/functions/", "/api/training-data/")


def _parse_ip(value):
    try:
        return ipaddress.ip_address((value or "").strip())
    except ValueError:
        return None


def client_key(request):
    """The identity a client is limited by.

    Fly's proxy is the TCP peer for every public request, so the real
    client is the ``Fly-Client-IP`` header, which the proxy sets itself (a
    client can't supply its own). ``REMOTE_ADDR`` is the fallback for
    anything that didn't come through the proxy. IPv6 clients are grouped
    by /64, because one host usually controls a whole /64.
    """
    addr = _parse_ip(request.META.get("HTTP_FLY_CLIENT_IP")) or _parse_ip(
        request.META.get("REMOTE_ADDR")
    )
    if addr is None:
        return "unknown"
    if addr.version == 6:
        if addr.ipv4_mapped:
            return str(addr.ipv4_mapped)
        return str(ipaddress.IPv6Network((addr, 64), strict=False))
    return str(addr)


def is_in_container(request):
    """A direct loopback connection, i.e. the in-container cron.

    The cron POSTs to http://127.0.0.1:8080/ without going through Fly's
    proxy, so it has no Fly-Client-IP header. Anything the proxy forwarded
    has that header and is limited, whatever its peer address.
    """
    if request.META.get("HTTP_FLY_CLIENT_IP"):
        return False
    addr = _parse_ip(request.META.get("REMOTE_ADDR"))
    return addr is not None and addr.is_loopback


def _limit_for(request):
    path = request.path_info
    if path == "/api/incidents.tsv":
        return TSV
    if path == "/admin/login/" and request.method == "POST":
        return ADMIN_LOGIN
    if path in API_PATHS:
        return API
    return PAGES


def _hit(store, limit, client, now):
    """Count one request; return (cache key, count so far, Retry-After)."""
    window = int(now // limit.period)
    key = f"rl:{limit.name}:{client}:{window}"
    # add + incr are each atomic in LocMemCache, so concurrent threads
    # can't lose counts.
    store.add(key, 0, timeout=limit.period)
    try:
        count = store.incr(key)
    except ValueError:  # expired between add and incr
        store.set(key, 1, timeout=limit.period)
        count = 1
    retry_after = max(1, math.ceil((window + 1) * limit.period - now))
    return key, count, retry_after


def _too_many_requests(limit, client, count, retry_after):
    if count == limit.requests + 1:  # log once per client per window
        logger.warning(
            "Rate limit %r exceeded by %s; refusing for %ss",
            limit.name,
            client,
            retry_after,
        )
    response = HttpResponse(
        "Too many requests. Please try again later.\n",
        status=429,
        content_type="text/plain; charset=utf-8",
    )
    response["Retry-After"] = str(retry_after)
    return response


class RateLimitMiddleware:
    """Return 429 with Retry-After once a client exceeds its limit.

    Sits right after WhiteNoise, so static files are never counted and a
    refused request never reaches the session or the database.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if is_in_container(request):
            return self.get_response(request)

        store = caches[CACHE_ALIAS]
        client = client_key(request)
        now = time.time()

        if request.path_info.startswith(KEY_PROTECTED_PREFIXES):
            return self._key_protected(request, store, client, now)

        limit = _limit_for(request)
        _, count, retry_after = _hit(store, limit, client, now)
        if count > limit.requests:
            return _too_many_requests(limit, client, count, retry_after)
        return self.get_response(request)

    def _key_protected(self, request, store, client, now):
        # Count the attempt up front, so concurrent guesses can't all slip
        # under the cap, then refund it if the key was accepted. Once over
        # the cap the client is refused even with the right key; otherwise
        # 429-vs-200 would still reveal a correct guess.
        limit = FUNCTIONS_KEY_FAILURES
        key, count, retry_after = _hit(store, limit, client, now)
        if count > limit.requests:
            return _too_many_requests(limit, client, count, retry_after)
        response = self.get_response(request)
        # A bad key is a 404 (see incidents.views._valid_functions_key).
        if response.status_code != 404:
            try:
                store.decr(key)
            except ValueError:  # window rolled over meanwhile
                pass
        return response


API_INCIDENTS_CACHE_KEY = "response:api_incidents"
API_INCIDENTS_CACHE_SECONDS = 60


def cache_api_incidents(view):
    """Serve /api/incidents/ from the default cache for up to 60 seconds.

    The data only changes when the 5-minute update runs, which clears this
    entry (UpdateIncidentsView), so a burst of polls costs one set of
    queries. The view takes no parameters, so one entry serves every query
    string, and random query strings can't be used to bypass it.
    """

    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if request.method not in ("GET", "HEAD"):
            return view(request, *args, **kwargs)
        response = cache.get(API_INCIDENTS_CACHE_KEY)
        if response is None:
            response = view(request, *args, **kwargs)
            if response.status_code == 200 and not response.streaming:
                cache.set(
                    API_INCIDENTS_CACHE_KEY,
                    response,
                    timeout=API_INCIDENTS_CACHE_SECONDS,
                )
        return response

    return wrapped


def invalidate_api_incidents_cache():
    cache.delete(API_INCIDENTS_CACHE_KEY)
