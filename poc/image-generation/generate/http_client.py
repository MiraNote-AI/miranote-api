"""Bearer-auth JSON over HTTP for the DashScope provider.

DashScope is plain REST -- no vendor SDK is installed, because the whole surface
needed is a POST, a GET and an image download. Keeping that here leaves
dashscope_image.py and dashscope_text.py holding only what differs between them:
the request body and where the result lives in the response.

Generic on purpose rather than DashScope-specific: `key_env` is a parameter, so a
second Bearer-auth REST provider is a new adapter module and no change here.

Credentials are read from the environment after load_dotenv(), the same way
shared/vertex_client.py does it, and read lazily so importing this module
without a key set is fine (the unit tests do exactly that).
"""

import os
import random
import time
from contextlib import contextmanager
from contextvars import ContextVar

import requests
import urllib3
from dotenv import load_dotenv

load_dotenv()


class ProviderError(RuntimeError):
    """A provider call failed. Carries text short enough to land in a CSV cell."""


class Timeout(ProviderError):
    """A call ran out of time: its own read timeout, or the request's budget.

    A subclass so every existing `except ProviderError` still catches it, while
    main.py can answer it as a 504 instead of the 502 an unknown failure gets.
    """


# Opening the connection is quick when the provider is up; a slow connect is a
# network problem that waiting longer does not fix.
CONNECT_TIMEOUT = 5.0

# The request's remaining time, as a time.monotonic() deadline, and the dict its
# step timings go into. Context variables rather than parameters: asyncio's
# to_thread copies them into the worker thread, so /generate sets them once and
# every provider call below it sees them without a signature change -- and
# without one, a call outside /generate (the bench, a unit test) keeps its own
# timeouts exactly as before.
_deadline: ContextVar[float | None] = ContextVar("deadline", default=None)
_timings: ContextVar[dict | None] = ContextVar("timings", default=None)


@contextmanager
def budget(seconds: float, timings: dict):
    """Bound every call made inside this block to `seconds` in total, and
    collect their timed() steps into `timings`."""
    deadline_token = _deadline.set(time.monotonic() + seconds)
    timings_token = _timings.set(timings)
    try:
        yield
    finally:
        _deadline.reset(deadline_token)
        _timings.reset(timings_token)


def remaining() -> float | None:
    """Seconds left in the current budget, or None outside one."""
    deadline = _deadline.get()
    return None if deadline is None else deadline - time.monotonic()


@contextmanager
def timed(name: str):
    """Add this block's wall-clock ms to the current budget's timings under
    `name`. Added rather than overwritten: the empty-answer retry makes a
    second image call, and both belong in the total. A no-op outside a budget."""
    timings = _timings.get()
    start = time.perf_counter()
    try:
        yield
    finally:
        if timings is not None:
            spent = (time.perf_counter() - start) * 1000
            timings[name] = round(timings.get(name, 0) + spent, 1)


def _timeout(read: float, what: str) -> tuple[float, float]:
    """(connect, read) for one attempt, cut down to the budget's remainder.

    Inside a budget the two add up to no more than what is left, so a call that
    connects slowly and then stalls still ends by the deadline."""
    left = remaining()
    if left is None:
        return min(CONNECT_TIMEOUT, read), read
    if left <= 0:
        raise Timeout(f"{what}: no time left in the request")
    connect = min(CONNECT_TIMEOUT, left / 2)
    return connect, min(read, left - connect)


def _transport_error(what: str, error: Exception) -> ProviderError:
    """The ProviderError for a call that never produced a whole response.

    A read timeout that strikes while the body is arriving reaches us as a
    ConnectionError (requests) or a bare ReadTimeoutError (urllib3), not as
    requests.Timeout, so all three count as running out of time. Anything else
    -- refused, reset, DNS -- is a ProviderError main.py can classify, rather
    than a bare exception that surfaces as a 500."""
    timed_out = (isinstance(error, (requests.Timeout, urllib3.exceptions.ReadTimeoutError))
                 or any(isinstance(arg, urllib3.exceptions.ReadTimeoutError)
                        for arg in error.args))
    if timed_out:
        return Timeout(f"{what} timed out: {type(error).__name__}")
    return ProviderError(f"{what} failed: {type(error).__name__}")


def api_key(env_var: str) -> str:
    key = os.environ.get(env_var, "").strip()
    if not key or key == "XXX":
        raise ProviderError(f"{env_var} is not set (see .env.example)")
    return key


# Retried: the platform is telling us to slow down, not that the request is
# wrong. Anything else (bad model id, unavailable model, content rejection) is
# a real answer and gets returned to the caller on the first try.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_RETRY_CODES = ("Throttling", "RequestTimeOut", "ServiceUnavailable")


def _should_retry(response: requests.Response) -> bool:
    if response.status_code in _RETRY_STATUS:
        return True
    try:
        code = str(response.json().get("code", ""))
    except ValueError:
        return False
    return any(c in code for c in _RETRY_CODES)


def _request(method: str, url: str, key_env: str, *, headers=None, json=None,
             timeout: float = 60.0, attempts: int = 4) -> dict:
    head = {"Authorization": f"Bearer {api_key(key_env)}",
            "Content-Type": "application/json"}
    head.update(headers or {})
    what = f"{method} {url.rsplit('/', 1)[-1]}"
    delay = 2.0
    for attempt in range(attempts):
        try:
            response = requests.request(method, url, headers=head, json=json,
                                        timeout=_timeout(timeout, what))
        except requests.RequestException as error:
            raise _transport_error(what, error) from error
        if response.ok:
            return response.json()
        pause = delay + random.uniform(0, 1)
        left = remaining()
        # A backoff the budget cannot cover only delays the same failure, so the
        # provider's answer goes back now instead.
        out_of_time = left is not None and left < pause + CONNECT_TIMEOUT
        if attempt == attempts - 1 or not _should_retry(response) or out_of_time:
            raise ProviderError(f"{what} -> "
                                f"{response.status_code} {response.text[:300]}")
        # Jitter so six concurrent providers backing off at once do not all
        # come back in the same instant.
        time.sleep(pause)
        delay *= 2
    raise AssertionError("unreachable")


def post_json(url: str, payload: dict, key_env: str, *, headers=None,
              timeout: float = 60.0, attempts: int = 4) -> dict:
    return _request("POST", url, key_env, headers=headers, json=payload,
                    timeout=timeout, attempts=attempts)


def get_json(url: str, key_env: str, *, timeout: float = 60.0) -> dict:
    return _request("GET", url, key_env, timeout=timeout)


def fetch_image(url: str, timeout: float = 60.0) -> bytes:
    """Download a generated image.

    Both platforms hand back a URL valid for 24 hours rather than bytes, so
    every provider ends with this call and the pipeline keeps its list[bytes]
    contract -- main.py, the matte step and the benchmark stay unaware that
    anything but Gemini was involved.

    Streamed, with the deadline checked between chunks: requests' read timeout
    only bounds each gap between socket reads, so a ~1 MB image trickling in a
    few bytes at a time would otherwise never trip it.
    """
    what = "image download"
    try:
        with requests.get(url, timeout=_timeout(timeout, what), stream=True) as response:
            if not response.ok:
                raise ProviderError(f"{what} -> {response.status_code}")
            return _read_body(response, what)
    except (requests.RequestException, urllib3.exceptions.HTTPError) as error:
        raise _transport_error(what, error) from error


_CHUNK = 64 * 1024


def _read_body(response: requests.Response, what: str) -> bytes:
    """The whole body, read inside the current budget (if any)."""
    body = bytearray()
    while True:
        left = remaining()
        if left is not None:
            if left <= 0:
                raise Timeout(f"{what}: no time left in the request")
            # The socket's read timeout was set from the budget when the call
            # began; a stall late in the body must not get that full amount again.
            sock = getattr(getattr(response.raw, "connection", None), "sock", None)
            if sock is not None:
                sock.settimeout(left)
        # read1, not read: it hands back whatever one socket read produced
        # instead of blocking until the whole chunk has arrived.
        chunk = response.raw.read1(_CHUNK, decode_content=True)
        if not chunk:
            return bytes(body)
        body += chunk
