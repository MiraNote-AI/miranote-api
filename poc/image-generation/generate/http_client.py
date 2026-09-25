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

import requests
from dotenv import load_dotenv

load_dotenv()


class ProviderError(RuntimeError):
    """A provider call failed. Carries text short enough to land in a CSV cell."""


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
    delay = 2.0
    for attempt in range(attempts):
        response = requests.request(method, url, headers=head, json=json,
                                    timeout=timeout)
        if response.ok:
            return response.json()
        if attempt == attempts - 1 or not _should_retry(response):
            raise ProviderError(f"{method} {url.rsplit('/', 1)[-1]} -> "
                                f"{response.status_code} {response.text[:300]}")
        # Jitter so six concurrent providers backing off at once do not all
        # come back in the same instant.
        time.sleep(delay + random.uniform(0, 1))
        delay *= 2
    raise AssertionError("unreachable")


def post_json(url: str, payload: dict, key_env: str, *, headers=None,
              timeout: float = 60.0) -> dict:
    return _request("POST", url, key_env, headers=headers, json=payload,
                    timeout=timeout)


def get_json(url: str, key_env: str, *, timeout: float = 60.0) -> dict:
    return _request("GET", url, key_env, timeout=timeout)


def fetch_image(url: str, timeout: float = 60.0) -> bytes:
    """Download a generated image.

    Both platforms hand back a URL valid for 24 hours rather than bytes, so
    every provider ends with this call and the pipeline keeps its list[bytes]
    contract -- main.py, the matte step and the benchmark stay unaware that
    anything but Gemini was involved.
    """
    response = requests.get(url, timeout=timeout)
    if not response.ok:
        raise ProviderError(f"image download -> {response.status_code}")
    return response.content
