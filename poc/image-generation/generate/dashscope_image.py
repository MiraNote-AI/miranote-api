"""DashScope text-to-image: qwen-image-3.0, z-image-turbo, wan2.2-t2i-flash,
wanx2.0-t2i-turbo.

Three request shapes live here because DashScope genuinely has three, and which
one a model answers on is not something the docs settle -- the older wan models
are documented on a path the current pages no longer describe, and availability
varies by region. config.IMAGE_MODELS names the shape per model, and each entry
there means "answered on this path once", not "is guaranteed to" -- confirm an id
with one cheap call before trusting it in a paid run.

    mm_sync    POST /services/aigc/multimodal-generation/generation
               input.messages, answers immediately,
               image at output.choices[0].message.content[0].image
    t2i_async  POST /services/aigc/text2image/image-synthesis  (+ async header)
               input.prompt, returns a task id to poll,
               image at output.results[].url
    img_async  POST /services/aigc/image-generation/generation (+ async header)
               input.messages, returns a task id to poll,
               image at output.choices[0].message.content[0].image

Every shape ends in a list of URLs; the caller downloads them. The functions are
split so that building a request and reading a response can be unit tested with
no network (tests/test_image_providers.py).
"""

import time
from concurrent.futures import ThreadPoolExecutor

import config
from generate import http_client
from generate.http_client import ProviderError

KEY_ENV = "DASHSCOPE_API_KEY"

_PATHS = {
    "mm_sync":   "/services/aigc/multimodal-generation/generation",
    "t2i_async": "/services/aigc/text2image/image-synthesis",
    "img_async": "/services/aigc/image-generation/generation",
}
_ASYNC_HEADER = {"X-DashScope-Async": "enable"}


def size_of(model: str, aspect_ratio: str) -> str:
    """DashScope spells a size with a star: 1024*1024."""
    width, height = config.size_for(model, aspect_ratio)
    return f"{width}*{height}"


# --------------------------------------------------------------------------- #
# Requests
# --------------------------------------------------------------------------- #
def build_body(shape: str, model: str, prompt: str, aspect_ratio: str,
               n: int) -> dict:
    size = size_of(model, aspect_ratio)
    # prompt_extend off everywhere: DashScope's own rewriter would replace the
    # text the expansion model produced, which is the thing being measured.
    parameters = {"size": size, "prompt_extend": config.PROMPT_EXTEND,
                  "watermark": False}
    if shape == "t2i_async":
        parameters["n"] = n
        return {"model": model, "input": {"prompt": prompt},
                "parameters": parameters}
    # mm_sync and img_async share the messages-style input.
    return {
        "model": model,
        "input": {"messages": [{"role": "user",
                                "content": [{"text": prompt}]}]},
        "parameters": parameters,
    }


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
def urls_of(shape: str, data: dict) -> list[str]:
    output = data.get("output") or {}
    if shape == "t2i_async":
        results = output.get("results") or []
        # A per-image failure arrives as a result entry with `code`/`message`
        # and no url; surfacing it beats returning a silently short list.
        urls = [r["url"] for r in results if r.get("url")]
        if not urls and results:
            raise ProviderError(str(results[0])[:300])
        return urls
    urls = []
    for choice in output.get("choices") or []:
        for part in (choice.get("message") or {}).get("content") or []:
            if part.get("image"):
                urls.append(part["image"])
    return urls


def _poll(task_id: str, deadline: float) -> dict:
    url = f"{config.DASHSCOPE_BASE_URL}/tasks/{task_id}"
    while True:
        data = http_client.get_json(url, KEY_ENV, timeout=30.0)
        status = (data.get("output") or {}).get("task_status")
        if status == "SUCCEEDED":
            return data
        if status in ("FAILED", "CANCELED", "UNKNOWN"):
            output = data.get("output") or {}
            raise ProviderError(f"task {status}: "
                                f"{output.get('message') or output.get('code') or ''}"[:300])
        if time.monotonic() > deadline:
            raise ProviderError(f"task still {status} after "
                                f"{config.IMAGE_TIMEOUT:.0f}s")
        time.sleep(config.POLL_INTERVAL)


# --------------------------------------------------------------------------- #
# The call
# --------------------------------------------------------------------------- #
def generate_urls(model: str, prompt: str, aspect_ratio: str, n: int) -> list[str]:
    shape = config.IMAGE_MODELS[model]["shape"]
    url = config.DASHSCOPE_BASE_URL + _PATHS[shape]
    deadline = time.monotonic() + config.IMAGE_TIMEOUT

    if shape == "mm_sync":
        # No n parameter on this shape -- one image per call, so ask n times.
        def one() -> list[str]:
            body = build_body(shape, model, prompt, aspect_ratio, 1)
            return urls_of(shape, http_client.post_json(
                url, body, KEY_ENV, timeout=config.IMAGE_TIMEOUT))

        if n == 1:
            return one()
        with ThreadPoolExecutor(max_workers=n) as pool:
            return [u for batch in pool.map(lambda _: one(), range(n))
                    for u in batch]

    body = build_body(shape, model, prompt, aspect_ratio, n)
    created = http_client.post_json(url, body, KEY_ENV, headers=_ASYNC_HEADER,
                                    timeout=60.0)
    task_id = (created.get("output") or {}).get("task_id")
    if not task_id:
        raise ProviderError(f"no task_id in response: {str(created)[:300]}")
    return urls_of(shape, _poll(task_id, deadline))


def generate(model: str, prompt: str, aspect_ratio: str, n: int) -> list[bytes]:
    urls = generate_urls(model, prompt, aspect_ratio, n)
    if not urls:
        raise ProviderError(f"{model} returned no image")
    return [http_client.fetch_image(u) for u in urls]
