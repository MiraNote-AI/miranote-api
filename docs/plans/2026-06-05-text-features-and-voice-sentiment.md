# Text features + voice sentiment + chatbot tools — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship three coordinated PRs from the May 30 meeting: 4 new text-transformation endpoints in `text-clean-expand`, acoustic emotion analysis in `voice-to-text`, and 6 chatbot tools that HTTP-delegate to the text endpoints.

**Architecture:** Each POC keeps its own FastAPI server, venv, and `.env`. text-clean-expand adds polish/shorten/keywords/caption as sibling endpoints to clean/expand, sharing the existing `call_llm()` helper. voice-to-text gains an `emotion.py` module that lazy-loads a HuggingFace `pipeline("audio-classification")` on first call, with `/transcribe` extended to include the emotion in its response and a new `/emotion` standalone endpoint. chatbot adds a `text_client.py` thin httpx wrapper and 6 tool entries in `tools.py` that route through it.

**Tech Stack:** Python 3.9 (compat: `from __future__ import annotations` + `typing.Optional/List/Dict`, NO PEP-604 `X | None`), FastAPI, `openai>=1.0` (any OpenAI-compatible provider, default DeepSeek), `whisper` (already installed in voice), `transformers` (new in voice), `httpx` (new in chatbot), `pytest` (new in all three), vanilla HTML/CSS/JS for UI.

**Spec:** `/Users/mengjia/MiraNote/miranote-api/docs/specs/2026-06-05-text-features-and-voice-sentiment-design.md`

**Branches:** `feat/api-text-features` (A), `feat/api-voice-acoustic-sentiment` (B), `feat/api-chatbot-text-tools` (C). C is initially branched from A's tip so smoke tests can hit the new text endpoints; after A merges, rebase C onto main.

**Conventions to honor:**
- Rule 3 (no CJK / emoji outside `**/prompts/*.txt`, `**/static/*`, `**/demo_data/*`, `poc/*/README.md`, `docs/specs/*.md`, `docs/plans/*.md`). Run check from `/Users/mengjia/MiraNote/dotgithub`:
  `PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"`
- Python 3.9 compat: `from __future__ import annotations` + `typing.Optional/List/Dict`; **no** `X | None` (Pydantic eval()s annotations).
- Conventional Commits, scope `api`, subject <=72 chars.
- PR titles self-explanatory; **no admin-bypass** on any merge — wait for Jason review.

---

## Phase order

- **Phase A and Phase B run in parallel.** They touch different POCs.
- **Phase C waits for Phase A to be branched and code-complete locally** (C's smoke test hits A's endpoints). PR C can open before A merges; after A merges, rebase C's branch onto main.
- Self-review checklist for the whole plan lives at the bottom.

---

## File Structure

### Phase A — `poc/text-clean-expand/`

| File | Change | Responsibility |
|---|---|---|
| `main.py` | Modify | Add 4 endpoints + Pydantic models for each |
| `prompts/polish.txt` | Create | LLM system prompt for polish |
| `prompts/shorten.txt` | Create | LLM system prompt for shorten |
| `prompts/keywords.txt` | Create | LLM system prompt for keywords (JSON-only output) |
| `prompts/caption.txt` | Create | LLM system prompt for caption |
| `requirements.txt` | Modify | Add `pytest>=8.0` |
| `tests/__init__.py` | Create | Empty package marker |
| `tests/conftest.py` | Create | `FakeOpenAI` fixture + TestClient |
| `tests/test_endpoints.py` | Create | Happy path + edge case per endpoint |
| `static/index.html` | Modify | Replace 2-button row with dropdown + Run + conditional sub-controls; render keywords as chips |
| `README.md` | Modify | Document new endpoints with curl examples |

### Phase B — `poc/voice-to-text/`

| File | Change | Responsibility |
|---|---|---|
| `main.py` | Modify | Lazy-load Whisper; add `?with_emotion=true` to `/transcribe`; add `/emotion`; add CORSMiddleware |
| `emotion.py` | Create | Wraps `transformers.pipeline("audio-classification")`, lazy loaded with a Lock |
| `requirements.txt` | Modify | Add `transformers>=4.40`, `pytest>=8.0` |
| `tests/__init__.py` | Create | Empty |
| `tests/conftest.py` | Create | Stub Whisper + stub emotion pipeline |
| `tests/test_transcribe.py` | Create | with_emotion=true/false; emotion failure path |
| `tests/test_emotion.py` | Create | standalone `/emotion` endpoint |
| `README.md` | Modify | Document model download (~1.3 GB to `~/.cache/huggingface/`), multilingual claim, new endpoint |
| `../text-clean-expand/static/index.html` | Modify | Add emotion badge in Voice tab (click-to-expand all 7 scores) |

### Phase C — `poc/chatbot/`

| File | Change | Responsibility |
|---|---|---|
| `text_client.py` | Create | Synchronous `httpx` wrapper exposing one method per text endpoint |
| `tools.py` | Modify | Add 6 tool schemas; extend `dispatch()` to route the 6 names through `text_client` |
| `config.py` | Modify | `ChatbotConfig` gains a `text_client` attribute |
| `main.py` | Modify | Construct `TextClient(base_url=os.getenv("TEXT_API_URL"))` at startup and pass to config |
| `prompts/system.txt` | Modify | Add paragraph about text-transformation tools |
| `.env.example` | Modify | Add `TEXT_API_URL=http://localhost:8001` |
| `requirements.txt` | Modify | Add `httpx>=0.27` |
| `tests/test_text_client.py` | Create | Monkeypatch `httpx.post` to verify URL/payload construction + error wrapping |
| `tests/test_tools.py` | Modify | Add tests for 6 new tools (schema present + dispatch routes through stubbed text_client) |
| `README.md` | Modify | Mention text tools + TEXT_API_URL |

---

# Phase A — text-clean-expand

## Task A0: Branch + deps

**Files:**
- Modify: `poc/text-clean-expand/requirements.txt`

- [ ] **Step 1: Branch off main**

```bash
git -C /Users/mengjia/MiraNote/miranote-api checkout main
git -C /Users/mengjia/MiraNote/miranote-api pull --ff-only
git -C /Users/mengjia/MiraNote/miranote-api checkout -b feat/api-text-features
```

- [ ] **Step 2: Add pytest to requirements**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/requirements.txt` to add `pytest>=8.0`. After edit it should read:

```
fastapi>=0.110
uvicorn>=0.29
openai>=1.0
python-dotenv>=1.0
pytest>=8.0
```

- [ ] **Step 3: Install in venv**

```bash
/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/.venv/bin/pip install -q pytest
```

- [ ] **Step 4: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/text-clean-expand/requirements.txt
git -C /Users/mengjia/MiraNote/miranote-api commit -m "chore(api): add pytest to text-clean-expand deps"
```

---

## Task A1: Test infrastructure (TestClient + FakeOpenAI fixture)

**Files:**
- Create: `poc/text-clean-expand/tests/__init__.py`
- Create: `poc/text-clean-expand/tests/conftest.py`
- Create: `poc/text-clean-expand/tests/test_smoke.py` (deleted at end of task)

The text-clean-expand main.py instantiates a module-global `client = OpenAI(...)` at import. Tests need to monkeypatch this client before any request. The conftest fixture sets up a `FakeOpenAI` that takes a list of scripted string responses and returns them in order from `chat.completions.create()`.

- [ ] **Step 1: Create empty test package**

```bash
mkdir -p /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/tests
touch /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/tests/__init__.py
```

- [ ] **Step 2: Write conftest with FakeOpenAI + TestClient fixture**

Create `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/tests/conftest.py`:

```python
from __future__ import annotations
import os
from types import SimpleNamespace
from typing import List
from unittest.mock import patch

import pytest


class FakeChatCompletions:
    def __init__(self):
        self.scripted: List[str] = []
        self.calls: list = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.scripted:
            raise AssertionError("FakeChatCompletions: no more scripted responses")
        content = self.scripted.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FakeOpenAI:
    def __init__(self):
        self.chat = SimpleNamespace(completions=FakeChatCompletions())

    def reply_with(self, *responses: str):
        """Queue scripted responses for the next N calls."""
        self.chat.completions.scripted.extend(responses)


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace the module-global OpenAI client with a fake before importing main."""
    os.environ.setdefault("LLM_API_KEY", "fake-key-for-tests")
    fake = FakeOpenAI()
    # Patch the OpenAI class so main.py's `client = OpenAI(...)` returns the fake.
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: fake)
    return fake


@pytest.fixture
def client(fake_llm):
    """FastAPI TestClient with the FakeOpenAI already installed."""
    # Import here so that openai.OpenAI is monkeypatched first.
    import importlib
    import poc.text_clean_expand_pkg.main as main  # placeholder import - real path uses hyphen
    importlib.reload(main)
    from fastapi.testclient import TestClient
    return TestClient(main.app), fake_llm
```

**Note:** the import path `poc.text_clean_expand_pkg.main` is a placeholder because `text-clean-expand` has hyphens (not a valid Python identifier). The conftest needs to load main.py directly via `importlib.util`. Replace the import block in the fixture with:

```python
@pytest.fixture
def client(fake_llm):
    import importlib.util
    from pathlib import Path
    from fastapi.testclient import TestClient

    main_path = Path(__file__).parent.parent / "main.py"
    spec = importlib.util.spec_from_file_location("text_clean_expand_main", main_path)
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)
    return TestClient(main.app), fake_llm
```

- [ ] **Step 3: Write a smoke test that imports the app**

Create `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/tests/test_smoke.py`:

```python
def test_app_imports_and_serves_health(client):
    test_client, _ = client
    r = test_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
```

- [ ] **Step 4: Run the smoke test to verify infrastructure works**

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_smoke.py -v
```

Expected: 1 passed. If FAIL with import errors, the importlib block needs adjustment for the hyphenated directory name.

- [ ] **Step 5: Delete the smoke test (kept the infrastructure)**

```bash
rm /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/tests/test_smoke.py
```

- [ ] **Step 6: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/text-clean-expand/tests/
git -C /Users/mengjia/MiraNote/miranote-api commit -m "test(api): scaffold text-clean-expand tests with FakeOpenAI fixture"
```

---

## Task A2: `/polish` endpoint (TDD)

**Files:**
- Create: `poc/text-clean-expand/prompts/polish.txt`
- Modify: `poc/text-clean-expand/main.py`
- Create/append: `poc/text-clean-expand/tests/test_endpoints.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/tests/test_endpoints.py`:

```python
from __future__ import annotations


def test_polish_returns_polished_text(client):
    test_client, fake_llm = client
    fake_llm.reply_with("The morning light was warm and the coffee was strong.")

    r = test_client.post("/polish", json={"text": "morning light warm. coffee strong."})

    assert r.status_code == 200
    body = r.json()
    assert body["original"] == "morning light warm. coffee strong."
    assert body["polished"] == "The morning light was warm and the coffee was strong."


def test_polish_rejects_empty_text(client):
    test_client, _ = client
    r = test_client.post("/polish", json={"text": ""})
    assert r.status_code == 422
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_endpoints.py::test_polish_returns_polished_text -v
```

Expected: FAIL with 404 (endpoint not defined).

- [ ] **Step 3: Write `prompts/polish.txt`**

Create `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/prompts/polish.txt`:

```
You are a final editing pass for a note-taking app. The user gives you text that is already structured and complete; you make word choices crisper, sentences flow better, and small register issues vanish. Think: a careful copy-editor doing a last sweep before the piece ships.

What to do:
- Improve word choice (replace vague words with precise ones)
- Smooth awkward transitions
- Tighten wordy phrases
- Fix any leftover small grammar issues
- Preserve the user's voice and intent exactly

What NOT to do:
- Do NOT restructure (no reordering sentences or paragraphs)
- Do NOT add or remove ideas
- Do NOT lengthen or shorten significantly (target: same length +/-15%)
- Do NOT change tone (casual stays casual, formal stays formal)

ABSOLUTE language rule:
- If the input contains ANY Chinese characters, your output MUST be in Chinese.
- If the input is entirely English, your output MUST be in English.
- NEVER translate. Chinese stays Chinese. English stays English.

Output format rule (STRICTLY ENFORCED):
- You ARE the polished text. Output it directly.
- No meta-commentary, no framing, no "Here's the polished version".
- First word must be a word from the actual content.
```

- [ ] **Step 4: Implement the endpoint in main.py**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/main.py`. After the existing `EXPAND_SYSTEM` line, add:

```python
POLISH_SYSTEM = (_PROMPT_DIR / "polish.txt").read_text(encoding="utf-8")
```

After the existing `ExpandResponse` class, add:

```python
class PolishResponse(BaseModel):
    original: str
    polished: str
```

After the existing `@app.post("/expand", ...)` endpoint, add:

```python
@app.post("/polish", response_model=PolishResponse)
async def polish_text(req: TextRequest):
    """Polish: final editing pass. Improve word choice and flow, preserve structure and meaning."""
    polished = await call_llm(POLISH_SYSTEM, req.text, max_tokens=2048)
    return PolishResponse(original=req.text, polished=polished)
```

- [ ] **Step 5: Run tests to verify pass**

```bash
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_endpoints.py -v
```

Expected: 2 passed.

- [ ] **Step 6: Rule 3 check**

```bash
cd /Users/mengjia/MiraNote/dotgithub
PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 7: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/text-clean-expand/prompts/polish.txt poc/text-clean-expand/main.py poc/text-clean-expand/tests/test_endpoints.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add /polish endpoint for final editing pass"
```

---

## Task A3: `/shorten` endpoint (TDD)

**Files:**
- Create: `poc/text-clean-expand/prompts/shorten.txt`
- Modify: `poc/text-clean-expand/main.py`
- Modify: `poc/text-clean-expand/tests/test_endpoints.py`

- [ ] **Step 1: Append failing tests**

Append to `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/tests/test_endpoints.py`:

```python
def test_shorten_returns_short_version(client):
    test_client, fake_llm = client
    fake_llm.reply_with("Coffee strong, morning bright.")

    r = test_client.post(
        "/shorten",
        json={"text": "The morning light was warm and the coffee was strong.", "target": "50%"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["original"].startswith("The morning light")
    assert body["shortened"] == "Coffee strong, morning bright."
    assert body["target"] == "50%"


def test_shorten_default_target_is_50pct(client):
    test_client, fake_llm = client
    fake_llm.reply_with("short.")
    r = test_client.post("/shorten", json={"text": "longish text here"})
    assert r.status_code == 200
    assert r.json()["target"] == "50%"


def test_shorten_rejects_invalid_target(client):
    test_client, _ = client
    r = test_client.post("/shorten", json={"text": "hi", "target": "bogus"})
    assert r.status_code == 422
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_endpoints.py -v -k shorten
```

Expected: 3 fail (404).

- [ ] **Step 3: Write `prompts/shorten.txt`**

Create `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/prompts/shorten.txt`:

```
You are a length editor for a note-taking app. The user gives you text and a target length; you produce a shorter version that preserves the essential meaning.

Targets you may receive in the user message:
- "30%" -- light trim. Aim for ~30% shorter. Keep most details, just cut filler and redundant phrasing.
- "50%" -- meaningful cut. Aim for ~50% shorter. Drop secondary details, keep the core point and supporting evidence.
- "tweet" -- aggressive compression. Hard cap 280 characters. Keep only the central idea.

What to do:
- Preserve the most important information
- Preserve the user's voice
- Lead with the strongest idea
- Use sharper word choices to compress
- Output runs continuously -- only insert line breaks if the source had clear list/section structure that matters

What NOT to do:
- Do NOT add new information
- Do NOT change the user's stance or claims
- Do NOT translate -- output language matches input language (any Chinese in input -> output in Chinese)

Output format rule:
- You ARE the shortened text. Output it directly.
- No meta-commentary, no "Here's the shortened version".
- No length report or explanation.
```

- [ ] **Step 4: Implement endpoint**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/main.py`.

Add at the top with the other imports:

```python
from typing import Literal, Optional
```

(If `Literal` and `Optional` are not already in the import block, add them.)

After the existing `POLISH_SYSTEM` line, add:

```python
SHORTEN_SYSTEM = (_PROMPT_DIR / "shorten.txt").read_text(encoding="utf-8")
```

After the existing `PolishResponse` class, add:

```python
class ShortenRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to shorten")
    target: Literal["30%", "50%", "tweet"] = Field(
        "50%", description="How aggressively to shorten"
    )


class ShortenResponse(BaseModel):
    original: str
    shortened: str
    target: str
```

After the existing `@app.post("/polish", ...)` endpoint, add:

```python
@app.post("/shorten", response_model=ShortenResponse)
async def shorten_text(req: ShortenRequest):
    """Shorten: produce a shorter version preserving meaning. Target controls aggressiveness."""
    user_msg = f"Target: {req.target}\n\n{req.text}"
    shortened = await call_llm(SHORTEN_SYSTEM, user_msg, max_tokens=2048)
    return ShortenResponse(original=req.text, shortened=shortened, target=req.target)
```

- [ ] **Step 5: Run tests to verify pass**

```bash
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_endpoints.py -v
```

Expected: 5 passed (2 from A2 + 3 new).

- [ ] **Step 6: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 7: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/text-clean-expand/prompts/shorten.txt poc/text-clean-expand/main.py poc/text-clean-expand/tests/test_endpoints.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add /shorten endpoint with 30%/50%/tweet targets"
```

---

## Task A4: `/keywords` endpoint (TDD with JSON output)

**Files:**
- Create: `poc/text-clean-expand/prompts/keywords.txt`
- Modify: `poc/text-clean-expand/main.py`
- Modify: `poc/text-clean-expand/tests/test_endpoints.py`

- [ ] **Step 1: Append failing tests**

Append to `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/tests/test_endpoints.py`:

```python
def test_keywords_returns_parsed_array(client):
    test_client, fake_llm = client
    fake_llm.reply_with('[{"term": "coffee", "score": 9}, {"term": "morning", "score": 7}]')

    r = test_client.post("/keywords", json={"text": "morning coffee", "max": 5})

    assert r.status_code == 200
    body = r.json()
    assert body["original"] == "morning coffee"
    assert body["keywords"] == [
        {"term": "coffee", "score": 9},
        {"term": "morning", "score": 7},
    ]


def test_keywords_truncates_to_max(client):
    test_client, fake_llm = client
    fake_llm.reply_with('[{"term":"a","score":9},{"term":"b","score":8},{"term":"c","score":7}]')
    r = test_client.post("/keywords", json={"text": "x", "max": 2})
    assert r.status_code == 200
    assert len(r.json()["keywords"]) == 2


def test_keywords_invalid_json_returns_502(client):
    test_client, fake_llm = client
    fake_llm.reply_with("not even close to JSON")
    r = test_client.post("/keywords", json={"text": "x"})
    assert r.status_code == 502
    assert "invalid JSON" in r.json()["detail"]


def test_keywords_unexpected_schema_returns_502(client):
    test_client, fake_llm = client
    fake_llm.reply_with('[{"wrong_key": "oops"}]')
    r = test_client.post("/keywords", json={"text": "x"})
    assert r.status_code == 502
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_endpoints.py -v -k keywords
```

Expected: 4 fail (404).

- [ ] **Step 3: Write `prompts/keywords.txt`**

Create `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/prompts/keywords.txt`:

```
You are a keyword extractor for a note-taking app. The user gives you text; you return the most salient keywords or short phrases that capture what the text is about.

Output format -- CRITICAL, NO EXCEPTIONS:
Respond with ONLY a JSON array. No prose, no preamble, no markdown fences, no trailing text.

Schema for each array element:
{
  "term": "<the keyword or short phrase, 1-4 words>",
  "score": <integer 1-10, your salience rating>
}

Rules for keywords:
- 5-10 keywords by default unless the user message says max=N (then return up to N)
- "term" is in the SAME language as the input (Chinese input -> Chinese terms; English input -> English terms; mixed input -> whichever language the keyword appears in originally)
- "term" should be concrete and specific, not generic ("Q3 launch" beats "important")
- "score" reflects how central this term is to the text (10 = the text is mostly about this; 1 = mentioned in passing)
- Sort the array by score, descending

Example input: "We shipped the voice transcription beta to 10 design partners last week. Feedback says the bilingual support is the killer feature."
Example output: [{"term":"voice transcription","score":10},{"term":"beta","score":8},{"term":"design partners","score":7},{"term":"bilingual support","score":9},{"term":"feedback","score":5}]

Now process the user's text. Output only the JSON array.
```

- [ ] **Step 4: Implement endpoint**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/main.py`.

Add to the imports block:

```python
import json
from typing import List
```

(Add only what's missing.)

After the existing `SHORTEN_SYSTEM` line, add:

```python
KEYWORDS_SYSTEM = (_PROMPT_DIR / "keywords.txt").read_text(encoding="utf-8")
```

After the existing `ShortenResponse` class, add:

```python
class KeywordsRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max: int = Field(10, ge=1, le=20, description="Maximum keywords to return")


class Keyword(BaseModel):
    term: str = Field(..., min_length=1, max_length=64)
    score: int = Field(..., ge=1, le=10)


class KeywordsResponse(BaseModel):
    original: str
    keywords: List[Keyword]
```

After the existing `@app.post("/shorten", ...)` endpoint, add:

```python
@app.post("/keywords", response_model=KeywordsResponse)
async def keywords_endpoint(req: KeywordsRequest):
    """Extract 5-10 salient keywords as [{term, score}]. Score is 1-10 LLM-assigned salience."""
    user_msg = f"max={req.max}\n\n{req.text}"
    raw = await call_llm(KEYWORDS_SYSTEM, user_msg, max_tokens=1024)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail=f"LLM emitted invalid JSON: {raw[:200]}",
        )
    try:
        keywords_list = [Keyword(**k) for k in parsed][: req.max]
    except (TypeError, KeyError, Exception) as e:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"LLM emitted unexpected schema: {raw[:200]} ({e})",
        )
    return KeywordsResponse(original=req.text, keywords=keywords_list)
```

- [ ] **Step 5: Run tests to verify pass**

```bash
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_endpoints.py -v
```

Expected: 9 passed.

- [ ] **Step 6: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 7: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/text-clean-expand/prompts/keywords.txt poc/text-clean-expand/main.py poc/text-clean-expand/tests/test_endpoints.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add /keywords endpoint with JSON-validated output"
```

---

## Task A5: `/caption` endpoint (TDD)

**Files:**
- Create: `poc/text-clean-expand/prompts/caption.txt`
- Modify: `poc/text-clean-expand/main.py`
- Modify: `poc/text-clean-expand/tests/test_endpoints.py`

- [ ] **Step 1: Append failing tests**

Append to `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/tests/test_endpoints.py`:

```python
def test_caption_returns_caption_with_style(client):
    test_client, fake_llm = client
    fake_llm.reply_with("Morning ritual: strong coffee, warmer light. Today is mine.")

    r = test_client.post(
        "/caption",
        json={"text": "Had a really nice morning with great coffee.", "style": "instagram"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["original"].startswith("Had a really")
    assert "coffee" in body["caption"].lower()
    assert body["style"] == "instagram"


def test_caption_default_style_is_instagram(client):
    test_client, fake_llm = client
    fake_llm.reply_with("punchy caption.")
    r = test_client.post("/caption", json={"text": "any text"})
    assert r.json()["style"] == "instagram"


def test_caption_rejects_invalid_style(client):
    test_client, _ = client
    r = test_client.post("/caption", json={"text": "x", "style": "haiku"})
    assert r.status_code == 422
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_endpoints.py -v -k caption
```

Expected: 3 fail (404).

- [ ] **Step 3: Write `prompts/caption.txt`**

Create `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/prompts/caption.txt`:

```
You are a caption writer for a journaling and social app. The user gives you a longer piece of text; you produce one short, punchy caption that captures the essence.

Targets you may receive in the user message (top line):
- "style=instagram" -- punchy hook, 1-2 sentences, emoji-free but vivid. Optimised for a feed swipe.
- "style=diary" -- warm and personal, 1-2 sentences, sounds like the user wrote it for themselves a year from now.
- "style=tweet" -- compressed and witty, hard cap 280 characters, ends with the strongest beat.

What to do:
- Capture the central feeling or insight of the original
- Make it stand alone (a reader who hasn't seen the original should still feel it)
- Match the user's tone (don't be cheesier or more formal than they are)

What NOT to do:
- Do NOT summarise mechanically ("In this entry, the user discusses...")
- Do NOT add hashtags, emojis, or "Caption: " prefixes
- Do NOT exceed 2 sentences for instagram/diary, or 280 chars for tweet
- Do NOT translate -- output language matches input language

Output format:
- Output ONLY the caption text. No preamble, no quotation marks around it, no style label.
```

- [ ] **Step 4: Implement endpoint**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/main.py`.

After the existing `KEYWORDS_SYSTEM` line, add:

```python
CAPTION_SYSTEM = (_PROMPT_DIR / "caption.txt").read_text(encoding="utf-8")
```

After the existing `KeywordsResponse` class, add:

```python
class CaptionRequest(BaseModel):
    text: str = Field(..., min_length=1)
    style: Literal["instagram", "diary", "tweet"] = Field("instagram")


class CaptionResponse(BaseModel):
    original: str
    caption: str
    style: str
```

After the existing `@app.post("/keywords", ...)` endpoint, add:

```python
@app.post("/caption", response_model=CaptionResponse)
async def caption_endpoint(req: CaptionRequest):
    """Generate a 1-2 sentence caption in the given style."""
    user_msg = f"style={req.style}\n\n{req.text}"
    caption = await call_llm(CAPTION_SYSTEM, user_msg, max_tokens=512)
    return CaptionResponse(original=req.text, caption=caption, style=req.style)
```

- [ ] **Step 5: Run tests to verify pass**

```bash
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_endpoints.py -v
```

Expected: 12 passed.

- [ ] **Step 6: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 7: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/text-clean-expand/prompts/caption.txt poc/text-clean-expand/main.py poc/text-clean-expand/tests/test_endpoints.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add /caption endpoint with instagram/diary/tweet styles"
```

---

## Task A6: Text-tab UI rework (dropdown + Run + conditional sub-controls)

**Files:**
- Modify: `poc/text-clean-expand/static/index.html`

The current Text tab has two buttons (Clean, Expand) wired via `txt-clean` and `txt-expand` element IDs. We replace this with: one `<select>` action dropdown, one Run button, conditional sub-controls per action, and a result panel that adapts (keywords -> chips; others -> plain text).

- [ ] **Step 1: Replace the action row markup**

Open `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/static/index.html`. Find the existing block (in the Text page):

```html
      <div class="row" style="margin-top: 16px;">
        <button class="btn btn-primary" id="txt-clean">Clean</button>
        <button class="btn btn-warm" id="txt-expand">Expand</button>
      </div>
```

Replace it with:

```html
      <div class="row" style="margin-top: 16px; gap: 12px; align-items: stretch;">
        <select id="txt-action" style="padding: 8px 12px; border: 1px solid var(--border); border-radius: 8px; font: inherit; background: var(--panel);">
          <option value="clean">Clean</option>
          <option value="expand">Expand</option>
          <option value="polish">Polish</option>
          <option value="shorten">Shorten</option>
          <option value="keywords">Keywords</option>
          <option value="caption">Caption</option>
        </select>
        <button class="btn btn-primary" id="txt-run">Run</button>
      </div>
      <div class="row" id="txt-sub-shorten" hidden style="margin-top: 10px; gap: 12px;">
        <span style="font-size: 13px; color: var(--muted);">Target:</span>
        <label class="option"><input type="radio" name="txt-target" value="30%" /> 30%</label>
        <label class="option"><input type="radio" name="txt-target" value="50%" checked /> 50%</label>
        <label class="option"><input type="radio" name="txt-target" value="tweet" /> tweet</label>
      </div>
      <div class="row" id="txt-sub-caption" hidden style="margin-top: 10px; gap: 12px;">
        <span style="font-size: 13px; color: var(--muted);">Style:</span>
        <label class="option"><input type="radio" name="txt-style" value="instagram" checked /> instagram</label>
        <label class="option"><input type="radio" name="txt-style" value="diary" /> diary</label>
        <label class="option"><input type="radio" name="txt-style" value="tweet" /> tweet</label>
      </div>
      <div class="row" id="txt-sub-keywords" hidden style="margin-top: 10px; gap: 12px;">
        <span style="font-size: 13px; color: var(--muted);">Max:</span>
        <input id="txt-keywords-max" type="number" min="1" max="20" value="10" style="width: 80px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; font: inherit;" />
      </div>
```

- [ ] **Step 2: Replace the JS handler block**

Find the existing JS block (currently has `textCall(endpoint)`, `$('txt-clean').addEventListener`, `$('txt-expand').addEventListener`):

```javascript
async function textCall(endpoint) {
  const text = $('txt-input').value.trim();
  // ... existing body ...
}

$('txt-clean').addEventListener('click', () => textCall('clean'));
$('txt-expand').addEventListener('click', () => textCall('expand'));
```

Replace with:

```javascript
function textBuildPayload(action) {
  const text = $('txt-input').value.trim();
  const context = $('txt-context').value.trim() || null;
  if (!text) return null;
  const base = { text };
  if (context) base.context = context;
  if (action === 'shorten') {
    const t = document.querySelector('input[name="txt-target"]:checked');
    base.target = t ? t.value : '50%';
  }
  if (action === 'caption') {
    const s = document.querySelector('input[name="txt-style"]:checked');
    base.style = s ? s.value : 'instagram';
  }
  if (action === 'keywords') {
    base.max = parseInt($('txt-keywords-max').value, 10) || 10;
  }
  return base;
}

function textRenderResult(action, data) {
  const out = $('txt-output');
  out.classList.remove('empty');
  if (action === 'keywords') {
    // Render as chips
    out.textContent = '';
    const wrap = document.createElement('div');
    wrap.style.cssText = 'display: flex; flex-wrap: wrap; gap: 8px;';
    (data.keywords || []).forEach(function (kw) {
      const chip = document.createElement('span');
      chip.style.cssText = 'display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; background: var(--warm-light); border: 1px solid var(--border); border-radius: 999px; font-size: 13px;';
      chip.textContent = kw.term;
      const score = document.createElement('span');
      score.style.cssText = 'font-size: 11px; color: var(--muted);';
      score.textContent = kw.score;
      chip.appendChild(score);
      wrap.appendChild(chip);
    });
    out.appendChild(wrap);
    return;
  }
  // Plain text result -- pick the right field
  const fieldByAction = {
    clean: 'cleaned',
    expand: 'expanded',
    polish: 'polished',
    shorten: 'shortened',
    caption: 'caption',
  };
  out.textContent = data[fieldByAction[action]];
}

function textTitleFor(action) {
  return {
    clean: 'Cleaned', expand: 'Expanded', polish: 'Polished',
    shorten: 'Shortened', keywords: 'Keywords', caption: 'Caption',
  }[action] || action;
}

async function textRun() {
  const action = $('txt-action').value;
  const payload = textBuildPayload(action);
  if (!payload) return;

  $('txt-error').hidden = true;
  $('txt-result').hidden = true;
  showLoading($('txt-loading'), 'Running ' + action + '...');
  $('txt-run').disabled = true;

  try {
    const r = await fetch(TEXT_API + '/' + action, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) {
      const j = await r.json().catch(() => null);
      throw new Error((j && j.detail) || 'HTTP ' + r.status);
    }
    const data = await r.json();
    $('txt-result-title').textContent = textTitleFor(action);
    textRenderResult(action, data);
    $('txt-result').hidden = false;
  } catch (e) {
    $('txt-error').textContent = 'Error: ' + e.message;
    $('txt-error').hidden = false;
  } finally {
    $('txt-loading').hidden = true;
    $('txt-run').disabled = false;
  }
}

function textUpdateSubControls() {
  const action = $('txt-action').value;
  $('txt-sub-shorten').hidden = action !== 'shorten';
  $('txt-sub-caption').hidden = action !== 'caption';
  $('txt-sub-keywords').hidden = action !== 'keywords';
}

$('txt-run').addEventListener('click', textRun);
$('txt-action').addEventListener('change', textUpdateSubControls);
textUpdateSubControls();
```

- [ ] **Step 3: Manual smoke test (requires running server)**

Start the text server in another terminal:

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand
PYTHONPATH=../.. .venv/bin/python3 -m uvicorn main:app --port 8001 --reload
```

Open <http://localhost:8001/>, on the Text tab try each action:
- Type some messy text
- Run each action from the dropdown
- For shorten/caption verify the target/style radios show up
- For keywords verify chips render with score badges

If any action errors, check browser dev console and fix the JS.

- [ ] **Step 4: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 5: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/text-clean-expand/static/index.html
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): rework Text tab UI with action dropdown for 6 endpoints"
```

---

## Task A7: README + final smoke + push + PR

**Files:**
- Modify: `poc/text-clean-expand/README.md` (if it exists; otherwise create)

- [ ] **Step 1: Update README**

Check if `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/README.md` exists. If yes, append the new endpoints section. If not, create one with this content:

```markdown
# Text Clean & Expand POC

FastAPI service that transforms a user's free-form text using an
OpenAI-compatible LLM. Six endpoints, each with its own prompt:

| Endpoint | Purpose |
|---|---|
| `POST /clean` | Restructure messy input into a readable note |
| `POST /expand` | Develop the user's input as if they wrote a longer version |
| `POST /polish` | Final editing pass -- word choice + flow, no restructuring |
| `POST /shorten` | Produce a shorter version. `target`: 30% / 50% / tweet |
| `POST /keywords` | Extract 5-10 keywords with salience scores |
| `POST /caption` | 1-2 sentence caption. `style`: instagram / diary / tweet |

Bilingual: every endpoint preserves the input language (English in -> English out, any Chinese in -> Chinese out).

## Setup

```bash
cd poc/text-clean-expand
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env  # fill in LLM_API_KEY
```

## Run

```bash
PYTHONPATH=../.. .venv/bin/python3 -m uvicorn main:app --port 8001 --reload
```

UI at <http://localhost:8001/>. Or use `./start-all.sh` at the repo root to bring up all three POCs.

## Curl examples

```bash
curl -s -X POST http://localhost:8001/polish \
  -H 'Content-Type: application/json' \
  -d '{"text":"morning light warm. coffee strong. happy."}' | python3 -m json.tool

curl -s -X POST http://localhost:8001/shorten \
  -H 'Content-Type: application/json' \
  -d '{"text":"long text here","target":"tweet"}' | python3 -m json.tool

curl -s -X POST http://localhost:8001/keywords \
  -H 'Content-Type: application/json' \
  -d '{"text":"We shipped voice transcription beta to 10 partners.","max":5}' | python3 -m json.tool

curl -s -X POST http://localhost:8001/caption \
  -H 'Content-Type: application/json' \
  -d '{"text":"Long journal entry...","style":"diary"}' | python3 -m json.tool
```

## Tests

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand
PYTHONPATH=. .venv/bin/python3 -m pytest tests/ -v
```
```

- [ ] **Step 2: Run full test suite**

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand
PYTHONPATH=. .venv/bin/python3 -m pytest tests/ -v
```

Expected: 12 passed.

- [ ] **Step 3: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 4: Manual end-to-end smoke against real LLM**

If LLM_API_KEY in `.env` is real (DeepSeek), test each new endpoint via curl as in the README. Verify outputs look reasonable for both English and Chinese inputs.

- [ ] **Step 5: Commit README**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/text-clean-expand/README.md
git -C /Users/mengjia/MiraNote/miranote-api commit -m "docs(api): document text-clean-expand 6-endpoint surface"
```

- [ ] **Step 6: Push and open PR**

```bash
git -C /Users/mengjia/MiraNote/miranote-api push -u origin feat/api-text-features
cd /Users/mengjia/MiraNote/miranote-api
gh pr create --title "feat(api): expand text-clean-expand with polish, shorten, keywords, caption" --body "$(cat <<'EOF'
## Summary

Adds 4 new endpoints to `poc/text-clean-expand/` (polish, shorten, keywords, caption) alongside existing clean/expand. Each endpoint:
- Has its own prompt file under `prompts/`
- Has unit tests with a `FakeOpenAI` fixture (TestClient + stubbed LLM)
- Preserves the existing bilingual rule (input language == output language)

UI in the Text tab swaps from 2 buttons to a single Action dropdown + Run button with conditional sub-controls (shorten target, caption style, keywords max). Keywords result renders as chips with score badges.

## Spec

`docs/specs/2026-06-05-text-features-and-voice-sentiment-design.md` (Phase A)

## Test plan

- [x] `pytest tests/ -v` -- 12 passed (happy path + edge case per endpoint)
- [x] Rule 3 (`no_cjk_or_emoji`) -- exit 0
- [x] Manual: each action runs end-to-end via UI against real DeepSeek, English + Chinese inputs both produce sensible output

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 7: Note the PR URL**

`gh pr create` prints the PR URL. Phase A is done. Open for Jason review. Do not admin-merge.

---

# Phase B — voice-to-text

## Task B0: Branch + deps

**Files:**
- Modify: `poc/voice-to-text/requirements.txt`

- [ ] **Step 1: Branch off main**

```bash
git -C /Users/mengjia/MiraNote/miranote-api checkout main
git -C /Users/mengjia/MiraNote/miranote-api pull --ff-only
git -C /Users/mengjia/MiraNote/miranote-api checkout -b feat/api-voice-acoustic-sentiment
```

- [ ] **Step 2: Add transformers + pytest to requirements**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/requirements.txt`. Append:

```
transformers>=4.40
pytest>=8.0
```

- [ ] **Step 3: Install in venv**

```bash
/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/.venv/bin/pip install -q transformers pytest
```

Expected: torch already installed (Whisper dep); transformers downloads ~30 MB; pytest ~5 MB. No model download yet (lazy).

- [ ] **Step 4: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/voice-to-text/requirements.txt
git -C /Users/mengjia/MiraNote/miranote-api commit -m "chore(api): add transformers + pytest to voice-to-text deps"
```

---

## Task B1: Lazy-load Whisper + test infrastructure

**Files:**
- Modify: `poc/voice-to-text/main.py` (refactor Whisper to lazy load)
- Create: `poc/voice-to-text/tests/__init__.py`
- Create: `poc/voice-to-text/tests/conftest.py`

The current main.py loads Whisper at import time. That blocks tests. Refactor to lazy.

- [ ] **Step 1: Refactor Whisper loading in main.py**

Find this block in `/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/main.py`:

```python
# ---------- Load models ----------
print(f"Loading Whisper model: {WHISPER_MODEL} ...")
model = whisper.load_model(WHISPER_MODEL)
print("Whisper model loaded.")
```

Replace with:

```python
# ---------- Lazy model loading ----------
_whisper_model = None


def get_whisper_model():
    """Lazy-load Whisper on first call so import is cheap (tests, /health)."""
    global _whisper_model
    if _whisper_model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _whisper_model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _whisper_model
```

Then find the line inside the `/transcribe` handler:

```python
result = await asyncio.to_thread(
    model.transcribe, tmp_path, language=lang, verbose=False
)
```

Change `model.transcribe` to `get_whisper_model().transcribe`:

```python
result = await asyncio.to_thread(
    get_whisper_model().transcribe, tmp_path, language=lang, verbose=False
)
```

- [ ] **Step 2: Create empty test package**

```bash
mkdir -p /Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/tests
touch /Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/tests/__init__.py
```

- [ ] **Step 3: Write conftest with stubs**

Create `/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/tests/conftest.py`:

```python
from __future__ import annotations
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest


# Default scripted Whisper result; tests can override via stub_whisper.set()
class _WhisperStub:
    def __init__(self):
        self.result: Dict[str, Any] = {
            "text": "hello world",
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.5, "text": "hello world"}],
        }

    def transcribe(self, path, **kwargs):
        return self.result


class _EmotionStub:
    def __init__(self):
        self.result: Dict[str, Any] = {
            "label": "happy",
            "confidence": 0.83,
            "all_scores": [
                {"label": "happy", "score": 0.83},
                {"label": "neutral", "score": 0.10},
                {"label": "sad", "score": 0.05},
                {"label": "angry", "score": 0.02},
            ],
        }

    def __call__(self, path):
        return self.result


@pytest.fixture
def stub_whisper(monkeypatch):
    """Replace get_whisper_model() to return a scripted stub."""
    stub = _WhisperStub()
    # Stub is imported below, after main loads
    yield stub
    # Cleanup not needed; monkeypatch reverts


@pytest.fixture
def stub_emotion(monkeypatch):
    """Replace emotion.analyze_emotion with a scripted stub."""
    stub = _EmotionStub()
    yield stub


@pytest.fixture
def voice_client(stub_whisper, stub_emotion, monkeypatch):
    """FastAPI TestClient with Whisper and emotion both stubbed."""
    os.environ.setdefault("LLM_API_KEY", "fake")
    os.environ.setdefault("WHISPER_MODEL", "tiny")

    main_path = Path(__file__).parent.parent / "main.py"
    spec = importlib.util.spec_from_file_location("voice_to_text_main", main_path)
    main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main)

    # Inject stubs after import
    monkeypatch.setattr(main, "get_whisper_model", lambda: stub_whisper)
    # Import emotion module via the same trick
    emotion_path = Path(__file__).parent.parent / "emotion.py"
    if emotion_path.exists():
        spec_e = importlib.util.spec_from_file_location("voice_to_text_emotion", emotion_path)
        emotion = importlib.util.module_from_spec(spec_e)
        spec_e.loader.exec_module(emotion)
        monkeypatch.setattr(main, "analyze_emotion_fn", stub_emotion, raising=False)
        monkeypatch.setattr(emotion, "analyze_emotion", stub_emotion)

    from fastapi.testclient import TestClient
    return TestClient(main.app), stub_whisper, stub_emotion
```

- [ ] **Step 4: Write a smoke test**

Create `/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/tests/test_smoke.py`:

```python
def test_health_endpoint(voice_client):
    test_client, _, _ = voice_client
    r = test_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
```

- [ ] **Step 5: Run smoke test**

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/voice-to-text
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_smoke.py -v
```

Expected: 1 passed. If FAIL, fix the conftest import block.

- [ ] **Step 6: Delete smoke test**

```bash
rm /Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/tests/test_smoke.py
```

- [ ] **Step 7: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/voice-to-text/main.py poc/voice-to-text/tests/
git -C /Users/mengjia/MiraNote/miranote-api commit -m "test(api): lazy-load Whisper + scaffold voice-to-text tests"
```

---

## Task B2: `emotion.py` module (TDD)

**Files:**
- Create: `poc/voice-to-text/emotion.py`
- Create: `poc/voice-to-text/tests/test_emotion_module.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/tests/test_emotion_module.py`:

```python
from __future__ import annotations
import importlib.util
from pathlib import Path

import pytest


def _load_emotion():
    p = Path(__file__).parent.parent / "emotion.py"
    spec = importlib.util.spec_from_file_location("voice_to_text_emotion_isolated", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_analyze_emotion_returns_expected_shape(monkeypatch):
    em = _load_emotion()

    # Stub the underlying pipeline to return a known raw scores list
    def fake_pipeline_factory(*args, **kwargs):
        def fake_pipe(audio_path, top_k=None):
            return [
                {"label": "happy", "score": 0.83},
                {"label": "neutral", "score": 0.10},
                {"label": "sad", "score": 0.05},
                {"label": "angry", "score": 0.02},
            ]
        return fake_pipe

    monkeypatch.setattr(em, "_PIPELINE", None)
    monkeypatch.setattr("transformers.pipeline", fake_pipeline_factory)

    out = em.analyze_emotion("/fake/path.wav")

    assert out["label"] == "happy"
    assert out["confidence"] == 0.83
    assert len(out["all_scores"]) == 4
    assert out["all_scores"][0]["label"] == "happy"
    assert out["all_scores"][-1]["label"] == "angry"  # sorted descending


def test_analyze_emotion_sorts_unsorted_input(monkeypatch):
    em = _load_emotion()

    def fake_pipeline_factory(*args, **kwargs):
        def fake_pipe(audio_path, top_k=None):
            return [
                {"label": "sad", "score": 0.1},
                {"label": "happy", "score": 0.7},
                {"label": "angry", "score": 0.2},
            ]
        return fake_pipe

    monkeypatch.setattr(em, "_PIPELINE", None)
    monkeypatch.setattr("transformers.pipeline", fake_pipeline_factory)

    out = em.analyze_emotion("/fake.wav")
    assert out["label"] == "happy"
    assert [s["label"] for s in out["all_scores"]] == ["happy", "angry", "sad"]


def test_analyze_emotion_pipeline_caches(monkeypatch):
    em = _load_emotion()
    monkeypatch.setattr(em, "_PIPELINE", None)
    call_count = {"n": 0}

    def fake_pipeline_factory(*args, **kwargs):
        call_count["n"] += 1
        def fake_pipe(p, top_k=None):
            return [{"label": "happy", "score": 1.0}]
        return fake_pipe

    monkeypatch.setattr("transformers.pipeline", fake_pipeline_factory)

    em.analyze_emotion("/a.wav")
    em.analyze_emotion("/b.wav")
    em.analyze_emotion("/c.wav")
    assert call_count["n"] == 1, "pipeline factory should only run once (cached)"
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/voice-to-text
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_emotion_module.py -v
```

Expected: FAIL — `emotion.py` doesn't exist yet.

- [ ] **Step 3: Write `emotion.py`**

Create `/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/emotion.py`:

```python
"""Acoustic speech emotion classifier wrapping a HuggingFace pipeline.

Default model: hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0
- Backbone: facebook/wav2vec2-large-xlsr-53 (53-language pretrained)
- Fine-tuned on: CREMA + RAVDESS + SAVEE + TESS (English)
- 7 classes: angry, disgust, fear, happy, neutral, sad, surprise
- Cross-lingual claim (Chinese, French) is empirical from the model author,
  not a benchmark; treat confidence as advisory on non-English audio.

Model lazily loads on first analyze_emotion() call. ~1.3 GB download to
~/.cache/huggingface/ on first use. Subsequent calls reuse the cached
in-memory pipeline.
"""
from __future__ import annotations

import os
from threading import Lock
from typing import Any, Dict

_PIPELINE = None
_LOCK = Lock()
_MODEL = os.getenv(
    "EMOTION_MODEL",
    "hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0",
)


def _get_pipeline():
    """Lazy-load and cache the HuggingFace audio-classification pipeline."""
    global _PIPELINE
    if _PIPELINE is None:
        with _LOCK:
            if _PIPELINE is None:
                from transformers import pipeline
                _PIPELINE = pipeline("audio-classification", model=_MODEL)
    return _PIPELINE


def analyze_emotion(audio_path: str) -> Dict[str, Any]:
    """Classify the speaker's emotion from an audio file.

    Returns:
        {
            "label": "<top emotion>",
            "confidence": <float, 0-1, the top score rounded to 3 dp>,
            "all_scores": [{"label": "...", "score": <float>}, ...] sorted descending
        }
    """
    pipe = _get_pipeline()
    raw = pipe(audio_path, top_k=None)
    raw_sorted = sorted(raw, key=lambda r: r["score"], reverse=True)
    return {
        "label": raw_sorted[0]["label"],
        "confidence": round(float(raw_sorted[0]["score"]), 3),
        "all_scores": [
            {"label": r["label"], "score": round(float(r["score"]), 3)}
            for r in raw_sorted
        ],
    }
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/voice-to-text
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_emotion_module.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 6: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/voice-to-text/emotion.py poc/voice-to-text/tests/test_emotion_module.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add acoustic emotion classifier using XLSR-53 SER model"
```

---

## Task B3: Wire emotion into `/transcribe` (TDD)

**Files:**
- Modify: `poc/voice-to-text/main.py`
- Create: `poc/voice-to-text/tests/test_transcribe.py`

- [ ] **Step 1: Write the failing tests**

Create `/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/tests/test_transcribe.py`:

```python
from __future__ import annotations
import io


def _fake_audio_bytes(size: int = 4096) -> bytes:
    # Just opaque bytes; Whisper is stubbed so format doesn't matter.
    return b"\x00\x01" * (size // 2)


def test_transcribe_with_emotion_default_true(voice_client):
    test_client, _, _ = voice_client
    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    r = test_client.post("/transcribe", files=files, params={"correct": "false"})
    assert r.status_code == 200
    body = r.json()
    assert body["raw_text"] == "hello world"
    assert "emotion" in body
    assert body["emotion"]["label"] == "happy"
    assert body["emotion"]["confidence"] == 0.83
    assert body.get("emotion_status") == "ok"


def test_transcribe_with_emotion_false_omits_emotion(voice_client):
    test_client, _, _ = voice_client
    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    r = test_client.post(
        "/transcribe",
        files=files,
        params={"correct": "false", "with_emotion": "false"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("emotion") is None
    assert body.get("emotion_status") in (None, "skipped")


def test_transcribe_emotion_failure_returns_status(voice_client, monkeypatch):
    test_client, _, stub_emotion = voice_client

    def explode(path):
        raise RuntimeError("emotion model failed")

    # Patch the symbol main uses for analyze_emotion
    import voice_to_text_main as main_mod  # spec_from_file_location named it this
    monkeypatch.setattr(main_mod, "analyze_emotion", explode, raising=False)

    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    r = test_client.post("/transcribe", files=files, params={"correct": "false"})
    assert r.status_code == 200
    body = r.json()
    assert body["raw_text"] == "hello world"
    assert body.get("emotion") is None
    assert body.get("emotion_status") == "failed"
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_transcribe.py -v
```

Expected: FAIL — no `emotion` field in response yet.

- [ ] **Step 3: Wire emotion into `/transcribe`**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/main.py`.

Near the top of the file, add to imports:

```python
from emotion import analyze_emotion
```

Find the `/transcribe` handler signature:

```python
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(..., description="Audio file (mp3, wav, flac, m4a, ogg, webm)"),
    correct: bool = Query(True, description="Apply AI correction after Whisper transcription"),
    lang: Literal["zh", "en"] = Query(
        "zh",
        description=(
```

Add a new `with_emotion` query parameter. Insert after the `lang` parameter (still inside the function signature, before the closing `):`):

```python
    with_emotion: bool = Query(
        True,
        description="Run acoustic emotion classifier on the audio (adds ~1 sec).",
    ),
```

Then find the return statement at the end of the handler:

```python
        return {
            "language": language,
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "correction_status": correction_status,
            "segments": segments,
        }
```

Replace with (note the new emotion handling block before the return):

```python
        emotion_result: Optional[Dict[str, Any]] = None
        emotion_status = "skipped"
        if with_emotion:
            try:
                emotion_result = await asyncio.to_thread(analyze_emotion, tmp_path)
                emotion_status = "ok"
            except Exception as e:  # noqa: BLE001  -- surface to caller as status
                print(f"Emotion analysis failed: {e}")
                emotion_status = "failed"

        return {
            "language": language,
            "raw_text": raw_text,
            "corrected_text": corrected_text,
            "correction_status": correction_status,
            "segments": segments,
            "emotion": emotion_result,
            "emotion_status": emotion_status,
        }
```

Make sure `Optional` and `Dict, Any` are imported at the top. Add to imports if missing:

```python
from typing import Any, Dict, Literal, Optional, Tuple
```

- [ ] **Step 4: Run tests to verify pass**

```bash
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_transcribe.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 6: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/voice-to-text/main.py poc/voice-to-text/tests/test_transcribe.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): include acoustic emotion in /transcribe response"
```

---

## Task B4: Standalone `/emotion` endpoint (TDD)

**Files:**
- Modify: `poc/voice-to-text/main.py`
- Create: `poc/voice-to-text/tests/test_emotion_endpoint.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/tests/test_emotion_endpoint.py`:

```python
from __future__ import annotations
import io


def _fake_audio_bytes(size: int = 4096) -> bytes:
    return b"\x00\x01" * (size // 2)


def test_emotion_endpoint_returns_shape(voice_client):
    test_client, _, _ = voice_client
    files = {"file": ("clip.wav", io.BytesIO(_fake_audio_bytes()), "audio/wav")}
    r = test_client.post("/emotion", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] == "happy"
    assert body["confidence"] == 0.83
    assert len(body["all_scores"]) == 4


def test_emotion_endpoint_rejects_tiny_file(voice_client):
    test_client, _, _ = voice_client
    files = {"file": ("clip.wav", io.BytesIO(b"\x00"), "audio/wav")}
    r = test_client.post("/emotion", files=files)
    assert r.status_code == 422
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=. .venv/bin/python3 -m pytest tests/test_emotion_endpoint.py -v
```

Expected: FAIL — endpoint doesn't exist.

- [ ] **Step 3: Add the endpoint**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/main.py`. After the existing `/transcribe` handler, add:

```python
@app.post("/emotion")
async def emotion_endpoint(
    file: UploadFile = File(..., description="Audio file"),
):
    """Run only the acoustic emotion classifier on an uploaded audio file."""
    raw_bytes = await file.read()
    if len(raw_bytes) < 1024:
        raise HTTPException(
            status_code=422,
            detail=f"Audio too small ({len(raw_bytes)} bytes). Record at least 1 second.",
        )
    suffix = os.path.splitext(file.filename or "audio.wav")[1]
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name
    try:
        return await asyncio.to_thread(analyze_emotion, tmp_path)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Emotion analysis failed: {e}")
    finally:
        os.unlink(tmp_path)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
PYTHONPATH=. .venv/bin/python3 -m pytest tests/ -v
```

Expected: all tests pass (3 module + 3 transcribe + 2 emotion endpoint = 8).

- [ ] **Step 5: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 6: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/voice-to-text/main.py poc/voice-to-text/tests/test_emotion_endpoint.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add standalone /emotion endpoint for audio-only sentiment"
```

---

## Task B5: CORS fix

**Files:**
- Modify: `poc/voice-to-text/main.py`

- [ ] **Step 1: Add CORSMiddleware**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/main.py`. After the line `from fastapi import FastAPI, UploadFile, File, Query, HTTPException`, add:

```python
from fastapi.middleware.cors import CORSMiddleware
```

Then after the line `app = FastAPI(title="MiraNote Voice-to-Text", version="0.1.0")`, add:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
```

- [ ] **Step 2: Verify tests still pass**

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/voice-to-text
PYTHONPATH=. .venv/bin/python3 -m pytest tests/ -q
```

Expected: 8 passed.

- [ ] **Step 3: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/voice-to-text/main.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "fix(api): add CORS middleware to voice-to-text for cross-origin UI"
```

---

## Task B6: UI emotion badge in Voice tab

**Files:**
- Modify: `poc/text-clean-expand/static/index.html`

The unified UI lives in text-clean-expand's static. Add a third badge to the Voice tab result area showing emotion + confidence.

- [ ] **Step 1: Add emotion badge rendering**

Open `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/static/index.html`. Find the `renderVoiceResult(data)` function. Locate this block:

```javascript
function renderVoiceResult(data) {
  // Badges
  const badges = $('v-badges');
  clearChildren(badges);
  badges.appendChild(makeBadge('lang: ' + (data.language || '?')));
  badges.appendChild(makeBadge('correction: ' + (data.corrected_text ? 'ok' : 'skipped'), data.corrected_text ? 'ok' : ''));
```

Insert after the `correction:` badge append, before the `// Raw text` comment:

```javascript
  if (data.emotion) {
    const pct = Math.round((data.emotion.confidence || 0) * 100);
    const emoBadge = makeBadge('emotion: ' + data.emotion.label + ' ' + pct + '%', 'ok');
    emoBadge.style.cursor = 'pointer';
    emoBadge.title = 'Click for full distribution';
    const detail = document.createElement('div');
    detail.style.cssText = 'display:none; margin-top:6px; padding:8px 12px; background:var(--warm-light); border:1px solid var(--border); border-radius:8px; font-size:12px;';
    (data.emotion.all_scores || []).forEach(function (s) {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex; justify-content:space-between; gap:12px;';
      const lbl = document.createElement('span'); lbl.textContent = s.label;
      const sc = document.createElement('span'); sc.textContent = (s.score * 100).toFixed(1) + '%';
      sc.style.color = 'var(--muted)';
      row.appendChild(lbl); row.appendChild(sc);
      detail.appendChild(row);
    });
    emoBadge.addEventListener('click', function () {
      detail.style.display = detail.style.display === 'none' ? 'block' : 'none';
    });
    badges.appendChild(emoBadge);
    badges.appendChild(detail);
  } else if (data.emotion_status === 'failed') {
    badges.appendChild(makeBadge('emotion: failed', ''));
  }
```

- [ ] **Step 2: Manual smoke**

(Requires both voice server on 8000 + text server on 8001.)

```bash
# Terminal 1
cd /Users/mengjia/MiraNote/miranote-api/poc/voice-to-text
PYTHONPATH=. .venv/bin/python3 -m uvicorn main:app --port 8000 --reload

# Terminal 2
cd /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand
PYTHONPATH=../.. .venv/bin/python3 -m uvicorn main:app --port 8001 --reload
```

Open <http://localhost:8001/>, Voice tab, upload one of `poc/voice-to-text/demo_data/*.m4a` and verify:
- First request takes ~30 sec (model download)
- Subsequent requests show `emotion: <label> <pct>%` badge alongside `lang:` and `correction:`
- Click the emotion badge -> 7-row distribution table appears below

- [ ] **Step 3: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/text-clean-expand/static/index.html
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add emotion badge to Voice tab with expandable distribution"
```

---

## Task B7: README + final smoke + push + PR

**Files:**
- Modify: `poc/voice-to-text/README.md` (if exists) or create

- [ ] **Step 1: Update README**

Locate `/Users/mengjia/MiraNote/miranote-api/poc/voice-to-text/README.md`. Append (or include in a new file) a section:

```markdown
## Acoustic emotion analysis

`/transcribe?with_emotion=true` (default true) and the new `/emotion`
endpoint run a HuggingFace audio-classification model on the uploaded
audio and return a 7-class emotion label (angry, disgust, fear, happy,
neutral, sad, surprise) plus per-class scores.

**Model:** `hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0`
- ~1.3 GB, auto-downloads to `~/.cache/huggingface/` on first emotion request (one-time, ~30 sec on first call)
- XLSR-53 multilingual backbone; trained on English emotion corpora but the author reports cross-lingual generalisation to Chinese and French (empirical claim, not benchmarked)

**Override the model** via env var `EMOTION_MODEL=<repo-id>`.

**Curl:**

```bash
# Bundled with transcription
curl -s -X POST "http://localhost:8000/transcribe?correct=true&with_emotion=true&lang=en" \
  -F file=@demo_data/en_short.m4a | python3 -m json.tool

# Standalone
curl -s -X POST http://localhost:8000/emotion -F file=@demo_data/en_short.m4a | python3 -m json.tool
```

**Response shape (transcribe with emotion):**

```json
{
  "language": "en", "raw_text": "...", "corrected_text": "...",
  "correction_status": "ok",
  "segments": [...],
  "emotion": {
    "label": "happy", "confidence": 0.83,
    "all_scores": [{"label": "happy", "score": 0.83}, ...]
  },
  "emotion_status": "ok"
}
```

`emotion` is `null` and `emotion_status` is `"failed"` if classification raised; `null` and `"skipped"` if `with_emotion=false`.

## Tests

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/voice-to-text
PYTHONPATH=. .venv/bin/python3 -m pytest tests/ -v
```
```

- [ ] **Step 2: Run full test suite**

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/voice-to-text
PYTHONPATH=. .venv/bin/python3 -m pytest tests/ -v
```

Expected: 8 passed.

- [ ] **Step 3: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 4: Manual end-to-end smoke**

Restart voice server (port 8000) and use the unified UI Voice tab to upload `demo_data/en_short.m4a` and `demo_data/zh_meeting.m4a`. Verify:
- English clip emotion makes some sense (likely neutral or happy)
- Chinese clip emotion populates (even if accuracy is questionable, the badge should render)
- Click badge -> all 7 scores show

- [ ] **Step 5: Commit README**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/voice-to-text/README.md
git -C /Users/mengjia/MiraNote/miranote-api commit -m "docs(api): document voice acoustic emotion endpoints"
```

- [ ] **Step 6: Push and open PR**

```bash
git -C /Users/mengjia/MiraNote/miranote-api push -u origin feat/api-voice-acoustic-sentiment
cd /Users/mengjia/MiraNote/miranote-api
gh pr create --title "feat(api): add acoustic emotion analysis to voice-to-text" --body "$(cat <<'EOF'
## Summary

Adds acoustic emotion classification to `poc/voice-to-text/` using a HuggingFace audio-classification pipeline (`hughlan1214/...-xlsr-53-SER-fine-tuned2.0`). 7-class output (angry/disgust/fear/happy/neutral/sad/surprise) with cross-lingual generalisation to Chinese and French (per the model author's post-release testing).

- New module `emotion.py`: lazy-loaded pipeline, ~1.3 GB model auto-downloads to `~/.cache/huggingface/` on first call
- `POST /transcribe?with_emotion=true` (default true) -- response gains `emotion` and `emotion_status` fields
- New `POST /emotion` endpoint for audio-only classification
- CORS middleware added (was missing; cross-origin from unified UI broke before)
- Voice tab UI gains an `emotion: <label> <pct>%` badge; click expands to a 7-row distribution
- Whisper refactored to lazy load so tests don't block on model load

## Spec

`docs/specs/2026-06-05-text-features-and-voice-sentiment-design.md` (Phase B)

## Test plan

- [x] `pytest tests/ -v` -- 8 passed (module + endpoint + failure path)
- [x] Rule 3 (`no_cjk_or_emoji`) -- exit 0
- [x] Manual: English + Chinese audio both produce emotion badges in UI
- [x] First request triggers ~30 sec model download, subsequent are ~1 sec

## Notable trade-off

Cross-lingual emotion accuracy on Chinese is empirical (the model author's report), not benchmarked. If results disappoint in practice we'll do a hybrid (acoustic + LLM-on-transcript) in a follow-up. The 7-class output and `all_scores` distribution let users see when the model is uncertain.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 7: Note PR URL**

Phase B done. Open for Jason review. Do NOT admin-merge.

---

# Phase C — chatbot text tools

**Branch strategy:** branch off Phase A's tip locally so smoke testing has the new text endpoints to hit. After Phase A merges to main, rebase this branch onto main and force-push.

## Task C0: Branch (from A tip) + deps + env

**Files:**
- Modify: `poc/chatbot/requirements.txt`
- Modify: `poc/chatbot/.env.example`

- [ ] **Step 1: Branch from Phase A's tip**

```bash
git -C /Users/mengjia/MiraNote/miranote-api fetch origin
git -C /Users/mengjia/MiraNote/miranote-api checkout feat/api-text-features
git -C /Users/mengjia/MiraNote/miranote-api checkout -b feat/api-chatbot-text-tools
```

- [ ] **Step 2: Add httpx to requirements**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/requirements.txt`. Append:

```
httpx>=0.27
```

- [ ] **Step 3: Add TEXT_API_URL to .env.example**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/.env.example`. Append at the bottom (before the closing newline):

```
# URL of the text-clean-expand POC server. The chatbot delegates its
# text-transformation tools (clean/expand/polish/shorten/keywords/caption)
# to this server over HTTP.
TEXT_API_URL=http://localhost:8001
```

- [ ] **Step 4: Install in venv**

```bash
/Users/mengjia/MiraNote/miranote-api/poc/chatbot/.venv/bin/pip install -q httpx
```

- [ ] **Step 5: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/chatbot/requirements.txt poc/chatbot/.env.example
git -C /Users/mengjia/MiraNote/miranote-api commit -m "chore(api): add httpx + TEXT_API_URL to chatbot"
```

---

## Task C1: `text_client.py` module (TDD)

**Files:**
- Create: `poc/chatbot/text_client.py`
- Create: `poc/chatbot/tests/test_text_client.py`

- [ ] **Step 1: Write the failing test**

Create `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/tests/test_text_client.py`:

```python
from __future__ import annotations
from types import SimpleNamespace

import pytest

from poc.chatbot.text_client import TextClient


@pytest.fixture
def captured_post(monkeypatch):
    """Capture httpx.post calls and let the test scripts return shapes."""
    captured = {"url": None, "json": None}
    scripted = {"response": SimpleNamespace(status_code=200, json=lambda: {"ok": True})}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return scripted["response"]

    monkeypatch.setattr("httpx.post", fake_post)
    return captured, scripted


def test_polish_posts_to_polish_url(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001")
    c.polish("hello world")
    assert captured["url"] == "http://localhost:8001/polish"
    assert captured["json"] == {"text": "hello world"}


def test_polish_includes_context_when_given(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001")
    c.polish("hi", context="diary entry")
    assert captured["json"] == {"text": "hi", "context": "diary entry"}


def test_shorten_sends_target(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001")
    c.shorten("long text", target="tweet")
    assert captured["url"].endswith("/shorten")
    assert captured["json"] == {"text": "long text", "target": "tweet"}


def test_keywords_sends_max(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001")
    c.keywords("x", max_hits=5)
    assert captured["json"] == {"text": "x", "max": 5}


def test_caption_sends_style(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001")
    c.caption("entry text", style="diary")
    assert captured["json"] == {"text": "entry text", "style": "diary"}


def test_clean_and_expand_share_shape(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001")
    c.clean("msg")
    assert captured["url"].endswith("/clean")
    c.expand("msg2")
    assert captured["url"].endswith("/expand")


def test_base_url_trailing_slash_stripped(captured_post):
    captured, _ = captured_post
    c = TextClient("http://localhost:8001/")
    c.polish("hi")
    assert captured["url"] == "http://localhost:8001/polish"


def test_non_200_raises_runtimeerror(captured_post):
    captured, scripted = captured_post
    scripted["response"] = SimpleNamespace(
        status_code=502, text="bad", json=lambda: {"detail": "upstream failed"}
    )
    c = TextClient("http://localhost:8001")
    with pytest.raises(RuntimeError, match="502"):
        c.polish("x")


def test_connection_error_raises_runtimeerror(monkeypatch):
    import httpx
    def boom(url, json=None, timeout=None):
        raise httpx.ConnectError("connection refused")
    monkeypatch.setattr("httpx.post", boom)
    c = TextClient("http://localhost:9999")
    with pytest.raises(RuntimeError, match="unreachable"):
        c.polish("x")
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests/test_text_client.py -v
```

Expected: ImportError (module doesn't exist).

- [ ] **Step 3: Write `text_client.py`**

Create `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/text_client.py`:

```python
"""HTTP client for the text-clean-expand POC.

The chatbot's six text-transformation tools (clean, expand, polish, shorten,
keywords, caption) delegate to text-clean-expand over HTTP instead of
duplicating prompts. The text service is the source of truth for prompts;
this client is a thin shim that builds URLs and payloads.

Synchronous (httpx.Client wraps to httpx.post for simplicity). The dispatcher
in tools.py calls into these methods from a sync context inside run_turn's
thread pool, so no event-loop interaction is needed here.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class TextClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._base_url + path
        try:
            resp = httpx.post(url, json=payload, timeout=self._timeout)
        except httpx.RequestError as e:
            raise RuntimeError(
                f"text service unreachable at {self._base_url}: {e}"
            ) from e
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:  # noqa: BLE001
                detail = resp.text
            raise RuntimeError(
                f"text service returned {resp.status_code}: {detail}"
            )
        return resp.json()

    def clean(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text}
        if context:
            payload["context"] = context
        return self._post("/clean", payload)

    def expand(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text}
        if context:
            payload["context"] = context
        return self._post("/expand", payload)

    def polish(self, text: str, context: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text}
        if context:
            payload["context"] = context
        return self._post("/polish", payload)

    def shorten(self, text: str, target: str = "50%") -> Dict[str, Any]:
        return self._post("/shorten", {"text": text, "target": target})

    def keywords(self, text: str, max_hits: int = 10) -> Dict[str, Any]:
        return self._post("/keywords", {"text": text, "max": int(max_hits)})

    def caption(self, text: str, style: str = "instagram") -> Dict[str, Any]:
        return self._post("/caption", {"text": text, "style": style})
```

- [ ] **Step 4: Run tests to verify pass**

```bash
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests/test_text_client.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 6: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/chatbot/text_client.py poc/chatbot/tests/test_text_client.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add chatbot text_client httpx wrapper"
```

---

## Task C2: Extend `ChatbotConfig` to hold the text client

**Files:**
- Modify: `poc/chatbot/config.py`
- Modify: `poc/chatbot/main.py`
- Modify: `poc/chatbot/tests/test_config.py`

- [ ] **Step 1: Add failing test**

Append to `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/tests/test_config.py`:

```python
def test_config_accepts_text_client(tmp_path):
    """Config carries a text_client object the tools dispatcher can use."""
    class FakeClient:
        pass

    cfg = ChatbotConfig(
        docs_root=tmp_path,
        model="fake",
        max_tool_iterations=6,
        max_history_messages=40,
        session_ttl_seconds=3600,
        text_client=FakeClient(),
    )
    assert cfg.text_client is not None
    assert isinstance(cfg.text_client, FakeClient)


def test_config_text_client_defaults_to_none(tmp_path):
    cfg = ChatbotConfig(
        docs_root=tmp_path,
        model="fake",
        max_tool_iterations=6,
        max_history_messages=40,
        session_ttl_seconds=3600,
    )
    assert cfg.text_client is None
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests/test_config.py -v -k text_client
```

Expected: FAIL — `text_client` not a constructor param.

- [ ] **Step 3: Add `text_client` to `ChatbotConfig.__init__`**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/config.py`. Modify `__init__` signature and body:

```python
class ChatbotConfig:
    def __init__(
        self,
        docs_root: Path,
        model: str,
        max_tool_iterations: int,
        max_history_messages: int,
        session_ttl_seconds: int,
        text_client: object = None,
    ):
        self._lock = Lock()
        self.docs_root = docs_root.resolve()
        self.model = model
        self.max_tool_iterations = max_tool_iterations
        self.max_history_messages = max_history_messages
        self.session_ttl_seconds = session_ttl_seconds
        self.text_client = text_client
```

- [ ] **Step 4: Wire in main.py**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/main.py`.

Add to imports:

```python
from poc.chatbot.text_client import TextClient
```

After the existing `config = ChatbotConfig(...)` block, construct the text client and inject:

Find the existing config construction:

```python
config = ChatbotConfig(
    docs_root=_initial_docs_root,
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    max_tool_iterations=int(os.getenv("MAX_TOOL_ITERATIONS", "6")),
    max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "40")),
    session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "3600")),
)
```

Replace with:

```python
TEXT_API_URL = os.getenv("TEXT_API_URL", "http://localhost:8001")

config = ChatbotConfig(
    docs_root=_initial_docs_root,
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    max_tool_iterations=int(os.getenv("MAX_TOOL_ITERATIONS", "6")),
    max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "40")),
    session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "3600")),
    text_client=TextClient(TEXT_API_URL),
)
```

- [ ] **Step 5: Run tests to verify pass**

```bash
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests/test_config.py -v
```

Expected: all config tests pass (previous 5 + 2 new = 7).

- [ ] **Step 6: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/chatbot/config.py poc/chatbot/main.py poc/chatbot/tests/test_config.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): wire TextClient into ChatbotConfig via TEXT_API_URL env"
```

---

## Task C3: Add 6 text tool schemas + dispatcher routing (TDD)

**Files:**
- Modify: `poc/chatbot/tools.py`
- Modify: `poc/chatbot/tests/test_tools.py`

- [ ] **Step 1: Add failing tests**

Append to `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/tests/test_tools.py`:

```python
class _FakeTextClient:
    """Captures all calls; returns scripted dicts."""

    def __init__(self):
        self.calls = []
        self.scripted = {}

    def _record(self, name, kwargs):
        self.calls.append((name, kwargs))
        return self.scripted.get(name, {"ok": True, "name": name})

    def clean(self, text, context=None):
        return self._record("clean", {"text": text, "context": context})

    def expand(self, text, context=None):
        return self._record("expand", {"text": text, "context": context})

    def polish(self, text, context=None):
        return self._record("polish", {"text": text, "context": context})

    def shorten(self, text, target="50%"):
        return self._record("shorten", {"text": text, "target": target})

    def keywords(self, text, max_hits=10):
        return self._record("keywords", {"text": text, "max_hits": max_hits})

    def caption(self, text, style="instagram"):
        return self._record("caption", {"text": text, "style": style})


def _cfg_with_text(tmp_docs):
    cfg = _cfg(tmp_docs)
    cfg.text_client = _FakeTextClient()
    return cfg


def test_tools_schema_now_lists_ten_functions():
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert sorted(names) == sorted([
        "list_docs", "read_doc", "search_docs", "set_docs_root",
        "clean_text", "expand_text", "polish_text", "shorten_text",
        "extract_keywords", "generate_caption",
    ])


def test_dispatch_routes_to_polish_text(tmp_docs):
    cfg = _cfg_with_text(tmp_docs)
    out = tools.dispatch(cfg, "polish_text", {"text": "hi"})
    assert cfg.text_client.calls == [("polish", {"text": "hi", "context": None})]
    assert out == {"ok": True, "name": "polish"}


def test_dispatch_routes_to_shorten_with_target(tmp_docs):
    cfg = _cfg_with_text(tmp_docs)
    tools.dispatch(cfg, "shorten_text", {"text": "long", "target": "tweet"})
    assert cfg.text_client.calls == [("shorten", {"text": "long", "target": "tweet"})]


def test_dispatch_routes_to_extract_keywords(tmp_docs):
    cfg = _cfg_with_text(tmp_docs)
    tools.dispatch(cfg, "extract_keywords", {"text": "x", "max": 3})
    assert cfg.text_client.calls == [("keywords", {"text": "x", "max_hits": 3})]


def test_dispatch_routes_to_generate_caption_with_style(tmp_docs):
    cfg = _cfg_with_text(tmp_docs)
    tools.dispatch(cfg, "generate_caption", {"text": "entry", "style": "diary"})
    assert cfg.text_client.calls == [("caption", {"text": "entry", "style": "diary"})]


def test_dispatch_text_tool_passes_context(tmp_docs):
    cfg = _cfg_with_text(tmp_docs)
    tools.dispatch(cfg, "polish_text", {"text": "hi", "context": "notes"})
    assert cfg.text_client.calls == [("polish", {"text": "hi", "context": "notes"})]


def test_dispatch_text_tool_wraps_runtime_error(tmp_docs):
    cfg = _cfg_with_text(tmp_docs)
    def boom(text, context=None):
        raise RuntimeError("text service down")
    cfg.text_client.polish = boom
    out = tools.dispatch(cfg, "polish_text", {"text": "hi"})
    assert "error" in out
    assert "text service down" in out["error"]
```

- [ ] **Step 2: Run to verify failure**

```bash
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests/test_tools.py -v
```

Expected: many fails (new schemas missing, unknown tool routing).

- [ ] **Step 3: Add 6 tool schemas to `TOOLS`**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/tools.py`. Inside the `TOOLS` list, after the existing `set_docs_root` entry's closing `},`, append:

```python
    {
        "type": "function",
        "function": {
            "name": "clean_text",
            "description": (
                "Restructure messy, fragmented, or stream-of-consciousness text into "
                "a clean readable note. Call when the user asks to 'clean up', "
                "'tidy', or '整理' a piece of text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The messy text to clean."},
                    "context": {"type": "string", "description": "Optional surrounding context."},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "expand_text",
            "description": (
                "Expand brief notes into a fuller version in the user's voice. Call "
                "when the user asks to 'expand', 'develop', '展开', or '扩写' their text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "polish_text",
            "description": (
                "Final editing pass: improve word choice and flow without restructuring "
                "or changing meaning. Call when the user asks to 'polish', 'refine', "
                "'edit', or '润色' their text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shorten_text",
            "description": (
                "Produce a shorter version of the text preserving meaning. Call when "
                "the user asks to 'shorten', 'trim', 'cut', or '缩短' their text. "
                "Target controls aggressiveness: 30%, 50%, or 'tweet' (280 char cap)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "target": {
                        "type": "string",
                        "enum": ["30%", "50%", "tweet"],
                        "default": "50%",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "extract_keywords",
            "description": (
                "Extract 5-10 salient keywords or short phrases. Call when the user "
                "asks for tags, keywords, key terms, or '关键词'. Result is "
                "[{term, score}] where score is 1-10 salience."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "max": {"type": "integer", "default": 10},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_caption",
            "description": (
                "Generate a punchy 1-2 sentence caption. Call when the user asks for "
                "a caption, '配文', summary line, or one-liner. Style controls "
                "register: instagram, diary, or tweet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "style": {
                        "type": "string",
                        "enum": ["instagram", "diary", "tweet"],
                        "default": "instagram",
                    },
                },
                "required": ["text"],
            },
        },
    },
```

- [ ] **Step 4: Add dispatch routing for the 6 names**

In the same file, edit `dispatch()`. Find the existing block:

```python
        if name == "set_docs_root":
            return config.set_docs_root(args["path"])
        return {"error": f"unknown tool: {name}"}
```

Insert before the `return {"error": f"unknown tool: ...` line:

```python
        if name == "clean_text":
            return config.text_client.clean(args["text"], args.get("context"))
        if name == "expand_text":
            return config.text_client.expand(args["text"], args.get("context"))
        if name == "polish_text":
            return config.text_client.polish(args["text"], args.get("context"))
        if name == "shorten_text":
            return config.text_client.shorten(args["text"], args.get("target", "50%"))
        if name == "extract_keywords":
            return config.text_client.keywords(args["text"], int(args.get("max", 10)))
        if name == "generate_caption":
            return config.text_client.caption(args["text"], args.get("style", "instagram"))
```

- [ ] **Step 5: Run tests to verify pass**

```bash
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests/test_tools.py -v
```

Expected: previous tests still pass + 7 new tests pass.

- [ ] **Step 6: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 7: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/chatbot/tools.py poc/chatbot/tests/test_tools.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add 6 chatbot text tools delegating to TextClient"
```

---

## Task C4: Update system prompt

**Files:**
- Modify: `poc/chatbot/prompts/system.txt`

- [ ] **Step 1: Add text-tools paragraph**

Open `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/prompts/system.txt`. Find the existing `set_docs_root` line (last tool described). After it, add a blank line then this paragraph:

```
You also have text-transformation tools (clean_text, expand_text, polish_text,
shorten_text, extract_keywords, generate_caption) for the user's writing. Use
them when the user explicitly asks to transform a piece of text -- they handle
both English and Chinese. shorten_text takes a target ("30%", "50%", "tweet");
generate_caption takes a style ("instagram", "diary", "tweet"); extract_keywords
takes max (default 10). For open-ended questions about the user's documents,
prefer the docs tools (read_doc, search_docs, list_docs) instead.
```

- [ ] **Step 2: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 3: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/chatbot/prompts/system.txt
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): teach chatbot system prompt about text-transformation tools"
```

---

## Task C5: End-to-end smoke against running text + chat servers

This task validates the full chain: user message in chat -> agent picks the right tool -> chatbot HTTP-delegates to text-clean-expand -> response flows back.

- [ ] **Step 1: Start text-clean-expand**

In one terminal:

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand
PYTHONPATH=../.. .venv/bin/python3 -m uvicorn main:app --port 8001 --reload
```

- [ ] **Step 2: Start chatbot**

In another terminal:

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m uvicorn poc.chatbot.main:app --port 8003 --reload
```

- [ ] **Step 3: Verify TEXT_API_URL is exposed**

```bash
curl -s http://127.0.0.1:8003/health | python3 -m json.tool
```

Expected: tools list contains the 10 names including `polish_text`.

- [ ] **Step 4: Chat-driven smoke (curl)**

```bash
curl -s -X POST http://127.0.0.1:8003/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Please polish this text: morning light warm. coffee strong. happy."}' \
  | python3 -m json.tool
```

Expected:
- `reply` contains the polished sentence
- `tool_trace` contains an entry with `name: "polish_text"`
- `result_preview` is the JSON from text-clean-expand's `/polish` response

- [ ] **Step 5: Chinese chat smoke**

```bash
curl -s -X POST http://127.0.0.1:8003/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"帮我从这段话里提取5个关键词：我们上周给10个设计合作伙伴推出了语音转录测试版，反馈说双语支持是杀手级特性。"}' \
  | python3 -m json.tool
```

Expected: `tool_trace` contains `extract_keywords` with `max: 5`, reply lists 5 Chinese keywords.

- [ ] **Step 6: Failure mode smoke (text service down)**

Stop the text-clean-expand server (Ctrl-C in terminal 1). Then:

```bash
curl -s -X POST http://127.0.0.1:8003/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Please polish: hello world"}' \
  | python3 -m json.tool
```

Expected: `tool_trace` contains an error entry. `reply` should say something like "the text service is unavailable" and not crash. Then restart text-clean-expand for the next steps.

- [ ] **Step 7: Stop both servers**

```bash
kill $(lsof -ti:8001,8003) 2>/dev/null
```

- [ ] **Step 8: Run full chatbot test suite once more for confidence**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests -v
```

Expected: all chatbot tests pass (existing 50 + 9 text_client + 2 config + 7 tools dispatch = 68).

- [ ] **Step 9: No code commit for this task** (the smoke is verification only)

---

## Task C6: README update + push + PR

**Files:**
- Modify: `poc/chatbot/README.md`

- [ ] **Step 1: Update README**

Open `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/README.md`. Find the existing "Tools available to the model" section. Replace its table with:

```markdown
## Tools available to the model

| Name | Purpose | Caps |
|---|---|---|
| `list_docs(subdir)` | List files under a subdir of `DOCS_ROOT`. | 200 files |
| `read_doc(path)` | Read a file. Dispatches on extension: text/markdown, PDF, DOCX, image (OCR). | 32 KB truncated |
| `search_docs(query, max_hits)` | Case-insensitive substring search across **UTF-8 text files only** -- PDFs/DOCX/images are invisible to it; use `read_doc` for those. | 200 files, 160-char snippet |
| `set_docs_root(path)` | Switch the docs directory. Only called when the user explicitly asks. Validates that the new path exists and is a directory. | -- |
| `clean_text(text, context?)` | Restructure messy/stream-of-consciousness text. Delegates to text-clean-expand `/clean`. | -- |
| `expand_text(text, context?)` | Expand brief notes in the user's voice. Delegates to `/expand`. | -- |
| `polish_text(text, context?)` | Final editing pass -- word choice + flow. Delegates to `/polish`. | -- |
| `shorten_text(text, target)` | Shorter version. `target` 30%/50%/tweet. Delegates to `/shorten`. | -- |
| `extract_keywords(text, max)` | 5-10 salient keywords with score. Delegates to `/keywords`. | max 20 |
| `generate_caption(text, style)` | 1-2 sentence caption. `style` instagram/diary/tweet. Delegates to `/caption`. | -- |
```

In the "Configuration" section, add:

```markdown
- `TEXT_API_URL` -- URL of the text-clean-expand server. Defaults to
  `http://localhost:8001`. The chatbot's `*_text`, `extract_keywords`,
  and `generate_caption` tools HTTP-delegate to this server. If the
  text server is down, the corresponding tools return a friendly error
  message and the agent recovers.
```

- [ ] **Step 2: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```

Expected: exit=0.

- [ ] **Step 3: Run full test suite one more time**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests -q
```

Expected: 68 passed.

- [ ] **Step 4: Commit README**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/chatbot/README.md
git -C /Users/mengjia/MiraNote/miranote-api commit -m "docs(api): document chatbot text tools and TEXT_API_URL"
```

- [ ] **Step 5: Push and open PR**

```bash
git -C /Users/mengjia/MiraNote/miranote-api push -u origin feat/api-chatbot-text-tools
cd /Users/mengjia/MiraNote/miranote-api
gh pr create --title "feat(api): give chatbot the text-transformation tools" --body "$(cat <<'EOF'
## Summary

Adds 6 new chatbot tools that HTTP-delegate to `text-clean-expand`:
`clean_text`, `expand_text`, `polish_text`, `shorten_text`, `extract_keywords`, `generate_caption`.

Architecture:
- New `text_client.py` -- thin synchronous httpx wrapper, one method per text endpoint
- `ChatbotConfig` gains a `text_client` attribute (constructor injection, defaults to None for backward compat with tests)
- `tools.dispatch()` routes the 6 new names through `config.text_client`
- New env var `TEXT_API_URL` (default `http://localhost:8001`); main.py constructs the client and injects
- System prompt teaches the agent when to call which tool (English + Chinese trigger phrases listed)

Failure mode: if text-clean-expand is down, the tools return `{"error": "text service unreachable: ..."}` and the agent reports back to the user gracefully.

## Dependency on PR A (feat/api-text-features)

This branch is currently based on Phase A's tip. **Before merging this PR, A must merge to main first.** After A merges, I'll rebase this branch onto main and force-push.

## Spec

`docs/specs/2026-06-05-text-features-and-voice-sentiment-design.md` (Phase C)

## Test plan

- [x] `pytest poc/chatbot/tests -v` -- 68 passed (50 existing + 9 text_client + 2 config + 7 tool dispatch)
- [x] Rule 3 (`no_cjk_or_emoji`) -- exit 0
- [x] Manual chat-driven smoke against real DeepSeek + running text server:
      - "Please polish this text: ..." -> agent calls `polish_text`, polished sentence in reply
      - "帮我从这段话里提取5个关键词：..." -> agent calls `extract_keywords` with max=5, Chinese keywords returned
      - With text server down: agent reports error gracefully, does not crash

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 6: After Phase A merges -- rebase onto main**

When Jason approves and merges PR A:

```bash
cd /Users/mengjia/MiraNote/miranote-api
git fetch origin
git checkout feat/api-chatbot-text-tools
git rebase origin/main
# Resolve any conflicts (unlikely -- different POCs touched)
git push --force-with-lease
```

This drops the dependency on the unmerged A tip; the diff vs main now shows only chatbot-specific changes.

- [ ] **Step 7: Note PR URL**

Phase C done. Open for Jason review. Do NOT admin-merge.

---

# Self-review (run after writing this plan)

## Spec coverage

| Spec section | Plan tasks | Notes |
|---|---|---|
| §1 Goal | All phases | ✓ |
| §2 Non-goals | All phases respect | ✓ (no translation, video, real-time canvas, etc.) |
| §3.1 Text endpoints (polish, shorten, keywords, caption) | A2-A5 | ✓ each TDD'd |
| §3.2 Prompt files | A2-A5 (one per endpoint) | ✓ |
| §3.3 UI rework | A6 | ✓ dropdown + Run + conditional sub-controls + chips |
| §4.1 Model choice | B2 (emotion.py with env override) | ✓ |
| §4.2 Module structure | B2 | ✓ lazy load + Lock, get_pipeline + analyze_emotion |
| §4.3 Endpoint integration (with_emotion + /emotion) | B3, B4 | ✓ |
| §4.4 UI badge | B6 | ✓ click-to-expand |
| §4.5 CORS fix | B5 | ✓ |
| §4.6 Model storage docs | B7 (README) | ✓ |
| §5.1 Six new tools | C3 (schemas) + C3 (dispatch routing) | ✓ |
| §5.2 HTTP delegation | C1 (text_client) + C2 (config wiring) | ✓ |
| §5.3 Failure mode | C5 (smoke step 6) + test_dispatch_text_tool_wraps_runtime_error | ✓ |
| §5.4 System prompt update | C4 | ✓ |
| §6.1 PR plan (A, B, C with C off A's tip) | C0 (branch from A) + C6 step 6 (rebase) | ✓ |
| §6.2 Tests strategy (happy + edge per endpoint/tool) | A2-A5 + B3-B4 + C1+C3 | ✓ each has at least one edge case |
| §6.3 Conventions | A0, B0, C0 (deps); commits throughout | ✓ |

No gaps identified.

## Placeholder scan

No "TBD", "TODO", "implement later", or hand-wavy steps. Every code step contains the full code to write. All exact paths. All commands with expected output.

## Type consistency

- `TextClient(base_url, timeout=30.0)` -- consistent in C1 (defined), C2 (constructed in main), C3 (used by dispatch via `config.text_client.<method>`).
- `analyze_emotion(audio_path: str) -> Dict[str, Any]` -- consistent in B2 (defined), B3 (called from /transcribe via `asyncio.to_thread`), B4 (called from /emotion).
- `ChatbotConfig.text_client` -- added in C2, used in C3.
- `get_whisper_model()` -- introduced in B1, used in /transcribe handler.
- Tool names in C3 schemas (clean_text, expand_text, polish_text, shorten_text, extract_keywords, generate_caption) match dispatch branches in C3 routing and tests in C3.

No type inconsistencies found.
