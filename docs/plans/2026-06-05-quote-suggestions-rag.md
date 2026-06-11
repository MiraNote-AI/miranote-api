# Quote suggestions via local RAG -- Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a local RAG-based quote-suggestion feature: new `poc/retrieval/` POC that serves bilingual quote recommendations from a ~1,000-entry curated corpus via BGE-M3 embeddings + sqlite-vec + LLM reranker, plus integration into the Text tab UI and the chatbot's tool registry.

**Architecture:** Two-stage retrieval. Stage 1 (Retriever): user text -> BGE-M3 embedding -> top-K cosine search over sqlite-vec store. Stage 2 (Reranker): top-K candidates + user text -> LLM picks <=N best with one-sentence "why", structured JSON output. Corpus lives as version-controlled JSON, built by an offline `build_corpus.py` script that copies `(text, author, source)` triples from trusted source datasets and lets the LLM only tag themes -- never generate text or authors.

**Tech Stack:** Python 3.9 (compat: `from __future__ import annotations` + `typing.Optional/List/Dict`, NO PEP-604 `X | None`), FastAPI, `sentence-transformers` (BGE-M3 model), `sqlite-vec`, `httpx` (already in chatbot from PR #12), `openai>=1.0` for reranker, `pytest`. Vanilla HTML/CSS/JS for UI.

**Spec:** `/Users/mengjia/MiraNote/miranote-api/docs/specs/2026-06-05-quote-suggestions-rag-design.md`

**Branches:**
- Phase 0: `feat/allowlist-corpus-json` in `MiraNote-AI/.github`
- PR alpha: `feat/api-retrieval-poc` in `MiraNote-AI/miranote-api`
- PR beta: `feat/api-quote-integration` in `MiraNote-AI/miranote-api`

**Phase order:**
- Phase 0 must MERGE before Phase alpha tasks touch corpus JSON (CI would fail Rule 3 otherwise).
- Phase alpha and beta are independent of the open PRs (#10, #11, #12). Goes up in parallel.
- Phase beta branches from main; depends on Phase alpha being open (the smoke test hits retrieval server) but tests stub HTTP so the PR itself can ship.

**Conventions to honor:**
- Python 3.9 compat throughout.
- Rule 3: source ASCII; corpus JSONs allowed CJK only after Phase 0 merges.
- Conventional Commits, scope `api`, subject <=72 chars.
- PR titles self-explanatory (no internal indices).
- **NO admin-bypass** on any merge. Wait for Jason review.
- Rule 6: PR body must include `#<issue>`, a URL, or `spec:`/`design:`/`adr:`/`rfc:` token.

---

## File Structure

### Phase 0 -- `MiraNote-AI/.github`

| File | Change | Responsibility |
|---|---|---|
| `checks/no_cjk_or_emoji.py` | Modify | Append `"poc/*/corpus/*.json"` and `"poc/*/corpus/*.md"` to `ALLOWLIST_PATTERNS` |
| `checks/tests/test_no_cjk_or_emoji.py` | Modify | Pin the new allowlist entries with a test |

### PR alpha -- `poc/retrieval/`

| File | Change | Responsibility |
|---|---|---|
| `__init__.py` | Create | Empty package marker |
| `requirements.txt` | Create | `fastapi`, `uvicorn`, `sentence-transformers>=2.7`, `sqlite-vec>=0.1`, `openai>=1.0`, `python-dotenv`, `pytest>=8.0`, `numpy>=1.24` |
| `.env.example` | Create | `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL`, `CORPUS_DIR`, `INDEX_DB_PATH`, `MAX_PICKS` |
| `.gitignore` | Create | `.env`, `.venv/`, `__pycache__/`, `*.pyc`, `.pytest_cache/`, `data/index.db`, `sources/` |
| `config.py` | Create | Load env, hold paths, expose constants |
| `embedder.py` | Create | BGE-M3 wrapper. Lazy load via Lock. `encode(texts)` and `encode_one(text)` |
| `store.py` | Create | `Store` class wrapping sqlite-vec. `insert(id, payload, vec)`, `search(vec, k)`, `count()` |
| `retriever.py` | Create | `Retriever(store)`. `search(query, k, lang_filter)` -> list of {id, text, author, source, lang, score} |
| `reranker.py` | Create | `rerank(client, model, user_text, candidates, max_picks)` -> [{index, why}]. JSON parse with friendly errors |
| `main.py` | Create | FastAPI: `/health`, `/search`, `/quotes`. CORS allow-all |
| `scripts/__init__.py` | Create | Empty |
| `scripts/build_corpus.py` | Create | CLI: read `sources/*.json` + chinese-poetry git clone -> emit `corpus/quotes_en.json` + `corpus/quotes_zh.json` |
| `scripts/build_index.py` | Create | CLI: read corpus JSONs -> embed all -> write to `data/index.db` |
| `corpus/quotes_en.json` | Create (via script) | ~500 English entries |
| `corpus/quotes_zh.json` | Create (via script) | ~500 Chinese entries |
| `corpus/README.md` | Create | Source datasets, licenses, dates of pull, how to rebuild |
| `data/.gitkeep` | Create | Keep dir; index.db itself is gitignored |
| `tests/__init__.py` | Create | Empty |
| `tests/conftest.py` | Create | Fixtures: in-memory Store, stub embedder, stub LLM client, TestClient |
| `tests/test_embedder.py` | Create | Lazy load, caching, shape |
| `tests/test_store.py` | Create | Insert, search ordering, count |
| `tests/test_retriever.py` | Create | Pipeline composition with stubs |
| `tests/test_reranker.py` | Create | Happy path, malformed JSON, out-of-range index, empty result |
| `tests/test_api.py` | Create | `/health`, `/search`, `/quotes` shapes, lang filtering |
| `tests/test_corpus.py` | Create | Validate JSON against schema, no dupes, themes in taxonomy |
| `README.md` | Create | Setup, run, model download note, curl examples, build_corpus + build_index commands |

### PR beta -- text + chatbot integration

| File | Change | Responsibility |
|---|---|---|
| `poc/chatbot/retrieval_client.py` | Create | Sync `httpx` wrapper for `/quotes`. Mirrors `text_client.py` pattern |
| `poc/chatbot/config.py` | Modify | Add `retrieval_client` attribute to `ChatbotConfig` |
| `poc/chatbot/main.py` | Modify | Construct `RetrievalClient(RETRIEVAL_API_URL)` and inject |
| `poc/chatbot/.env.example` | Modify | Add `RETRIEVAL_API_URL=http://localhost:8004` |
| `poc/chatbot/tools.py` | Modify | Add `find_quote` schema + dispatch routing |
| `poc/chatbot/prompts/tool_descriptions.txt` | Modify | Add `find_quote` row with bilingual triggers |
| `poc/chatbot/prompts/system.txt` | Modify | One-line addition about `find_quote` |
| `poc/chatbot/tests/test_retrieval_client.py` | Create | Stub `httpx.post`, verify URL/payload |
| `poc/chatbot/tests/test_tools.py` | Modify | Add 3 tests: schema includes find_quote, dispatch routes correctly, error wrapping |
| `poc/chatbot/tests/test_config.py` | Modify | Add 1 test: config accepts retrieval_client |
| `poc/chatbot/README.md` | Modify | Document `find_quote` + `RETRIEVAL_API_URL` |
| `poc/text-clean-expand/static/index.html` | Modify | Add `Quote` action to dropdown + sub-controls + result card rendering |
| `poc/text-clean-expand/README.md` | Modify | Note Quote action depends on retrieval server |
| `start-all.sh` | Modify | Add retrieval as 4th service on port 8004 |

---

# Phase 0 -- dotgithub allowlist for corpus JSONs

This phase is in a DIFFERENT REPO (`MiraNote-AI/.github`, locally at `/Users/mengjia/MiraNote/dotgithub`). It must merge before PR alpha can commit any corpus JSON with CJK content.

## Task 0.1: Branch + add allowlist patterns

**Files:**
- Modify: `/Users/mengjia/MiraNote/dotgithub/checks/no_cjk_or_emoji.py`

- [ ] **Step 1: Branch off main in dotgithub**

```bash
git -C /Users/mengjia/MiraNote/dotgithub checkout main
git -C /Users/mengjia/MiraNote/dotgithub pull --ff-only
git -C /Users/mengjia/MiraNote/dotgithub checkout -b feat/allowlist-corpus-json
```

- [ ] **Step 2: Find the ALLOWLIST_PATTERNS list**

```bash
grep -n "ALLOWLIST_PATTERNS\|demo_data" /Users/mengjia/MiraNote/dotgithub/checks/no_cjk_or_emoji.py | head
```
Find the existing block where patterns like `"**/demo_data/*"` are listed.

- [ ] **Step 3: Add the corpus patterns**

In `/Users/mengjia/MiraNote/dotgithub/checks/no_cjk_or_emoji.py`, find the `ALLOWLIST_PATTERNS` list and append two new entries. Pick a sensible spot near `"**/demo_data/*"`:

```python
    # Curated content datasets for retrieval / RAG POCs. Same rationale
    # as demo_data: this is data, not source code, and the CJK content
    # is the whole point (bilingual quotes / poetry).
    "poc/*/corpus/*.json",
    "poc/*/corpus/*.md",
```

- [ ] **Step 4: Verify the file parses**

```bash
python3 -c "import ast; ast.parse(open('/Users/mengjia/MiraNote/dotgithub/checks/no_cjk_or_emoji.py').read()); print('ok')"
```
Expected: `ok`.

## Task 0.2: Pin the allowlist entries with tests

**Files:**
- Modify: `/Users/mengjia/MiraNote/dotgithub/checks/tests/test_no_cjk_or_emoji.py`

- [ ] **Step 1: Look at existing tests to learn the pattern**

```bash
grep -nE "def test_|allow" /Users/mengjia/MiraNote/dotgithub/checks/tests/test_no_cjk_or_emoji.py | head -30
```
Identify how existing allowlist entries are tested (probably setting up a tmp repo with files and asserting `validate(...)` returns no errors for allowlisted paths).

- [ ] **Step 2: Append two new tests**

Append to `/Users/mengjia/MiraNote/dotgithub/checks/tests/test_no_cjk_or_emoji.py`. Adapt to your existing test infrastructure (the helper that creates a fake repo); the assertions to make:

```python
def test_corpus_json_with_cjk_is_allowlisted(tmp_path):
    """poc/*/corpus/*.json files are allowed to contain CJK characters."""
    # Use your existing helper for setting up a fake repo. The shape:
    # 1. Make tmp_path/poc/retrieval/corpus/quotes_zh.json with CJK content.
    # 2. Make a top-level file with the minimal repo skeleton (.git etc).
    # 3. Call validate(tmp_path) (or the equivalent in your test file).
    # 4. Assert no errors are reported for the corpus json path.
    from checks import no_cjk_or_emoji
    corpus_dir = tmp_path / "poc" / "retrieval" / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "quotes_zh.json").write_text(
        '[{"text": "山重水复", "author": "陆游"}]',
        encoding="utf-8",
    )
    # Mirror existing test setup -- this assumes a git-init or similar
    # exists in the helper; check existing test for the right shape
    errors = no_cjk_or_emoji.validate(tmp_path)
    assert not any("corpus/quotes_zh.json" in e for e in errors), (
        f"Corpus JSON should be allowlisted; got errors: {errors}"
    )


def test_corpus_md_with_cjk_is_allowlisted(tmp_path):
    """poc/*/corpus/README.md (and similar) allowed to contain CJK."""
    from checks import no_cjk_or_emoji
    corpus_dir = tmp_path / "poc" / "retrieval" / "corpus"
    corpus_dir.mkdir(parents=True)
    (corpus_dir / "README.md").write_text(
        "Corpus sources: 全唐诗", encoding="utf-8",
    )
    errors = no_cjk_or_emoji.validate(tmp_path)
    assert not any("corpus/README.md" in e for e in errors), (
        f"Corpus README should be allowlisted; got errors: {errors}"
    )
```

**Note:** if your existing tests require additional setup (e.g. `git init`), use the same helper they use. The two tests above show the assertion shape only.

- [ ] **Step 3: Run the dotgithub test suite**

```bash
cd /Users/mengjia/MiraNote/dotgithub
PYTHONPATH=. python3 -m unittest discover checks/tests -v 2>&1 | tail -15
```
Expected: all tests pass (existing + 2 new).

- [ ] **Step 4: Run meta-rule + Rule 3 self-check**

```bash
cd /Users/mengjia/MiraNote/dotgithub
PYTHONPATH=. python3 -m checks._meta.all_rules_have_checks .
PYTHONPATH=. python3 -m checks.no_cjk_or_emoji .
```
Both expected exit 0.

## Task 0.3: Commit + push + open PR

- [ ] **Step 1: Commit**

```bash
git -C /Users/mengjia/MiraNote/dotgithub add checks/
git -C /Users/mengjia/MiraNote/dotgithub commit -m "feat(checks): allow CJK in poc/*/corpus/*.json and *.md"
```

- [ ] **Step 2: Push and open PR**

```bash
git -C /Users/mengjia/MiraNote/dotgithub push -u origin feat/allowlist-corpus-json
cd /Users/mengjia/MiraNote/dotgithub
gh pr create --title "feat(checks): allow CJK in poc/*/corpus/* for RAG POC" --body "$(cat <<'EOF'
## Summary

Widens Rule 3 allowlist for `poc/*/corpus/*.json` and `poc/*/corpus/*.md`. Same rationale as the existing `**/demo_data/*` entry: this is curated content data, not source code, and the CJK content (bilingual quotes / Chinese poetry) is the entire point of the dataset.

Unblocks the upcoming `poc/retrieval/` RAG POC, which lands bilingual quote / poetry JSONs as the source-of-truth corpus.

spec: /Users/mengjia/MiraNote/miranote-api/docs/specs/2026-06-05-quote-suggestions-rag-design.md (Phase 0)

## Test plan

- [x] Two new tests pin the allowlist entries for both `.json` and `.md` paths
- [x] All existing dotgithub tests still pass
- [x] Meta-rule + self-check pass

EOF
)"
```

- [ ] **Step 3: Note PR URL, wait for Jason review + merge**

Phase 0 MUST merge before any Phase alpha task touches corpus JSONs. Do NOT admin-bypass.

---

# Phase alpha -- `poc/retrieval/`

## Task alpha.0: Branch + scaffold POC dir

**Files:**
- Create: `poc/retrieval/__init__.py`
- Create: `poc/retrieval/requirements.txt`
- Create: `poc/retrieval/.env.example`
- Create: `poc/retrieval/.gitignore`

- [ ] **Step 1: Branch off main in miranote-api**

```bash
git -C /Users/mengjia/MiraNote/miranote-api checkout main
git -C /Users/mengjia/MiraNote/miranote-api pull --ff-only
git -C /Users/mengjia/MiraNote/miranote-api checkout -b feat/api-retrieval-poc
```

- [ ] **Step 2: Create directories**

```bash
mkdir -p /Users/mengjia/MiraNote/miranote-api/poc/retrieval/scripts \
         /Users/mengjia/MiraNote/miranote-api/poc/retrieval/corpus \
         /Users/mengjia/MiraNote/miranote-api/poc/retrieval/data \
         /Users/mengjia/MiraNote/miranote-api/poc/retrieval/tests
touch /Users/mengjia/MiraNote/miranote-api/poc/retrieval/__init__.py \
      /Users/mengjia/MiraNote/miranote-api/poc/retrieval/scripts/__init__.py \
      /Users/mengjia/MiraNote/miranote-api/poc/retrieval/tests/__init__.py \
      /Users/mengjia/MiraNote/miranote-api/poc/retrieval/data/.gitkeep
```

- [ ] **Step 3: Write requirements.txt**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/requirements.txt`:

```
fastapi>=0.110
uvicorn>=0.29
openai>=1.0
python-dotenv>=1.0
pytest>=8.0
numpy>=1.24
sentence-transformers>=2.7
sqlite-vec>=0.1.6
```

- [ ] **Step 4: Write .env.example**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/.env.example`:

```
# -- LLM (for the reranker step + the build_corpus.py script) --
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://api.deepseek.com/v1
LLM_MODEL=deepseek-v4-flash

# -- Embedding model --
EMBEDDING_MODEL=BAAI/bge-m3

# -- Paths (relative to poc/retrieval/) --
CORPUS_DIR=./corpus
INDEX_DB_PATH=./data/index.db

# -- Reranker tuning --
MAX_PICKS=3
RETRIEVE_K=10
```

- [ ] **Step 5: Write .gitignore**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/.gitignore`:

```
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
data/index.db
sources/
```

- [ ] **Step 6: Create the venv + install dependencies**

```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/retrieval
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
```

First-time install pulls torch + transformers via sentence-transformers (~2 GB). Expect 1-3 minutes.

- [ ] **Step 7: Verify imports**

```bash
/Users/mengjia/MiraNote/miranote-api/poc/retrieval/.venv/bin/python3 -c "
import fastapi, uvicorn, openai, dotenv, pytest, numpy, sentence_transformers, sqlite_vec
print('all imports ok')
print('sqlite_vec', sqlite_vec.__version__)
print('sentence_transformers', sentence_transformers.__version__)
"
```
Expected: `all imports ok` + versions.

- [ ] **Step 8: Rule 3 check**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```
Expected: exit=0.

- [ ] **Step 9: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/
git -C /Users/mengjia/MiraNote/miranote-api commit -m "chore(api): scaffold poc/retrieval/ with deps"
```

## Task alpha.1: config.py

**Files:**
- Create: `poc/retrieval/config.py`

- [ ] **Step 1: Write config.py**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/config.py`:

```python
"""Runtime config for the retrieval POC.

Reads env vars once at import. POC-only; no live mutation surface
(unlike chatbot/config.py which is mutable).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
CORPUS_DIR = Path(os.getenv("CORPUS_DIR", "./corpus"))
INDEX_DB_PATH = Path(os.getenv("INDEX_DB_PATH", "./data/index.db"))
MAX_PICKS = int(os.getenv("MAX_PICKS", "3"))
RETRIEVE_K = int(os.getenv("RETRIEVE_K", "10"))

# Make paths absolute relative to the package dir (so they work no
# matter the caller's CWD).
_PKG_DIR = Path(__file__).parent
if not CORPUS_DIR.is_absolute():
    CORPUS_DIR = (_PKG_DIR / CORPUS_DIR).resolve()
if not INDEX_DB_PATH.is_absolute():
    INDEX_DB_PATH = (_PKG_DIR / INDEX_DB_PATH).resolve()
```

- [ ] **Step 2: Smoke-import**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -c "
from poc.retrieval import config
print('LLM_MODEL', config.LLM_MODEL)
print('EMBEDDING_MODEL', config.EMBEDDING_MODEL)
print('CORPUS_DIR', config.CORPUS_DIR)
print('INDEX_DB_PATH', config.INDEX_DB_PATH)
"
```
Expected: 4 lines printed; paths absolute and end with `poc/retrieval/corpus` / `poc/retrieval/data/index.db`.

- [ ] **Step 3: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```
Expected: exit=0.

- [ ] **Step 4: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/config.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add retrieval config with env-driven paths"
```

## Task alpha.2: Test infrastructure (conftest)

**Files:**
- Create: `poc/retrieval/tests/conftest.py`

- [ ] **Step 1: Write conftest with shared stubs**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/tests/conftest.py`:

```python
from __future__ import annotations
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest


# --- Embedder stub ---------------------------------------------------

class _StubModel:
    """Drop-in for SentenceTransformer that returns deterministic vectors."""

    def __init__(self, dim: int = 1024):
        self.dim = dim
        self.encode_calls: List[List[str]] = []

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True):
        self.encode_calls.append(list(texts))
        # Deterministic: hash-based, normalised
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = abs(hash(t)) % (2**32 - 1)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-12
            out[i] = v
        return out


@pytest.fixture
def stub_embedder_model(monkeypatch):
    """Replace SentenceTransformer with a stub that returns deterministic vectors."""
    stub = _StubModel()
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        lambda *args, **kwargs: stub,
    )
    # Also reset embedder._MODEL so a fresh load picks up the stub.
    from poc.retrieval import embedder
    monkeypatch.setattr(embedder, "_MODEL", None)
    return stub


# --- Store fixture ---------------------------------------------------

@pytest.fixture
def in_memory_store():
    """sqlite-vec store backed by :memory: -- isolated per test."""
    from poc.retrieval.store import Store
    return Store(db_path=":memory:", dim=1024)


# --- LLM stub --------------------------------------------------------

class _FakeChatCompletions:
    def __init__(self):
        self.scripted: List[str] = []
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.scripted:
            raise AssertionError("FakeChatCompletions: no more scripted responses")
        content = self.scripted.pop(0)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeLLM:
    def __init__(self):
        self.chat = SimpleNamespace(completions=_FakeChatCompletions())

    def reply_with(self, *responses: str):
        self.chat.completions.scripted.extend(responses)


@pytest.fixture
def fake_llm():
    return _FakeLLM()


# --- TestClient ------------------------------------------------------

@pytest.fixture
def api_client(stub_embedder_model, fake_llm, monkeypatch):
    """FastAPI TestClient with all externals stubbed."""
    os.environ.setdefault("LLM_API_KEY", "fake-key-for-tests")
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: fake_llm)

    main_path = Path(__file__).parent.parent / "main.py"
    if "retrieval_main_for_tests" in sys.modules:
        del sys.modules["retrieval_main_for_tests"]
    spec = importlib.util.spec_from_file_location(
        "retrieval_main_for_tests", main_path
    )
    main = importlib.util.module_from_spec(spec)
    sys.modules["retrieval_main_for_tests"] = main
    spec.loader.exec_module(main)

    from fastapi.testclient import TestClient
    return TestClient(main.app), fake_llm, stub_embedder_model
```

- [ ] **Step 2: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```
Expected: exit=0.

- [ ] **Step 3: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/tests/
git -C /Users/mengjia/MiraNote/miranote-api commit -m "test(api): scaffold retrieval tests with stubbed externals"
```

## Task alpha.3: embedder.py (TDD)

**Files:**
- Create: `poc/retrieval/embedder.py`
- Create: `poc/retrieval/tests/test_embedder.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/tests/test_embedder.py`:

```python
from __future__ import annotations
import numpy as np


def test_encode_one_returns_1024d_vector(stub_embedder_model):
    from poc.retrieval import embedder
    v = embedder.encode_one("hello")
    assert v.shape == (1024,)
    assert v.dtype == np.float32


def test_encode_batch_returns_n_by_dim(stub_embedder_model):
    from poc.retrieval import embedder
    v = embedder.encode(["a", "b", "c"])
    assert v.shape == (3, 1024)


def test_embedder_lazy_loads_once(stub_embedder_model):
    """Second call must not re-instantiate the SentenceTransformer."""
    from poc.retrieval import embedder
    embedder.encode_one("a")
    embedder.encode_one("b")
    embedder.encode_one("c")
    # _MODEL is set after first call and not replaced
    assert embedder._MODEL is stub_embedder_model
    # Stub recorded all three encode invocations
    assert len(stub_embedder_model.encode_calls) == 3


def test_encoded_vectors_are_normalised(stub_embedder_model):
    from poc.retrieval import embedder
    v = embedder.encode_one("anything")
    assert abs(np.linalg.norm(v) - 1.0) < 1e-5
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_embedder.py -v
```
Expected: ImportError (embedder doesn't exist yet).

- [ ] **Step 3: Implement embedder.py**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/embedder.py`:

```python
"""BGE-M3 multilingual embedder.

Wraps sentence-transformers in a lazy-loading module so importing the
package is cheap. First call to encode() downloads the model (~1.3 GB
to ~/.cache/huggingface/) and pins it in memory for subsequent calls.

Embeddings are L2-normalised so cosine similarity == dot product, and
ranking against a vector store can use either metric interchangeably.
"""
from __future__ import annotations

from threading import Lock
from typing import List

import numpy as np

from poc.retrieval import config

_MODEL = None
_LOCK = Lock()


def _get_model():
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer
                _MODEL = SentenceTransformer(config.EMBEDDING_MODEL)
    return _MODEL


def encode(texts: List[str]) -> np.ndarray:
    """Embed N texts to a (N, dim) float32 numpy array, L2-normalised."""
    return _get_model().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)


def encode_one(text: str) -> np.ndarray:
    """Embed a single text to a (dim,) float32 vector."""
    return encode([text])[0]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_embedder.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/embedder.py poc/retrieval/tests/test_embedder.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add lazy-loading BGE-M3 embedder"
```

## Task alpha.4: store.py (TDD)

**Files:**
- Create: `poc/retrieval/store.py`
- Create: `poc/retrieval/tests/test_store.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/tests/test_store.py`:

```python
from __future__ import annotations
import numpy as np


def _unit(seed: int, dim: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def test_store_starts_empty(in_memory_store):
    assert in_memory_store.count() == 0


def test_insert_and_count(in_memory_store):
    in_memory_store.insert("a", {"text": "alpha"}, _unit(1))
    in_memory_store.insert("b", {"text": "beta"}, _unit(2))
    assert in_memory_store.count() == 2


def test_search_returns_nearest_first(in_memory_store):
    """Inserting a vector and then querying with the exact same vector
    must return that item as the top hit."""
    target = _unit(42)
    other = _unit(99)
    in_memory_store.insert("target", {"text": "the one"}, target)
    in_memory_store.insert("other", {"text": "not it"}, other)
    hits = in_memory_store.search(target, k=2)
    assert hits[0]["id"] == "target"
    assert hits[0]["payload"]["text"] == "the one"
    assert hits[0]["distance"] < hits[1]["distance"]


def test_search_k_limits_results(in_memory_store):
    for i in range(5):
        in_memory_store.insert(f"item_{i}", {"text": f"#{i}"}, _unit(i))
    hits = in_memory_store.search(_unit(0), k=3)
    assert len(hits) == 3


def test_insert_replaces_on_same_id(in_memory_store):
    in_memory_store.insert("dup", {"text": "first"}, _unit(1))
    in_memory_store.insert("dup", {"text": "second"}, _unit(2))
    assert in_memory_store.count() == 1
    hits = in_memory_store.search(_unit(2), k=1)
    assert hits[0]["payload"]["text"] == "second"
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_store.py -v
```
Expected: ImportError (store.py doesn't exist).

- [ ] **Step 3: Implement store.py**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/store.py`:

```python
"""sqlite-vec wrapper for vector storage + similarity search.

Each item has a string `id`, a JSON `payload` (the full metadata),
and a `dim`-float vector. sqlite-vec is loaded as a SQLite extension;
the data lives in a single .db file (or `:memory:` for tests).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import sqlite_vec


class Store:
    def __init__(self, db_path: Union[str, Path], dim: int = 1024):
        self._dim = dim
        path_str = str(db_path) if isinstance(db_path, Path) else db_path
        self._conn = sqlite3.connect(path_str)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._setup()

    def _setup(self) -> None:
        # Main metadata table -- one row per item.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                id TEXT UNIQUE NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        # Vec virtual table -- one row per item, keyed by rowid linking to items.
        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vecs USING vec0(
                embedding float[{self._dim}]
            )
            """
        )
        self._conn.commit()

    def count(self) -> int:
        cur = self._conn.execute("SELECT COUNT(*) FROM items")
        return int(cur.fetchone()[0])

    def insert(self, id_: str, payload: Dict[str, Any], embedding: np.ndarray) -> None:
        # If id_ already exists, delete the old row first (REPLACE on id).
        cur = self._conn.execute("SELECT rowid FROM items WHERE id = ?", (id_,))
        existing = cur.fetchone()
        if existing is not None:
            old_rowid = existing[0]
            self._conn.execute("DELETE FROM items WHERE rowid = ?", (old_rowid,))
            self._conn.execute("DELETE FROM vecs WHERE rowid = ?", (old_rowid,))

        cur = self._conn.execute(
            "INSERT INTO items (id, payload) VALUES (?, ?)",
            (id_, json.dumps(payload, ensure_ascii=False)),
        )
        rowid = cur.lastrowid
        vec_blob = embedding.astype(np.float32).tobytes()
        self._conn.execute(
            "INSERT INTO vecs (rowid, embedding) VALUES (?, ?)",
            (rowid, vec_blob),
        )
        self._conn.commit()

    def search(self, query: np.ndarray, k: int = 10) -> List[Dict[str, Any]]:
        vec_blob = query.astype(np.float32).tobytes()
        cur = self._conn.execute(
            """
            SELECT items.id, items.payload, vecs.distance
            FROM vecs
            JOIN items ON items.rowid = vecs.rowid
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (vec_blob, k),
        )
        return [
            {"id": row[0], "payload": json.loads(row[1]), "distance": float(row[2])}
            for row in cur.fetchall()
        ]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_store.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/store.py poc/retrieval/tests/test_store.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add sqlite-vec backed Store with insert+search"
```

## Task alpha.5: retriever.py (TDD)

**Files:**
- Create: `poc/retrieval/retriever.py`
- Create: `poc/retrieval/tests/test_retriever.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/tests/test_retriever.py`:

```python
from __future__ import annotations
import numpy as np


def _unit(seed: int, dim: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def test_retriever_returns_hits_with_score_and_payload(stub_embedder_model, in_memory_store):
    from poc.retrieval.retriever import Retriever
    in_memory_store.insert(
        "zh_1",
        {"text": "山重水复", "author": "陆游", "lang": "zh"},
        _unit(1),
    )
    in_memory_store.insert(
        "en_1",
        {"text": "Rest when weary", "author": "anon", "lang": "en"},
        _unit(2),
    )
    r = Retriever(in_memory_store)
    hits = r.search("anything", k=2)
    assert len(hits) == 2
    assert all("score" in h for h in hits)
    assert all("text" in h and "author" in h and "lang" in h for h in hits)


def test_retriever_filters_by_lang(stub_embedder_model, in_memory_store):
    from poc.retrieval.retriever import Retriever
    in_memory_store.insert("zh_1", {"text": "z1", "lang": "zh"}, _unit(1))
    in_memory_store.insert("zh_2", {"text": "z2", "lang": "zh"}, _unit(2))
    in_memory_store.insert("en_1", {"text": "e1", "lang": "en"}, _unit(3))
    r = Retriever(in_memory_store)
    hits = r.search("query", k=10, lang="zh")
    assert {h["id"] for h in hits} == {"zh_1", "zh_2"}


def test_retriever_lang_none_returns_all(stub_embedder_model, in_memory_store):
    from poc.retrieval.retriever import Retriever
    in_memory_store.insert("zh_1", {"text": "z", "lang": "zh"}, _unit(1))
    in_memory_store.insert("en_1", {"text": "e", "lang": "en"}, _unit(2))
    r = Retriever(in_memory_store)
    hits = r.search("query", k=10, lang=None)
    assert {h["id"] for h in hits} == {"zh_1", "en_1"}


def test_retriever_score_in_0_to_1_range(stub_embedder_model, in_memory_store):
    from poc.retrieval.retriever import Retriever
    in_memory_store.insert("a", {"text": "a"}, _unit(1))
    r = Retriever(in_memory_store)
    hits = r.search("any", k=1)
    assert 0.0 <= hits[0]["score"] <= 1.0
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_retriever.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement retriever.py**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/retriever.py`:

```python
"""Compose embedder + store into a search pipeline."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from poc.retrieval import embedder
from poc.retrieval.store import Store


class Retriever:
    def __init__(self, store: Store):
        self._store = store

    def search(
        self,
        query: str,
        k: int = 10,
        lang: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Return up to k hits ordered by descending similarity.

        Each hit: {id, text, author?, source?, lang?, themes?, score}.
        score in [0, 1] (1 = identical vector).
        If lang is given ('en' or 'zh'), filter to that language.
        Filtering is done post-hoc on a slightly oversized k to compensate.
        """
        # Oversample so post-filter still has k results in mixed corpora.
        retrieval_k = k * 4 if lang else k
        vec = embedder.encode_one(query)
        raw = self._store.search(vec, k=retrieval_k)

        hits: List[Dict[str, Any]] = []
        for r in raw:
            payload = r["payload"]
            if lang and payload.get("lang") != lang:
                continue
            # Cosine distance from sqlite-vec is in [0, 2]; normalised
            # embeddings make cosine_sim = 1 - distance / 2.
            # Empirically sqlite-vec returns distances close to 0 for
            # identical vectors and ~1 for orthogonal; use the more
            # common mapping `score = max(0, 1 - distance)` for display.
            score = max(0.0, 1.0 - r["distance"])
            hits.append({
                "id": r["id"],
                "score": round(score, 4),
                **payload,
            })
            if len(hits) >= k:
                break
        return hits
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_retriever.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/retriever.py poc/retrieval/tests/test_retriever.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add Retriever composing embedder + store with lang filter"
```

## Task alpha.6: reranker.py (TDD)

**Files:**
- Create: `poc/retrieval/reranker.py`
- Create: `poc/retrieval/tests/test_reranker.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/tests/test_reranker.py`:

```python
from __future__ import annotations
import pytest


def _candidates(n: int = 3):
    return [
        {"id": f"c{i}", "text": f"text {i}", "author": f"a{i}",
         "source": f"s{i}", "lang": "en", "score": 0.8 - 0.1 * i}
        for i in range(n)
    ]


def test_rerank_returns_picks(fake_llm):
    from poc.retrieval.reranker import rerank
    fake_llm.reply_with('[{"id": 2, "why": "second matches mood"}, {"id": 1, "why": "first works too"}]')
    out = rerank(fake_llm, "fake-model", "any user text", _candidates(3), max_picks=3)
    assert out == [
        {"index": 2, "why": "second matches mood"},
        {"index": 1, "why": "first works too"},
    ]


def test_rerank_empty_array_is_valid(fake_llm):
    from poc.retrieval.reranker import rerank
    fake_llm.reply_with('[]')
    out = rerank(fake_llm, "fake-model", "totally off-topic", _candidates(3), max_picks=3)
    assert out == []


def test_rerank_malformed_json_raises_valueerror(fake_llm):
    from poc.retrieval.reranker import rerank
    fake_llm.reply_with('not even close to JSON')
    with pytest.raises(ValueError, match="invalid JSON"):
        rerank(fake_llm, "fake-model", "x", _candidates(3))


def test_rerank_out_of_range_index_raises_valueerror(fake_llm):
    from poc.retrieval.reranker import rerank
    fake_llm.reply_with('[{"id": 99, "why": "..."}]')
    with pytest.raises(ValueError, match="out of range"):
        rerank(fake_llm, "fake-model", "x", _candidates(3))


def test_rerank_missing_keys_raises_valueerror(fake_llm):
    from poc.retrieval.reranker import rerank
    fake_llm.reply_with('[{"wrong_key": 1}]')
    with pytest.raises(ValueError, match="unexpected schema"):
        rerank(fake_llm, "fake-model", "x", _candidates(3))
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_reranker.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement reranker.py**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/reranker.py`:

```python
"""LLM reranker: pick the best matches from candidates with one-line whys.

Single LLM call; structured JSON output. Picks are indexed 1..N matching
the order of candidates passed in. Hallucination is structurally
impossible -- the LLM never sees the corpus, only the K candidates.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

SYSTEM = """You re-rank candidate quotes for emotional / semantic fit with a user's text.

Rules:
- Pick at most {max_picks} of the {n} candidates that truly fit the user's text.
- For each pick, write a one-sentence "why" in the SAME LANGUAGE as the user's text.
- If none of the candidates feels right, return an empty array.
- NEVER invent quotes. NEVER edit text or author. Only choose from the given list.

Output strictly as JSON: [{{"id": <int 1..{n}>, "why": "<one sentence>"}}].
No prose, no markdown fences, no preamble."""

USER_TEMPLATE = """Text: {user_text}

Candidates:
{candidates_block}"""


def _format_candidates(candidates: List[Dict[str, Any]]) -> str:
    lines = []
    for i, c in enumerate(candidates, start=1):
        author = c.get("author", "anon")
        source = c.get("source", "")
        suffix = f" -- {author}" + (f", {source}" if source else "")
        lines.append(f"{i}. [{c.get('lang', '?')}] {c['text']}{suffix}")
    return "\n".join(lines)


def rerank(
    client: Any,
    model: str,
    user_text: str,
    candidates: List[Dict[str, Any]],
    max_picks: int = 3,
) -> List[Dict[str, Any]]:
    """Return a list of {"index": <1..N>, "why": <str>}.

    Raises ValueError if the LLM emits malformed JSON, an out-of-range
    index, or a row missing the expected keys.
    """
    n = len(candidates)
    if n == 0:
        return []
    system = SYSTEM.format(max_picks=max_picks, n=n)
    user = USER_TEMPLATE.format(
        user_text=user_text,
        candidates_block=_format_candidates(candidates),
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"reranker emitted invalid JSON: {raw[:200]}") from e
    if not isinstance(parsed, list):
        raise ValueError(f"reranker emitted unexpected schema (not a list): {raw[:200]}")
    out: List[Dict[str, Any]] = []
    for item in parsed[:max_picks]:
        if not isinstance(item, dict) or "id" not in item or "why" not in item:
            raise ValueError(f"reranker emitted unexpected schema: {raw[:200]}")
        idx = item["id"]
        if not isinstance(idx, int) or idx < 1 or idx > n:
            raise ValueError(f"reranker pick out of range (got {idx}, n={n}): {raw[:200]}")
        out.append({"index": idx, "why": str(item["why"])})
    return out
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_reranker.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/reranker.py poc/retrieval/tests/test_reranker.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add LLM reranker with structured JSON picks"
```

## Task alpha.7: main.py with /health (TDD)

**Files:**
- Create: `poc/retrieval/main.py`
- Create: `poc/retrieval/tests/test_api.py`

- [ ] **Step 1: Write failing test for /health**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/tests/test_api.py`:

```python
from __future__ import annotations


def test_health_returns_status_and_config(api_client):
    test_client, _, _ = api_client
    r = test_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["embedder"] == "BAAI/bge-m3"
    assert body["store"] == "sqlite-vec"
    assert "corpus_size" in body
    assert body["namespaces"] == ["quotes"]
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_api.py -v
```
Expected: import error or fixture failure (main.py doesn't exist).

- [ ] **Step 3: Implement main.py with /health**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/main.py`:

```python
"""MiraNote POC -- semantic retrieval over a local quote corpus.

Backend-only. Two endpoints:
- POST /search  -- generic top-K semantic search over a corpus
- POST /quotes  -- business endpoint: /search + LLM reranker + "why" lines

Uses BGE-M3 (multilingual) embeddings via sentence-transformers and a
sqlite-vec store. The reranker runs through any OpenAI-compatible LLM.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI

from poc.retrieval import config, reranker
from poc.retrieval.retriever import Retriever
from poc.retrieval.store import Store


if not config.LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY is required. Set it in .env")

client_kwargs: Dict[str, Any] = {"api_key": config.LLM_API_KEY}
if config.LLM_BASE_URL:
    client_kwargs["base_url"] = config.LLM_BASE_URL
llm = OpenAI(**client_kwargs)


def _open_store() -> Store:
    """Open the index DB if it exists; create empty one otherwise."""
    index_path = config.INDEX_DB_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)
    return Store(db_path=index_path, dim=1024)


store = _open_store()
retriever = Retriever(store)


app = FastAPI(title="MiraNote Retrieval", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "embedder": config.EMBEDDING_MODEL,
        "store": "sqlite-vec",
        "corpus_size": store.count(),
        "namespaces": ["quotes"],
    }
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_api.py::test_health_returns_status_and_config -v
```
Expected: PASS.

- [ ] **Step 5: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/main.py poc/retrieval/tests/test_api.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add retrieval FastAPI app with /health"
```

## Task alpha.8: /search endpoint (TDD)

**Files:**
- Modify: `poc/retrieval/main.py`
- Modify: `poc/retrieval/tests/test_api.py`

- [ ] **Step 1: Append failing tests**

Append to `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/tests/test_api.py`:

```python
import numpy as np


def _unit(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(1024).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-12)


def _seed_store(test_client_app, entries):
    """Helper: insert rows directly into the live store the API uses."""
    import retrieval_main_for_tests as main
    for id_, payload in entries:
        main.store.insert(id_, payload, _unit(abs(hash(id_)) % 1000))


def test_search_returns_top_k_with_score(api_client):
    test_client, _, _ = api_client
    _seed_store(test_client, [
        ("zh_1", {"text": "z1", "lang": "zh"}),
        ("en_1", {"text": "e1", "lang": "en"}),
    ])
    r = test_client.post("/search", json={"query": "anything", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert "hits" in body
    assert len(body["hits"]) == 2
    assert all("score" in h for h in body["hits"])
    assert all("metadata" in h for h in body["hits"])


def test_search_rejects_unknown_namespace(api_client):
    test_client, _, _ = api_client
    r = test_client.post("/search", json={"query": "x", "namespace": "bogus"})
    assert r.status_code == 422


def test_search_default_k_is_10(api_client):
    test_client, _, _ = api_client
    for i in range(15):
        _seed_store(test_client, [(f"x_{i}", {"text": f"#{i}"})])
    r = test_client.post("/search", json={"query": "x"})
    assert r.status_code == 200
    assert len(r.json()["hits"]) == 10
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_api.py -v -k search
```
Expected: 3 fail (404).

- [ ] **Step 3: Add /search to main.py**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/main.py`. Add new Pydantic models after the imports:

```python
class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(10, ge=1, le=50)
    namespace: str = Field("quotes")


class SearchHit(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    hits: List[SearchHit]
```

Add this endpoint after `@app.get("/health")`:

```python
@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest):
    """Generic top-K semantic search over the corpus."""
    if req.namespace != "quotes":
        raise HTTPException(status_code=422, detail=f"unknown namespace: {req.namespace}")
    hits = retriever.search(req.query, k=req.k)
    return SearchResponse(
        hits=[
            SearchHit(
                id=h["id"],
                text=h["text"],
                score=h["score"],
                metadata={k: v for k, v in h.items() if k not in ("id", "text", "score")},
            )
            for h in hits
        ]
    )
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_api.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/main.py poc/retrieval/tests/test_api.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add /search generic top-K endpoint"
```

## Task alpha.9: /quotes endpoint (TDD)

**Files:**
- Modify: `poc/retrieval/main.py`
- Modify: `poc/retrieval/tests/test_api.py`

- [ ] **Step 1: Append failing tests**

Append to `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/tests/test_api.py`:

```python
def test_quotes_returns_reranked_matches(api_client):
    test_client, fake_llm, _ = api_client
    _seed_store(test_client, [
        ("q_1", {"text": "first quote", "author": "A", "source": "S1", "lang": "en"}),
        ("q_2", {"text": "second quote", "author": "B", "source": "S2", "lang": "en"}),
        ("q_3", {"text": "third quote", "author": "C", "source": "S3", "lang": "en"}),
    ])
    fake_llm.reply_with('[{"id": 2, "why": "perfect fit"}, {"id": 1, "why": "also good"}]')

    r = test_client.post("/quotes", json={"text": "test feeling", "max": 3})
    assert r.status_code == 200
    body = r.json()
    assert "matches" in body
    assert len(body["matches"]) == 2
    m = body["matches"][0]
    assert m["text"] == "second quote"
    assert m["author"] == "B"
    assert m["why"] == "perfect fit"


def test_quotes_empty_match_array_is_ok(api_client):
    test_client, fake_llm, _ = api_client
    _seed_store(test_client, [
        ("q_1", {"text": "irrelevant", "lang": "en"}),
    ])
    fake_llm.reply_with('[]')
    r = test_client.post("/quotes", json={"text": "anything"})
    assert r.status_code == 200
    assert r.json()["matches"] == []


def test_quotes_lang_filter(api_client):
    test_client, fake_llm, _ = api_client
    _seed_store(test_client, [
        ("zh_1", {"text": "zh quote", "author": "Z", "lang": "zh"}),
        ("en_1", {"text": "en quote", "author": "E", "lang": "en"}),
    ])
    # Reranker picks index 1, which after lang=zh filter is the zh one.
    fake_llm.reply_with('[{"id": 1, "why": "yes"}]')
    r = test_client.post("/quotes", json={"text": "x", "lang": "zh", "max": 1})
    assert r.status_code == 200
    assert r.json()["matches"][0]["text"] == "zh quote"


def test_quotes_reranker_invalid_json_returns_502(api_client):
    test_client, fake_llm, _ = api_client
    _seed_store(test_client, [("q_1", {"text": "x", "lang": "en"})])
    fake_llm.reply_with('not JSON')
    r = test_client.post("/quotes", json={"text": "y"})
    assert r.status_code == 502
    assert "invalid JSON" in r.json()["detail"]
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_api.py -v -k quotes
```
Expected: 4 fail (404).

- [ ] **Step 3: Add /quotes to main.py**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/main.py`. Add new Pydantic models near the others:

```python
from typing import Literal


class QuotesRequest(BaseModel):
    text: str = Field(..., min_length=1)
    max: int = Field(3, ge=1, le=5)
    lang: Literal["auto", "en", "zh", "both"] = Field("auto")


class QuoteMatch(BaseModel):
    text: str
    author: Optional[str] = None
    source: Optional[str] = None
    lang: Optional[str] = None
    score: float
    why: str


class QuotesResponse(BaseModel):
    matches: List[QuoteMatch]
```

Add this endpoint after `/search`:

```python
@app.post("/quotes", response_model=QuotesResponse)
async def quotes(req: QuotesRequest):
    """Pick the best quotes from the corpus for the user's text."""
    lang_filter = None if req.lang in ("auto", "both") else req.lang
    candidates = retriever.search(req.text, k=config.RETRIEVE_K, lang=lang_filter)
    if not candidates:
        return QuotesResponse(matches=[])

    try:
        picks = reranker.rerank(
            llm, config.LLM_MODEL, req.text, candidates, max_picks=req.max,
        )
    except ValueError as e:
        raise HTTPException(status_code=502, detail=str(e))

    matches: List[QuoteMatch] = []
    for p in picks:
        c = candidates[p["index"] - 1]  # 1-indexed from reranker
        matches.append(QuoteMatch(
            text=c["text"],
            author=c.get("author"),
            source=c.get("source"),
            lang=c.get("lang"),
            score=c["score"],
            why=p["why"],
        ))
    return QuotesResponse(matches=matches)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_api.py -v
```
Expected: 8 passed.

- [ ] **Step 5: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/main.py poc/retrieval/tests/test_api.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add /quotes endpoint with reranker + lang filter"
```

## Task alpha.10: build_corpus.py script + run + commit corpus JSONs

**Files:**
- Create: `poc/retrieval/scripts/build_corpus.py`
- Create: `poc/retrieval/corpus/quotes_en.json`
- Create: `poc/retrieval/corpus/quotes_zh.json`
- Create: `poc/retrieval/corpus/README.md`

**Prerequisite:** Phase 0 PR (`feat/allowlist-corpus-json`) must be MERGED to main of `MiraNote-AI/.github`. Otherwise Rule 3 CI will fail when this task commits CJK content. **Verify first:**

```bash
gh pr view <phase-0-pr-number> -R MiraNote-AI/.github --json state --jq .state
# Expected: "MERGED". If "OPEN", wait.
```

- [ ] **Step 1: Stage source datasets (these are gitignored)**

```bash
mkdir -p /Users/mengjia/MiraNote/miranote-api/poc/retrieval/sources
cd /Users/mengjia/MiraNote/miranote-api/poc/retrieval/sources

# Chinese poetry: clone the canonical dataset (MIT-licensed)
git clone --depth=1 https://github.com/chinese-poetry/chinese-poetry.git

# English: stage a Wikiquote-style JSON file by hand or via the
# wikiquote Python library. For the build_corpus script's contract,
# the format expected is:
# en_raw.json -> [{"text": "...", "author": "...", "source": "..."}]
# Pull a representative sample of Marcus Aurelius, Thoreau, Lao Tzu
# (English translations), Emerson, Whitman from Project Gutenberg
# plain-text files, slice into quotation-sized chunks, save as
# sources/en_raw.json. Aim for 1,500-2,000 raw entries so the LLM
# tagger has selection room to land on the best 500.
```

(The "stage source datasets" step is manual. If automating, the build_corpus.py script should accept a `--sources-dir` flag.)

- [ ] **Step 2: Write build_corpus.py**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/scripts/build_corpus.py`:

```python
"""Offline corpus builder.

Reads source datasets from poc/retrieval/sources/ (gitignored) and
emits the curated corpus JSONs to poc/retrieval/corpus/. The LLM is
used only to:
  1. Filter entries for journal relevance (drop too-long, too-obscure,
     duplicate content).
  2. Assign 1-3 theme tags from the fixed taxonomy.

The LLM NEVER generates text or author -- those are copied verbatim
from the source.

Usage:
    PYTHONPATH=../.. ./poc/retrieval/.venv/bin/python3 scripts/build_corpus.py \\
        --sources poc/retrieval/sources \\
        --out poc/retrieval/corpus \\
        --target-en 500 --target-zh 500
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI


THEMES = [
    "love", "loss", "grief", "hope", "despair", "loneliness",
    "friendship", "family", "nostalgia", "time", "nature", "seasons",
    "home", "travel", "work", "ambition", "rest", "joy", "gratitude",
    "regret", "turning-point", "perseverance", "solitude", "change",
    "farewell",
]


TAG_SYSTEM = (
    "You assign theme tags to a quote. Output ONLY a JSON array of "
    "1-3 tags from this list (lowercase, exact spelling): "
    + ", ".join(THEMES)
    + ". No explanations, no other text."
)


def load_zh_from_chinese_poetry(sources_dir: Path) -> List[Dict[str, Any]]:
    """Walk chinese-poetry/ and assemble candidate entries.

    Returns rows of {text, author, source, lang='zh', era}.
    The chinese-poetry repo organises poems as JSON files; pull from
    quan_tang_shi (Tang poetry) and song-ci (Song lyrics). Each file
    is a JSON array of objects with 'paragraphs' (lines) and 'author'.
    """
    out: List[Dict[str, Any]] = []
    repo = sources_dir / "chinese-poetry"
    if not repo.exists():
        return out
    # Tang
    tang_dir = repo / "quan_tang_shi" / "json"
    if tang_dir.exists():
        for f in sorted(tang_dir.glob("*.json"))[:50]:  # ~50 files = thousands of poems
            data = json.loads(f.read_text(encoding="utf-8"))
            for poem in data:
                for line in poem.get("paragraphs", []):
                    if 10 <= len(line) <= 30:
                        out.append({
                            "text": line.strip(),
                            "author": poem.get("author", ""),
                            "source": poem.get("title", ""),
                            "lang": "zh",
                            "era": "Tang",
                        })
    # Song-ci (similar)
    ci_dir = repo / "ci"
    if ci_dir.exists():
        for f in sorted(ci_dir.glob("ci.song.*.json"))[:30]:
            data = json.loads(f.read_text(encoding="utf-8"))
            for poem in data:
                for line in poem.get("paragraphs", []):
                    if 10 <= len(line) <= 30:
                        out.append({
                            "text": line.strip(),
                            "author": poem.get("author", ""),
                            "source": poem.get("rhythmic", ""),
                            "lang": "zh",
                            "era": "Song",
                        })
    return out


def load_en_from_sources(sources_dir: Path) -> List[Dict[str, Any]]:
    """Read sources/en_raw.json into the candidate list shape."""
    f = sources_dir / "en_raw.json"
    if not f.exists():
        return []
    raw = json.loads(f.read_text(encoding="utf-8"))
    return [
        {
            "text": r["text"].strip(),
            "author": r.get("author", "anon"),
            "source": r.get("source", ""),
            "lang": "en",
        }
        for r in raw
    ]


def dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in rows:
        key = r["text"]
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def tag_themes(client: OpenAI, model: str, row: Dict[str, Any]) -> List[str]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": TAG_SYSTEM},
            {"role": "user", "content": row["text"]},
        ],
    )
    raw = resp.choices[0].message.content.strip()
    try:
        tags = json.loads(raw)
    except Exception:
        return []
    if not isinstance(tags, list):
        return []
    return [t for t in tags if isinstance(t, str) and t in THEMES][:3]


def assign_ids(rows: List[Dict[str, Any]], lang: str) -> List[Dict[str, Any]]:
    return [
        {**r, "id": f"{lang}_{i:04d}"}
        for i, r in enumerate(rows, start=1)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-en", type=int, default=500)
    parser.add_argument("--target-zh", type=int, default=500)
    args = parser.parse_args()

    load_dotenv(args.out.parent / ".env")
    key = os.getenv("LLM_API_KEY")
    if not key:
        sys.exit("LLM_API_KEY required")
    client = OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL"))
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    print(f"Loading sources from {args.sources}")
    zh = dedupe(load_zh_from_chinese_poetry(args.sources))[: args.target_zh]
    en = dedupe(load_en_from_sources(args.sources))[: args.target_en]
    print(f"  ZH candidates: {len(zh)}, EN candidates: {len(en)}")

    args.out.mkdir(parents=True, exist_ok=True)

    print("Tagging ZH...")
    for i, r in enumerate(zh):
        r["themes"] = tag_themes(client, model, r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(zh)}")
    zh = assign_ids(zh, "zh")
    (args.out / "quotes_zh.json").write_text(
        json.dumps(zh, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"  wrote {len(zh)} entries to quotes_zh.json")

    print("Tagging EN...")
    for i, r in enumerate(en):
        r["themes"] = tag_themes(client, model, r)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(en)}")
    en = assign_ids(en, "en")
    (args.out / "quotes_en.json").write_text(
        json.dumps(en, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"  wrote {len(en)} entries to quotes_en.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run the script**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 poc/retrieval/scripts/build_corpus.py \
    --sources poc/retrieval/sources \
    --out poc/retrieval/corpus \
    --target-en 500 --target-zh 500
```

Expect ~10-15 minutes of LLM calls (500 + 500 entries, ~1 sec each through DeepSeek).
Watch the progress prints; if rate-limited, the script will exit with a 429 -- in that case, increase a sleep or batch. Adjust as needed.

- [ ] **Step 4: Inspect the output**

```bash
jq 'length' /Users/mengjia/MiraNote/miranote-api/poc/retrieval/corpus/quotes_en.json
jq 'length' /Users/mengjia/MiraNote/miranote-api/poc/retrieval/corpus/quotes_zh.json
jq '.[0]' /Users/mengjia/MiraNote/miranote-api/poc/retrieval/corpus/quotes_en.json
jq '.[0]' /Users/mengjia/MiraNote/miranote-api/poc/retrieval/corpus/quotes_zh.json
```
Expected: both files report 500 entries; each entry has `id, text, author, source, lang, themes`.

- [ ] **Step 5: Write corpus/README.md**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/corpus/README.md`:

```markdown
# Corpus sources

This directory holds the curated quote / poetry corpus used by the
retrieval POC. The files here are the source of truth -- the
`build_corpus.py` script can re-generate them but the JSON here is
what ships.

## Files

- `quotes_en.json` -- English quotes (~500)
- `quotes_zh.json` -- Chinese poetry lines (~500)

## Per-entry schema

```json
{
  "id": "<lang>_<4-digit zero-padded>",
  "text": "the quote / line",
  "author": "attributed author",
  "source": "work / context",
  "lang": "en" | "zh",
  "era": "Tang" | "Song" | "20c" | ...,   // ZH only
  "themes": ["hope", "perseverance", ...]
}
```

## Sources

| Dataset | License | URL | Pulled | Used for |
|---|---|---|---|---|
| chinese-poetry/chinese-poetry | MIT | https://github.com/chinese-poetry/chinese-poetry | 2026-06-05 | quotes_zh.json (Tang + Song) |
| Project Gutenberg public-domain works | Public domain | https://www.gutenberg.org | 2026-06-05 | quotes_en.json (Marcus Aurelius, Thoreau, etc.) |

## How to rebuild

```bash
cd poc/retrieval
.venv/bin/python3 scripts/build_corpus.py \
    --sources sources \
    --out corpus \
    --target-en 500 --target-zh 500
```

The script copies (text, author, source) triples verbatim from the
sources; the LLM only assigns theme tags.

## Rule 3

CJK is permitted in this directory via the `poc/*/corpus/*.json`
and `poc/*/corpus/*.md` allowlist entries (see
`MiraNote-AI/.github` `checks/no_cjk_or_emoji.py`).
```

- [ ] **Step 6: Rule 3 check (this is the moment the allowlist matters)**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```
Expected: exit=0.

If FAIL with "forbidden character U+...", Phase 0 hasn't taken effect on this dotgithub checkout. Pull dotgithub main and retry:
```bash
git -C /Users/mengjia/MiraNote/dotgithub checkout main && git -C /Users/mengjia/MiraNote/dotgithub pull --ff-only
```

- [ ] **Step 7: Commit corpus + script**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add \
    poc/retrieval/scripts/build_corpus.py \
    poc/retrieval/corpus/
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add corpus build script and 1k-entry quote corpus"
```

## Task alpha.11: build_index.py + initial index build

**Files:**
- Create: `poc/retrieval/scripts/build_index.py`

- [ ] **Step 1: Write build_index.py**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/scripts/build_index.py`:

```python
"""Embed the corpus JSONs and write to the sqlite-vec index.

Reads quotes_en.json + quotes_zh.json from the corpus dir, embeds every
entry via the configured embedder, inserts into the store. Runs once;
re-run when the corpus changes (it will replace existing rows by id).

Usage:
    PYTHONPATH=../.. ./poc/retrieval/.venv/bin/python3 scripts/build_index.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from poc.retrieval import config, embedder
from poc.retrieval.store import Store


def main():
    corpus_dir = config.CORPUS_DIR
    index_path = config.INDEX_DB_PATH
    index_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for f in sorted(corpus_dir.glob("quotes_*.json")):
        loaded = json.loads(f.read_text(encoding="utf-8"))
        print(f"  {f.name}: {len(loaded)} entries")
        entries.extend(loaded)

    if not entries:
        sys.exit(f"No corpus entries found in {corpus_dir}")

    print(f"Total: {len(entries)} entries")
    print(f"Loading embedder ({config.EMBEDDING_MODEL})...")
    print("(first run will download ~1.3 GB to ~/.cache/huggingface/)")

    # Batch embedding for speed.
    texts = [e["text"] for e in entries]
    batch = 32
    print(f"Encoding {len(texts)} texts in batches of {batch}...")
    vectors = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i+batch]
        vecs = embedder.encode(chunk)
        vectors.extend(vecs)
        print(f"  {min(i+batch, len(texts))}/{len(texts)}")

    print(f"Writing to {index_path}...")
    if index_path.exists():
        index_path.unlink()
    store = Store(db_path=index_path, dim=1024)
    for entry, vec in zip(entries, vectors):
        store.insert(entry["id"], entry, vec)
    print(f"Done. {store.count()} entries indexed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it (downloads model on first run)**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 poc/retrieval/scripts/build_index.py
```
Expected first run: ~3-5 min model download + ~1-2 min encoding (CPU).
Subsequent runs: ~1-2 min encoding only.

End state: `poc/retrieval/data/index.db` exists (~4 MB).

- [ ] **Step 3: Verify**

```bash
PYTHONPATH=/Users/mengjia/MiraNote/miranote-api ./poc/retrieval/.venv/bin/python3 -c "
from poc.retrieval.store import Store
from poc.retrieval import config
store = Store(db_path=config.INDEX_DB_PATH, dim=1024)
print('count', store.count())
"
```
Expected: `count 1000`.

- [ ] **Step 4: Rule 3 + commit (script only; index.db is gitignored)**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/scripts/build_index.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add build_index script to embed corpus into sqlite-vec"
```

## Task alpha.12: test_corpus.py validation tests

**Files:**
- Create: `poc/retrieval/tests/test_corpus.py`

- [ ] **Step 1: Write tests**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/tests/test_corpus.py`:

```python
from __future__ import annotations
import json
from pathlib import Path

import pytest


CORPUS_DIR = Path(__file__).parent.parent / "corpus"
THEMES = {
    "love", "loss", "grief", "hope", "despair", "loneliness",
    "friendship", "family", "nostalgia", "time", "nature", "seasons",
    "home", "travel", "work", "ambition", "rest", "joy", "gratitude",
    "regret", "turning-point", "perseverance", "solitude", "change",
    "farewell",
}


def _load(name: str):
    f = CORPUS_DIR / name
    if not f.exists():
        pytest.skip(f"{name} not present (corpus not built yet)")
    return json.loads(f.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name,expected_lang", [
    ("quotes_en.json", "en"),
    ("quotes_zh.json", "zh"),
])
def test_corpus_file_well_formed(name, expected_lang):
    data = _load(name)
    assert isinstance(data, list)
    assert len(data) >= 100, f"{name} should have at least 100 entries"
    ids = set()
    for entry in data:
        assert isinstance(entry, dict)
        for key in ("id", "text", "author", "source", "lang", "themes"):
            assert key in entry, f"missing {key} in {entry}"
        assert entry["lang"] == expected_lang
        assert entry["id"] not in ids, f"duplicate id {entry['id']}"
        ids.add(entry["id"])
        assert isinstance(entry["text"], str) and entry["text"].strip()
        assert isinstance(entry["themes"], list)
        for t in entry["themes"]:
            assert t in THEMES, f"unknown theme {t!r} in {entry['id']}"


def test_no_duplicate_text_across_corpus():
    seen = {}
    for name in ("quotes_en.json", "quotes_zh.json"):
        f = CORPUS_DIR / name
        if not f.exists():
            continue
        for entry in json.loads(f.read_text(encoding="utf-8")):
            text = entry["text"]
            if text in seen:
                pytest.fail(f"duplicate text across files: {text[:40]!r} (both {seen[text]} and {entry['id']})")
            seen[text] = entry["id"]
```

- [ ] **Step 2: Run tests**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests/test_corpus.py -v
```
Expected: 3 passed.

If a test fails on `theme` value, the build script let an out-of-taxonomy theme through. Add a filter step to build_corpus.py and re-run; or hand-edit the offending entry. Re-run tests.

- [ ] **Step 3: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/tests/test_corpus.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "test(api): validate corpus schema + theme taxonomy + uniqueness"
```

## Task alpha.13: README + push + PR alpha

**Files:**
- Create: `poc/retrieval/README.md`

- [ ] **Step 1: Write README**

Create `/Users/mengjia/MiraNote/miranote-api/poc/retrieval/README.md`:

```markdown
# Retrieval POC (RAG)

Local semantic-retrieval service. Two endpoints:

- `POST /quotes` -- pick the best quotes from a curated corpus for a
  given user text. Uses an LLM reranker on top of vector search to
  emit 0-3 picks with one-sentence "why" lines. Zero hallucination:
  the LLM only chooses from candidates we retrieved.
- `POST /search` -- generic top-K semantic search (reserved for
  future doc / note search reuse).

Bilingual (English + Chinese classical poetry).

Spec: `docs/specs/2026-06-05-quote-suggestions-rag-design.md`

## Pieces

- `embedder.py` -- BGE-M3 multilingual embeddings via sentence-transformers
- `store.py` -- sqlite-vec backed vector store (single-file SQLite)
- `retriever.py` -- embedder + store composition with language filter
- `reranker.py` -- LLM call with structured JSON output; picks indexed
- `main.py` -- FastAPI app
- `scripts/build_corpus.py` -- offline corpus builder (LLM tags themes
  only; never generates text or authors)
- `scripts/build_index.py` -- offline index builder (embeds corpus, writes index.db)

## Setup

```bash
cd poc/retrieval
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env  # fill in LLM_API_KEY
```

The BGE-M3 embedding model is ~1.3 GB and auto-downloads to
`~/.cache/huggingface/` on first call (one-time, several minutes).
Same pattern as the voice POC's emotion model.

## Build the corpus + index (one-time)

```bash
# 1. Stage source datasets (gitignored under sources/)
mkdir sources
cd sources && git clone --depth=1 https://github.com/chinese-poetry/chinese-poetry.git && cd ..
# Add sources/en_raw.json with English quotes (see corpus/README.md)

# 2. Build the corpus JSONs (10-15 min of LLM calls)
PYTHONPATH=../.. .venv/bin/python3 scripts/build_corpus.py \
    --sources sources --out corpus \
    --target-en 500 --target-zh 500

# 3. Embed and index (3-5 min including first-time model download)
PYTHONPATH=../.. .venv/bin/python3 scripts/build_index.py
```

The committed corpus JSONs are the source of truth; the index .db
is gitignored and rebuilt from JSONs by `build_index.py`.

## Run

```bash
PYTHONPATH=../.. .venv/bin/python3 -m uvicorn main:app --port 8004 --reload
```

## Try it (curl)

```bash
# English mood
curl -s -X POST http://localhost:8004/quotes \
  -H 'Content-Type: application/json' \
  -d '{"text":"I feel exhausted but I have to keep going","max":3}' \
  | python3 -m json.tool

# Chinese mood
curl -s -X POST http://localhost:8004/quotes \
  -H 'Content-Type: application/json' \
  -d '{"text":"今天有点丧","max":3,"lang":"zh"}' \
  | python3 -m json.tool

# Generic search
curl -s -X POST http://localhost:8004/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"perseverance","k":5}' \
  | python3 -m json.tool

# Health
curl -s http://localhost:8004/health | python3 -m json.tool
```

## Tests

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests -v
```

## Configuration

See `.env.example`. The important knobs:

- `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` -- reranker provider
- `EMBEDDING_MODEL` -- defaults to `BAAI/bge-m3`
- `CORPUS_DIR` -- defaults to `./corpus`
- `INDEX_DB_PATH` -- defaults to `./data/index.db`
- `MAX_PICKS` -- default 3 (reranker output cap)
- `RETRIEVE_K` -- default 10 (retriever top-K before reranker)
```

- [ ] **Step 2: Run full test suite**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests -v 2>&1 | tail -25
```
Expected: 4 embedder + 5 store + 4 retriever + 5 reranker + 8 api + 3 corpus = **29 passed**.

- [ ] **Step 3: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```
Expected: exit=0.

- [ ] **Step 4: Manual smoke against live model**

```bash
# Start the server
cd /Users/mengjia/MiraNote/miranote-api/poc/retrieval
PYTHONPATH=../.. .venv/bin/python3 -m uvicorn main:app --port 8004 &
SERVER_PID=$!
sleep 3

# Probe with three test queries
curl -s -X POST http://localhost:8004/quotes -H 'Content-Type: application/json' -d '{"text":"I feel exhausted","max":3}' | python3 -m json.tool
curl -s -X POST http://localhost:8004/quotes -H 'Content-Type: application/json' -d '{"text":"今天有点丧","max":3,"lang":"zh"}' | python3 -m json.tool
curl -s -X POST http://localhost:8004/health | python3 -m json.tool

kill $SERVER_PID
```

Eyeball the picks; if none feel right (i.e. all noise), tune the corpus.

- [ ] **Step 5: Commit README**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/retrieval/README.md
git -C /Users/mengjia/MiraNote/miranote-api commit -m "docs(api): document retrieval POC -- setup, build, run, curl"
```

- [ ] **Step 6: Push and open PR**

```bash
git -C /Users/mengjia/MiraNote/miranote-api push -u origin feat/api-retrieval-poc
cd /Users/mengjia/MiraNote/miranote-api
gh pr create --title "feat(api): add local RAG retrieval POC for quote/poetry suggestions" --body "$(cat <<'EOF'
## Summary

New POC at `poc/retrieval/` -- local semantic retrieval over a curated
~1,000-entry bilingual quote corpus (500 EN + 500 ZH classical poetry).
BGE-M3 multilingual embeddings + sqlite-vec store + LLM reranker that
picks from candidates (zero hallucination by construction).

Two endpoints: `/quotes` (business: ranked picks with one-sentence
"why" lines) and `/search` (generic top-K; reserved for future doc /
note search reuse). Plus offline scripts `build_corpus.py` (LLM only
tags themes; never generates text or authors) and `build_index.py`.

29 unit tests + 3 corpus validation tests. Rule 3 clean (corpus JSONs
allowlisted via the just-merged `MiraNote-AI/.github` PR for
`poc/*/corpus/*.json` and `*.md`).

## Spec

spec: docs/specs/2026-06-05-quote-suggestions-rag-design.md

## Test plan

- [x] `pytest poc/retrieval/tests -v` -- 29 passed
- [x] Rule 3 (`no_cjk_or_emoji`) -- exit 0
- [x] Manual: 3 live queries against DeepSeek + the indexed corpus
      (English mood, Chinese mood, /health), eyeballed for relevance

## Notable design choices

- Corpus is fully automated via `build_corpus.py`; the LLM only tags
  themes (`(text, author, source)` come straight from trusted source
  datasets -- chinese-poetry/chinese-poetry MIT-licensed for Chinese,
  Project Gutenberg public-domain for English). No human spot-check.
- Reranker indexes picks by 1-based position from the candidate list,
  not by content -- makes hallucination structurally impossible (a
  pick that doesn't resolve to a candidate is treated as parse error).
- `/search` exists alongside `/quotes` so the chatbot's future
  semantic doc-search can ride the same plumbing.

EOF
)"
```

- [ ] **Step 7: HARD STOP -- no merge**

PR alpha is open for Jason review. Do NOT admin-bypass.

---

# Phase beta -- text-clean-expand UI + chatbot integration

## Task beta.0: Branch off main

- [ ] **Step 1: Branch**

```bash
git -C /Users/mengjia/MiraNote/miranote-api checkout main
git -C /Users/mengjia/MiraNote/miranote-api pull --ff-only
git -C /Users/mengjia/MiraNote/miranote-api checkout -b feat/api-quote-integration
```

PR β is independent at the test level -- tests stub HTTP. It will need
to wait for PR α to merge before end-to-end smoke is possible, but the
branch / PR can ship before α merges.

## Task beta.1: retrieval_client.py + tests (TDD)

**Files:**
- Create: `poc/chatbot/retrieval_client.py`
- Create: `poc/chatbot/tests/test_retrieval_client.py`

- [ ] **Step 1: Write failing tests**

Create `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/tests/test_retrieval_client.py`:

```python
from __future__ import annotations
from types import SimpleNamespace

import pytest

from poc.chatbot.retrieval_client import RetrievalClient


@pytest.fixture
def captured_post(monkeypatch):
    captured = {"url": None, "json": None}
    scripted = {"response": SimpleNamespace(status_code=200, json=lambda: {"matches": []})}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return scripted["response"]

    monkeypatch.setattr("httpx.post", fake_post)
    return captured, scripted


def test_quotes_posts_to_quotes_url(captured_post):
    captured, _ = captured_post
    c = RetrievalClient("http://localhost:8004")
    c.quotes("I feel tired")
    assert captured["url"] == "http://localhost:8004/quotes"
    assert captured["json"]["text"] == "I feel tired"


def test_quotes_default_max_is_3(captured_post):
    captured, _ = captured_post
    RetrievalClient("http://localhost:8004").quotes("x")
    assert captured["json"]["max"] == 3


def test_quotes_passes_lang_when_set(captured_post):
    captured, _ = captured_post
    RetrievalClient("http://localhost:8004").quotes("x", lang="zh")
    assert captured["json"]["lang"] == "zh"


def test_trailing_slash_stripped(captured_post):
    captured, _ = captured_post
    RetrievalClient("http://localhost:8004/").quotes("x")
    assert captured["url"] == "http://localhost:8004/quotes"


def test_non_200_raises_runtimeerror(captured_post):
    captured, scripted = captured_post
    scripted["response"] = SimpleNamespace(
        status_code=502, text="bad", json=lambda: {"detail": "upstream"}
    )
    with pytest.raises(RuntimeError, match="502"):
        RetrievalClient("http://localhost:8004").quotes("x")


def test_connection_error_raises_runtimeerror(monkeypatch):
    import httpx
    def boom(url, json=None, timeout=None):
        raise httpx.ConnectError("refused")
    monkeypatch.setattr("httpx.post", boom)
    with pytest.raises(RuntimeError, match="unreachable"):
        RetrievalClient("http://localhost:9999").quotes("x")
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests/test_retrieval_client.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement retrieval_client.py**

Create `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/retrieval_client.py`:

```python
"""HTTP client for the retrieval POC.

Mirrors text_client.py: thin synchronous httpx wrapper. The chatbot
dispatcher binds this to a tool ('find_quote') so the agent can call
the retrieval service when the user wants a quote suggestion.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import httpx


class RetrievalClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._base_url + path
        try:
            resp = httpx.post(url, json=payload, timeout=self._timeout)
        except httpx.RequestError as e:
            raise RuntimeError(
                f"retrieval service unreachable at {self._base_url}: {e}"
            ) from e
        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:  # noqa: BLE001
                detail = resp.text
            raise RuntimeError(
                f"retrieval service returned {resp.status_code}: {detail}"
            )
        return resp.json()

    def quotes(self, text: str, max_picks: int = 3, lang: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"text": text, "max": int(max_picks)}
        if lang is not None:
            payload["lang"] = lang
        return self._post("/quotes", payload)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests/test_retrieval_client.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add poc/chatbot/retrieval_client.py poc/chatbot/tests/test_retrieval_client.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add chatbot RetrievalClient httpx wrapper"
```

## Task beta.2: ChatbotConfig.retrieval_client + main.py wiring (TDD)

**Files:**
- Modify: `poc/chatbot/config.py`
- Modify: `poc/chatbot/main.py`
- Modify: `poc/chatbot/.env.example`
- Modify: `poc/chatbot/tests/test_config.py`

- [ ] **Step 1: Append failing test**

Append to `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/tests/test_config.py`:

```python
def test_config_accepts_retrieval_client(tmp_path):
    class FakeClient:
        pass
    cfg = ChatbotConfig(
        docs_root=tmp_path,
        model="fake",
        max_tool_iterations=6,
        max_history_messages=40,
        session_ttl_seconds=3600,
        retrieval_client=FakeClient(),
    )
    assert cfg.retrieval_client is not None
    assert isinstance(cfg.retrieval_client, FakeClient)


def test_config_retrieval_client_defaults_to_none(tmp_path):
    cfg = ChatbotConfig(
        docs_root=tmp_path,
        model="fake",
        max_tool_iterations=6,
        max_history_messages=40,
        session_ttl_seconds=3600,
    )
    assert cfg.retrieval_client is None
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests/test_config.py -v -k retrieval_client
```
Expected: 2 fail.

- [ ] **Step 3: Add retrieval_client to ChatbotConfig**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/config.py`. Modify the `__init__` signature to add `retrieval_client`:

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
        retrieval_client: object = None,
    ):
        self._lock = Lock()
        self.docs_root = docs_root.resolve()
        self.model = model
        self.max_tool_iterations = max_tool_iterations
        self.max_history_messages = max_history_messages
        self.session_ttl_seconds = session_ttl_seconds
        self.text_client = text_client
        self.retrieval_client = retrieval_client
```

- [ ] **Step 4: Wire in main.py**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/main.py`. Add to imports:

```python
from poc.chatbot.retrieval_client import RetrievalClient
```

Find the existing `TEXT_API_URL` line. After the config construction with `text_client=TextClient(TEXT_API_URL)`, modify to also pass `retrieval_client`:

```python
TEXT_API_URL = os.getenv("TEXT_API_URL", "http://localhost:8001")
RETRIEVAL_API_URL = os.getenv("RETRIEVAL_API_URL", "http://localhost:8004")

config = ChatbotConfig(
    docs_root=_initial_docs_root,
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    max_tool_iterations=int(os.getenv("MAX_TOOL_ITERATIONS", "6")),
    max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "40")),
    session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "3600")),
    text_client=TextClient(TEXT_API_URL),
    retrieval_client=RetrievalClient(RETRIEVAL_API_URL),
)
```

- [ ] **Step 5: Update .env.example**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/.env.example`. Append at the bottom:

```

# URL of the retrieval POC server (RAG over quote / poetry corpus).
# The chatbot delegates its find_quote tool to this server.
RETRIEVAL_API_URL=http://localhost:8004
```

- [ ] **Step 6: Run tests**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests/test_config.py -v
```
Expected: previous 7 + 2 new = 9 passed.

- [ ] **Step 7: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add \
    poc/chatbot/config.py poc/chatbot/main.py \
    poc/chatbot/.env.example poc/chatbot/tests/test_config.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): wire RetrievalClient into ChatbotConfig"
```

## Task beta.3: find_quote tool schema + dispatch + tool_descriptions (TDD)

**Files:**
- Modify: `poc/chatbot/tools.py`
- Modify: `poc/chatbot/prompts/tool_descriptions.txt`
- Modify: `poc/chatbot/tests/test_tools.py`

- [ ] **Step 1: Append failing tests**

Append to `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/tests/test_tools.py`:

```python
class _FakeRetrievalClient:
    def __init__(self):
        self.calls = []
        self.scripted = {"matches": [{"text": "stub", "score": 0.9, "why": "ok"}]}

    def quotes(self, text, max_picks=3, lang=None):
        self.calls.append({"text": text, "max_picks": max_picks, "lang": lang})
        return self.scripted


def _cfg_with_retrieval(tmp_docs):
    cfg = _cfg(tmp_docs)
    cfg.retrieval_client = _FakeRetrievalClient()
    return cfg


def test_tools_schema_now_lists_eleven_functions():
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert "find_quote" in names
    assert len(names) == 11  # 10 from prior PRs + find_quote


def test_dispatch_routes_to_find_quote(tmp_docs):
    cfg = _cfg_with_retrieval(tmp_docs)
    out = tools.dispatch(cfg, "find_quote", {"text": "I am tired"})
    assert cfg.retrieval_client.calls == [
        {"text": "I am tired", "max_picks": 3, "lang": None}
    ]
    assert "matches" in out


def test_dispatch_find_quote_passes_max_and_lang(tmp_docs):
    cfg = _cfg_with_retrieval(tmp_docs)
    tools.dispatch(cfg, "find_quote", {"text": "x", "max": 2, "lang": "zh"})
    assert cfg.retrieval_client.calls == [
        {"text": "x", "max_picks": 2, "lang": "zh"}
    ]


def test_dispatch_find_quote_wraps_runtime_error(tmp_docs):
    cfg = _cfg_with_retrieval(tmp_docs)
    def boom(text, max_picks=3, lang=None):
        raise RuntimeError("retrieval service down")
    cfg.retrieval_client.quotes = boom
    out = tools.dispatch(cfg, "find_quote", {"text": "x"})
    assert "error" in out
    assert "retrieval service down" in out["error"]
```

- [ ] **Step 2: Run to verify failure**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests/test_tools.py -v -k find_quote
```
Expected: tests fail (schema missing / unknown tool).

- [ ] **Step 3: Add tool description**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/prompts/tool_descriptions.txt`. Append:

```
find_quote|Find 1-3 famous quotes or pieces of poetry that fit the user's expressed mood or feeling. Call this when the user shares a feeling and might appreciate a quote, or when they explicitly ask for a quote, '名言', '诗句', or '配文'. Returns 0 matches if nothing in the corpus is a good fit (not an error). Optional 'lang' parameter ('en' / 'zh') restricts the search.
```

- [ ] **Step 4: Add find_quote schema to TOOLS list**

Edit `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/tools.py`. After the last tool entry in the `TOOLS` list (after `generate_caption`'s closing `},`), append:

```python
    {
        "type": "function",
        "function": {
            "name": "find_quote",
            "description": _TOOL_DESCRIPTIONS["find_quote"],
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "max": {"type": "integer", "default": 3},
                    "lang": {
                        "type": "string",
                        "enum": ["en", "zh"],
                    },
                },
                "required": ["text"],
            },
        },
    },
```

- [ ] **Step 5: Add dispatch routing for find_quote**

In the same file, edit `dispatch()`. Find the existing `if name == "generate_caption":` branch. After it (still before the final `return {"error": ...}`), add:

```python
        if name == "find_quote":
            lang = args.get("lang")
            return config.retrieval_client.quotes(
                args["text"],
                max_picks=int(args.get("max", 3)),
                lang=lang,
            )
```

- [ ] **Step 6: Run tests**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests/test_tools.py -v 2>&1 | tail -10
```
Expected: existing tests + 4 new = 21 passed (in test_tools.py).

- [ ] **Step 7: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add \
    poc/chatbot/tools.py \
    poc/chatbot/prompts/tool_descriptions.txt \
    poc/chatbot/tests/test_tools.py
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add chatbot find_quote tool delegating to retrieval"
```

## Task beta.4: System prompt update

**Files:**
- Modify: `poc/chatbot/prompts/system.txt`

- [ ] **Step 1: Read current system.txt**

```bash
cat /Users/mengjia/MiraNote/miranote-api/poc/chatbot/prompts/system.txt
```

Identify the paragraph that mentions the text-transformation tools (added in PR #12).

- [ ] **Step 2: Append a paragraph for find_quote**

Append to `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/prompts/system.txt`:

```

When the user shares a mood, a feeling, or asks for a quote or poem
('quote', 'poem', 'verse', '名言', '诗句', '配文'), you may call
find_quote(text, max?, lang?) to pick the most fitting items from a
curated bilingual corpus. find_quote may return 0 matches -- that is
a normal outcome (no good fit), not an error. Tell the user honestly
when no quote felt right, rather than improvising one.
```

- [ ] **Step 3: Rule 3**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```
Expected: exit=0.

- [ ] **Step 4: Run all chatbot tests to confirm no regression**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests -q 2>&1 | tail -5
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git -C /Users/mengjia/MiraNote/miranote-api add poc/chatbot/prompts/system.txt
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): teach chatbot prompt about find_quote and 0-match outcome"
```

## Task beta.5: Text tab UI Quote action

**Files:**
- Modify: `poc/text-clean-expand/static/index.html`

- [ ] **Step 1: Add the Quote option to the action dropdown**

Open `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/static/index.html`. Find the existing action `<select>`:

```html
<select id="txt-action" style="...">
  <option value="clean">Clean</option>
  <option value="expand">Expand</option>
  <option value="polish">Polish</option>
  <option value="shorten">Shorten</option>
  <option value="keywords">Keywords</option>
  <option value="caption">Caption</option>
</select>
```

Add a new `<option>`:

```html
  <option value="quote">Quote</option>
```

- [ ] **Step 2: Add Quote sub-control row**

Find the existing sub-control rows (`txt-sub-shorten`, `txt-sub-caption`, `txt-sub-keywords`). Add a new one after them:

```html
<div class="row" id="txt-sub-quote" hidden style="margin-top: 10px; gap: 12px;">
  <span style="font-size: 13px; color: var(--muted);">Lang:</span>
  <label class="option"><input type="radio" name="txt-quote-lang" value="auto" checked /> auto</label>
  <label class="option"><input type="radio" name="txt-quote-lang" value="en" /> en</label>
  <label class="option"><input type="radio" name="txt-quote-lang" value="zh" /> zh</label>
  <label class="option"><input type="radio" name="txt-quote-lang" value="both" /> both</label>
  <span style="font-size: 13px; color: var(--muted); margin-left: 12px;">Max:</span>
  <input id="txt-quote-max" type="number" min="1" max="5" value="3" style="width: 60px; padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px; font: inherit;" />
</div>
```

- [ ] **Step 3: Add RETRIEVAL_API constant**

Find the `TEXT_API` / `VOICE_API` / `CHAT_API` constants near the top of the JS block. Add:

```javascript
const RETRIEVAL_API = 'http://localhost:8004';   // retrieval / quotes server
```

- [ ] **Step 4: Update textBuildPayload + textUpdateSubControls + textRun**

Find `textBuildPayload`. After the existing `if (action === 'keywords') { ... }` block, append:

```javascript
  if (action === 'quote') {
    const l = document.querySelector('input[name="txt-quote-lang"]:checked');
    base.lang = l ? l.value : 'auto';
    base.max = parseInt($('txt-quote-max').value, 10) || 3;
    // /quotes expects 'text' (already in base); 'lang' and 'max' as above.
  }
  return base;  // (existing return)
```

Find `textUpdateSubControls`. Add a line for `txt-sub-quote`:

```javascript
function textUpdateSubControls() {
  const action = $('txt-action').value;
  $('txt-sub-shorten').hidden = action !== 'shorten';
  $('txt-sub-caption').hidden = action !== 'caption';
  $('txt-sub-keywords').hidden = action !== 'keywords';
  $('txt-sub-quote').hidden = action !== 'quote';
}
```

Find `textRun`. The function currently builds `TEXT_API + '/' + action`. For Quote, the URL is `RETRIEVAL_API + '/quotes'`. Add a check at the top of the try block:

```javascript
async function textRun() {
  const action = $('txt-action').value;
  const payload = textBuildPayload(action);
  if (!payload) return;

  $('txt-error').hidden = true;
  $('txt-result').hidden = true;
  showLoading($('txt-loading'), 'Running ' + action + '...');
  $('txt-run').disabled = true;

  try {
    const url = action === 'quote'
      ? RETRIEVAL_API + '/quotes'
      : TEXT_API + '/' + action;
    const r = await fetch(url, {
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
```

- [ ] **Step 5: Update textRenderResult + textTitleFor**

Find `textRenderResult`. Add a branch for `quote`:

```javascript
function textRenderResult(action, data) {
  const out = $('txt-output');
  out.classList.remove('empty');
  if (action === 'keywords') {
    // existing keywords chip rendering ...
    return;
  }
  if (action === 'quote') {
    out.textContent = '';
    const matches = data.matches || [];
    if (matches.length === 0) {
      out.textContent = 'Nothing felt quite right -- try rephrasing or adding context.';
      out.classList.add('empty');
      return;
    }
    matches.forEach(function (m) {
      const card = document.createElement('div');
      card.style.cssText = 'margin: 0 0 14px; padding: 14px 16px; background: var(--warm-light); border: 1px solid var(--border); border-radius: 10px;';
      const qt = document.createElement('div');
      qt.style.cssText = 'font-size: 16px; font-style: italic; line-height: 1.6; margin-bottom: 8px;';
      qt.textContent = '"' + m.text + '"';
      const meta = document.createElement('div');
      meta.style.cssText = 'font-size: 12px; color: var(--muted); margin-bottom: 6px;';
      const author = m.author || 'anon';
      const source = m.source ? ', ' + m.source : '';
      const pct = Math.round((m.score || 0) * 100);
      meta.textContent = '-- ' + author + source + '  |  match: ' + pct + '%';
      const why = document.createElement('div');
      why.style.cssText = 'font-size: 13px; color: var(--accent); font-style: italic;';
      why.textContent = m.why ? '~ ' + m.why : '';
      card.appendChild(qt);
      card.appendChild(meta);
      if (m.why) card.appendChild(why);
      out.appendChild(card);
    });
    return;
  }
  // ... existing plain-text rendering branch ...
  const fieldByAction = {
    clean: 'cleaned',
    expand: 'expanded',
    polish: 'polished',
    shorten: 'shortened',
    caption: 'caption',
  };
  out.textContent = data[fieldByAction[action]];
}
```

Find `textTitleFor`. Add `quote`:

```javascript
function textTitleFor(action) {
  return {
    clean: 'Cleaned', expand: 'Expanded', polish: 'Polished',
    shorten: 'Shortened', keywords: 'Keywords', caption: 'Caption',
    quote: 'Quotes',
  }[action] || action;
}
```

- [ ] **Step 6: HTML parse + grep sanity**

```bash
python3 -c "
import html.parser
class P(html.parser.HTMLParser):
    def error(self, m): raise SyntaxError(m)
p = P()
p.feed(open('/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/static/index.html').read())
print('html ok')
"
grep -c "value=\"quote\"\|RETRIEVAL_API\|txt-sub-quote" /Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/static/index.html
```
Expected: `html ok` and grep count >= 3.

- [ ] **Step 7: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add poc/text-clean-expand/static/index.html
git -C /Users/mengjia/MiraNote/miranote-api commit -m "feat(api): add Quote action to Text tab calling retrieval server"
```

## Task beta.6: Update start-all.sh + READMEs

**Files:**
- Modify: `start-all.sh`
- Modify: `poc/text-clean-expand/README.md`
- Modify: `poc/chatbot/README.md`

- [ ] **Step 1: Update start-all.sh to launch the retrieval server**

Open `/Users/mengjia/MiraNote/miranote-api/start-all.sh`. Find the existing block at the bottom with three `start` lines (text / voice / chat). Add a 4th line for retrieval:

```bash
start retrieval 32 "$REPO_ROOT/poc/retrieval"   "$REPO_ROOT"  "--reload --reload-dir poc/retrieval"  poc.retrieval.main:app  8004
```

(Use ANSI colour code 32 = green so it's visually distinct.)

- [ ] **Step 2: Append Quote section to text-clean-expand/README.md**

Append to `/Users/mengjia/MiraNote/miranote-api/poc/text-clean-expand/README.md`:

```markdown
## Quote action (NEW, depends on retrieval server)

The 7th Text-tab action -- `Quote` -- does NOT call this server.
Instead it calls the retrieval POC at `http://localhost:8004/quotes`
(see `poc/retrieval/`). Make sure that server is running, or use
`./start-all.sh` from the repo root.

Sub-controls: `Lang` (auto / en / zh / both) and `Max` (1-5).
Result is rendered as quote cards with author, source, match %, and a
one-sentence "why" line. Zero matches is a valid response.
```

- [ ] **Step 3: Update chatbot/README.md**

Open `/Users/mengjia/MiraNote/miranote-api/poc/chatbot/README.md`. Find the tools table (added in PR #12). Append a new row:

```markdown
| `find_quote(text, max, lang)` | Find quotes/poetry that fit a mood. Delegates to retrieval POC `/quotes`. May return 0 matches. | -- |
```

Find the Configuration section. Append:

```markdown
- `RETRIEVAL_API_URL` -- URL of the retrieval POC server. Defaults to
  `http://localhost:8004`. The chatbot's `find_quote` tool
  HTTP-delegates to this server. Server down -> tool returns an error
  and the agent recovers.
```

- [ ] **Step 4: Run full chatbot test suite once more**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. ./poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests -q 2>&1 | tail -5
```
Expected: all pass.

- [ ] **Step 5: Rule 3 + commit**

```bash
cd /Users/mengjia/MiraNote/dotgithub && PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
git -C /Users/mengjia/MiraNote/miranote-api add \
    start-all.sh \
    poc/text-clean-expand/README.md \
    poc/chatbot/README.md
git -C /Users/mengjia/MiraNote/miranote-api commit -m "docs(api): document Quote action + find_quote tool + 4th server"
```

## Task beta.7: End-to-end smoke + push + PR beta

**Prerequisite:** PR alpha branch (`feat/api-retrieval-poc`) is up. PR alpha doesn't need to be merged yet -- you just need to run the retrieval server locally from that branch's checkout. If you're on PR beta's branch, the corpus & code aren't on this branch, so for the smoke you'll temporarily check out alpha's branch in a worktree OR alternatively merge alpha into beta locally (don't push the merge).

**Easier path:** use a git worktree for the smoke.

- [ ] **Step 1: Worktree for retrieval, start all 4 servers**

```bash
# Worktree the retrieval branch in a sibling dir so files don't clash
git -C /Users/mengjia/MiraNote/miranote-api worktree add ../miranote-api-retrieval feat/api-retrieval-poc

# Start retrieval from the worktree (port 8004)
cd /Users/mengjia/MiraNote/miranote-api-retrieval
PYTHONPATH=. ./poc/retrieval/.venv/bin/python3 -m uvicorn poc.retrieval.main:app --port 8004 &
RETRIEVAL_PID=$!

# Start text-clean-expand, voice, chat from main repo via start-all.sh
cd /Users/mengjia/MiraNote/miranote-api
./start-all.sh &
ALL_PID=$!

sleep 6
```

(NOTE: the chatbot venv .venv/ and demo data are present in the original checkout, not the worktree. Worktree shares tracked files only. The retrieval .venv was created in the original checkout (`poc/retrieval/.venv/`), so referencing `./poc/retrieval/.venv/bin/python3` from the worktree might not resolve. If so, run retrieval from the original checkout AFTER you check out alpha there:
```bash
git -C /Users/mengjia/MiraNote/miranote-api stash
git -C /Users/mengjia/MiraNote/miranote-api checkout feat/api-retrieval-poc
# Start retrieval
# After smoke: stash pop + checkout back to feat/api-quote-integration
```
)

- [ ] **Step 2: Probe everyone's /health**

```bash
for port in 8000 8001 8003 8004; do
  echo "--- port $port ---"
  curl -s --max-time 3 http://127.0.0.1:$port/health | python3 -m json.tool 2>&1 | head -5
done
```

- [ ] **Step 3: UI smoke -- Quote action**

(Manual, open in browser.) Visit `http://localhost:8001/`, pick Text tab, dropdown -> Quote. Enter an English mood, click Run. Verify card-style rendering with author + match %. Pick `Lang: zh` and enter a Chinese mood. Verify Chinese quotes returned.

- [ ] **Step 4: Chat smoke -- find_quote**

```bash
curl -s -X POST http://127.0.0.1:8003/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"I feel exhausted but I have to keep going. Got a quote?"}' \
  | python3 -m json.tool | head -30
```
Verify `tool_trace` includes `find_quote`; `reply` references at least one matched quote.

```bash
curl -s -X POST http://127.0.0.1:8003/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"今天有点丧，给我一句应景的诗"}' \
  | python3 -m json.tool | head -30
```
Verify Chinese chat works similarly.

- [ ] **Step 5: Failure-mode smoke**

```bash
# Stop retrieval server
kill $RETRIEVAL_PID 2>/dev/null
sleep 1

# Hit the Quote action via chat -- should report graceful unavailability
curl -s -X POST http://127.0.0.1:8003/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Got a quote for being tired?"}' | python3 -m json.tool | head -15
```
Verify reply is graceful (doesn't crash; mentions service unavailable).

- [ ] **Step 6: Stop everything**

```bash
kill $ALL_PID 2>/dev/null
pkill -f "uvicorn.*--port (8000|8001|8003|8004)" 2>/dev/null
git -C /Users/mengjia/MiraNote/miranote-api worktree remove ../miranote-api-retrieval --force 2>/dev/null
```

- [ ] **Step 7: Push and open PR beta**

```bash
git -C /Users/mengjia/MiraNote/miranote-api push -u origin feat/api-quote-integration
cd /Users/mengjia/MiraNote/miranote-api
gh pr create --title "feat(api): integrate quote retrieval into Text tab + chatbot" --body "$(cat <<'EOF'
## Summary

Wires the retrieval POC (PR alpha: `feat/api-retrieval-poc`) into the two existing user-facing surfaces:

- **Text tab `Quote` action** -- 7th option in the action dropdown, calls the retrieval server's `/quotes` endpoint cross-origin (port 8004). Sub-controls for `Lang` (auto/en/zh/both) and `Max` (1-5). Result is rendered as quote cards with author, source, match %, and "why" lines. 0-match is rendered as a friendly empty-state.
- **Chatbot `find_quote` tool** -- 11th tool in the chatbot's registry, HTTP-delegates to the retrieval server via a thin `RetrievalClient` (mirrors the existing `TextClient` pattern). System prompt teaches the agent when to call it and that 0 matches is a normal outcome (never improvise a quote).

Also: `start-all.sh` learns the 4th service (port 8004); README updates in both POCs to point at the new feature and the new env var.

## Dependency

Needs PR alpha (`feat/api-retrieval-poc`) running for live UI / chat smoke. Tests on this PR stub HTTP so they don't require it.

## Spec

spec: docs/specs/2026-06-05-quote-suggestions-rag-design.md (integration sections)

## Test plan

- [x] `pytest poc/chatbot/tests -v` -- previous 68 + 6 retrieval_client + 2 config + 4 dispatch = 80 passed
- [x] Rule 3 (`no_cjk_or_emoji`) -- exit 0
- [x] Manual end-to-end smoke against running retrieval server: English mood + Chinese mood via both UI and chat, plus failure-mode smoke (retrieval server down -> graceful error)

EOF
)"
```

- [ ] **Step 8: HARD STOP**

PR beta open. Do NOT admin-bypass. Wait for Jason review.

---

# Self-review (run after writing this plan)

## Spec coverage

| Spec section | Tasks | Notes |
|---|---|---|
| §1 Goal | All phases | ✓ |
| §2 Non-goals | Respected throughout (no Wikiquote runtime verify, no hosted vector stores, no corpus UI, no personalisation, no search_docs migration) | ✓ |
| §3 Architecture diagram | Tasks alpha.3-9 build each box | ✓ |
| §4 Corpus -- 1000 entries, auto build, fixed taxonomy, provenance | alpha.10 (script + run + JSON + README) | ✓ |
| §5.1 Embedder BGE-M3 + lazy load | alpha.3 | ✓ |
| §5.2 Store sqlite-vec | alpha.4 | ✓ |
| §6.1 /quotes endpoint shape | alpha.9 | ✓ |
| §6.2 /search endpoint shape | alpha.8 | ✓ |
| §6.3 /health endpoint shape | alpha.7 | ✓ |
| §7 Reranker prompt + 1-index picks + error modes | alpha.6 | ✓ |
| §8.1 text-clean-expand UI integration | beta.5 | ✓ |
| §8.2 chatbot find_quote tool | beta.1-4 | ✓ |
| §9 Tests (embedder, store, retriever, reranker, api, corpus) | alpha.3-9, alpha.12 | ✓ |
| §10 PR plan | Phases alpha + beta | ✓ |
| §11 Conventions + Phase 0 allowlist | Phase 0 tasks + Rule 3 checks throughout | ✓ |
| §12 Risks | Mitigations baked into individual tasks (lazy loading, 502 wrapping, README notes) | ✓ |
| §13 Open follow-ups | Deferred as spec says | ✓ |

No gaps identified.

## Placeholder scan

No "TBD", "TODO", or "implement later". Every code step contains full code. Every command has expected output. The one place that says "adapt to your existing test infrastructure" is in Phase 0 step 0.2 step 2 -- it's an honest acknowledgement that we don't know the dotgithub test helper's exact API without reading it. The shape of the assertions IS provided.

## Type consistency

- `Store(db_path, dim=1024)` -- defined alpha.4; used in alpha.5 retriever fixture, alpha.7 main `_open_store`, alpha.11 build_index, conftest `in_memory_store`. Consistent.
- `Retriever(store).search(query, k, lang)` -- defined alpha.5; used in alpha.7-9 main.py. Consistent.
- `embedder.encode(texts) / encode_one(text)` -- defined alpha.3; used by alpha.5, alpha.11. Consistent.
- `reranker.rerank(client, model, user_text, candidates, max_picks)` returning `[{index, why}]` -- defined alpha.6; used in alpha.9 main. Consistent.
- `RetrievalClient(base_url).quotes(text, max_picks, lang)` -- defined beta.1; used in beta.2 main wiring, beta.3 dispatch. Consistent.
- `ChatbotConfig.retrieval_client` -- added in beta.2; used in beta.3. Consistent.

All signatures match across tasks.

## Phase 0 dependency check

Task alpha.10 calls out "Phase 0 must be merged first" explicitly with a `gh pr view` command to verify. Task alpha.10 step 6 has a fallback `git pull` instruction if Rule 3 fails locally.

## End state

After all phases:
- Phase 0 PR merged into `MiraNote-AI/.github` main
- PR alpha (poc/retrieval/) open, awaiting Jason review
- PR beta (integration) open, awaiting Jason review, noted as depending on alpha
- main of miranote-api unchanged; all work isolated to feature branches

---

## Iterations (loop resumed 2026-06-11, Refs #18)

1. Surveyed true state past the stale 06-05 handoff: an interim session
   had already completed every alpha.10-13 artifact (corpus 500+500
   committed, fuzzy-dedup regen, build_index, test_corpus, README) plus
   hardening (reranker prompt-injection guard, Store transaction
   atomicity). Verified live: 29/29 tests, index rebuilt locally
   (1000 indexed, store count 1000), /quotes smoke EN+ZH both return
   on-topic picks with why-lines. Brought plan+spec onto this branch.
   Known limitation (backlog, not fixed here): zh corpus is
   author-skewed -- candidates come from the head of quan_tang_shi, so
   Emperor Taizong dominates; diversify sampling in a follow-up.
2. Hardening from maker-checker round 1 (verdict DONE with 5 WARN):
   cosine distance_metric on the vec0 table (scores were flooring at 0;
   live re-smoke now 0.646/0.621 on the EN probe), loop-until-stable
   delimiter strip + regression test (nested-tag bypass reproduced by
   reviewer), None-content guard, max_length on /quotes (2000) and
   /search (500) inputs, sanitized 502 detail (raw LLM output no longer
   echoed; existing test updated to the stricter contract), corpus
   README rewritten to state the true zh composition (500 Tang, single
   author) and the 75 empty-theme entries from 3 failed tagging batches.
   Suite 31 passed. Index rebuilt (cosine), 1000 indexed. Backlog (not
   this loop): corpus diversity rebuild, re-tag failed batches,
   atomicity test. Ops note: this entry was first committed to the
   wrong repo (dotgithub main, direct push) by a stateful-cd mistake
   and reverted there within minutes; lesson recorded.
