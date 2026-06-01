# Chatbot with tool calling — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a chatbot POC at `miranote-api/poc/chatbot/` that supports multi-turn chat and OpenAI-style native function calling, with a read-only file-system tool kit so the agent can answer questions about markdown docs under a configurable directory.

**Architecture:** FastAPI app + OpenAI-compatible client (DeepSeek by default). The chat loop runs server-side: model emits `tool_calls` → dispatcher executes each → results sent back to model → repeat until the model returns plain text or the iteration cap is hit. Sessions live in-memory keyed by uuid4. Tools are sandboxed under `DOCS_ROOT` and all reject paths that resolve outside it.

**Tech Stack:** Python 3.9+, FastAPI, uvicorn, `openai>=1.0` (used against DeepSeek's OpenAI-compatible endpoint), python-dotenv, pytest. Vanilla HTML/CSS/JS for the UI (no build step).

**Spec:** `/Users/mengjia/MiraNote/miranote-api/docs/superpowers/specs/2026-05-28-chatbot-with-tools-design.md`

**Branch:** Continue on `spec/chatbot-with-tools` (currently holds 2 spec commits). Rename to `feat/api-chatbot-poc` before opening the PR (Task 15).

**Conventions to honor:**
- Rule 3: no CJK / emoji in committed text outside `**/prompts/*.txt`, `**/static/*`, `**/demo_data/*`, `poc/*/README.md`. Run the check from `/Users/mengjia/MiraNote/dotgithub` against this repo before pushing.
- Python 3.9 compat: use `from __future__ import annotations` + `Optional[X]`, `List[X]` (Pydantic still eval()s annotations).
- Conventional Commits with `api` scope, ≤72 chars on the subject.
- PR title self-explanatory (no internal indices); references the spec doc.

---

## File Structure

| File | Responsibility |
|---|---|
| `poc/chatbot/__init__.py` | Empty; package marker. |
| `poc/chatbot/main.py` | FastAPI app, routes only. Wires the singletons (client, session store, dispatcher) into the chat loop. |
| `poc/chatbot/chat_loop.py` | Pure `run_turn(...)` that drives the tool-calling loop. All deps injected for testability. |
| `poc/chatbot/tools.py` | OpenAI tool-schema list (`TOOLS`) + `dispatch(name, args)` function. Wraps exceptions as `{"error": "..."}`. |
| `poc/chatbot/tools_fs.py` | `list_docs`, `read_doc`, `search_docs` + `_resolve_path` sandbox helper. Pure; takes `docs_root` as argument. |
| `poc/chatbot/session.py` | `SessionStore` — in-memory dict + TTL eviction; clock injected. |
| `poc/chatbot/prompts/system.txt` | English system prompt naming the tools. |
| `poc/chatbot/static/index.html` | Single-page chat UI, warm/cream palette matching siblings. |
| `poc/chatbot/demo_data/docs/product_overview.md` | Bilingual sample doc. |
| `poc/chatbot/demo_data/docs/roadmap_2026.md` | Bilingual sample doc. |
| `poc/chatbot/demo_data/docs/team.md` | Bilingual sample doc. |
| `poc/chatbot/tests/__init__.py` | Empty. |
| `poc/chatbot/tests/conftest.py` | pytest fixtures (`tmp_docs`, `fake_client_factory`). |
| `poc/chatbot/tests/test_tools_fs.py` | Path traversal, list/read/search behaviour + caps. |
| `poc/chatbot/tests/test_tools.py` | Dispatcher routing + error wrapping. |
| `poc/chatbot/tests/test_session.py` | Create/get/delete/TTL eviction. |
| `poc/chatbot/tests/test_chat_loop.py` | No-tool turn, single-tool turn, multi-step tool turn, cap-hit. |
| `poc/chatbot/.env.example` | LLM_*, DOCS_ROOT, caps. |
| `poc/chatbot/requirements.txt` | Pinned deps. |
| `poc/chatbot/README.md` | Bilingual usage doc + 3 canonical demo questions. |

---

## Task 0: Scaffold POC directory

**Files:**
- Create: `poc/chatbot/__init__.py`
- Create: `poc/chatbot/tests/__init__.py`
- Create: `poc/chatbot/requirements.txt`
- Create: `poc/chatbot/.env.example`
- Create: `poc/chatbot/.gitignore`

- [ ] **Step 1: Create the directory skeleton**

```bash
mkdir -p poc/chatbot/{prompts,static,demo_data/docs,tests}
touch poc/chatbot/__init__.py poc/chatbot/tests/__init__.py
```

- [ ] **Step 2: Write `poc/chatbot/requirements.txt`**

```
fastapi>=0.110
uvicorn>=0.29
openai>=1.0
python-dotenv>=1.0
pytest>=8.0
```

- [ ] **Step 3: Write `poc/chatbot/.env.example`**

```
# -- LLM (OpenAI-compatible API) --
# DeepSeek (default):
LLM_API_KEY=your-api-key-here
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
#
# Gemini OpenAI shim:
# LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
# LLM_MODEL=gemini-2.5-flash
#
# OpenAI:
# LLM_BASE_URL=
# LLM_MODEL=gpt-4o

# -- Chatbot config --
DOCS_ROOT=./demo_data/docs
MAX_TOOL_ITERATIONS=6
MAX_HISTORY_MESSAGES=40
SESSION_TTL_SECONDS=3600
```

- [ ] **Step 4: Write `poc/chatbot/.gitignore`**

```
.env
.venv/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 5: Verify Rule 3 still passes**

Run from `/Users/mengjia/MiraNote/dotgithub`:
```bash
PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```
Expected: `exit=0`.

- [ ] **Step 6: Commit**

```bash
cd /Users/mengjia/MiraNote/miranote-api
git add poc/chatbot/
git commit -m "feat(api): scaffold chatbot POC directory"
```

---

## Task 1: `tools_fs._resolve_path` (path-traversal sandbox)

**Files:**
- Create: `poc/chatbot/tools_fs.py`
- Create: `poc/chatbot/tests/conftest.py`
- Create: `poc/chatbot/tests/test_tools_fs.py`

- [ ] **Step 1: Write the conftest fixture**

Create `poc/chatbot/tests/conftest.py`:

```python
from __future__ import annotations
from pathlib import Path
import pytest


@pytest.fixture
def tmp_docs(tmp_path: Path) -> Path:
    """A temporary DOCS_ROOT with three small markdown files."""
    root = tmp_path / "docs"
    root.mkdir()
    (root / "alpha.md").write_text("# Alpha\nFirst doc body.\nApples and oranges.\n", encoding="utf-8")
    (root / "beta.md").write_text("# Beta\nSecond doc.\nBananas only.\n", encoding="utf-8")
    sub = root / "nested"
    sub.mkdir()
    (sub / "gamma.md").write_text("# Gamma\nNested doc body.\nApples again.\n", encoding="utf-8")
    return root
```

- [ ] **Step 2: Write the failing tests for `_resolve_path`**

Create `poc/chatbot/tests/test_tools_fs.py`:

```python
from __future__ import annotations
from pathlib import Path
import pytest

from poc.chatbot import tools_fs


def test_resolve_path_accepts_relative_inside_root(tmp_docs: Path):
    p = tools_fs._resolve_path(tmp_docs, "alpha.md")
    assert p == tmp_docs / "alpha.md"


def test_resolve_path_accepts_nested(tmp_docs: Path):
    p = tools_fs._resolve_path(tmp_docs, "nested/gamma.md")
    assert p == tmp_docs / "nested" / "gamma.md"


def test_resolve_path_rejects_parent_escape(tmp_docs: Path):
    with pytest.raises(ValueError, match="outside"):
        tools_fs._resolve_path(tmp_docs, "../escape.md")


def test_resolve_path_rejects_absolute_outside(tmp_docs: Path):
    with pytest.raises(ValueError, match="outside"):
        tools_fs._resolve_path(tmp_docs, "/etc/passwd")
```

- [ ] **Step 3: Run the tests to confirm they fail**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_tools_fs.py -v
```
Expected: `ModuleNotFoundError: No module named 'poc.chatbot.tools_fs'`.

- [ ] **Step 4: Implement `_resolve_path`**

Create `poc/chatbot/tools_fs.py`:

```python
"""Read-only file-system tools sandboxed under a docs root.

Every public function takes `docs_root: Path` so the module is pure and easy
to test. The dispatcher in `tools.py` binds the active `DOCS_ROOT` at call
time.
"""
from __future__ import annotations

from pathlib import Path


def _resolve_path(docs_root: Path, rel_or_abs: str) -> Path:
    """Resolve a user-supplied path and reject anything outside docs_root.

    Raises ValueError if the resolved path escapes the sandbox.
    """
    root = docs_root.resolve()
    candidate = (root / rel_or_abs).resolve() if not Path(rel_or_abs).is_absolute() else Path(rel_or_abs).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as e:
        raise ValueError(f"path '{rel_or_abs}' resolves outside DOCS_ROOT") from e
    return candidate
```

- [ ] **Step 5: Run the tests to confirm they pass**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_tools_fs.py -v
```
Expected: 4 passed.

- [ ] **Step 6: Commit**

```bash
git add poc/chatbot/tools_fs.py poc/chatbot/tests/conftest.py poc/chatbot/tests/test_tools_fs.py
git commit -m "feat(api): add chatbot fs sandbox path resolver"
```

---

## Task 2: `tools_fs.list_docs`

**Files:**
- Modify: `poc/chatbot/tools_fs.py` (add function)
- Modify: `poc/chatbot/tests/test_tools_fs.py` (add tests)

- [ ] **Step 1: Add failing tests**

Append to `poc/chatbot/tests/test_tools_fs.py`:

```python
def test_list_docs_root_returns_top_level(tmp_docs):
    out = tools_fs.list_docs(tmp_docs, ".")
    paths = sorted(item["path"] for item in out)
    assert paths == ["alpha.md", "beta.md", "nested/gamma.md"]
    for item in out:
        assert isinstance(item["size_bytes"], int) and item["size_bytes"] > 0


def test_list_docs_subdir(tmp_docs):
    out = tools_fs.list_docs(tmp_docs, "nested")
    paths = [item["path"] for item in out]
    assert paths == ["nested/gamma.md"]


def test_list_docs_rejects_escape(tmp_docs):
    with pytest.raises(ValueError, match="outside"):
        tools_fs.list_docs(tmp_docs, "../")
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_tools_fs.py::test_list_docs_root_returns_top_level -v
```
Expected: AttributeError or similar.

- [ ] **Step 3: Implement `list_docs`**

Append to `poc/chatbot/tools_fs.py`:

```python
from typing import Dict, List

_MAX_FILES = 200


def list_docs(docs_root: Path, subdir: str = ".") -> List[Dict[str, object]]:
    """Recursively list files under docs_root/subdir.

    Returns at most _MAX_FILES entries, each {"path": str (relative to docs_root), "size_bytes": int}.
    Hidden files and directories (leading dot) are skipped.
    """
    root = docs_root.resolve()
    start = _resolve_path(docs_root, subdir)
    if not start.exists() or not start.is_dir():
        raise ValueError(f"subdir '{subdir}' is not a directory under DOCS_ROOT")
    out: List[Dict[str, object]] = []
    for p in sorted(start.rglob("*")):
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if not p.is_file():
            continue
        out.append({
            "path": p.relative_to(root).as_posix(),
            "size_bytes": p.stat().st_size,
        })
        if len(out) >= _MAX_FILES:
            break
    return out
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_tools_fs.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add poc/chatbot/tools_fs.py poc/chatbot/tests/test_tools_fs.py
git commit -m "feat(api): add chatbot list_docs tool"
```

---

## Task 3: `tools_fs.read_doc`

**Files:**
- Modify: `poc/chatbot/tools_fs.py`
- Modify: `poc/chatbot/tests/test_tools_fs.py`

- [ ] **Step 1: Add failing tests**

Append to `poc/chatbot/tests/test_tools_fs.py`:

```python
def test_read_doc_returns_content(tmp_docs):
    out = tools_fs.read_doc(tmp_docs, "alpha.md")
    assert out["path"] == "alpha.md"
    assert "First doc body." in out["content"]
    assert out["truncated"] is False


def test_read_doc_truncates_large_file(tmp_docs):
    big = tmp_docs / "big.md"
    big.write_text("x" * (40 * 1024), encoding="utf-8")  # 40 KB
    out = tools_fs.read_doc(tmp_docs, "big.md")
    assert len(out["content"].encode("utf-8")) <= 32 * 1024
    assert out["truncated"] is True


def test_read_doc_rejects_escape(tmp_docs):
    with pytest.raises(ValueError, match="outside"):
        tools_fs.read_doc(tmp_docs, "../escape.md")


def test_read_doc_missing_file(tmp_docs):
    with pytest.raises(FileNotFoundError):
        tools_fs.read_doc(tmp_docs, "nope.md")
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_tools_fs.py -v -k read_doc
```
Expected: 4 failures (AttributeError: no `read_doc`).

- [ ] **Step 3: Implement `read_doc`**

Append to `poc/chatbot/tools_fs.py`:

```python
_MAX_BYTES = 32 * 1024


def read_doc(docs_root: Path, path: str) -> Dict[str, object]:
    """Read a UTF-8 text file under docs_root. Truncates to 32 KB."""
    target = _resolve_path(docs_root, path)
    if not target.exists() or not target.is_file():
        raise FileNotFoundError(f"file '{path}' not found under DOCS_ROOT")
    raw = target.read_bytes()
    truncated = len(raw) > _MAX_BYTES
    head = raw[:_MAX_BYTES]
    # Decode safely; replace any partial multibyte at the boundary.
    content = head.decode("utf-8", errors="replace")
    return {
        "path": target.relative_to(docs_root.resolve()).as_posix(),
        "content": content,
        "truncated": truncated,
    }
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_tools_fs.py -v
```
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add poc/chatbot/tools_fs.py poc/chatbot/tests/test_tools_fs.py
git commit -m "feat(api): add chatbot read_doc tool with 32KB cap"
```

---

## Task 4: `tools_fs.search_docs`

**Files:**
- Modify: `poc/chatbot/tools_fs.py`
- Modify: `poc/chatbot/tests/test_tools_fs.py`

- [ ] **Step 1: Add failing tests**

Append to `poc/chatbot/tests/test_tools_fs.py`:

```python
def test_search_docs_finds_substring(tmp_docs):
    hits = tools_fs.search_docs(tmp_docs, "apples")
    paths = sorted(h["path"] for h in hits)
    assert paths == ["alpha.md", "nested/gamma.md"]
    for h in hits:
        assert "apples" in h["snippet"].lower()
        assert h["line"] >= 1


def test_search_docs_case_insensitive(tmp_docs):
    assert tools_fs.search_docs(tmp_docs, "BANANAS")[0]["path"] == "beta.md"


def test_search_docs_max_hits(tmp_docs):
    busy = tmp_docs / "busy.md"
    busy.write_text("\n".join(["needle"] * 50), encoding="utf-8")
    hits = tools_fs.search_docs(tmp_docs, "needle", max_hits=5)
    assert len(hits) == 5


def test_search_docs_snippet_truncated(tmp_docs):
    long = tmp_docs / "long.md"
    long.write_text("x" * 500 + "needle" + "y" * 500, encoding="utf-8")
    hits = tools_fs.search_docs(tmp_docs, "needle")
    assert len(hits) == 1
    assert len(hits[0]["snippet"]) <= 160
    assert "needle" in hits[0]["snippet"]


def test_search_docs_empty_query_rejected(tmp_docs):
    with pytest.raises(ValueError, match="query"):
        tools_fs.search_docs(tmp_docs, "")
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_tools_fs.py -v -k search_docs
```
Expected: 5 failures.

- [ ] **Step 3: Implement `search_docs`**

Append to `poc/chatbot/tools_fs.py`:

```python
_SNIPPET_LEN = 160


def search_docs(docs_root: Path, query: str, max_hits: int = 20) -> List[Dict[str, object]]:
    """Case-insensitive substring search across files under docs_root.

    Returns at most max_hits entries of {"path", "line", "snippet"}.
    Skipped: hidden files/dirs, binaries (non-UTF-8), files larger than 1 MB.
    """
    if not query:
        raise ValueError("query must be non-empty")
    needle = query.lower()
    root = docs_root.resolve()
    hits: List[Dict[str, object]] = []
    files_scanned = 0
    for p in sorted(root.rglob("*")):
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        if not p.is_file():
            continue
        if p.stat().st_size > 1 * 1024 * 1024:
            continue
        files_scanned += 1
        if files_scanned > _MAX_FILES:
            break
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_no, line in enumerate(text.splitlines(), start=1):
            if needle in line.lower():
                col = line.lower().index(needle)
                start = max(0, col - 40)
                snippet = line[start:start + _SNIPPET_LEN]
                hits.append({
                    "path": p.relative_to(root).as_posix(),
                    "line": line_no,
                    "snippet": snippet,
                })
                if len(hits) >= max_hits:
                    return hits
    return hits
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_tools_fs.py -v
```
Expected: 16 passed.

- [ ] **Step 5: Commit**

```bash
git add poc/chatbot/tools_fs.py poc/chatbot/tests/test_tools_fs.py
git commit -m "feat(api): add chatbot search_docs substring tool"
```

---

## Task 5: Tool registry + dispatcher (`tools.py`)

**Files:**
- Create: `poc/chatbot/tools.py`
- Create: `poc/chatbot/tests/test_tools.py`

- [ ] **Step 1: Write failing tests**

Create `poc/chatbot/tests/test_tools.py`:

```python
from __future__ import annotations
from pathlib import Path

from poc.chatbot import tools


def test_tools_schema_lists_three_functions():
    names = [t["function"]["name"] for t in tools.TOOLS]
    assert sorted(names) == ["list_docs", "read_doc", "search_docs"]
    for t in tools.TOOLS:
        assert t["type"] == "function"
        assert "parameters" in t["function"]


def test_dispatch_routes_to_list_docs(tmp_docs: Path):
    out = tools.dispatch(tmp_docs, "list_docs", {"subdir": "."})
    assert isinstance(out, list)
    assert any(item["path"] == "alpha.md" for item in out)


def test_dispatch_routes_to_read_doc(tmp_docs: Path):
    out = tools.dispatch(tmp_docs, "read_doc", {"path": "alpha.md"})
    assert "First doc body." in out["content"]


def test_dispatch_routes_to_search_docs(tmp_docs: Path):
    out = tools.dispatch(tmp_docs, "search_docs", {"query": "apples"})
    assert len(out) >= 1


def test_dispatch_unknown_tool_returns_error(tmp_docs: Path):
    out = tools.dispatch(tmp_docs, "nope", {})
    assert "error" in out
    assert "unknown tool" in out["error"].lower()


def test_dispatch_wraps_exceptions(tmp_docs: Path):
    out = tools.dispatch(tmp_docs, "read_doc", {"path": "../escape.md"})
    assert "error" in out
    assert "outside" in out["error"].lower()


def test_dispatch_missing_required_arg(tmp_docs: Path):
    out = tools.dispatch(tmp_docs, "search_docs", {})
    assert "error" in out
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_tools.py -v
```
Expected: `ModuleNotFoundError: poc.chatbot.tools`.

- [ ] **Step 3: Implement `tools.py`**

Create `poc/chatbot/tools.py`:

```python
"""Tool registry + dispatcher for the chatbot POC."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from poc.chatbot import tools_fs


TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_docs",
            "description": "List all files under a subdirectory of the docs root. Returns relative paths and sizes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subdir": {
                        "type": "string",
                        "description": "Subdirectory relative to the docs root. Use '.' for the root itself.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_doc",
            "description": "Read the UTF-8 contents of a file under the docs root. Truncates at 32 KB.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to the docs root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Case-insensitive substring search across all files under the docs root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Substring to search for."},
                    "max_hits": {"type": "integer", "description": "Maximum hits to return (default 20).", "default": 20},
                },
                "required": ["query"],
            },
        },
    },
]


def dispatch(docs_root: Path, name: str, args: Dict[str, Any]) -> Any:
    """Route a model-issued tool call to the underlying implementation.

    Always returns a JSON-serialisable value. Exceptions become {"error": "..."}
    so the model can recover on the next turn.
    """
    try:
        if name == "list_docs":
            return tools_fs.list_docs(docs_root, args.get("subdir", "."))
        if name == "read_doc":
            return tools_fs.read_doc(docs_root, args["path"])
        if name == "search_docs":
            return tools_fs.search_docs(docs_root, args["query"], int(args.get("max_hits", 20)))
        return {"error": f"unknown tool: {name}"}
    except KeyError as e:
        return {"error": f"missing required argument: {e.args[0]}"}
    except Exception as e:  # noqa: BLE001  -- intentionally broad; model needs the message
        return {"error": str(e)}
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_tools.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add poc/chatbot/tools.py poc/chatbot/tests/test_tools.py
git commit -m "feat(api): add chatbot tool registry and dispatcher"
```

---

## Task 6: Session store (`session.py`)

**Files:**
- Create: `poc/chatbot/session.py`
- Create: `poc/chatbot/tests/test_session.py`

- [ ] **Step 1: Write failing tests**

Create `poc/chatbot/tests/test_session.py`:

```python
from __future__ import annotations
import pytest

from poc.chatbot.session import SessionStore


def test_create_returns_uuid():
    clock = [0.0]
    store = SessionStore(ttl_seconds=60, clock=lambda: clock[0])
    sid = store.create(seed=[{"role": "system", "content": "hi"}])
    assert isinstance(sid, str) and len(sid) >= 32
    assert store.get(sid) == [{"role": "system", "content": "hi"}]


def test_append_extends_history():
    clock = [0.0]
    store = SessionStore(ttl_seconds=60, clock=lambda: clock[0])
    sid = store.create(seed=[{"role": "system", "content": "hi"}])
    store.append(sid, {"role": "user", "content": "hello"})
    msgs = store.get(sid)
    assert msgs[-1] == {"role": "user", "content": "hello"}


def test_replace_overwrites_history():
    clock = [0.0]
    store = SessionStore(ttl_seconds=60, clock=lambda: clock[0])
    sid = store.create(seed=[{"role": "system", "content": "hi"}])
    new = [{"role": "system", "content": "hi"}, {"role": "user", "content": "x"}]
    store.replace(sid, new)
    assert store.get(sid) == new


def test_delete_removes_session():
    clock = [0.0]
    store = SessionStore(ttl_seconds=60, clock=lambda: clock[0])
    sid = store.create(seed=[])
    store.delete(sid)
    with pytest.raises(KeyError):
        store.get(sid)


def test_unknown_session_raises():
    store = SessionStore(ttl_seconds=60)
    with pytest.raises(KeyError):
        store.get("bogus")


def test_ttl_evicts_old_sessions():
    clock = [0.0]
    store = SessionStore(ttl_seconds=10, clock=lambda: clock[0])
    sid = store.create(seed=[{"role": "system", "content": "hi"}])
    clock[0] = 5
    assert store.get(sid)  # still alive
    clock[0] = 20  # past TTL
    with pytest.raises(KeyError):
        store.get(sid)


def test_access_refreshes_ttl():
    clock = [0.0]
    store = SessionStore(ttl_seconds=10, clock=lambda: clock[0])
    sid = store.create(seed=[])
    clock[0] = 8
    store.get(sid)        # refresh
    clock[0] = 15         # 7 since last touch -> alive
    assert store.get(sid) == []
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_session.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement `session.py`**

Create `poc/chatbot/session.py`:

```python
"""In-memory session store keyed by uuid4, with TTL eviction.

`clock` is injected so tests can advance time deterministically.
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Dict, List, Optional


class SessionStore:
    def __init__(self, ttl_seconds: int, clock: Optional[Callable[[], float]] = None):
        self._ttl = ttl_seconds
        self._clock = clock or time.time
        self._data: Dict[str, List[Dict[str, Any]]] = {}
        self._last_touched: Dict[str, float] = {}

    def create(self, seed: List[Dict[str, Any]]) -> str:
        sid = uuid.uuid4().hex
        self._data[sid] = list(seed)
        self._last_touched[sid] = self._clock()
        return sid

    def get(self, sid: str) -> List[Dict[str, Any]]:
        self._evict_expired()
        if sid not in self._data:
            raise KeyError(sid)
        self._last_touched[sid] = self._clock()
        return self._data[sid]

    def append(self, sid: str, msg: Dict[str, Any]) -> None:
        self.get(sid).append(msg)

    def replace(self, sid: str, msgs: List[Dict[str, Any]]) -> None:
        self.get(sid)  # also refreshes TTL / asserts exists
        self._data[sid] = list(msgs)

    def delete(self, sid: str) -> None:
        self._data.pop(sid, None)
        self._last_touched.pop(sid, None)

    def _evict_expired(self) -> None:
        now = self._clock()
        dead = [sid for sid, t in self._last_touched.items() if now - t > self._ttl]
        for sid in dead:
            self.delete(sid)
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_session.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add poc/chatbot/session.py poc/chatbot/tests/test_session.py
git commit -m "feat(api): add chatbot in-memory session store with TTL"
```

---

## Task 7: Chat loop — no-tool path (`chat_loop.py`)

**Files:**
- Create: `poc/chatbot/chat_loop.py`
- Create: `poc/chatbot/tests/test_chat_loop.py`

- [ ] **Step 1: Write failing test for the simplest path (model returns text immediately)**

Create `poc/chatbot/tests/test_chat_loop.py`:

```python
from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from poc.chatbot.chat_loop import run_turn, ChatTurnResult
from poc.chatbot.session import SessionStore


def _msg(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _resp(message):
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeChatCompletions:
    def __init__(self, scripted_responses):
        self._scripted = list(scripted_responses)
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._scripted.pop(0)


class FakeClient:
    def __init__(self, scripted_responses):
        self.chat = SimpleNamespace(completions=FakeChatCompletions(scripted_responses))


def test_no_tool_call_returns_text(tmp_path):
    client = FakeClient([_resp(_msg(content="hello back"))])
    store = SessionStore(ttl_seconds=60)
    result = run_turn(
        client=client,
        session_store=store,
        session_id=None,
        user_message="hi",
        model="fake-model",
        tools=[],
        tool_dispatcher=lambda name, args: {"error": "no tools"},
        max_iterations=6,
        max_history=40,
        system_prompt="you are a helper",
    )
    assert isinstance(result, ChatTurnResult)
    assert result.reply == "hello back"
    assert result.tool_trace == []
    history = store.get(result.session_id)
    assert history[0] == {"role": "system", "content": "you are a helper"}
    assert history[1] == {"role": "user", "content": "hi"}
    assert history[2]["role"] == "assistant"
    assert history[2]["content"] == "hello back"
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_chat_loop.py -v
```
Expected: ModuleNotFoundError.

- [ ] **Step 3: Implement the no-tool path**

Create `poc/chatbot/chat_loop.py`:

```python
"""Server-side chat loop. Pure -- all deps injected for testability."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from poc.chatbot.session import SessionStore


@dataclass
class ChatTurnResult:
    session_id: str
    reply: str
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)


def run_turn(
    *,
    client: Any,
    session_store: SessionStore,
    session_id: Optional[str],
    user_message: str,
    model: str,
    tools: List[Dict[str, Any]],
    tool_dispatcher: Callable[[str, Dict[str, Any]], Any],
    max_iterations: int,
    max_history: int,
    system_prompt: str,
) -> ChatTurnResult:
    if session_id is None:
        session_id = session_store.create(seed=[{"role": "system", "content": system_prompt}])
    history = session_store.get(session_id)
    history.append({"role": "user", "content": user_message})

    trace: List[Dict[str, Any]] = []
    reply: Optional[str] = None
    for _ in range(max_iterations):
        kwargs = {"model": model, "messages": history}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        if not getattr(msg, "tool_calls", None):
            reply = msg.content or ""
            history.append({"role": "assistant", "content": reply})
            break
        # tool-call branch -- implemented in Task 8
        raise NotImplementedError("tool-call path lands in Task 8")

    if reply is None:
        reply = "(stopped: hit MAX_TOOL_ITERATIONS -- partial tool use, no final answer)"
        history.append({"role": "assistant", "content": reply})

    _trim_history(history, max_history)
    session_store.replace(session_id, history)
    return ChatTurnResult(session_id=session_id, reply=reply, tool_trace=trace)


def _trim_history(history: List[Dict[str, Any]], max_history: int) -> None:
    """Drop oldest non-system messages in pairs until we're under the cap."""
    non_system = [i for i, m in enumerate(history) if m.get("role") != "system"]
    while len(non_system) > max_history:
        drop_idx = non_system[0]
        del history[drop_idx]
        non_system = [i for i, m in enumerate(history) if m.get("role") != "system"]
```

- [ ] **Step 4: Run test to confirm pass**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_chat_loop.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add poc/chatbot/chat_loop.py poc/chatbot/tests/test_chat_loop.py
git commit -m "feat(api): add chatbot run_turn skeleton (no-tool path)"
```

---

## Task 8: Chat loop — tool-call path

**Files:**
- Modify: `poc/chatbot/chat_loop.py`
- Modify: `poc/chatbot/tests/test_chat_loop.py`

- [ ] **Step 1: Add failing tests for single-step and multi-step tool use**

Append to `poc/chatbot/tests/test_chat_loop.py`:

```python
def _tool_call(call_id, name, args_dict):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args_dict)),
    )


# Re-import to keep test file self-contained
import json  # noqa: E402


def test_single_tool_call_then_text():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("call_1", "search_docs", {"query": "apples"})])),
        _resp(_msg(content="found apples in alpha.md")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)
    captured = []

    def dispatcher(name, args):
        captured.append((name, args))
        return [{"path": "alpha.md", "line": 3, "snippet": "Apples and oranges."}]

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="find apples", model="fake", tools=[{"type": "function", "function": {"name": "search_docs"}}],
        tool_dispatcher=dispatcher, max_iterations=6, max_history=40,
        system_prompt="sys",
    )
    assert result.reply == "found apples in alpha.md"
    assert captured == [("search_docs", {"query": "apples"})]
    assert len(result.tool_trace) == 1
    assert result.tool_trace[0]["name"] == "search_docs"
    assert result.tool_trace[0]["args"] == {"query": "apples"}
    history = store.get(result.session_id)
    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "call_1"


def test_multi_step_tool_use():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "list_docs", {"subdir": "."})])),
        _resp(_msg(tool_calls=[_tool_call("c2", "read_doc", {"path": "alpha.md"})])),
        _resp(_msg(content="alpha says hello")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)

    def dispatcher(name, args):
        return {"ok": True, "name": name}

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="look around", model="fake", tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=dispatcher, max_iterations=6, max_history=40,
        system_prompt="sys",
    )
    assert result.reply == "alpha says hello"
    assert [t["name"] for t in result.tool_trace] == ["list_docs", "read_doc"]


def test_dispatcher_errors_are_wrapped_for_model():
    scripted = [
        _resp(_msg(tool_calls=[_tool_call("c1", "read_doc", {"path": "../escape"})])),
        _resp(_msg(content="oh that was rejected")),
    ]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)

    def dispatcher(name, args):
        raise ValueError("path outside DOCS_ROOT")

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="read escape", model="fake", tools=[{"type": "function", "function": {"name": "x"}}],
        tool_dispatcher=dispatcher, max_iterations=6, max_history=40,
        system_prompt="sys",
    )
    assert result.reply == "oh that was rejected"
    history = store.get(result.session_id)
    tool_msgs = [m for m in history if m.get("role") == "tool"]
    assert "error" in tool_msgs[0]["content"]
    assert "outside" in tool_msgs[0]["content"]
```

- [ ] **Step 2: Run to confirm failure**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_chat_loop.py -v
```
Expected: 3 new tests fail with `NotImplementedError: tool-call path lands in Task 8`.

- [ ] **Step 3: Implement the tool-call branch**

Replace the `# tool-call branch -- implemented in Task 8` block in `poc/chatbot/chat_loop.py` with this loop body. The full `run_turn` function should look like:

```python
def run_turn(
    *,
    client: Any,
    session_store: SessionStore,
    session_id: Optional[str],
    user_message: str,
    model: str,
    tools: List[Dict[str, Any]],
    tool_dispatcher: Callable[[str, Dict[str, Any]], Any],
    max_iterations: int,
    max_history: int,
    system_prompt: str,
) -> ChatTurnResult:
    if session_id is None:
        session_id = session_store.create(seed=[{"role": "system", "content": system_prompt}])
    history = session_store.get(session_id)
    history.append({"role": "user", "content": user_message})

    trace: List[Dict[str, Any]] = []
    reply: Optional[str] = None
    for _ in range(max_iterations):
        kwargs = {"model": model, "messages": history}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        if not getattr(msg, "tool_calls", None):
            reply = msg.content or ""
            history.append({"role": "assistant", "content": reply})
            break

        assistant_entry: Dict[str, Any] = {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        }
        history.append(assistant_entry)

        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = tool_dispatcher(tc.function.name, args)
            except Exception as e:  # noqa: BLE001
                result = {"error": str(e)}
            result_str = json.dumps(result, ensure_ascii=False)
            history.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": result_str,
            })
            trace.append({
                "name": tc.function.name,
                "args": args,
                "result_preview": result_str[:300],
            })

    if reply is None:
        reply = "(stopped: hit MAX_TOOL_ITERATIONS -- partial tool use, no final answer)"
        history.append({"role": "assistant", "content": reply})

    _trim_history(history, max_history)
    session_store.replace(session_id, history)
    return ChatTurnResult(session_id=session_id, reply=reply, tool_trace=trace)
```

- [ ] **Step 4: Run tests to confirm pass**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_chat_loop.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add poc/chatbot/chat_loop.py poc/chatbot/tests/test_chat_loop.py
git commit -m "feat(api): wire chatbot tool-call loop with error wrapping"
```

---

## Task 9: Chat loop — iteration-cap path

**Files:**
- Modify: `poc/chatbot/tests/test_chat_loop.py`

- [ ] **Step 1: Add failing test**

Append to `poc/chatbot/tests/test_chat_loop.py`:

```python
def test_iteration_cap_returns_synthetic_reply():
    # 8 scripted responses, but cap = 3 -> loop should give up at 3.
    scripted = [_resp(_msg(tool_calls=[_tool_call(f"c{i}", "list_docs", {"subdir": "."})])) for i in range(8)]
    client = FakeClient(scripted)
    store = SessionStore(ttl_seconds=60)

    result = run_turn(
        client=client, session_store=store, session_id=None,
        user_message="loop forever", model="fake",
        tools=[{"type": "function", "function": {"name": "list_docs"}}],
        tool_dispatcher=lambda n, a: [],
        max_iterations=3, max_history=100, system_prompt="sys",
    )
    assert "stopped" in result.reply.lower()
    assert "max_tool_iterations" in result.reply.lower()
    # The fake should have been called exactly 3 times.
    assert len(client.chat.completions.calls) == 3
    assert len(result.tool_trace) == 3
```

- [ ] **Step 2: Run -- should already pass thanks to Task 8 implementation**

```bash
PYTHONPATH=. python3 -m pytest poc/chatbot/tests/test_chat_loop.py::test_iteration_cap_returns_synthetic_reply -v
```
Expected: 1 passed. (If it fails, the cap behaviour in `run_turn` needs a tweak — re-check Task 8 implementation.)

- [ ] **Step 3: Commit**

```bash
git add poc/chatbot/tests/test_chat_loop.py
git commit -m "test(api): pin chatbot iteration-cap behaviour"
```

---

## Task 10: System prompt + demo docs

**Files:**
- Create: `poc/chatbot/prompts/system.txt`
- Create: `poc/chatbot/demo_data/docs/product_overview.md`
- Create: `poc/chatbot/demo_data/docs/roadmap_2026.md`
- Create: `poc/chatbot/demo_data/docs/team.md`

- [ ] **Step 1: Write `poc/chatbot/prompts/system.txt`**

```
You are MiraNote's documentation assistant. Answer the user's questions
about the documents stored under the docs root. You have these tools:

- list_docs(subdir): list all files under a subdirectory.
- read_doc(path): read the contents of a specific file.
- search_docs(query, max_hits): case-insensitive substring search.

Be concise. When the answer comes from a doc, cite the filename. If you
do not have enough information after using the tools, say so plainly --
do not invent facts.

The user may write in English or Chinese. Reply in the language they used.
```

(Note: `prompts/system.txt` is allowlisted by Rule 3 — the Chinese reference
in the last line is fine here. Source code must still stay ASCII.)

- [ ] **Step 2: Write `poc/chatbot/demo_data/docs/product_overview.md`**

```markdown
# MiraNote — Product Overview

MiraNote is an AI-augmented note-taking workspace for product, design,
and engineering teams. It blends a fast Markdown editor with on-device
voice capture, real-time text cleanup, and a retrieval layer that lets
teammates ask questions across the shared knowledge base.

## Surfaces

- **iOS app** -- primary capture surface (voice + quick text).
- **Web app** -- the workspace, where notes live and are organized.
- **Discord bot** -- low-friction question/answer in the team channel.

## Why teams pick MiraNote

1. Voice-first capture that respects bilingual speech (Chinese + English).
2. AI cleanup is a button press, never automatic. Users stay in control.
3. The knowledge base is yours -- no vendor lock-in, export anytime.

中文摘要：MiraNote 是一个面向中英双语团队的 AI 笔记工作区，强调录音
转写、人工触发的文本清洁，以及跨团队的知识检索。
```

- [ ] **Step 3: Write `poc/chatbot/demo_data/docs/roadmap_2026.md`**

```markdown
# MiraNote 2026 Roadmap

## Q1 2026 (shipped)

- Day-0 repo hygiene -- `.github` checks, CLAUDE.md, sub-project tracker.
- Voice-to-text POC (Whisper + LLM correction).
- Text clean & expand POC.

## Q2 2026 (in progress)

- Chatbot POC with native function calling.
- iOS capture beta -- closed alpha with 10 design partners.
- Discord bot first-touch responses.

## Q3 2026 (planned)

- Web app private beta. Markdown editor, folders, search.
- Voice-to-text moves out of POC into shared `api/` service.
- Retrieval layer over the shared knowledge base (vector index).

## Q4 2026 (planned)

- iOS public beta.
- Pricing experiments. Pilot with 3 paying teams.
- SOC 2 Type 1 scoping kickoff.

中文摘要：2026 Q1 已完成基础设施和两个 POC；Q2 重点是 chatbot 和 iOS；
Q3 推出 Web 私测；Q4 进入公测和商业化。
```

- [ ] **Step 4: Write `poc/chatbot/demo_data/docs/team.md`**

```markdown
# MiraNote Team

## Roles

- **mengjia** -- founder, full-stack, owns infrastructure and the web app.
- **Jason (Jiachen Zhong)** -- API and POC work, owns text-clean-expand
  and contributes to chatbot tooling.
- **(unfilled)** -- iOS lead. Currently shared between mengjia and contractors.

## Working agreements

- Every PR has either an issue, a spec, an ADR, or a URL in its body.
- Day-0 rules in `CONTRIBUTING.md` are enforced by `checks/` scripts.
- Voice-to-text and text-clean-expand POCs are bilingual by design.
- Internal communication happens in Discord (#general for announcements,
  DMs reserved for genuinely private topics).

中文备注：iOS 负责人空缺；所有 PR 需要附 issue/spec 引用；Discord 群
默认讨论，DM 仅用于私事。
```

- [ ] **Step 5: Verify Rule 3 still passes**

```bash
cd /Users/mengjia/MiraNote/dotgithub
PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```
Expected: `exit=0`.

- [ ] **Step 6: Commit**

```bash
cd /Users/mengjia/MiraNote/miranote-api
git add poc/chatbot/prompts/ poc/chatbot/demo_data/
git commit -m "feat(api): add chatbot system prompt and demo docs"
```

---

## Task 11: FastAPI app (`main.py`)

**Files:**
- Create: `poc/chatbot/main.py`

- [ ] **Step 1: Write `poc/chatbot/main.py`**

```python
"""MiraNote POC -- Chatbot with native function calling."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

from poc.chatbot import tools
from poc.chatbot.chat_loop import ChatTurnResult, run_turn
from poc.chatbot.session import SessionStore


load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")
MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
DOCS_ROOT = Path(os.getenv("DOCS_ROOT", "./demo_data/docs")).resolve()
MAX_TOOL_ITERATIONS = int(os.getenv("MAX_TOOL_ITERATIONS", "6"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "40"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "3600"))

if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY is required. Set it in .env")
if not DOCS_ROOT.exists() or not DOCS_ROOT.is_dir():
    raise RuntimeError(f"DOCS_ROOT does not exist or is not a directory: {DOCS_ROOT}")

client_kwargs: Dict[str, Any] = {"api_key": LLM_API_KEY}
if LLM_BASE_URL:
    client_kwargs["base_url"] = LLM_BASE_URL
client = OpenAI(**client_kwargs)

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.txt").read_text(encoding="utf-8")

sessions = SessionStore(ttl_seconds=SESSION_TTL_SECONDS)


def _dispatcher(name: str, args: Dict[str, Any]) -> Any:
    return tools.dispatch(DOCS_ROOT, name, args)


app = FastAPI(title="MiraNote Chatbot", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_trace: List[Dict[str, Any]]


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result: ChatTurnResult = await asyncio.to_thread(
            run_turn,
            client=client,
            session_store=sessions,
            session_id=req.session_id,
            user_message=req.message,
            model=MODEL,
            tools=tools.TOOLS,
            tool_dispatcher=_dispatcher,
            max_iterations=MAX_TOOL_ITERATIONS,
            max_history=MAX_HISTORY_MESSAGES,
            system_prompt=SYSTEM_PROMPT,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session_id")
    except Exception as e:  # noqa: BLE001 -- surface LLM/network errors
        raise HTTPException(status_code=502, detail=f"chat failed: {e}")
    return ChatResponse(
        session_id=result.session_id,
        reply=result.reply,
        tool_trace=result.tool_trace,
    )


@app.get("/sessions/{sid}")
async def get_session(sid: str):
    try:
        return {"messages": sessions.get(sid)}
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session_id")


@app.delete("/sessions/{sid}")
async def delete_session(sid: str):
    sessions.delete(sid)
    return {"status": "deleted"}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "tools": [t["function"]["name"] for t in tools.TOOLS],
        "docs_root": str(DOCS_ROOT),
    }


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
```

- [ ] **Step 2: Sanity-check the import (no LLM call yet)**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. LLM_API_KEY=fake DOCS_ROOT=poc/chatbot/demo_data/docs python3 -c "from poc.chatbot import main; print('imported ok, tools:', [t['function']['name'] for t in main.tools.TOOLS])"
```
Expected: `imported ok, tools: ['list_docs', 'read_doc', 'search_docs']`.

- [ ] **Step 3: Commit**

```bash
git add poc/chatbot/main.py
git commit -m "feat(api): add chatbot FastAPI app with /chat /sessions /health"
```

---

## Task 12: Web UI (`static/index.html`)

**Files:**
- Create: `poc/chatbot/static/index.html`

- [ ] **Step 1: Write `poc/chatbot/static/index.html`**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MiraNote Chatbot POC</title>
  <style>
    :root {
      --bg: #f8f7f4;
      --panel: #ffffff;
      --border: #e5e2dc;
      --text: #1a1a1a;
      --muted: #8a8580;
      --accent: #5b4f3f;
      --accent-hover: #3d3429;
      --user-bg: #f5f0e6;
      --tool-bg: #f0ede6;
      --warm: #c9a96e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }
    .container { max-width: 920px; margin: 0 auto; padding: 32px 24px 120px; }
    .logo { font-size: 24px; font-weight: 700; letter-spacing: -0.5px; }
    .logo span { color: var(--warm); }
    .subtitle { color: var(--muted); font-size: 13px; margin: 2px 0 24px; }

    .chat { display: flex; flex-direction: column; gap: 12px; min-height: 50vh; }
    .turn { display: flex; }
    .turn.user { justify-content: flex-end; }
    .bubble {
      max-width: 78%;
      padding: 12px 16px;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: var(--panel);
      white-space: pre-wrap;
      word-wrap: break-word;
    }
    .turn.user .bubble { background: var(--user-bg); }
    .turn.assistant .bubble { background: var(--panel); }

    .tools { margin-top: 8px; display: flex; flex-direction: column; gap: 4px; }
    .tool-chip {
      font-size: 12px;
      color: var(--muted);
      background: var(--tool-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 6px 10px;
      cursor: pointer;
      user-select: none;
    }
    .tool-chip pre {
      display: none;
      margin-top: 6px;
      font-size: 11px;
      white-space: pre-wrap;
      word-wrap: break-word;
      color: var(--text);
    }
    .tool-chip.open pre { display: block; }

    .composer {
      position: fixed; bottom: 0; left: 0; right: 0;
      background: var(--bg);
      border-top: 1px solid var(--border);
      padding: 14px 24px;
    }
    .composer-inner { max-width: 920px; margin: 0 auto; display: flex; gap: 8px; }
    textarea {
      flex: 1; min-height: 56px; max-height: 200px;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: var(--panel);
      font: inherit;
      resize: vertical;
    }
    button {
      border: none; padding: 0 18px; border-radius: 10px;
      background: var(--accent); color: white;
      font: inherit; cursor: pointer;
    }
    button:hover { background: var(--accent-hover); }
    button.secondary { background: transparent; color: var(--accent); border: 1px solid var(--border); }
    button:disabled { opacity: 0.4; cursor: not-allowed; }

    .status { font-size: 11px; color: var(--muted); margin-top: 8px; text-align: center; }
    .err { color: #c45c5c; }
  </style>
</head>
<body>
  <div class="container">
    <div class="logo">Mira<span>Note</span> Chatbot</div>
    <div class="subtitle">Ask questions about the docs in <code id="docs-root">DOCS_ROOT</code>.</div>

    <div id="chat" class="chat"></div>
  </div>

  <div class="composer">
    <div class="composer-inner">
      <textarea id="input" placeholder="Ask about the docs... (Cmd/Ctrl+Enter to send)"></textarea>
      <button id="send">Send</button>
      <button id="reset" class="secondary">Reset</button>
    </div>
    <div class="status" id="status">model: ... | docs: ...</div>
  </div>

  <script>
    const chatEl = document.getElementById('chat');
    const inputEl = document.getElementById('input');
    const sendBtn = document.getElementById('send');
    const resetBtn = document.getElementById('reset');
    const statusEl = document.getElementById('status');
    const docsRootEl = document.getElementById('docs-root');

    let sessionId = null;

    async function loadHealth() {
      try {
        const r = await fetch('/health').then(r => r.json());
        statusEl.textContent = `model: ${r.model}  |  docs: ${r.docs_root}  |  tools: ${r.tools.join(', ')}`;
        docsRootEl.textContent = r.docs_root;
      } catch (e) {
        statusEl.textContent = 'health check failed';
        statusEl.classList.add('err');
      }
    }

    function renderTurn(role, content, toolTrace) {
      const turn = document.createElement('div');
      turn.className = `turn ${role}`;
      const bubble = document.createElement('div');
      bubble.className = 'bubble';
      bubble.textContent = content;
      turn.appendChild(bubble);
      if (toolTrace && toolTrace.length) {
        const tools = document.createElement('div');
        tools.className = 'tools';
        toolTrace.forEach(t => {
          const chip = document.createElement('div');
          chip.className = 'tool-chip';
          const label = document.createElement('div');
          label.textContent = `tool: ${t.name}(${JSON.stringify(t.args)})`;
          const pre = document.createElement('pre');
          pre.textContent = t.result_preview;
          chip.appendChild(label);
          chip.appendChild(pre);
          chip.onclick = () => chip.classList.toggle('open');
          tools.appendChild(chip);
        });
        bubble.appendChild(tools);
      }
      chatEl.appendChild(turn);
      window.scrollTo(0, document.body.scrollHeight);
    }

    async function send() {
      const text = inputEl.value.trim();
      if (!text) return;
      sendBtn.disabled = true;
      renderTurn('user', text);
      inputEl.value = '';
      try {
        const r = await fetch('/chat', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({ session_id: sessionId, message: text }),
        });
        if (!r.ok) {
          const err = await r.json().catch(() => ({ detail: r.statusText }));
          renderTurn('assistant', `error ${r.status}: ${err.detail}`);
          return;
        }
        const data = await r.json();
        sessionId = data.session_id;
        renderTurn('assistant', data.reply, data.tool_trace);
      } catch (e) {
        renderTurn('assistant', `network error: ${e.message}`);
      } finally {
        sendBtn.disabled = false;
        inputEl.focus();
      }
    }

    async function reset() {
      if (sessionId) {
        try { await fetch(`/sessions/${sessionId}`, { method: 'DELETE' }); } catch {}
      }
      sessionId = null;
      chatEl.innerHTML = '';
      inputEl.focus();
    }

    sendBtn.onclick = send;
    resetBtn.onclick = reset;
    inputEl.addEventListener('keydown', e => {
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        send();
      }
    });

    loadHealth();
    inputEl.focus();
  </script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add poc/chatbot/static/index.html
git commit -m "feat(api): add chatbot single-page UI"
```

---

## Task 13: README + end-to-end manual demo

**Files:**
- Create: `poc/chatbot/README.md`

- [ ] **Step 1: Write `poc/chatbot/README.md`**

```markdown
# Chatbot POC

A FastAPI demo of multi-turn chat with native OpenAI-style function
calling. The agent answers questions about the markdown documents in
`DOCS_ROOT` using three read-only tools: `list_docs`, `read_doc`,
`search_docs`.

Design spec: `docs/superpowers/specs/2026-05-28-chatbot-with-tools-design.md`.

## Setup

```bash
cd poc/chatbot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# edit .env: put your LLM_API_KEY in, pick a provider
```

DeepSeek is the default. Any OpenAI-compatible provider that supports
tool calling works (Gemini's OpenAI shim, OpenAI proper, Moonshot).

## Run

```bash
cd poc/chatbot
PYTHONPATH=../.. uvicorn main:app --reload --port 8002
```

Then open <http://localhost:8002/>.

## Try it (curl)

```bash
# First turn (no session_id -> server mints one and returns it)
curl -s -X POST http://localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"what files are available?"}' | python3 -m json.tool

# Follow-up using the returned session_id
curl -s -X POST http://localhost:8002/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"<paste-here>","message":"summarize the Q3 plans"}' | python3 -m json.tool
```

## Three demo questions

Try these in the web UI against the bundled `demo_data/docs/`:

1. **English / `list_docs`:** _"What docs do we have?"_
2. **English / `read_doc`:** _"What ships in Q3 2026?"_
3. **中文 / `search_docs`:** _"团队里谁负责 iOS?"_

You should see tool chips under each assistant reply showing exactly
which tools were called and what came back.

## Tests

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. python3 -m pytest poc/chatbot/tests -v
```

## Tools available to the model

| Name | Purpose | Caps |
|---|---|---|
| `list_docs(subdir)` | List files under a subdir of `DOCS_ROOT`. | 200 files |
| `read_doc(path)` | Read a UTF-8 file. | 32 KB truncated |
| `search_docs(query, max_hits)` | Case-insensitive substring search. | 200 files, 160-char snippet |

All tools resolve paths under `DOCS_ROOT`; anything escaping is rejected
with `{"error": "...outside DOCS_ROOT"}`.

## Configuration

See `.env.example`. The important knobs:

- `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` -- provider switch.
- `DOCS_ROOT` -- the directory the agent can read from.
- `MAX_TOOL_ITERATIONS` -- safety cap on the tool-call loop.
- `SESSION_TTL_SECONDS` -- idle eviction for in-memory sessions.
```

- [ ] **Step 2: Run full test suite**

```bash
cd /Users/mengjia/MiraNote/miranote-api
PYTHONPATH=. python3 -m pytest poc/chatbot/tests -v
```
Expected: all tests passed (16 fs + 7 tools + 7 session + 4 chat_loop = 34).

- [ ] **Step 3: Rule 3 check**

```bash
cd /Users/mengjia/MiraNote/dotgithub
PYTHONPATH=. python3 -m checks.no_cjk_or_emoji /Users/mengjia/MiraNote/miranote-api; echo "exit=$?"
```
Expected: `exit=0`.

- [ ] **Step 4: Manual end-to-end smoke (requires a real `LLM_API_KEY`)**

Start the server in one terminal:
```bash
cd /Users/mengjia/MiraNote/miranote-api/poc/chatbot
source .venv/bin/activate
PYTHONPATH=../.. uvicorn main:app --reload --port 8002
```

In another, run the three canonical questions from the README via curl
(or open the UI). Confirm:
- Tool chips show `list_docs` / `read_doc` / `search_docs` being used.
- The final reply cites a filename when answering a content question.
- Session reuse works (second curl with `session_id` continues the convo).

If the demo fails, debug before committing. Do not move on with a
broken server.

- [ ] **Step 5: Commit README**

```bash
cd /Users/mengjia/MiraNote/miranote-api
git add poc/chatbot/README.md
git commit -m "docs(api): add chatbot POC README with demo questions"
```

---

## Task 14: Rename branch + open PR

- [ ] **Step 1: Rename the branch to a feat/* name**

```bash
cd /Users/mengjia/MiraNote/miranote-api
git branch -m feat/api-chatbot-poc
```

- [ ] **Step 2: Push and open the PR**

```bash
git push -u origin feat/api-chatbot-poc
gh pr create --title "Add chatbot POC with native function calling" --body "$(cat <<'EOF'
## Summary
- New POC at `poc/chatbot/` with multi-turn chat and OpenAI-style native
  function calling.
- Read-only file-system tools (`list_docs`, `read_doc`, `search_docs`)
  sandboxed under `DOCS_ROOT`.
- Vanilla HTML/CSS/JS single-page UI matching the warm/cream palette of
  the sibling POCs.
- Bilingual demo docs under `demo_data/docs/` and three canonical demo
  questions in the README.

## Design
Spec: `docs/superpowers/specs/2026-05-28-chatbot-with-tools-design.md`
(this PR also includes the spec + an inline self-review fix commit).

## Test plan
- [ ] `PYTHONPATH=. python3 -m pytest poc/chatbot/tests -v` -- all green
- [ ] `python3 -m checks.no_cjk_or_emoji <repo>` -- exit 0
- [ ] Manual: start server, run the three demo questions via UI, confirm
      tool chips render and replies cite the right files

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Verify PR URL is returned**

`gh pr create` prints the PR URL on success. Share it with the user.

---

## Self-review

**Spec coverage:**
- §1 Goal -> Tasks 7-8 (chat loop) + Task 11 (HTTP) cover multi-turn + tool calling. ✓
- §2 Non-goals -> Plan explicitly avoids auth, persistent sessions, streaming, vector RAG, write tools. ✓
- §3 File layout -> Task 0 scaffolds, Tasks 1-12 fill in. ✓
- §4 Tools -> Tasks 1-5. Caps match spec (32 KB, 200 files, 160 chars, max_hits=20). ✓
- §5 Chat loop -> Tasks 7-9. Session-seeding (system prompt on first call), tool-call append/dispatch, cap-hit synthetic reply all present. ✓
- §6 HTTP API -> Task 11. All four endpoints + `/` + `/static/*` present. ✓
- §7 UI -> Task 12. Single column, user-right / assistant-left, tool chips, Cmd/Ctrl+Enter send, Reset. ✓
- §8 Env -> Tasks 0 (.env.example) + 11 (`os.getenv` reads). ✓
- §9 Safety -> Path-traversal guard in Task 1, read-only by construction, iteration cap in Task 9, read-size cap in Task 3, history cap in Task 7, no env exposure (Tasks 1-5 only touch DOCS_ROOT). ✓
- §10 Testing -> Tasks 1-9 cover tools_fs + chat_loop unit tests; Task 13 covers manual demo. ✓
- §11 Demo content -> Task 10. ✓
- §12 Open follow-ups -> deferred; out of scope as spec says. ✓

**Placeholder scan:** No TBDs, no "implement later", no "similar to Task N". Every code step contains the actual code. ✓

**Type consistency:**
- `_resolve_path(docs_root: Path, rel_or_abs: str) -> Path` -- referenced consistently by `list_docs`, `read_doc`, `search_docs`. ✓
- `dispatch(docs_root: Path, name: str, args: Dict) -> Any` -- matches `_dispatcher(name, args)` wrapper in `main.py` (binds `DOCS_ROOT`). ✓
- `SessionStore.create(seed=...)` / `.get(sid)` / `.append(sid, msg)` / `.replace(sid, msgs)` / `.delete(sid)` -- consistent in `session.py` impl and in `chat_loop.py` callers. ✓
- `ChatTurnResult(session_id, reply, tool_trace)` -- consistent. ✓
- `run_turn(*, client, session_store, session_id, user_message, model, tools, tool_dispatcher, max_iterations, max_history, system_prompt)` -- callsites in tests and `main.py` use the exact same kwargs. ✓
