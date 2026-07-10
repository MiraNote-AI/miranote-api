"""Tool registry + dispatcher for the chatbot POC."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from poc.chatbot import tools_fs
from poc.chatbot.config import ChatbotConfig


# Load tool descriptions from prompts file to support CJK trigger phrases.
_TOOL_DESC_PATH = Path(__file__).parent / "prompts" / "tool_descriptions.txt"
_TOOL_DESCRIPTIONS = {}
for _line in _TOOL_DESC_PATH.read_text(encoding="utf-8").splitlines():
    _line = _line.strip()
    if not _line or _line.startswith("#"):
        continue
    _name, _, _desc = _line.partition("|")
    _TOOL_DESCRIPTIONS[_name.strip()] = _desc.strip()


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
            "description": (
                "Read the contents of a file under the docs root. Supports plain "
                "text / markdown, PDF (.pdf), Word (.docx), and images "
                "(.png/.jpg/.jpeg/.gif/.webp/.bmp/.tiff) via OCR. The response "
                "includes a content_type field telling you which extractor was "
                "used. Output is truncated at 32 KB."
            ),
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
    {
        "type": "function",
        "function": {
            "name": "set_docs_root",
            "description": (
                "Switch the docs root directory the agent reads from. ONLY call "
                "this when the user explicitly asks to switch to a different "
                "folder. Do not call it speculatively. The new path must be an "
                "existing local directory. On success, confirm the new path back "
                "to the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute local path, or '~'-prefixed home-relative path.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clean_text",
            "description": _TOOL_DESCRIPTIONS["clean_text"],
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
            "description": _TOOL_DESCRIPTIONS["expand_text"],
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
            "description": _TOOL_DESCRIPTIONS["polish_text"],
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
            "description": _TOOL_DESCRIPTIONS["shorten_text"],
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
            "description": _TOOL_DESCRIPTIONS["extract_keywords"],
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
            "description": _TOOL_DESCRIPTIONS["generate_caption"],
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
]

# Tools that delegate to the text-clean-expand service via config.text_client.
_TEXT_TOOL_NAMES = frozenset({
    "clean_text", "expand_text", "polish_text",
    "shorten_text", "extract_keywords", "generate_caption",
})


def dispatch(config: ChatbotConfig, name: str, args: Dict[str, Any]) -> Any:
    """Route a model-issued tool call to the underlying implementation.

    `config.docs_root` is read fresh on each call so set_docs_root takes
    effect immediately for subsequent fs-tool calls in the same turn.

    Always returns a JSON-serialisable value. Exceptions become
    {"error": "..."} so the model can recover on the next turn.
    """
    try:
        if name == "list_docs":
            return tools_fs.list_docs(config.docs_root, args.get("subdir", "."))
        if name == "read_doc":
            return tools_fs.read_doc(config.docs_root, args["path"])
        if name == "search_docs":
            return tools_fs.search_docs(config.docs_root, args["query"], int(args.get("max_hits", 20)))
        if name == "set_docs_root":
            return config.set_docs_root(args["path"])
        # Text tools delegate to text-clean-expand; if TEXT_API_URL was not
        # configured, text_client is None. Guard here so the model gets a
        # clear message instead of a cryptic AttributeError on None.
        if name in _TEXT_TOOL_NAMES and config.text_client is None:
            return {"error": "text tools unavailable: TEXT_API_URL not configured"}
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
        if name == "create_note":
            # Journal mode: the page lives on the user's device. The app
            # reads title/body from tool_trace; this result just tells
            # the model the hand-off happened.
            return {"status": "draft handed to the app for the user to shape"}
        if name == "find_quote":
            lang = args.get("lang")
            return config.retrieval_client.quotes(
                args["text"],
                max_picks=int(args.get("max", 3)),
                lang=lang,
            )
        return {"error": f"unknown tool: {name}"}
    except KeyError as e:
        return {"error": f"missing required argument: {e.args[0]}"}
    except Exception as e:  # noqa: BLE001  -- intentionally broad; model needs the message
        return {"error": str(e)}
