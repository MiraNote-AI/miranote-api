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
