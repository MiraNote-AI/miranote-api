"""MiraNote POC -- Chatbot with native function calling.

Backend-only. The shared web UI lives in poc/text-clean-expand/static/
and is served by that POC; it talks to this server cross-origin (CORS
allow-all). See README for run instructions.
"""
from __future__ import annotations

import asyncio
import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

from poc.chatbot import canvas, journal, tools
from poc.chatbot.chat_loop import ActionGuard, ChatTurnResult, drop_unrenderable, run_turn
from poc.chatbot.config import ChatbotConfig
from poc.chatbot.image_client import ImageClient
from poc.chatbot.session import SessionStore
from poc.chatbot.retrieval_client import RetrievalClient
from poc.chatbot.text_client import TextClient


load_dotenv(Path(__file__).parent / ".env")

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")

# A relative DOCS_ROOT is anchored to this file's directory, matching how the
# .env next to it is loaded, so startup works regardless of the process CWD.
_docs_root_env = Path(os.getenv("DOCS_ROOT", "./demo_data/docs"))
_initial_docs_root = (
    _docs_root_env if _docs_root_env.is_absolute() else Path(__file__).parent / _docs_root_env
).resolve()

if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY is required. Set it in .env")
if not _initial_docs_root.exists() or not _initial_docs_root.is_dir():
    raise RuntimeError(f"DOCS_ROOT does not exist or is not a directory: {_initial_docs_root}")

TEXT_API_URL = os.getenv("TEXT_API_URL", "http://localhost:8001")
RETRIEVAL_API_URL = os.getenv("RETRIEVAL_API_URL", "http://localhost:8004")
IMAGE_API_URL = os.getenv("IMAGE_API_URL", "http://localhost:8002")

config = ChatbotConfig(
    docs_root=_initial_docs_root,
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    max_tool_iterations=int(os.getenv("MAX_TOOL_ITERATIONS", "6")),
    max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "40")),
    session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "3600")),
    text_client=TextClient(TEXT_API_URL),
    retrieval_client=RetrievalClient(RETRIEVAL_API_URL),
)

client_kwargs: Dict[str, Any] = {"api_key": LLM_API_KEY}
if LLM_BASE_URL:
    client_kwargs["base_url"] = LLM_BASE_URL
client = OpenAI(**client_kwargs)

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.txt").read_text(encoding="utf-8")
JOURNAL_PROMPT = (Path(__file__).parent / "prompts" / "journal.txt").read_text(encoding="utf-8")
CANVAS_PROMPT = canvas.CANVAS_PROMPT_PATH.read_text(encoding="utf-8")
JOURNAL_TOOLS = journal.journal_tools(tools.TOOLS)
CANVAS_TOOLS = canvas.canvas_tools(tools.TOOLS)

image_client = ImageClient(IMAGE_API_URL)

sessions = SessionStore(ttl_seconds=config.session_ttl_seconds)


def _dispatcher(name: str, args: Dict[str, Any]) -> Any:
    return tools.dispatch(config, name, args)


app = FastAPI(title="MiraNote Chatbot", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NoteIn(BaseModel):
    title: str = ""
    body: str = ""
    date: str = ""


class ElementIn(BaseModel):
    handle: str
    kind: str
    x: float = 0
    y: float = 0
    w: float = 0
    h: float = 0
    says: str = ""
    point_size: Optional[float] = None
    color: Optional[str] = None
    rotation: Optional[float] = None
    treatment: Optional[str] = None


class PageIn(BaseModel):
    width: float
    height: float
    background: str = ""
    # Text color names the app will accept; stated so the model picks
    # from the palette instead of inventing "warm beige".
    palette: List[str] = []
    elements: List[ElementIn] = []
    omitted: int = 0
    # base64 JPEG of the page as the user sees it. Present on every
    # canvas turn; spent only if the model calls look_at_page.
    image: Optional[str] = None


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=4000)
    # Present (even empty) = journal mode: ground the reply in the
    # user's own pages and withhold the docs tools.
    notes: Optional[List[NoteIn]] = None
    # Present = canvas mode: the user is editing this page right now.
    # Wins over notes -- the page in front of them is the context.
    page: Optional[PageIn] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_trace: List[Dict[str, Any]]


class ConfigRequest(BaseModel):
    docs_root: str = Field(..., min_length=1)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    canvas_mode = req.page is not None
    journal_mode = req.notes is not None
    message = req.message
    transient_prefix = None
    dispatcher = _dispatcher
    turn_tools = tools.TOOLS
    prompt = SYSTEM_PROMPT
    action_guard = None

    if canvas_mode:
        page = req.page.model_dump()
        image_b64 = page.pop("image", None)
        try:
            image_bytes = base64.b64decode(image_b64) if image_b64 else None
        except Exception:  # noqa: BLE001 -- a bad image only costs the look
            image_bytes = None
        # The map rides along on every call but never enters the
        # session: stale coordinates and renumbered handles would have
        # the model confidently edit the wrong element.
        transient_prefix = canvas.render_page(page)
        dispatcher = canvas.build_dispatcher(image_bytes, image_client, _dispatcher)
        turn_tools = CANVAS_TOOLS
        prompt = CANVAS_PROMPT
        # A look with no follow-through is how a weak model "tidies" a
        # page into a chat bubble; the guard gives it one corrective
        # turn with the tools still on the table.
        action_guard = ActionGuard(
            look_tool="look_at_page",
            action_tools=sorted(canvas.CANVAS_TOOL_NAMES - {"look_at_page"}),
            nudge=canvas.ACTION_NUDGE,
        )
    elif journal_mode:
        message = journal.compose_user_message(
            req.message, [n.model_dump() for n in req.notes]
        )
        turn_tools = JOURNAL_TOOLS
        prompt = JOURNAL_PROMPT

    try:
        result: ChatTurnResult = await asyncio.to_thread(
            run_turn,
            client=client,
            session_store=sessions,
            session_id=req.session_id,
            user_message=message,
            # journal mode composes notes into the message; language must
            # follow what the USER typed, not the notes block
            language_ref=req.message,
            model=config.model,
            tools=turn_tools,
            tool_dispatcher=dispatcher,
            max_iterations=config.max_tool_iterations,
            max_history=config.max_history_messages,
            system_prompt=prompt,
            transient_prefix=transient_prefix,
            action_guard=action_guard,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail="unknown session_id")
    except Exception as e:  # noqa: BLE001 -- surface LLM/network errors
        raise HTTPException(status_code=502, detail=f"chat failed: {e}")
    if not result.reply.strip():
        # Never hand the app a blank bubble; let it show its retry card.
        raise HTTPException(status_code=502, detail="the model returned an empty reply")
    return ChatResponse(
        session_id=result.session_id,
        reply=drop_unrenderable(result.reply),
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
        "model": config.model,
        "tools": [t["function"]["name"] for t in tools.TOOLS],
        "docs_root": str(config.docs_root),
    }


@app.post("/config")
async def update_config(req: ConfigRequest):
    """Mutate runtime config. Currently only docs_root is mutable."""
    try:
        return config.set_docs_root(req.docs_root)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
