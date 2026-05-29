"""MiraNote POC -- Chatbot with native function calling.

Backend-only. The shared web UI lives in poc/text-clean-expand/static/
and is served by that POC; it talks to this server cross-origin (CORS
allow-all). See README for run instructions.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from openai import OpenAI

from poc.chatbot import tools
from poc.chatbot.chat_loop import ChatTurnResult, run_turn
from poc.chatbot.config import ChatbotConfig
from poc.chatbot.session import SessionStore


load_dotenv()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL")

_initial_docs_root = Path(os.getenv("DOCS_ROOT", "./demo_data/docs")).resolve()

if not LLM_API_KEY:
    raise RuntimeError("LLM_API_KEY is required. Set it in .env")
if not _initial_docs_root.exists() or not _initial_docs_root.is_dir():
    raise RuntimeError(f"DOCS_ROOT does not exist or is not a directory: {_initial_docs_root}")

config = ChatbotConfig(
    docs_root=_initial_docs_root,
    model=os.getenv("LLM_MODEL", "deepseek-chat"),
    max_tool_iterations=int(os.getenv("MAX_TOOL_ITERATIONS", "6")),
    max_history_messages=int(os.getenv("MAX_HISTORY_MESSAGES", "40")),
    session_ttl_seconds=int(os.getenv("SESSION_TTL_SECONDS", "3600")),
)

client_kwargs: Dict[str, Any] = {"api_key": LLM_API_KEY}
if LLM_BASE_URL:
    client_kwargs["base_url"] = LLM_BASE_URL
client = OpenAI(**client_kwargs)

SYSTEM_PROMPT = (Path(__file__).parent / "prompts" / "system.txt").read_text(encoding="utf-8")

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


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_trace: List[Dict[str, Any]]


class ConfigRequest(BaseModel):
    docs_root: str = Field(..., min_length=1)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result: ChatTurnResult = await asyncio.to_thread(
            run_turn,
            client=client,
            session_store=sessions,
            session_id=req.session_id,
            user_message=req.message,
            model=config.model,
            tools=tools.TOOLS,
            tool_dispatcher=_dispatcher,
            max_iterations=config.max_tool_iterations,
            max_history=config.max_history_messages,
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
