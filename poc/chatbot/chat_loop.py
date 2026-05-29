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


def _trim_history(history: List[Dict[str, Any]], max_history: int) -> None:
    """Drop oldest non-system messages in pairs until we're under the cap."""
    non_system = [i for i, m in enumerate(history) if m.get("role") != "system"]
    while len(non_system) > max_history:
        drop_idx = non_system[0]
        del history[drop_idx]
        non_system = [i for i, m in enumerate(history) if m.get("role") != "system"]
