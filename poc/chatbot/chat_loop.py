"""Server-side chat loop. Pure -- all deps injected for testability."""
from __future__ import annotations

import json
import os
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from poc.chatbot.session import SessionStore


@dataclass
class ChatTurnResult:
    session_id: str
    reply: str
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)


def drop_unrenderable(text: str) -> str:
    """Strip characters no client font can draw -- the model occasionally
    emits private-use or unassigned codepoints (bad tokens), which iOS
    renders as the LastResort box-with-question-mark glyph. Real emoji and
    CJK pass through untouched (unless MIRA_STRIP_EMOJI=1: demo-video
    recording aid for simulator runtimes whose emoji font is broken);
    newlines and tabs survive."""
    strip_emoji = os.getenv("MIRA_STRIP_EMOJI") == "1"

    def is_emoji(o: int) -> bool:
        return (
            0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF
            or 0x2B00 <= o <= 0x2BFF or o in (0xFE0F, 0x200D, 0x20E3)
        )

    def keep(ch: str) -> bool:
        if ch == chr(0xFFFD):  # REPLACEMENT CHARACTER
            return False
        if strip_emoji and is_emoji(ord(ch)):
            return False
        category = unicodedata.category(ch)
        if category in ("Co", "Cn", "Cs"):
            return False
        if category == "Cc" and ch not in "\n\t":
            return False
        return True

    return "".join(ch for ch in text if keep(ch)).strip()


def _contains_cjk(text: str) -> bool:
    # CJK Unified Ideographs, U+4E00..U+9FFF (chr() keeps this file ASCII-only)
    return any(chr(0x4E00) <= ch <= chr(0x9FFF) for ch in text)


def _language_correction(ref: str, reply: str) -> Optional[str]:
    """Corrective instruction when the reply's language does not match the
    user's message (the model ignores the prompt's language rule; same
    class of bug fixed in the text POC's call_llm)."""
    if _contains_cjk(ref) == _contains_cjk(reply):
        return None
    if _contains_cjk(ref):
        return (
            "Your reply is in the wrong language. The user wrote in Chinese, "
            "so rewrite your entire reply in Chinese. Keep English technical "
            "terms and cited page titles as-is."
        )
    return (
        "Your reply is in the wrong language. The user wrote in English, so "
        "rewrite your entire reply in English only, with no Chinese "
        "characters. Keep cited page titles exactly as they appear."
    )


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
    language_ref: Optional[str] = None,
) -> ChatTurnResult:
    if session_id is None:
        session_id = session_store.create(seed=[{"role": "system", "content": system_prompt}])
    history = session_store.get(session_id)
    history.append({"role": "user", "content": user_message})

    trace: List[Dict[str, Any]] = []
    reply: Optional[str] = None
    empty_retries = 0
    for _ in range(max_iterations):
        kwargs = {"model": model, "messages": history}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        resp = client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        if not getattr(msg, "tool_calls", None):
            reply = msg.content or ""
            if not reply.strip() and empty_retries < 2:
                # Thinking-mode models occasionally return empty content
                # (the answer stays in reasoning); ask again, appending
                # nothing so the transcript stays clean.
                empty_retries += 1
                reply = None
                continue
            note = _language_correction(language_ref or user_message, reply)
            if note:
                # One corrective retry, no tools: wrong-language text is
                # rewritten in place; if it drifts again we keep it.
                fix = client.chat.completions.create(model=model, messages=history + [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": note},
                ])
                fixed = fix.choices[0].message.content or ""
                if fixed.strip():
                    reply = fixed
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
        # Thinking-mode models (DeepSeek v4-flash, Reasoning) require the
        # reasoning_content field to be round-tripped with the tool result.
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning is not None:
            assistant_entry["reasoning_content"] = reasoning
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
