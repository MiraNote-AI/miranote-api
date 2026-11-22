"""Server-side chat loop. Pure -- all deps injected for testability."""
from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from poc.chatbot.session import SessionStore


@dataclass
class ChatTurnResult:
    session_id: str
    reply: str
    tool_trace: List[Dict[str, Any]] = field(default_factory=list)


class ToolNotOffered(Exception):
    """A tool dispatcher raises this for a name the turn did not offer the
    model. run_turn answers the call with {"error": "unknown tool: <name>"},
    and the tool never runs. The call stays in the trace even when the
    action guard drops its nudge exchange: a response never hides a call
    the model made."""

    def __init__(self, name: str):
        super().__init__("unknown tool: " + name)


@dataclass
class ActionGuard:
    """One corrective turn when the model ran `look_tool` and then
    answered in prose without calling any of `action_tools` -- the weak
    model's way of DESCRIBING a change instead of making it. The nudge
    rides as a user-role message so the loop can hand the model its
    tools again. The nudge asks for the `keep` word when no change was
    asked for. If the model answers with it (see _says_keep) and makes no
    change, the first reply was the answer (a question, or words the user
    asked for): it is returned and the nudge exchange -- messages and tool
    calls, except calls refused as not offered (ToolNotOffered) -- is
    dropped. Any other answer stands, as the model's reply to
    "make the change now"; the keep word itself is never shown."""

    look_tool: str
    action_tools: List[str]
    nudge: str
    keep: str


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


# Where a reply's clauses break: sentence and clause punctuation, dashes,
# markdown marks and line breaks, with the CJK full-width forms (escaped:
# this file stays ASCII-only).
_CLAUSE_BREAKS = re.compile(
    r"[.!?;:,*_`\n\u2013\u2014\u3001\u3002\uff01\uff0c\uff1a\uff1b\uff1f-]+"
)


def _says_keep(reply: str, keep: str) -> bool:
    """Whether a reply to the guard's nudge is its keep word rather than an
    answer: the word alone in any case ("KEEP", "Keep.", "**keep**"), or
    in capitals as a clause of its own ("KEEP -- nothing to change",
    "Moved it. KEEP"). Inside a phrase or in quotes it is page text the
    user can see ("KEEP GOING is 36pt now"), and that reply is shown."""
    if re.sub(r"[\W_]+", "", reply).lower() == keep.lower():
        return True
    return any(part.strip() == keep for part in _CLAUSE_BREAKS.split(reply))


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


def _messages_for_model(
    history: List[Dict[str, Any]],
    transient_prefix: Optional[str],
    user_index: int,
) -> List[Dict[str, Any]]:
    """History as the model should see it this turn.

    The prefix (the page map) is context the model needs on every call
    but must never persist: it goes stale the moment the page changes,
    and its element handles are renumbered each turn, so an old t1 may
    name a different element than the current one.
    """
    if not transient_prefix:
        return history
    view = list(history)
    original = view[user_index].get("content", "")
    view[user_index] = {"role": "user", "content": transient_prefix + "\n\n" + original}
    return view


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
    transient_prefix: Optional[str] = None,
    language_ref: Optional[str] = None,
    action_guard: Optional[ActionGuard] = None,
) -> ChatTurnResult:
    if session_id is None:
        session_id = session_store.create(seed=[{"role": "system", "content": system_prompt}])
    history = session_store.get(session_id)
    user_index = len(history)
    history.append({"role": "user", "content": user_message})

    trace: List[Dict[str, Any]] = []
    reply: Optional[str] = None
    empty_retries = 0
    # Where the nudge exchange starts in history and trace, and the reply
    # it would replace; set once the guard has nudged.
    nudged_at: Optional[int] = None
    trace_at = 0
    first_reply = ""
    # Trace positions of calls refused as not offered; see ToolNotOffered.
    refused: Set[int] = set()

    def acted() -> bool:
        return action_guard is not None and bool(
            set(action_guard.action_tools) & {t["name"] for t in trace}
        )

    def keep_first_reply() -> str:
        del history[nudged_at:]
        trace[trace_at:] = [t for i, t in enumerate(trace[trace_at:], trace_at) if i in refused]
        return first_reply

    for _ in range(max_iterations):
        kwargs = {
            "model": model,
            "messages": _messages_for_model(history, transient_prefix, user_index),
        }
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
            if nudged_at is not None and not acted() and (
                not reply.strip() or _says_keep(reply, action_guard.keep)
            ):
                # The model confirmed no change was asked for (or said
                # nothing), so the first reply was the answer. Any other
                # reply without a change -- "I can't change fonts", a
                # question back -- is the model's answer to "make the
                # change now", and stands: the first reply may claim a
                # change that never happened.
                reply = keep_first_reply()
            elif nudged_at is not None and _says_keep(reply, action_guard.keep):
                # It changed the page and still said KEEP: the change
                # stands, and the first reply describes it.
                reply = first_reply
            elif (
                action_guard is not None
                and nudged_at is None
                and reply.strip()
                and action_guard.look_tool in {t["name"] for t in trace}
                and not acted()
            ):
                # The model LOOKED and then answered in prose without
                # changing the page. One corrective turn, with the tools
                # still on the table, in case it described a change
                # instead of making it.
                nudged_at = len(history)
                trace_at = len(trace)
                first_reply = reply
                history.append({"role": "assistant", "content": reply})
                history.append({"role": "user", "content": action_guard.nudge})
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
            except ToolNotOffered as e:
                refused.add(len(trace))
                result = {"error": str(e)}
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

    if reply is None and nudged_at is not None and not acted():
        # Ran out of iterations after the nudge without a change: the
        # first reply still stands.
        reply = keep_first_reply()
        history.append({"role": "assistant", "content": reply})
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
