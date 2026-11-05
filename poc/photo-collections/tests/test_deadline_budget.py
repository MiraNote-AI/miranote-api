"""The job cap at real latencies (full check 2026-11-05).

Latencies are simulated at 1/100 scale (1 s here = 100 s real). The fake
honours the timeout the pipeline passes, as llm.complete does. A usable
answer must never be thrown away as a timeout because an optional round
(shorten, cover rewrite) ran into the cap, and call B failing must not
spend a third call B on the cover rewrite.
"""

import json
import re
import time

import pytest

import compose
import config
import llm
from tests.test_compose import IMAGES, _a_json, _b_json, _meta

S = 0.01


class Sim:
    def __init__(self, plan):
        self.plan = list(plan)  # (call, real latency s, reply text or callable)
        self.log = []

    def __call__(self, messages, *, call, timeout):
        assert self.plan and self.plan[0][0] == call, (call, self.plan[:1], self.log)
        _, lat, reply = self.plan.pop(0)
        lat *= S
        self.log.append((call, round(lat / S), round(timeout / S)))
        if timeout <= 0:
            raise llm.ModelTimeout("none left")
        if lat > timeout:
            time.sleep(timeout)
            raise llm.ModelTimeout("sim")
        time.sleep(lat)
        text = reply(messages) if callable(reply) else reply
        return llm.LlmReply(text=text, prompt_tokens=1, completion_tokens=1,
                            latency_ms=int(lat * 1000), cost_cny=0.0)


def _auto_c(cover_texts):
    def answer(messages):
        task = messages[1]["content"][-1]["text"]
        ss = re.findall(r"^(S\d+)｜(.*)$", task, re.M)
        qs = re.findall(r"^(Q\d+)｜(.*)$", task, re.M)
        return json.dumps({
            "sentences": [{"id": i, "grounded": True, "mentions_cover": t in cover_texts,
                           "reason": ""} for i, t in ss],
            "questions": [{"id": i, "grounded": True, "safe": True, "reason": ""} for i, _ in qs],
        }, ensure_ascii=False)
    return answer


NON_PEAK = [{"text": "长椅上，一只橘猫睡着了", "photo_ids": ["P2"]}]
ON_PEAK = [{"text": "台中央，落下一束光", "photo_ids": ["P4"]}]


@pytest.fixture
def scaled(monkeypatch, tmp_log):
    for k in ("CALL_A_TIMEOUT", "CALL_B_TIMEOUT", "CALL_C_TIMEOUT"):
        monkeypatch.setattr(config, k, getattr(config, k) * S)
    monkeypatch.setattr(compose, "CALL_MARGIN_SECONDS", compose.CALL_MARGIN_SECONDS * S)
    return monkeypatch


def _go(monkeypatch, plan):
    sim = Sim(plan)
    monkeypatch.setattr(llm, "complete", sim)
    deadline = time.monotonic() + config.JOB_CAP_SECONDS * S
    out = compose.run(_meta(), dict(IMAGES), deadline)
    return out, sim, deadline


def test_cover_rewrite_running_into_the_cap_keeps_the_ready_answer(scaled, tmp_log):
    # A 150 s, B 45 s, C 45 s: at ~240 s the cat sentence is pushable (the
    # cover moves to P2). The optional cover rewrite's B takes 50 s and its C
    # cannot finish: the ready answer goes out, not a timeout.
    plan = [("a", 150, _a_json()), ("b", 45, _b_json(candidates=NON_PEAK)),
            ("c", 45, _auto_c(set())),
            ("b", 50, _b_json(candidates=ON_PEAK, ranked=[])),
            ("c", 30, _auto_c({"台中央，落下一束光"}))]
    out, sim, deadline = _go(scaled, plan)
    assert out["status"] == "ok", (out, sim.log)
    assert out["memory"]["push"]["body"] == "长椅上，一只橘猫睡着了"
    assert out["memory"]["push"]["coverPhotoId"] == "P2"
    assert time.monotonic() <= deadline


def test_call_b_failing_twice_skips_the_cover_rewrite(scaled, tmp_log):
    # B times out twice: templates and fallback questions go out; no third
    # call B is spent on the cover.
    plan = [("a", 125, _a_json()), ("b", 999, ""), ("b", 999, "")]
    out, sim, _ = _go(scaled, plan)
    assert out["status"] == "ok", (out, sim.log)
    assert [c for c, _, _ in sim.log] == ["a", "b", "b"]
    assert out["memory"]["generation"]["bodySource"] != "llm"
    assert out["memory"]["ending"]["questions"][0]["source"] == "fallback_B1"


def test_cap_reached_in_call_b_gives_templates_not_timeout(scaled, tmp_log):
    # A timed out once and took most of the job; B is cut by the cap: the
    # templates are still a push (like call C failing twice, decision
    # 2026-11-04 item 2).
    plan = [("a", 999, ""), ("a", 100, _a_json()), ("b", 999, "")]
    out, sim, deadline = _go(scaled, plan)
    assert out["status"] == "ok", (out, sim.log)
    assert out["memory"]["generation"]["bodySource"] != "llm"
    assert time.monotonic() <= deadline


def test_cap_reached_in_call_c_gives_templates_not_timeout(scaled, tmp_log):
    plan = [("a", 999, ""), ("a", 60, _a_json()), ("b", 40, _b_json(candidates=ON_PEAK)),
            ("c", 999, "")]
    out, sim, deadline = _go(scaled, plan)
    assert out["status"] == "ok", (out, sim.log)
    assert out["memory"]["generation"]["bodySource"] != "llm"
    assert out["memory"]["ending"]["questions"][0]["source"] == "fallback_B1"
    assert time.monotonic() <= deadline


def test_every_call_ends_before_the_cap(scaled, tmp_log):
    # No call is given time that runs past the job cap (minus the margin
    # the program needs to put the answer together).
    plan = [("a", 999, ""), ("a", 999, "")]
    out, sim, deadline = _go(scaled, plan)
    assert (out["status"], out["reason"]) == ("error", "timeout")
    assert time.monotonic() <= deadline
