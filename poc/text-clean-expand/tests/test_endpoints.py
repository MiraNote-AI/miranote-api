from __future__ import annotations

import pytest

# Chinese sample text, built with chr() to keep this file ASCII-only (rule 3).
ZH_NOTE = chr(0x4ECA) + chr(0x5929) + chr(0x5F88) + chr(0x7D2F)  # "today very tired"
ZH_REPLY = chr(0x4ECA) + chr(0x5929) + chr(0x771F) + chr(0x7684) + chr(0x5F88) + chr(0x7D2F)
# "kai hui ji de" in simplified (kai/hui/ji are variant chars) and traditional.
ZH_SIMP = chr(0x5F00) + chr(0x4F1A) + chr(0x8BB0) + chr(0x5F97)
ZH_TRAD = chr(0x958B) + chr(0x6703) + chr(0x8A18) + chr(0x5F97)


def test_polish_returns_polished_text(client):
    test_client, fake_llm = client
    fake_llm.reply_with("The morning light was warm and the coffee was strong.")

    r = test_client.post("/polish", json={"text": "morning light warm. coffee strong."})

    assert r.status_code == 200
    body = r.json()
    assert body["original"] == "morning light warm. coffee strong."
    assert body["polished"] == "The morning light was warm and the coffee was strong."


def test_polish_rejects_empty_text(client):
    test_client, _ = client
    r = test_client.post("/polish", json={"text": ""})
    assert r.status_code == 422


def test_shorten_returns_short_version(client):
    test_client, fake_llm = client
    fake_llm.reply_with("Coffee strong, morning bright.")

    r = test_client.post(
        "/shorten",
        json={"text": "The morning light was warm and the coffee was strong.", "target": "50%"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["original"].startswith("The morning light")
    assert body["shortened"] == "Coffee strong, morning bright."
    assert body["target"] == "50%"


def test_shorten_default_target_is_50pct(client):
    test_client, fake_llm = client
    fake_llm.reply_with("short.")
    r = test_client.post("/shorten", json={"text": "longish text here"})
    assert r.status_code == 200
    assert r.json()["target"] == "50%"


def test_shorten_rejects_invalid_target(client):
    test_client, _ = client
    r = test_client.post("/shorten", json={"text": "hi", "target": "bogus"})
    assert r.status_code == 422


def test_keywords_returns_parsed_array(client):
    test_client, fake_llm = client
    fake_llm.reply_with('[{"term": "coffee", "score": 9}, {"term": "morning", "score": 7}]')

    r = test_client.post("/keywords", json={"text": "morning coffee", "max": 5})

    assert r.status_code == 200
    body = r.json()
    assert body["original"] == "morning coffee"
    assert body["keywords"] == [
        {"term": "coffee", "score": 9},
        {"term": "morning", "score": 7},
    ]


def test_keywords_truncates_to_max(client):
    test_client, fake_llm = client
    fake_llm.reply_with('[{"term":"a","score":9},{"term":"b","score":8},{"term":"c","score":7}]')
    r = test_client.post("/keywords", json={"text": "x", "max": 2})
    assert r.status_code == 200
    assert len(r.json()["keywords"]) == 2


def test_keywords_invalid_json_returns_502(client):
    test_client, fake_llm = client
    fake_llm.reply_with("not even close to JSON")
    r = test_client.post("/keywords", json={"text": "x"})
    assert r.status_code == 502
    assert "invalid JSON" in r.json()["detail"]


def test_keywords_unexpected_schema_returns_502(client):
    test_client, fake_llm = client
    fake_llm.reply_with('[{"wrong_key": "oops"}]')
    r = test_client.post("/keywords", json={"text": "x"})
    assert r.status_code == 502


def test_caption_returns_caption_with_style(client):
    test_client, fake_llm = client
    fake_llm.reply_with("Morning ritual: strong coffee, warmer light. Today is mine.")

    r = test_client.post(
        "/caption",
        json={"text": "Had a really nice morning with great coffee.", "style": "instagram"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["original"].startswith("Had a really")
    assert "coffee" in body["caption"].lower()
    assert body["style"] == "instagram"


def test_caption_default_style_is_instagram(client):
    test_client, fake_llm = client
    fake_llm.reply_with("punchy caption.")
    r = test_client.post("/caption", json={"text": "any text"})
    assert r.json()["style"] == "instagram"


def test_caption_rejects_invalid_style(client):
    test_client, _ = client
    r = test_client.post("/caption", json={"text": "x", "style": "haiku"})
    assert r.status_code == 422


def test_english_input_chinese_reply_triggers_language_retry(client):
    test_client, fake_llm = client
    fake_llm.reply_with(ZH_REPLY, "Today was really tiring.")

    r = test_client.post("/polish", json={"text": "today very tired"})

    assert r.status_code == 200
    assert r.json()["polished"] == "Today was really tiring."
    calls = fake_llm.chat.completions.calls
    assert len(calls) == 2
    retry_messages = calls[1]["messages"]
    assert retry_messages[2] == {"role": "assistant", "content": ZH_REPLY}
    assert "wrong language" in retry_messages[3]["content"]
    assert "entirely in English" in retry_messages[3]["content"]


def test_chinese_input_english_reply_triggers_language_retry(client):
    test_client, fake_llm = client
    fake_llm.reply_with("Today was really tiring.", ZH_REPLY)

    r = test_client.post("/polish", json={"text": ZH_NOTE})

    assert r.status_code == 200
    assert r.json()["polished"] == ZH_REPLY
    calls = fake_llm.chat.completions.calls
    assert len(calls) == 2
    assert "rewrite your entire reply in Chinese" in calls[1]["messages"][3]["content"]


def test_matching_language_does_not_retry(client):
    test_client, fake_llm = client
    fake_llm.reply_with(ZH_REPLY)

    r = test_client.post("/polish", json={"text": ZH_NOTE})

    assert r.status_code == 200
    assert len(fake_llm.chat.completions.calls) == 1


def test_language_retry_happens_at_most_once(client):
    test_client, fake_llm = client
    fake_llm.reply_with(ZH_REPLY, ZH_REPLY)  # drifts twice; reply returned as-is

    r = test_client.post("/polish", json={"text": "today very tired"})

    assert r.status_code == 200
    assert r.json()["polished"] == ZH_REPLY
    assert len(fake_llm.chat.completions.calls) == 2


def test_simplified_input_traditional_reply_triggers_script_retry(client):
    test_client, fake_llm = client
    fake_llm.reply_with(ZH_TRAD, ZH_SIMP)

    r = test_client.post("/polish", json={"text": ZH_SIMP})

    assert r.status_code == 200
    assert r.json()["polished"] == ZH_SIMP
    calls = fake_llm.chat.completions.calls
    assert len(calls) == 2
    assert "simplified Chinese characters" in calls[1]["messages"][3]["content"]


def test_matching_script_does_not_retry(client):
    test_client, fake_llm = client
    fake_llm.reply_with(ZH_SIMP)

    r = test_client.post("/polish", json={"text": ZH_SIMP})

    assert r.status_code == 200
    assert len(fake_llm.chat.completions.calls) == 1


def test_expand_over_length_reply_triggers_retry(client):
    test_client, fake_llm = client
    text = "short diary note"
    long_reply = "novelized " * 60  # 600 chars, over 2x + 150
    fake_llm.reply_with(long_reply, "A short, grounded expansion.", "[]")

    r = test_client.post("/expand", json={"text": text})

    assert r.status_code == 200
    assert r.json()["expanded"] == "A short, grounded expansion."
    calls = fake_llm.chat.completions.calls
    assert len(calls) == 3  # first try, length retry, grounding check
    note = calls[1]["messages"][3]["content"]
    assert "too long" in note
    assert str(2 * len(text) + 150) in note


def test_expand_within_length_does_not_retry(client):
    test_client, fake_llm = client
    fake_llm.reply_with("A short, grounded expansion.", "[]")

    r = test_client.post("/expand", json={"text": "short diary note"})

    assert r.status_code == 200
    # first try plus the always-on grounding check; no correction retries
    assert len(fake_llm.chat.completions.calls) == 2


def test_expand_language_and_length_violations_share_one_retry(client):
    test_client, fake_llm = client
    long_chinese = ZH_REPLY * 40  # wrong language AND over length
    fake_llm.reply_with(long_chinese, "A short, grounded expansion.", "[]")

    r = test_client.post("/expand", json={"text": "short diary note"})

    assert r.status_code == 200
    assert r.json()["expanded"] == "A short, grounded expansion."
    calls = fake_llm.chat.completions.calls
    assert len(calls) == 3
    note = calls[1]["messages"][3]["content"]
    assert "wrong language" in note
    assert "too long" in note


def test_expand_invented_details_trigger_grounded_rewrite(client):
    test_client, fake_llm = client
    fake_llm.reply_with(
        "Called mom; dad has been busy fixing the roof.",
        '["dad has been busy fixing the roof"]',
        "Called mom; it was good to hear her voice.",
    )

    r = test_client.post("/expand", json={"text": "called mom today"})

    assert r.status_code == 200
    assert r.json()["expanded"] == "Called mom; it was good to hear her voice."
    calls = fake_llm.chat.completions.calls
    assert len(calls) == 3  # first try, grounding check, grounded rewrite
    checker_system = calls[1]["messages"][0]["content"]
    assert "fact checker" in checker_system
    rewrite_note = calls[2]["messages"][3]["content"]
    assert "invented" in rewrite_note
    assert "dad has been busy fixing the roof" in rewrite_note


def test_expand_grounding_checker_garbage_fails_open(client):
    test_client, fake_llm = client
    fake_llm.reply_with("A grounded expansion.", "not json at all")

    r = test_client.post("/expand", json={"text": "short diary note"})

    assert r.status_code == 200
    assert r.json()["expanded"] == "A grounded expansion."
    assert len(fake_llm.chat.completions.calls) == 2


def test_expand_language_follows_text_not_context(client):
    test_client, fake_llm = client
    fake_llm.reply_with("A fine English expansion.", "[]")

    r = test_client.post(
        "/expand",
        json={"text": "meeting notes about stickers", "context": ZH_NOTE},
    )

    assert r.status_code == 200
    assert r.json()["expanded"] == "A fine English expansion."
    # Composite prompt contains Chinese context, but the reply matches the
    # note's language, so no correction retry (just the grounding check).
    assert len(fake_llm.chat.completions.calls) == 2


@pytest.mark.parametrize("mood", ["joyful", "calm", "tender", "low", "tense", "neutral"])
def test_mood_passes_each_value_through(client, mood):
    test_client, fake_llm = client
    fake_llm.reply_with(mood)

    r = test_client.post("/mood", json={"text": "a page about my day"})

    assert r.status_code == 200
    assert r.json() == {"mood": mood}
    call = fake_llm.chat.completions.calls[0]
    assert call["temperature"] <= 0.2
    assert call["max_tokens"] <= 16


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("  Calm.\n", "calm"),
        ("JOYFUL", "joyful"),
        ('"tender"', "tender"),
        ('{"mood": " Low "}', "low"),
    ],
)
def test_mood_normalises_case_whitespace_and_json(client, reply, expected):
    test_client, fake_llm = client
    fake_llm.reply_with(reply)

    r = test_client.post("/mood", json={"text": "a page"})

    assert r.status_code == 200
    assert r.json() == {"mood": expected}


@pytest.mark.parametrize("reply", ["happy", "The mood is calm", "", '{"x": 1}', "[]"])
def test_mood_off_set_reply_maps_to_neutral(client, reply):
    test_client, fake_llm = client
    fake_llm.reply_with(reply)

    r = test_client.post("/mood", json={"text": "a page"})

    assert r.status_code == 200
    assert r.json() == {"mood": "neutral"}


def test_mood_chinese_page_does_not_trigger_language_retry(client):
    test_client, fake_llm = client
    fake_llm.reply_with("low")

    r = test_client.post("/mood", json={"text": ZH_NOTE})

    assert r.status_code == 200
    assert r.json() == {"mood": "low"}
    assert len(fake_llm.chat.completions.calls) == 1


def test_mood_chinese_reply_gets_one_english_retry_then_neutral(client):
    test_client, fake_llm = client
    fake_llm.reply_with(ZH_NOTE, ZH_NOTE)

    r = test_client.post("/mood", json={"text": ZH_NOTE})

    assert r.status_code == 200
    assert r.json() == {"mood": "neutral"}
    calls = fake_llm.chat.completions.calls
    assert len(calls) == 2
    assert "entirely in English" in calls[1]["messages"][3]["content"]


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_mood_rejects_empty_or_whitespace_text(client, text):
    test_client, fake_llm = client
    r = test_client.post("/mood", json={"text": text})
    assert r.status_code == 422
    assert fake_llm.chat.completions.calls == []


def test_mood_rejects_text_over_the_cap(client):
    test_client, fake_llm = client
    fake_llm.reply_with("calm")
    assert test_client.post("/mood", json={"text": "a" * 4000}).status_code == 200
    r = test_client.post("/mood", json={"text": "a" * 4001})
    assert r.status_code == 422
    assert len(fake_llm.chat.completions.calls) == 1
