# Text features expansion + voice acoustic sentiment + chatbot integration

- **Date:** 2026-06-05
- **Author:** mengjia (Claude-assisted)
- **Status:** Draft, awaiting implementation plan
- **Scope:** three POCs touched in one day's work
  - `poc/text-clean-expand/` -- 4 new endpoints + UI rework
  - `poc/voice-to-text/` -- acoustic emotion analysis + CORS fix + UI badge
  - `poc/chatbot/` -- 6 new tools wrapping the text endpoints over HTTP
- **Reference:** action items from `meeting_3_summary.md` (2026-05-30).
  Specifically Meng's "Expand text AI features" and the meeting's call for
  voice-emotion analysis on mobile journal entries.

## 1. Goal

Three things from the May 30 meeting:

1. **Expand text AI surface** -- the existing POC ships only `clean` and
   `expand`. Add the most-asked-for transforms (polish, shorten, keywords,
   caption) so the mobile journal app has a real text toolkit.
2. **Voice acoustic sentiment** -- after Whisper transcribes a voice note,
   classify the speaker's emotion from the audio waveform (not just the
   transcript) so the journal can show "this entry sounded happy" badges.
   Must work for both English and Chinese audio.
3. **Chatbot can do everything text can do** -- expose all six text
   transforms as chatbot tools so the user can ask in chat (e.g.
   "polish this paragraph: ...") and the agent runs the right transform.

## 2. Non-goals (today)

- Image, video, music -- not Meng's lane (Gloria + Jason + Zhao Yan).
- Real-time canvas reading (deferred at the meeting, see meeting `3.3`).
- Translation, quote suggestions, multi-style caption variants -- cut from
  today's "cores only" scope; defer to a follow-up.
- Production deployment, auth, persistent storage.
- "Real" multi-day acoustic emotion model (custom training, dataset
  curation). Today uses an off-the-shelf model with reasonable cross-lingual
  generalisation, accepting the empirical-only Chinese performance claim.
- Mobile UI -- Jason owns. This spec only updates the dev-facing unified
  web UI for demo.
- User research survey (Meng's other action item; non-engineering).

## 3. Feature 1 -- Text endpoints

### 3.1 New endpoints in `text-clean-expand`

Add four endpoints alongside the existing `/clean` and `/expand`. Each
follows the same shape as the existing two: load prompt at startup,
call `call_llm()` helper, return Pydantic response.

| Endpoint | Request | Response |
|---|---|---|
| `POST /polish` | `{text, context?}` | `{original, polished}` |
| `POST /shorten` | `{text, target?: "30%" \| "50%" \| "tweet"}` | `{original, shortened, target}` |
| `POST /keywords` | `{text, max?: 10}` | `{original, keywords: [{term, score}]}` |
| `POST /caption` | `{text, style?: "instagram" \| "diary" \| "tweet"}` | `{original, caption, style}` |

Behaviour notes:
- `polish`: final editing pass -- improve word choice and flow, do NOT
  restructure or add ideas. Differs from `clean` (which restructures
  messy input) and `expand` (which adds substance).
- `shorten`: produce a shorter version preserving meaning. `target`
  controls aggressiveness: `30%` (light), `50%` (cut in half), `tweet`
  (max 280 chars). The prompt file describes what each target means in
  natural language; the server doesn't enforce length itself, the LLM
  does -- response includes the `target` echoed back so the UI knows
  what was asked.
- `keywords`: extract 5-10 keywords as `[{term, score}]`. Score is an
  LLM-assigned salience integer 1-10 (not a calibrated probability);
  useful for relative ranking. The prompt forces a JSON-only response
  in the keyword schema; the endpoint `json.loads` the LLM output and
  returns a Pydantic-validated list. If the LLM emits invalid JSON the
  endpoint returns 502 with the raw output included so we can diagnose.
- `caption`: produce a 1-2 sentence caption. `style` controls register:
  `instagram` (punchy, hook-y), `diary` (warm, personal), `tweet`
  (compressed, hook).

All four obey the existing bilingual rule from `clean.txt` (output
language matches input language; English tech terms stay as-is).

### 3.2 Prompt files

New files under `poc/text-clean-expand/prompts/`:

- `polish.txt`
- `shorten.txt`
- `keywords.txt`
- `caption.txt`

Each follows the existing `clean.txt` / `expand.txt` template:
- "You ARE the output" rule (no meta-commentary).
- Bilingual rule (output language == input language).
- Output format strictness section.

Each prompt is allowlisted by Rule 3 (`**/prompts/*.txt`), so Chinese
few-shot examples are permitted in the prompt files themselves.

### 3.3 UI rework in the Text tab

The Text tab in `poc/text-clean-expand/static/index.html` currently has
two buttons (Clean, Expand). Replace with:

- One `<select>` dropdown labelled "Action": clean / expand / polish /
  shorten / keywords / caption.
- One "Run" button.
- Conditional sub-controls that appear based on action:
  - `shorten`: radio for target (30% / 50% / tweet).
  - `caption`: radio for style (instagram / diary / tweet).
  - `keywords`: number input for max (default 10).
- Result panel adapts:
  - `keywords`: render as chips with score badges.
  - Everything else: render as plain text in the existing result-box.

The existing Input textarea, Context textarea, and result-box layout
stay. Only the action surface changes.

## 4. Feature 2 -- Voice acoustic sentiment

### 4.1 Model choice

`hughlan1214/Speech_Emotion_Recognition_wav2vec2-large-xlsr-53_240304_SER_fine-tuned2.0`

- Backbone: `facebook/wav2vec2-large-xlsr-53` (pretrained on 53 languages
  including Mandarin).
- Fine-tuned on: CREMA + RAVDESS + SAVEE + TESS (12,000+ English clips).
- Output: 7-class label -- `angry`, `disgust`, `fear`, `happy`,
  `neutral`, `sad`, `surprise` + softmax scores.
- Cross-lingual performance: the model author reports it "performs well
  in Chinese and French" via post-release testing. This is an empirical
  claim by the author, not a published benchmark; treat as "best
  practical bet" rather than "guaranteed". If Chinese accuracy is poor
  in practice, the fallback is a hybrid acoustic + LLM-on-transcript
  approach (see Open Follow-ups).
- Model size: ~1.3 GB.

### 4.2 Module structure

New file `poc/voice-to-text/emotion.py`:

```python
def analyze_emotion(audio_path: str) -> Dict[str, Any]:
    """Run the emotion classifier on an audio file.

    Returns:
        {
            "label": "happy",
            "confidence": 0.83,
            "all_scores": [{"label": "happy", "score": 0.83}, ...]
        }
    """
```

- Lazy load: model loads on first `analyze_emotion()` call (not at
  startup). Whisper-only requests are not blocked by the emotion model.
- Loaded model is cached at module level for subsequent calls.
- First call: ~5 sec warm-up. Subsequent: ~1 sec on CPU.

### 4.3 Endpoint integration

- `POST /transcribe` gains an optional `?with_emotion=true` query param.
  Default is `true` so the unified UI gets emotion without changing its
  request shape. Response gains an `emotion` field (shape from 4.2)
  when `with_emotion=true`.
- New `POST /emotion` endpoint accepts just an audio upload and returns
  only the emotion shape. For reuse cases that already have a transcript
  or don't need one.

When `with_emotion=true` but emotion analysis fails (e.g. file format
issue), the `emotion` field is `null` and a sibling `emotion_status`
field reports `"failed"` (mirroring the existing `correction_status`
pattern in voice-to-text).

### 4.4 UI integration

In the Voice tab result area:
- Add a third badge next to the existing `lang:` and `correction:`
  badges: `emotion: happy 83%`.
- Click/hover on the badge expands to show all 7 scores as a small
  vertical list.

### 4.5 CORS fix

`poc/voice-to-text/main.py` currently has no `CORSMiddleware`. The
unified UI in `text-clean-expand` calls `/transcribe` cross-origin, so
this should have been there from the start. Add it (allow-all, matching
the other POCs).

### 4.6 Model storage

Default HuggingFace cache: `~/.cache/huggingface/hub/`. Mirrors the
existing Whisper pattern (`~/.cache/whisper/`). Not committed to git,
not inside the repo, no `.gitignore` entry needed. README documents
the ~1.3 GB first-call download so teammates expect it.

## 5. Feature 3 -- Chatbot text-transformation tools

### 5.1 Six new tools in the chatbot's tool registry

In `poc/chatbot/tools.py`, add:

| Tool name | Backed by | When the agent should call it |
|---|---|---|
| `clean_text` | `POST /clean` | user asks to "clean up", "整理", or pastes messy text wanting structure |
| `expand_text` | `POST /expand` | user asks to "expand", "develop", "扩写" |
| `polish_text` | `POST /polish` | user asks to "polish", "refine", "edit", "润色" |
| `shorten_text` | `POST /shorten` | user asks to "shorten", "trim", "缩短" |
| `extract_keywords` | `POST /keywords` | user asks to extract tags, keywords, key terms |
| `generate_caption` | `POST /caption` | user asks for a caption, summary one-liner, "配文" |

Each tool's JSON-schema description includes both English and Chinese
trigger phrases so the model can match the user's actual language.

### 5.2 HTTP delegation

Chatbot calls `text-clean-expand` over HTTP rather than importing or
duplicating its prompts. Rationale:

- Single source of truth for prompts (lives in text-clean-expand).
- text-clean-expand can evolve (caching, rate-limit, A/B testing) without
  touching chatbot.
- Avoids cross-POC import paths and shared-directory coupling.

Implementation:
- New env var `TEXT_API_URL` in chatbot's `.env.example`, defaults to
  `http://localhost:8001`.
- New dependency `httpx` in `poc/chatbot/requirements.txt`.
- New module `poc/chatbot/text_client.py`: thin wrapper around
  `httpx.AsyncClient` exposing one function per text endpoint. Pure;
  receives `base_url` as argument so tests can stub.
- `tools.dispatch(config, name, args)` routes the six text tool calls
  through `text_client`. Existing fs tools (`list_docs` etc.) unchanged.

### 5.3 Failure mode

If text-clean-expand is unreachable (connection refused, timeout, 5xx),
the tool returns `{"error": "text service unavailable: <message>"}` and
the chat loop's existing error-wrapping bubbles it to the agent, which
reports back to the user naturally. No new failure-handling primitives
required.

### 5.4 System prompt update

Add one paragraph to `poc/chatbot/prompts/system.txt`:

> You also have text-transformation tools (`clean_text`, `expand_text`,
> `polish_text`, `shorten_text`, `extract_keywords`, `generate_caption`)
> for the user's writing. Use them when the user explicitly asks to
> transform a piece of text (English or Chinese -- the tools handle
> both). For open-ended questions about the user's documents, prefer
> the docs tools (`read_doc`, `search_docs`, `list_docs`).

## 6. Architecture / cross-cutting

### 6.1 PR plan

Three PRs:

- **PR A: text endpoints + UI rework** (`poc/text-clean-expand/`).
  Ships independently. Mergeable as soon as Jason approves.
- **PR B: voice acoustic sentiment + CORS** (`poc/voice-to-text/`).
  Independent of A. Parallel review.
- **PR C: chatbot text tools** (`poc/chatbot/`). Depends on A being
  merged (or at least running locally) for end-to-end smoke. Tests can
  ship before A merges because they stub the HTTP client.

If we run short on time, **PR C is the cut**. The demo-critical pieces
are the text endpoints (A) and the voice emotion badge (B). Chatbot
tool integration is quality-of-life and can land tomorrow.

### 6.2 Testing strategy

Each POC currently has no tests except chatbot. Add a minimal `tests/`
per touched POC following the chatbot pattern:

- `poc/text-clean-expand/tests/`:
  - `conftest.py`: pytest fixture providing a FastAPI `TestClient` with
    a stubbed OpenAI client (returns scripted strings).
  - `test_endpoints.py`: one test per new endpoint asserting request
    shape -> response shape and that the prompt file is loaded.
- `poc/voice-to-text/tests/`:
  - `conftest.py`: stub `emotion.analyze_emotion` to return a known
    payload; stub Whisper to return a known transcript.
  - `test_transcribe.py`: assert `/transcribe?with_emotion=true` returns
    the emotion field; assert `with_emotion=false` does not.
  - `test_emotion.py`: standalone `/emotion` endpoint shape.
- `poc/chatbot/tests/`:
  - Extend `test_tools.py`: assert the 6 new tools are registered;
    assert dispatch routes to a stubbed `text_client` and the result
    flows back to the dispatcher.
  - Extend `test_text_client.py` (new): stub `httpx` and verify the
    client builds the right URLs / payloads.

Manual smoke for both A and B: hit the new endpoints via curl with real
LLM/model, then open the unified UI Chat / Text / Voice tabs and
exercise the new buttons.

### 6.3 Conventions

- Rule 3: source ASCII; new prompts go under `prompts/` (allowlisted);
  no new content files outside allowlisted paths.
- Conventional Commits, scope `api`, max 72 char subject.
- Each PR title self-explanatory, no internal indices.
- Python 3.9 compat (existing constraint): `from __future__ import
  annotations` + `typing.Optional/List/Dict`, no PEP-604 `X | None`.

## 7. Open follow-ups (post-today)

- **Chinese acoustic accuracy:** if `hughlan1214/...` performs poorly on
  Chinese journal audio in practice, switch to a hybrid: keep the
  acoustic label as `vocal_emotion` (honest about what it measures) and
  add a text-derived `transcript_sentiment` via LLM. The UI shows both.
- **Translation endpoint:** cut from today; meeting listed it under text
  AI. Own spec.
- **Quote suggestions, multi-style caption:** cut; future text PR.
- **Shared prompts directory:** if drift between text-clean-expand
  prompts and chatbot prompt usage becomes painful, factor to
  `poc/_shared_prompts/`.
- **Production deployment:** Whisper + emotion model are heavy. Plan
  out-of-process model loading or pre-bake into container images.

## 8. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| 1.3 GB model download on first call surprises new teammates | UX friction | README "first emotion request downloads ~1.3 GB to `~/.cache/huggingface/`" |
| Cross-lingual emotion accuracy on Chinese is empirical, not benchmarked | Wrong labels in Chinese audio demos | Show `confidence` so users see uncertainty; document fallback path in Open Follow-ups |
| `httpx` adds inter-POC operational coupling for chatbot | Chatbot tool calls fail if text-clean-expand is down | Graceful error wrapping, `start-all.sh` already brings both up, error message points at the right server |
| PR A + B + C in one day is tight | We slip the demo | Pre-declared cut order: PR C drops first |
| Sentiment model emits a label that doesn't match user expectation (e.g. `disgust` on a neutral journal entry) | Demo lands awkwardly | Show top-3 instead of just top-1 on click/hover, so users see "happy 30%, neutral 28%, sad 18%" -- model is hedging, label feels honest |

