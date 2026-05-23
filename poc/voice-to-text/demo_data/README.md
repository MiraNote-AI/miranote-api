# Voice-to-Text demo data

Small set of TTS-generated audio clips for trying the POC without
needing your own recordings. All clips are generated with macOS `say`
(public-domain output) and converted to mono 16 kHz AAC m4a for a
small repo footprint.

| File              | Lang | Duration | What it exercises |
|-------------------|------|----------|-------------------|
| `zh_meeting.m4a`  | zh   | ~10 s    | Pure Mandarin, meeting-style monologue with no source punctuation. Good for showing the LLM correction step (adds commas/periods, fixes the `is/shi` homophone). |
| `zh_en_mixed.m4a` | zh   | ~10 s    | Chinese with inline English tech terms (`product roadmap`, `Q3`, `demo`, `share`). Confirms the multilingual Whisper model preserves code-switching when called with `lang=zh`. |
| `en_short.m4a`    | en   | ~5 s     | Pure English smoke test. With `lang=en` the result should be near-identical raw vs. corrected (the LLM prompt is Chinese-tuned and barely changes English). |

## Use them from the web UI

Start the server (see the parent README), open
<http://localhost:8000/>, pick the matching language from the
**Language** row at the top, switch to the **Upload file** tab, and
select one of the files in this folder.

## Use them from curl

```bash
# Chinese meeting (default lang=zh)
curl -s -F file=@demo_data/zh_meeting.m4a \
     "http://localhost:8000/transcribe" | python3 -m json.tool

# Mixed Chinese + English (lang=zh handles inline English)
curl -s -F file=@demo_data/zh_en_mixed.m4a \
     "http://localhost:8000/transcribe?lang=zh" | python3 -m json.tool

# Pure English
curl -s -F file=@demo_data/en_short.m4a \
     "http://localhost:8000/transcribe?lang=en" | python3 -m json.tool
```

## Regenerating

These files were produced with:

```bash
say -v Meijia -o _zh_meeting.aiff   "<the Chinese paragraph>"
say -v Meijia -o _zh_en_mixed.aiff  "<the mixed paragraph>"
say -v Alex   -o _en_short.aiff     "<the English paragraph>"
for src in _*.aiff; do
  out="${src#_}"; out="${out%.aiff}.m4a"
  ffmpeg -y -i "$src" -ac 1 -ar 16000 -c:a aac -b:a 64k "$out"
done
rm _*.aiff
```

The source phrases are intentionally not committed in this README to
keep the repo Rule-3 clean (no CJK in source). To re-record, ask the
team or pick fresh phrases that exercise the same scenarios.
