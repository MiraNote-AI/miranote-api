# Text Clean & Expand POC

AI-powered text processing: turn messy input into polished notes.

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /clean` | Fix typos, punctuation, grammar + restructure into readable text. Light expansion to fill gaps. |
| `POST /expand` | Fully expand fragments into complete paragraphs, like drafting an email from bullet points. |
| `GET /health` | Health check |
| `GET /` | Web UI (unified POC frontend with Text / Voice / Image tabs) |

## Setup

```bash
cd poc/text-clean-expand
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your API key
uvicorn main:app --host 0.0.0.0 --port 8001
```

Open http://localhost:8001 for the web UI.

## LLM Provider

Uses OpenAI-compatible API. Switch provider by editing `.env`:

- **DeepSeek**: `LLM_BASE_URL=https://api.deepseek.com`, `LLM_MODEL=deepseek-chat`
- **Gemini**: `LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`, `LLM_MODEL=gemini-2.5-flash`
- **OpenAI**: leave `LLM_BASE_URL` empty, `LLM_MODEL=gpt-4o`

## API Usage

```bash
# Clean
curl -X POST http://localhost:8001/clean \
  -H "Content-Type: application/json" \
  -d '{"text": "今天开会讨论了三个事情 第一是产品路线图 第二是美化功能怎么做"}'

# Expand
curl -X POST http://localhost:8001/expand \
  -H "Content-Type: application/json" \
  -d '{"text": "明天要做的事 早上先把poc跑通 下午对一下调研结果"}'

# Expand with context
curl -X POST http://localhost:8001/expand \
  -H "Content-Type: application/json" \
  -d '{"text": "voice memo做好了 下一步搞text", "context": "MiraNote是AI日记应用"}'
```
