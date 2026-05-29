# ADR 002: DeepSeek as the default LLM provider

- **Date:** 2026-05-20
- **Status:** Accepted
- **Deciders:** mengjia

## Context

POCs need a default LLM. We benchmarked DeepSeek v4-flash, Gemini 2.5
Flash, OpenAI GPT-4o-mini, and Moonshot Kimi-K1.

## Decision

**DeepSeek v4-flash is the default** across all MiraNote POCs. Each
POC's `.env.example` keeps `LLM_BASE_URL=https://api.deepseek.com` and
`LLM_MODEL=deepseek-chat`. Switching to another provider is a one-file
edit since every POC talks to an OpenAI-compatible API.

## Why

1. **Bilingual quality.** DeepSeek punctuation/grammar repair on
   Chinese voice transcripts beats GPT-4o-mini and ties Gemini.
2. **Function-calling support.** Returns OpenAI-compatible `tool_calls`,
   which the chatbot POC depends on. Caveat: v4-flash is a
   thinking-mode model and requires `reasoning_content` to be
   round-tripped (see chatbot loop fix `0c85eef`).
3. **Price.** ~1/10 the cost of GPT-4o for our typical workload.

## Consequences

- Engineers should test against DeepSeek before merging POC changes.
- Provider abstraction stays cheap because every option uses the
  OpenAI SDK; no DeepSeek-specific code in the POCs.

## Related

- ADR-001: bilingual prompts.
