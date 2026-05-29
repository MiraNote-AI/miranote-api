# ADR 001: Bilingual product, English-only system prompts

- **Date:** 2026-05-12
- **Status:** Accepted
- **Deciders:** mengjia, Jason

## Context

MiraNote ships to a bilingual user base (Chinese + English). Every LLM
call needs a system prompt. We had to decide whether system prompts
should be English-only, Chinese-only, or a mix.

## Decision

System prompts are written in **English**, but include a few-shot
example or two in the target language(s) when the task is
language-specific (e.g. punctuation/grammar cleanup of Chinese text).

## Why

1. English is the canonical language of the open-source models we use.
   Instructions land more reliably in English.
2. Few-shot examples in the target language make up the gap by showing
   the model what good output looks like.
3. Source code stays Rule-3 clean (no CJK in `.py` files). Prompt files
   under `prompts/` are allowlisted, so target-language examples live
   there.

## Consequences

- Adding a new task type: write the English instructions first, then add
  one or two Chinese/English examples as needed.
- Reviewing prompts is easier for non-Chinese readers because the
  control logic is always English.

## Related

- ADR-002: DeepSeek as default LLM provider.
- Spec: `docs/specs/2026-05-28-chatbot-with-tools-design.md`.
