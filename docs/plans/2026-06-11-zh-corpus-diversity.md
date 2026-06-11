# zh corpus diversity rebuild

Refs #20
Branch: `fix/api-zh-corpus-diversity` (worktree `../miranote-api-wt-corpus`)

## Goal (acceptance criteria)

- [ ] AC1: no single author exceeds 10 percent of quotes_zh.json
      (test_zh_author_diversity).
- [ ] AC2: Tang and Song both represented, at least 50 entries each
      (test_zh_both_eras_present).
- [ ] AC3: empty-theme entries at most 5 percent of zh corpus; the build
      script retries failed tagging batches up to 3 attempts
      (test_zh_empty_theme_cap + script logic).
- [ ] AC4: text/author/source remain verbatim from sources; LLM tags only.
      EN corpus byte-identical (rebuild runs with --langs zh).
- [ ] AC5: full retrieval suite green including the 3 new tests; index
      rebuilds to corpus size; live ZH smoke returns picks from more than
      one author.
- [ ] AC6: Rule 3 exit 0.
- [ ] AC7: PR open, CI green. Merge stays human.

HUMAN: none.

## Stop conditions

Cap 5 iterations; no-progress 2; escalate on check-weakening or scope
growth beyond #20.

## Verifier

Retrieval suite via the canonical venv; Rule 3 from dotgithub main;
build script run with --langs zh; index build + store count; live
/quotes smoke.

## Method note (TDD)

The three diversity tests are written FIRST and must FAIL against the
committed corpus (single-author, Song=0, 75 empty themes) before the
rebuild makes them pass -- the RED run is the proof they can detect the
regression they guard.

## Iterations

## Deviations and decisions
