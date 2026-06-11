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

1. RED baseline: 3 new corpus tests written first, all FAILED against
   the committed corpus (author 100 percent Taizong / Song 0 / 75 empty
   themes) -- detection power proven. Implemented diverse_select
   (seeded shuffle over full pool, author cap 2 percent, era split,
   fuzzy dedupe), widened source reading to 40 files per glob with
   empty-author and U+25A1 filters, batch-tagging retries (parse
   failure now returns None -> up to 3 attempts), --langs/--seed args.
   Rebuild (zh only): pool 263,524 lines -> 500 selected, 254 authors,
   top author Du Fu at 10 (the cap), eras 250/250, empty themes 0; one
   live retry observed succeeding (batch at 425). Suite 34/34 GREEN
   (31 prior + 3 new). EN file untouched (git status clean for it).
   Index rebuilt 1000. Live ZH smoke: 3 picks from 3 distinct authors
   (Zeng Di, Pi Rixiu, Li Liuqian), scores ~0.57. README updated to
   the new reality.

## Deviations and decisions

- EN corpus deliberately untouched (--langs zh) to avoid re-tagging
  churn on a healthy file; the --langs flag is the durable mechanism.
- sources/ staged via a temporary symlink to the main checkout's copy
  (removed before commit -- the dir-pattern gitignore does not cover
  symlinks, so it would have shown as untracked).
