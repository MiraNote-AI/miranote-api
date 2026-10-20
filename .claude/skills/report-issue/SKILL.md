---
name: report-issue
description: Use when someone found a problem or has an idea while USING a MiraNote app (often a screenshot, a Discord message or a one-line description) and wants it on the team's shared bug list -- dedupes against open and recently closed issues, reproduces it, and files a Problem / Context / Acceptance criteria GitHub issue like miranote-ios#120.
---

# report-issue

## Overview

The team's shared bug list is GitHub Issues (team meeting, 2026-10-19).
Reference example: [miranote-ios#120](https://github.com/MiraNote-AI/miranote-ios/issues/120).
This skill turns "I saw X" into an issue like that in a few minutes, so anyone
can pick it up. For planned work you are about to build, use **create-ticket**.

**Every MiraNote-AI repo is public.** Anything in an issue or comment --
words, screenshots, logs -- can be read by anyone (see step 0).

## When to use

- A tester reports something from TestFlight or the simulator ("the keyboard
  won't go down", "Add does nothing"), with or without a screenshot.
- A Discord message lists several findings -- file one issue per problem.
- An idea or a "should this work like X?" question about existing behaviour.

## Workflow

0. **Keep it public-safe.**
   - Screenshots from the simulator or a test page; a tester's own
     screenshot is cropped or blurred first (their photos, words, places).
   - Never copy keys, tokens, `Authorization` headers, `.env` or
     `Secrets.xcconfig` values, or anyone's name, email or details.
   - A security or data-exposure finding goes to the team in Discord, never
     into a public issue.
1. **Pick the repo** -- create-ticket's repo picker: app behaviour ->
   `miranote-ios`; server results (image generation, chat replies,
   transcription) -> `miranote-api`; web -> `miranote-web`; the Discord bot
   -> `mirabot`.
2. **Dedupe first, open AND closed.** Search with 2-3 short English keywords
   or the accessibility identifier (not the tester's wording):
   `gh search issues --owner MiraNote-AI "<keywords>"` (every repo, any state),
   or per repo `gh issue list -R MiraNote-AI/<repo> --state all --search "<keywords>"`.
   - **Open match**: comment the new evidence there
     (`gh issue comment <n> -R MiraNote-AI/<repo> --body-file <file>`), share
     that issue's URL with the reporter, do step 7 for it, then go on to the
     next finding. One problem = one issue.
   - **Closed match** (fixed on dev -- issues close when the fix merges, but
     testers keep their build until the next one): if the tester's build
     predates the fix, reply that it is fixed on dev and coming in the next
     build; if their build already has the fix, file a new issue that links
     the closed one.
3. **Reproduce** (if impossible, say so in Context). If it cannot be
   reproduced and the report lacks the screen, the action, what was expected
   and the build, ask the reporter first.
   - App (`miranote-ios`): a scratch worktree of the latest dev, removed
     afterwards: `git fetch origin`, `git worktree add <dir> origin/dev`,
     `xcodegen generate` in it. Find the code path (search the accessibility
     identifier or the label in the screenshot), drive it on one simulator by
     hand or with a throwaway UI test, using the `-UITEST` mocks and the
     delay hooks for timing problems (`docs/testing/adversarial-harness.md`).
     `git -C <dir> rev-parse --short HEAD` is the commit you ran. Never commit
     the throwaway test.
   - Server (`miranote-api`, `mirabot`): `-UITEST` scripts transcription, chat
     and image generation in the app, so reproduce against the backend
     itself; record which backend answered, the model names from its
     `/health`, and what was sent (no tokens -- step 0).
4. **Label** (GitHub defaults only, never invent labels):
   - `bug` -- broken or behaves wrongly;
   - `enhancement` -- works, but should be better;
   - `question` -- the code does what the design says, or a product call is
     needed (Jason / Gloria decide). Combine with `bug`/`enhancement` if both
     apply (`--label question --label bug`).
5. **Write the issue** into a temp file OUTSIDE the repo (`mktemp`), in
   English (team convention):

   ```
   ## Problem
   <what the user sees, in user terms, one short paragraph. A screenshot the
   reporter could not attach is described in words.>

   Seen on: TestFlight <version (build)>, <device>, iOS <x>   (from the reporter)
   Reproduced on: <simulator>, <repo> @ <commit>             (what you ran)

   <how to reproduce: numbered steps. Use a small table when several places
   behave differently (place | action | what happens).>

   ## Context
   <relevant code (types, files, identifiers); the cause if known for sure;
   related issues / PRs (#n, or MiraNote-AI/<repo>#n in another repo); what
   was NOT checked.>

   ## Acceptance criteria
   - [ ] <outcome the user would observe>
   - [ ] <...>
   - [ ] <the test that proves it, failing before the fix>
   ```

   Title: a problem-focused sentence, at most ~80 characters, no `#N`, no
   backticks (e.g. `Only the Send key lowers the keyboard (Home ask bar,
   editor fields)`). Criteria are observable outcomes, not implementation
   steps.
6. **File it** -- the title from a file too, so quotes, backticks and `$`
   stay as typed:
   `gh issue create -R MiraNote-AI/<repo> --title "$(cat <title-file>)" --label <label> --body-file <body-file>`
   and share the URL.
7. **Screenshots**: the CLI cannot attach images. Ask the reporter to drag the
   cropped screenshot (problem circled in red) into the issue's comment box; a
   TestFlight tester without a GitHub account sends it to the filer, who adds
   it.
8. **Ownership**: assign nobody unless asked; whoever takes it assigns
   themselves. The fixing PR says `Closes #<n>` (see create-pr), which closes
   the issue on merge.

## Batch mode

Several findings at once: group them by problem, dedupe each problem, file one
issue per problem, then reply with a short list -- `MiraNote-AI/<repo>#n title
(label)` (repo-qualified: one batch can file into several repos).

## Common mistakes

| Mistake | Fix |
|---|---|
| A second issue for a known problem | Comment on the existing issue instead |
| Searching open issues only | A fix merged to dev closes the issue before testers get it: search all states |
| `gh issue comment <n>` without `-R` | It lands on #n of the repo the shell is in; always pass `-R MiraNote-AI/<repo>` |
| A tester's photo, name or a token in a public issue | Crop or blur, describe in words, keep secrets out; security goes to Discord |
| A guess written as the cause | Only state causes you confirmed; otherwise "not checked" |
| Criteria are code changes ("add a focus state") | Rewrite as what the user sees ("a tap outside lowers the keyboard") |
| Two problems in one issue | Split them, cross-link with `#n` |
| No repro steps | Steps + Seen on + Reproduced on, or say why it could not be reproduced |
