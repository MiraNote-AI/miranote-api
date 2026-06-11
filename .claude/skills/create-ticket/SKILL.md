---
name: create-ticket
description: Use when starting non-trivial work in any MiraNote-AI repo, to file a GitHub issue first so the subsequent pull request can link back to it and satisfy Rule 6 (PR-has-reference) without extra effort.
---

# create-ticket

## Overview

Creates a GitHub issue in the right MiraNote-AI repo before any non-trivial change is started. The issue number returned by `gh issue create` becomes the natural reference for the eventual PR body (`Closes #<n>`), which satisfies Rule 6 without the author having to invent a URL or `spec:` token.

## When to use

- About to start a new feature, refactor, multi-commit fix, or anything bigger than a typo.
- Discussing scope with a teammate -- the issue gives a place for the conversation to live.
- Working under create-pr but the body has no reference yet.

## When NOT to use

- Trivial typo / comment / formatting fixes -- overhead outweighs the benefit, and Rule 6 can still be satisfied with a URL.
- The change already has an external tracking link (Linear, design doc URL, RFC) -- that URL alone satisfies Rule 6.
- Bot sync PRs (`chore/sync-ai-docs-*`) -- exempt from Rule 6.

## Quick reference

### Repo picker

| Change scope | Repo |
|---|---|
| API endpoints, backend logic | `MiraNote-AI/miranote-api` |
| iOS app | `MiraNote-AI/miranote-ios` |
| Web frontend | `MiraNote-AI/miranote-web` |
| Discord bot, MCP server, mirabot infra | `MiraNote-AI/mirabot` |
| Cross-cutting: rules, sync workflow, CLAUDE.md, CONTRIBUTING.md, skills | `MiraNote-AI/.github` |

### Label picker

Use GitHub's default labels (present in every MiraNote-AI repo):

| Nature of work | Label |
|---|---|
| Behavior is wrong | `bug` |
| Net-new functionality, refactor, chore | `enhancement` |
| Documentation only | `documentation` |
| Discussion / open question | `question` |
| Newcomer-friendly | `good first issue` |
| Needs more hands | `help wanted` |

Labels are advisory, not enforced. If the existing set feels too coarse, raise a
separate issue to introduce a `feature`/`chore`/`refactor` split across all five
repos rather than inventing labels ad-hoc.

Check what exists before filing:

```bash
gh label list --repo MiraNote-AI/<repo>
```

### Title

- Imperative mood, problem-focused, <= ~80 chars.
- Good: `Remove unused fields from user JSON response`, `iOS sign-in crashes on slow networks`.
- Bad: `Bug`, `Fix things`, `Cleanup`, `MIRA-42`.

### Body template

```
## Problem
<what is wrong or what we need; one short paragraph>

## Context
<why this matters; links, screenshots, prior discussion>

## Acceptance criteria
- [ ] <observable outcome 1>
- [ ] <observable outcome 2>
```

Keep acceptance criteria observable (`response no longer contains field X`), not implementation steps (`refactor serializer`).

## Workflow

1. Pick the right repo from the table above.
2. Draft the title.
3. Fill in the body using the template.
4. Pick the label.
5. Run:

   ```bash
   gh issue create \
     --repo MiraNote-AI/<repo> \
     --title "<title>" \
     --label <label> \
     --body "$(cat <<'EOF'
   ## Problem
   ...

   ## Context
   ...

   ## Acceptance criteria
   - [ ] ...
   EOF
   )"
   ```

6. Note the issue number gh prints (e.g. `https://github.com/MiraNote-AI/<repo>/issues/12`).
7. Hand off to **create-pr**. In the PR body, use `Closes #<n>` (auto-closes on merge) or `Refs #<n>` (for partial work that doesn't finish the issue).

## Common mistakes

| Mistake | Fix |
|---|---|
| Filed in the wrong repo (issue is in `.github` but PR is in `miranote-api`) | Cross-repo `#<n>` does not auto-close; file the issue in the repo where the PR lands |
| Title is a one-word noun (`Cleanup`) | Make it a sentence: subject + what should change |
| Acceptance criteria are implementation steps | Rewrite as observable outcomes -- what would a reviewer check? |
| No label | Add the closest from the picker; do not invent new labels ad-hoc |
| Forgot to copy the issue number | `gh issue list --repo MiraNote-AI/<repo> --author @me --limit 5` to recover |

## Relation to create-pr

`create-ticket` is the upstream half of a two-step flow:

1. `create-ticket` -> issue #N filed
2. `create-pr` -> branch + commits + `gh pr create` with `Closes #N` in body

The `#N` reference is what makes the PR satisfy Rule 6. If you skip this skill, you must instead include a URL, a `spec:`/`design:`/`adr:`/`rfc:` token in the PR body, or both.
