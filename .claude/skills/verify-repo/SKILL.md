---
name: verify-repo
description: Use when an agent needs a trustable pass/fail signal in any MiraNote-AI repo -- before claiming work is done, before opening a PR, and as the stop-condition check inside any loop (see run-loop).
---

# verify-repo

## Overview

The CI status check (`checks / checks`) proves GOVERNANCE compliance only:
docs format, no CJK/emoji, CLAUDE.md size, skills registry, PR title/body,
protected paths. It never runs tests, lint, or builds for application code.
Code-correctness signals are local and repo-specific. This skill is the
registry of what an agent may trust, per repo -- and of where no trustable
signal exists yet.

## Governance checks (every repo)

The check scripts live only in `MiraNote-AI/.github`, and CI fetches them
from `@main` at run time -- so never run them from a feature branch, and do
not trust a target repo's synced docs (they can lag `@main`). To match CI
exactly without disturbing anyone's working branch, use a temporary
worktree pinned to `origin/main`:

```bash
git -C path/to/MiraNote-AI/.github fetch origin
git -C path/to/MiraNote-AI/.github worktree add /tmp/mn-checks origin/main
cd /tmp/mn-checks
PYTHONPATH=. python3 -m checks.contributing_format <repo-path> --mode target
PYTHONPATH=. python3 -m checks.no_cjk_or_emoji <repo-path>
PYTHONPATH=. python3 -m checks.claude_md_size <repo-path> --max 80
PYTHONPATH=. python3 -m checks.skills_registry <repo-path>
git -C path/to/MiraNote-AI/.github worktree remove /tmp/mn-checks
```

(If the clone already sits clean on up-to-date `main`, running from it
directly is fine; the worktree dance is for every other case.)

Note: the CJK/emoji check scans untracked files too (`git ls-files --cached
--others --exclude-standard`), so local scratch files count. PR title/body
preflight (`pr_title_format`, `pr_has_reference`) is covered by `create-pr`.

## Per-repo code verifiers

### MiraNote-AI/.github (source repo)

```bash
PYTHONPATH=. python3 -m checks.contributing_format . --mode source
PYTHONPATH=. python3 -m checks._meta.all_rules_have_checks .
PYTHONPATH=. python3 -m checks.no_cjk_or_emoji .
PYTHONPATH=. python3 -m checks.claude_md_size . --max 80
PYTHONPATH=. python3 -m checks.skills_registry .
PYTHONPATH=. python3 -m unittest discover checks/tests -v
```

All six must pass. Green here implies green CI for the non-PR-event stages.
Remember: a push to `main` here triggers the sync workflow, which opens bot
PRs in all four code repos -- changes in this repo have org-wide blast radius.

### miranote-api

Four offline pytest suites (counts grow as the suites do -- trust the
collected totals from the commands below). No network and no API keys
needed: LLM and model calls are stubbed inside each suite, in its
`tests/conftest.py` or in the test modules themselves.

```bash
# from the repo root:
PYTHONPATH=. poc/chatbot/.venv/bin/python3 -m pytest poc/chatbot/tests -v
PYTHONPATH=. poc/retrieval/.venv/bin/python3 -m pytest poc/retrieval/tests -v
# these two run from their POC directory:
cd poc/voice-to-text     && PYTHONPATH=. .venv/bin/python3 -m pytest tests/ -v
cd poc/text-clean-expand && PYTHONPATH=. .venv/bin/python3 -m pytest tests/ -v
```

Caveats:
- Each POC needs its own `.venv` (setup commands in `poc/<name>/README.md`).
- Code must stay Python 3.9-compatible (`from __future__ import annotations`,
  `typing.Optional`/`List`/`Dict`; no PEP 604 `X | None`).
- Run the CJK/emoji governance check after every commit.
- `start-all.sh` is a long-running smoke harness for humans, not a loop
  verifier.

### mirabot

NO functional verifier exists: zero tests, no lint, no typecheck, and CI
checks docs only. The only trustable signals are syntax-level:

```bash
python3 -m py_compile bot.py
node --check mcp-server/src/server.js
```

`docker build .` also works but is heavy (Playwright chromium). Running the
bot or the MCP server requires live Discord credentials and is never a loop
signal.

RULE: no behavioral-correctness claims in this repo. A loop here may only
claim what syntax checks prove. The first loop task in this repo should be
adding a real test harness (pytest + mocked Discord/AI clients) so later
loops have something to trust.

### miranote-web / miranote-ios

No application code exists, so no code verifier exists -- governance checks
only. A loop that scaffolds the app MUST bring its own verifier: tests and
lint land in the same PR as the first code ("no verifier, no loop").
Neither repo has a `.gitignore` yet; add one before generating any local
artifacts.

## Trust rules

1. Green CI does not mean correct code -- in code repos CI is governance
   only.
2. No verifier -> no claim. State exactly what was verified and how.
3. Never weaken, skip, or delete a failing check or test to get to green;
   stop and escalate instead.
4. Verify from a clean state: fresh shell, no leftover env exports, no
   half-installed dependencies.
5. The `.github` repo at `origin/main` defines what CI enforces; a target
   repo's synced docs may be older. When in doubt, trust `@main`.

## Common mistakes

| Mistake | Fix |
|---|---|
| Declaring an api change done because CI is green | Run the four pytest suites; CI never runs them |
| Claiming a mirabot behavior change works | Only syntax-level claims are available; say so explicitly |
| Verifying with a stale or feature-branch `.github` checkout | Fetch, then run from a temp worktree at `origin/main`; `git pull` only advances the current branch |
| Scratch files trip the CJK/emoji check | It scans untracked files; keep scratch English-only or outside the repo |
| Editing a test until it passes | That is weakening the verifier; stop and escalate |
