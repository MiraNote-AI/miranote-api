---
name: run-loop
description: Use when starting any multi-iteration autonomous task (agent loop) in a MiraNote-AI repo -- defines the goal contract, stop conditions, budgets, state persistence, and the terminal state every loop must reach.
---

# run-loop

## Overview

A loop is an autonomous work cycle: plan -> act -> verify -> repeat, until the
acceptance criteria pass or a stop condition fires. This skill is the org-wide
contract every loop must follow. It exists so that a loop's progress is
verifiable from disk at any moment, and so that a loop can never "succeed" by
weakening its own checks or outrunning its budget.

Companions: `verify-repo` (what "passing" means in each repo),
`create-ticket` (file the issue first), `create-pr` (compliant PR at the end).

## When to use

- Any task expected to take more than one act-verify iteration.
- Executing a checkbox plan from `docs/plans/`.
- Scheduled or background agent runs.

Do not use:
- Single-edit tasks (typo, comment, config tweak) -- just do them.
- Pure questions / analysis with no repo changes.

## The loop contract

Write these down in the plan file BEFORE the first iteration:

0. **Issue** -- file it first (`create-ticket`) and put `Refs #N` in the plan
   file header. The issue's acceptance criteria and the plan's Goal must
   match; scope creep is measured against this issue.
1. **Goal** -- observable acceptance criteria. Every criterion must be
   checkable by a command from `verify-repo`, or be explicitly marked
   `HUMAN:` (a reviewer judgment, out of loop scope).
2. **Stop conditions** -- all of:
   - success: every acceptance criterion passes AND the independent check
     (below) passes;
   - iteration cap: default **5** per loop. An iteration is one act+verify
     cycle ending in a recorded verify result. Raising the cap is a written
     decision recorded in the iteration ledger; removing it is forbidden;
   - no-progress rule: 2 consecutive iterations without progress -> stop
     and write the handoff. Progress = a criterion newly passing, or the
     failing test/check count strictly decreasing;
   - escalation triggers: a protected path would need editing; a failing
     check would need weakening; the work grows beyond the referenced issue.
3. **Budget** (scheduled loops): wall-clock and/or token ceiling. A loop
   that hits its budget stops mid-task and writes the handoff -- it does
   not sprint to a worse answer.

## The iteration ledger

The plan file carries an `## Iterations` section. Append one line at the end
of every iteration:

```
N. <what changed> -- criteria X/Y passing, <verify summary or exit codes>
```

Cap raises and budget consumed are recorded here too. The ledger is how the
cap survives context loss and session boundaries. If the ledger is missing
or stale, the loop must stop: an uncounted loop is an unbounded loop.

## Independent verification (maker-checker)

The agent that wrote a change does not get to declare it done:

- Run the `verify-repo` commands from a clean state and record the results
  in the iteration ledger.
- Before SUCCESS, a fresh-context subagent reviews the work. It receives
  ONLY the acceptance criteria and `git diff <base>...HEAD` -- not the
  conversation that produced them. Record its verdict in the ledger. A
  "not done" verdict means the iteration failed: iterate again (it counts
  against the cap) or hand off.
- Only exception: changes that touch no application code (docs/plan-only)
  may skip the subagent; `verify-repo` alone gates those.
- "Looks right" is not a signal. Only command output is.

## State persistence

The loop's memory lives on disk, never only in the context window:

- **Plan**: `docs/plans/YYYY-MM-DD-<slug>.md` -- committed. Carries the
  contract, checkboxes, and the iteration ledger. If `docs/plans/` or
  `docs/specs/` does not exist in the repo yet, create it -- they are
  repo-local, NOT protected paths (only `docs/ai/**` is). Plans may carry
  a `REQUIRED SUB-SKILL` header naming the execution skill to use.
- **Spec** (if any): `docs/specs/YYYY-MM-DD-<slug>-design.md`, committed.
- **Deviations and decisions**: anything that departs from the plan gets
  written into the committed plan (or the PR body) before the loop ends --
  never only into the handoff.
- **Session handoff**: `.handoff-<slug>.md` at repo root -- UNTRACKED,
  rewritten in place. Written when a stop condition fires or a session ends
  before SUCCESS: branch state, open PRs, iteration count, deviations, and
  a decision tree for the next session. On SUCCESS, fold anything still
  useful into the plan or PR body, then delete the handoff. Keep it
  English-only (the Rule 3 check scans untracked files too). Add
  `.handoff-*.md` to the repo's `.gitignore` in the loop's first commit if
  it is not already there.

## Worktrees (parallel loops)

- One loop = one branch = one checkout. Never run two loops in the same
  working directory.
- Parallel loops use sibling worktrees:
  `git worktree add ../<repo>-wt-<slug> -b <type>/<slug>`
- Remove the worktree only after the PR is open and the handoff content has
  been folded into the plan or PR body.

## Scheduled loops

- A scheduled loop carries the same contract PLUS a budget, with consumption
  recorded in the iteration ledger. No exceptions.
- Scheduler / cron / MCP configuration is machine-local: do not commit it
  to any repo.
- A scheduled loop that hits a stop condition writes the handoff and exits;
  it never retries past its own cap.

## Terminal state

Every loop ends in exactly one of two states:

1. **SUCCESS**: all criteria pass, fresh-context check passed ->
   `create-pr` -> PR open with CI green -> STOP. A human reviews and
   merges. Never `gh pr merge --admin`, never self-approve, never merge
   your own PR. The PR body must say, in plain language: what changed, why
   this approach, and what was NOT verified (every `HUMAN:` criterion). If
   the diff is too large to review in one sitting, split it -- an
   unreviewable PR is a failed loop, not a finished one.
2. **HANDOFF**: a stop condition fired -> handoff file written -> STOP and
   report which condition fired and why.

Never, in any state:
- edit protected paths in a code repo (`CLAUDE.md`, `CONTRIBUTING.md`,
  `docs/ai/**`, `.claude/skills/**`, `.github/workflows/checks.yml`);
- use the reserved bot branch prefix `chore/sync-ai-docs-*`;
- weaken, skip, or delete a failing check or test to reach green
  (`verify-repo`, trust rule 3).

## Red flags

- An acceptance criterion no command can check, and it is not marked `HUMAN:`.
- The plan file has no iteration ledger, or the ledger is stale.
- The loop "fixed" a failing check by editing the check.
- The fresh-context check was skipped for a change that touches code.
- Two loops sharing one checkout.
- A PR merged by the loop that opened it.
- A PR that cannot be understood from its body plus the plan file.

Any of these = stop the loop, write the handoff, escalate to a human.
