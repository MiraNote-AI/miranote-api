# Contributing to MiraNote

MiraNote ships as five repos under `MiraNote-AI`. This file is the **single
source of truth** for engineering rules across all five. It lives in
`MiraNote-AI/.github` and is synced to each code repo via the
`sync-ai-docs.yml` workflow.

## How to propose a change

1. Branch from `main` in `MiraNote-AI/.github`.
2. Edit `CLAUDE.md`, `CONTRIBUTING.md`, or files under `docs/ai/`.
3. If you add a rule, follow the procedure below.
4. Open a PR; the self-check workflow must pass before merge.

## How to add a rule

Every rule in this document must have a corresponding check script. To add
a new rule:

1. Write a check script at `checks/<name>.py` following the existing
   patterns (validator function + thin CLI wrapper).
2. Add a `### Rule N: <title>` section below, with the required
   `**Rationale:**` and `**Enforced by:**` lines.
3. Locally run:
   ```bash
   PYTHONPATH=. python3 -m checks.contributing_format . --mode source
   PYTHONPATH=. python3 -m checks._meta.all_rules_have_checks .
   ```
4. PR into `MiraNote-AI/.github`.

Rule IDs are stable: do not renumber on deletion. Gaps are allowed.

## Rules

### Rule 1: Every rule has an enforcement mechanism

Every rule documented here must be paired with at least one executable check
script in `checks/`. Rules without programmatic enforcement do not belong in
this file; if a constraint cannot be checked, do not document it as a rule.

The alpha check also detects **orphan** check scripts -- every `.py` file at the
direct top level of `checks/` (excluding `__init__.py`) must be referenced
by at least one rule's `Enforced by:` line.

When gamma reports parse errors (CONTRIBUTING.md structural problems), alpha
skips orphan detection to avoid false-positive cascades; fix the parse errors
first, then rerun alpha to see any remaining orphans.

**Rationale:** Without paired checks, rules degrade to wall-art over time.
**Enforced by:** `checks/_meta/all_rules_have_checks.py`

### Rule 2: CONTRIBUTING.md follows the canonical structure

This file must match the structure that the gamma check parses: a `## Rules`
section, with each rule starting `### Rule N: <title>` (unique integers,
gaps allowed) and containing exactly one `**Rationale:**` line and one
`**Enforced by:**` line. Path tokens on the `Enforced by:` line are
comma-separated; surrounding backticks and whitespace are stripped.

In `mode=source` (running inside `MiraNote-AI/.github`), every path on
every `Enforced by:` line must resolve to an existing file. In
`mode=target` (running inside a code repo), path resolution is skipped
because the check scripts live only in the source repo.

**Rationale:** alpha and downstream automation depend on this file being
mechanically parseable; structural drift here would silently break the
entire framework.
**Enforced by:** `checks/contributing_format.py`

### Rule 3: No CJK characters or emoji in committed files

Source files, docs, and any other text committed to the repo must not
contain Chinese, Japanese, Korean, fullwidth-form, or emoji characters
within the Unicode ranges scanned by beta. Text-presentation symbols (check
marks, arrows, box-drawing) are allowed because they have legitimate use in
tables and diagrams.

The beta check scans the file set returned by
`git ls-files --cached --others --exclude-standard` -- including
locally-staged-but-untracked files, so local results match CI.

**Rationale:** A multilingual team can easily leak input-method residue
into committed source. DASGPT was bitten by this in production.
**Enforced by:** `checks/no_cjk_or_emoji.py`

### Rule 4: CLAUDE.md is at most 80 lines

The `CLAUDE.md` at the root of each repo is consumed by every Claude Code
session as part of its context. It must stay short -- it is an entry point
and navigation aid, not the place for detailed rules. Detail belongs in
`docs/ai/` and in this `CONTRIBUTING.md`.

**Rationale:** Context budget is a finite resource; entry-point files that
grow unchecked degrade every downstream task.
**Enforced by:** `checks/claude_md_size.py`

### Rule 5: Skills and MCP servers are registered in `docs/ai/skills.md`

Any Claude Code skill, MCP server, or external tool integration adopted by
the team must be declared in `docs/ai/skills.md` under the `## Skills` or
`## MCP Servers` section. The day-0 check verifies only that the file
exists with both required headers; future iterations will diff this
registry against actual configuration.

**Rationale:** The "50 MCP servers running" anti-pattern destroys tool
discovery; a deliberate registry forces every adoption to be a conscious
decision.
**Enforced by:** `checks/skills_registry.py`

### Rule 6: PR descriptions reference an issue, spec, or URL

Every pull request body must contain at least one of:
- `#<integer>` referencing an issue or PR,
- a URL (`http://` or `https://`),
- one of `spec:`, `design:`, `adr:`, `rfc:` followed by a token.

PRs from the sync bot (branch matching `chore/sync-ai-docs-*`) are exempt.

**Rationale:** A PR without a reference is opaque to future readers; the
"why" must be discoverable from the PR itself.
**Enforced by:** `checks/pr_has_reference.py`

### Rule 7: Synced files cannot be modified inside a target repo

The files synced from `MiraNote-AI/.github` to each code repo
(`CLAUDE.md`, `CONTRIBUTING.md`, `docs/ai/**`, `.claude/skills/**`,
`.github/workflows/checks.yml`) must be edited only at the source. Direct
edits to these paths inside a target repo will cause the eta check to fail.

The bot's own sync PRs (branch matching `chore/sync-ai-docs-*`) are
exempt -- those are the legitimate update path.

This is soft enforcement: branch names are spoofable. Hard enforcement
requires branch protection rules and CODEOWNERS, which are tracked
separately under sub-project F.

**Rationale:** Without this check, accidental local edits to synced files
silently diverge from the canonical source until the next sync round-trip,
producing confusion and lost edits.
**Enforced by:** `checks/protected_paths.py`

### Rule 8: PR title follows Conventional Commits format

Every pull request title must be self-explanatory to readers outside the
immediate working session. Concretely:

- Title starts with a Conventional Commits prefix from this set: `feat`,
  `fix`, `chore`, `docs`, `refactor`, `test`, `ci`, `perf`, `build`,
  `revert`. An optional `!` may follow the prefix to mark breaking changes.
- An optional scope in parentheses must come from the whitelist:
  `api`, `web`, `ios`, `bot`, `infra`. Omit the scope for cross-cutting
  changes. Internal codenames (`F-2`, `F-final`, `step-3`) are rejected
  because the scope must be a real component name.
- The prefix is followed by exactly `: ` and a description.
- Description starts with a lowercase letter and does not end with a period.
- Description uses the imperative mood: `add` not `added`/`adding`,
  `fix` not `fixed`/`fixing`.
- Total title length is at most 72 characters.
- Title must not contain `#<integer>` issue references (move them to the
  body, where Rule 6 already requires a reference).
- Title must not contain `WIP`, `DRAFT`, `FIXME`, or `TODO` markers; use
  GitHub's Draft PR state instead.

PRs from the sync bot (branch matching `chore/sync-ai-docs-*`) are exempt.

**Rationale:** PR titles surface in GitHub lists, release-tooling input,
changelog generators, and Slack/email notifications. A title that means
something only to the author at write-time becomes opaque to teammates and
to the author's future self; mechanical enforcement removes the social cost
of asking for renames.
**Enforced by:** `checks/pr_title_format.py`
