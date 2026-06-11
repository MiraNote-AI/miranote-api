# MiraNote — Rules for Claude Code

This file is the canonical entry point for AI coding assistants working in
any `MiraNote-AI/*` repo. Edit this file only in `MiraNote-AI/.github`;
the `sync-ai-docs.yml` workflow propagates changes to every code repo.

## Required reading order

1. [CONTRIBUTING.md](CONTRIBUTING.md) -- the full rule set with rationale
   and enforcement. Every line in this file is enforced by a check script
   in `checks/`.
2. [docs/ai/README.md](docs/ai/README.md) -- navigation for engineering
   docs (architecture, workflows, playbooks, ADRs as they're added).
3. [docs/ai/skills.md](docs/ai/skills.md) -- registry of skills and MCP
   servers adopted across MiraNote-AI.

## The seven day-0 rules (summarised)

1. **Meta-rule** -- every rule has a check.
2. **CONTRIBUTING.md format** -- registry structure is mechanically
   parseable.
3. **No CJK/emoji** -- committed text stays in ASCII-compatible
   ranges. Text-presentation symbols are OK.
4. **CLAUDE.md <= 80 lines** -- entry point stays tight.
5. **Skills/MCP registry** -- every adopted skill/MCP listed in
   `docs/ai/skills.md`.
6. **PR has reference** -- every PR body references an issue, URL,
   spec, design, ADR, or RFC.
7. **Protected paths** -- synced files (this file,
   `CONTRIBUTING.md`, `docs/ai/`, `.github/workflows/checks.yml`) are
   edited only in `MiraNote-AI/.github`.

Full text and enforcement details are in [CONTRIBUTING.md](CONTRIBUTING.md).

## Post-day-0 rules

8. **PR title format** -- Conventional Commits prefix (`feat`/`fix`/`ci`/...),
   whitelisted scopes (`api`/`web`/`ios`/`bot`/`infra`), imperative mood,
   no `#N` / `WIP`/`TODO` markers, <= 72 chars.

## Quick local commands

```bash
PYTHONPATH=. python3 -m checks.contributing_format . --mode source
PYTHONPATH=. python3 -m checks._meta.all_rules_have_checks .
PYTHONPATH=. python3 -m checks.no_cjk_or_emoji .
PYTHONPATH=. python3 -m checks.claude_md_size . --max 80
PYTHONPATH=. python3 -m checks.skills_registry .
PYTHONPATH=. python3 -m unittest discover checks/tests -v
```

## How to add a rule

See the procedure in [CONTRIBUTING.md](CONTRIBUTING.md). Briefly:
write the check, register the rule, run the two meta-validators locally,
PR.

## Out of scope (deferred)

- CODEOWNERS, required-status-check enforcement -- sub-project F follow-ups.
  Branch protection itself is live on all 5 repos.
- Per-stack harness (web/api/ios linters, tests, settings.json hooks) --
  sub-project D.
- Shared org-level skills and memory -- sub-project E.
- Local pre-commit hooks -- nice-to-have, post-day-0.
