# Skills and MCP Servers

This file is the canonical registry of Claude Code skills and MCP servers
adopted across MiraNote-AI repos. Entries here are required by Rule 5 in
[CONTRIBUTING.md](../../CONTRIBUTING.md).

To add a skill or MCP server:

1. Add an entry to the relevant section below with a short description and
   a link to its configuration / source.
2. If the skill/MCP is configured at the repo level (e.g., in
   `.claude/settings.json`), reference that configuration here.
3. PR into `MiraNote-AI/.github`.

## Skills

Org skills live in [`skills/`](https://github.com/MiraNote-AI/.github/tree/main/skills)
in `MiraNote-AI/.github` and are synced into each code repo's
`.claude/skills/` by `sync-ai-docs.yml`.

| Skill | Purpose |
|---|---|
| `create-ticket` | File a GitHub issue before non-trivial work so the PR body can reference it (Rule 6). |
| `create-pr` | Make a PR satisfy Rules 6 and 8 -- title/body preflight using the same scripts CI runs. |
| `run-loop` | Contract for autonomous multi-iteration agent loops: goal, stop conditions, budgets, state persistence, terminal state. |
| `verify-repo` | Per-repo registry of trustable verification commands, and what an agent may claim where no verifier exists. |

## MCP Servers

_None registered. Machine-local MCP configs (`.mcp.json`) are per-user and
stay uncommitted; register here only servers adopted as team infrastructure._
