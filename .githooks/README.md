# Git hooks for author enforcement

This repository includes a local Git hook that blocks commits attributed to an AI coding agent (Claude, Codex, OpenAI, etc.). It does not hardcode any identity: the expected author is whatever you have set in `git config user.name` / `user.email`.

## Setup

From the repo root:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

## What it does

- Blocks commits when `GIT_AUTHOR_*` / `GIT_COMMITTER_*` look like an AI agent.
- Blocks commits when the effective `user.name` / `user.email` is an agent identity.

## Notes

It protects future commits only; it does not rewrite existing history.
