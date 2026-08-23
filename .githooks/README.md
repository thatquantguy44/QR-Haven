# Git hooks for author enforcement

This repository includes a local Git hook that ensures commit metadata stays attributed to Joshua Lutkemuller, CFA, even if an AI coding agent tries to create a commit.

## Setup

From the repo root:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

## What it does

- Forces the local Git identity to:
  - Name: Joshua Lutkemuller, CFA
  - Email: 110635594+joshualutkemuller@users.noreply.github.com
- Blocks commits when the author or committer looks like an AI agent such as Claude or Codex.

## Notes

This prevents casual agent commits from being recorded under a non-human identity within this repo.
It does not rewrite existing history; it protects future commits.
