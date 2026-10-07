# Git author guard

This repository includes a Git hook and a global Git template that keep commits attributed to Joshua Lutkemuller, CFA instead of AI agents such as Claude or Codex.

## How it works

Git stores both an author and a committer for every commit. If an AI tool writes those values using its own identity, the commit history will show the agent instead of the human owner.

To prevent that, this setup does two things:

1. Sets the global Git identity to the real GitHub owner
   - Name: Joshua Lutkemuller, CFA
2. Installs a pre-commit hook that blocks commit metadata when it resolves to known AI agent names such as Claude, Codex, or OpenAI.

## Global setup

The global guard is configured through the following Git settings:

```bash
git config --global user.name "Joshua Lutkemuller, CFA"
git config --global user.email "110635594+joshualutkemuller@users.noreply.github.com"
git config --global init.templatedir ~/.git-template
git config --global core.hooksPath ~/.git-template/hooks
```

The actual hook lives at:

```bash
~/.git-template/hooks/pre-commit
```

Any new repository created with `git init` inherits this hook automatically.

## Existing repo setup

To apply the same protection to an existing repository:

```bash
git config core.hooksPath ~/.git-template/hooks
```

## Verification

A commit using Claude-style metadata is rejected with an error such as:

```text
Blocked: commit identity resolved to an AI agent (GIT_AUTHOR_NAME=Claude).
Required author/committer: Joshua Lutkemuller, CFA <110635594+joshualutkemuller@users.noreply.github.com>
```

## Notes

This protects future repository history, and it is separate from rewriting past commit history. For a repo that already contains old agent-authored commits, those commits must be rewritten explicitly before the history is fully cleaned.
