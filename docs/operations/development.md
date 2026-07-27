# Development

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,research]"
```

## Validation

```bash
pytest
ruff check .
mypy src
```

## Working Agreement

- Keep reusable logic inside `src/qr_haven`.
- Keep exploratory work in `research/` and `notebooks/`.
- Add tests when behavior becomes reusable or terminal-facing.
- Prefer typed interfaces for anything that will cross into `market_terminal`.

