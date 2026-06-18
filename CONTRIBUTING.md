# Contributing to TrajectoryCache

Thank you for your interest in contributing! This document outlines the process for getting involved.

## Development Setup

```bash
git clone https://github.com/your-org/trajectorycache.git
cd trajectorycache
pip install -e ".[all]"
pre-commit install   # optional but recommended
```

## Branching Model

- `main` — stable releases only
- `develop` — integration branch, PRs target here
- `feature/<name>` — new features
- `fix/<name>` — bug fixes

## Code Style

We use **black** (formatter) and **ruff** (linter):

```bash
make format    # auto-format
make lint      # check issues
make type-check
```

All CI checks must pass before a PR is merged.

## Adding a New Cache Policy

1. Create `src/trajectorycache/cache/mypolicy.py` subclassing `BaseCache`.
2. Implement `request(item_id, item_location, current_time, **kwargs) -> bool`.
3. Register it in `src/trajectorycache/cache/__init__.py` → `REGISTRY`.
4. Add unit tests in `tests/unit/test_baselines.py` or a new file.
5. The benchmark runner automatically picks it up.

## Testing

```bash
make test            # full suite
make test-unit       # fast unit tests only
make smoke           # quick sanity check
```

All new code requires unit tests. Integration tests are required for new modules.

## Pull Request Checklist

- [ ] Tests pass (`make test`)
- [ ] Code formatted (`make format`)
- [ ] Lint clean (`make lint`)
- [ ] Docstrings on public functions/classes
- [ ] `IMPLEMENTATION_STATUS.md` updated if applicable
- [ ] No hardcoded secrets or credentials

## Reporting Bugs

Open a GitHub issue with:
- Python version
- Steps to reproduce
- Expected vs actual behaviour
- Relevant log output
