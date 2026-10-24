# Getting Started Guide

## Prerequisites & Tooling (`asdf`)

This repository uses `asdf` for `uv` and `pre-commit`, running on **Python 3.15**:

```bash
asdf install
uv sync --all-groups
pre-commit install
```

## Running the CLI

```bash
# Inspect CLI hierarchy and help
uv run agy-acp --help
uv run agy-acp acp --help
uv run agy-acp auth --help
uv run agy-acp quota --help
uv run agy-acp docs --help
```

## Building & Serving the Zensical Documentation

```bash
# Build static site into site/
uv run agy-acp docs build --clean

# Serve locally with live reload
uv run agy-acp docs serve
```

## Running Verification & Unit Tests

```bash
# Run all integration requirement verifications and unit tests
uv run pytest
```
