# Commit and Release Checklist

## Pre-commit
- [ ] Run `python3 -m pytest -q`
- [ ] Run `python3 -m pytest --cov=prediction_bot --cov-report=term-missing -q`
- [ ] Confirm no unintended file moves or generated artifacts are staged
- [ ] Review diffs for `prediction_bot/`, `tests/`, `README.md`, and `pyproject.toml`
- [ ] Ensure model/data/cache artifacts are excluded from commit

## Suggested commit grouping
- [ ] Package rename to `prediction_bot`
- [ ] Temporal split/schema/evaluator improvements
- [ ] Optimization pass
- [ ] CLI and docs improvements
- [ ] Test additions

## Pre-release validation
- [ ] Install package in editable mode: `python3 -m pip install -e .[dev]`
- [ ] Verify CLI help: `prediction-bot --help`
- [ ] Verify module entrypoint: `python3 -m prediction_bot --help`
- [ ] Run a train command on a small symbol set
- [ ] Run a predict command against a produced artifact
- [ ] Run an evaluate command and inspect output JSON

## Documentation
- [ ] Confirm README examples still match actual CLI flags
- [ ] Confirm historical docs mentioning `src` are either updated or clearly archival
- [ ] Document trusted-local artifact assumption for `joblib.load`

## Release prep
- [ ] Bump version in `pyproject.toml` if releasing
- [ ] Tag release in git if desired
- [ ] Save final test/coverage outputs for traceability
- [ ] Optionally generate a changelog summary

## Current known non-blockers
- [ ] `prediction_bot/__main__.py` may remain lightly covered unless explicitly smoke-tested
- [ ] External dependency warnings from environment are not currently app failures
