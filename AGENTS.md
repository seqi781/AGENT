# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 project for a terminal-bench capable agent. Core runtime code lives in `agent/`: `core.py` owns the agent loop, `llm.py` handles model calls, `executor.py` runs commands, `trajectory.py` writes run logs, and `config.py` stores defaults. `adapters/` contains integration code such as `harbor_agent.py`. `run_task.py` is the local CLI entry point. Prompts are stored in `prompts/`, task descriptions in `tasks/`, documentation sources in `docs/`, and generated docs in `site/`. Runtime traces are written to `runs/*.jsonl`.

## Build, Test, and Development Commands

- `uv sync`: install project and development dependencies from `pyproject.toml` and `uv.lock`.
- `uv run python run_task.py --task "your task" --workdir /path/to/workdir`: run one local agent task.
- `uv run python run_task.py --task-file tasks/demo.md --model deepseek/deepseek-v4-pro`: run a saved task with an explicit model.
- `./serve_docs.sh`: serve the MkDocs documentation locally, if the script is executable.
- `uv run mkdocs build`: rebuild the static documentation in `site/`.

## Coding Style & Naming Conventions

Use standard Python formatting with 4-space indentation, type hints, and small functions with explicit responsibilities. Module and function names should use `snake_case`; classes should use `PascalCase`; constants should use `UPPER_SNAKE_CASE`. Keep comments useful and brief. Existing source and docs include Chinese text, so preserve that style where it clarifies user-facing behavior.

## Testing Guidelines

There is no formal test suite directory yet. Validate changes with focused agent runs and inspect the resulting `runs/*.jsonl` trajectory. For smoke coverage, follow the scenario documented in `docs/06-smoke-test.md`: reproduce the issue, make the minimal fix, verify output, and confirm the final artifact. When adding tests, prefer `tests/test_<module>.py` and keep fixtures small.

## Commit & Pull Request Guidelines

This checkout does not include Git history, so use clear imperative commit subjects such as `Add Harbor adapter validation` or `Fix trajectory cost logging`. Pull requests should describe the behavior change, list commands or agent runs used for validation, link related issues, and include screenshots only for documentation or UI changes.

## Security & Configuration Tips

Store API keys in `.env`, for example `OPENROUTER_API_KEY=...`, and never commit secrets. Treat `runs/` as potentially sensitive because trajectories may include prompts, file contents, command output, and model responses.
