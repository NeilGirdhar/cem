# CEM — Causal Evidence Model

Instrument discovery via adversarial factor purification with phasor-based evidence tracking.
Built with JAX + Equinox on Python 3.14.

## Stable context

Assume these facts are stable unless this file or the user says otherwise:

- CEM uses JAX + Equinox on Python 3.14.
- Primary references: `@~/src/typst/thesis.typ`, `@~/src/typst/architecture_figures.typ`.
- Prior implementation: `@~/backup/cem`.
- Run all project commands with `uv run`.
- Use the import aliases in this file; ruff enforces them.

Do not re-scan the repo to rediscover these facts unless they appear inconsistent.

## Token discipline

- Prefer targeted reads over broad repo scans.
- Avoid recursive discovery unless the task requires it.
- Reuse established context instead of rereading the same files.
- Keep summaries short and avoid repeating unchanged conclusions.

## Resume discipline

If interrupted, rate-limited, or resuming after a pause:

1. Continue from the last completed step.
2. Do not restart with repo-wide discovery.
3. Reuse the stable context and prior verified facts.
4. Read only the minimum additional files needed.
5. Verify specific files or symbols rather than rescanning the project.

## Clear discipline

For a new unrelated task, clear working context and rebuild only from:

- this `CLAUDE.md`
- the user’s request
- the specific files needed for the task

Do not carry incidental context across unrelated tasks.

## Long-session discipline

- Every 10 turns, compress the working context into a short summary before continuing.
- Include only: current task, relevant files, decisions made, next step.

## Setup

```bash
uv sync --all-extras
```

## Common commands

```bash
uv run pytest
uv run ruff check --fix
uv run ruff format
uv run ty check
uv run lefthook run pre-commit
```

## Style

- Line length: 100
- Google docstrings
- Do not move imports into TYPE_CHECKING blocks
