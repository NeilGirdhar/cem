# Contributing

Code should be consistent with the surrounding style, fully type annotated, and pass every check
listed below.

## Development install

```bash
git clone git@github.com:NeilGirdhar/cem.git
cd cem
uv sync --all-extras
uv run lefthook install
```

`lefthook install` activates the pre-commit hooks; `uv sync --all-extras` installs runtime and
dev dependencies into `.venv`.

## Running checks

Full pre-commit suite (same thing CI runs):

```bash
uv run lefthook run pre-commit
```

Individual tools:

- `uv run pytest` — tests
- `uv run ruff check` — lint
- `uv run ruff format` — format
- `uv run ty` — type check
- `uv run rumdl fmt` — Markdown format
- `uv run taplo fmt` — TOML format

## Style

- Line length: 100 (set in `.editorconfig`).
- File naming: kebab-case except Python, which uses snake_case.
- Prefer abstract base classes to `Protocol` unless structural typing is required.
- Canadian spelling with *-ize* verbs (*colour*, *centre*, *modelling*).
- Use `jnp.vecdot`, not `jnp.dot` — array API functions over NumPy-era aliases.
- In Equinox and `tjax`, `static=True` means static to JAX tracing, not non-trainable. Keep
  array-bearing fields dynamic; see [`docs/architecture.md`](docs/architecture.md#parameter-system)
  for the `LearnableParameter` / `FixedParameter` contract.

## Commit messages

- Imperative subject, at most 72 characters.
- When the *why* isn't obvious, add a blank line and a body wrapped at 72 that explains why, not
  how.
