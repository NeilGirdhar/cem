# CEM — Causal Evidence Model

Instrument discovery via adversarial factor purification with phasor-based evidence tracking.
Built with JAX + Equinox on Python 3.14.

## Architecture

Read `README.rst` before making structural changes — it documents the full `cem/structure/`
design: `Problem`/`DataSource`, `Model`/`ModelResult`/`NodeConfiguration`, the parameter
system, `DisModel`/`DisGradientTransformation`, `Solver`, and `Telemetry`/`Plotter`.

Key invariant: every JAX array leaf inside a `Model` pytree must be wrapped in either
`LearnableParameter` (optimized by the gradient transformation) or `FixedParameter` (present
in the pytree for vmap/jit but never differentiated).  Use `eqx.field(static=True)` only for
Python scalars and shapes that must be fully invisible to JAX.  `verify_model_has_no_free_parameters`
enforces this at model-creation time.

## Stable facts

- Run all project commands with `uv run`.
- Primary references: `~/src/typst/thesis.typ`, `~/src/typst/architecture_figures.typ`.
- Prior implementation: `~/backup/cem`.

## Working discipline

- Prefer targeted reads (known path → `Read`, known symbol → `Grep`) over broad scans.
- Reuse established context; do not re-read files already in context.
- On resuming: continue from the last completed step without repo-wide rediscovery.
- On an unrelated new task: clear working context and rebuild only from this file, the user's
  request, and the specific files needed.
- In long sessions (~10 turns): compress context to current task, relevant files, decisions, next step.

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
- Do not move imports into `TYPE_CHECKING` blocks

## Import aliases (enforced by ruff)

`equinox` → `eqx`, `jax.numpy` → `jnp`, `jax.random` → `jr`, `numpy.typing` → `npt`,
`itertools` → `it`, `datetime` → `dt`, `numpy` → `np` (inferred by ruff, not in alias list)
