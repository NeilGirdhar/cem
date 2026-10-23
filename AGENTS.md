# CEM — Causal Evidence Model

Instrument discovery via adversarial factor purification with phasor-based evidence tracking.
Built with JAX + Equinox on Python 3.14.

## Architecture

Read `README.md` before making structural changes — it documents the full `cem/structure/`
design: `Problem`/`DataSource`, `Model`/`ModelResult`/`NodeConfiguration`, the parameter
system, `DisModel`/`DisGradientTransformation`, `Solver`, and `Telemetry`/`Plotter`.

Key invariant: every JAX array leaf inside a `Model` pytree must be wrapped in either
`LearnableParameter` (optimized by the gradient transformation) or `FixedParameter` (present
in the pytree for vmap/jit but never differentiated).  Use `eqx.field(static=True)` only for
Python scalars and shapes that must be fully invisible to JAX.  `verify_model_has_no_free_parameters`
enforces this at model-creation time.

## Stable facts

- Install with `uv sync --all-extras`.
- Primary references: `~/src/typst/thesis.typ`, `~/src/typst/architecture-figures.typ`.
- Prior implementation: `~/backup/cem`.
- Shared test fixtures (`streams`, `np_rng`, `log`) live in `tests/conftest.py`.

## Working discipline

- On an unrelated new task: clear working context and rebuild only from this file, the user's
  request, and the specific files needed.
- In long sessions (~10 turns): compress context to current task, relevant files, decisions, next step.
