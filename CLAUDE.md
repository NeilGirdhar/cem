# CEM — Causal Evidence Model

Instrument discovery via adversarial factor purification with phasor-based evidence tracking.
Built with JAX + Equinox on Python 3.14.

## Documentation

Read these files to understand the theory and architecture before implementing:

- @~/src/typst/thesis.typ
- @~/src/typst/architecture_figures.typ

## Prior work

`@~/backup/cem` is the old version of this project. Draw on it heavily when implementing — it contains working code and patterns to reference.

## Setup

```bash
uv sync --all-extras
```

All commands must be run via `uv run`.

## Common commands

```bash
uv run pytest                        # tests
uv run ruff check --fix              # lint (auto-fix)
uv run ruff format                   # format
uv run ty check                      # type-check
uv run lefthook run pre-commit       # run all pre-commit checks
```

## Import aliases

Always use these aliases — they are enforced by ruff:

| Alias  | Module                        |
|--------|-------------------------------|
| `xpx`  | `array_api_extra`             |
| `ct`   | `ctypes`                      |
| `dt`   | `datetime`                    |
| `eqx`  | `equinox`                     |
| `it`   | `itertools`                   |
| `jnp`  | `jax.numpy`                   |
| `jr`   | `jax.random`                  |
| `jss`  | `jax.scipy.special`           |
| `nx`   | `networkx`                    |
| `npt`  | `numpy.typing`                |
| `optx` | `optimistix`                  |
| `op`   | `optype`                      |
| `onp`  | `optype.numpy`                |
| `npc`  | `optype.numpy.compat`         |
| `opt`  | `optype.typing`               |
| `sc`   | `scipy.special`               |
| `sns`  | `seaborn`                     |

## Style

- Line length: 100
- Docstring style: Google
- Don't move imports into `TYPE_CHECKING` blocks (TC001–TC003 suppressed)
