# Architecture

All reusable training infrastructure lives in `cem/structure/`. The framework follows a strict
layered design: a *problem* produces data, a *model* consumes it, a *solver* wires them together,
and a *plotter* turns telemetry into chart series. Experiments register in
`cem/commands/registry.py` and are then available under every CLI verb.

## Problem layer

`Problem` (`cem/structure/problem/problem.py`)\
An Equinox module that owns a `DataSource` and optionally narrows a full `ProblemState` down to a
`ProblemObservation` via `extract_observation`. The distinction matters for RL-style problems where
the agent sees less than the ground truth; feed-forward problems simply return the state unchanged.

`DataSource` (`cem/structure/problem/data_source.py`)\
Generates `ProblemState` instances one at a time from a JAX key via `initial_problem_state`. The
framework vmaps this call across a batch.

`ProblemState / ProblemObservation`\
Plain Equinox modules (dataclasses of JAX arrays) that carry one sample of data. States may include
ground-truth information that is only used in plotters; observations are what the model actually
sees.

## Model layer

`Model` (`cem/structure/graph/model.py`)\
Abstract base class for every neural network. Subclasses implement:

```python
def infer(
    self,
    observation: object,
    state: object,
    *,
    streams: Mapping[str, RngStream],
    inference: bool,
) -> ModelResult: ...
```

- `observation` is the current `ProblemObservation`.
- `state` is the recurrent carry (`frozendict()` for feed-forward models).
- `streams` provides named RNG streams (`"inference"` for dropout, etc.).
- `inference=True` disables stochastic operations at eval time.
- Returns a `ModelResult` with a `frozendict` of named `NodeConfiguration` objects (telemetry), an
  updated `state`, a fixed-weight loss, and optional named balanced losses. Its `loss` is their
  sum.

`NodeConfiguration` (`cem/structure/graph/node.py`)\
Base class for per-node telemetry. Subclasses carry whatever arrays are useful for plotting.
`TargetConfiguration` adds a `loss` dict and implements `total_loss()` by summing over fields.

## Loss reduction policy

Reduction semantics are part of the model contract.

Use `jnp.sum` for semantic objective terms inside one model inference. Field axes and per-node
objective terms are additive evidence or additive penalties for the same example.
`ModelResult.loss` should therefore be a per-example scalar formed by summing those semantic
contributions.

Use `jnp.mean` for empirical averages over independent examples, samples, or reporting windows.
The training infrastructure already averages a batch of per-example `ModelResult.loss` values in
`Inference._v_infer`.

Telemetry may store normalized values for readability, but names and docstrings must say so. If a
demo score should report the objective, first sum all non-example axes to recover the per-example
objective and then average over examples. Plotters may average non-time axes when the goal is
visual normalization rather than the training objective.

## Parameter system

Every JAX array leaf inside a `Model` pytree must be wrapped in one of two parameter categories in
`cem/structure/graph/parameters.py`:

`LearnableParameter[A]`\
Carries a JAX array updated by the optimizer. A `SummedGradientParameter` uses the sum of the
gradients that reach it. A `LossBalancedParameter` marks a parameter for gradient balancing;
each balancing group uses its own subclass.

`FixedParameter[A]`\
Carries a JAX array that is part of the model pytree but is *never* differentiated or updated
(dropout rates, flatteners containing distribution constants, …).

Both wrap their payload in a `.value` attribute. The framework enforces completeness at
model-creation time via `verify_model_has_no_free_parameters`, which raises if any bare JAX array
leaf exists in the tree.

**Why not** `eqx.field(static=True)`?

`static=True` removes a field from the pytree entirely — JAX cannot vmap or jit-trace over it.
`FixedParameter` keeps the field *inside* the pytree so it participates in vmap/jit without
receiving gradient updates. This matters for anything that might vary across a vmapped batch of
models or that contains arrays (e.g. a variance matrix stored inside a `Flattener`). The rule of
thumb: use `static=True` for Python scalars and shapes; use `FixedParameter` for JAX arrays that
are architectural constants.

**Nonnegative learnable parameters.**

Store the unconstrained `LearnableParameter[JaxRealArray]` in ℝ and apply `jax.nn.softplus` at
every read site. This is the canonical pattern — no projection step, smooth autodiff, bounded
nonnegativity at the API surface. Use the same pattern for scalars (Lagrange multipliers, loss
scales) and arrays (per-channel precisions, loss weights). Alternatives like projected updates
(`jnp.maximum(value, 0)` after each step) require a post-update hook the gradient-transform pipeline
doesn't currently provide.

## Training infrastructure

`DisModel` and `DisGradientTransformation` (`cem/structure/graph/disassembled.py`)\
The disassembled model splits the full model pytree into one sub-tree per `ParameterType`.
`DisGradientTransformation` maps each type to an optimizer (or `None` for fixed parameters) and
drives `init`/`update` independently. This enables per-parameter-type optimizer routing without
special-casing anything in the model.

A multi-loss transform from tjax, such as `MTAdam`, balances the losses named by its `loss_ids` and
receives their gradients separately. Its group must receive gradient from those losses alone: a
`checkify` check rejects any finite nonzero gradient from other losses, and the trainer
functionalizes the check around each training chunk.

The default routing in `Solver.gradient_transformations` is:

```python
DisGradientTransformation(
    [
        (ParameterType(FixedParameter), None),  # not optimized
        (ParameterType(LearnableParameter), Adam(...)),  # Adam
    ]
)
```

`TrainingSolution` (`cem/structure/solution/training_solution.py`)\
The central training object. At creation it:

1. Calls `verify_model_has_no_free_parameters` to catch bare leaves.
2. Splits the model into `fixed_parameters` (stored in `Inference`) and `dis_learnable_parameters`
   (stored in `SolutionState`).
3. Initialises optimizer state.

During training, fixed parameters stay in `Inference` and never move; only the learnable sub-tree
is differentiated and updated.

`Inference` (`cem/structure/solution/inference.py`)\
Recombines the fixed and learnable parameter trees before each batched forward pass. The losses
that the multi-loss transforms balance come from the `ModelResult`, and the remaining objective is
the fixed-weight loss and every other balanced loss. Inference pulls back each requested batch mean
and the remaining objective from the same forward pass.

## Solver

`Solver[P]` (`cem/structure/solver/solver.py`)\
Constructs the model and the problem for one experiment configuration. Subclasses override:

- `problem() -> P` — return the problem instance.
- `create_model(data_source, problem, *, streams) -> Model` — construct and return the model (all
  parameters initialised here).
- Optionally `gradient_transformations()` for custom optimizer routing.

`Solver` also manages Optuna hyperparameter search: fields annotated with `float_field` /
`int_field` and `optimize=True` are exposed as tunable hyperparameters.

## Telemetry and plotting

`Telemetry` (`cem/structure/solution/telemetry.py`)\
Extracts a snapshot of data from each training or inference episode. Snapshots are accumulated
across episodes and handed to plotters. Return `None` to skip an episode (e.g. before any training
has run).

`Plotter` (`cem/structure/plotter/plotter.py`)\
Declares which `Telemetry` objects it needs via `telemetries()`, then pivots the accumulated
`Run`s into chart series in `plotted_series()`. The `name` field becomes the plot key in the
output JSON; the `title` field becomes the chart title.

## Experiment harness

`Experiment` (`cem/structure/plotter/experiment.py`)\
A named bundle of **variants × sweep × plotters**. One `Experiment` represents everything the CLI
needs to train, infer, and render for one entry in `cem/commands/registry.py`.

`Variant` (frozen dataclass)\
One alternative solver inside an experiment. `label` is the chart series key (empty for a
single-variant experiment; required and non-empty for multi-variant experiments). `solver` is the
base `Solver` to run.

`Experiment(name, variants, plotters, sweep=…, shared_hyperparameter_names=…)`\
The `sweep` is a sequence of parameter overrides. Every `(variant, sweep_entry)` pair becomes one
`Run`. Subclasses override `resolve_solver(variant, sweep_entry, hyperparameters)` to apply sweep
entries and hyperparameters to the base solver — for example, to set `dataset_kind`, `seed`, or
`hidden_size`.

`Run` (frozen dataclass)\
One `(variant, sweep_entry)` execution. Carries `variant_label`, `sweep_entry`,
`training_results`, and `inference_results`. Plotters receive all runs and pivot them into chart
data.

`OptimizableExperiment`\
Subclass that adds `experiment_loss` and plugs into `cem optimize`. Multi-variant experiments
optimize jointly; each variant's hyperparameters are prefixed with `"{variant.label}."` except for
names listed in `shared_hyperparameter_names`.

## Adding a new experiment

The current pattern groups related experiments under one directory. For example,
`cem/npn/experiments/supervised/` holds both `supervised-missingness` and `supervised-calibration`,
sharing `datasets.py` and `models.py`.

1. **Data.** Create `cem/<module>/experiments/<group>/datasets.py` with a `ProblemState`, a
   `DataSource`, and a `Problem`.
2. **Model and solver.** Create `cem/<module>/experiments/<group>/models.py` with a `Model` and a
   `Solver`. Every JAX array field must be `LearnableParameter` or `FixedParameter`.
3. **Experiment.** Create `cem/<module>/experiments/<group>/<name>.py` with a `Plotter` subclass
   (implements `plotted_series`) and an `Experiment` subclass. Override `resolve_solver` to apply
   each sweep entry to the base solver. Instantiate the experiment as a module-level singleton:

   ```python
   supervised_missingness_experiment = SupervisedMissingnessExperiment()
   ```

4. **Register.** Import that singleton in `cem/commands/registry.py` and add it to the
   `experiments` dict. The CLI picks it up under every verb (`list`, `generate`, `visualize`,
   `optimize` when applicable).

Existing examples: `cem/npn/experiments/supervised/missingness.py` and
`cem/npn/experiments/supervised/calibration.py`.
