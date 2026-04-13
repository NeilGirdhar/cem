=====================
Causal Evidence Model
=====================

.. role:: bash(code)
    :language: bash

.. role:: python(code)
    :language: python

This repository implements the Causal Evidence Model (CEM), which is a model for causal instrument
discovery via adversarial factor purification with phasor-based evidence tracking.

Running
=======

- Install UV: :bash:`pip install uv`

- Clone repository: :bash:`git clone git@github.com:NeilGirdhar/cem.git`

- Install dependencies in virtual environment:

.. code:: bash

   cd cem
   uv sync --all-extras

- Visualize:

  - Just save the graphs:

  .. code:: bash

     uv run visualize afp_iv

  - Show the graphs in a window:

  .. code:: bash

     uv run visualize afp_iv --display

- Optimize:

  - Using one thread:

  .. code:: bash

     uv run optimize afp_iv single --trials 30

  - Using 8 threads:

  .. code:: bash

     uv run optimize afp_iv multi --trials 30 --jobs 8

Architecture
============

All reusable training infrastructure lives in ``cem/structure/``.  The
framework follows a strict layered design: a *problem* produces data, a *model*
consumes it, a *solver* wires them together, and a *plotter* visualises the
result.

Problem layer
-------------

``Problem`` (``cem/structure/problem/problem.py``)
    An Equinox module that owns a ``DataSource`` and optionally narrows a full
    ``ProblemState`` down to a ``ProblemObservation`` via ``extract_observation``.
    The distinction matters for RL-style problems where the agent sees less than
    the ground truth; feed-forward problems simply return the state unchanged.

``DataSource`` (``cem/structure/problem/data_source.py``)
    Generates ``ProblemState`` instances one at a time from a JAX key via
    ``initial_problem_state``.  The framework vmaps this call across a batch.

``ProblemState / ProblemObservation``
    Plain Equinox modules (dataclasses of JAX arrays) that carry one sample of
    data.  States may include ground-truth information that is only used in
    plotters; observations are what the model actually sees.

Model layer
-----------

``Model`` (``cem/structure/graph/model.py``)
    Abstract base class for every neural network.  Subclasses implement:

    .. code:: python

        def infer(
            self,
            observation: object,
            state: object,
            *,
            streams: Mapping[str, RngStream],
            inference: bool,
        ) -> ModelResult: ...

    - ``observation`` is the current ``ProblemObservation``.
    - ``state`` is the recurrent carry (``frozendict()`` for feed-forward models).
    - ``streams`` provides named RNG streams (``"inference"`` for dropout, etc.).
    - ``inference=True`` disables stochastic operations at eval time.
    - Returns a ``ModelResult`` with a scalar ``loss``, a ``frozendict`` of named
      ``NodeConfiguration`` objects (telemetry), and an updated ``state``.

``NodeConfiguration`` (``cem/structure/graph/node.py``)
    Base class for per-node telemetry.  Subclasses carry whatever arrays are
    useful for plotting.  ``TargetConfiguration`` adds a ``loss`` dict and
    implements ``total_loss()`` by summing over fields.

Parameter system
----------------

Every JAX array leaf inside a ``Model`` pytree must be wrapped in exactly one
of two types defined in ``cem/structure/graph/parameters.py``:

``LearnableParameter[A]``
    Carries a JAX array that is updated by the optimizer each training step
    (weights, biases, normalization parameters, rivalry logits, …).

``FixedParameter[A]``
    Carries a JAX array that is part of the model pytree but is *never*
    differentiated or updated (dropout rates, frequency grids, flatteners
    containing distribution constants, …).

Both wrap their payload in a ``.value`` attribute.  The framework enforces
completeness at model-creation time via ``verify_model_has_no_free_parameters``,
which raises if any bare JAX array leaf exists in the tree.

**Why not** ``eqx.field(static=True)``?

``static=True`` removes a field from the pytree entirely — JAX cannot vmap or
jit-trace over it.  ``FixedParameter`` keeps the field *inside* the pytree so
it participates in vmap/jit without receiving gradient updates.  This matters
for anything that might vary across a vmapped batch of models or that contains
arrays (e.g. a variance matrix stored inside a ``Flattener``).  The rule of
thumb: use ``static=True`` for Python scalars and shapes; use ``FixedParameter``
for JAX arrays that are architectural constants.

Training infrastructure
-----------------------

``DisModel`` and ``DisGradientTransformation`` (``cem/structure/graph/disassembled.py``)
    The disassembled model splits the full model pytree into one sub-tree per
    ``ParameterType``.  ``DisGradientTransformation`` maps each type to an
    optimizer (or ``None`` for fixed parameters) and drives ``init``/``update``
    independently.  This enables per-parameter-type optimizer routing without
    special-casing anything in the model.

    The default routing in ``Solver.gradient_transformations`` is:

    .. code:: python

        DisGradientTransformation([
            (ParameterType(FixedParameter), None),          # not optimized
            (ParameterType(LearnableParameter), Adam(...)), # Adam
        ])

``TrainingSolution`` (``cem/structure/solution/training_solution.py``)
    The central training object.  At creation it:

    1. Calls ``verify_model_has_no_free_parameters`` to catch bare leaves.
    2. Splits the model into ``fixed_parameters`` (stored in ``Inference``) and
       ``dis_learnable_parameters`` (stored in ``SolutionState``).
    3. Initialises optimizer state.

    During training, fixed parameters stay in ``Inference`` and never move;
    only the learnable sub-tree is differentiated and updated.

``Inference`` (``cem/structure/solution/inference.py``)
    Holds the fixed parameters and runs batched forward and gradient passes via
    ``vmap`` + ``grad``.  ``Inference.assemble_model`` recombines fixed and
    learnable sub-trees via ``eqx.combine`` before each forward pass, so the
    model always sees a complete pytree.

Solver
------

``Solver[P]`` (``cem/structure/solver/solver.py``)
    The single entry point for a demo.  Subclasses override:

    - ``problem() -> P`` — return the problem instance.
    - ``create_model(data_source, problem, *, streams) -> Model`` — construct
      and return the model (all parameters initialised here).
    - Optionally ``gradient_transformations()`` for custom optimizer routing.

    ``Solver`` also manages Optuna hyperparameter search: fields annotated with
    ``float_field``/``int_field`` and ``optimize=True`` are exposed as tunable
    hyperparameters.

Telemetry and plotting
----------------------

``Telemetry`` (``cem/structure/solution/telemetry.py``)
    Extracts a snapshot of data from each training or inference episode.
    Snapshots are accumulated across episodes and handed to plotters.
    Return ``None`` to skip an episode (e.g. before any training has run).

``Plotter`` (``cem/structure/plotter/plotter.py``)
    Declares which ``Telemetry`` objects it needs via ``telemetries()``, then
    receives the accumulated snapshots in ``plot()``.

Adding a new demo
-----------------

1. Create ``cem/demos/<name>/problem.py`` with a ``ProblemState``, a
   ``DataSource``, and a ``Problem``.
2. Create ``cem/demos/<name>/solution.py`` with a ``Model`` and a ``Solver``.
   Every JAX array field in the model must be ``LearnableParameter`` or
   ``FixedParameter``.
3. Optionally add ``cem/demos/<name>/plotter.py`` with ``Telemetry`` and
   ``Plotter`` subclasses.
4. Register the solver in ``cem/commands/demos.py``.

Contribution guidelines
=======================

The implementation should be consistent with the surrounding style, be type annotated, and pass the
linters below.

- How to start development: :bash:`uv run lefthook install`

- How to run tests: :bash:`uv run pytest`

- How to clean the source:

  - :bash:`uv run ruff check`
  - :bash:`uv run ruff format`
  - :bash:`uv run ty`
