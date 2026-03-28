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

     uv run visualize afp

  - Show the graphs in a window:

  .. code:: bash

     uv run visualize afp --display

- Optimize:

  - Using one thread:

  .. code:: bash

     uv run optimize afp single --trials 30

  - Using 8 threads:

  .. code:: bash

     uv run optimize afp multi --trials 30 --jobs 8

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
