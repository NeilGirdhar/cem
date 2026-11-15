# Running the CLI

## Install

```bash
git clone git@github.com:NeilGirdhar/cem.git
cd cem
uv sync --all-extras
```

(Install `uv` first with `pip install uv` if it isn't on PATH.)

## Running commands

The commands below assume the project's virtualenv is active — e.g. `source
.venv/bin/activate.fish`. Otherwise prefix each command with `uv run` (so `cem list` becomes
`uv run cem list`).

The `cem` CLI is a Typer app with three verbs. Every experiment registered in
`cem/commands/registry.py` is available by name; `cem list` prints the current catalogue.

## list — show registered experiments

```bash
cem list
```

Prints each experiment's name and the verbs it supports (`generate` and, for
`OptimizableExperiment` subclasses, `optimize`).

## generate — write chart JSON

```bash
cem generate supervised-missingness
```

Writes `results/supervised-missingness.json` containing every plotter's series for every run.
Each key in the JSON is a plotter `name` (e.g. `missing-data-distributional`,
`calibration-scatter`). Override the directory with `--output-dir <path>`.

## optimize — hyperparameter search

```bash
cem optimize <name> single --trials 30
cem optimize <name> multi --trials 30 --jobs 8
```

Only works for `OptimizableExperiment` subclasses. `single` runs trials in one process; `multi`
dispatches across workers (`--jobs -1` uses all CPUs).

## Chart JSON in the thesis

The thesis draws `cem generate` output with `chart.typ` in the thesis repository, which reads each
plotter's dictionary from the JSON. That file documents the keys it reads (`line plots`,
`bar plots`, `x labels`, `scatter series`, and so on).
