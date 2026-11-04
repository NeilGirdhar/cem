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

The `cem` CLI is a Typer app with four verbs. Every experiment registered in
`cem/commands/registry.py` is available by name; `cem list` prints the current catalogue.

## list — show registered experiments

```bash
cem list
```

Prints each experiment's name and the verbs it supports (`generate`, `visualize`, and, for
`OptimizableExperiment` subclasses, `optimize`).

## generate — write chart JSON

```bash
cem generate supervised-missingness
```

Writes `results/supervised-missingness.json` containing every plotter's series for every run.
Each key in the JSON is a plotter `name` (e.g. `missing-data-distributional`,
`calibration-scatter`). Override the directory with `--output-dir <path>`.

## visualize — generate, compile, and open the PDF

```bash
cem visualize supervised-missingness
```

Generates the JSON, compiles a PDF with Typst/Lilaq, and opens it in the configured viewer.
`--no-display` skips the viewer; `--viewer <bin>` overrides it.

## optimize — hyperparameter search

```bash
cem optimize <name> single --trials 30
cem optimize <name> multi --trials 30 --jobs 8
```

Only works for `OptimizableExperiment` subclasses. `single` runs trials in one process; `multi`
dispatches across workers (`--jobs -1` uses all CPUs).

## Rendering saved JSON in Typst

The `cem generate` output is directly readable by `results/chart.typ`. Import one chart into
another Typst document:

```typst
#import "results/chart.typ": chart-from-json

#figure(
  chart-from-json(
    "supervised-missingness.json",
    "missing-data-distributional",
    width: 85%,
  ),
  caption: [Distributional loss on four tabular datasets.],
)
```

To re-render saved JSON directly (one PDF per JSON):

```bash
typst compile --input source=supervised-missingness.json \
    results/render.typ results/supervised-missingness.pdf
```
