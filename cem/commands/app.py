"""The ``cem`` Typer app with ``list``, ``generate``, ``visualize``, ``optimize``."""

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from typer import Argument, Option

from cem.structure import Experiment, OptimizableExperiment, set_up_logging, solver_context_manager

from ._optimize import OptimizationMode, run_optimize
from ._render import compile_pdf, open_pdf
from .registry import experiments
from .settings import jax_cache_dir

_log = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _lookup(name: str) -> Experiment:
    try:
        return experiments[name]
    except KeyError:
        available = ", ".join(sorted(experiments))
        msg = f"Unknown experiment '{name}'. Available: {available}"
        raise typer.BadParameter(msg) from None


def _write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    typer.echo(str(path))


@app.command("list")
def list_experiments() -> None:
    """List every registered experiment and the verbs available on it."""
    width = max(len(name) for name in experiments)
    for name in sorted(experiments):
        experiment = experiments[name]
        verbs = "generate, visualize"
        if isinstance(experiment, OptimizableExperiment):
            verbs += ", optimize"
        typer.echo(f"{name:<{width}}  ({verbs})")


@app.command()
def generate(
    name: str,
    *,
    output_dir: Annotated[Path, Option("--output-dir")] = Path("results"),
    log: bool = True,
) -> None:
    """Write ``<output_dir>/<name>.json`` for the named experiment."""
    experiment = _lookup(name)
    if log:
        set_up_logging()
    else:
        logging.disable()
    with solver_context_manager(jax_cache_dir=jax_cache_dir, thread_limit=None):
        data = experiment.generate()
    _write_json(data, output_dir / f"{name}.json")


@app.command()
def visualize(
    name: str,
    *,
    output_dir: Annotated[Path, Option("--output-dir")] = Path("results"),
    viewer: Annotated[str | None, Option("--viewer")] = None,
    display: Annotated[bool, Option("--display/--no-display")] = True,
    log: bool = True,
) -> None:
    """Generate ``<name>.json``, compile ``<name>.pdf`` with typst, and open it."""
    experiment = _lookup(name)
    if log:
        set_up_logging()
    else:
        logging.disable()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{name}.json"
    pdf_path = output_dir / f"{name}.pdf"
    with solver_context_manager(jax_cache_dir=jax_cache_dir, thread_limit=None):
        data = experiment.generate()
    _write_json(data, json_path)
    compile_pdf(json_path, pdf_path)
    if display:
        open_pdf(pdf_path, viewer=viewer)


@app.command()
def optimize(  # ruff:ignore[too-many-arguments]
    name: str,
    *,
    mode: Annotated[OptimizationMode, Argument()] = OptimizationMode.single_task,
    jobs: Annotated[int, Option(help="Number of jobs; -1 uses all CPUs.")] = -1,
    trials: int = 1,
    log: bool = True,
    progress_bar: bool = True,
    wandb: bool = False,
    profiling: bool = False,
    restart: bool = False,
) -> None:
    """Search hyperparameters for an :class:`OptimizableExperiment`."""
    experiment = _lookup(name)
    if not isinstance(experiment, OptimizableExperiment):
        msg = f"Experiment '{name}' has no hyperparameters to optimize."
        raise typer.BadParameter(msg)
    if log:
        set_up_logging()
    else:
        logging.disable()
    run_optimize(
        experiment,
        mode=mode,
        jobs=jobs,
        trials=trials,
        progress_bar=progress_bar,
        wandb=wandb,
        profiling=profiling,
        restart=restart,
    )
