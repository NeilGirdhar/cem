"""Commands for listing, running, visualizing, and tuning experiments."""

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from typer import Argument, Option

from cem.framework import Experiment, OptimizableExperiment, set_up_logging, solver_context_manager

from ._optimize import OptimizationMode, run_optimize
from ._render import compile_pdf, open_pdf
from .registry import experiments
from .settings import jax_cache_dir

app = typer.Typer(pretty_exceptions_enable=False, no_args_is_help=True)


def _lookup(name: str) -> Experiment:
    """Find an experiment by name or report the available names as a usage error."""
    try:
        return experiments[name]
    except KeyError:
        available = ", ".join(sorted(experiments))
        msg = f"Unknown experiment '{name}'. Available: {available}"
        raise typer.BadParameter(msg) from None


def _configure_logging(*, log: bool) -> None:
    if log:
        set_up_logging()
    else:
        logging.disable()


def _write_json(data: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(data, file, indent=2)
        file.write("\n")
    typer.echo(str(path))


def _run_to_json(experiment: Experiment, output_dir: Path, *, log: bool) -> Path:
    _configure_logging(log=log)
    with solver_context_manager(jax_cache_dir=jax_cache_dir, thread_limit=None):
        data = experiment.generate()
    json_path = output_dir / f"{experiment.name}.json"
    _write_json(data, json_path)
    return json_path


@app.command("list")
def list_experiments() -> None:
    """List registered experiments and the commands each supports."""
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
    """Run the named experiment and write its chart data as JSON."""
    _run_to_json(_lookup(name), output_dir, log=log)


@app.command()
def visualize(
    name: str,
    *,
    output_dir: Annotated[Path, Option("--output-dir")] = Path("results"),
    viewer: Annotated[str | None, Option("--viewer")] = None,
    display: Annotated[bool, Option("--display/--no-display")] = True,
    log: bool = True,
) -> None:
    """Run the named experiment, render its chart as a PDF, and optionally open it."""
    json_path = _run_to_json(_lookup(name), output_dir, log=log)
    pdf_path = output_dir / f"{name}.pdf"
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
    """Search for better hyperparameters and save the best trial as experiment defaults."""
    experiment = _lookup(name)
    if not isinstance(experiment, OptimizableExperiment):
        msg = f"Experiment '{name}' has no hyperparameters to optimize."
        raise typer.BadParameter(msg)
    _configure_logging(log=log)
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
