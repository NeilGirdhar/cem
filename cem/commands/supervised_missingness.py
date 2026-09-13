"""Generate the supervised missing-input comparison used by the thesis."""

import json
from pathlib import Path

import typer

from cem.demos.supervised.missingness import run_missingness_comparison
from cem.structure import solver_context_manager

from .settings import jax_cache_dir

app = typer.Typer(pretty_exceptions_enable=False)
_MINIMUM_SEED_COUNT = 2


@app.command()
def supervised_missingness(
    *,
    output: Path = Path("typst/supervised-missingness.json"),
    seeds: int = 3,
) -> None:
    """Compare Gaussian NPN and mask-aware MLP missing-input performance."""
    if seeds < _MINIMUM_SEED_COUNT:
        msg = "--seeds must be at least 2 so the output can estimate run-to-run variation"
        raise typer.BadParameter(msg)

    def report(dataset: object, model: str, seed: int, loss: float) -> None:
        dataset_name = getattr(dataset, "value", str(dataset))
        typer.echo(f"{dataset_name}: {model}, seed {seed}: {loss:.8f}", err=True)

    with solver_context_manager(jax_cache_dir=jax_cache_dir, thread_limit=None):
        result = run_missingness_comparison(seed_count=seeds, report=report)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    typer.echo(output)
