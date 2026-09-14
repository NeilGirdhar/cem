"""Generate the SIP identification results used by the thesis."""

import json
from dataclasses import asdict
from pathlib import Path

import typer

from cem.sip import (
    CausalBenchmarkResult,
    run_direct_injection_benchmark,
    run_inherited_instrument_benchmark,
)
from cem.structure import solver_context_manager

from .settings import jax_cache_dir

app = typer.Typer(pretty_exceptions_enable=False)
_DIRECT_SEED = 200
_INHERITED_SEED = 300


def _conditions(
    results: dict[str, CausalBenchmarkResult],
) -> dict[str, dict[str, float]]:
    conditions = {}
    for name, result in results.items():
        values = {key: value for key, value in asdict(result).items() if value is not None}
        values["effect_error"] = abs(result.estimated_effect - result.true_effect)
        conditions[name] = values
    return conditions


@app.command()
def sip_identification(
    *,
    output: Path = Path("typst/sip-identification.json"),
    count: int = 64,
    steps: int = 960,
) -> None:
    """Run the direct-injection and inherited-instrument thesis benchmarks."""
    if count < 1:
        msg = "--count must be positive"
        raise typer.BadParameter(msg)
    if steps < 1:
        msg = "--steps must be positive"
        raise typer.BadParameter(msg)

    with solver_context_manager(jax_cache_dir=jax_cache_dir, thread_limit=None):
        direct = run_direct_injection_benchmark(
            count=count,
            steps=steps,
            seed=_DIRECT_SEED,
        )
        inherited = run_inherited_instrument_benchmark(
            count=count,
            steps=steps,
            seed=_INHERITED_SEED,
        )

    result = {
        "configuration": {
            "count": count,
            "steps": steps,
            "direct_seed": _DIRECT_SEED,
            "inherited_seed": _INHERITED_SEED,
        },
        "direct-injection": _conditions(direct),
        "instrument-inheritance": _conditions(inherited),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    typer.echo(output)
