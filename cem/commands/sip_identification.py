"""Generate the SIP identification results used by the thesis."""

import json
from dataclasses import asdict
from pathlib import Path

import typer

from cem.sip import (
    CausalBenchmarkResult,
    CausalBenchmarkTrajectory,
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
) -> dict[str, dict[str, object]]:
    conditions = {}
    for name, result in results.items():
        values = {
            key: value
            for key, value in asdict(result).items()
            if value is not None and key != "trajectory"
        }
        values["effect_error"] = abs(result.estimated_effect - result.true_effect)
        conditions[name] = values
    return conditions


def _trajectory(result: CausalBenchmarkResult) -> CausalBenchmarkTrajectory:
    if result.trajectory is None:
        msg = "the benchmark did not record a training trajectory"
        raise ValueError(msg)
    return result.trajectory


def _direct_charts(
    results: dict[str, CausalBenchmarkResult],
) -> dict[str, dict[str, object]]:
    zero = results["zero"]
    random = results["random"]
    zero_trajectory = _trajectory(zero)
    random_trajectory = _trajectory(random)
    if zero_trajectory.training_examples != random_trajectory.training_examples:
        msg = "direct-injection conditions recorded different training checkpoints"
        raise ValueError(msg)
    training_examples = zero_trajectory.training_examples
    return {
        "direct-injection-effect": {
            "iteration": training_examples,
            "line plots": {
                "zero": "Noise magnitude 0",
                "random": "Noise magnitude 1",
                "true": "True effect",
            },
            "zero": zero_trajectory.estimated_effects,
            "random": random_trajectory.estimated_effects,
            "true": [random.true_effect] * len(training_examples),
        },
        "direct-injection-reconstruction-loss": {
            "iteration": training_examples,
            "line plots": {
                "zero": "Noise magnitude 0",
                "random": "Noise magnitude 1",
            },
            "zero": zero_trajectory.reconstruction_losses,
            "random": random_trajectory.reconstruction_losses,
        },
    }


def _inherited_charts(
    results: dict[str, CausalBenchmarkResult],
) -> dict[str, dict[str, object]]:
    policy = _trajectory(results["policy"])
    injected = _trajectory(results["injected"])
    if policy.training_examples != injected.training_examples:
        msg = "inherited-instrument conditions recorded different training checkpoints"
        raise ValueError(msg)
    if not injected.instrument_magnitudes:
        msg = "inherited-instrument benchmark did not record instrument metrics"
        raise ValueError(msg)
    training_examples = injected.training_examples
    return {
        "inherited-instrument-effect": {
            "iteration": training_examples,
            "line plots": {
                "policy": "Policy",
                "injected": "Injected",
                "true": "True effect",
                "instrument-y": "instrument(Y) magnitude",
            },
            "line styles": {"instrument-y": "dashed"},
            "line colors": {
                "policy": "dark-peach",
                "injected": "dark-blue",
                "true": "dark-green",
                "instrument-y": "dark-blue",
            },
            "policy": policy.estimated_effects,
            "injected": injected.estimated_effects,
            "true": [results["injected"].true_effect] * len(training_examples),
            "instrument-y": injected.instrument_magnitudes,
        },
        "inherited-instrument-reconstruction-loss": {
            "iteration": training_examples,
            "line plots": {
                "policy": "Policy Z reconstruction",
                "injected": "Injected Z reconstruction",
            },
            "line styles": {},
            "line colors": {
                "policy": "dark-peach",
                "injected": "dark-blue",
            },
            "policy": policy.reconstruction_losses,
            "injected": injected.reconstruction_losses,
        },
    }


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
        **_direct_charts(direct),
        **_inherited_charts(inherited),
        "instrument-inheritance": _conditions(inherited),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    typer.echo(output)
