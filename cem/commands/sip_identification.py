"""Generate the SIP identification results used by the thesis."""

import json
from dataclasses import asdict
from pathlib import Path

import typer

from cem.sip import (
    CausalBenchmarkResult,
    CausalBenchmarkTrajectory,
    CreditBenchmarkResult,
    run_direct_injection_benchmark,
    run_inherited_instrument_benchmark,
    run_td_error_benchmark,
    simulate_remaining_food,
)
from cem.structure import solver_context_manager

from .settings import jax_cache_dir

app = typer.Typer(pretty_exceptions_enable=False)
_DIRECT_SEED = 200
_INHERITED_SEED = 300
_TD_ERROR_SEED = 400


def _conditions(
    results: dict[str, CausalBenchmarkResult],
) -> dict[str, dict[str, object]]:
    conditions: dict[str, dict[str, object]] = {}
    for name, result in results.items():
        values: dict[str, object] = {
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
                "zero": "No instrument",
                "random": "Injected instrument",
                "true": "True effect",
            },
            "zero": zero_trajectory.estimated_effects,
            "random": random_trajectory.estimated_effects,
            "true": [random.true_effect] * len(training_examples),
        },
        "direct-injection-reconstruction-loss": {
            "iteration": training_examples,
            "line plots": {
                "zero": "No instrument",
                "random": "Injected instrument",
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
                "policy": "No instrument",
                "injected": "Injected instrument",
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
                "policy": "No instrument",
                "injected": "Injected instrument",
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


def _credit_conditions(
    results: dict[str, CreditBenchmarkResult],
) -> dict[str, dict[str, object]]:
    conditions: dict[str, dict[str, object]] = {}
    for name, result in results.items():
        values: dict[str, object] = {
            key: value
            for key, value in asdict(result).items()
            if value is not None and key != "trajectory"
        }
        values["effect_error"] = abs(result.estimated_effect - result.true_effect)
        conditions[name] = values
    return conditions


def _td_error_charts(
    results: dict[str, CreditBenchmarkResult],
) -> dict[str, dict[str, object]]:
    ordinary = results["ordinary"]
    td = results["td"]
    ordinary_trajectory = ordinary.trajectory
    td_trajectory = td.trajectory
    if ordinary_trajectory is None or td_trajectory is None:
        msg = "the TD-error benchmark did not record a training trajectory"
        raise ValueError(msg)
    training_examples = td_trajectory.training_examples
    balance = simulate_remaining_food(count=8, n=16, seed=_TD_ERROR_SEED)
    if not td_trajectory.link_strengths or not td_trajectory.expected_td_errors:
        msg = "TD-error benchmark did not record link and error metrics"
        raise ValueError(msg)
    requested = (0, 64, 128, 192, 256)
    checkpoints = {
        example: td_trajectory.td_error_by_step[index]
        for index, example in enumerate(training_examples)
        if example in requested and index < len(td_trajectory.td_error_by_step)
    }
    if len(checkpoints) != len(requested):
        msg = "TD-error benchmark did not record the requested training checkpoints"
        raise ValueError(msg)
    return {
        "td-error-link-strength": {
            "iteration": training_examples,
            "line plots": {
                "td": "P to R link strength",
            },
            "td": td_trajectory.link_strengths,
        },
        "td-error-expected-error": {
            "iteration": training_examples,
            "line plots": {
                "td": "Expected TD-error magnitude",
            },
            "td": td_trajectory.expected_td_errors,
        },
        "td-error-mean-by-step": {
            "iteration": list(range(len(next(iter(checkpoints.values()))))),
            "line plots": {"0": "Before training", "256": "After training"},
            **{
                str(example): td_trajectory.td_error_mean_by_step[index]
                for index, example in enumerate(training_examples)
                if example in {0, 256}
            },
        },
        "td-error-balance": {
            "iteration": list(range(balance.shape[1])),
            "line plots": {
                f"trajectory-{index + 1}": f"Trajectory {index + 1}"
                for index in range(balance.shape[0])
            },
            **{
                f"trajectory-{index + 1}": balance[index].tolist()
                for index in range(balance.shape[0])
            },
        },
    }


@app.command()
def sip_identification(
    *,
    output: Path = Path("typst/sip-identification.json"),
    count: int = 64,
    steps: int = 256,
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
        td_error = run_td_error_benchmark(
            count=count,
            steps=steps,
            seed=_TD_ERROR_SEED,
        )

    result = {
        "configuration": {
            "count": count,
            "steps": steps,
            "direct_seed": _DIRECT_SEED,
            "inherited_seed": _INHERITED_SEED,
            "td_error_seed": _TD_ERROR_SEED,
        },
        "direct-injection": _conditions(direct),
        **_direct_charts(direct),
        **_inherited_charts(inherited),
        "instrument-inheritance": _conditions(inherited),
        "td-error": _credit_conditions(td_error),
        **_td_error_charts(td_error),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    typer.echo(output)
