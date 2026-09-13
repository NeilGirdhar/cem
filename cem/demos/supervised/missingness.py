"""Capacity-matched supervised missing-input comparison."""

from collections.abc import Callable, Mapping, Sequence
from statistics import fmean, pstdev

import jax.numpy as jnp

from cem.structure.solution import ExecutionPacket, LossTelemetry, Telemetries

from .solution import DatasetKind, LinkKind, SupervisedSolver

type ProgressReporter = Callable[[DatasetKind, str, int, float], None]

_MODEL_LABELS = {
    "npn": "Gaussian NPN",
    "mask-mlp": "Mask-aware MLP",
}
_SETTINGS = {
    DatasetKind.iris: (220, 133, 1168),
    DatasetKind.bike_sharing_demand: (98, 57, 1216),
    DatasetKind.cpu_activity: (20, 11, 882),
    DatasetKind.elevators: (85, 46, 2456),
}
_DATASET_LABELS = {
    DatasetKind.iris: "Iris",
    DatasetKind.bike_sharing_demand: "Bike",
    DatasetKind.cpu_activity: "CPU",
    DatasetKind.elevators: "Elevators",
}


def _held_out_distributional_loss(solver: SupervisedSolver) -> float:
    telemetry = LossTelemetry(selected_node="target")
    training = solver.training_results(packet=ExecutionPacket(scan_chunk_size=64))
    inference = solver.inference_results(
        training.final_state,
        packet=ExecutionPacket(
            telemetries=Telemetries((telemetry,)),
            scan_chunk_size=16,
        ),
    )
    return float(jnp.mean(inference.telemetries[telemetry])) / solver.inference_batch_size


def summarize_missingness_losses(
    losses: Mapping[str, Sequence[Sequence[float]]],
) -> dict[str, object]:
    """Create thesis chart data from per-dataset, per-seed losses."""
    means = {key: [fmean(dataset) for dataset in values] for key, values in losses.items()}
    deviations = {key: [pstdev(dataset) for dataset in values] for key, values in losses.items()}
    return {
        "iteration": list(range(len(_SETTINGS))),
        "x labels": [_DATASET_LABELS[dataset] for dataset in _SETTINGS],
        "bar plots": _MODEL_LABELS,
        "bar errors": deviations,
        **means,
        "seed losses": {
            key: [list(dataset) for dataset in values] for key, values in losses.items()
        },
        "error bars": "one population standard deviation across matched seeds",
    }


def run_missingness_comparison(
    *,
    seed_count: int = 3,
    report: ProgressReporter | None = None,
) -> dict[str, dict[str, object]]:
    """Train both models and return the thesis chart's JSON object."""
    losses: dict[str, list[list[float]]] = {key: [] for key in _MODEL_LABELS}
    for dataset_kind, (npn_width, mlp_width, training_examples) in _SETTINGS.items():
        for key, link_kind, hidden_size in (
            ("npn", LinkKind.natural_parameter, npn_width),
            ("mask-mlp", LinkKind.perceptron, mlp_width),
        ):
            seed_losses = []
            for seed in range(seed_count):
                solver = SupervisedSolver(
                    dataset_kind=dataset_kind,
                    link_kind=link_kind,
                    hidden_size=hidden_size,
                    training_examples=training_examples,
                    training_batch_size=128,
                    inference_examples=64,
                    inference_batch_size=128,
                    learning_rate=0.01,
                    parameters_seed=seed,
                    training_seed=100 + seed,
                    inference_seed=200 + seed,
                    missing_probability=0.5,
                    random_missing_values=False,
                    include_missingness_mask=link_kind == LinkKind.perceptron,
                )
                loss = _held_out_distributional_loss(solver)
                seed_losses.append(loss)
                if report is not None:
                    report(dataset_kind, key, seed, loss)
            losses[key].append(seed_losses)
    return {"missing-data-distributional": summarize_missingness_losses(losses)}
