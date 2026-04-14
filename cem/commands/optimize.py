from __future__ import annotations

import logging
from dataclasses import replace
from enum import StrEnum
from typing import Annotated, Any

import networkx as nx
import rich.progress as rp
import typer
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.study import create_study, delete_study
from optuna.trial import Trial
from tjax import GenericString, register_graph_as_jax_pytree
from typer import Argument, BadParameter, Option

from cem.structure import (
    Demo,
    ExecutionPacket,
    console_progress_bar,
    jax_is_initialized,
    set_up_logging,
    solver_context_manager,
)

from .demos import DemoEnum, demo_registry
from .settings import (
    get_optuna_storage,
    jax_cache_dir,
    optuna_sampler,
    wandb_settings,
)

_log = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_enable=False)


class OptimizationMode(StrEnum):
    single_task = "single"
    multi_task = "multi"


class InvalidTrialsError(BadParameter):
    def __init__(self) -> None:
        super().__init__("must be greater than 0", param_hint="trials")


class InvalidJobsError(BadParameter):
    def __init__(self) -> None:
        super().__init__("must be -1 or greater than 0", param_hint="jobs")


def suggest_from_distribution(trial: Trial, name: str, distribution: BaseDistribution) -> object:
    if isinstance(distribution, FloatDistribution):
        return trial.suggest_float(
            name,
            distribution.low,
            distribution.high,
            step=distribution.step,
            log=distribution.log,
        )
    if isinstance(distribution, IntDistribution):
        return trial.suggest_int(
            name,
            distribution.low,
            distribution.high,
            step=distribution.step,
            log=distribution.log,
        )
    if isinstance(distribution, CategoricalDistribution):
        return trial.suggest_categorical(name, distribution.choices)
    msg = f"Unsupported Optuna distribution for {name}: {type(distribution).__name__}"
    raise TypeError(msg)


def objective(
    demo: Demo,
    hyperparameters: dict[str, Any],
    *,
    wandb: bool,
    profiling: bool,
    progress_bar: bool,
) -> float:
    adjusted_wandb_settings = (
        replace(wandb_settings, name=demo.name, config=hyperparameters, reinit=True)
        if wandb
        else None
    )
    solver = demo.create_solver().populate_from_hyperparameters(hyperparameters)
    progress_manager = console_progress_bar() if progress_bar else rp.Progress(disable=True)
    packet = ExecutionPacket(
        progress_manager=progress_manager,
        telemetries=demo.all_telemetries(),
        wandb_settings=adjusted_wandb_settings,
        enable_profiling=profiling,
    )
    with solver_context_manager(jax_cache_dir=jax_cache_dir, thread_limit=None):
        training_results, inference_results = solver.training_and_inference_result(packet=packet)
        return demo.demo_loss(training_results, inference_results)


@app.command()
def optimize(  # noqa: C901
    name: DemoEnum,
    *,
    mode: Annotated[OptimizationMode, Argument()] = OptimizationMode.single_task,
    jobs: Annotated[int, Option(help="The number of jobs.  Using -1 sets jobs to all CPUs.")] = -1,
    trials: int = 1,
    log: bool = True,
    progress_bar: bool = True,
    wandb: bool = False,
    profiling: bool = False,
    continue_study: bool = True,
) -> None:
    demo = demo_registry[name]
    register_graph_as_jax_pytree(nx.DiGraph)
    if log:
        set_up_logging()
    else:
        logging.disable()
    if trials <= 0:
        raise InvalidTrialsError
    if jobs != -1 and jobs <= 0:
        raise InvalidJobsError
    solver = demo.create_solver()
    hyper_space = solver.create_hyperparameters()
    storage = get_optuna_storage()
    if not continue_study:
        _log.info("Deleting study")
        delete_study(study_name=demo.name, storage=storage)
    study = create_study(
        storage=storage,
        sampler=optuna_sampler,
        study_name=demo.name,
        load_if_exists=continue_study,
    )
    if jax_is_initialized():
        raise RuntimeError

    _log.info("Optimizing: %s", GenericString(tuple(hyper_space)))

    def bound_objective(hyperparameters: dict[str, Any]) -> float:
        return objective(
            demo,
            hyperparameters,
            wandb=wandb,
            profiling=profiling,
            progress_bar=progress_bar and mode != OptimizationMode.multi_task,
        )

    match mode:
        case OptimizationMode.single_task:
            for _ in range(trials):
                trial = study.ask(hyper_space)
                value = bound_objective(trial.params)
                study.tell(trial, values=value)
        case OptimizationMode.multi_task:

            def parallel_objective(trial: Trial) -> float:
                hyperparameters = {
                    dist_name: suggest_from_distribution(trial, dist_name, distribution)
                    for dist_name, distribution in hyper_space.items()
                }
                return bound_objective(hyperparameters)

            study.optimize(
                parallel_objective,
                n_trials=trials,
                n_jobs=jobs,
                show_progress_bar=progress_bar,
            )
    _log.info("Best parameters found:")
    _log.info(GenericString(study.best_params))
