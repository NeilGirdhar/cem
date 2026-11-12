"""Run Optuna searches and save the best trial's hyperparameters as experiment defaults."""

import logging
from collections.abc import Callable
from dataclasses import replace
from enum import StrEnum
from typing import Any

import rich.progress as rp
import typer
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from optuna.storages import JournalStorage
from optuna.study import Study, create_study, delete_study, load_study
from optuna.trial import FrozenTrial, Trial, TrialState
from tjax import GenericString

from cem.framework import (
    console_progress_bar,
    jax_is_initialized,
    solver_context_manager,
)
from cem.framework.plotter import OptimizableExperiment
from cem.tuned_defaults import update_tuned_defaults

from .settings import get_optuna_storage, jax_cache_dir, optuna_sampler, wandb_settings

_log = logging.getLogger(__name__)


class OptimizationMode(StrEnum):
    """Choose sequential trials with per-run progress or Optuna-managed concurrent trials."""

    single_task = "single"
    multi_task = "multi"


class InvalidTrialsError(typer.BadParameter):
    """The requested trial count is zero or negative."""

    def __init__(self) -> None:
        super().__init__("must be greater than 0", param_hint="trials")


class InvalidJobsError(typer.BadParameter):
    """The requested worker count is invalid."""

    def __init__(self) -> None:
        super().__init__("must be -1 or greater than 0", param_hint="jobs")


def _suggest_from_distribution(trial: Trial, name: str, distribution: BaseDistribution) -> object:
    """Suggest a trial parameter without changing its declared Optuna distribution."""
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


def _objective(
    experiment: OptimizableExperiment,
    hyperparameters: dict[str, Any],
    *,
    wandb: bool,
    profiling: bool,
    progress_manager: rp.Progress | None,
) -> float:
    """Run an experiment with one trial's settings and evaluate its objective."""
    adjusted_wandb_settings = (
        replace(wandb_settings, name=experiment.name, config=hyperparameters, reinit=True)
        if wandb
        else None
    )
    with solver_context_manager(jax_cache_dir=jax_cache_dir, thread_limit=None):
        runs = experiment.runs(
            hyperparameters,
            progress_manager=progress_manager,
            wandb_settings=adjusted_wandb_settings,
            enable_profiling=profiling,
        )
        return experiment.experiment_loss(runs, hyperparameters)


type BoundObjective = Callable[[dict[str, Any], rp.Progress | None], float]


def _progress_manager(*, enabled: bool) -> rp.Progress:
    return console_progress_bar() if enabled else rp.Progress(disable=True)


def _run_single_task_trials(
    study: Study,
    hyper_space: dict[str, BaseDistribution],
    trials: int,
    bound_objective: BoundObjective,
    *,
    progress_bar: bool,
    sync_defaults_if_better: Callable[[], None],
) -> None:
    progress_manager = _progress_manager(enabled=progress_bar)
    with progress_manager:
        task_id = progress_manager.add_task("Optimization", total=trials)
        for _ in range(trials):
            trial = study.ask(hyper_space)
            value = bound_objective(trial.params, progress_manager)
            study.tell(trial, values=value)
            sync_defaults_if_better()
            progress_manager.advance(task_id, 1)


def _existing_trial_count(study_name: str, storage: JournalStorage) -> int | None:
    """Count trials in an existing study, or return none if the study is absent."""
    try:
        study = load_study(study_name=study_name, storage=storage, sampler=optuna_sampler)
    except KeyError:
        return None
    return len(study.get_trials(deepcopy=False))


def _delete_study_with_confirmation(study_name: str, storage: JournalStorage) -> None:
    """Delete an existing study, asking for confirmation if it already contains trials."""
    trial_count = _existing_trial_count(study_name, storage)
    if trial_count is None:
        return
    if trial_count > 0:
        typer.confirm(
            f"Delete Optuna study '{study_name}' with {trial_count} existing trials?",
            abort=True,
        )
    _log.info("Deleting study")
    delete_study(study_name=study_name, storage=storage)


def _enqueue_default_trial(
    study: Study,
    hyper_space: dict[str, BaseDistribution],
    hyperparameters: dict[str, Any],
) -> None:
    """Queue the experiment defaults for parameters in the current search space."""
    trial_params = {k: v for k, v in hyperparameters.items() if k in hyper_space}
    _log.info("Queueing default parameters as first trial: %s", GenericString(trial_params))
    study.enqueue_trial(trial_params)


def _best_trial_number(study: Study) -> int | None:
    try:
        return study.best_trial.number
    except ValueError:
        return None


def _sync_best_defaults(study: Study, experiment_name: str) -> int | None:
    """Save the best completed trial's parameters and return its trial number, if available."""
    try:
        best_trial = study.best_trial
    except ValueError:
        return None
    update_tuned_defaults(experiment_name, best_trial.params)
    _log.info("Updated tuned defaults from trial %s", best_trial.number)
    return best_trial.number


def run_optimize(  # ruff:ignore[complex-structure]
    experiment: OptimizableExperiment,
    *,
    mode: OptimizationMode,
    jobs: int,
    trials: int,
    progress_bar: bool,
    wandb: bool,
    profiling: bool,
    restart: bool,
) -> None:
    """Tune an experiment against a journal-backed Optuna study.

    A restart replaces the study and queues the experiment defaults as the first trial.
    Otherwise the search resumes an existing study or starts a new one.
    """
    if trials <= 0:
        raise InvalidTrialsError
    if jobs != -1 and jobs <= 0:
        raise InvalidJobsError
    hyper_space = experiment.create_hyperparameters()
    storage = get_optuna_storage()
    if restart:
        _delete_study_with_confirmation(experiment.name, storage)
    study = create_study(
        storage=storage,
        sampler=optuna_sampler,
        study_name=experiment.name,
        load_if_exists=not restart,
    )
    if restart:
        _enqueue_default_trial(study, hyper_space, experiment.default_hyperparameters())
    best_trial_number = _sync_best_defaults(study, experiment.name)
    if jax_is_initialized():
        raise RuntimeError

    def bound_objective(
        hyperparameters: dict[str, Any],
        progress_manager: rp.Progress | None,
    ) -> float:
        return _objective(
            experiment,
            hyperparameters,
            wandb=wandb,
            profiling=profiling,
            progress_manager=progress_manager,
        )

    _log.info("Optimizing: %s", GenericString(tuple(hyper_space)))
    match mode:
        case OptimizationMode.single_task:

            def sync_defaults_if_better() -> None:
                nonlocal best_trial_number
                current_best_trial_number = _best_trial_number(study)
                if current_best_trial_number != best_trial_number:
                    best_trial_number = _sync_best_defaults(study, experiment.name)

            _run_single_task_trials(
                study,
                hyper_space,
                trials,
                bound_objective,
                progress_bar=progress_bar,
                sync_defaults_if_better=sync_defaults_if_better,
            )
        case OptimizationMode.multi_task:

            def parallel_objective(trial: Trial) -> float:
                hyperparameters = {
                    dist_name: _suggest_from_distribution(trial, dist_name, distribution)
                    for dist_name, distribution in hyper_space.items()
                }
                return bound_objective(hyperparameters, None)

            def sync_defaults_callback(study: Study, trial: FrozenTrial) -> None:
                if trial.state == TrialState.COMPLETE and study.best_trial.number == trial.number:
                    _sync_best_defaults(study, experiment.name)

            study.optimize(
                parallel_objective,
                n_trials=trials,
                n_jobs=jobs,
                show_progress_bar=progress_bar,
                callbacks=[sync_defaults_callback],
            )
    _log.info("Best parameters found:")
    _log.info(GenericString(study.best_params))
