import logging
from collections.abc import Callable
from dataclasses import replace
from enum import StrEnum
from typing import Annotated, Any

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
from typer import Argument, BadParameter, Option

from cem.structure import (
    Demo,
    ExecutionPacket,
    console_progress_bar,
    jax_is_initialized,
    set_up_logging,
    solver_context_manager,
)
from cem.tuned_defaults import update_tuned_defaults

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
    progress_manager: rp.Progress | None,
) -> float:
    adjusted_wandb_settings = (
        replace(wandb_settings, name=demo.name, config=hyperparameters, reinit=True)
        if wandb
        else None
    )
    variant_results = []
    for variant in demo.variants:
        if len(demo.variants) > 1:
            prefix = f"{variant.label}."
            shared = variant.shared_hyperparameter_names()
            variant_hyper = {k: v for k, v in hyperparameters.items() if k in shared}
            variant_hyper.update(
                {k[len(prefix) :]: v for k, v in hyperparameters.items() if k.startswith(prefix)}
            )
        else:
            variant_hyper = hyperparameters
        solver = variant.create_solver().populate_from_hyperparameters(variant_hyper)
        packet = ExecutionPacket(
            progress_manager=progress_manager,
            run_label=variant.label or None,
            telemetries=variant.all_telemetries(),
            wandb_settings=adjusted_wandb_settings,
            enable_profiling=profiling,
        )
        with solver_context_manager(jax_cache_dir=jax_cache_dir, thread_limit=None):
            training_results, inference_results = solver.training_and_inference_result(
                packet=packet
            )
            variant_results.append((variant, training_results, inference_results))
    with solver_context_manager(jax_cache_dir=jax_cache_dir, thread_limit=None):
        return demo.demo_loss(variant_results, hyperparameters)


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
    try:
        study = load_study(study_name=study_name, storage=storage, sampler=optuna_sampler)
    except KeyError:
        return None
    return len(study.get_trials(deepcopy=False))


def _delete_study_with_confirmation(study_name: str, storage: JournalStorage) -> None:
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
    trial_params = {k: v for k, v in hyperparameters.items() if k in hyper_space}
    _log.info("Queueing default parameters as first trial: %s", GenericString(trial_params))
    study.enqueue_trial(trial_params)


def _best_trial_number(study: Study) -> int | None:
    try:
        return study.best_trial.number
    except ValueError:
        return None


def _sync_best_defaults(study: Study, demo_name: str) -> int | None:
    try:
        best_trial = study.best_trial
    except ValueError:
        return None
    update_tuned_defaults(demo_name, best_trial.params)
    _log.info("Updated tuned defaults from trial %s", best_trial.number)
    return best_trial.number


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
    restart: bool = False,
) -> None:
    demo = demo_registry[name]
    if log:
        set_up_logging()
    else:
        logging.disable()
    if trials <= 0:
        raise InvalidTrialsError
    if jobs != -1 and jobs <= 0:
        raise InvalidJobsError
    hyper_space = demo.create_hyperparameters()
    storage = get_optuna_storage()
    if restart:
        _delete_study_with_confirmation(demo.name, storage)
    study = create_study(
        storage=storage,
        sampler=optuna_sampler,
        study_name=demo.name,
        load_if_exists=not restart,
    )
    if restart:
        _enqueue_default_trial(study, hyper_space, demo.default_hyperparameters())
    best_trial_number = _sync_best_defaults(study, demo.name)
    if jax_is_initialized():
        raise RuntimeError

    def bound_objective(
        hyperparameters: dict[str, Any], progress_manager: rp.Progress | None
    ) -> float:
        return objective(
            demo,
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
                    best_trial_number = _sync_best_defaults(study, demo.name)

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
                    dist_name: suggest_from_distribution(trial, dist_name, distribution)
                    for dist_name, distribution in hyper_space.items()
                }
                return bound_objective(hyperparameters, None)

            def sync_defaults_callback(study: Study, trial: FrozenTrial) -> None:
                if trial.state == TrialState.COMPLETE and study.best_trial.number == trial.number:
                    _sync_best_defaults(study, demo.name)

            study.optimize(
                parallel_objective,
                n_trials=trials,
                n_jobs=jobs,
                show_progress_bar=progress_bar,
                callbacks=[sync_defaults_callback],
            )
    _log.info("Best parameters found:")
    _log.info(GenericString(study.best_params))
