from __future__ import annotations

import logging

import networkx as nx
import typer
from optuna.study import load_study
from tjax import GenericString, register_graph_as_jax_pytree

from cem.structure import (
    ExecutionPacket,
    console_progress_bar,
    jax_is_initialized,
    set_up_logging,
    solver_context_manager,
)

from .demos import DemoEnum, demo_registry
from .generate_figures import generate_figures
from .settings import get_optuna_storage, jax_cache_dir, optuna_sampler

_log = logging.getLogger(__name__)
app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def visualize(
    name: DemoEnum,
    *,
    display: bool = False,
    log: bool = True,
) -> None:
    demo = demo_registry[name]
    register_graph_as_jax_pytree(nx.DiGraph)
    if log:
        set_up_logging()
    else:
        logging.disable()
    solver = demo.create_solver()
    storage = get_optuna_storage()
    study = load_study(study_name=demo.name, storage=storage, sampler=optuna_sampler)
    total_trials = len(study.get_trials(deepcopy=False))
    trial = study.best_trial
    solver = solver.populate_from_hyperparameters(trial.params)
    _log.info("Choosing best trial out of %d trials", total_trials)
    _log.info(GenericString(trial.params))
    assert not jax_is_initialized()
    with solver_context_manager(jax_cache_dir=jax_cache_dir, thread_limit=None):
        packet = ExecutionPacket(
            progress_manager=console_progress_bar(), telemetries=demo.all_telemetries()
        )
        results = solver.training_and_inference_result(packet=packet)
    generate_figures(demo, results, display=display)
