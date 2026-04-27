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
    defaults: bool = True,
    display: bool = False,
    log: bool = True,
) -> None:
    demo = demo_registry[name]
    register_graph_as_jax_pytree(nx.DiGraph)
    if log:
        set_up_logging()
    else:
        logging.disable()
    assert not jax_is_initialized()
    if defaults:
        hyperparameters = demo.default_hyperparameters()
        _log.info("Visualizing with defaults: %s", GenericString(hyperparameters))
    else:
        storage = get_optuna_storage()
        try:
            study = load_study(study_name=demo.name, storage=storage, sampler=optuna_sampler)
        except KeyError:
            msg = f"No Optuna study found for '{demo.name}'. Run the optimization first."
            raise SystemExit(msg) from None
        total_trials = len(study.get_trials(deepcopy=False))
        trial = study.best_trial
        _log.info("Choosing best trial out of %d trials", total_trials)
        _log.info(GenericString(trial.params))
        hyperparameters = trial.params
    labeled_results = []
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
        with solver_context_manager(jax_cache_dir=jax_cache_dir, thread_limit=None):
            packet = ExecutionPacket(
                progress_manager=console_progress_bar(), telemetries=variant.all_telemetries()
            )
            results = solver.training_and_inference_result(packet=packet)
        labeled_results.append((variant.label, results))
    generate_figures(demo, labeled_results, display=display)
