"""Train each variant of a :class:`Demo`, apply its plotters, and return chart JSON."""

from collections.abc import Mapping
from typing import Any

from cem.structure import ExecutionPacket, console_progress_bar, solver_context_manager
from cem.structure.plotter import Demo

from .settings import jax_cache_dir


def generate_from_demo(
    demo: Demo,
    hyperparameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Train each variant, apply each plotter, and return the chart JSON."""
    resolved = (
        dict(hyperparameters) if hyperparameters is not None else demo.default_hyperparameters()
    )
    telemetries = demo.all_telemetries()
    labeled_results = []
    for variant in demo.variants:
        if len(demo.variants) > 1:
            prefix = f"{variant.label}."
            variant_hyper = {
                k: v for k, v in resolved.items() if k in demo.shared_hyperparameter_names
            }
            variant_hyper.update(
                {k[len(prefix) :]: v for k, v in resolved.items() if k.startswith(prefix)}
            )
        else:
            variant_hyper = resolved
        solver = variant.solver.populate_from_hyperparameters(variant_hyper)
        with solver_context_manager(jax_cache_dir=jax_cache_dir, thread_limit=None):
            packet = ExecutionPacket(
                progress_manager=console_progress_bar(),
                run_label=variant.label or None,
                telemetries=telemetries,
            )
            results = solver.training_and_inference_result(packet=packet)
        labeled_results.append((variant.label, results))

    result: dict[str, dict[str, Any]] = {}
    for plotter in demo.plotters:
        plot_data: dict[str, Any] = {}
        line_plots: dict[str, str] = {}
        for label, (training_results, inference_results) in labeled_results:
            variant_data = plotter.plotted_series(training_results, inference_results, label)
            line_plot_titles = plotter.line_plot_titles(label)
            for key, values in variant_data.items():
                output_key = key if key == "iteration" or not label else f"{label}.{key}"
                plot_data[output_key] = values
                if key in line_plot_titles:
                    line_plots[output_key] = line_plot_titles[key]
        plot_data["line plots"] = line_plots
        result[plotter.name] = plot_data
    return result
