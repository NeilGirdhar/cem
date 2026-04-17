from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import equinox as eqx
import matplotlib as mpl
import seaborn as sns
from scipy.constants import golden_ratio

from cem.structure import Demo
from cem.structure.solution import InferenceResults, TrainingResults


class _MatplotlibSettings(eqx.Module):
    backend: str
    dpi: float
    figure_size: tuple[float, float]
    context: dict[str, Any] | Literal["paper", "notebook", "talk", "poster"]
    style: dict[str, Any] | Literal["white", "dark", "whitegrid", "darkgrid", "ticks"]
    font_scale: float


def _pdf_matplotlib_settings() -> _MatplotlibSettings:
    dpi = 1200
    width = 8.5 - 0.5 / 2.54
    height = width / golden_ratio
    figure_size = (width, height)
    return _MatplotlibSettings(
        backend="pdf",
        dpi=dpi,
        figure_size=figure_size,
        context="paper",
        style="whitegrid",
        font_scale=0.8,
    )


def _qt_matplotlib_settings() -> _MatplotlibSettings:
    return _MatplotlibSettings(
        backend="QtAgg",
        dpi=120.0,
        figure_size=(10.0, 10.0 / golden_ratio),
        context="notebook",
        style="darkgrid",
        font_scale=1.0,
    )


def generate_figures(
    demo: Demo,
    labeled_results: Sequence[tuple[str, tuple[TrainingResults, InferenceResults]]],
    *,
    display: bool,
) -> None:
    settings = _qt_matplotlib_settings() if display else _pdf_matplotlib_settings()
    mpl.use(settings.backend)
    mpl.rcParams.update(
        {"figure.dpi": settings.dpi, "figure.figsize": settings.figure_size, "text.usetex": False}
    )
    sns.set_theme(context=settings.context, style=settings.style, font_scale=settings.font_scale)
    from matplotlib import pyplot as plt  # noqa: PLC0415

    for plotter in demo.plotters():
        figure = plt.figure(num=plotter.title, constrained_layout=True, clear=True)
        for label, results in labeled_results:
            plotter.plot(figure, results[0], results[1], label)
        if not display:
            figure.savefig(f"{demo.name}_{plotter.name}.pdf", format="pdf", bbox_inches="tight")
            plt.close(figure)

    if display:
        plt.show()
