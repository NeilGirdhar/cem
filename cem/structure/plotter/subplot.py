from __future__ import annotations

import math
from typing import Any, cast, override

import numpy as np
from matplotlib.axes import Axes
from tjax import JaxArray, NumpyIntegralArray, NumpyRealArray

from .plot_labeler import DefaultLabeler, Labeler


class Subplot:
    @override
    def __init__(self, axis: Axes) -> None:
        super().__init__()
        self.axis = axis
        self.drawn: dict[tuple[int, ...], Any] = {}

    def set_title(self, title: str) -> None:
        self.axis.set_title(title)

    def plot(
        self,
        times: NumpyRealArray,
        values: NumpyIntegralArray | NumpyRealArray,
        *,
        max_plots: int = 0,
        labeler: Labeler | None = None,
        base_index: tuple[int, ...] = (),
        log_scale: bool = False,
    ) -> None:
        """Plot the times and values.

        Args:
            times: The times corresponding to the values.  Typically, this is np.arange(n) where n
                is the number of examples in the data set.
            values: The values to be plotted.  Typically, this is a single example in a data set.
            max_plots: The maximum number of plots; any more, and we plot means and deviations.
            labeler: The labeler that produces labels for each plot.
            base_index: If plot is called multiple times, each call should have a different
                base_index.
            log_scale: Make the plot use log scale.
        """
        if labeler is None:
            labeler = DefaultLabeler()

        values = _validate_values(values)

        if log_scale:
            self.axis.set_yscale("log")
        number_of_graph_lines = math.prod(values.shape[1:])
        if number_of_graph_lines <= max_plots:
            plot_indices = list(np.ndindex(*values.shape[1:]))
            labels = labeler.create_labels(plot_indices)
            for plot_index, label in zip(plot_indices, labels, strict=True):
                total_index: tuple[slice | int, ...] = (slice(None), *plot_index)
                ys = values[total_index]
                self._plot_internal(base_index + plot_index, times, ys, label)
            if number_of_graph_lines > 0:
                self.show_legend()
        else:
            (label,) = labeler.create_labels([base_index])
            axes = tuple(range(1, values.ndim))
            means = np.mean(values, axis=axes)
            deviations = np.std(values, axis=axes)
            self._plot_internal(base_index, times, means, label, deviations=deviations)
            self.show_legend()

    def show_legend(self) -> None:
        self.axis.legend()

    def show_image(self, image: JaxArray) -> None:
        self.axis.imshow(image, cmap="Greys")

    def scatter(
        self,
        x: NumpyRealArray,
        y: NumpyRealArray,
        *,
        label: str | None,
        color: str | NumpyRealArray | None = None,
        cmap: str | None = None,
    ) -> None:
        self.axis.scatter(x=x, y=y, label=label, c=color, cmap=cmap)

    def contourf(
        self,
        x: NumpyRealArray,
        y: NumpyRealArray,
        z: NumpyRealArray,
        *,
        alpha: float | None = None,
        levels: int | None = None,
    ) -> None:
        self.axis.contourf(x, y, z, alpha=alpha, levels=levels)

    def _plot_internal(
        self,
        plot_index: tuple[int, ...],
        times: NumpyRealArray,
        values: NumpyIntegralArray | NumpyRealArray,
        label: str,
        *,
        deviations: NumpyRealArray | None = None,
    ) -> None:
        if plot_index not in self.drawn:
            (drawn,) = self.axis.plot(times, values, label=label)
            if deviations is not None:
                self.axis.fill_between(
                    times,
                    values - deviations,
                    values + deviations,
                    alpha=0.3,
                    facecolor=drawn.get_color(),
                )
            self.drawn[plot_index] = drawn
            return
        self.drawn[plot_index].set_data(times, values)


def _validate_values(
    values: NumpyIntegralArray | NumpyRealArray,
) -> NumpyIntegralArray | NumpyRealArray:
    assert isinstance(values, np.ndarray)
    finite_values = np.isfinite(values)
    if not np.all(finite_values):
        index = np.argmin(finite_values)
        print(f"Graph contains infinite numbers starting at index {index}.")  # noqa: T201
        return cast("NumpyRealArray", np.where(finite_values, values, 0.0))
    return values
