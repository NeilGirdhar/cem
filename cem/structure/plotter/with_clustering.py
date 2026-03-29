from __future__ import annotations

from dataclasses import KW_ONLY

from optuna.distributions import IntDistribution
from tjax import JaxArray

from cem.structure.solver import int_field

from .subplot import Subplot
from .with_projection import PlotterWithProjection


class PlotterWithClustering(PlotterWithProjection):
    _: KW_ONLY
    num_clustering_modes: int = int_field(default=3, domain=IntDistribution(1, 9))

    def _plot_using_kde_and_modes(
        self,
        xy_values: JaxArray,
        mode_values: JaxArray,
        subplot: Subplot,
        resolution: int,
        color_index: int | None = None,
        alpha: float | None = None,
    ) -> None:
        assert xy_values.ndim == 2  # noqa: PLR2004
        assert 1 <= xy_values.shape[-1] <= 2  # noqa: PLR2004
        assert mode_values.ndim == 1
        assert mode_values.shape[0] == xy_values.shape[0]
        for mode in range(self.num_clustering_modes):
            these_xy_values = xy_values[mode_values == mode, :]
            self._plot_using_kde(
                subplot, these_xy_values, resolution=resolution, color_index=mode, alpha=alpha
            )
