from __future__ import annotations

from dataclasses import KW_ONLY

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.stats import gaussian_kde
from jaxkd.extras import k_means
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from tjax import JaxArray
from tjax.dataclasses import field

from .plotter import Plotter


class ProjectionOptions(eqx.Module):
    projection_seed: int = eqx.field(static=True)
    clustering_seed: int = eqx.field(static=True)
    clustering_iterations: int = eqx.field(static=True)

    def cluster_with_k_means(self, points: JaxArray, clustering_modes: int) -> JaxArray:
        key = jr.key(self.clustering_seed)
        _means, labels = k_means(
            key, points, k=clustering_modes, steps=self.clustering_iterations, pairwise=True
        )
        return labels


def _add_border(low: JaxArray, high: JaxArray, border: float = 0.1) -> tuple[JaxArray, JaxArray]:
    delta = high - low
    delta *= border
    delta += 1e-4
    low -= delta
    high += delta
    return low, high


def plot_kde(
    ax: Axes,
    xy_values: JaxArray,
    *,
    resolution: int,
    color_index: int | None = None,
    alpha: float | None = None,
    levels: int = 20,
) -> None:
    """Plot a Gaussian KDE of xy_values as a filled contour on ax."""
    assert xy_values.ndim == 2  # noqa: PLR2004
    assert 1 <= xy_values.shape[-1] <= 2  # noqa: PLR2004
    kde = gaussian_kde(xy_values.T, bw_method="scott")
    min_values = jnp.min(xy_values, axis=0)
    max_values = jnp.max(xy_values, axis=0)
    positions_along_each_dim = [
        jnp.linspace(*_add_border(min_values[i], max_values[i]), resolution)
        for i in range(min_values.shape[0])
    ]
    grid_x, grid_y = jnp.meshgrid(*positions_along_each_dim, indexing="ij")
    assert grid_x.shape == grid_y.shape == (resolution, resolution)
    grid = jnp.reshape(jnp.stack((grid_x, grid_y), axis=0), (2, -1))
    assert grid.shape == (2, resolution * resolution)
    z = jnp.reshape(kde(grid), grid_x.shape)
    np_grid_x = np.asarray(grid_x)
    np_grid_y = np.asarray(grid_y)
    np_z = np.asarray(z)
    if color_index is None:
        ax.contourf(np_grid_x, np_grid_y, np_z, levels=levels, alpha=alpha)
    else:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color = default_colors[color_index]
        colors = [(color, i / levels) for i in range(levels + 1)]
        ax.contourf(np_grid_x, np_grid_y, np_z, levels=len(colors), colors=colors)


class PlotterWithProjection(Plotter):
    """A plotter that can project data to a line or plane."""

    _: KW_ONLY
    projection_options: ProjectionOptions = field(
        default_factory=lambda: ProjectionOptions(0, 0, 1000)
    )
