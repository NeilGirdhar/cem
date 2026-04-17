"""Plotting: projections, clustering overlays, and demo harness."""

from .demo import Demo, Variant
from .plotter import Plotter
from .with_clustering import PlotterWithClustering
from .with_projection import PlotterWithProjection, ProjectionOptions, plot_kde
from .with_smooth_graph import PlotterWithSmoothGraph, absolute_percentile, smooth_data

__all__ = [
    "Demo",
    "Plotter",
    "PlotterWithClustering",
    "PlotterWithProjection",
    "PlotterWithSmoothGraph",
    "ProjectionOptions",
    "Variant",
    "absolute_percentile",
    "plot_kde",
    "smooth_data",
]
