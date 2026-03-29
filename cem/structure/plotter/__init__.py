"""Plotting: subplots, projections, clustering overlays, labelers, and demo harness."""

from .demo import Demo
from .plot import Plot
from .plot_labeler import DefaultLabeler, Labeler, ListLabeler, SimpleLabeler
from .plotter import ExecutedSolverResults, Plotter
from .subplot import Subplot
from .with_clustering import PlotterWithClustering
from .with_projection import PlotterWithProjection, ProjectionOptions
from .with_smooth_graph import PlotterWithSmoothGraph

__all__ = [
    "DefaultLabeler",
    "Demo",
    "ExecutedSolverResults",
    "Labeler",
    "ListLabeler",
    "Plot",
    "Plotter",
    "PlotterWithClustering",
    "PlotterWithProjection",
    "PlotterWithSmoothGraph",
    "ProjectionOptions",
    "SimpleLabeler",
    "Subplot",
]
