"""Plotting: projections, clustering overlays, and demo harness."""

from .demo import Demo, Variant
from .plotter import Plotter
from .with_smooth_graph import PlotterWithSmoothGraph, absolute_percentile, smooth_data

__all__ = [
    "Demo",
    "Plotter",
    "PlotterWithSmoothGraph",
    "Variant",
    "absolute_percentile",
    "smooth_data",
]
