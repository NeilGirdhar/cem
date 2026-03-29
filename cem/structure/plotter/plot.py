from __future__ import annotations

from typing import override

import numpy as np
import numpy.typing as npt
from matplotlib.figure import Figure

from .subplot import Subplot


class Plot:
    @override
    def __init__(self, figure: Figure) -> None:
        super().__init__()
        self.figure = figure
        self.axes: npt.NDArray[np.object_] | None = None

    def create(self, rows: int, columns: int, title: str) -> None:
        if self.axes is None or self.axes.shape != (rows, columns):
            axes = self.figure.subplots(rows, columns, squeeze=False)
            self.axes = np.vectorize(Subplot)(axes)

    def subplot(self, row: int, column: int) -> Subplot:
        assert self.axes is not None
        subplot = self.axes[row, column]
        assert isinstance(subplot, Subplot)
        return subplot

    def clear_subplots(self) -> None:
        if self.axes is None:
            return
        self.figure.clear()
        self.axes = None
