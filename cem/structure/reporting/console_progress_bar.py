from __future__ import annotations

import rich.progress as rp


def console_progress_bar() -> rp.Progress:
    return rp.Progress(
        rp.TextColumn("[progress.description]{task.description:24}"),
        rp.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        rp.BarColumn(),
        rp.MofNCompleteColumn(),
        rp.TextColumn("•"),
        rp.TimeElapsedColumn(),
        rp.TextColumn("•"),
        rp.TimeRemainingColumn(compact=True, elapsed_when_finished=True),
    )
