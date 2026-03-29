from __future__ import annotations

import logging

import rich.logging as rl


def set_up_logging() -> None:
    rich_handler = rl.RichHandler(rich_tracebacks=True)
    formatter = logging.Formatter(fmt="%(message)s", datefmt="[%X]")
    rich_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[rich_handler])
