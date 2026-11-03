import logging

import rich.logging as rl

_NOISY_LOGGERS = (
    "httpx",
    "httpcore",
    "huggingface_hub",
    "datasets",
    "urllib3",
    "filelock",
    "fsspec",
)


def set_up_logging() -> None:
    rich_handler = rl.RichHandler(rich_tracebacks=True)
    formatter = logging.Formatter(fmt="%(message)s", datefmt="[%X]")
    rich_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO, handlers=[rich_handler])
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
