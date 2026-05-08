from collections.abc import Generator
from contextlib import contextmanager

import jax._src.xla_bridge as xb  # noqa: PLC2701
import numpy as np
from jax import (
    enable_custom_prng,
    enable_x64,
    log_compiles,
    softmax_custom_jvp,
    threefry_partitionable,
)
from jax.experimental.compilation_cache import compilation_cache as cc
from threadpoolctl import ThreadpoolController

assert np  # Required for ThreadpoolController.
tc = ThreadpoolController()


def jax_is_initialized() -> bool:
    return bool(xb._backends)  # noqa: SLF001


@contextmanager
def solver_context_manager(
    *,
    jax_cache_dir: str,
    thread_limit: int | None,
    log_compilation: bool = False,
) -> Generator[None]:
    cc.set_cache_dir(jax_cache_dir)
    with (
        tc.limit(limits=thread_limit),
        enable_x64(),
        log_compiles(log_compilation),
        enable_custom_prng(),
        softmax_custom_jvp(),
        threefry_partitionable(new_val=True),
    ):
        yield
