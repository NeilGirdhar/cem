from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, InitVar

import equinox as eqx
from tjax import RngStream


class Module(eqx.Module):
    """Base class for all CEM modules.

    Subclasses receive an RNG stream mapping at construction time and use it to initialize
    learnable parameters via ``streams['parameters'].key()``.
    """

    _: KW_ONLY
    streams: InitVar[Mapping[str, RngStream]]

    def __post_init__(self, streams: Mapping[str, RngStream]) -> None:
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # ty: ignore[unresolved-attribute]
