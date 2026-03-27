from __future__ import annotations

from collections.abc import Mapping
from dataclasses import KW_ONLY, InitVar
from typing import override

import equinox as eqx
from tjax import JaxArray, RngStream

from cem.structure import Module

from .gate import phasor_gate
from .linear import Linear
from .rivalry import RivalryNorm


class Nonlinear(Module):
    """GLU-style nonlinear projection followed by rivalry normalization.

    h(z) = r(f3(g(f1(z), f2(z))))

    where f1, f2, f3 are linear links, g is a phasor gate, and r is rivalry normalization.
    f1 produces the gate signal; f2 produces the content; their gated product is projected by f3
    and then normalized.

    Attributes:
        f1: Gate signal projection, in_features → mid_features.
        f2: Content projection, in_features → mid_features.
        f3: Output projection, mid_features → out_features.
        rivalry_norm: Rivalry normalization applied to the output.
    """

    in_features: InitVar[int]
    out_features: InitVar[int]
    num_groups: InitVar[int]
    _: KW_ONLY
    mid_features: InitVar[int | None] = None
    f1: Linear = eqx.field(init=False)
    f2: Linear = eqx.field(init=False)
    f3: Linear = eqx.field(init=False)
    rivalry_norm: RivalryNorm = eqx.field(init=False)

    @override
    def __post_init__(
        self,
        streams: Mapping[str, RngStream],
        in_features: int,
        out_features: int,
        num_groups: int,
        mid_features: int | None,
    ) -> None:
        super().__post_init__(streams=streams)
        if mid_features is None:
            mid_features = out_features
        self.f1 = Linear(in_features, mid_features, streams=streams)
        self.f2 = Linear(in_features, mid_features, streams=streams)
        self.f3 = Linear(mid_features, out_features, streams=streams)
        self.rivalry_norm = RivalryNorm(out_features, num_groups, streams=streams)

    def infer(self, z: JaxArray) -> JaxArray:
        """Apply GLU-style nonlinear transform with rivalry normalization.

        Args:
            z: Input phasors, shape (..., in_features).

        Returns:
            Output phasors, shape (..., out_features).
        """
        gated = phasor_gate(self.f1.infer(z), self.f2.infer(z))
        return self.rivalry_norm.infer(self.f3.infer(gated))
