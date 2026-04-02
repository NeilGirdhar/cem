from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import equinox as eqx
from tjax import JaxArray, RngStream

from .gate import phasor_gate
from .linear import Linear
from .rivalry import RivalryNorm


class Nonlinear(eqx.Module):
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

    f1: Linear
    f2: Linear
    f3: Linear
    rivalry_norm: RivalryNorm

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        num_groups: int,
        *,
        mid_features: int | None = None,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if mid_features is None:
            mid_features = out_features
        return cls(
            f1=Linear.create(in_features, mid_features, streams=streams),
            f2=Linear.create(in_features, mid_features, streams=streams),
            f3=Linear.create(mid_features, out_features, streams=streams),
            rivalry_norm=RivalryNorm.create(out_features, num_groups, streams=streams),
        )

    def infer(self, z: JaxArray) -> JaxArray:
        """Apply GLU-style nonlinear transform with rivalry normalization.

        Args:
            z: Input phasors, shape (..., in_features).

        Returns:
            Output phasors, shape (..., out_features).
        """
        gated = phasor_gate(self.f1.infer(z), self.f2.infer(z))
        return self.rivalry_norm.infer(self.f3.infer(gated))
