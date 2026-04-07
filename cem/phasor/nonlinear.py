from __future__ import annotations

from collections.abc import Mapping
from typing import Self

import equinox as eqx
import jax.numpy as jnp
from tjax import JaxArray, JaxRealArray, RngStream

from cem.phasor.gate import phasor_gate
from cem.phasor.linear import Linear
from cem.phasor.message import PhasorMessage
from cem.phasor.rivalry import RivalryNorm


class Nonlinear(eqx.Module):
    """GLU-style nonlinear projection followed by rivalry normalization, with optional dropout.

    h(z) = d(r(f3(g(f1(z), f2(z)))))

    where f1, f2, f3 are linear links, g is a phasor gate, r is rivalry normalization,
    and d is an optional final phasor dropout.  Pass ``inference=True`` to :meth:`infer`
    to skip all dropout at eval time.

    Attributes:
        f1: Gate signal projection, in_features → mid_features.
        f2: Content projection, in_features → mid_features.
        f3: Output projection, mid_features → out_features.
        rivalry_norm: Rivalry normalization applied to the output.
        dropout_rate: Fraction of outputs zeroed after rivalry normalization.  0.0 disables.
    """

    f1: Linear
    f2: Linear
    f3: Linear
    rivalry_norm: RivalryNorm
    dropout_rate: JaxRealArray

    @classmethod
    def create(
        cls,
        in_features: int,
        out_features: int,
        num_groups: int,
        *,
        mid_features: int | None = None,
        dropout_rate: float = 0.0,
        streams: Mapping[str, RngStream],
    ) -> Self:
        if mid_features is None:
            mid_features = out_features
        return cls(
            f1=Linear.create(in_features, mid_features, streams=streams),
            f2=Linear.create(in_features, mid_features, streams=streams),
            f3=Linear.create(mid_features, out_features, streams=streams),
            rivalry_norm=RivalryNorm.create(out_features, num_groups, streams=streams),
            dropout_rate=jnp.asarray(dropout_rate),
        )

    def infer(self, z: JaxArray, *, streams: Mapping[str, RngStream], inference: bool) -> JaxArray:
        """Apply GLU-style nonlinear transform with rivalry normalization and optional dropout.

        Args:
            z: Input phasors, shape (..., in_features).
            streams: RNG streams; the ``"inference"`` stream is used for dropout.
            inference: When ``True``, dropout is skipped.

        Returns:
            Output phasors, shape (..., out_features).
        """
        gated = phasor_gate(self.f1.infer(z), self.f2.infer(z))
        result = self.rivalry_norm.infer(self.f3.infer(gated))
        if inference:
            return result
        return PhasorMessage(result).dropout(streams["inference"].key(), self.dropout_rate).data
